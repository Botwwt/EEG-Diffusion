import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import paper_rgb_object as rgb


class MatchedRowsTest(unittest.TestCase):
    def setUp(self):
        self.rows = np.array([31, 9, 58, 4, 38, 64, 51, 77, 3, 99, 80, 82])
        self.labels = np.repeat(np.arange(6), 2)
        self.layout = np.array([[0], [2], [5], [6], [3], [9], [4], [8], [2], [0], [9], [7]], dtype=float)

    def test_modes_share_all_other_rows_and_draws(self):
        plan = rgb.make_plan(self.rows, self.labels, self.layout, steps=6, seed=37, negatives=3)
        order = np.arange(len(self.rows))[::-1]
        self.assertEqual(plan, rgb.make_plan(self.rows[order], self.labels[order], self.layout[order], 6, 37, 3))
        lookup = {row: i for i, row in enumerate(self.rows)}
        query_classes = []
        for episode in plan:
            q = lookup[episode['query']]
            s = lookup[episode['sibling']]
            self.assertNotEqual(q, s)
            self.assertEqual(self.labels[q], self.labels[s])
            query_classes.append(self.labels[q])
            seen = {self.labels[q]}
            expected = []
            for i in sorted(range(len(self.rows)), key=lambda i: (abs(self.layout[i, 0] - self.layout[q, 0]), self.rows[i])):
                if self.labels[i] not in seen:
                    expected.append(int(self.rows[i]))
                    seen.add(self.labels[i])
            self.assertEqual(rgb.target_rows(episode, 'token-rgb-photo'), [episode['query'], *expected[:3]])
            self.assertEqual(rgb.target_rows(episode, 'token-rgb-sibling'), [episode['sibling'], *expected[:3]])
        self.assertEqual(len(set(query_classes)), 6)
        self.assertEqual(len({item['noise_seed'] for item in plan}), 6)

    def test_invalid_registration_and_unknown_mode(self):
        with self.assertRaises(ValueError):
            rgb.make_plan(self.rows, self.labels, self.layout, 7, 9, 3)
        with self.assertRaises(ValueError):
            rgb.make_plan(np.zeros_like(self.rows), self.labels, self.layout, 1, 9, 1)
        with self.assertRaises(ValueError):
            rgb.target_rows({'query': 0, 'sibling': 1, 'negatives': [2]}, 'other')

    def test_same_size_split_with_a_heldout_row_is_rejected(self):
        labels = np.repeat(np.arange(1654), 10)
        fold = np.arange(1654) % 5
        rows = np.flatnonzero(fold[labels] != 0)
        ids = [f'photo_{i}' for i in range(len(labels))]
        paths = [f'training_images/{int(c) + 1:05d}_thing/{s}.jpg' for c, s in zip(labels, ids)]
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'common': SimpleNamespace(concept_folds=lambda n: fold)}):
            prep = Path(folder)
            np.save(prep / 'train_concept.npy', labels)
            (prep / 'train_ids.json').write_text(json.dumps(dict(ids=ids, paths=paths)))
            split = prep / 'paper_split.json'
            split.write_text(json.dumps(dict(train_rows=rows.tolist())))
            actual, _, members = rgb.registered_photos(prep)
            np.testing.assert_equal(actual, rows)
            self.assertEqual(set(members), set(rows))
            rows[0] = 0
            split.write_text(json.dumps(dict(train_rows=rows.tolist())))
            with self.assertRaises(ValueError):
                rgb.registered_photos(prep)


def archive_rollout():
    # The deployed module imports HPC-only dependencies at file scope. Exercise the actual
    # archived Euler function on CPU with a tiny differentiable U-Net and scheduler.
    import math
    path = Path(__file__).with_name('exp6_draft_intra4.py')
    func = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if getattr(n, 'name', '') == 'rollout_4step_step_ips')
    ns = {'torch': torch, 'math': math}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), 'exec'), ns)
    return ns[func.name]


class RenderGradientTest(unittest.TestCase):
    def test_all_four_steps_and_vae_match_render_and_gradient(self):
        class UNet(torch.nn.Module):
            dtype = torch.float32

            def forward(self, x, t, *, encoder_hidden_states, added_cond_kwargs, cross_attention_kwargs):
                tokens = cross_attention_kwargs['ip_hidden_states'].mean()
                return SimpleNamespace(sample=(x + tokens * (t.item() + 1) * .1).tanh())

        class VAE(torch.nn.Module):
            config = SimpleNamespace(scaling_factor=.6)

            def decode(self, z):
                return SimpleNamespace(sample=z[:, :3].tanh())

        unet, vae = UNet(), VAE()
        scheduler = SimpleNamespace(timesteps=torch.tensor([3, 2, 1, 0]), sigmas=torch.tensor([1., .7, .35, .12, 0.]))
        prompts = (torch.zeros(1, 2, 3), torch.zeros(1, 3), torch.zeros(1, 6))
        noise = torch.randn(1, 5, 4, 6, 6, generator=torch.Generator().manual_seed(42))
        rollout = archive_rollout()
        a = torch.tensor([[[.35]]], requires_grad=True)
        b = a.detach().clone().requires_grad_()
        actual = rgb.render_rgb(unet, scheduler, vae, prompts, a, noise, rollout)
        eps, *steps = noise.unbind(1)
        z = rollout(unet, scheduler, [b] * 4, torch.zeros_like(eps), eps, steps, *prompts, 0., None)
        expected = (vae.decode(z / vae.config.scaling_factor).sample * .5 + .5).clamp(0, 1)
        torch.testing.assert_close(actual, expected)
        grad = torch.autograd.grad(actual.sum(), a)[0]
        self.assertGreater(grad.abs().item(), 1e-6)
        torch.testing.assert_close(grad, torch.autograd.grad(expected.sum(), b)[0])

    def test_dino_crop_and_gradients(self):
        pixels = torch.full((1, 3, 512, 512), .5, requires_grad=True)
        processed = rgb.dino_pixels(pixels)
        self.assertEqual(processed.shape, (1, 3, 224, 224))
        expected = (torch.tensor(.5) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(processed[0, :, 100, 100], expected)
        processed.sum().backward()
        self.assertGreater(pixels.grad.abs().sum().item(), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'the production optimizer records CUDA peak memory')
    def test_one_step_updates_only_the_last_weight_and_responds_to_the_positive(self):
        nn = torch.nn

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = nn.Linear(4, 4)
                self.block = nn.Module()
                layer = nn.Module()
                layer.ff = nn.Module()
                layer.ff.w2 = nn.Linear(4, 4, bias=False)
                self.block.layers = nn.ModuleList([layer])

            def forward(self, x):
                return self.block.layers[0].ff.w2(self.proj_in(x))

        def encoder(device):
            model = nn.Module()
            model.backbone = Backbone()
            return model.to(device)

        class UNet(nn.Module):
            dtype = torch.float16

            def forward(self, x, t, *, encoder_hidden_states, added_cond_kwargs, cross_attention_kwargs):
                return SimpleNamespace(sample=x * .1 + cross_attention_kwargs['ip_hidden_states'][:, :, None, None])

        class VAE(nn.Module):
            config = SimpleNamespace(scaling_factor=.8)

            def decode(self, x):
                return SimpleNamespace(sample=x[:, :3].tanh())

        class Dino(nn.Module):
            def forward(self, pixel_values):
                rgb_mean = pixel_values.mean((2, 3))
                return SimpleNamespace(last_hidden_state=torch.cat([rgb_mean, torch.ones_like(rgb_mean[:, :1])], dim=1)[:, None])

        torch.manual_seed(17)
        state = encoder('cpu').state_dict()
        eeg = torch.tensor([[.2, .3, .1, -.5]], device='cuda')
        plan = [dict(query=0, sibling=1, negatives=[2, 3, 4, 5], noise_seed=81)]
        targets = torch.cat([torch.eye(4), -torch.eye(4)[:2]], dim=0)
        sigmas = torch.tensor([1., .7, .35, .12, 0.], device='cuda')
        scheduler = SimpleNamespace(timesteps=torch.tensor([3, 2, 1, 0], device='cuda'), sigmas=sigmas)
        prompts = tuple(torch.zeros(*size, device='cuda') for size in [(1, 2, 3), (1, 3), (1, 6)])
        pipe = SimpleNamespace(unet=UNet().cuda(), scheduler=scheduler, vae=VAE().cuda())
        pipeline = (pipe, nn.Identity(), nn.Identity(), prompts)
        with patch('paper_instance_residual.make_encoder', side_effect=encoder):
            results = [rgb.train_arm(mode, state, eeg, plan, targets, pipeline, Dino().cuda(), archive_rollout()) for mode in rgb.MODES]
        for updated, stats in results:
            changed = {k for k in state if not torch.equal(state[k], updated[k])}
            self.assertEqual(changed, {rgb.LAST_WEIGHT})
            self.assertGreater(stats['updates'][0]['grad_norm'], 0.)
        self.assertFalse(torch.equal(results[0][0][rgb.LAST_WEIGHT], results[1][0][rgb.LAST_WEIGHT]))


if __name__ == '__main__':
    unittest.main()
