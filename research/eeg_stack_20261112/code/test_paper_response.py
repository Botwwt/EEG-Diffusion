from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import paper_response as response


class ResponseTests(unittest.TestCase):
    def test_anchor_queries_are_disjoint_and_every_timestep_is_used(self):
        labels = np.repeat(np.arange(60), 4)
        anchors, pairs = response.make_plan(labels, (1, 5, 9))
        self.assertEqual((anchors, pairs), response.make_plan(labels, (1, 5, 9)))
        self.assertEqual(len(anchors), 32)
        self.assertEqual([a.mode for a in anchors].count("d2"), 16)
        donors = {labels[a.row] for a in anchors}
        self.assertEqual(len(donors), len(anchors))
        self.assertEqual(len(pairs), 160)
        self.assertEqual(len({(p.anchor, p.time) for p in pairs[:128]}), 128)
        within_by_time = [0] * 4
        for i, p in enumerate(pairs):
            self.assertNotIn(labels[p.left], donors)
            self.assertNotIn(labels[p.right], donors)
            self.assertNotEqual(p.left, p.right)
            self.assertEqual(labels[p.left] == labels[p.right], i % 4 == 3)
            if i % 4 == 3:
                within_by_time[p.time] += 1
        self.assertEqual(within_by_time, [10] * 4)

    def test_losses_ignore_shared_template_and_penalize_swapped_identity(self):
        target = torch.tensor([[2., -3., 1.], [-1., 1., 4.]])
        baseline = torch.tensor([[5., -9., 2.]])
        pos, pair = response.response_terms(target + baseline, target + baseline, baseline)
        self.assertLess(abs(float(pos)) + abs(float(pair)), 1e-6)
        student = (target.flip(0) + baseline).requires_grad_()
        pos, pair = response.response_terms(student, target + baseline, baseline)
        self.assertGreater(float(pair.detach()), .5)
        both = response.response_terms(student + 13, target + baseline + 13, baseline + 13)
        torch.testing.assert_close(pos, both[0])
        torch.testing.assert_close(pair, both[1])
        (pos + .5 * pair).backward()
        self.assertTrue(torch.isfinite(student.grad).all() and student.grad.norm() > 0)

    def test_shared_denoiser_state_preserves_timestep_dtype_and_token_gradient(self):
        from types import SimpleNamespace

        class Denoiser(torch.nn.Module):
            def forward(self, state, time, encoder_hidden_states, added_cond_kwargs, cross_attention_kwargs):
                self.inputs = state, time
                token = cross_attention_kwargs["ip_hidden_states"].flatten(1).sum(1)
                return SimpleNamespace(sample=state + token[:, None, None, None])

        unet = Denoiser()
        x = torch.ones(1, 4, 2, 2)
        t = torch.tensor([749.], dtype=torch.float64)
        tokens = torch.ones(2, 4, 3, requires_grad=True)
        prompts = (torch.zeros(1, 3, 4), torch.zeros(1, 4), torch.zeros(1, 6))
        y = response.denoise(unet, x, t, tokens, prompts)
        self.assertEqual(unet.inputs[1].dtype, t.dtype)
        torch.testing.assert_close(unet.inputs[0][0], unet.inputs[0][1])
        y.sum().backward()
        self.assertTrue((tokens.grad > 0).all())

    def test_capture_snapshots_mutable_inputs_and_removes_its_hook(self):
        class Unet(torch.nn.Module):
            def forward(self, state, time):
                return state

        unet = Unet()
        current = torch.zeros(1, 4, 2, 2)
        schedule = torch.tensor([999., 749., 499., 249.])

        def run():
            for i, t in enumerate(schedule):
                current.fill_(i)
                unet(current, t.reshape(1))

        states, times = response.capture_trajectory(unet, run)
        self.assertEqual(states[:, 0, 0, 0].tolist(), [0, 1, 2, 3])
        torch.testing.assert_close(times, schedule)
        self.assertFalse(unet._forward_pre_hooks)
        with self.assertRaisesRegex(ValueError, "four finite"):
            response.capture_trajectory(unet, lambda: unet(current, schedule[:1]))
        self.assertFalse(unet._forward_pre_hooks)

    def test_equal_writer_reuses_first_complete_bank_and_refuses_a_different_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "bank.pt"
            first = dict(complete=True, request_sha256="paired", value=torch.tensor([1]))
            second = dict(first, value=torch.tensor([2]))
            response.publish(file, first)
            winning = response.publish(file, second)
            torch.testing.assert_close(winning["value"], first["value"])
            with self.assertRaisesRegex(ValueError, "incompatible"):
                response.publish(file, dict(second, request_sha256="changed"))
            self.assertEqual(list(Path(tmp).iterdir()), [file])


if __name__ == "__main__":
    unittest.main()
