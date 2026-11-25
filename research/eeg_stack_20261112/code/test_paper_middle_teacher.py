import importlib
from pathlib import Path
import unittest

import numpy as np
import torch


class MiddleTeacherTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("paper_middle_teacher"), "The two new fitting arms are missing")
        self.study = importlib.import_module("paper_middle_teacher")

    def test_manifest_excludes_the_held_fold_and_matches_photographs_to_eeg(self):
        labels = np.repeat(np.arange(3), 10)
        folds = np.array([1, 0, 2])
        ids = [f"photo_{i}" for i in range(30)]
        paths = [f"training_images/class{labels[i]}/{s}.jpg" for i, s in enumerate(ids)]
        rows = np.r_[0:10, 20:30]
        self.assertEqual(self.study.aligned_members(rows, labels, folds, ids, paths, ids), paths[:10] + paths[20:])
        with self.assertRaises(ValueError):
            self.study.aligned_members(np.r_[0:9, 10, 20:30], labels, folds, ids, paths, ids)
        paths[1], paths[2] = paths[2], paths[1]
        with self.assertRaises(ValueError):
            self.study.aligned_members(rows, labels, folds, ids, paths, ids)

    def test_one_image_forward_pools_only_the_block12_patches_and_retains_final_projection(self):
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        torch.manual_seed(42)
        cfg = CLIPVisionConfig(hidden_size=12, intermediate_size=16, projection_dim=6, num_attention_heads=3,
                               num_hidden_layers=13, image_size=8, patch_size=4)
        net = CLIPVisionModelWithProjection(cfg).eval().requires_grad_(False)
        px = torch.randn(2, 3, 8, 8)
        with torch.inference_mode():
            middle, final = self.study.visual_views(net, px)
            ref = net(px, output_hidden_states=True)
        torch.testing.assert_close(middle, ref.hidden_states[12][:, 1:].mean(1))
        torch.testing.assert_close(final, ref.image_embeds)
        self.assertFalse(torch.allclose(middle, ref.hidden_states[12].mean(1)))

    def test_train_only_pca_uses_identity_preserving_full_rank_control(self):
        torch.manual_seed(100)
        v = torch.randn(25, 8, dtype=torch.float32)
        projection, retained = self.study.pca_view(v, 8)
        unit = torch.nn.functional.normalize(v.double())
        expected = torch.nn.functional.normalize(unit-unit.mean(0))
        torch.testing.assert_close(projection @ projection.T, (expected @ expected.T).float(), atol=1.e-5, rtol=1.e-5)
        self.assertAlmostEqual(retained, 1., places=6)
        compressed, retained = self.study.pca_view(v, 5)
        self.assertEqual(compressed.shape, (25, 5))
        self.assertTrue(0 < retained < 1)

    def test_correct_auxiliary_pairs_score_better_while_the_token_objective_is_unchanged(self):
        eeg = torch.eye(4, requires_grad=True)
        tokens = torch.arange(24, dtype=torch.float32).view(4, 2, 3)
        prediction = tokens.flip(-1)
        center = tokens.mean(0)
        variance = (tokens - center).square().mean((0, 2))
        token_loss, aligned = self.study.losses(prediction, tokens, center, variance, eeg, torch.eye(4))
        other_token, wrong = self.study.losses(prediction, tokens, center, variance, eeg, torch.eye(4).roll(1, 0))
        self.assertEqual(float(token_loss), float(other_token))
        self.assertLess(float(aligned.detach()), float(wrong.detach()))
        wrong.backward()
        self.assertGreater(float(eeg.grad.norm()), 0.)


if __name__ == "__main__":
    unittest.main()
