import importlib.util
import unittest

import numpy as np
import torch
from torch import nn


class TinyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.block = nn.Module()
        layer = nn.Module()
        layer.norm = nn.LayerNorm(3)
        self.backbone.block.layers = nn.ModuleList([layer])
        self.backbone.proj_in = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            layer.norm.weight.copy_(torch.tensor([.75, 1.5, 2.]))
            layer.norm.bias.copy_(torch.tensor([-1., 2., .5]))

    def forward(self, x):
        return self.backbone.block.layers[-1].norm(self.backbone.proj_in(x))


class SpreadTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("paper_native_spread"), "Missing native spread calibration")
        self.study = importlib.import_module("paper_native_spread")

    def test_center_and_scale_use_different_native_and_shadow_distributions(self):
        native = torch.tensor([[0., 1.], [6., 5.]] * 3)
        shadow = torch.tensor([[100., -9.], [102., -7.]] * 3)
        visual = torch.tensor([[10., 1.], [14., 5.]] * 3)
        cal = self.study.moments(visual, native, shadow)
        self.assertEqual(cal["mean_visual"], [12., 3.])
        self.assertEqual(cal["mean_native"], [3., 3.])
        self.assertEqual(cal["mean_shadow"], [101., -8.])
        self.assertEqual(cal["gain"], 2.)
        shifted = self.study.moments(visual, native, shadow - 99.)
        self.assertEqual(shifted["gain"], 2.)
        with self.assertRaises(ValueError):
            self.study.moments(visual, native, shadow[:1].expand_as(shadow))

    def test_folded_state_reloads_into_an_ordinary_encoder_for_both_arms(self):
        torch.manual_seed(7)
        raw = TinyBrain().eval()
        x = torch.randn(9, 4)
        vis = torch.tensor([3., -1., 0.])
        pred = torch.tensor([0., 2., 1.])
        for gain in (1., 2.5):
            state = self.study.folded_state(raw, vis, pred, gain)
            folded = TinyBrain().eval()
            folded.load_state_dict(state, strict=True)
            torch.testing.assert_close(folded(x), vis + gain*(raw(x)-pred))
            torch.testing.assert_close(folded.backbone.proj_in.weight, raw.backbone.proj_in.weight)
        other = TinyBrain()
        other.backbone.block.layers[0].norm = nn.Identity()
        with self.assertRaises(ValueError):
            self.study.folded_state(other, vis, pred, 1.)

    def test_shadow_audit_rejects_row_disjoint_but_concept_overlapping_predictions(self):
        from paper_instance_residual import digest_array, inner_partition

        labels = np.repeat(np.arange(30), 10)
        rows = np.arange(len(labels)) + 500
        folds, parts = inner_partition(labels)
        def records():
            out = []
            for k in range(3):
                train = rows[(folds != k) & (parts == "fit")]
                pred = rows[folds == k]
                out.append(dict(fold=k, train_rows=len(train), predict_rows=len(pred),
                                train_rows_sha256=digest_array(train), predicted_rows_sha256=digest_array(pred)))
            return out
        proof = self.study.audit_oof(rows, labels, folds, parts, records())
        self.assertEqual(proof["predicted_rows"], 300)
        i = np.flatnonzero(parts == "fit")[0]
        folds[i] = (folds[i]+1) % 3
        with self.assertRaises(ValueError):
            self.study.audit_oof(rows, labels, folds, parts, records())


if __name__ == "__main__":
    unittest.main()
