import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from paper_h14_value_probe import acquisition_h14
from paper_recording_transfer import AcquisitionReadout, select_external
from paper_render_cache import PinnedFiles, sha


class RecordingTransferTest(unittest.TestCase):
    def test_external_selection_cycles_actual_recordings(self):
        present = np.array([[True, False, True], [False, True, True], [True, False, False]])
        photo = np.array([2, 0, 1])
        offset = np.zeros(3, dtype=int)
        first = select_external(present, photo, offset)
        second = select_external(present, photo, offset + 1)
        self.assertTrue(present[first, photo].all())
        self.assertTrue(present[second, photo].all())
        np.testing.assert_array_equal(first, [0, 0, 1])
        np.testing.assert_array_equal(second, [1, 2, 1])

    def test_both_heads_share_the_same_early_path(self):
        torch.manual_seed(2)
        model = AcquisitionReadout()
        x = torch.randn(5, 17, 100)
        y = x.clone()
        y[..., 30:] = torch.randn(5, 17, 70)
        a, first = model(x)
        b, second = model(y)
        for left, right, target in zip(a, b, model.early(x[..., :30]), strict=True):
            torch.testing.assert_close(left, target)
            torch.testing.assert_close(left, right)
        for left, right in zip(first, second, strict=True):
            self.assertEqual(left.shape, (5, 1024))
            torch.testing.assert_close(left.norm(dim=-1), torch.ones(5))
            self.assertFalse(torch.equal(left, right))

    def test_render_requires_the_same_physical_eeg_and_row_order(self):
        registry = dict(split="a", development="b", labels="c")
        physical = dict(array_sha256="digest", receipt_sha256="receipt")
        rows = [10, 22]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "sub-01.npz"
            values = np.ones((2, 1024), dtype=np.float32)
            np.savez_compressed(result, row=rows, eeg1=values, eeg2=-values)
            receipt = dict(subject=1, rows=rows, splits=registry, repeats=list(range(4)),
                           eeg=dict(array="digest", manifest="receipt"), predictions=sha(result))
            result.with_suffix(".json").write_text(json.dumps(receipt), encoding="utf-8")
            read, _ = acquisition_h14(root, 1, rows, registry, physical, PinnedFiles())
            torch.testing.assert_close(read["eeg1"], -read["eeg2"])
            changed = dict(physical, receipt_sha256="unmatched")
            with self.assertRaisesRegex(ValueError, "physical queries"):
                acquisition_h14(root, 1, rows, registry, changed, PinnedFiles())
            with self.assertRaisesRegex(ValueError, "physical queries"):
                acquisition_h14(root, 1, rows[::-1], registry, physical, PinnedFiles())
