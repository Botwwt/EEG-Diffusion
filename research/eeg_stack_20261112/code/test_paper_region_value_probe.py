from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from paper_region_value_probe import MASK_RECEIPTS, REFERENCE_VIEWS, appearance_views, load_views
from paper_render_cache import PinnedFiles, save_json, sha


class RegionValueTest(unittest.TestCase):
    def test_complementary_regions_receive_the_same_opacity_budget(self):
        image = torch.ones(2, 3, 16, 16)
        image[:, :, ::2] = 0
        mask = torch.zeros(2, 1, 16, 16)
        mask[0, :, :4] = 1
        mask[1, :, :12] = 1
        coarse, foreground, background, weights = appearance_views(image, mask)

        self.assertTrue(torch.equal(weights[0].sum((1, 2, 3)), weights[1].sum((1, 2, 3))))
        self.assertTrue(torch.equal(foreground, coarse + weights[0] * (image - coarse)))
        self.assertTrue(torch.equal(background, coarse + weights[1] * (image - coarse)))
        self.assertTrue(torch.equal(foreground[0, :, 4:], coarse[0, :, 4:]))
        self.assertTrue(torch.equal(background[1, :, :12], coarse[1, :, :12]))

    def test_empty_binary_region_adds_no_detail_to_either_view(self):
        image = torch.rand(2, 3, 16, 16, generator=torch.Generator().manual_seed(3))
        mask = torch.zeros(2, 1, 16, 16)
        mask[1] = 1
        coarse, foreground, background, weights = appearance_views(image, mask)

        self.assertTrue(all(torch.isfinite(w).all() and torch.count_nonzero(w) == 0 for w in weights))
        self.assertTrue(torch.equal(coarse, foreground) and torch.equal(coarse, background))

    def test_reference_bank_must_follow_the_held_rows_and_scope(self):
        with TemporaryDirectory() as scratch:
            bank = Path(scratch) / "dev.npz"
            rows = np.array([2, 3])
            registry = {"development": "fixed"}
            np.savez_compressed(bank, row=rows, **{name: np.ones((2, 1024), np.float32) for name in REFERENCE_VIEWS})
            save_json(bank.with_suffix(".json"), dict(phase="dev", registration=registry, mask=MASK_RECEIPTS["dev"],
                                                     archive_sha256=sha(bank), scope="reference-only"))

            features, _ = load_views(bank, "dev", rows, registry, PinnedFiles())
            self.assertEqual(tuple(features), REFERENCE_VIEWS)
            with self.assertRaisesRegex(ValueError, "ordering"):
                load_views(bank, "dev", rows[::-1], registry, PinnedFiles())
            with self.assertRaisesRegex(ValueError, "photo panel"):
                load_views(bank, "train", rows, registry, PinnedFiles())
