from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

import paper_peer_teacher as peer


class CrossingTests(unittest.TestCase):
    def test_mean_of_four_and_time_bins_preserve_channel_order(self):
        x = torch.arange(2 * 4 * 17 * 250, dtype=torch.float32).reshape(2, 4, 17, 250)
        z = peer.binned_eeg(x)
        self.assertEqual(z.shape, (2, 850))
        for channel, sample in [(0, 0), (3, 17), (16, 49)]:
            torch.testing.assert_close(z[:, channel * 50 + sample], x[:, :, channel, sample * 5: sample * 5 + 5].mean((1, 2)))

    def test_concept_folds_and_derangement_are_stable_under_row_reordering(self):
        rows = np.array([61, 23, 4, 77, 9, 13, 16, 21, 18, 73, 71, 19])
        labels = np.repeat([11, 24, 6, 57, 3, 14], 2)
        f = peer.concept_partition(labels)
        p = peer.sibling_permutation(rows, labels)
        order = np.arange(len(rows))[::-1]
        np.testing.assert_equal(f[order], peer.concept_partition(labels[order]))
        np.testing.assert_equal(rows[p][order], rows[order][peer.sibling_permutation(rows[order], labels[order])])
        self.assertEqual(sorted(p), list(range(len(rows))))
        self.assertFalse(np.any(p == np.arange(len(rows))))
        np.testing.assert_equal(labels[p], labels)
        for label in set(labels):
            self.assertEqual(len(set(f[labels == label])), 1)
        np.testing.assert_equal(np.bincount(f), [4, 4, 4])

    def test_oof_matches_explicit_ridge_and_is_independent_of_its_held_targets(self):
        rng = np.random.default_rng(20)
        x = rng.normal(size=(30, 6)); x[:, -1] = 4.
        y = rng.normal(size=(30, 4))
        folds = peer.concept_partition(np.repeat(np.arange(10), 3))
        x_t, y_t = torch.tensor(x), torch.tensor(y)
        predicted, stats = peer.oof_predict(x_t, y_t, folds)
        self.assertEqual(predicted.dtype, torch.float32)
        for h in range(3):
            tr, va = folds != h, folds == h
            mean, std = x[tr].mean(0), np.maximum(x[tr].std(0), 1e-6)
            a = (x[tr] - mean) / std
            cov = a.T @ a / sum(tr)
            reg = np.trace(cov) / cov.shape[0]
            w = np.linalg.solve(cov + np.eye(x.shape[1]) * reg, a.T @ (y[tr] - y[tr].mean(0)) / sum(tr))
            np.testing.assert_allclose(predicted.numpy()[va], ((x[va]-mean)/std) @ w + y[tr].mean(0), atol=1e-6)
            self.assertAlmostEqual(stats[h]['lambda'], reg, places=8)
        alternate = y_t.clone(); alternate[folds == 0] += 1e5
        after, _ = peer.oof_predict(x_t, alternate, folds)
        torch.testing.assert_close(predicted[folds == 0], after[folds == 0], atol=0, rtol=0)

    def test_other_people_only_and_sibling_views_share_center(self):
        torch.manual_seed(23)
        subject_ids = list(range(1, 11))
        matrix = torch.randn(10, 30, 12).half()
        own, ids = peer.other_people(matrix, subject_ids, 3, torch.device('cpu'))
        self.assertEqual(ids, [1, 2, 4, 5, 6, 7, 8, 9, 10])
        torch.testing.assert_close(own, matrix[[0, 1, 3, 4, 5, 6, 7, 8, 9]].float().mean(0))
        matrix[2].fill_(float('nan'))
        after, _ = peer.other_people(matrix, subject_ids, 3, torch.device('cpu'))
        torch.testing.assert_close(own, after, atol=0, rtol=0)
        perm = peer.sibling_permutation(np.arange(30), np.repeat(np.arange(3), 10))
        torch.testing.assert_close(own.mean(0), own[perm].mean(0))
        reference = torch.eye(4)
        center = reference.mean(0)
        tokens = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4)
        token, aligned = peer.peer_losses(tokens, tokens, tokens.mean(0), torch.ones(2), reference, reference, center)
        other_token, shuffled = peer.peer_losses(tokens, tokens, tokens.mean(0), torch.ones(2), reference, reference.roll(1, 0), center)
        torch.testing.assert_close(token, other_token)
        normalized = F.normalize(reference - center)
        expected = F.cross_entropy(normalized @ normalized.T / .07, torch.arange(4))
        torch.testing.assert_close(aligned, expected)
        self.assertLess(aligned.item(), shuffled.item())

    def test_cached_row_order_sources_and_all_ten_prediction_hashes_are_bound(self):
        rows = np.arange(9, dtype=np.int64) * 3
        labels = np.repeat([4, 7, 9], 3)
        folds = peer.concept_partition(labels)
        perm = peer.sibling_permutation(rows, labels)
        sources = {'eeg': 'ten inputs', 'F': 'visual data'}
        pred = torch.randn(10, len(rows), 1024, generator=torch.Generator().manual_seed(12)).half()
        pack = dict(schema=1, subject_ids=peer.SUBJECTS, sources=sources, predictions=pred,
                    rows=torch.from_numpy(rows.copy()), labels=torch.from_numpy(labels), folds=torch.from_numpy(folds),
                    sibling_rows=torch.from_numpy(rows[perm]),
                    prediction_sha256={str(i): peer.array_hash(pred[i - 1].numpy()) for i in peer.SUBJECTS})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.pt'
            torch.save(pack, path)
            data = peer.read_cache(path, rows, labels, folds, perm, sources)
            torch.testing.assert_close(data['predictions'], pred)
            with self.assertRaises(ValueError):
                peer.read_cache(path, rows[::-1], labels, folds, perm, sources)
            with self.assertRaises(ValueError):
                peer.read_cache(path, rows, labels, folds, perm, dict(sources, F='changed'))
            pack['predictions'][9, 3, 17] += 4
            torch.save(pack, path)
            with self.assertRaises(ValueError):
                peer.read_cache(path, rows, labels, folds, perm, sources)


if __name__ == '__main__':
    unittest.main()
