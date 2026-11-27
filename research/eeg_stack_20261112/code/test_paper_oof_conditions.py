import numpy as np
import pytest
import torch

from paper_oof_conditions import condition_stats, concept_splits


def test_concept_splits_never_train_on_a_predicted_concept():
    labels = np.repeat(np.arange(8), 10)
    folds = np.tile(np.arange(1, 5).repeat(10), 2)
    assigned = []
    for held, fit, predict in concept_splits(labels, folds):
        assert np.all(folds[predict] == held)
        assert np.intersect1d(labels[fit], labels[predict]).size == 0
        assigned.extend(predict)
    assert sorted(assigned) == list(range(80))
    bad = folds.copy()
    bad[0] = 2
    with pytest.raises(ValueError, match="concept"):
        concept_splits(labels, bad)


def test_condition_stats_measure_angle_and_signal_radius_separately():
    target = torch.tensor([[3., 2.], [1., 2.], [2., 4.], [2., 0.]])
    noisy = target.mean(0) + .25 * (target - target.mean(0))
    stats = condition_stats(noisy, target)
    assert stats["mean_centered_target_cos"] == pytest.approx(1.)
    assert stats["radius_over_target_rms"] == pytest.approx(.25)
    assert stats["rmse"] > 0
