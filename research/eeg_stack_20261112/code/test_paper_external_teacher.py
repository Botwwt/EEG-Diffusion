import numpy as np
import torch

from paper_external_teacher import covered_donors, fused_teacher, safe_alignment


def test_alignment_training_cannot_include_the_target_fold_in_context():
    fold = 2
    context = np.array([1 << 1, 1 << 1 | 1 << 2, 1 << 2, 1 << 3, 1 << 4])
    actual = np.array([1, 1, 2, 3, 4])
    fit, predict = safe_alignment(context, actual, fold)
    assert fit.tolist() == [True, False, False, True, True]
    assert predict.tolist() == [False, False, True, False, False]


def test_same_object_wrong_teacher_has_exact_same_mask_and_balanced_coverage():
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    counts = np.array([2, 4, 0, 1, 0, 0, 5, 1])
    donor, support = covered_donors(labels, counts)
    take = np.flatnonzero(support)
    assert take.tolist() == [0, 1, 6, 7]
    assert sorted(donor[take].tolist()) == take.tolist()
    assert np.all(labels[donor[take]] == labels[take]) and np.all(donor[take] != take)
    assert np.array_equal(support[take], np.minimum(counts[take],counts[donor[take]]))
    assert np.all(donor[support == 0] == -1)


def test_combining_external_eeg_keeps_the_same_internal_signal_for_controls():
    internal = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    extra = torch.tensor([[0., 2.], [0., -3.], [1., 1.]])
    donor = torch.tensor([1, 0, -1])
    count = torch.tensor([3., 3., 0.])
    exact, different = fused_teacher(internal,extra,donor,count)
    assert exact[0,1] > 0 and different[0,1] < 0
    torch.testing.assert_close(exact[0],different[1])
    torch.testing.assert_close(exact[2],internal[2])
    torch.testing.assert_close(different[2],internal[2])
