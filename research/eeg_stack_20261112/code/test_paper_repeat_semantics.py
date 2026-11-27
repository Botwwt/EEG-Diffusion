import numpy as np
import torch

from paper_repeat_semantics import (
    apply_language_targets,
    matched_margin,
    paired_batches,
    select_candidate,
    split_trials,
)


def test_two_views_are_disjoint_and_use_one_trial_from_each_actual_session():
    sessions = torch.tensor([[3, 0, 3, 0], [2, 1, 1, 2]])
    n = len(sessions)
    index = torch.arange(n)[:, None]
    sides = split_trials(sessions, torch.tensor([[1, 0], [0, 1]]))
    assert sides.shape == (n, 2, 2)
    assert sides.flatten(1).sort(1).values.tolist() == [[0, 1, 2, 3]] * n
    assert torch.equal(sessions[index, sides[:, 0]], sessions[index, sides[:, 1]])
    assert torch.all(sessions[index, sides[:, 0]][:, 0] != sessions[index, sides[:, 0]][:, 1])


def test_matching_cost_prefers_correct_correspondence_within_each_object():
    x = torch.tensor([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]]])
    aligned = matched_margin(x, x)
    exchanged = matched_margin(x, x.flip(1))
    assert exchanged > aligned + 10.
    common = x.mean(1, keepdim=True).expand_as(x)
    torch.testing.assert_close(matched_margin(common, common), torch.tensor(np.log(2), dtype=x.dtype))


def test_paired_batches_leave_no_registered_photo_out():
    members = np.arange(60).reshape(6, 10)
    batches = list(paired_batches(members, 1, 4, 17))
    assert np.array_equal(np.sort(np.concatenate(batches, axis=None)), members.ravel())
    for pairs in batches:
        assert len(np.unique(pairs[:, 0] // 10)) == len(pairs)
        assert np.array_equal(pairs[:, 0] // 10, pairs[:, 1] // 10)
        assert np.all(pairs[:, 0] != pairs[:, 1])


def test_identical_name_vectors_share_one_training_positive():
    names = torch.tensor([[1., 0., 1.], [0., 1., 1.], [1., 0., 1.]])
    ids = torch.tensor([0, 0, 1, 2, 1, 2])
    prototypes, targets = apply_language_targets(names, ids)
    assert prototypes.shape == (2, 3)
    assert targets[0] == targets[3]
    assert targets[2] == targets[4] and targets[2] != targets[0]


def test_pooled_object_gain_cannot_hide_loss_on_one_participant():
    def by(objects, photos):
        return {i: {"full_object": a, "same_object_photo": b}
                for i, a, b in zip((1, 3), objects, photos, strict=True)}
    native = by((4, 5), (40, 40))
    base = by((5, 5), (39, 40))
    comparisons = {"native-paper": native, "photo": base, "repeat": by((3, 25), (43, 44)),
                   "language": by((7, 7), (43, 35)), "language-repeat": by((9, 8), (30, 30))}
    assert select_candidate(comparisons) == "language"
    comparisons["language"] = by((7, 7), (35, 35))
    assert select_candidate(comparisons) is None
