import numpy as np
import torch

from paper_d2_object_swap import DONOR, exchange_loss, make_episodes, photo_loss


def test_episodes_are_balanced_reproducible_and_have_two_distinct_objects():
    groups = np.arange(24).reshape(6, 4)
    appearance = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    subjects, rows, seeds = make_episodes(groups, appearance, steps=20)
    repeat = make_episodes(groups, appearance, steps=20)
    for array, reproduced in zip((subjects, rows, seeds), repeat, strict=True):
        np.testing.assert_array_equal(array, reproduced)
    assert np.bincount(subjects, minlength=10).tolist() == [2] * 10
    assert all(len(set(row)) == 4 for row in rows)
    objects = rows // 4
    assert np.all(objects[:, 0] == objects[:, 1]) and np.all(objects[:, 2] == objects[:, 3])
    assert np.all(objects != objects[:, DONOR])


def test_matched_noise_and_layout_cancel_common_object_margin():
    labels = torch.tensor([0, 0, 1, 1])
    donor = labels[list(DONOR)]
    logits = torch.nn.functional.one_hot(labels, 2).float()
    swapped = torch.nn.functional.one_hot(donor, 2).float()
    bias = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    matched = exchange_loss(logits, swapped, labels, donor)
    torch.testing.assert_close(matched, exchange_loss(logits + bias, swapped + bias, labels, donor))
    assert matched < exchange_loss(swapped, logits, labels, donor)


def test_photo_loss_compares_only_same_object_photographs():
    similarity = torch.eye(4)
    changed = similarity.clone()
    changed[:2, 2:] = 10.
    changed[2:, :2] = 9.
    torch.testing.assert_close(photo_loss(similarity), photo_loss(changed))
    changed[0, 1] = 2.
    assert photo_loss(changed) > photo_loss(similarity)
