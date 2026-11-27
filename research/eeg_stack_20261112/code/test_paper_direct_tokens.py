import numpy as np
import torch

from paper_direct_tokens import (
    RelativeTokenEncoder,
    coefficient_loss,
    coordinate_weights,
    photo_batches,
    restore_relative,
    visual_basis,
)


def test_weight_uses_both_held_object_recovery_and_visual_object_variance():
    weight = coordinate_weights(torch.tensor([.2, .2, -.5, .0]), torch.tensor([1., .01, 1., 0.]))
    assert weight[0] > weight[1] > weight[2] == weight[3] > 0
    torch.testing.assert_close(weight.mean(), torch.tensor(1.))
    torch.testing.assert_close(coordinate_weights(torch.zeros(4), torch.ones(4)), torch.ones(4))


def test_pca_reassembles_true_relative_tokens_without_changing_the_query_mean():
    generator = torch.Generator().manual_seed(81)
    x = torch.randn(30, 4, 5, generator=generator)
    x = x - x.mean(dim=1, keepdim=True)
    basis, coefficients = visual_basis(x, 15)
    reproduced = restore_relative(coefficients, basis)
    torch.testing.assert_close(reproduced, x, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(reproduced.mean(dim=1), torch.zeros_like(x[:, 0]), atol=1e-6, rtol=0)
    guess = restore_relative(coefficients, basis, calibrate=True)
    torch.testing.assert_close((guess - basis["center"]).flatten(1).norm(dim=1),
                               basis["radius"].expand(len(x)))


def test_batches_cover_exact_photographs_and_preserve_index_alignment():
    rows = np.arange(21).reshape(7, 3)
    first = list(photo_batches(rows, 5, 3, 4, 31))
    second = list(photo_batches(rows, 5, 3, 4, 31))
    assert len(first) == 6
    assert np.array_equal(np.sort(np.concatenate([chosen for _, chosen in first])), rows.ravel())
    for ((viewers, selected), (same_viewers, same_selected)) in zip(first, second, strict=True):
        assert len(set(viewers)) == 4
        assert len(set(selected // 3)) == len(selected)
        np.testing.assert_array_equal(viewers, same_viewers)
        np.testing.assert_array_equal(selected, same_selected)


def test_same_photo_has_one_positive_even_with_distinct_eeg_viewers():
    truth = torch.eye(7)
    matched = truth.expand(4, -1, -1)
    aligned = coefficient_loss(matched, truth, torch.ones(7))
    switched = coefficient_loss(matched.roll(1, dims=1), truth, torch.ones(7))
    assert aligned.item() < .01
    assert switched.item() > aligned.item() + 5


def test_encoder_reads_one_montage_and_produces_one_prediction_per_query():
    center = torch.zeros(3, 17, 40)
    model = RelativeTokenEncoder(center, torch.ones_like(center), 19).eval()
    response = model(torch.zeros(4, 17, 40), torch.tensor([2, 1, 0, 2]))
    assert response.shape == (4, 19)
