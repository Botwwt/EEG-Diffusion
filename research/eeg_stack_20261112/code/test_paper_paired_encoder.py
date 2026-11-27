import numpy as np
import torch

from paper_paired_encoder import (
    PairedEncoder,
    factorized_pair_loss,
    matching_batches,
    recording_basis,
    scale_condition,
    visual_components,
)


def test_batches_select_distinct_people_and_distinct_photographs_of_each_object():
    members = np.arange(50).reshape(5, 10)
    batches = list(matching_batches(members, people=4, epochs=6, concepts_per_batch=2, seed=12))
    assert len(batches) == 18
    for first, second, rows in batches:
        assert 0 <= first < 4 and 0 <= second < 4 and first != second
        assert np.array_equal(rows // 10, rows[:, :1] // 10 + np.zeros_like(rows))
        assert np.all(rows[:, 0] != rows[:, 1])
        assert len(set(rows[:, 0] // 10)) == len(rows)
    replay = list(matching_batches(members, people=4, epochs=6, concepts_per_batch=2, seed=12))
    assert all(a == c and b == d and np.array_equal(x, y)
               for (a, b, x), (c, d, y) in zip(batches, replay, strict=True))


def test_photo_residual_is_centered_within_each_object_and_recomposes_its_photo():
    fused = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    members = torch.tensor([[0, 3, 4, 7], [1, 2, 5, 6]])
    center, common, residual = visual_components(fused, members)
    assert torch.allclose(residual[members].mean(1), torch.zeros(2, 4))
    assert torch.equal(fused, center + common + residual)


def test_same_object_loss_preserves_which_photo_was_seen_by_both_people():
    midpoint = torch.eye(5)[:, None, :].expand(-1, 2, -1)
    first = torch.eye(5)
    photo = torch.stack((first, -first), 1)
    common = torch.stack((midpoint, midpoint))
    aligned = factorized_pair_loss(common, torch.stack((photo, photo)))
    swapped = factorized_pair_loss(common, torch.stack((photo, photo.flip(1))))
    reversed_both = factorized_pair_loss(common, torch.stack((photo.flip(1), photo.flip(1))))
    assert swapped > aligned + 2
    torch.testing.assert_close(aligned, reversed_both)


def test_comparing_residual_directions_is_invariant_to_each_photograph_pairs_scale():
    rng = torch.Generator().manual_seed(23)
    common = torch.randn(2, 4, 2, 19, generator=rng)
    residual = torch.randn(2, 4, 2, 19, generator=rng)
    scales = torch.tensor([[.2, 1., 3., 5.], [4., .5, .1, 2.]])[:, :, None, None]
    torch.testing.assert_close(factorized_pair_loss(common, residual),
                               factorized_pair_loss(common, residual * scales))


def test_encoder_has_one_query_output_and_two_supervised_components():
    mean = torch.zeros(3, 17, 40)
    encoder = PairedEncoder(mean, torch.ones_like(mean), torch.zeros(24), torch.ones(2)).eval()
    common, residual = encoder(torch.randn(5, 17, 40), torch.tensor([0, 1, 2, 0, 1]))
    assert common.shape == residual.shape == (5, 24)


def test_same_photo_differences_isolate_recording_offsets_without_a_query_session():
    source = torch.eye(4)
    incidence = source[[0, 0, 1, 1]] - source[[2, 3, 2, 3]]
    offsets = torch.cat((source, torch.zeros(4, 2)), dim=1)
    basis = recording_basis(incidence, incidence @ offsets, torch.ones(6))
    difference = incidence @ offsets
    torch.testing.assert_close((difference @ basis) @ basis.T, difference)
    assert not basis[4:].any()


def test_training_radius_calibrates_each_query_and_preserves_its_direction():
    center = torch.tensor([1., -1.])
    direction = torch.tensor([[3., 4.], [-2., 0.], [0., 0.]])
    calibrated = scale_condition(center, direction, 10.)
    torch.testing.assert_close(calibrated - center, torch.tensor([[6., 8.], [-10., 0.], [0., 0.]]))
