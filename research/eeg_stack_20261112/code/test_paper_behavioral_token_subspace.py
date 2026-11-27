import numpy as np
import pytest
import torch

from paper_behavioral_token_subspace import attribute_axes, replace_coordinates
from paper_prepare_behavior_dimensions import aligned_attributes


def test_replacement_retains_directions_outside_the_attribute_space():
    axes = torch.tensor([[1., 0.], [0., 0.], [0., 1.]])
    native = torch.tensor([[2., 4., 6.], [3., 5., 7.]])
    readout = torch.tensor([[10., 30., 50.], [11., 40., 51.]])
    actual = replace_coordinates(native, readout, axes)
    torch.testing.assert_close(actual, torch.tensor([[10., 4., 50.], [11., 5., 51.]]))


def test_attributes_find_a_centered_orthonormal_subspace():
    tokens = torch.tensor([[1., 0., 7.], [-1., 0., 7.], [0., 1., 7.], [0., -1., 7.]])
    attribute = tokens[:, :2] + torch.tensor([31., 13.])
    axes = attribute_axes(tokens, attribute)
    torch.testing.assert_close(axes.T @ axes, torch.eye(2))
    torch.testing.assert_close(axes @ axes.T, torch.diag(torch.tensor([1., 1., 0.])))


def test_behavioral_rows_are_reordered_by_exact_filename_and_keep_only_registered_photos():
    original = ["delta.jpg", "bc.jpg", "c.jpg"]
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    selected, rows = aligned_attributes(original, values, ["c.jpg", "delta.jpg"])
    np.testing.assert_array_equal(rows, [2, 0])
    np.testing.assert_array_equal(selected, values[[2, 0]])
    with pytest.raises(ValueError, match="not found"):
        aligned_attributes(original, values, ["b.jpg"])
