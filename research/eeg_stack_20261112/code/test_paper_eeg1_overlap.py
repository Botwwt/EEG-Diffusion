import numpy as np
import torch

from paper_eeg1_overlap import response_design


def test_neighbor_design_never_includes_target_and_is_causal():
    geometry = (np.array([0, 0, 0]), np.array([0, 1, 2]),
                np.array([0., .10, -.10]), np.array([True, False, False]))
    picture = torch.eye(3)
    _, neighbors = response_design(1, geometry, picture, np.ones(3, dtype=bool))
    traced = neighbors.reshape(1, 12, 15, 3)[0]
    assert torch.count_nonzero(traced[...,0]) == 0
    assert torch.count_nonzero(traced[:6,:,1]) == 0
    assert traced[6,0,1] == 1
    assert torch.count_nonzero(traced[0,:,2]) == 1


def test_neighbor_picture_from_a_held_fold_contributes_nothing():
    geometry = (np.array([0,0]), np.array([0,1]), np.array([0.,.10]), np.array([True,False]))
    total, neighbors = response_design(1, geometry, torch.eye(2), np.array([True,False]))
    assert torch.count_nonzero(neighbors) == 0
    assert torch.count_nonzero(total[...,0::2]) > 0
    assert torch.count_nonzero(total[...,1::2]) == 0
