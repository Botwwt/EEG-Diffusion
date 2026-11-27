import numpy as np
import torch

from paper_rsvp_context_training import context_changes, context_donors, lift_changes
from paper_rsvp_deconvolution import wave_bins


def test_context_donors_are_balanced_within_session_and_fold():
    groups = np.repeat(np.arange(4), 12)
    concepts = np.tile(np.repeat(np.arange(6), 2), 4)
    first, second = context_donors(groups, concepts)
    for donor in first, second:
        np.testing.assert_array_equal(np.sort(donor), np.arange(len(groups)))
        np.testing.assert_array_equal(groups[donor], groups)
        assert np.all(concepts[donor] != concepts)
    assert np.all(first != second)


def test_true_change_replaces_context_and_control_matches_its_strength():
    current = torch.arange(5., 13.).reshape(4, 2)
    context = torch.tensor([[1., 0.], [2., 0.], [4., 1.], [2., 5.]])
    replacement, placebo = [2, 3, 0, 1], [3, 2, 1, 0]
    exchanged, shuffled = context_changes(context, replacement, placebo)
    torch.testing.assert_close(current + context + exchanged, current + context[replacement])
    torch.testing.assert_close(exchanged.norm(dim=1), shuffled.norm(dim=1))
    assert not torch.allclose(exchanged, shuffled)


def test_temporal_adjustment_is_confined_to_the_two_modeled_windows():
    changes = torch.arange(680.).reshape(2, 2, 170)
    lifted = lift_changes(changes)
    assert lifted.shape == (2, 17, 250)
    torch.testing.assert_close(wave_bins(lifted), changes)
    assert torch.count_nonzero(lifted[..., 100:]) == 0
