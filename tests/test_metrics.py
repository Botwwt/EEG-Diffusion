import numpy as np
import pytest

from eeg_diffusion.metrics import image_two_way, paired_query_bootstrap, retrieval_top1


def test_pairwise_identification_is_not_gallery_top1():
    scores = np.array([[0.8, 0.9, 0.1], [0.9, 0.8, 0.1]])
    targets = np.array([0, 1])
    assert retrieval_top1(scores, targets).mean() == 0
    assert image_two_way(scores, targets).mean() == 0.5


def test_tie_and_target_exclusion():
    assert image_two_way([[1, 1, 0]], [0]).item() == 0.75
    assert retrieval_top1([[1, 1, 0]], [0]).item() == 1
    assert retrieval_top1([[1, 1, 0]], [1]).item() == 0


def test_paired_resampling_preserves_shared_variation():
    # Independently bootstrapping the two arms would invent variance here.
    a = np.array([0.2, 0.4, 0.6, 0.8])
    r = paired_query_bootstrap(a + 0.1, a, repetitions=1000, seed=19)
    assert np.allclose([r.difference, r.lower, r.upper], 0.1)


@pytest.mark.parametrize(
    "scores,targets", [([[1, np.nan]], [0]), ([[1, 2]], [2]), ([[1, 2]], [0.5])]
)
def test_invalid_inputs_are_explicit(scores, targets):
    with pytest.raises(ValueError):
        image_two_way(scores, targets)
