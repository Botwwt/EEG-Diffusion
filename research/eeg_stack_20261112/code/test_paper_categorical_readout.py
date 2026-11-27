import torch

from paper_categorical_readout import decode_posterior, quantize
from paper_h14_value_probe import match_relative_norm


def test_residual_selection_uses_previous_stage_reconstruction():
    codebooks = torch.tensor([[[0.0, 0.0], [4.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]])
    values = torch.tensor([[4.0, 2.0], [0.0, 0.0]])
    ids = quantize(values, codebooks)
    torch.testing.assert_close(ids, torch.tensor([[1, 1], [0, 0]]))


def test_exact_mean_and_mode_use_same_factorized_posterior():
    codebooks = torch.tensor([[[0.0, 0.0], [4.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]])
    probabilities = torch.tensor([[[0.2, 0.8], [0.75, 0.25]], [[0.9, 0.1], [0.25, 0.75]]])
    mean, mode = decode_posterior(probabilities, codebooks)
    torch.testing.assert_close(mean, torch.tensor([[3.2, 0.5], [0.4, 1.5]]))
    torch.testing.assert_close(mode, torch.tensor([[4.0, 0.0], [0.0, 2.0]]))
    altered = probabilities.clone()
    altered[1] = altered[1].flip(-1)
    for before, after in zip((mean, mode), decode_posterior(altered, codebooks), strict=True):
        torch.testing.assert_close(before[0], after[0])


def test_matching_relative_norm_keeps_the_candidate_direction_and_mean():
    reference = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]])
    candidate = torch.tensor([[[2.0, 7.0], [2.0, 5.0]]])
    comparable = match_relative_norm(candidate, reference)
    torch.testing.assert_close(comparable, torch.tensor([[[2.0, 8.0], [2.0, 4.0]]]))
