import numpy as np
import torch

from paper_rsvp_deconvolution import complete_context, fit_kernels


def test_context_requires_both_successors_and_predecessors_for_one_epoch():
    edges = np.array([[0, 1, 1, 2, 0, 1, 3, 50], [1, 2, 2, 3, 0, 1, 2, 50],
                      [1, 1, 4, 0, 1, 2, 3, 50]])
    np.testing.assert_array_equal(complete_context(edges), np.array([[1, 2, 0, 2]]))


def test_tied_kernels_recover_individual_responses_when_three_images_overlap():
    generator = torch.Generator().manual_seed(12)
    own, old, new = torch.randn(3, 64, 3, generator=generator).unbind()
    early = torch.randn(3, 4, generator=generator)
    late = torch.randn(3, 4, generator=generator)
    first = own @ early + old @ late
    second = own @ late + new @ early
    actual = fit_kernels(own, old, new, first, second, penalty=0.)
    torch.testing.assert_close(actual[0], early, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], late, atol=1e-5, rtol=1e-5)
