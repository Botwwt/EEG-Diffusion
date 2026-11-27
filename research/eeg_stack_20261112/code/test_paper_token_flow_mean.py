import torch

from paper_token_flow_mean import expected_endpoint


def test_expected_endpoint_cancels_independent_noise_with_paired_starts():
    context = torch.tensor([[1.5, .5], [-1., 2.]])
    noise = torch.tensor([[[1., .7], [3., 0.]], [[-1., .4], [2., 9.]]])

    def initial_velocity(state, time, condition):
        assert time.count_nonzero() == 0
        return condition - state

    torch.testing.assert_close(expected_endpoint(initial_velocity, context, noise), context, atol=1e-6, rtol=0.)
