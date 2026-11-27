import torch

from paper_forward_guidance import bounded_descent, forward_weights, update_condition


def test_forward_weights_apply_trial_noise_precision_to_observed_and_predicted_eeg():
    visual = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    eeg = visual @ torch.tensor([[2., 1.], [.1, -1.]])
    half_difference = torch.tensor([[2., 1.], [3., .5], [1., 2.], [.5, 1.]])
    fit = forward_weights(visual, eeg, half_difference)

    native = fit["plain"]
    torch.testing.assert_close(fit["weighted"] @ fit["noise_factor"].T, native)


def test_bounded_descent_updates_each_preview_toward_its_own_observation():
    x = torch.ones(2, 3, 2, 2)
    target = torch.tensor([0., 3.])[:, None, None, None]
    with torch.no_grad():
        delta = bounded_descent(x, lambda preview: (preview - target).square().mean(), .15)

    torch.testing.assert_close(delta.flatten(1).square().mean(1).sqrt(), torch.full((2,), .15))
    assert float((x + delta - target).square().mean()) < float((x - target).square().mean())
    assert delta[0].max() < 0 and delta[1].min() > 0


def test_condition_update_preserves_the_unobserved_direction_and_caps_displacement():
    initial = torch.tensor([[2., 3.]])
    bridge = torch.tensor([[1.], [0.]])
    lifted = update_condition(initial, torch.zeros(2), bridge, torch.eye(1), torch.tensor([[20.]]))
    assert lifted[0, 0] > initial[0, 0]
    assert lifted[0, 1] == initial[0, 1]
    torch.testing.assert_close((lifted - initial).norm(), initial.norm() * .5)
