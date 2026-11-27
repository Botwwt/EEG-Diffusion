import torch

from paper_paired_token_flow import choose_confusers, pair_penalty, paired_path, solve_flow


def test_pair_path_shares_initial_noise_with_both_photographs():
    noise = torch.tensor([[2., 5.], [1., 3.]])
    target = torch.tensor([[[4., 8.], [0., 2.]], [[7., 9.], [3., 8.]]])
    state, velocity = paired_path(noise, target, torch.tensor([0., .5]))
    torch.testing.assert_close(state[0], noise[:1].expand(2, -1))
    torch.testing.assert_close(state[1], (noise[1] + target[1]) / 2)
    torch.testing.assert_close(state + torch.tensor([1., .5])[:, None, None] * velocity, target)


def test_pair_penalty_prefers_correct_cross_object_assignment():
    target = torch.tensor([[[1., 0.], [-1., 0.]]])
    own = pair_penalty(target, target.flip(1), target)
    swapped = pair_penalty(target.flip(1), target, target)
    assert own.item() < .01 and swapped.item() > 1.
    torch.testing.assert_close(pair_penalty(target.flip(1), target, target.flip(1)), own)


def test_confusers_exclude_every_photo_of_the_same_object():
    visual = torch.tensor([[1., 0.], [1., 0.], [1., 1.], [-1., 0.], [0., -1.]])
    labels = torch.tensor([0, 0, 1, 2, 2])
    choices = choose_confusers(visual, labels, count=2)
    assert choices.shape == (5, 2)
    assert (labels[choices] != labels[:, None]).all()
    assert choices[0, 0].item() == 2


def test_flow_integrates_a_constant_conditional_field_once_to_its_endpoint():
    initial = torch.tensor([[.3, 1.], [-.7, 2.]])
    signal = torch.tensor([[.2, .4], [.6, -1.]])
    times = []

    def field(state, time, condition):
        times.append(float(time[0]))
        return condition

    torch.testing.assert_close(solve_flow(field, initial, signal, steps=4), initial + signal)
    assert len(times) == 8 and times[0] == 0. and times[-1] == 1.
