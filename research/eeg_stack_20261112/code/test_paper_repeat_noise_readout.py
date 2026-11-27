import torch

from paper_repeat_noise_readout import repeat_moments, semantic_weights, subject_condition


def test_variance_of_four_repeat_mean_uses_unbiased_repeat_residuals():
    offsets = torch.tensor([-3., -1., 1., 3.]).reshape(1, 4, 1)
    stimulus = torch.tensor([-2., -1., 1., 2.]).reshape(4, 1, 1)
    labels = torch.tensor([0, 0, 1, 1])
    measured = repeat_moments(stimulus + offsets, labels)
    torch.testing.assert_close(measured["noise"], torch.tensor([[5. / 3.]]))
    torch.testing.assert_close(measured["class_mean"], torch.tensor([[-1.5], [1.5]]))


def test_budget_changes_repeat_noise_but_keeps_the_photograph_penalty():
    cov = torch.diag(torch.tensor([.8, .2]))
    photo = torch.diag(torch.tensor([0., .4]))
    teacher = torch.eye(2)
    four = semantic_weights(cov, photo, teacher, 4, .75)
    eighty = semantic_weights(cov, photo, teacher, 80, .75)
    torch.testing.assert_close(four, torch.diag(torch.tensor([1., 1. / 1.3])))
    torch.testing.assert_close(eighty, torch.diag(torch.tensor([1. / .24, 1. / 1.11])))


def test_magnitude_control_uses_plain_direction_and_the_adjusted_norm():
    anchor = torch.tensor([[1., 0.], [0., 1.]])
    plain = torch.tensor([[1., 1.], [1., 1.]])
    corrected = torch.tensor([[3., 0.], [0., 3.]])
    donors = [1, 0]
    variants = subject_condition(anchor, plain, corrected, donors, 10.)
    torch.testing.assert_close(variants["repeat-aware"], (anchor + corrected) / 2)
    torch.testing.assert_close(variants["repeat-wrong"], (anchor + corrected[donors]) / 2)
    torch.testing.assert_close(variants["equal-magnitude"].norm(dim=1), variants["repeat-aware"].norm(dim=1))
    assert torch.all(variants["equal-magnitude"][:, 0] > 0) and torch.all(variants["equal-magnitude"][:, 1] > 0)
