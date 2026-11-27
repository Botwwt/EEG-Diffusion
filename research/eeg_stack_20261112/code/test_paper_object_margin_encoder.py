import torch

from paper_object_margin_encoder import confuser_log_prior, object_loss


def test_hard_negatives_exclude_the_owned_object_and_preserve_negative_mass():
    prototypes = torch.tensor([[1., 0.], [.8, .6], [0., 1.], [-1., 0.]])
    photos = prototypes[[0, 2]]
    labels = torch.tensor([0, 2])
    prior = confuser_log_prior(photos, prototypes, labels, count=1, multiplier=4.)

    assert torch.equal(prior.exp().argmax(1), torch.tensor([1, 1]))
    assert torch.allclose(prior.exp().sum(1), torch.full((2,), 3.))
    assert not prior.exp()[torch.arange(2), labels].any()


def test_confuser_weight_changes_the_gradient_toward_the_same_photo():
    prototypes = torch.tensor([[1., 0.], [.8, .6], [0., 1.], [-1., 0.]])
    photo = prototypes[[0]]
    labels = torch.tensor([0])
    vector = torch.tensor([[.4, .9]], requires_grad=True)
    uniform = confuser_log_prior(photo, prototypes, labels, count=1, multiplier=1.)
    hard = confuser_log_prior(photo, prototypes, labels, count=1, multiplier=4.)

    a = object_loss(vector, photo, prototypes, labels, uniform)
    b = object_loss(vector, photo, prototypes, labels, hard)
    grad_a = torch.autograd.grad(a, vector, retain_graph=True)[0]
    grad_b = torch.autograd.grad(b, vector)[0]
    assert torch.isfinite(torch.stack((a, b))).all()
    assert not torch.allclose(grad_a, grad_b)
