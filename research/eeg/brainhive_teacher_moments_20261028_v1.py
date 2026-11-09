"""Frozen symmetric InfoNCE gradients, including non-unit empirical mean targets."""
import torch


def symmetric_readout(condition, targets, temperature):
    """Targets are already normalized draws OR their unnormalized mean."""
    assert condition.ndim == 2 and targets.shape == condition.shape
    assert condition.dtype == targets.dtype == torch.float64
    radius=condition.norm(dim=1,keepdim=True)
    unit=condition/radius
    logits=float(temperature)*unit@targets.T
    row=logits.softmax(1);col=logits.softmax(0)
    loss=.5*(-2*logits.diag()+logits.logsumexp(1)+logits.logsumexp(0))
    direction=.5*(row+col)@targets-targets
    gradient=float(temperature)/radius*(direction-unit*(unit*direction).sum(1,keepdim=True))
    return loss,gradient


def finite_moment_summary(condition,deterministic,draws,temperature):
    """Small-batch mathematical check; production accumulates draws sequentially."""
    assert draws.ndim==3 and len(draws)>=2 and len(draws)%2==0
    values=[symmetric_readout(condition,x,temperature) for x in draws]
    losses=torch.stack([x[0] for x in values]);gradients=torch.stack([x[1] for x in values])
    at_mean=symmetric_readout(condition,draws.mean(0),temperature)
    direct=symmetric_readout(condition,deterministic,temperature)
    return dict(jensen_gap=losses.mean(0)-at_mean[0],mean_gradient=gradients.mean(0),
                mean_target_gradient=at_mean[1],deterministic_gradient=direct[1])
