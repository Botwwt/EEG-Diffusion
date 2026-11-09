"""Exact first derivatives for direction AND the existing scalar logit scale.

No new neural head; no current candidate fit is performed by this module.
"""
import numpy as np
from brainhive_EEG_mixture_moment_objective_CPU_20261029_v1 import EEG_cross_entropy_value_and_gradient


def joint_value_and_gradients(condition,teacher_moment,weights,means,concentrations,log_scale):
    scale=float(np.exp(log_scale));assert np.isfinite(scale)
    value,gradient,information=EEG_cross_entropy_value_and_gradient(condition,teacher_moment,weights,means,concentrations,scale)
    direction=condition/np.linalg.norm(condition)
    scale_gradient=scale*float(direction@(information['posterior_mean']-teacher_moment))
    return value,gradient,scale_gradient,information


def Torch_joint_first_order_loss(condition,teacher_moment,log_scale,weights,means,concentrations):
    import torch
    assert log_scale.numel()==1
    c=condition.detach().double().cpu().numpy();q=teacher_moment.detach().double().cpu().numpy();alpha=float(log_scale.detach())
    records=[joint_value_and_gradients(row,moment,weights,means,concentrations,alpha) for row,moment in zip(c,q)]
    value=torch.tensor([record[0] for record in records],device=condition.device,dtype=torch.float64)
    gradient=torch.tensor(np.stack([record[1] for record in records]),device=condition.device,dtype=torch.float64)
    log_gradient=torch.tensor([record[2] for record in records],device=condition.device,dtype=torch.float64)
    surrogate=(condition.double()*gradient).sum(dim=-1)+log_scale.double()*log_gradient
    return value+surrogate-surrogate.detach()
