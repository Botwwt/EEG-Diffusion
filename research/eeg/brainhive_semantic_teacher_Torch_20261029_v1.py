"""Normalized training-only Power-Spherical teacher and frozen mixture prior.

Import only after the pinned runtime is configured. This is an auxiliary latent
model. F-native coordinates use the actual output affine and the zero-mean
subspace, so an intrinsic 1023-vector restores a 1024-vector at a fixed T radius.
"""
import math
import torch
import torch.nn.functional as F


def power_parameters(raw_direction,raw_concentration):
    if bool((torch.linalg.vector_norm(raw_direction.detach(),dim=-1)<=1e-12).any()):
        raise ValueError('Power-Spherical direction is zero')
    mean=F.normalize(raw_direction.double(),dim=-1)
    concentration=F.softplus(raw_concentration.double())
    dimension=mean.shape[-1]
    beta=torch.full_like(concentration,(dimension-1)/2)
    alpha=beta+concentration
    return mean,concentration,alpha,beta


def power_rsample(raw_direction,raw_concentration,sample_shape=torch.Size()):
    mean,concentration,alpha,beta=power_parameters(raw_direction,raw_concentration)
    axial=2*torch.distributions.Beta(alpha,beta).rsample(sample_shape)-1
    tangent=torch.randn(axial.shape+(mean.shape[-1]-1,),device=mean.device,dtype=mean.dtype)
    tangent=F.normalize(tangent,dim=-1)
    value=torch.cat((axial[...,None],torch.sqrt((1-axial.square()).clamp_min(0))[...,None]*tangent),dim=-1)
    sign=torch.where(mean[...,0]>=0,torch.ones_like(mean[...,0]),-torch.ones_like(mean[...,0]))
    first=torch.zeros_like(mean);first[...,0]=sign
    householder=F.normalize(mean+first,dim=-1)
    signed_first=-sign*value[...,0]
    signed=torch.cat((signed_first[...,None],value[...,1:]),dim=-1)
    return signed-2*(signed*householder).sum(dim=-1,keepdim=True)*householder


def power_entropy(raw_direction,raw_concentration):
    mean,kappa,alpha,beta=power_parameters(raw_direction,raw_concentration)
    log_z=(alpha+beta)*math.log(2)+beta*math.log(math.pi)+torch.lgamma(alpha)-torch.lgamma(alpha+beta)
    expected_log=math.log(2)+torch.digamma(alpha)-torch.digamma(alpha+beta)
    return log_z-kappa*expected_log


def power_mean(raw_direction,raw_concentration):
    mean,kappa,alpha,beta=power_parameters(raw_direction,raw_concentration)
    return (kappa/(kappa+mean.shape[-1]-1))[...,None]*mean


def mixture_prior_log_density(value,weights,means,concentrations,component_log_normalizers,evidence,log_evidence):
    """p0(v) exp(b.v)/Z; b can be one EEG or a fixed KL barycenter evidence."""
    for constant in (component_log_normalizers,log_evidence):
        if torch.is_tensor(constant) and constant.dtype!=torch.float64:
            raise ValueError('Probability log-normalizers must retain float64 precision')
    component_log_normalizers=torch.as_tensor(component_log_normalizers,device=value.device,dtype=torch.float64)
    log_evidence=torch.as_tensor(log_evidence,device=value.device,dtype=torch.float64)
    logits=torch.log(weights.double())+component_log_normalizers+(value.double()@means.double().T)*concentrations.double()
    return torch.logsumexp(logits,dim=-1)+(value.double()*evidence.double()).sum(dim=-1)-log_evidence


def _centering_householder(dimension,device):
    vector=torch.ones(dimension,dtype=torch.float64,device=device)/math.sqrt(dimension)
    vector[-1]-=1
    return F.normalize(vector,dim=-1)


def F_native_coordinates(condition,F_weight,F_bias):
    """Inverse F affine then orthogonal coordinates; no EEG affine substitution."""
    raw=(condition.double()-F_bias.double())/F_weight.double()
    centered=raw-raw.mean(dim=-1,keepdim=True)
    vector=_centering_householder(raw.shape[-1],raw.device)
    coordinates=(centered-2*(centered*vector).sum(dim=-1,keepdim=True)*vector)[...,:-1]
    return coordinates


def native_F_condition(value,F_weight,F_bias,reference_pre_affine_radius):
    """Intrinsic sphere -> actual native F output affine at a T-only radius."""
    if value.shape[-1]+1!=F_weight.numel():
        raise ValueError('Intrinsic F sphere dimension is wrong')
    norm=torch.linalg.vector_norm(value.double(),dim=-1,keepdim=True)
    if bool((norm<=1e-12).any()):
        raise ValueError('Native F direction is zero')
    unit=value.double()/norm
    augmented=torch.cat((unit,torch.zeros_like(unit[...,:1])),dim=-1)
    vector=_centering_householder(F_weight.numel(),value.device)
    pre=augmented-2*(augmented*vector).sum(dim=-1,keepdim=True)*vector
    radius=torch.as_tensor(reference_pre_affine_radius,device=value.device,dtype=torch.float64)
    if radius.ndim:
        radius=radius[...,None]
    return F_bias.double()+F_weight.double()*radius*pre


def semantic_teacher_loss(feature,target,semantic_concentration,semantic_log_normalizer,teacher_entropy,prior_log_density):
    feature=F.normalize(feature.double(),dim=-1)
    target=F.normalize(target.detach().double(),dim=-1)
    nll=-semantic_log_normalizer-semantic_concentration*(feature*target).sum(dim=-1)
    sampled_KL=-teacher_entropy-prior_log_density
    return nll+sampled_KL,dict(normalized_semantic_NLL=nll,sampled_KL_to_working_prior=sampled_KL)


def exact_moment_M_step(condition,teacher_moment,weights_numpy,means_numpy,concentrations_numpy,scale):
    """Exact first derivative for fixed q; second derivatives are not supplied.

    The numerical moment computation is intentionally detached. This injects
    the proven first derivative, suitable for ordinary first-order optimizers.
    Values omit q-dependent but theta-independent terms and are not full KL.
    """
    from brainhive_EEG_mixture_moment_objective_CPU_20261029_v1 import EEG_cross_entropy_value_and_gradient
    import numpy as np
    c=condition.detach().double().cpu().numpy()
    teacher=teacher_moment.detach().double().cpu().numpy()
    records=[EEG_cross_entropy_value_and_gradient(row,moment,weights_numpy,means_numpy,concentrations_numpy,float(scale)) for row,moment in zip(c,teacher)]
    values=torch.tensor([record[0] for record in records],device=condition.device,dtype=torch.float64)
    gradients=torch.tensor(np.stack([record[1] for record in records]),device=condition.device,dtype=torch.float64)
    surrogate=(condition.double()*gradients).sum(dim=-1)
    return values+surrogate-surrogate.detach()
