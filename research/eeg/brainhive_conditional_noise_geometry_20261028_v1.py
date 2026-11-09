"""D25 sampling covariance operator, independent of the ordinary condition module.

For each four-channel latent location, a nonzero native conditional-response
direction v defines a three-dimensional null space. Among Gaussian laws on
that subspace with trace covariance four, maximum entropy gives (4/3)(I-vv^T).
This is a declared sampling hypothesis, not an EEG likelihood or exact posterior.
"""
import math
import torch

MODES=('isotropic','response_null','orthogonal_control')

def orthogonal_channel_direction(x):
    assert x.ndim==4 and x.shape[1]==4
    return torch.stack((-x[:,1],x[:,0],-x[:,3],x[:,2]),dim=1)

def unit_direction(field,mode):
    assert mode in MODES and field.ndim==4 and field.shape[1]==4
    d=field.float()
    if mode=='orthogonal_control':d=orthogonal_channel_direction(d)
    norm=torch.linalg.vector_norm(d,dim=1,keepdim=True)
    valid=norm>0
    unit=d/norm.clamp_min(torch.finfo(d.dtype).tiny)
    unit=torch.where(valid,unit,torch.zeros_like(unit))
    return unit,valid

def transform_ancestral_noise(noise,field,mode):
    assert noise.shape==field.shape and noise.ndim==4 and noise.shape[1]==4
    assert mode in MODES and torch.isfinite(noise).all() and torch.isfinite(field).all()
    if mode=='isotropic':return noise
    v,valid=unit_direction(field,mode)
    if not valid.any():return noise
    z=noise.float();projected=math.sqrt(4/3)*(z-v*(z*v).sum(dim=1,keepdim=True))
    result=torch.where(valid,projected,z).to(noise.dtype)
    assert torch.isfinite(result).all()
    return result
