"""Native-coordinate EEG measurement precision and encoder functional anchoring.

These Gaussian/quadratic constructions are classical. This module neither
loads a host nor claims posterior calibration or image improvement.
"""
import math
import torch
from torch.nn import functional as F

ARMS=('unanchored','isotropic','measurement','permuted_measurement')
TAU=.25
SHRINKAGE=.1


def unit_and_jvp(eeg,perturbations,state):
    """Exact native BrainMLP pre-affine LN coordinates and input JVPs."""
    assert eeg.ndim==2 and eeg.shape[1]==4250 and perturbations.shape==(len(eeg),4,4250)
    W=state['proj_in.weight'];bias=state['proj_in.bias']
    W1=state['block.layers.0.ff.w1.weight'];W2=state['block.layers.0.ff.w2.weight']
    hidden=F.linear(eeg,W,bias);dh=F.linear(perturbations,W)
    pre=F.linear(hidden,W1)
    slope=.5*(1+torch.erf(pre/math.sqrt(2)))+pre*torch.exp(-.5*pre.square())/math.sqrt(2*math.pi)
    z=hidden+F.linear(F.gelu(pre),W2)
    dz=dh+F.linear(F.linear(dh,W1)*slope[:,None],W2)
    centered=z-z.mean(-1,keepdim=True);inverse=torch.rsqrt(centered.square().mean(-1,keepdim=True)+1e-6)
    unit=centered*inverse
    du=(dz-dz.mean(-1,keepdim=True)-unit[:,None]*(dz*unit[:,None]).mean(-1,keepdim=True))*inverse[:,None]
    return unit,du


def normalized_precision(covariance,shrinkage=SHRINKAGE):
    """Equal isotropic expected penalty: tr(Q)/D=1 for all anchored controls."""
    N=covariance.double();N=(N+N.T)/2;D=len(N)
    assert D==N.shape[1] and shrinkage==.1 and torch.isfinite(N).all() and N.trace()>0
    C=(1-shrinkage)*N/(N.trace()/D)+shrinkage*torch.eye(D,dtype=N.dtype,device=N.device)
    Q=torch.linalg.inv(C);Q=(Q+Q.T)/2;Q=Q/(Q.trace()/D)
    assert torch.linalg.eigvalsh(Q).min()>0 and torch.isfinite(Q).all()
    return Q


def anchor_penalty(unit,original,precision,arm,tau=TAU):
    assert arm in ARMS and tau==.25 and unit.shape==original.shape
    delta=unit.float()-original.float();D=unit.shape[-1]
    if arm=='unanchored':return delta.square().mean()*0
    if arm=='isotropic':return .5*delta.square().mean()/(tau*tau)
    return .5*((delta@precision.float())*delta).sum(-1).mean()/(D*tau*tau)


def local_quadratic_map(J,residual,precision,tau=TAU):
    """Exact local residual-linearized objective; not a global neural MAP solver.

    min_d ||r+Jd||^2 + d^T Q d/(2 D tau^2).
    """
    assert J.ndim==2 and J.shape[1]==len(precision) and tau==.25
    return torch.linalg.solve(J.T@J+precision/(2*len(precision)*tau*tau),-J.T@residual)
