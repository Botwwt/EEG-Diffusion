"""D18 global state/time transport from native EEG c, no independent source noise."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
import brainhive_gain_calibration_20261027_v1 as U
from brainhive_evidence_guided_prior_20261027_v1 import Marginal as StateFlow

SCHEMA='BrainHIVE_D18_global_native_state_transport_20261027_v1'
ARMS=('paired_transport','sameconcept_wrong_transport')
STEPS=64

def shared_coordinates(c,f):
    assert c.ndim==f.ndim==2 and c.shape[1]==f.shape[1]==1024
    mean=.5*(c.double().mean(0)+f.double().mean(0))
    # One shared scalar RMS, not a separate whitening or query norm adjustment.
    scale=(.5*((c.double()-mean).square().mean()+(f.double()-mean).square().mean())).sqrt()
    assert torch.isfinite(mean).all() and torch.isfinite(scale) and scale>.01
    return mean.float(),scale.float()

def heun_state(model,x,steps=STEPS,reverse=False):
    assert steps in (64,128) and x.ndim==2 and x.shape[1]==1024 and x.dtype==torch.float32
    value=x.clone();dt=(-1. if reverse else 1.)/steps
    for k in range(steps):
        t=1-k/steps if reverse else k/steps
        time=torch.full((len(x),),t,device=x.device,dtype=torch.float32)
        v0=model(value,time).float();trial=value+dt*v0
        v1=model(trial,time+dt).float();value=value+.5*dt*(v0+v1)
        assert torch.isfinite(value).all()
    return value

@torch.no_grad()
def transport(model,raw,mean,scale,steps=STEPS,reverse=False):
    assert raw.dtype==mean.dtype==scale.dtype==torch.float32
    initial=(raw-mean)/scale
    # Field evaluation is FP32. BF16 state quantization would change the ODE.
    with torch.autocast(raw.device.type,enabled=False):final=heun_state(model,initial,steps,reverse)
    return raw+scale*(final-initial)

def concept_endpoint_permutation(array_rows,metadata):
    positions={}
    for i,row in enumerate(array_rows):positions.setdefault(metadata[row]['concept_id'],[]).append(i)
    assert len(positions)==1323 and all(len(x)==10 for x in positions.values())
    permutation=np.empty(len(array_rows),np.int64)
    for concept,ids in sorted(positions.items()):
        offset=1+int(hashlib.sha256((SCHEMA+'|1729|'+concept).encode()).hexdigest()[:8],16)%9
        for j,i in enumerate(ids):permutation[i]=ids[(j+offset)%10]
    assert sorted(permutation.tolist())==list(range(len(array_rows))) and np.all(permutation!=np.arange(len(array_rows)))
    assert all(metadata[array_rows[i]]['concept_id']==metadata[array_rows[j]]['concept_id'] for i,j in enumerate(permutation))
    return permutation

def self_test():
    torch.set_num_threads(2);torch.manual_seed(1729)
    c=torch.randn(6,1024);f=torch.randn(3,1024)+.4;mean,scale=shared_coordinates(c,f)
    class Zero(nn.Module):
        def forward(self,x,t):return torch.zeros_like(x)
    assert torch.equal(transport(Zero(),c,mean,scale),c)
    drift=torch.randn(1,1024)*.02
    class Constant(nn.Module):
        def forward(self,x,t):return drift.expand_as(x)
    out=transport(Constant(),c,mean,scale);back=transport(Constant(),out,mean,scale,reverse=True)
    assert torch.allclose(out,c+scale*drift,atol=2e-5,rtol=2e-5) and torch.allclose(back,c,atol=2e-5,rtol=2e-5)
    class Linear(nn.Module):
        def forward(self,x,t):return .2*x
    x=torch.randn(6,1024);out=heun_state(Linear(),x);back=heun_state(Linear(),out,reverse=True)
    assert torch.allclose(out,x*math.exp(.2),atol=3e-6,rtol=3e-6) and torch.allclose(back,x,atol=3e-6,rtol=3e-6)
    model=StateFlow();assert sum(p.numel() for p in model.parameters())==4483072
    assert list(__import__('inspect').signature(model.forward).parameters)==['x','t']
    assert torch.count_nonzero(model(c,torch.rand(6)))==0
    x=torch.randn(20,1024);y=torch.randn_like(x);time=torch.rand(20);leaf=(1-time[:,None])*x+time[:,None]*y
    loss=(model(leaf,time)-(y-x)).square().mean();loss.backward()
    assert model.out.weight.grad.norm()>0 and torch.isfinite(model.out.weight.grad).all()
    # A shared change of raw coordinates changes both sources identically.
    mu2,s2=shared_coordinates(c+3.,f+3.);assert torch.allclose(mu2,mean+3.) and torch.allclose(s2,scale)
    return dict(status='D18_SHARED_COORDINATE_ZERO_IDENTITY_CONSTANT_LINEAR_ODE_AND_NATIVE_FIELD_CPU_CHECKS_PASSED',checks=6,
        parameters_per_field=4483072,field_inputs=['state','time'],CUDA_initialized=torch.cuda.is_initialized(),real_training=False)
