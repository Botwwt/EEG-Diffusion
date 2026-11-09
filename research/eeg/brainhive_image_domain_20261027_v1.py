"""D19 real/native-image latent domain classification and bounded endpoint edits.

This is image-domain postprocessing, not a diffusion-score or posterior sampler.
"""
import hashlib,math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA='BrainHIVE_D19_actual_image_domain_endpoint_correction_20261027_v1'
FIELDS=('conditional','unconditional','wrong_condition')
ARMS=('vae_roundtrip',)+FIELDS
SIGMAS=(0.,.25,.5,1.)
RADIUS=.1
EDIT_STEPS=8

class DomainHead(nn.Module):
    def __init__(self):
        super().__init__();layers=[];channels=4
        for width in (32,64,128,128):
            layers.extend([nn.Conv2d(channels,width,3,stride=2,padding=1),nn.GroupNorm(8,width),nn.SiLU()]);channels=width
        self.image=nn.Sequential(*layers,nn.Flatten(),nn.Linear(2048,128),nn.SiLU())
        self.condition=nn.Sequential(nn.Linear(1024,128),nn.SiLU())
        self.hidden=nn.Sequential(nn.Linear(400,512),nn.SiLU())
        self.out=nn.Linear(512,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,sigma,c):
        assert x.shape[1:]==(4,64,64) and c.shape==(len(x),1024) and sigma.shape==(len(x),)
        image=self.image(x);condition=self.condition(c)
        freq=torch.arange(1,9,device=x.device,dtype=x.dtype);phase=sigma[:,None]*freq*math.pi
        time=torch.cat([phase.sin(),phase.cos()],-1)
        return self.out(self.hidden(torch.cat([image,condition,image*condition,time],-1))).squeeze(-1)

def wrong_positions(queries):
    groups={}
    for i,q in enumerate(queries):groups.setdefault(q['concept_id'],[]).append(i)
    assert all(len(ids)==10 for ids in groups.values());result=np.empty(len(queries),np.int64)
    for concept,ids in sorted(groups.items()):
        shift=1+int(hashlib.sha256((SCHEMA+'|1729|'+concept).encode()).hexdigest()[:8],16)%9
        for j,i in enumerate(ids):result[i]=ids[(j+shift)%10]
    assert sorted(result.tolist())==list(range(len(queries))) and np.all(result!=np.arange(len(queries)))
    assert all(queries[i]['concept_id']==queries[j]['concept_id'] for i,j in enumerate(result))
    return result

def corrupt(real,generated,noise):
    assert real.shape==generated.shape==noise.shape and real.ndim==4
    sigma=torch.tensor(SIGMAS,device=real.device,dtype=real.dtype)[None,:,None,None,None]
    alpha=(1-sigma.square()).clamp_min(0).sqrt()
    positive=alpha*real[:,None]+sigma*noise[:,None];negative=alpha*generated[:,None]+sigma*noise[:,None]
    return torch.stack([positive,negative],2) # query,sigma,class,channel,H,W; same epsilon perpair

def proper(logits):
    assert logits.ndim==3 and logits.shape[1:]==(4,2)
    labels=torch.tensor([1.,0.],device=logits.device).expand_as(logits)
    p=logits.sigmoid();return dict(log_loss=F.binary_cross_entropy_with_logits(logits,labels,reduction='none').mean(-1),
        Brier=(p-labels).square().mean(-1),accuracy=((p>=.5)==labels.bool()).float().mean(-1))

def edit(model,z,c,radius=RADIUS,steps=EDIT_STEPS):
    assert radius in (0.,RADIUS) and steps==EDIT_STEPS and z.dtype==c.dtype==torch.float32
    original=z.detach();value=original.clone();r=radius*original.square().mean((1,2,3),keepdim=True).sqrt()
    if radius==0:return value,dict(max_relative_change=0.,steps=0)
    for _ in range(steps):
        value=value.detach().requires_grad_(True)
        with torch.autocast(value.device.type,enabled=False):score=model(value,torch.zeros(len(value),device=value.device),c).sum()
        gradient=torch.autograd.grad(score,value)[0];assert torch.isfinite(gradient).all()
        rms=gradient.square().mean((1,2,3),keepdim=True).sqrt()
        delta=(value.detach()-original)+(r/steps)*gradient.detach()/rms.clamp_min(1e-12)
        norm=delta.square().mean((1,2,3),keepdim=True).sqrt();delta=delta*torch.minimum(torch.ones_like(norm),r/norm.clamp_min(1e-12))
        value=original+delta
    relative=(value-original).square().mean((1,2,3)).sqrt()/original.square().mean((1,2,3)).sqrt().clamp_min(1e-12)
    assert torch.isfinite(value).all() and relative.max()<=radius+1e-6
    return value.detach(),dict(max_relative_change=float(relative.max()),mean_relative_change=float(relative.mean()),steps=steps)

def self_test():
    torch.set_num_threads(2);torch.manual_seed(1729);z=torch.randn(3,4,64,64);c=torch.randn(3,1024)
    class Linear(nn.Module):
        def forward(self,x,t,c):return (x.flatten(1)*torch.linspace(-1.,1.,16384)).mean(-1)
    zero,proof=edit(Linear(),z,c,radius=0);assert torch.equal(zero,z)
    out,proof=edit(Linear(),z,c);assert proof['max_relative_change']<=.100001 and (Linear()(out,None,c)>Linear()(z,None,c)).all()
    class Flat(nn.Module):
        def forward(self,x,t,c):return x.flatten(1).sum(-1)*0
    unchanged,_=edit(Flat(),z,c);assert torch.equal(unchanged,z)
    noisy=corrupt(z,z+1,torch.randn_like(z));assert noisy.shape==(3,4,2,4,64,64) and torch.equal(noisy[:,3,0],noisy[:,3,1])
    a=proper(torch.zeros(3,4,2));assert torch.allclose(a['log_loss'],torch.full((3,4),math.log(2))) and torch.all(a['Brier']==.25)
    model=DomainHead().eval().requires_grad_(False)
    with torch.no_grad():model.out.weight.fill_(.001)
    leaf=z.clone().requires_grad_(True);gradient=torch.autograd.grad(model(leaf,torch.zeros(3),c).sum(),leaf)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0 and all(p.grad is None for p in model.parameters())
    qs=[dict(concept_id=str(i//10),array_row=i) for i in range(40)];perm=wrong_positions(qs);assert len(set(perm))==40
    assert torch.equal(model(z,torch.zeros(3),torch.zeros_like(c)),model(z,torch.zeros(3),torch.zeros_like(c+2)))
    return dict(status='D19_EIGHT_CPU_RADIUS_SIGN_IDENTITY_PAIRED_NOISE_PROPER_LOSS_AND_INPUT_GRADIENT_CHECKS_PASSED',checks=8,
        parameters_per_field=sum(p.numel() for p in model.parameters()),CUDA_initialized=torch.cuda.is_initialized(),real_fit=False)

if __name__=='__main__':print(__import__('json').dumps(self_test()))
