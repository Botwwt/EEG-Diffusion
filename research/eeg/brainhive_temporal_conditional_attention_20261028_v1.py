"""D37 candidate: native-coordinate temporal tangents and constrained IP routing.

This module does not load a host, train a model, or establish image efficacy.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

TIME_EDGES=(0,50,100,150,200,250)


def temporal_tangents(eeg,state):
    """Five native first-layer contributions, pushed through the true EEG MLP JVP."""
    assert eeg.ndim==2 and eeg.shape[1]==4250
    w=state['proj_in.weight'];b=state['proj_in.bias']
    w1=state['block.layers.0.ff.w1.weight'];w2=state['block.layers.0.ff.w2.weight']
    gamma=state['block.layers.0.norm.weight'];beta=state['block.layers.0.norm.bias']
    x=eeg.reshape(-1,17,250);space_time=w.reshape(1024,17,250)
    contributions=torch.stack([F.linear(x[:,:,lo:hi].reshape(len(x),-1),space_time[:,:,lo:hi].reshape(1024,-1))
                               for lo,hi in zip(TIME_EDGES[:-1],TIME_EDGES[1:])],dim=1)
    h=F.linear(eeg,w,b)
    residual=contributions-contributions.mean(dim=1,keepdim=True)
    pre=F.linear(h,w1)
    activation=F.gelu(pre)
    derivative=.5*(1+torch.erf(pre/math.sqrt(2)))+pre*torch.exp(-.5*pre.square())/math.sqrt(2*math.pi)
    z=h+F.linear(activation,w2)
    dz=residual+F.linear(F.linear(residual,w1)*derivative[:,None,:],w2)
    centered=z-z.mean(dim=-1,keepdim=True)
    inv_std=torch.rsqrt(centered.square().mean(dim=-1,keepdim=True)+1e-6)
    unit=centered*inv_std
    dunit=(dz-dz.mean(dim=-1,keepdim=True)-unit[:,None,:]*(dz*unit[:,None,:]).mean(dim=-1,keepdim=True))*inv_std[:,None,:]
    tangents=dunit*gamma
    condition=unit*gamma+beta
    return condition,tangents,dict(hidden=h,contributions=contributions,hidden_residuals=residual)


class TemporalReadout(nn.Module):
    """Unused synthetic draft fixture; the primary candidate uses frozen JVPs."""
    def __init__(self,width=128):
        super().__init__()
        assert width==128
        self.down=nn.Linear(1024,width)
        self.time_embedding=nn.Parameter(torch.randn(5,width)*.02)
        self.key=nn.Linear(width,2048,bias=False)
        self.value=nn.Linear(width,2048,bias=False)
        nn.init.zeros_(self.value.weight)

    def forward(self,temporal,base_tokens):
        assert temporal.ndim==3 and temporal.shape[1:]==(5,1024)
        assert base_tokens.shape==(len(temporal),4,2048)
        x=F.layer_norm(temporal,(1024,),eps=1e-6)
        h=F.gelu(self.down(x)+self.time_embedding)
        keys=self.key(h);values=self.value(h)
        keys=keys-keys.mean(dim=1,keepdim=True)
        values=values-values.mean(dim=1,keepdim=True)
        anchor=base_tokens/base_tokens.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(1e-6)
        return anchor[:,:,None,:]*keys[:,None,:,:],anchor[:,:,None,:]*values[:,None,:,:]


def frozen_prior_temporal_tokens(condition,temporal,prior,radius=.25):
    """Push true EEG time tangents through frozen ordinary/A/P, adding no weights.

    `prior` maps native c to the existing four IP tokens. Radius is a fixed
    engineering trust budget, not an empirically established quality bound.
    """
    assert condition.shape==(len(temporal),1024) and temporal.shape[1:]==(5,1024)
    assert radius==.25
    pieces=[];base=None
    for t in range(5):
        value,direction=torch.autograd.functional.jvp(prior,condition,temporal[:,t],create_graph=False,strict=False)
        assert value.shape==direction.shape==(len(condition),4,2048)
        if base is None:base=value
        pieces.append(direction)
    details=torch.stack(pieces,dim=2)
    details=details-details.mean(dim=2,keepdim=True)
    norm=details.square().sum(-1).mean(-1,keepdim=True).sqrt().unsqueeze(-1)
    budget=radius*base.norm(dim=-1,keepdim=True).unsqueeze(-1)
    scale=torch.minimum(torch.ones_like(norm),budget/norm.clamp_min(1e-12))
    return base,details*scale,dict(unbounded=details,scale=scale,budget=budget)


def attention_delta(query,base_key,detail_key,detail_value,mode='conditional',query_chunk=256,return_probabilities=False):
    """Residual only. Both conditional and flat_residual return zero at zero values.

    query [B,H,N,D]; base_key [B,H,4,D]; details [B,H,4,5,D].
    """
    assert mode in ['conditional','flat_residual'] and query_chunk>0
    B,H,N,D=query.shape
    assert base_key.shape==(B,H,4,D) and detail_key.shape==detail_value.shape==(B,H,4,5,D)
    dtype=torch.float64 if query.dtype==torch.float64 else torch.float32
    k=base_key.to(dtype);dk=detail_key.to(dtype);dv=detail_value.to(dtype)
    pieces=[];marginals=[];conditionals=[];joint_probabilities=[]
    for lo in range(0,N,query_chunk):
        q=query[:,:,lo:lo+query_chunk].to(dtype)
        root_logits=torch.einsum('bhnd,bhkd->bhnk',q,k)/math.sqrt(D)
        temporal_logits=torch.einsum('bhnd,bhktd->bhnkt',q,dk)/math.sqrt(D)
        conditional=temporal_logits.softmax(dim=-1)
        marginal=root_logits.softmax(dim=-1) if mode=='conditional' else (root_logits+temporal_logits.logsumexp(dim=-1)).softmax(dim=-1)
        joint=marginal[...,None]*conditional
        pieces.append(torch.einsum('bhnkt,bhktd->bhnd',joint,dv))
        if return_probabilities:
            marginals.append(marginal);conditionals.append(conditional);joint_probabilities.append(joint)
    delta=torch.cat(pieces,dim=2).to(query.dtype)
    if not return_probabilities:return delta
    return delta,dict(marginal=torch.cat(marginals,dim=2),conditional=torch.cat(conditionals,dim=2),joint=torch.cat(joint_probabilities,dim=2))
