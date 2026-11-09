"""Unadmitted matrix-free margin geometry; classical VJP/JVP, not novelty.

This module is NOT imported by any sealed training run or GPU queue. It computes
local derivatives at an immutable parameter/input snapshot, not nonlinear finite
feasibility. No full parameter Jacobian or Gram matrix is stored by the operator.
"""
import torch
from torch.nn import functional as F


def _snapshot(x):
    if isinstance(x,torch.Tensor):return x.detach().clone()
    if isinstance(x,tuple):return tuple(_snapshot(v) for v in x)
    if isinstance(x,list):return [_snapshot(v) for v in x]
    if isinstance(x,dict):return {k:_snapshot(v) for k,v in x.items()}
    return x


def functional_encoder_snapshot(model,names,args,kwargs=None):
    """Explicit parameter subset; native forward, frozen head and buffers retained."""
    if any(m.training for m in model.modules()):
        raise ValueError('all modules must be eval; stochastic/stateful training unsupported')
    parameters=dict(model.named_parameters())
    if not names or len(set(names))!=len(names) or any(n not in parameters for n in names):
        raise ValueError('invalid active parameter names')
    theta=tuple(parameters[n].detach().clone() for n in names)
    selected=set(names)
    fixed={n:p.detach().clone() for n,p in parameters.items() if n not in selected}
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    args,kwargs=_snapshot(args),_snapshot(kwargs or {})
    def encoder(values):
        state={**fixed,**dict(zip(names,values))}
        return torch.func.functional_call(model,(state,buffers),args,kwargs,strict=True)
    return encoder,theta


class MatrixFreeMarginMap:
    def __init__(self,encoder,theta,gallery,own_indices,gallery_groups,row_scale=None):
        self.encoder=encoder
        self.theta=tuple(t.detach().clone() for t in theta)
        raw,self.pullback=torch.func.vjp(encoder,self.theta)
        if raw.ndim!=2 or gallery.ndim!=2 or raw.shape[1]!=gallery.shape[1]:
            raise ValueError('wrong semantic interface')
        batch=raw.shape[0]
        if own_indices.shape!=(batch,) or gallery_groups.shape!=(len(gallery),):
            raise ValueError('wrong gallery identity/group shape')
        with torch.no_grad():
            norm=raw.norm(dim=-1,keepdim=True)
            if not bool(torch.isfinite(raw).all() and (norm>1e-12).all()):
                raise ValueError('nonfinite/degenerate normalization')
            yy=F.normalize(raw,dim=-1);score=yy@gallery.T
            ii=torch.arange(batch,device=raw.device);own=score[ii,own_indices].clone()
            score[ii,own_indices]=-torch.inf
            same=gallery_groups[own_indices,None]==gallery_groups[None,:]
            if not bool((same.sum(-1)>=2).all()):raise ValueError('no within-concept negative')
            wi=score.masked_fill(~same,-torch.inf).argmax(-1);gi=score.argmax(-1)
            self.negatives=torch.cat((wi,gi));self.rows=torch.arange(2*batch,device=raw.device)%batch
            diff=gallery[own_indices[self.rows]]-gallery[self.negatives]
            yr=yy[self.rows]
            self.cotangents=((diff-yr*(yr*diff).sum(-1,keepdim=True))/norm[self.rows]).detach()
            self.margins=torch.cat((own-score[ii,wi],own-score[ii,gi])).detach()
            self.row_scale=(torch.ones_like(self.margins) if row_scale is None else row_scale.detach().clone())
            if self.row_scale.shape!=self.margins.shape or not bool((self.row_scale>0).all() and torch.isfinite(self.row_scale).all()):
                raise ValueError('row scaling must be finite and positive')
            self.raw_template=raw.detach()
            self.batch=batch

    def transpose(self,weights):
        """J^T w as a parameter tuple; handles both margins per EEG query."""
        if weights.shape!=self.margins.shape:raise ValueError('wrong dual vector shape')
        with torch.no_grad():
            coefficients=(weights/self.row_scale).to(self.cotangents.dtype)
            seed=torch.zeros_like(self.raw_template)
            seed.index_add_(0,self.rows,coefficients[:,None]*self.cotangents)
        gradients,=self.pullback(seed)
        return tuple(g.detach() for g in gradients)

    def forward(self,direction):
        """J v via native forward-mode AD. Unsupported operators raise, no fallback."""
        if len(direction)!=len(self.theta) or any(v.shape!=t.shape for v,t in zip(direction,self.theta)):
            raise ValueError('wrong parameter tangent')
        _,raw_tangent=torch.func.jvp(self.encoder,(self.theta,),(tuple(direction),))
        return ((self.cotangents*raw_tangent[self.rows]).sum(-1)/self.row_scale).detach()

    def gram(self,weights):
        """(J J^T) w, without either J or J J^T materialization."""
        return self.forward(self.transpose(weights))

    def closest_step(self,proposal,dual):
        """Recover p+J^T lambda from an externally solved local dual problem."""
        correction=self.transpose(dual)
        return tuple(p+v for p,v in zip(proposal,correction))
