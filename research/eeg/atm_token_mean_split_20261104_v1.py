"""Existing four-token mean/contrast substitution; no fitted parameters."""
import numpy as np
MODES=('native_old','native_new','old_mean_new_center','new_mean_old_center','token_ensemble')

def numpy_tokens(old,new,mode):
    assert mode in MODES and old.shape==new.shape and old.shape[-2:]==(4,2048)
    if mode=='native_old':return old
    if mode=='native_new' or np.array_equal(old,new):return new
    a=old.astype(np.float64);b=new.astype(np.float64)
    if mode=='old_mean_new_center':return b-b.mean(-2,keepdims=True)+a.mean(-2,keepdims=True)
    if mode=='new_mean_old_center':return a-a.mean(-2,keepdims=True)+b.mean(-2,keepdims=True)
    return .5*(a+b)

def torch_tokens(old,new,mode):
    import torch
    assert mode in MODES and old.shape==new.shape and tuple(old.shape[-2:])==(4,2048)
    if mode=='native_old':return old
    if mode=='native_new' or torch.equal(old,new):return new
    a=old.float();b=new.float()
    if mode=='old_mean_new_center':v=b-b.mean(-2,keepdim=True)+a.mean(-2,keepdim=True)
    elif mode=='new_mean_old_center':v=a-a.mean(-2,keepdim=True)+b.mean(-2,keepdim=True)
    else:v=.5*(a+b)
    return v.to(new.dtype)

def attention64(q,k,v):
    logits=q@np.swapaxes(k,-1,-2)/np.sqrt(q.shape[-1]);weights=np.exp(logits-logits.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
    return weights@v,weights
