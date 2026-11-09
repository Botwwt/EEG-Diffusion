"""D24: geometry-preserving coordinate alignment folded into existing weights.

This is functionally an orthogonal output map, not added representational power.
Orthogonal Procrustes and coupled weight rotations are established techniques.
Only native one-block BrainMLP with bias-free internal linear layers is supported.
"""
import torch

ARMS=('paired_coordinate_gauge','same_concept_wrong_gauge','visual_affine_only')
PREFIX='block.layers.0.'

def zero_mean_basis(d,device=None):
    assert d>1
    one=torch.ones(d,device=device,dtype=torch.float64)/(d**.5)
    v=-one;v[-1]+=1;v=v/v.norm()
    h=torch.eye(d,device=device,dtype=torch.float64)-2*v[:,None]*v[None,:]
    return h[:,:-1]

def procrustes_rotation(x,y):
    """Column rotation R: min ||x R^T-y||F, R^TR=I and R1=1."""
    assert x.shape==y.shape and x.ndim==2 and torch.isfinite(x).all() and torch.isfinite(y).all()
    d=x.shape[1];b=zero_mean_basis(d,x.device);xx=x.double()@b;yy=y.double()@b
    cross=yy.T@xx;u,s,vh=torch.linalg.svd(cross,full_matrices=False)
    r=b@(u@vh)@b.T+torch.ones((d,d),device=x.device,dtype=torch.float64)/d
    return r,dict(singular_min=float(s.min()),singular_max=float(s.max()),numerical_rank=int((s>s.max()*1e-10).sum()),
        objective_before=float((x.double()-y.double()).square().mean()),objective_after=float((x.double()@r.T-y.double()).square().mean()))

def remove_affine(value,weight,bias):
    assert (weight.abs()>1e-8).all()
    return (value.double()-bias.double())/weight.double()

def fold_native(state,rotation,visual_weight,visual_bias):
    """a'=Ra, W1'=W1 R^T, W2'=R W2, LN affine'=visual affine."""
    expected={'proj_in.weight','proj_in.bias',PREFIX+'ff.w1.weight',PREFIX+'ff.w2.weight',PREFIX+'norm.weight',PREFIX+'norm.bias'}
    assert set(state)==expected
    r=rotation.double();d=len(r);eye=torch.eye(d,device=r.device,dtype=r.dtype)
    assert (r@r.T-eye).abs().max()<1e-8 and (r.sum(1)-1).abs().max()<1e-8
    assert visual_weight.shape==visual_bias.shape==(d,)
    out={k:v.detach().clone() for k,v in state.items()}
    def put(k,v):out[k]=v.to(device=state[k].device,dtype=state[k].dtype)
    put('proj_in.weight',r@state['proj_in.weight'].double())
    put('proj_in.bias',r@state['proj_in.bias'].double())
    put(PREFIX+'ff.w1.weight',state[PREFIX+'ff.w1.weight'].double()@r.T)
    put(PREFIX+'ff.w2.weight',r@state[PREFIX+'ff.w2.weight'].double())
    put(PREFIX+'norm.weight',visual_weight)
    put(PREFIX+'norm.bias',visual_bias)
    assert sum(x.numel() for x in out.values())==sum(x.numel() for x in state.values())
    return out

def predict_from_state(x,state):
    """Small pure native formula for independent folding checks, evaluation only."""
    f=torch.nn.functional
    a=f.linear(x,state['proj_in.weight'],state['proj_in.bias'])
    h=a+f.linear(f.gelu(f.linear(a,state[PREFIX+'ff.w1.weight'])),state[PREFIX+'ff.w2.weight'])
    return f.layer_norm(h,(h.shape[-1],),state[PREFIX+'norm.weight'],state[PREFIX+'norm.bias'],1e-6)

def same_concept_derangement(concepts):
    """Fixed +1 cyclic stimulus shift within each role-FIT concept; no labels at inference."""
    groups={}
    for i,c in enumerate(concepts):groups.setdefault(c,[]).append(i)
    ids=list(range(len(concepts)))
    for positions in groups.values():
        assert len(positions)==10
        for j,i in enumerate(positions):ids[i]=positions[(j+1)%10]
    assert sorted(ids)==list(range(len(ids))) and all(i!=v for i,v in enumerate(ids))
    return ids
