"""Physical-repeat task-risk extrapolation, with explicit matched controls.

Classical count extrapolation is a working statistical model, not a theorem
about EEG noise or a claim of originality. Never consumes R80 EEG/answers.
"""
import itertools
import numpy as np

SUBSETS=tuple(s for n in range(1,5) for s in itertools.combinations(range(4),n))
COUNTS=np.array([len(s) for s in SUBSETS])
ARMS=('R4_continued','positive_views','count_corrected')

def coefficients(target_count=80):
    assert isinstance(target_count,int) and target_count>=1
    nodes=np.array([1.,.5,.25]);x=1./target_count
    w=np.array([np.prod([(x-nodes[j])/(nodes[i]-nodes[j]) for j in range(3) if i!=j]) for i in range(3)])
    return np.array([w[0],w[1],0.,w[2]])

def weights(arm):
    assert arm in ARMS
    w=coefficients()
    return np.array([0.,0.,0.,1.]) if arm=='R4_continued' else np.abs(w)/np.abs(w).sum() if arm=='positive_views' else w

def loss_bounds(batch,arm,scale_cap=100.):
    assert batch>=2 and scale_cap>0
    w=weights(arm);upper=np.log(batch)+2*scale_cap
    return float(w[w<0].sum()*upper),float(w[w>0].sum()*upper)

def collect_training_gradients(net,meta,block,targets,seed,objective,guard=lambda:None):
    """All 15 views at common dropout; retain exactly ONE R4 BN update.

    R3 is logged only and receives zero coefficient in all three arms. Each arm
    computes the same views. Its own optimizer state evolves on its own risk.
    """
    import torch
    pars=meta['parameters'];device=pars[0].device
    buffers={n:b.detach().clone() for n,b in net.named_buffers()}
    Y=torch.nn.functional.normalize(torch.as_tensor(targets,device=device,dtype=pars[0].dtype),dim=1)
    grads_by_count=torch.zeros((4,meta['dimension']),device=device,dtype=torch.float64)
    losses=[];active=None
    for subset in SUBSETS:
        guard()
        with torch.no_grad():
            for n,b in net.named_buffers():b.copy_(buffers[n])
        torch.cuda.manual_seed(seed)
        x=torch.as_tensor(np.ascontiguousarray(block[:,subset].mean(1)),device=device,dtype=pars[0].dtype)
        pred=net(x,torch.ones(len(block),device=device,dtype=torch.long))
        logits=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(pred,dim=1)@Y.T
        loss,_=objective(logits,'nce_continued')
        assert torch.isfinite(loss)
        gs=torch.autograd.grad(loss,pars,allow_unused=True)
        used=[g is not None for g in gs]
        if active is None:active=used
        assert active==used
        flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(pars,gs)])
        assert torch.isfinite(flat).all()
        grads_by_count[len(subset)-1].add_(flat.double(),alpha=1/int((COUNTS==len(subset)).sum()))
        losses.append(float(loss.detach()))
    # Last view is R4: buffers now equal a single native R4 training forward.
    assert SUBSETS[-1]==(0,1,2,3)
    means=np.array([np.mean(np.array(losses)[COUNTS==n]) for n in range(1,5)])
    return grads_by_count,means,active

def combine_and_assign(net,meta,gradients,losses,active,arm):
    import torch
    w=weights(arm);g=sum(float(a)*v for a,v in zip(w,gradients));assert torch.isfinite(g).all()
    risk=float(w@losses);lower,upper=loss_bounds(160,arm)
    assert lower-1e-4<=risk<=upper+1e-4
    net.zero_grad(set_to_none=True)
    for p,(a,b),use in zip(meta['parameters'],meta['slices'],active):p.grad=g[a:b].reshape_as(p).to(p.dtype).clone() if use else None
    norm=float(torch.nn.utils.clip_grad_norm_(meta['parameters'],1.));assert np.isfinite(norm)
    cosine=float(torch.dot(g,gradients[3])/(torch.linalg.vector_norm(g)*torch.linalg.vector_norm(gradients[3])+1e-30))
    return dict(risk=risk,losses_by_count=losses.tolist(),preclip_norm=norm,gradient_cosine_vs_own_R4=cosine,
                effective_logit_scale=float(net.logit_scale.exp().clamp_max(100).detach()),weights=w.tolist())

def math_checks():
    w=coefficients();assert np.allclose(w,[.30875,-1.87625,0.,2.5675],atol=1e-13)
    x=1/np.array([1,2,3,4],float)
    for power in range(3):assert abs(w@(x**power)-(1/80)**power)<1e-13
    assert np.allclose(coefficients(3),[-1/27,12/27,0,16/27])
    for a in ARMS:
        assert abs(weights(a).sum()-1)<1e-13
        lo,hi=loss_bounds(160,a);assert np.isfinite([lo,hi]).all()
    assert np.all(weights('positive_views')>=0)
    # Cubic residual: exact quadratics do not establish universal extrapolation.
    cubic_error=float(w@(x**3)-(1/80)**3);assert abs(cubic_error)>.01
    return dict(status='PASS_COUNT_COEFFICIENTS_BOUNDS_AND_NONUNIVERSAL_COUNTEREXAMPLE',weights=w.tolist(),
                positive_control=weights('positive_views').tolist(),quadratic_moments_exact=True,
                cubic_extrapolation_error=cubic_error,normalised_capped_NCE_bounded=True,
                bounds={a:loss_bounds(160,a) for a in ARMS},unbiasedness_or_R80_validity_proven=False)

if __name__=='__main__':
    import json
    print(json.dumps(math_checks(),indent=2))
