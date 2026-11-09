"""Physical R1/R4 task-step proximal rule. Classical solve, empirical hypothesis."""
import numpy as np
ARMS=('repeat_step','sign_control','norm_control')
VIEWS=((0,),(1,),(2,),(3,),(0,1,2,3))

def collect(net,meta,block,targets,seed,objective,guard=lambda:None):
    import torch
    ps=meta['parameters'];device=ps[0].device;dtype=ps[0].dtype
    buffers={n:b.detach().clone() for n,b in net.named_buffers()};Y=torch.nn.functional.normalize(torch.as_tensor(targets,device=device,dtype=dtype),dim=1)
    vectors=[];losses=[];active=None
    for subset in VIEWS:
        guard()
        with torch.no_grad():
            for n,b in net.named_buffers():b.copy_(buffers[n])
        if device.type=='cuda':torch.cuda.manual_seed(seed)
        else:torch.manual_seed(seed)
        x=torch.as_tensor(np.ascontiguousarray(block[:,subset].mean(1)),device=device,dtype=dtype)
        z=net(x,torch.ones(len(block),device=device,dtype=torch.long))
        S=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(z,dim=1)@Y.T
        loss,_=objective(S,'nce_continued');gs=torch.autograd.grad(loss,ps,allow_unused=True)
        mask=[g is not None for g in gs]
        if active is None:active=mask
        assert mask==active and torch.isfinite(loss)
        flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(ps,gs)]).double()
        assert torch.isfinite(flat).all();vectors.append(flat);losses.append(float(loss.detach()))
    return torch.stack(vectors),np.asarray(losses),active

def signs_for(meta,device):
    import torch
    # Fixed parameter sign conjugation commutes with diagonal D, preserving U^TDU.
    return torch.as_tensor(np.random.default_rng(1729).choice([-1.,1.],size=meta['dimension']),device=device,dtype=torch.float64)

def proximal(data,g4,G,D,signs=None):
    import torch
    U=(G-g4).T/2
    if signs is not None:U=U*signs[:,None]
    DU=D[:,None]*U;B=U.T@DU;B=(B+B.T)/2;tau=torch.dot(g4,D*g4);b=U.T@data
    e,V=torch.linalg.eigh(B);assert float(e.min())>=-1e-10*max(1.,float(e.max()))
    e=e.clamp_min(0);tol=torch.finfo(torch.float64).eps*torch.maximum(e.max(),tau)*16
    inv=torch.where(e+tau>tol,(e+tau).reciprocal(),torch.zeros_like(e));coef=V@(inv*(V.T@b))
    change=-DU@coef;new=data+change;res=(B+tau*torch.eye(4,device=B.device,dtype=B.dtype))@coef-b
    rel=float(torch.linalg.vector_norm(res)/(torch.linalg.vector_norm(b)+1e-30));assert rel<1e-6
    before=float(b@b);after=float((U.T@new).square().sum());assert after<=before+1e-10*max(1.,before)
    return new,dict(B=B,tau=tau,b=b,coefficient=coef,eigenvalues=e,residual=rel,sensitivity_before=before,sensitivity_after=after),U

def provisional_Adam(net,opt,meta,gradients,active):
    """Execute exactly original FP32 Adam first; retain its actual updated state.

    Decompose the rounded native parameter operation at its decayed intermediate,
    so disabling the correction preserves the real native step bit-for-bit.
    """
    import torch
    assert len(opt.param_groups)==1
    group=opt.param_groups[0];assert group['lr']==1e-4 and group['weight_decay']==.01 and group['betas']==(.9,.999) and group['eps']==1e-8
    net.zero_grad(set_to_none=True)
    initial=torch.cat([p.detach().reshape(-1).double() for p in meta['parameters']])
    decayed=[]
    for p,(a,b),use in zip(meta['parameters'],meta['slices'],active):
        p.grad=gradients[4,a:b].reshape_as(p).to(p.dtype).clone() if use else None
        decayed.append((p.detach()*(1-group['lr']*group['weight_decay']) if use else p.detach()).reshape(-1).double())
    decayed=torch.cat(decayed);norm=float(torch.nn.utils.clip_grad_norm_(meta['parameters'],1.));opt.step()
    actual=torch.cat([p.detach().reshape(-1).double() for p in meta['parameters']]);data=actual-decayed
    D=[];step=None
    for p,use in zip(meta['parameters'],active):
        state=opt.state.get(p)
        assert bool(state)==bool(use)
        if use:
            st=int(state['step']);step=st if step is None else step;assert st==step
            v=state['exp_avg_sq'].reshape(-1).double()/(1-group['betas'][1]**st)
            D.append((v.sqrt()+group['eps']).reciprocal())
        else:D.append(torch.zeros(p.numel(),device=p.device,dtype=torch.float64))
    return dict(initial=initial,actual=actual,decay=decayed-initial,data=data,D=torch.cat(D),preclip_norm=norm,Adam_step=step)

def finish(net,meta,gradients,active,provisional,arm,signs,record=False):
    import torch
    assert arm in ARMS or arm=='native'
    native=provisional['data'];D=provisional['D'];g=gradients[4]
    correct,proof,U=proximal(native,g,gradients[:4],D)
    if arm=='repeat_step':chosen=correct
    elif arm=='sign_control':chosen,proof,U=proximal(native,g,gradients[:4],D,signs)
    elif arm=='norm_control':
        norm=torch.linalg.vector_norm(native);chosen=native*(torch.linalg.vector_norm(correct)/norm) if float(norm)>0 else native.clone()
    else:chosen=native
    correction=chosen-native
    with torch.no_grad():
        for p,(a,b),use in zip(meta['parameters'],meta['slices'],active):
            if use:p.copy_((provisional['actual'][a:b]+correction[a:b]).reshape_as(p).to(p.dtype))
            else:assert torch.equal(p.reshape(-1).double(),provisional['initial'][a:b])
    actual=torch.cat([p.detach().reshape(-1).double() for p in meta['parameters']]);assert torch.isfinite(actual).all()
    norm0=torch.linalg.vector_norm(native);norm1=torch.linalg.vector_norm(chosen)
    summary=dict(arm=arm,preclip_norm=provisional['preclip_norm'],Adam_step=provisional['Adam_step'],
        data_step_norm=float(norm1),original_data_step_norm=float(norm0),data_step_cosine=float(torch.dot(chosen,native)/(norm0*norm1+1e-30)),
        R4_first_order_data_slope=float(g@chosen),R1_mean_first_order_data_slope=float(gradients[:4].mean(0)@chosen),
        original_R4_first_order_data_slope=float(g@native),original_R1_mean_first_order_data_slope=float(gradients[:4].mean(0)@native),
        proximal_residual=proof['residual'],registered_sensitivity_before=proof['sensitivity_before'],registered_sensitivity_after=proof['sensitivity_after'],
        chosen_R1_R4_sensitivity=float((((gradients[:4]-g)@chosen)/2).square().sum()),
        native_R1_R4_sensitivity=float((((gradients[:4]-g)@native)/2).square().sum()),
        actual_parameter_step_norm=float(torch.linalg.vector_norm(actual-provisional['initial'])),
        FP32_application_max_error=float((actual-provisional['actual']-correction).abs().max()))
    fixture=None
    if record:
        ix=torch.as_tensor(meta['indices'],device=g.device);fixture={k:v.detach().cpu().numpy() for k,v in proof.items() if torch.is_tensor(v)}
        fixture.update(D= D[ix].cpu().numpy(),U=U[ix].cpu().numpy(),chosen=chosen[ix].cpu().numpy(),
            original=native[ix].cpu().numpy(),actual=actual[ix].cpu().numpy(),native_actual=provisional['actual'][ix].cpu().numpy(),
            parameter_dtype='float32',global_correct_norm=float(torch.linalg.vector_norm(correct)),global_native_norm=float(norm0))
    return summary,fixture
