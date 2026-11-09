"""Native global NCE gradients and counterfactual AdamW updates at fixed state."""
import numpy as np
import atm_full_repeat_gradient_core_20261105_v1 as F
SUBSETS=F.SUBSETS;COUNTS=F.COUNTS
VARIANTS=('R1','R2','R3','R4','linear_R3','quadratic_R3','R4_gradient_norm_matched_to_quadratic')

def restore_buffers(net,state):
    import torch
    with torch.no_grad():
        for name,b in net.named_buffers():b.copy_(state[name].to(device=b.device,dtype=b.dtype))

def collect(net,meta,block,targets,state,seed,device,dtype,objective,record=False,guard=lambda:None):
    import torch
    Y=torch.nn.functional.normalize(torch.as_tensor(targets,device=device,dtype=dtype),dim=1)
    counts=torch.zeros((4,meta['dimension']),device=device,dtype=torch.float64)
    ls=[];preds=[];samples=[];norms=[];mask=None;logits4=None
    for j,subset in enumerate(SUBSETS):
        guard();restore_buffers(net,state)
        if device=='cuda':torch.cuda.manual_seed(seed)
        else:torch.manual_seed(seed)
        x=torch.as_tensor(np.ascontiguousarray(block[:,subset].mean(1)),device=device,dtype=dtype)
        z=net(x,torch.ones(len(x),device=device,dtype=torch.long))
        logits=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(z,dim=1)@Y.T
        loss,_=objective(logits,'nce_continued')
        grads=torch.autograd.grad(loss,meta['parameters'],allow_unused=True)
        used=[g is not None for g in grads]
        if mask is None:mask=used
        assert mask==used
        flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(meta['parameters'],grads)])
        assert torch.isfinite(flat).all();counts[len(subset)-1].add_(flat.double(),alpha=1/int((COUNTS==len(subset)).sum()))
        ls.append(float(loss.detach()))
        if record:
            preds.append(z[:8].detach().cpu().numpy());samples.append(flat[torch.as_tensor(meta['indices'],device=device)].detach().cpu().double().numpy())
            norms.append([float(flat[a:b].double().square().sum()) for a,b in meta['slices']])
            if len(subset)==4:logits4=logits.detach().cpu().numpy()
    restore_buffers(net,state)
    result=dict(losses=np.array(ls),active_tensors=np.array(mask,bool))
    if record:result.update(first8_predictions=np.stack(preds,1),sampled_gradients=np.stack(samples),tensor_squared_norms=np.array(norms),R4_logits=logits4)
    return counts,result

def optimizer_reference(meta,checkpoint,device):
    import torch
    op=checkpoint['optimizer'];assert len(op['param_groups'])==1
    group=op['param_groups'][0];ids=group['params'];assert len(ids)==len(meta['names'])
    assert group['lr']==1e-4 and group['weight_decay']==.01 and group['betas']==(.9,.999) and group['eps']==1e-8
    assert not group.get('amsgrad',False) and not group.get('maximize',False)
    first=[];second=[];steps=[];has=[];theta=[]
    for name,p,i in zip(meta['names'],meta['parameters'],ids):
        s=op['state'].get(i);has.append(s is not None)
        first.append((s['exp_avg'] if s is not None else torch.zeros_like(p,device='cpu')).reshape(-1).to(device=device,dtype=torch.float64))
        second.append((s['exp_avg_sq'] if s is not None else torch.zeros_like(p,device='cpu')).reshape(-1).to(device=device,dtype=torch.float64))
        step=int(s['step']) if s is not None else 0
        if s is not None:assert step==830
        steps.append(torch.full((p.numel(),),step+1,device=device,dtype=torch.float64))
        theta.append(checkpoint['model'][name].reshape(-1).to(device=device,dtype=torch.float64))
    return dict(m=torch.cat(first),v=torch.cat(second),steps=torch.cat(steps),theta=torch.cat(theta),has_state=has,
        lr=group['lr'],weight_decay=group['weight_decay'],betas=group['betas'],eps=group['eps'])

def adam_vectors(g,meta,active,reference):
    import torch
    assert active.tolist()==reference['has_state']
    mask=torch.cat([torch.full((b-a,),bool(use),device=g.device,dtype=torch.bool) for (a,b),use in zip(meta['slices'],active)])
    norm=torch.linalg.vector_norm(g);clipped=g*torch.clamp(1/(norm+1e-6),max=1)
    b1,b2=reference['betas'];m=b1*reference['m']+(1-b1)*clipped;v=b2*reference['v']+(1-b2)*clipped.square()
    mhat=m/(1-b1**reference['steps']);vhat=v/(1-b2**reference['steps'])
    data=-reference['lr']*mhat/(vhat.sqrt()+reference['eps']);data=torch.where(mask,data,0.)
    decay=torch.where(mask,-reference['lr']*reference['weight_decay']*reference['theta'],0.)
    return clipped,data,data+decay

def measurements(g,meta,active,reference):
    import torch
    quad=(-g[0]+12*g[1]+16*g[3])/27
    vectors=[*g,g[1]/3+2*g[3]/3,quad,g[3]*torch.linalg.vector_norm(quad)/torch.linalg.vector_norm(g[3])]
    spaces={'raw_gradient':torch.stack(vectors)}
    parts=[adam_vectors(x,meta,active,reference) for x in vectors]
    spaces.update(clipped_gradient=torch.stack([p[0] for p in parts]),Adam_data_update=torch.stack([p[1] for p in parts]),Adam_total_update=torch.stack([p[2] for p in parts]))
    summary={};grams={};norms={};samples={}
    for key,V in spaces.items():
        H=V@V.T;N=torch.linalg.vector_norm(V,dim=1);truth=V[2];tn=N[2]
        grams[key]=H.detach().cpu().numpy();norms[key]=N.detach().cpu().numpy()
        samples[key]=V[:,torch.as_tensor(meta['indices'],device=V.device)].detach().cpu().numpy()
        summary[key]={}
        for j,name in enumerate(VARIANTS):
            p=V[j];summary[key][name]=dict(relative_L2_error=float(torch.linalg.vector_norm(p-truth)/tn),cosine=float(torch.dot(p,truth)/(N[j]*tn)),norm=float(N[j]))
        # Descriptive oracle scale explains magnitude; it is not a deployable control.
        alpha=H[3,2]/H[3,3];residual=torch.linalg.vector_norm(alpha*V[3]-truth)/tn
        summary[key]['oracle_single_scale_R4']=dict(alpha=float(alpha),relative_L2_error=float(residual),uses_true_R3=True)
    # Match the actual preconditioned update magnitude, preserving original decay.
    data=spaces['Adam_data_update'];decay=spaces['Adam_total_update'][3]-data[3]
    matched=data[3]*torch.linalg.vector_norm(data[5])/torch.linalg.vector_norm(data[3])
    for key,p,t in [('Adam_data_update',matched,data[2]),('Adam_total_update',matched+decay,spaces['Adam_total_update'][2])]:
        summary[key]['R4_update_norm_matched_to_quadratic']=dict(relative_L2_error=float(torch.linalg.vector_norm(p-t)/torch.linalg.vector_norm(t)),cosine=float(torch.dot(p,t)/(torch.linalg.vector_norm(p)*torch.linalg.vector_norm(t))),norm=float(torch.linalg.vector_norm(p)))
    return dict(summary=summary,grams=grams,norms=norms,samples=samples),spaces
