"""Evaluation-mode objectives and whole-encoder gradient evidence; no optimizer."""
import itertools
import numpy as np

SUBSETS=tuple(s for n in range(1,5) for s in itertools.combinations(range(4),n))
COUNTS=np.array([len(s) for s in SUBSETS])
OBJECTIVES=('raw_MSE','one_minus_cosine','within10_CE_temperature1')
METHODS=('linear_R2_R4','quadratic_R1_R2_R4','constant_R4')


def metadata(net):
    names=[];params=[];slices=[];indices=[];offset=0
    for name,p in net.named_parameters():
        names.append(name);params.append(p)
        slices.append((offset,offset+p.numel()))
        indices.extend((offset+np.unique(np.linspace(0,p.numel()-1,min(8,p.numel()),dtype=np.int64))).tolist())
        offset+=p.numel()
    assert offset==3200413
    return dict(names=names,parameters=params,slices=slices,indices=np.asarray(indices,np.int64),dimension=offset)


def losses(net,x,y10,ix):
    import torch
    z=net(x,torch.ones(len(x),dtype=torch.long,device=x.device))
    y=y10[torch.arange(len(x),device=x.device),ix]
    unit=torch.nn.functional.normalize(z,dim=1)
    v=torch.nn.functional.normalize(y10,dim=2)
    scores=(unit[:,None]*v).sum(2)
    ce=torch.nn.functional.cross_entropy(scores,ix)
    return z,torch.stack(((z-y).square().mean(),1-(unit*v[torch.arange(len(x),device=x.device),ix]).sum(1).mean(),ce))


def collect(net,meta,block,targets10,correct,device,dtype,record=False,guard=lambda:None):
    import torch
    dimension=meta['dimension'];pars=meta['parameters']
    counts=torch.zeros((4,3,dimension),device=device,dtype=torch.float64)
    values=np.empty((15,3),float);predictions=[];samples=[];norms=[]
    sample_indices=torch.as_tensor(meta['indices'],device=device)
    y10=torch.as_tensor(targets10,device=device,dtype=dtype)
    ix=torch.as_tensor(correct,device=device,dtype=torch.long)
    for j,subset in enumerate(SUBSETS):
        guard()
        # Preserve published float32 averaging before either neural execution precision.
        x=torch.as_tensor(np.ascontiguousarray(block[:,subset].mean(1)),device=device,dtype=dtype)
        pred,ls=losses(net,x,y10,ix)
        values[j]=ls.detach().cpu().double().numpy()
        recorded_samples=[];recorded_norms=[]
        for k in range(3):
            grads=torch.autograd.grad(ls[k],pars,retain_graph=k<2,allow_unused=True)
            flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(pars,grads)])
            assert torch.isfinite(flat).all()
            counts[len(subset)-1,k].add_(flat.double(),alpha=1/int((COUNTS==len(subset)).sum()))
            if record:
                recorded_samples.append(flat[sample_indices].detach().cpu().double().numpy())
                recorded_norms.append(np.array([float(flat[a:b].double().square().sum()) for a,b in meta['slices']]))
        if record:
            predictions.append(pred.detach().cpu().numpy())
            samples.append(np.stack(recorded_samples));norms.append(np.stack(recorded_norms))
    result=dict(losses=values)
    if record:result.update(predictions=np.stack(predictions,1),gradient_samples=np.stack(samples),tensor_squared_norms=np.stack(norms))
    return counts,result


def evidence(g,meta):
    import torch
    result={};whole_norms=torch.linalg.vector_norm(g,dim=-1)
    for k,key in enumerate(OBJECTIVES):
        true=g[2,k];tn=torch.linalg.vector_norm(true);assert tn>0
        result[key]={}
        pred=(g[1,k]/3+2*g[3,k]/3,(-g[0,k]+12*g[1,k]+16*g[3,k])/27,g[3,k])
        for name,p in zip(METHODS,pred):
            error=p-true;pn=torch.linalg.vector_norm(p)
            layer_error={group:float(sum(error[a:b].square().sum() for n,(a,b) in zip(meta['names'],meta['slices']) if n.split('.')[0]==group)) for group in sorted({n.split('.')[0] for n in meta['names']})}
            result[key][name]=dict(relative_L2_error=float(torch.linalg.vector_norm(error)/tn),
                cosine=float(torch.dot(p,true)/(pn*tn)),absolute_L2_error=float(torch.linalg.vector_norm(error)),
                squared_error_by_native_module=layer_error)
    normtable=np.stack([g[:,:,a:b].square().sum(-1).detach().cpu().numpy() for a,b in meta['slices']],axis=-1)
    return dict(comparison=result,norms=whole_norms.detach().cpu().numpy(),tensor_squared_norms=normtable,
                gradient_samples=g[:,:,torch.as_tensor(meta['indices'],device=g.device)].detach().cpu().numpy(),
                inactive_parameter_tensors=[name for j,name in enumerate(meta['names']) if np.all(normtable[:,:,j]==0)])


def perturb(net,meta,base,delta):
    import torch
    with torch.no_grad():
        for name,p,(a,b) in zip(meta['names'],meta['parameters'],meta['slices']):
            p.copy_(base[name].to(device=p.device,dtype=p.dtype)+delta[a:b].reshape_as(p).to(p.dtype))
