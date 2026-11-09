"""Complete-epoch EEG alignment and immutable exact-resume transactions."""
from contextlib import nullcontext
import hashlib,json,math,os,tempfile,time
from pathlib import Path
import torch
import torch.nn.functional as F

def objective(c,target,logit_scale):
    similarity=logit_scale.float().exp().clamp(max=100)*F.normalize(c.float(),dim=-1)@F.normalize(target.float(),dim=-1).T
    labels=torch.arange(len(c),device=c.device)
    nce=(F.cross_entropy(similarity,labels)+F.cross_entropy(similarity.T,labels))/2
    mse=F.mse_loss(c.float(),target.float())
    return .8*nce+.2*mse,nce,mse

def learning_rate(update,total,warmup):
    assert 0<=update<total and 1<warmup<total
    if update<warmup:return 1e-6+(5e-4-1e-6)*update/(warmup-1)
    position=(update-warmup)/max(1,total-1-warmup)
    return 1e-6+.5*(5e-4-1e-6)*(1+math.cos(math.pi*position))

def optimizer_for(model):
    groups={};names=[]
    for name,p in model.named_parameters():
        assert p.requires_grad
        if name.startswith('blocks.'):layer=int(name.split('.')[1])+1
        elif name.startswith(('head.','fc_norm.')) or name=='logit_scale':layer=13
        else:layer=0
        decay=0. if p.ndim<=1 or name in {'pos_embed','cls_token','time_embed'} else .05
        key=(layer,decay)
        if key not in groups:groups[key]=dict(params=[],parameter_names=[],lr_scale=.8**(13-layer),weight_decay=decay)
        groups[key]['params'].append(p);groups[key]['parameter_names'].append(name);names.append(name)
    assert len(names)==len(set(names))==len(list(model.parameters()))
    return torch.optim.AdamW(list(groups.values()),lr=5e-4,betas=(.9,.999),eps=1e-8)

def run_epoch(model,optimizer,shuffle_rng,eeg,targets,selected,positions,*,epoch,updates,total_updates,warmup_updates,batch,device,guard=lambda:None):
    model.train();order=selected[torch.randperm(len(selected),generator=shuffle_rng).numpy()]
    seen=[];totals=[0.,0.,0.];began=time.monotonic();gradient=0.
    for lo in range(0,len(order),batch):
        guard();ids=order[lo:lo+batch];seen.extend(ids.tolist())
        x=torch.tensor(eeg[ids],device=device);y=torch.tensor(targets[ids],device=device)
        lr=learning_rate(updates,total_updates,warmup_updates)
        for group in optimizer.param_groups:group['lr']=lr*group['lr_scale']
        optimizer.zero_grad(set_to_none=True)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
        with context:
            c=model(x,input_chans=positions)[0];loss,nce,mse=objective(c,y,model.logit_scale)
        assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():model.logit_scale.clamp_(max=math.log(100))
        for index,value in enumerate((loss,nce,mse)):totals[index]+=float(value.detach())*len(ids)
        updates+=1;gradient=float(norm);assert updates<=total_updates
    assert len(seen)==len(selected) and sorted(seen)==sorted(selected.tolist())
    return dict(epoch=epoch,updates=updates,queries=len(seen),order_sha256=hashlib.sha256(__import__('numpy').asarray(seen,dtype='<i8').tobytes()).hexdigest(),loss=totals[0]/len(seen),NCE=totals[1]/len(seen),MSE=totals[2]/len(seen),last_gradient_norm=gradient,last_base_lr=lr,seconds=time.monotonic()-began)

def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def snapshot(model,optimizer,shuffle_rng,*,epoch,updates,spec_sha256,role,device):
    return dict(schema='EEG_alignment_complete_epoch_v1',model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),torch_rng=torch.get_rng_state().clone(),cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()] if str(device).startswith('cuda') else [],shuffle_rng=shuffle_rng.get_state().clone(),epoch=epoch,updates=updates,spec_sha256=spec_sha256,role=role)

def restore(value,model,optimizer,shuffle_rng,*,spec_sha256,role,updates_per_epoch,total_epochs,device):
    assert value['schema']=='EEG_alignment_complete_epoch_v1' and value['spec_sha256']==spec_sha256 and value['role']==role
    assert 1<=value['epoch']<=total_epochs and value['updates']==value['epoch']*updates_per_epoch
    assert set(value['model'])==set(model.state_dict())
    for group,expected in zip(value['optimizer']['param_groups'],optimizer.param_groups):
        assert group['parameter_names']==expected['parameter_names'] and group['lr_scale']==expected['lr_scale'] and group['weight_decay']==expected['weight_decay']
        assert tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
    assert len(value['optimizer']['param_groups'])==len(optimizer.param_groups)
    assert len(value['optimizer']['state'])==len(list(model.parameters()))
    for state in value['optimizer']['state'].values():
        assert float(state['step'])==value['updates'] and all(torch.isfinite(state[k]).all() for k in ('exp_avg','exp_avg_sq'))
    model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
    shuffle_rng.set_state(value['shuffle_rng']);torch.set_rng_state(value['torch_rng'])
    if str(device).startswith('cuda'):
        assert len(value['cuda_rng'])==torch.cuda.device_count();torch.cuda.set_rng_state_all(value['cuda_rng'])
    else:assert not value['cuda_rng']
    return value['epoch'],value['updates']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def publish_epoch(root,value,row,*,fault=None):
    root=Path(root);root.mkdir(exist_ok=True);destination=root/f"epoch_{value['epoch']:03d}";assert not destination.exists()
    pending=Path(tempfile.mkdtemp(prefix=destination.name+'.pending.',dir=root))
    with (pending/'resume.pt').open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    with (pending/'epoch.json').open('x') as f:json.dump(row,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    commit=dict(epoch=value['epoch'],updates=value['updates'],role=value['role'],spec_sha256=value['spec_sha256'],resume_sha256=sha(pending/'resume.pt'),epoch_sha256=sha(pending/'epoch.json'))
    with (pending/'commit.json').open('x') as f:json.dump(commit,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if fault:fault('before_directory_commit')
    pending.rename(destination)
    return dict(path=str(destination/'commit.json'),sha256=sha(destination/'commit.json'),**commit)
