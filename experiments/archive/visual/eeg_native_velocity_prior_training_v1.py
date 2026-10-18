"""SERVER ONLY: known conditional DDPM prior versus matched point regression.

Uses reviewed, pinned ATM DiffusionPriorUNet with native BrainHIVE F coordinates.
No new-method claim: separate distribution modeling from a deterministic map.
"""
import ast
import ctypes
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

assert sys.platform=='linux' and os.uname().machine=='x86_64'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
OUT=Path('/root/mced_eeg_native_velocity_prior_training_v1')
QUERY=Path('/root/mced_eeg_brainhive_query_cache_v2')
AUDIT=Path('/root/mced_eeg_atm_prior_source_audit_v1')
SPLIT=Path('/root/mced_eeg_native_target_level_reference_v1')
REFERENCE=Path('/root/mced_eeg_native_conditional_prior_training_v1')
assert not OUT.exists();OUT.mkdir()
START=time.monotonic();FDS=[];SAVED=None
SEEDS=[2026110101,2026110102];EEG_SEEDS=[2026109501,2026109502]
COUNTS=dict(train_calls=0,train_states=0,backward_calls=0,optimizer_updates=0,eval_calls=0,eval_states=0)

def identity(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

def write(n,v,replace=False):
    p=OUT/n
    if replace:
        t=p.with_name(p.name+'.pending');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
    else:
        with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def adopt(path,wanted):
    fd=os.open(path,os.O_RDONLY);FDS.append(fd);own=Path(f'/proc/{os.getpid()}/fd/{fd}')
    assert fcntl.fcntl(fd,1034)==15 and identity(own)==wanted
    return own

def seal(name,tensor):
    lib=ctypes.CDLL(None,use_errno=True)
    fd=lib.syscall(ctypes.c_long(319),ctypes.c_char_p(('mced_native_prior_'+name).encode()),ctypes.c_uint(3))
    assert fd>=0,ctypes.get_errno();FDS.append(fd)
    with os.fdopen(os.dup(fd),'wb') as f:torch.save(tensor,f)
    fcntl.fcntl(fd,1033,15);assert fcntl.fcntl(fd,1034)==15
    p=Path(f'/proc/{os.getpid()}/fd/{fd}');(OUT/name).symlink_to(p)
    return dict(name=name,memfd_path=str(p),**identity(p))

try:
    write('plan.json',dict(script=identity(Path(__file__)),pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Known v-prediction control for unstable raw-native feature DDPM; preserve prior epsilon model and all image results',
        reference='Pinned ATM public DiffusionPriorUNet develop efb6b1d5d7dd14bea44ee5a952b72a951fa6da54, unmodified architecture',
        mathematical_change='VP noisy z_t=alpha*z+sigma*epsilon, v=alpha*epsilon-sigma*z. Predict v with one MSE; x0=alpha*z_t-sigma*v has bounded error coefficient sigma<=1, unlike epsilon inversion sigma/alpha.',
        prior_art='Standard v-prediction (Salimans/Ho Progressive Distillation ICLR2022), not new theory or original architecture',
        seeds=SEEDS,EEG_seeds=EEG_SEEDS,arms=['DDPM_velocity'],source_fit_images=15260,source_validation_images=1280,
        epochs=150,batch=1024,updates_per_model=2250,expected_updates=4500,expected_train_states=4578000,
        matching='Exact same architecture,initialization hashes,input sequence hashes,dropout program,noise/time generator program,optimizer and2250updates as existing epsilon models; compare existingpointcontrols without retraining',
        optimizer='Adam1e-3,wd0,cos2250steps,warmup500,clip1,batchwise10percent condition dropout',
        teacher_and_queries='Same frozen nativepreF raw1024visual targets and original source01mean4preAEEG queries; no officialtest/target data',
        scheduler='DDPMScheduler1000,linearbeta1e-4to.02,prediction_type=v_prediction,clip_sampleFalse; no additional clipping, normtarget orloss',
        inference='Same50ancestral steps,CFG5primary andCFG1feature sensitivity,seed2026110200+EEGseed coupled to oldepsilon path; no bestofN',
        evaluation='Alltwo models train beforevalidation. Raw feature error/norm are diagnostics, not imagequality; imagecomparison will retain same fixed32images andgeneratornoise as oldpriorstudy.',
        known_method_reference=True,full_paper_reproduction=False,novelty_established=False,SOTA=False,lease_seconds=43200))
    for d in [QUERY,AUDIT,REFERENCE]:
        c=json.loads((d/'final_commit.json').read_text());assert c['complete']
        for n,r in c['files'].items():assert identity(d/n)==r,(d,n)
    ready=json.loads((QUERY/'results.json').read_text());os.kill(ready['pid'],0)
    previous=json.loads((REFERENCE/'results.json').read_text());os.kill(previous['pid'],0)
    assert previous['counts']['optimizer_updates']==9000
    qpaths=[adopt(next(r for r in ready['exports'] if r['seed']==seed)['memfd_path'],
                 {k:next(r for r in ready['exports'] if r['seed']==seed)[k] for k in ['bytes','sha256']}) for seed in EEG_SEEDS]
    bindings=json.loads((QUERY/'input_bindings.json').read_text());feature=bindings['native_features']
    fpath=adopt(bindings['independently_owned_pre_feature_path'],{k:feature[k] for k in ['bytes','sha256']})
    assert identity(SPLIT/'split.json')==bindings['source_split']
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from diffusers import DDPMScheduler
    from diffusers.models.embeddings import Timesteps,TimestepEmbedding
    from diffusers.optimization import get_cosine_schedule_with_warmup
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>8<<30
    SAVED=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),torch.get_num_threads(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=AUDIT/'develop/Generation/diffusion_prior.py'
    assert identity(path)==dict(bytes=9681,sha256='7bccd517b5fa5d20c7f2738efdc417e785bcaf4a586713e95ca582ad8cd9ce82')
    nodes=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='DiffusionPriorUNet'];assert len(nodes)==1
    scope=dict(torch=torch,nn=nn,Timesteps=Timesteps,TimestepEmbedding=TimestepEmbedding)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    def make():return scope['DiffusionPriorUNet'](embed_dim=1024,cond_dim=1024,dropout=.1).cuda()
    target=torch.load(fpath,weights_only=True).float().cuda();assert target.shape==(16540,1024) and torch.isfinite(target).all()
    heldout=set(json.loads((SPLIT/'split.json').read_text())['heldout_concepts'])
    fit=torch.tensor([i for i in range(16540) if i//10 not in heldout]);val=torch.tensor([i for i in range(16540) if i//10 in heldout])
    assert len(fit)==15260 and len(val)==1280
    # Preflight the actual reviewed class's complete backwards path before costly training.
    torch.manual_seed(2026110099);probe=make();xx=target[fit[:4]].detach().clone().requires_grad_(True)
    cc=target[fit[4:8]].detach().clone().requires_grad_(True);tt=torch.tensor([0,17,500,999],device='cuda')
    yy=probe(xx,tt,cc);ll=yy.square().mean();ll.backward()
    assert yy.shape==(4,1024) and torch.isfinite(ll) and xx.grad.norm()>0 and cc.grad.norm()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in probe.parameters())
    parameter_count=sum(p.numel() for p in probe.parameters())
    write('gradient_preflight.json',dict(complete=True,parameter_count=parameter_count,probe_forward_calls=1,probe_states=4,
        input_gradient_norm=float(xx.grad.norm()),condition_gradient_norm=float(cc.grad.norm()),optimizer_updates=0,
        actual_public_class_complete_backward_finite=True))
    del probe,xx,cc,tt,yy,ll;torch.cuda.empty_cache()
    scheduler_config=dict(num_train_timesteps=1000,beta_start=.0001,beta_end=.02,beta_schedule='linear',prediction_type='v_prediction',clip_sample=False)
    noise_scheduler=DDPMScheduler(**scheduler_config)
    write('scheduler_config.json',dict(noise_scheduler.config))
    rows=[];exports=[];matched={};query_tensors=[]
    for seed,eeg_seed,qpath in zip(SEEDS,EEG_SEEDS,qpaths):
        query=torch.load(qpath,weights_only=True).float().cuda();assert query.shape==target.shape and torch.isfinite(query).all()
        query_tensors.append(query)
        for arm in ['DDPM_velocity']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=make()
            h=hashlib.sha256()
            for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
            initial=h.hexdigest()
            if seed in matched:assert matched[seed]['initial_state_sha256']==initial
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
            lr_scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=500,num_training_steps=2250)
            order=torch.Generator().manual_seed(seed+100);augmentation_cpu=torch.Generator().manual_seed(seed+200)
            augmentation_cuda=torch.Generator(device='cuda').manual_seed(seed+300)
            sequence=hashlib.sha256();loss_total=0.;updates=0;condition_dropped=0;model.train()
            for epoch in range(150):
                for ids in fit[torch.randperm(len(fit),generator=order)].split(1024):
                    sequence.update(ids.numpy().tobytes());n=len(ids)
                    dropped=bool(torch.rand((),generator=augmentation_cpu)<.1);condition_dropped+=int(dropped)
                    c=None if dropped else query[ids]
                    noise=torch.randn((n,1024),device='cuda',generator=augmentation_cuda)
                    timestep=torch.randint(0,1000,(n,),device='cuda',generator=augmentation_cuda)
                    z=noise_scheduler.add_noise(target[ids],noise,timestep)
                    label=noise_scheduler.get_velocity(target[ids],noise,timestep)
                    output=model(z,timestep,c);loss=F.mse_loss(output,label);assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();lr_scheduler.step();updates+=1;loss_total+=float(loss.detach())
                    for k in ['train_calls','optimizer_updates','backward_calls']:COUNTS[k]+=1
                    COUNTS['train_states']+=n
                if epoch==0 or (epoch+1)%25==0:
                    print(json.dumps(dict(stage='NATIVE_CONDITIONAL_PRIOR_TRAIN',arm=arm,seed=seed,epoch=epoch+1,updates=updates,mean_loss=loss_total/updates,counts=COUNTS)),flush=True)
            assert updates==2250
            row=dict(arm=arm,seed=seed,EEG_seed=eeg_seed,updates=updates,trainable_parameters=parameter_count,
                initial_state_sha256=initial,input_sequence_sha256=sequence.hexdigest(),condition_dropped_batches=condition_dropped,
                mean_training_loss=loss_total/updates)
            ref=next(r for r in previous['training'] if r['seed']==seed and r['arm']=='DDPM_epsilon')
            for key in ['initial_state_sha256','input_sequence_sha256','condition_dropped_batches','updates','trainable_parameters']:assert row[key]==ref[key],key
            row['matched_epsilon_reference']=ref
            if seed in matched:
                for key in ['initial_state_sha256','input_sequence_sha256','condition_dropped_batches','updates','trainable_parameters']:assert row[key]==matched[seed][key],key
            else:matched[seed]=row
            checkpoint=seal(f'{arm}_{eeg_seed}.pt',dict(state={k:v.cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,EEG_seed=eeg_seed,updates=updates))
            row['checkpoint']=checkpoint;exports.append(checkpoint);rows.append(row)
            write(f'{arm}_{eeg_seed}_ready.json',dict(complete=True,**row,validation_not_read=True))
            del model,optimizer,lr_scheduler,output,loss,noise,timestep,z,label,c;torch.cuda.empty_cache()
    assert COUNTS['optimizer_updates']==4500 and COUNTS['train_states']==4578000
    write('training_complete_before_validation.json',dict(complete=True,counts=dict(COUNTS),all_two_velocity_models_complete=True))
    predictions={};evaluations=[]
    fit_norm=target[fit].double().norm(dim=-1).mean().float()
    with torch.inference_mode():
        for seed,eeg_seed,query in zip(SEEDS,EEG_SEEDS,query_tensors):
            prediction={'unchanged':query[val].cpu()}
            row=next(r for r in rows if r['seed']==seed)
            model=make().eval().requires_grad_(False)
            state=torch.load(row['checkpoint']['memfd_path'],weights_only=True);model.load_state_dict(state['state'],strict=True)
            c=query[val]
            for cfg in [5.,1.]:
                scheduler=DDPMScheduler(**scheduler_config);scheduler.set_timesteps(50,device='cuda')
                gen=torch.Generator(device='cuda').manual_seed(2026110200+eeg_seed)
                out=torch.randn(c.shape,device='cuda',generator=gen)
                for tick in scheduler.timesteps:
                    t=torch.full((len(c),),float(tick),device='cuda')
                    eps=model(out,t,c);COUNTS['eval_calls']+=1;COUNTS['eval_states']+=len(c)
                    if cfg!=1.:
                        uncond=model(out,t,None);COUNTS['eval_calls']+=1;COUNTS['eval_states']+=len(c)
                        eps=uncond+cfg*(eps-uncond)
                    out=scheduler.step(eps,int(tick),out,generator=gen).prev_sample
                    assert torch.isfinite(out).all()
                prediction['velocity_CFG'+str(int(cfg))]=out.cpu()
            del model,state,c,out;torch.cuda.empty_cache()
            target_val=target[val].cpu()
            for name,q in prediction.items():
                cosine=F.cosine_similarity(q.double(),target_val.double(),dim=-1)
                scores=F.normalize(q,dim=-1)@F.normalize(target_val,dim=-1).T;rank=(scores>=scores.diag()[:,None]).sum(1)-1
                e=dict(EEG_seed=eeg_seed,arm=name,images=1280,mean_cosine=float(cosine.mean()),raw_MSE=float((q.double()-target_val.double()).square().mean()),
                    mean_norm=float(q.double().norm(dim=-1).mean()),source1280mean4_top1=float((rank<1).float().mean()),image_quality_metric=False)
                evaluations.append(e);print(json.dumps(e),flush=True)
            predictions[str(eeg_seed)]=prediction
        export=seal('private_velocity_prior_predictions.pt',dict(predictions=predictions,source_indices=val,fit_mean=target[fit].double().mean(0).float().cpu(),fit_mean_norm=fit_norm.cpu()))
        exports.append(export)
    assert COUNTS==dict(train_calls=4500,train_states=4578000,backward_calls=4500,optimizer_updates=4500,eval_calls=300,eval_states=384000),COUNTS
    write('input_bindings.json',dict(query_cache_commit=identity(QUERY/'final_commit.json'),query_seed_pairs=list(zip(SEEDS,EEG_SEEDS)),
        native_feature=feature,source_split=bindings['source_split'],source_audit_commit=identity(AUDIT/'final_commit.json'),epsilon_reference_commit=identity(REFERENCE/'final_commit.json'),
        public_source=identity(path),all_models_complete_before_heldout=True,no_official_test_or_target_read=True))
    write('results.json',dict(complete=True,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-START,
        counts=COUNTS,training=rows,exports=exports,evaluations=evaluations,
        matched_model_initialization_input_order_and_updates=True,frozen_EEG_queries=True,
        full_paper_reproduction=False,image_quality_established=False,novelty_established=False,SOTA=False))
    write('final_commit.json',dict(complete=True,files={p.name:identity(p) for p in OUT.glob('*.json')},volatile_exports=exports))
    print('READY_NATIVE_VELOCITY_PRIOR',flush=True)
    # Avoid retaining GPU views or parameter state while the descriptor keeper waits.
    del target,query_tensors,query,fit_norm,scope,predictions,prediction,q,target_val,eps,uncond,t
    import gc;gc.collect();torch.cuda.empty_cache()
    until=time.monotonic()+43200
    while time.monotonic()<until:
        write('keeper_status.json',dict(pid=os.getpid(),ready=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            lease_seconds_remaining=until-time.monotonic()),True)
        time.sleep(30)
except BaseException:
    write('failure.json',dict(traceback=traceback.format_exc(),counts=COUNTS,seconds=time.monotonic()-START));raise
finally:
    if SAVED is not None:
        cpu,cuda,threads,tf32,cudnn=SAVED;torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=cudnn
    for fd in FDS:os.close(fd)
