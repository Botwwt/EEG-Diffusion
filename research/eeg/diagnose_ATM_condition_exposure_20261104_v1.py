"""Same T identities: frozen encoder exposure and frozen OOF prior response."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,datetime,fcntl,gc,importlib.util,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
import atm_condition_exposure_contract_20261104_v1 as X

p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--mode',choices=('GPU','CPU'),required=True);a=p.parse_args()
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=X.setup(a.policy);s,F,b,K=(c[x] for x in ('s','F','base','K'));root=Path(s['output_root']);T=b['T'];lookup={r:i for i,r in enumerate(T)}
before=X.resources(c,starting=True)
def guard():
    assert time.monotonic()-started<855
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<s['maximum_extra_CPU_bytes']
    return X.resources(c)
def write(name,v):K.atomic(root/name,v)
def progress(phase,**kw):write('progress.json',dict(phase=phase,seconds=time.monotonic()-started,**kw))
import torch
torch.set_num_threads(2)
teacher=torch.load(b['parent']['original_visual_target']['path'],map_location='cpu',weights_only=True,mmap=True)
original_root=Path(s['true_forward_completion']['path']).parent
original_done=K.read(original_root/'GPU_completion.json');K.checked(original_done['metrics'])
baseline=np.load(original_done['metrics']['path'],mmap_mode='r',allow_pickle=False)
assert baseline.shape==(13230,50,4,8)
if a.mode=='GPU':
    repo=Path(b['ctx']['relative']['repository_root']);job=K.read(repo/'audit'/(s['job_name']+'_job.json'))
    assert job['pid']==os.getpid() and job['wrapper_pid']==os.getppid() and job['status']=='RUNNING'
    lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0,'/root/atm_official_20261016')
    from models.atms import ATMS
    assert K.ref('/root/atm_official_20261016/models/atms.py')['sha256']==b['parent']['native_encoder_source_sha256']
    raw=torch.load(b['sub']['input']['path'],map_location='cpu',weights_only=True,mmap=True)
    conditions=np.lib.format.open_memmap(root/'all_T_seen_unseen_conditions.npy',mode='w+',dtype=np.float32,shape=(13230,2,1024))
    quality=np.lib.format.open_memmap(root/'all_T_seen_unseen_quality.npy',mode='w+',dtype=np.float64,shape=(13230,2,6))
    conditions[:]=np.nan;quality[:]=np.nan
    role_seen=np.empty(13230,dtype=np.int8);role_unseen=np.empty(13230,dtype=np.int8);checks=[]
    for ri,bank in enumerate(b['banks']):
        role=bank['role'];permissions=K.read(bank['permissions']['path']);encoder=ATMS().eval().requires_grad_(False)
        encoder.load_state_dict(torch.load(b['sub']['models'][role]['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True),strict=True);encoder.cuda()
        for view,rows in ((0,permissions['training_rows']),(1,permissions['export_rows'])):
            assert len(rows)%10==0 and set(rows)<=set(T)
            for lo in range(0,len(rows),128):
                rr=rows[lo:lo+128];ix=[lookup[r] for r in rr]
                with torch.inference_mode():z=encoder(raw[rr].cuda(),torch.ones(len(rr),device='cuda',dtype=torch.long)).float()
                conditions[ix,view]=z.cpu().numpy()
                (role_seen if view==0 else role_unseen)[ix]=ri
                if view==1 and lo==0:
                    old=np.load(original_root/(role+'_first128_five_times.npz'))
                    assert old['array_rows'].tolist()==rr
                    err=float(np.max(abs(conditions[ix,view]-old['conditions'])));assert err<2e-5
                    checks.append(dict(role=role,full128_unseen_condition_vs_previous_trace_maxabs=err))
                if lo%1024==0:guard()
            conditions.flush();progress('FROZEN_ENCODER_ALL_T',role=role,view=X.VIEWS[view],queries=len(rows))
        del encoder;gc.collect();torch.cuda.empty_cache()
    assert np.isfinite(conditions).all() and np.all(role_seen!=role_unseen)
    # Conditions from the same image, different supervised exposure/encoder role.
    tt=teacher[T].cuda().double().reshape(1323,10,1024);unit=tt/tt.norm(dim=2,keepdim=True)
    for lo in range(0,1323,32):
        hi=min(lo+32,1323);truth=tt[lo:hi];u=unit[lo:hi];pos=torch.arange(10,device='cuda')[None,:,None]
        mask=~torch.eye(10,device='cuda',dtype=torch.bool)[None]
        for view in range(2):
            z=torch.tensor(np.array(conditions[lo*10:hi*10,view]),device='cuda').double().reshape(hi-lo,10,1024)
            su=z/z.norm(dim=2,keepdim=True);scores=su@u.transpose(1,2);positive=scores.diagonal(dim1=1,dim2=2)
            afc=(((positive[:,:,None]>scores).double()+.5*(positive[:,:,None]==scores).double())*mask).sum(2)/9
            values=torch.stack([(z-truth).square().mean(2),positive,(scores.argmax(2)==torch.arange(10,device='cuda')[None]).double(),afc,
                positive-(scores*mask).sum(2)/9,z.norm(dim=2)],dim=2)
            quality[lo*10:hi*10,view]=values.cpu().numpy().reshape(-1,6)
    quality.flush();del tt,unit,z,su,scores,positive,values;gc.collect();torch.cuda.empty_cache()
    spec=importlib.util.spec_from_file_location('frozen_exposure_prior',b['parent']['native_prior_source']['path'])
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    from diffusers import DDPMScheduler
    scheduler=DDPMScheduler();scheduler.set_timesteps(50);assert scheduler.timesteps.tolist()==F.TIMES
    alphas=scheduler.alphas_cumprod.clone().cuda()
    scores_out=np.lib.format.open_memmap(root/'all_T_seen_encoder_frozen_prior_metrics.npy',mode='w+',dtype=np.float64,shape=(13230,50,8));scores_out[:]=np.nan
    for bank in b['banks']:
        role=bank['role'];rows=bank['rows'];prior=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False)
        prior.load_state_dict(torch.load(bank['prior_final']['path'],map_location='cpu',weights_only=True)['model'],strict=True);prior.cuda()
        traces=dict(array_rows=np.array(rows[:128]),timesteps=np.array(s['CPU_replay_timesteps']),teacher=teacher[rows[:128]].numpy(),
            conditions=conditions[[lookup[r] for r in rows[:128]],0].copy(),context_teacher=teacher[rows[:130]].numpy(),context_rows=np.array(rows[:130]),
            alpha=scheduler.alphas_cumprod.cpu().numpy(),states=[],noise=[],predictions=[])
        for lo in range(0,len(rows),128):
            rr=rows[lo:lo+128];ix=[lookup[r] for r in rr];cond=torch.tensor(np.array(conditions[ix,0]),device='cuda');truth=teacher[rr].cuda().double()
            near=teacher[[[(r//10)*10+k for k in range(10)] for r in rr]].cuda().double();near=near/near.norm(dim=2,keepdim=True)
            position=torch.tensor([r%10 for r in rr],device='cuda');mask=torch.arange(10,device='cuda')[None]!=position[:,None]
            for j,t in enumerate(F.TIMES):
                noise=torch.stack([torch.randn(1024,generator=torch.Generator(device='cpu').manual_seed(F.seed(r,t))) for r in rr]).cuda()
                tt=torch.full((len(rr),),t,device='cuda',dtype=torch.long);state=scheduler.add_noise(teacher[rr].cuda(),noise,tt)
                with torch.inference_mode():prediction=prior(state,tt.float(),cond)
                clean=(state.double()-(1-alphas[t].double()).sqrt()*prediction.double())/alphas[t].double().sqrt();clipped=clean.clamp(-1,1)
                su=clipped/clipped.norm(dim=1,keepdim=True);sim=(su[:,None]*near).sum(2);positive=sim.gather(1,position[:,None])[:,0]
                afc=(((positive[:,None]>sim).double()+.5*(positive[:,None]==sim).double())*mask).sum(1)/9
                values=torch.stack([(prediction.double()-noise.double()).square().mean(1),(clean-truth).square().mean(1),
                    (clipped-truth).square().mean(1),positive,(sim.argmax(1)==position).double(),afc,clean.norm(dim=1),clipped.norm(dim=1)],dim=1)
                scores_out[ix,j]=values.cpu().numpy()
                if lo==0 and t in s['CPU_replay_timesteps']:
                    traces['states'].append(state.cpu().numpy());traces['noise'].append(noise.cpu().numpy());traces['predictions'].append(prediction.cpu().numpy())
            scores_out.flush()
            if lo%1024==0:guard();progress('ALL50_FROZEN_PRIOR_SEEN_ENCODER_DIAGNOSTIC',role=role,queries=lo+len(rr),role_queries=len(rows))
        np.savez_compressed(root/(role+'_first128_trace.npz'),**{k:np.asarray(v) for k,v in traces.items()})
        del prior;gc.collect();torch.cuda.empty_cache()
    assert np.isfinite(scores_out).all() and np.isfinite(quality).all()
    np.savez_compressed(root/'roles_and_rows.npz',array_rows=np.array(T),seen_encoder_role=role_seen,unseen_encoder_role=role_unseen)
    write('GPU_completion.json',dict(status='COMPLETE_FROZEN_ALL_T_EXPOSURE_AND_PRIOR_RESPONSE',policy=K.ref(a.policy),
        conditions=K.ref(root/'all_T_seen_unseen_conditions.npy'),quality=K.ref(root/'all_T_seen_unseen_quality.npy'),
        new_score=K.ref(root/'all_T_seen_encoder_frozen_prior_metrics.npy'),baseline_score=original_done['metrics'],checks=checks,
        seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
        resources_after=guard(),new_training_updates=0,new_images=0,encoder_exposure_and_encoder_identity_confounded=True,not_inference=True))
else:
    assert not torch.cuda.is_initialized()
    done=K.read(root/'GPU_completion.json');conditions=np.load(K.checked(done['conditions']),mmap_mode='r');quality=np.load(K.checked(done['quality']),mmap_mode='r');new_score=np.load(K.checked(done['new_score']),mmap_mode='r')
    qc,sc,bc,comparisons=X.summaries(quality,new_score,baseline,T)
    # Full quality metrics recomputed on CPU using explicit float64 dot products.
    maximum=0.;target=teacher.numpy()
    for j in range(1323):
        rows=T[j*10:(j+1)*10];truth=target[rows].astype(np.float64);unit=truth/np.linalg.norm(truth,axis=1,keepdims=True)
        mask=~np.eye(10,dtype=bool)
        for view in range(2):
            z=np.asarray(conditions[j*10:(j+1)*10,view],np.float64);su=z/np.linalg.norm(z,axis=1,keepdims=True);sim=su@unit.T;positive=np.diag(sim)
            afc=(((positive[:,None]>sim)+.5*(positive[:,None]==sim))*mask).sum(1)/9
            values=np.stack([((z-truth)**2).mean(1),positive,(sim.argmax(1)==np.arange(10)).astype(float),afc,
                positive-(sim*mask).sum(1)/9,np.linalg.norm(z,axis=1)],axis=1)
            maximum=max(maximum,float(np.max(abs(values-quality[j*10:(j+1)*10,view]))))
    assert maximum<2e-9
    replays=[];trace_metrics=[]
    spec=importlib.util.spec_from_file_location('frozen_CPU_exposure_prior',b['parent']['native_prior_source']['path']);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    for bank in b['banks']:
        tr=np.load(root/(bank['role']+'_first128_trace.npz'));prior=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False)
        prior.load_state_dict(torch.load(bank['prior_final']['path'],map_location='cpu',weights_only=True)['model'],strict=True)
        role_max=0.
        for j,t in enumerate(tr['timesteps']):
            regenerated=torch.stack([torch.randn(1024,generator=torch.Generator(device='cpu').manual_seed(F.seed(int(r),int(t)))) for r in tr['array_rows']]).numpy()
            assert np.array_equal(regenerated,tr['noise'][j])
            with torch.inference_mode():pred=prior(torch.from_numpy(tr['states'][j]),torch.full((128,),float(t)),torch.from_numpy(tr['conditions'])).numpy()
            err=float(np.max(abs(pred-tr['predictions'][j])/(1+abs(tr['predictions'][j]))));assert err<3e-5;role_max=max(role_max,err)
        replays.append(dict(role=bank['role'],CPU_sample_forwards=640,maximum_scaled_error=role_max,all640_actual_noise_vectors_exact=True));del prior
    np.savez_compressed(root/'all_concept_reductions.npz',quality=qc,seen_score=sc,baseline_score=bc)
    np.savez_compressed(root/'all_query_reductions.npz',quality=np.asarray(quality),seen_score=np.asarray(new_score).mean(1),baseline_score=np.asarray(baseline).mean(1))
    readout=dict(status='COMPLETE_FULL_T_FROZEN_CONDITION_EXPOSURE_ROLE_DIAGNOSTIC',quality_metrics=list(X.QUALITY),score_metrics=list(F.METRICS),
        quality_means=quality.mean(0).tolist(),seen_encoder_score_means=np.asarray(new_score).mean((0,1)).tolist(),baseline_score_means=np.asarray(baseline).mean((0,1)).tolist(),
        comparisons=comparisons,quality_CPU_float64_maximum_error=maximum,independent_CPU_prior_forward_replay=replays,
        all_T_prior_remains_genuinely_unseen_concept=True,swapped_encoder_saw_query_concepts=True,
        exposure_and_encoder_identity_confounded=True,not_normal_inference=True,no_D_V_effects=True,no_new_fit_or_images=True,
        prior_training_used_matched_encoder_seen_conditions=True,prior_input_quality_is_not_isolated_causal_effect=True,
        seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=guard())
    write('readout.json',readout);progress('COMPLETE_ALL_T_EXPOSURE_ROLE_DIAGNOSTIC')
