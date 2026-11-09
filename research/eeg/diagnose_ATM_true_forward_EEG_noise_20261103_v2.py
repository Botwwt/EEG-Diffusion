"""Compare frozen score branches on identical genuine OOF forward-noised targets."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,datetime,fcntl,gc,importlib.util,json,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
import atm_true_forward_noise_contract_20261103_v1 as F

ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('GPU','CPU'),required=True);args=ap.parse_args()
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=F.setup(args.policy);q=c['s'];K=c['K'];root=Path(q['output_root']);before=F.resources(c,starting=True)
def guard():
    assert time.monotonic()-started<855
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<q['maximum_extra_CPU_bytes']
    return F.resources(c)
def progress(phase,**values):K.atomic(root/'progress.json',dict(phase=phase,seconds=time.monotonic()-started,**values))
import torch
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('frozen_native_prior_forward',c['parent']['native_prior_source']['path'])
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
if args.mode=='GPU':
    repo=Path(c['ctx']['relative']['repository_root'])
    job=K.read(repo/'audit'/(q['job_name']+'_job.json'))
    assert job['pid']==os.getpid() and job['wrapper_pid']==os.getppid() and job['status']=='RUNNING'
    lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==q['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(q['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    from diffusers import DDPMScheduler
    scheduler=DDPMScheduler();scheduler.set_timesteps(50)
    assert scheduler.timesteps.tolist()==F.TIMES and scheduler.config.clip_sample
    alpha=scheduler.alphas_cumprod.cuda()
    sys.path.insert(0,'/root/atm_official_20261016')
    assert K.ref('/root/atm_official_20261016/models/atms.py')['sha256']==c['parent']['native_encoder_source_sha256']
    from models.atms import ATMS
    eeg=torch.load(c['sub']['input']['path'],map_location='cpu',weights_only=True,mmap=True)
    target=torch.load(c['parent']['original_visual_target']['path'],map_location='cpu',weights_only=True,mmap=True)
    raw=np.load(root/'all_query_time_method_metrics.npy',mmap_mode='r+')
    risk=np.load(root/'all_query_time_alignment_energy.npy',mmap_mode='r+')
    lookup={row:i for i,row in enumerate(c['T'])}
    kept=next(b for b in c['banks'] if b['role']==q['completed_role'])
    kept_indices=[lookup[row] for row in kept['rows']]
    import hashlib
    assert hashlib.sha256(np.ascontiguousarray(raw[kept_indices]).tobytes()).hexdigest()==q['preserved_role_metric_SHA256']
    assert hashlib.sha256(np.ascontiguousarray(risk[kept_indices]).tobytes()).hexdigest()==q['preserved_role_risk_SHA256']
    assert np.isfinite(raw[kept_indices]).all() and np.isfinite(risk[kept_indices]).all()
    traces=[K.ref(root/(q['completed_role']+'_first128_five_times.npz'))]
    for bank in c['banks']:
        if bank['role']==q['completed_role']:continue
        role=bank['role'];rows=bank['rows'];assert len(rows)%10==0
        encoder=ATMS().eval().requires_grad_(False)
        encoder.load_state_dict(torch.load(c['sub']['models'][role]['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True),strict=True);encoder.cuda()
        chunks=[]
        with torch.inference_mode():
            for lo in range(0,len(rows),128):chunks.append(encoder(eeg[rows[lo:lo+128]].cuda(),torch.ones(len(rows[lo:lo+128]),device='cuda',dtype=torch.long)).float())
        conditions=torch.cat(chunks);del chunks,encoder;gc.collect();torch.cuda.empty_cache()
        teacher=target[rows].cuda().float()
        prior=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False)
        prior.load_state_dict(torch.load(bank['prior_final']['path'],map_location='cpu',weights_only=True)['model'],strict=True);prior.cuda()
        trace=dict(timesteps=[],states=[],noise=[],predictions=[],conditions=conditions[:128].cpu().numpy(),
            wrong_conditions=conditions[torch.tensor([(j//10)*10+(j+1)%10 for j in range(128)],device='cuda')].cpu().numpy(),teacher=teacher[:128].cpu().numpy(),array_rows=np.array(rows[:128]),
            context_teacher=teacher[:130].cpu().numpy(),context_array_rows=np.array(rows[:130]),alpha=scheduler.alphas_cumprod.cpu().numpy())
        for lo in range(0,len(rows),128):
            ids=list(range(lo,min(lo+128,len(rows))));ix=[lookup[rows[i]] for i in ids]
            correct=conditions[ids];wrong=conditions[[(i//10)*10+(i+1)%10 for i in ids]]
            truth=teacher[ids].double()
            near=teacher[torch.tensor([[(i//10)*10+k for k in range(10)] for i in ids],device='cuda')].double()
            unit=near/near.norm(dim=2,keepdim=True);position=torch.tensor([i%10 for i in ids],device='cuda')
            for j,t in enumerate(F.TIMES):
                noise=torch.stack([torch.randn(1024,generator=torch.Generator(device='cpu').manual_seed(F.seed(rows[i],t))) for i in ids]).cuda()
                tt=torch.full((len(ids),),t,device='cuda',dtype=torch.long)
                state=scheduler.add_noise(teacher[ids],noise,tt)
                with torch.inference_mode():
                    missing=prior(state,tt.float(),None);paired=prior(state,tt.float(),correct);mismatched=prior(state,tt.float(),wrong)
                estimates=(missing,paired,mismatched,missing+5.*(paired-missing))
                for k,prediction in enumerate(estimates):
                    estimate=(state.double()-(1-alpha[t].double()).sqrt()*prediction.double())/alpha[t].double().sqrt()
                    clipped=estimate.clamp(-1,1)
                    su=clipped/clipped.norm(dim=1,keepdim=True)
                    scores=(su[:,None]*unit).sum(2);positive=scores.gather(1,position[:,None])[:,0]
                    top=(scores.argmax(1)==position).double()
                    mask=torch.arange(10,device='cuda')[None]!=position[:,None]
                    afc=(((positive[:,None]>scores).double()+.5*(positive[:,None]==scores).double())*mask).sum(1)/9
                    values=torch.stack([(prediction.double()-noise.double()).square().mean(1),
                        (estimate-truth).square().mean(1),(clipped-truth).square().mean(1),positive,top,afc,
                        estimate.norm(dim=1),clipped.norm(dim=1)],dim=1).cpu().numpy()
                    raw[ix,j,k]=values
                residual=noise.double()-missing.double();delta=paired.double()-missing.double()
                risk[ix,j]=torch.stack([2*(residual*delta).mean(1),delta.square().mean(1)],dim=1).cpu().numpy()
                if lo==0 and t in q['fixed_CPU_replay_timesteps']:
                    trace['timesteps'].append(t);trace['states'].append(state.cpu().numpy());trace['noise'].append(noise.cpu().numpy())
                    trace['predictions'].append(torch.stack([missing,paired,mismatched],dim=1).cpu().numpy())
            raw.flush();risk.flush()
            if lo%1024==0:
                assert torch.cuda.max_memory_allocated()<q['maximum_GPU_bytes'];guard()
                progress('FROZEN_TRUE_FORWARD_ALL50_TIMES',role=role,completed_queries=lo+len(ids),role_queries=len(rows))
        path=root/(role+'_first128_five_times.npz');np.savez_compressed(path,**{k:np.asarray(v) for k,v in trace.items()});traces.append(K.ref(path))
        del prior,conditions,teacher;gc.collect();torch.cuda.empty_cache()
        K.checked(bank['prior_final']);K.checked(c['sub']['models'][role]['checkpoint'])
    assert np.isfinite(raw).all() and np.isfinite(risk).all()
    assert hashlib.sha256(np.ascontiguousarray(raw[kept_indices]).tobytes()).hexdigest()==q['preserved_role_metric_SHA256']
    assert hashlib.sha256(np.ascontiguousarray(risk[kept_indices]).tobytes()).hexdigest()==q['preserved_role_risk_SHA256']
    del raw,risk
    K.atomic(root/'GPU_completion.json',dict(status='COMPLETE_FROZEN_ALL13230x50_NATIVE_PRIOR_TRUE_FORWARD_EEG',
        policy=K.ref(args.policy),metrics=K.ref(root/'all_query_time_method_metrics.npy'),risk=K.ref(root/'all_query_time_alignment_energy.npy'),
        fixed_batch128_traces=traces,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_before=before,resources_after=guard(),
        new_training_updates=0,new_images=0,all_model_weights_unchanged=True))
    progress('COMPLETE_GPU_ALL_T_TIMES_NO_FITS')
else:
    assert not torch.cuda.is_initialized()
    gpu=K.read(root/'GPU_completion.json');assert gpu['policy']==K.ref(args.policy)
    raw=np.load(K.checked(gpu['metrics']),mmap_mode='r');risk=np.load(K.checked(gpu['risk']),mmap_mode='r')
    cm,qm,means,time_means,comparisons=F.summarize(raw,c['T'])
    alpha=np.load(K.checked(gpu['fixed_batch128_traces'][0]))['alpha'].astype(np.float64)
    factor=(1-alpha[F.TIMES])/alpha[F.TIMES]
    clean_identity=float(np.max(abs(raw[:,:,:,1]-raw[:,:,:,0]*factor[None,:,None])/(1+abs(raw[:,:,:,1]))))
    correct_gain=raw[:,:,0,0]-raw[:,:,1,0];native_gain=raw[:,:,0,0]-raw[:,:,3,0]
    risk_identity=max(float(abs(correct_gain-(risk[:,:,0]-risk[:,:,1])).max()),
                      float(abs(native_gain-(5*risk[:,:,0]-25*risk[:,:,1])).max()))
    # add_noise forms the network input in FP32. Its rounding is amplified by
    # sqrt((1-alpha)/alpha); do not call that input-rounding term a score error.
    assert clean_identity<5e-5 and risk_identity<2e-6
    replay_error=0.
    for bank,tr in zip(c['banks'],gpu['fixed_batch128_traces']):
        z=np.load(K.checked(tr));prior=native.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval().requires_grad_(False)
        prior.load_state_dict(torch.load(bank['prior_final']['path'],map_location='cpu',weights_only=True)['model'],strict=True)
        for j,t in enumerate(z['timesteps']):
            conditions=(None,torch.from_numpy(z['conditions']),torch.from_numpy(z['wrong_conditions']))
            state=torch.from_numpy(z['states'][j]);tt=torch.full((128,),float(t))
            with torch.inference_mode():
                for k,cond in enumerate(conditions):
                    pred=prior(state,tt,cond).numpy().astype(np.float64);saved=z['predictions'][j,:,k].astype(np.float64)
                    replay_error=max(replay_error,float(np.max(abs(pred-saved)/(1+abs(saved)))))
            for i,row in enumerate(z['array_rows']):
                noise=torch.randn(1024,generator=torch.Generator(device='cpu').manual_seed(F.seed(int(row),int(t)))).numpy()
                assert np.array_equal(noise,z['noise'][j,i])
        del prior
    assert replay_error<3e-5
    groups=[[i for i,row in enumerate(c['T']) if row//10==concept] for concept in dict.fromkeys(row//10 for row in c['T'])]
    risk_cm=np.stack([np.asarray(risk[g],np.float64).mean(0) for g in groups])
    np.savez_compressed(root/'concept_and_query_summaries.npz',concept_metrics=cm,query_time_average=qm,
        concept_alignment_energy=risk_cm,array_rows=np.array(c['T']),alpha=alpha)
    readout=dict(status='COMPLETE_FULL_T_TRUE_FORWARD_NATIVE_EEG_SCORE_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        policy=K.ref(args.policy),methods=list(F.METHODS),metrics=list(F.METRICS),timesteps=F.TIMES,
        global_uniform_time_means={method:{metric:float(means[k,m]) for m,metric in enumerate(F.METRICS)} for k,method in enumerate(F.METHODS)},
        all50_time_means=time_means.tolist(),fixed_comparisons=comparisons,
        summaries=K.ref(root/'concept_and_query_summaries.npz'),raw_clean_noise_error_identity_max_scaled=clean_identity,
        raw_clean_identity_includes_FP32_forward_roundoff=True,
        all661500_query_time_risk_identity_max_error=risk_identity,real_CPU_vs_GPU_3840_frozen_predictions_max_scaled=replay_error,
        uniform_time_average_not_a_reverse_trajectory=True,teacher_forward_uses_true_T_images_and_cannot_be_normal_inference=True,
        classical_quadratic_risk_identity_not_new_theorem=True,no_CFG_fitting_or_scan=True,
        new_training_updates=0,new_images=0,no_D_or_V_effects=True,no_automatic_new_fit_or_sampling=True,
        seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=guard())
    assert not torch.cuda.is_initialized();K.atomic(root/'readout.json',readout);progress('COMPLETE_CPU_ALL_T_SCORE_DIAGNOSTIC')
