"""Frozen postfit replay of ALL 830 registered training targets; no optimizer."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import datetime,fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
plan=json.loads(Path(sys.argv[1]).read_bytes());sys.path.insert(0,str(Path(plan['parent_policy']).parent))
import atm_response_repair_contract_20261105_v196 as C
c=C.setup(plan['parent_policy']);q,K,tc=c['q'],c['K'],c['train'];out=Path(plan['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==plan['worker'];before=C.resources(c,True)
job=K.read(c['repo']/'audit'/(plan['job_name']+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==tc['physical']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path[:0]=tc['physical']['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from safetensors import safe_open
import atm_response_contrast_repair_core_20261105_v195 as H
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
def guard():
    assert time.monotonic()-started<850 and time.time()<plan['absolute_deadline_epoch']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30 and torch.cuda.max_memory_allocated()<20<<30
    assert K.allocated(out)<plan['maximum_bytes'];used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
    fs=os.statvfs(out);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def margin(response,visual):
    b,dim=visual.shape;assert b%10==0
    dot=response@visual.T/dim;vv=visual.square().mean(1);dist=response.square().mean(1)[:,None]+vv[None,:]-2*dot
    own=dist.diag();groups=torch.arange(b,device=dist.device).reshape(-1,10)
    other=dist[torch.arange(b,device=dist.device)[:,None],groups.repeat_interleave(10,0)].sum(1)-own
    return other/9-own
try:
    states={k:torch.load(K.checked(v),map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()}
    cn={n:states['NN'][n] if n.startswith('proj_eeg.') else v for n,v in states['CC'].items()}
    sources=[q['models']['NN'],q['models']['CC']]
    modelstates=[states['NN'],cn]
    for arm in C.ARMS[2:]:
        pin=K.read(c['fit']/arm/'fit_10_completion.json')['checkpoint'];sources.append(pin)
        modelstates.append(torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True)['model'])
    xall=torch.load(K.checked(tc['physical']['EEG']),map_location='cpu',weights_only=True,mmap=True)
    vall=torch.load(K.checked(tc['physical']['visual_target']),map_location='cpu',weights_only=True,mmap=True)
    T=np.asarray(tc['T']);assert len(T)==13230;conditions=[];cpu_checks=[];fixed=np.array([0,1,9,10,159,160,1023,13229])
    with torch.inference_mode():
        for arm,state in zip(C.ARMS,modelstates):
            guard();net=ATMS().cuda().eval().requires_grad_(False);net.load_state_dict(state,strict=True)
            v=torch.empty((len(T),1024),device='cuda')
            for lo in range(0,len(T),160):
                ids=T[lo:lo+160];v[lo:lo+len(ids)]=net(xall[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long))
            cpu=ATMS().double().eval().requires_grad_(False);cpu.load_state_dict(state,strict=True)
            actual=cpu(xall[T[fixed]].double(),torch.ones(len(fixed),dtype=torch.long)).numpy();saved=v[fixed].cpu().numpy()
            np.testing.assert_allclose(actual,saved,atol=3e-5,rtol=3e-5);cpu_checks.append(dict(arm=arm,queries=len(fixed),max_abs=float(abs(actual-saved).max())))
            conditions.append(v);del cpu,net
        visual_all=vall[T].cuda()
        Qdone=K.read(Path(q['project_root'])/'Qbank/completion.json');Qbank=np.load(K.checked(Qdone['bank']));w={};kv=[]
        with safe_open(K.checked(c['cv']['bq']['h94']),framework='pt',device='cpu') as f:
            for key,name in [('P_weight','image_proj.proj.weight'),('P_bias','image_proj.proj.bias'),('P_gamma','image_proj.norm.weight'),('P_beta','image_proj.norm.bias')]:w[key]=f.get_tensor(name).half().float().cuda()
            for lay in Qdone['layouts']:
                ix=lay['ip_index'];kv.append({k:f.get_tensor(f'ip_adapter.{ix}.{n}.weight').half().float().cuda() for k,n in [('K','to_k_ip'),('V','to_v_ip')]})
        rng=np.random.default_rng(1729);schedule=np.concatenate([rng.permutation(280) for _ in range(3)])[:830]
        keys=['departure_energy','CN_displacement_energy','ideal_projection_energy','CN_unwanted_energy','main_target_error','contrast_subspace_error','unwanted_energy','retained_projection_inner','weighted_contrast_error','CN_weighted_contrast_energy','own_training_target_error','within_margin','NN_within_margin','CN_within_margin']
        sums=np.zeros((13230,2,len(keys)),np.float64);coverage=np.zeros(13230,np.int16);batch=[];proofs=[];update=0
        for epoch in range(10):
            order=np.random.default_rng(1739+epoch).permutation(1323);epoch_cov=np.zeros(13230,np.uint8)
            for lo in range(0,1323,16):
                guard();pos=(order[lo:lo+16,None]*10+np.arange(10)).reshape(-1);n=len(pos);unit=int(schedule[update]);ti,li=divmod(unit,70)
                bank=Qbank[f'l{li:02d}'][:,ti];heads,dim=bank.shape[1],bank.shape[-1]
                Q=torch.tensor(bank.transpose(1,0,2,3).reshape(1,heads,8,dim).copy(),device='cuda').float();weights={**w,**kv[li]}
                outputs=[H.response(v[pos],Q,weights).double() for v in conditions];visual=H.response(visual_all[pos],Q,weights).double()
                old,updated=outputs[:2];delta=updated-old;D=visual-visual.mean(0,keepdim=True);scale=D.square().mean();G=D@D.T;lam,U=torch.linalg.eigh(G);active=lam>lam.max()*1e-12
                pinv=(U[:,active]/lam[active])@U[:,active].T
                def project(v):return ((v@D.T)@pinv)@D
                p=project(delta);CNorth=delta-p;norm=lambda a:a.square().mean(1)/scale
                rr=torch.randn(delta.shape,generator=torch.Generator(device='cuda').manual_seed(21729+update),device='cuda').double()
                dd=delta.square().sum(1);pp=p.square().sum(1);beta=(pp/dd.clamp_min(1e-300)).clamp(0,1);rr-=((rr*delta).sum(1)/dd.clamp_min(1e-300))[:,None]*delta;rr/=rr.norm(dim=1,keepdim=True)
                ctrl=beta[:,None]*delta+(beta*(1-beta)*dd).clamp_min(0).sqrt()[:,None]*rr
                assert float(((p-delta)@D.T).norm()/(delta@D.T).norm().clamp_min(1e-300))<1e-5
                mm=[margin(v,visual)/scale for v in outputs];result=[]
                for j,student in enumerate(outputs[2:]):
                    r=student-old;pr=project(r);orth=r-pr
                    weighted=lambda z:(z@D.T).square().mean(1)/(D.shape[1]*scale)**2
                    metrics=torch.stack([norm(r),norm(delta),norm(p),norm(CNorth),norm(r-p),norm(pr-p),norm(orth),(pr*p).mean(1)/scale,weighted(r-delta),weighted(delta),norm(r-(p if j==0 else ctrl)),mm[j+2],mm[0],mm[1]],dim=1)
                    assert torch.isfinite(metrics).all();np.testing.assert_allclose(metrics[:,4].cpu(),(metrics[:,5]+metrics[:,6]).cpu(),rtol=1e-7,atol=1e-9)
                    result.append(metrics.cpu().numpy())
                values=np.stack(result,1);sums[pos]+=values;coverage[pos]+=1;epoch_cov[pos]+=1
                batch.append(dict(update=update,epoch=epoch+1,unit=unit,queries=n,teacher_rank=int(active.sum()),scale=float(scale),means=values.mean(0).tolist()))
                if update in [0,829]:
                    # Independent CPU64 actual response and NumPy-SVD, not the GPU eig implementation.
                    cw={k:v.cpu().double() for k,v in weights.items()};cq=Q.cpu().double();cv=H.response(visual_all[pos].cpu().double(),cq,cw).numpy()
                    co=[H.response(v[pos].cpu().double(),cq,cw).numpy() for v in conditions]
                    for cpuo,gpuo in zip(co,outputs):np.testing.assert_allclose(cpuo,gpuo.cpu().numpy(),atol=3e-5,rtol=3e-5)
                    np.testing.assert_allclose(cv,visual.cpu().numpy(),atol=3e-5,rtol=3e-5)
                    cd=cv-cv.mean(0);_,sing,vt=np.linalg.svd(cd,full_matrices=False);basis=vt[sing>sing.max()*1e-6];cp=(co[1]-co[0])@basis.T@basis
                    pe=float(np.linalg.norm(cp-p.cpu().numpy())/max(np.linalg.norm(cp),1e-20));assert pe<1e-4
                    errors=[]
                    for j in range(2):
                        cr=co[j+2]-co[0];crp=cr@basis.T@basis;cs=np.mean(cd**2)
                        vals=np.stack([np.mean((cr-cp)**2,1)/cs,np.mean((crp-cp)**2,1)/cs,np.mean((cr-crp)**2,1)/cs],1)
                        e=float(np.max(np.abs(vals-values[:,j,4:7])));assert e<1e-4;errors.append(e)
                    proofs.append(dict(update=update,queries=n,unit=unit,projection_relative_error=pe,CPU64_metric_abs_errors=errors,CPU_SVD_rank=len(basis)))
                update+=1
            assert np.all(epoch_cov==1);K.write(out/'progress.json',dict(epochs=epoch+1,targets=update,query_exposures=int(coverage.sum()),new_updates=0))
        assert update==830 and np.all(coverage==10);assert np.array_equal(np.sort(np.unique(schedule)),np.arange(280))
        values=sums/coverage[:,None,None];assert np.isfinite(values).all()
        np.savez_compressed(out/'per_stimulus.npz',metrics=values,coverage=coverage,T_rows=T,batch_means=np.array([b['means'] for b in batch]),batch_queries=np.array([b['queries'] for b in batch]),batch_units=schedule)
        K.write(out/'batches.json',batch);means=values.mean(0);summary={}
        for j,arm in enumerate(C.ARMS[2:]):
            s={k:float(means[j,i]) for i,k in enumerate(keys)}
            s.update(target_error_fraction_of_initial=s['main_target_error']/s['CN_unwanted_energy'],unwanted_fraction_of_CN=s['unwanted_energy']/s['CN_unwanted_energy'],projection_alignment=s['retained_projection_inner']/s['ideal_projection_energy'],weighted_contrast_relative_RMSE=(s['weighted_contrast_error']/s['CN_weighted_contrast_energy'])**.5)
            summary[arm]=s
        guard();after=C.resources(c)
        for pin in sources:K.checked(pin)
        K.write(out/'completion.json',dict(status='COMPLETE_FROZEN_ALL830_TARGETS_132300_EXPOSURES_280_UNITS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=K.ref(sys.argv[1]),models=sources,arms=C.ARMS[2:],metric_names=keys,summary=summary,query_count=13230,concepts=1323,exposures=132300,units=280,CPU_conditions=cpu_checks,CPU64_SVD_proofs=proofs,arrays=K.ref(out/'per_stimulus.npz'),batches=K.ref(out/'batches.json'),new_updates=0,new_images=0,normal_inference_unchanged=True,seen_training_diagnostic_not_generalization=True,seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,peak_GPU_bytes=torch.cuda.max_memory_allocated(),resources=after))
except BaseException as exc:
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
