"""All-T fixed model inference and objective gradients. No optimizer, no selection."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,time,resource,subprocess,fcntl,traceback
from pathlib import Path
import numpy as np
import atm_repeat_margin_contract_20261105_v1 as C
import atm_repeat_margin_core_20261105_v1 as F
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(sys.argv[1]);q,K,out,pq=c['q'],c['K'],c['out'],c['physical'];assert K.ref(__file__)==q['sources']['GPU']
before=C.resources(c,True);repo=Path(pq['GPU_wrapper']['path']).parent
job=K.read(repo/'audit'/(q['job_prefix']+'_measure_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
import atm_full_repeat_gradient_core_20261105_v1 as G
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
def guard():
    assert time.monotonic()-started<850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
    assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
    used=K.allocated(q['project_root']);assert used<=q['maximum_output_bytes']
    st=os.statvfs(q['project_root']);assert st.f_bavail*st.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
try:
    obj=np.load(K.checked(pq['physical_source']),allow_pickle=True);obj=obj if isinstance(obj,dict) else obj.item()
    raw=obj['preprocessed_eeg_data'];assert raw.shape==(16540,4,63,250) and raw.dtype==np.float32
    mean4=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
    target=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy()
    T=c['T'];order=np.random.default_rng(1739).permutation(1323);positions=(order[:,None]*10+np.arange(10)).reshape(-1)
    numerator=np.zeros((4,13230,3,3,2,4));denominator=np.zeros((4,13230,3,3));coverage=np.zeros((4,13230,5),np.uint8)
    grams=np.zeros((4,83,2,2));losses=np.zeros((4,83,2));globalgrams=np.zeros((4,2,2));fixture={};scales=[]
    for ai,arm in enumerate(F.STATES):
        state=torch.load(K.checked(q['checkpoints'][arm]),map_location='cpu',weights_only=True,mmap=True);base=state['model']
        net=ATMS().cuda().eval();net.load_state_dict(base,strict=True);meta=G.metadata(net)
        total=torch.zeros((2,meta['dimension']),device='cuda',dtype=torch.float64)
        scales.append(float(net.logit_scale.exp().clamp_max(100)))
        for bi,lo in enumerate(range(0,13230,160)):
            guard();pos=positions[lo:lo+160];rows=T[pos];block=raw[rows].astype(np.float32)
            assert np.array_equal(block.mean(1),mean4[rows].numpy());Y=torch.nn.functional.normalize(torch.tensor(target[rows],device='cuda'),dim=1)
            ss=[];pred=[]
            with torch.no_grad():
                for r in range(4):
                    z=net(torch.tensor(block[:,r],device='cuda'),torch.ones(len(rows),device='cuda',dtype=torch.long))
                    s=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(z,dim=1)@Y.T
                    ss.append(s.cpu().numpy());pred.append(z[:8].cpu().numpy())
            z=net(torch.tensor(block.mean(1),device='cuda'),torch.ones(len(rows),device='cuda',dtype=torch.long))
            s=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(z,dim=1)@Y.T
            ss.append(s.detach().cpu().numpy());pred.append(z[:8].detach().cpu().numpy());ss=np.stack(ss)
            ll=F.torch_losses(s);grads=[];active=[]
            for ki,loss in enumerate(ll):
                if bi==0:
                    dl=torch.autograd.grad(loss,s,retain_graph=True)[0].detach().cpu().numpy();nl,ng,_=F.objective(ss[4],F.METHODS[ki])
                    assert abs(float(loss)-nl)<3e-5 and np.max(abs(dl-ng))<3e-6
                g=torch.autograd.grad(loss,meta['parameters'],retain_graph=(ki==0),allow_unused=True)
                active.append(np.array([v is not None for v in g]));flat=torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1) for p,v in zip(meta['parameters'],g)]).double()
                assert torch.isfinite(flat).all();grads.append(flat);losses[ai,bi,ki]=float(loss.detach())
            assert np.array_equal(active[0],active[1]);gg=torch.stack(grads);grams[ai,bi]=(gg@gg.T).cpu().numpy();total.add_(gg,alpha=len(rows)/13230)
            numerator[ai,pos],denominator[ai,pos]=F.measure(ss);coverage[ai,pos]+=1
            if bi==0:
                fixture[arm+'_logits']=ss;fixture[arm+'_first8_predictions']=np.stack(pred);fixture[arm+'_active']=active[0]
                fixture[arm+'_gradient_samples']=gg[:,torch.tensor(meta['indices'],device='cuda')].cpu().numpy()
                fixture[arm+'_tensor_squared_norms']=np.array([[float(v[a:b].square().sum()) for a,b in meta['slices']] for v in gg])
            del z,s,ll,g,gg,grads,flat
            K.write(out/'progress.json',dict(state=arm,completed=min(lo+160,13230),total=13230,seconds=time.monotonic()-started,new_updates=0))
        globalgrams[ai]=(total@total.T).cpu().numpy()
        for name,value in net.state_dict().items():assert torch.equal(value.cpu(),base[name])
        del net,total,state,base;torch.cuda.empty_cache()
    assert np.all(coverage==1)
    np.savez_compressed(out/'full_T_statistics.npz',T=T,positions=positions,concept_order=order,numerators=numerator,denominators=denominator,
        coverage=coverage,gradient_grams=grams,global_gradient_grams=globalgrams,losses=losses,effective_scales=scales)
    np.savez_compressed(out/'fixed160_fixture.npz',rows=T[positions[:160]],sampled_parameter_indices=meta['indices'],**fixture)
    K.write(out/'parameter_metadata.json',dict(names=meta['names'],slices=meta['slices'],dimension=meta['dimension']))
    guard();C.resources(c)
    K.write(out/'scientific_completion.json',dict(status='COMPLETE_FULL_T_FROZEN_REPEAT_MARGIN',policy=K.ref(sys.argv[1]),
        statistics=K.ref(out/'full_T_statistics.npz'),fixture=K.ref(out/'fixed160_fixture.npz'),parameters=K.ref(out/'parameter_metadata.json'),
        full_T_forwards=264600,parameter_backward_calls=664,new_updates=0,new_models=0,new_images=0,
        all_models_eval_frozen_buffers=True,training_data_NOT_OOF=True,repeat_agreement_NOT_independent_signal_estimate=True,
        seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
except BaseException as exc:
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
