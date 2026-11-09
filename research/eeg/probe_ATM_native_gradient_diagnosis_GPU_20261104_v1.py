"""Frozen four-state/all-T gradient diagnosis, no optimizer or generator calls."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,gc,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);a=ap.parse_args();q=json.loads(Path(a.policy).read_bytes())
sys.path.insert(0,str(Path(q['study_policy']['path']).parent));import atm_conditional_stimulus_contract_20261104_v1 as C
c=C.setup(q['study_policy']['path']);K=c['K'];root=Path(q['project_root']);out=Path(q['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['GPU']
done=K.read(K.checked(K.read(Path(q['code_root'])/'execution_admission.json')['whole_dependency_completion']))
assert done['status']=='COMPLETE_NATIVE_EEG_CONDITIONAL_STIMULUS_TARGET_DIAGNOSTIC'
launch=K.read(K.checked(q['dependency_launch']));assert not K.process(launch['PID'],launch['start_ticks'])['live']
for pin in q['sources'].values():K.checked(pin)
job=K.read(c['repo']/'audit'/(q['job_name']+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['parent']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
def resources(starting=False):
    r=C.resources(c,starting=starting);used=c['cv']['ctx']['base']['ctx']['C'].allocated(root);assert used<=q['maximum_output_bytes']
    assert r['free_bytes']>=r['required_including_this_project']+q['maximum_output_bytes']-used
    return dict(inherited=r,own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,required_including_this_project=r['required_including_this_project']+q['maximum_output_bytes']-used,free_bytes=r['free_bytes'])
before=resources(starting=True)
import torch
import torch.nn.functional as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(c['parent']['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
sys.path.insert(0,'/root/atm_official_20261016');from models.atms import ATMS
from atm_conditional_stimulus_objective_20261104_v1 import torch_loss
from atm_native_gradient_diagnosis_math_20261104_v1 import group,grouped
raw=torch.load(K.checked(c['q']['EEG']),map_location='cpu',weights_only=True,mmap=True);target=torch.load(K.checked(c['q']['visual_target']),map_location='cpu',weights_only=True,mmap=True)
T=c['T'];order=np.random.default_rng(1729).permutation(1323);batches=[T[(order[lo:lo+16,None]*10+np.arange(10)).reshape(-1)] for lo in range(0,1323,16)]
assert len(batches)==83 and sorted(np.concatenate(batches).tolist())==sorted(T.tolist())
K.write(out/'batches.json',dict(concept_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),array_rows=[r.tolist() for r in batches],all13230_T_once=True))
def guard():
    assert time.monotonic()-started<850 and torch.cuda.max_memory_allocated()<c['parent']['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
    resources()
def digest(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():h.update(name.encode());h.update(str(v.dtype).encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()
def flatten(values,parameters):
    return np.concatenate([(np.zeros(p.numel(),np.float32) if v is None else v.detach().float().reshape(-1).cpu().numpy()) for v,p in zip(values,parameters)])
summaries={};checks={};journal=out/'all332_batch_gradient_stats.jsonl';assert not journal.exists()
try:
    for state_name,item in q['states'].items():
        guard();pin=item['checkpoint'];value=torch.load(K.checked(pin),map_location='cpu',weights_only=True,mmap=True);weights=value if item['storage']=='plain_state_dict' else value['model']
        model=ATMS().cuda().eval();model.load_state_dict(weights,strict=True);model.requires_grad_(True);frozen=digest(model)
        named=list(model.named_parameters());params=[p for _,p in named];layout=[];pos=0
        for name,p in named:layout.append(dict(name=name,group=group(name),start=pos,stop=pos+p.numel()));pos+=p.numel()
        assert pos==3200413
        if not (out/'parameter_layout.json').exists():K.write(out/'parameter_layout.json',layout)
        else:assert K.read(out/'parameter_layout.json')==layout
        sums_c=np.zeros(pos,np.float64);sums_f=np.zeros(pos,np.float64);coverage=[];loss_sums=dict(global_loss=0.,correct_fine_loss=0.,coarse_loss=0.);state_checks=[]
        for bi,ids in enumerate(batches):
            guard();pred=model(raw[ids].cuda(),torch.ones(len(ids),device='cuda',dtype=torch.long));scale=model.logit_scale.exp().clamp_max(100)
            scores=scale*F.normalize(pred,dim=1)@F.normalize(target[ids].cuda(),dim=1).T;_,parts=torch_loss(scores,'conditional')
            gg=torch.autograd.grad(parts['global_loss'],params,retain_graph=True,allow_unused=True) if bi in (0,82) else None
            coarse=torch.autograd.grad(parts['coarse_loss'],params,retain_graph=True,allow_unused=True);fine=torch.autograd.grad(parts['correct_fine_loss'],params,allow_unused=True)
            vc=flatten(coarse,params);vf=flatten(fine,params);assert np.isfinite(vc).all() and np.isfinite(vf).all()
            if gg is not None:
                vg=flatten(gg,params);err=float(np.max(abs(vg.astype(float)-vc.astype(float)-vf.astype(float))/(1+abs(vg.astype(float)))));assert err<5e-5
                state_checks.append(dict(batch=bi,global_equals_coarse_plus_fine_gradient_scaled_max_error=err))
            if bi==0:
                cpu=ATMS().eval();cpu.load_state_dict(weights,strict=True);cpu.requires_grad_(True);cpu_params=list(cpu.parameters())
                cp=cpu(raw[ids],torch.ones(len(ids),dtype=torch.long));cs=cpu.logit_scale.exp().clamp_max(100)*F.normalize(cp,dim=1)@F.normalize(target[ids],dim=1).T;_,pp=torch_loss(cs,'conditional')
                cc=flatten(torch.autograd.grad(pp['coarse_loss'],cpu_params,retain_graph=True,allow_unused=True),cpu_params);cf=flatten(torch.autograd.grad(pp['correct_fine_loss'],cpu_params,allow_unused=True),cpu_params)
                errors={key:float(np.max(abs(g.astype(float)-h.astype(float))/(1+abs(h.astype(float))))) for key,g,h in [('coarse',vc,cc),('fine',vf,cf)]};assert max(errors.values())<5e-5
                state_checks.append(dict(batch=0,full160_sample_native_CPU_gradient_scaled_max_errors=errors,CPU_loss={k:float(v.detach()) for k,v in pp.items()}))
                np.savez_compressed(out/(state_name+'_first_batch_gradient_slices.npz'),coarse_GPU=vc[:4096],fine_GPU=vf[:4096],coarse_CPU=cc[:4096],fine_CPU=cf[:4096])
                del cpu,cpu_params,cp,cs,pp,cc,cf
            sums_c+=vc.astype(np.float64)*len(ids);sums_f+=vf.astype(np.float64)*len(ids);coverage.extend(ids.tolist())
            row=dict(state=state_name,batch=bi,queries=len(ids),array_row_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),gradient_stats=grouped(vc,vf,layout),losses={k:float(parts[k].detach()) for k in loss_sums})
            with journal.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            for key in loss_sums:loss_sums[key]+=row['losses'][key]*len(ids)
            del pred,scale,scores,parts,coarse,fine,gg,vc,vf
        assert sorted(coverage)==sorted(T.tolist()) and len(coverage)==13230 and digest(model)==frozen
        mean_c=(sums_c/13230).astype(np.float32);mean_f=(sums_f/13230).astype(np.float32)
        K.save_array(out/(state_name+'_coarse_mean_gradient.npy'),mean_c);K.save_array(out/(state_name+'_fine_mean_gradient.npy'),mean_f)
        summaries[state_name]=dict(checkpoint=pin,queries=13230,batches=83,mean_losses={k:v/13230 for k,v in loss_sums.items()},mean_gradient_stats=grouped(mean_c,mean_f,layout),
            coarse_gradient=K.ref(out/(state_name+'_coarse_mean_gradient.npy')),fine_gradient=K.ref(out/(state_name+'_fine_mean_gradient.npy')),all_weights_buffers_unchanged=True,model_FP32_digest=frozen)
        checks[state_name]=state_checks;assert K.sha(pin['path'])==pin['sha256'];del model,value,weights,params,named,sums_c,sums_f;gc.collect();torch.cuda.empty_cache()
        K.write(out/'progress.json',dict(completed_states=list(summaries),all_states=4))
    guard();K.write(out/'completion.json',dict(status='COMPLETE_FROZEN_FOUR_STATE_ALL_T_COARSE_FINE_GRADIENT_DIAGNOSIS',UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        policy=K.ref(a.policy),states=summaries,engineering_checks=checks,parameter_layout=K.ref(out/'parameter_layout.json'),batches=K.ref(out/'batches.json'),all332_stats=K.ref(journal),
        frozen_eval_mode_NOT_actual_training_dropout_gradients=True,all_T_same_fixed83_batches=True,full_T_mean_gradients_float64_accumulated_then_FP32_saved=True,
        no_new_model_updates_or_images=True,not_causal_evidence_for_image_gain=True,PCGrad_or_layer_branching_not_claimed_new=True,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_before=before,resources_after=resources()))
except BaseException as exc:
    K.write(out/'failure.json',dict(status='FAILED_PARTIAL_EVIDENCE_RETAINED_NO_AUTOMATIC_RETRY',error=repr(exc),traceback=traceback.format_exc()));raise
