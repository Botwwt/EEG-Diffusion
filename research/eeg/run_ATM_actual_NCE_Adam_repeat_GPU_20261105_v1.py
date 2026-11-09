"""Frozen native NCE and current Adam state across every physical repeat subset."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_actual_NCE_Adam_repeat_contract_20261105_v1 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);args=ap.parse_args()
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(args.policy);q,K,out,pq=c['q'],c['K'],c['out'],c['physical']
assert K.ref(__file__)==q['sources']['GPU'];before=C.resources(c,starting=True)
repo=Path(pq['GPU_wrapper']['path']).parent;job=K.read(repo/'audit'/(q['job_prefix']+'_measure_job.json'))
assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from atm_conditional_stimulus_objective_20261104_v1 import torch_loss,numpy_loss_gradient
import atm_actual_NCE_Adam_repeat_core_20261105_v1 as F
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
def guard():
    assert time.monotonic()-started<850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
    assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
    used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
    st=os.statvfs(q['project_root']);assert st.f_bavail*st.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
try:
    obj=np.load(K.checked(pq['physical_source']),allow_pickle=True)
    if not isinstance(obj,dict):obj=obj.item()
    raw=obj['preprocessed_eeg_data'];assert raw.shape==(16540,4,63,250) and raw.dtype==np.float32
    mean4=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
    target=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy()
    state=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
    assert state['updates']==830 and state['epoch']==10 and state['arm']=='nce_continued'
    base=state['model'];net=ATMS().cuda();net.load_state_dict(base,strict=True);meta=F.F.metadata(net)
    ref=F.optimizer_reference(meta,state,'cuda')
    T=c['T'];order=np.random.default_rng(1739).permutation(1323);positions=(order[:,None]*10+np.arange(10)).reshape(-1)
    rows=T[positions[:160]];block=raw[rows].astype(np.float32);assert np.array_equal(block.mean(1),mean4[rows].numpy())
    fixtures=dict(rows=rows,first8_targets=target[rows[:8]],sampled_parameter_indices=meta['indices'])
    profile={};masks={}
    for mode in q['modes']:
        net.train(mode=='train_common_RNG');g,r=F.collect(net,meta,block,target[rows],base,101729,'cuda',torch.float32,torch_loss,record=True,guard=guard)
        evidence,spaces=F.measurements(g,meta,r['active_tensors'],ref);masks[mode]=r['active_tensors']
        profile[mode]=evidence['summary']
        for key,value in r.items():fixtures[mode+'_'+key]=value
        for key,V in evidence['grams'].items():fixtures[mode+'_'+key+'_gram']=V
        for key,V in evidence['samples'].items():fixtures[mode+'_'+key+'_samples']=V
        # Exact same CUDA dropout seed/buffers and original objective must replay R4.
        F.restore_buffers(net,base);torch.cuda.manual_seed(101729)
        z=net(torch.tensor(block.mean(1),device='cuda'),torch.ones(160,device='cuda',dtype=torch.long))
        logits=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(z,dim=1)@torch.nn.functional.normalize(torch.tensor(target[rows],device='cuda'),dim=1).T
        loss,_=torch_loss(logits,'nce_continued');dl=torch.autograd.grad(loss,logits,retain_graph=True)[0]
        reference=numpy_loss_gradient(logits.detach().cpu().numpy(),'nce_continued')
        assert torch.equal(logits.detach().cpu(),torch.from_numpy(r['R4_logits']))
        err=float(abs(dl.detach().cpu().numpy()-reference['gradient']).max());assert err<3e-6
        profile[mode]['original_objective_proof']=dict(exact_R4_CUDA_logits_replay=True,NumPy_logit_gradient_max_error=err,loss_error=abs(float(loss)-reference['loss']))
        assert abs(float(loss)-reference['loss'])<3e-5
        del z,logits,loss,dl
        if mode=='train_common_RNG':
            # ONE temporary real torch.AdamW step, then restore model exactly.
            opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
            import copy
            opt.load_state_dict(copy.deepcopy(state['optimizer']))
            for p,(a,b),active in zip(meta['parameters'],meta['slices'],r['active_tensors']):p.grad=g[3,a:b].to(p.dtype).reshape_as(p).clone() if active else None
            clipped=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.));opt.step()
            expected=ref['theta']+spaces['Adam_total_update'][3]
            actual=torch.cat([p.detach().reshape(-1).double() for p in meta['parameters']])
            error=float((actual-expected).abs().max());assert error<3e-6
            profile[mode]['actual_torch_AdamW_step_proof']=dict(full_parameter_max_error=error,preclip_gradient_norm=clipped,
                temporary_engineering_steps=1,final_models_saved=0,optimizer_state_830_used=True)
            net.load_state_dict(base,strict=True);del opt,expected,actual
        for name,value in net.state_dict().items():assert torch.equal(value.detach().cpu(),base[name])
        del g,r,evidence,spaces;torch.cuda.empty_cache();guard()
    np.savez_compressed(out/'fixed160_global_NCE_fixture.npz',**fixtures);K.write(out/'profile.json',profile);del fixtures
    statistics=dict(T=T,concept_order=order,positions=positions,sampled_parameter_indices=meta['indices']);readouts={}
    for mode in q['modes']:
        net.train(mode=='train_common_RNG');total=torch.zeros((4,meta['dimension']),device='cuda',dtype=torch.float64)
        losses=np.zeros(15);coverage=np.zeros((13230,15),np.uint8)
        for bi,lo in enumerate(range(0,13230,160)):
            guard();pos=positions[lo:lo+160];rows=T[pos];block=raw[rows].astype(np.float32)
            assert np.array_equal(block.mean(1),mean4[rows].numpy())
            g,r=F.collect(net,meta,block,target[rows],base,101729+bi,'cuda',torch.float32,torch_loss,guard=guard)
            assert np.array_equal(r['active_tensors'],masks[mode]);total.add_(g,alpha=len(rows)/13230)
            losses+=r['losses']*len(rows)/13230;coverage[pos]+=1
            ev,spaces=F.measurements(g,meta,r['active_tensors'],ref)
            event=dict(mode=mode,batch=bi,rows=len(rows),first_position=lo,comparison=ev['summary'])
            with (out/'batch_log.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            K.write(out/'progress.json',dict(phase='NATIVE_GLOBAL_NCE_AND_ADAM_FIXED_STATE',mode=mode,completed=min(lo+160,13230),total=13230,seconds=time.monotonic()-started))
        assert np.all(coverage==1);ev,spaces=F.measurements(total,meta,masks[mode],ref)
        readouts[mode]=ev['summary'];statistics[mode+'_coverage']=coverage;statistics[mode+'_losses']=losses
        for key,V in ev['grams'].items():statistics[mode+'_'+key+'_gram']=V
        for key,V in ev['samples'].items():statistics[mode+'_'+key+'_samples']=V
        statistics[mode+'_active_tensors']=masks[mode]
        del total,g,ev,spaces;torch.cuda.empty_cache()
    np.savez_compressed(out/'full_T_global_NCE_Adam_statistics.npz',**statistics)
    K.write(out/'full_T_readout.json',dict(status='COMPLETE_NATIVE_GLOBAL_NCE_ADAM_FROZEN_STATE_COUNT_DIAGNOSIS',modes=readouts,
        parameter_names=meta['names'],parameter_slices=meta['slices'],original_logit_scale=float(net.logit_scale),effective_temperature_scale=float(net.logit_scale.exp().clamp_max(100)),
        global_means_are_sample_weighted_BATCH_InfoNCE_gradients=True,frozen_Adam_state_NOT_830_new_updates=True,
        train_common_RNG_resets_buffers_and_dropout_each_subset=True,all_T_training_data_NOT_OOF=True,no_R80_validation=True))
    for name,value in net.state_dict().items():assert torch.equal(value.detach().cpu(),base[name])
    guard();K.write(out/'scientific_completion.json',dict(status='COMPLETE_ACTUAL_GLOBAL_NCE_ADAM_ALL_T_TWO_MODES',policy=K.ref(args.policy),
        fixture=K.ref(out/'fixed160_global_NCE_fixture.npz'),statistics=K.ref(out/'full_T_global_NCE_Adam_statistics.npz'),readout=K.ref(out/'full_T_readout.json'),profile=K.ref(out/'profile.json'),
        full_T_forwards=396900,profile_forwards=5120,temporary_engineering_Adam_steps=1,new_models=0,new_updates=0,new_images=0,
        original_parameters_and_buffers_restored=True,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
except BaseException as exc:
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
