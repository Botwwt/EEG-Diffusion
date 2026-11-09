"""Two frozen ATMS models, all T/all15 physical subsets, actual LN derivatives."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import argparse, fcntl, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import atm_repeat_gradient_contract_20261105_v1 as C
from atm_repeat_gradient_math_20261105_v1 import loss_affine_gradients, compare_gradient_counts, gradient_summary

ap=argparse.ArgumentParser(); ap.add_argument('--policy',required=True); args=ap.parse_args()
started=time.monotonic(); rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(args.policy); q,K,out=c['q'],c['K'],c['out']; pq=c['prior']['q']
from enigma_repeat_task_risk_math_20261105_v1 import SUBSETS,COUNTS,task_metrics,summaries
assert K.ref(__file__)==q['sources']['GPU']
before=C.resources(c,starting=True)
repo=Path(pq['GPU_wrapper']['path']).parent
job=K.read(repo/'audit'/(q['job_prefix']+'_measure_job.json'))
assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)

def guard():
    assert time.monotonic()-started<850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
    assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
    used=K.allocated(q['project_root']); assert used<q['maximum_output_bytes']
    stat=os.statvfs(q['project_root'])
    assert stat.f_bavail*stat.f_frsize >= before['required_including_this_project']+before['own_used_bytes']-used

try:
    obj=np.load(K.checked(pq['physical_source']),allow_pickle=True)
    if not isinstance(obj,dict):obj=obj.item()
    raw=obj['preprocessed_eeg_data'];assert raw.shape==(16540,4,63,250) and raw.dtype==np.float32
    mean4=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
    target=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy()
    T=c['T']; totals={}; forwards=0
    fixture=dict(rows=T[:8],raw=raw[T[:8]],targets10=target[(T[:8]//10)[:,None]*10+np.arange(10)])
    for arm in q['arms']:
        state=torch.load(K.checked(q['checkpoints'][arm]),map_location='cpu',weights_only=True,mmap=True)
        if arm=='nce_continued':
            assert state['epoch']==10 and state['updates']==830 and state['exposures']==132300
            assert state['policy']==q['ATM_policy'];state=state['model']
        model=ATMS().cuda().eval().requires_grad_(False);model.load_state_dict(state,strict=True)
        assert sum(p.numel() for p in model.parameters())==3200413
        gamma=state['proj_eeg.2.weight'].numpy().astype(float);beta=state['proj_eeg.2.bias'].numpy().astype(float)
        assert gamma.shape==beta.shape==(1024,) and np.min(abs(gamma))>1e-4
        per=np.empty((13230,15,5),float);mean=np.empty((13230,4,5),float);disp=np.empty((13230,4),float)
        risks=np.empty((13230,15,3),float);comparisons=np.empty((13230,3,3,3),float)
        aggregate=np.zeros((15,3,2048),float); coverage=np.zeros((13230,15),np.uint8)
        max_ln_mean=0.;saved=[]
        with torch.inference_mode():
            for lo in range(0,13230,64):
                guard();rows=T[lo:lo+64];block=raw[rows].astype(np.float32)
                assert np.array_equal(block.mean(1),mean4[rows].numpy())
                targets10=target[(rows//10)[:,None]*10+np.arange(10)]
                accum=np.zeros((len(rows),4,1024),float);energy=np.zeros((len(rows),4),float)
                gradients=np.zeros((len(rows),4,3,2048),float)
                for j,subset in enumerate(SUBSETS):
                    x=np.ascontiguousarray(block[:,subset].mean(1))
                    pred=model(torch.from_numpy(x).cuda(),torch.ones(len(rows),device='cuda',dtype=torch.long)).cpu().numpy()
                    assert pred.dtype==np.float32 and np.isfinite(pred).all()
                    normalized=(pred.astype(float)-beta)/gamma
                    max_ln_mean=max(max_ln_mean,float(abs(normalized.mean(1)).max()))
                    assert max_ln_mean<2e-5
                    per[lo:lo+len(rows),j]=task_metrics(pred,targets10,rows%10)
                    loss,g=loss_affine_gradients(pred,targets10,rows%10,gamma,beta)
                    risks[lo:lo+len(rows),j]=loss;aggregate[j]+=g.sum(0)/13230
                    k=len(subset)-1;gradients[:,k]+=g;accum[:,k]+=pred;energy[:,k]+=np.square(pred.astype(float)).mean(1)
                    coverage[lo:lo+len(rows),j]+=1;forwards+=len(rows)
                    if lo==0:saved.append(pred[:8].copy())
                for k,n in enumerate(range(1,5)):
                    number=int((COUNTS==n).sum());gradients[:,k]/=number
                    avg=accum[:,k]/number
                    mean[lo:lo+len(rows),k]=task_metrics(avg,targets10,rows%10)
                    disp[lo:lo+len(rows),k]=energy[:,k]/number-np.square(avg).mean(1)
                comparisons[lo:lo+len(rows)]=compare_gradient_counts(gradients)
                if lo%1024==0:K.write(out/'progress.json',dict(arm=arm,completed=min(lo+64,13230),total_forwards=forwards))
        assert np.all(coverage==1)
        for key,v in model.state_dict().items():assert torch.equal(v.cpu(),state[key]),key
        readout,means=summaries(per,mean,disp)
        np.savez_compressed(out/(arm+'_arrays.npz'),per_view=per,loss_of_mean=mean,dispersion=disp,
            count_means=means,losses=risks,gradient_comparisons=comparisons,aggregate_gradients=aggregate,coverage=coverage,T=T)
        fixture[arm+'_prediction']=np.stack(saved,1);fixture[arm+'_gamma']=gamma;fixture[arm+'_beta']=beta
        result=dict(arm=arm,checkpoint=q['checkpoints'][arm],task_risk=readout,gradient_risk=gradient_summary(comparisons),
            actual_parameter_subset='proj_eeg.2.weight AND bias ONLY: 2048 parameters; NOT full-model gradients',
            maximum_inverse_affine_LN_mean_error=max_ln_mean,all_weights_and_buffers_unchanged=True,
            T_is_training_data_NOT_encoder_OOF=True,no_development_or_R80_inputs_used=True)
        K.write(out/(arm+'_readout.json'),result)
        totals[arm]=dict(arrays=K.ref(out/(arm+'_arrays.npz')),readout=K.ref(out/(arm+'_readout.json')))
        del model,state;torch.cuda.empty_cache()
    assert forwards==396900
    np.savez_compressed(out/'fixed8_two_models_fixture.npz',**fixture)
    guard()
    K.write(out/'scientific_completion.json',dict(status='COMPLETE_ATMS_BOTH_FROZEN_FULL_T_SUBSETS_AND_AFFINE_GRADIENTS',
        policy=K.ref(args.policy),arms=totals,fixture=K.ref(out/'fixed8_two_models_fixture.npz'),
        frozen_forwards=forwards,new_updates=0,new_images=0,seconds=time.monotonic()-started,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
except BaseException as exc:
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
