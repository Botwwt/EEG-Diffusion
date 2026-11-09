"""Continuous fixed-order T training with a durable commit after every attempt."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_continuous_margin_contract_20261106_v203 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['fit','conditions'],required=True)
ap.add_argument('--arm',default='');ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=0);a=ap.parse_args()
c=C.setup(a.policy);q,K,tc=c['q'],c['K'],c['train'];out=c['fit'];started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['fit'];before=C.resources(c,True)
suffix='fit_'+a.arm+'_'+str(a.start)+'_'+str(a.end) if a.mode=='fit' else 'conditions'
job=K.read(c['repo']/'audit'/(q['job_prefix']+'_'+suffix+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
launch=K.read(Path(q['code_root'])/'actual_launch.json')
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==tc['physical']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path[:0]=tc['physical']['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from safetensors import safe_open
import atm_continuous_margin_core_20261106_v202 as H
import atm_margin_checkpoint_20261106_v202 as CK
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
last_disk_check=-1.

def guard(force_disk=False):
    global last_disk_check
    elapsed=time.monotonic()-started
    assert elapsed<850 and time.time()<launch['absolute_deadline_epoch']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30 and torch.cuda.max_memory_allocated()<20<<30
    # No output writes happen during individual Jacobian rows. Avoid thousands
    # of repeated directory walks; force the same allocation check at each write.
    if force_disk or elapsed-last_disk_check>=1.:
        used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
        fs=os.statvfs(out);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
        last_disk_check=elapsed

try:
    src={k:torch.load(K.checked(v),map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()}
    initial={n:src['NN'][n] if n.startswith('proj_eeg.') else v for n,v in src['CC'].items()}
    xall=torch.load(K.checked(tc['physical']['EEG']),map_location='cpu',weights_only=True,mmap=True)
    policy_pin=K.ref(a.policy)
    if a.mode=='fit':
        assert a.arm in C.ARMS[2:] and 0<=a.start<a.end<=830
        assert a.end-a.start<=(q['constrained_attempts_per_job'] if a.arm=='margin_constrained' else 83)
        dest=out/a.arm;dest.mkdir(exist_ok=True)
        vall=torch.load(K.checked(tc['physical']['visual_target']),map_location='cpu',weights_only=True,mmap=True)
        anchor=ATMS().cuda().eval().requires_grad_(False);anchor.load_state_dict(src['NN'],strict=True)
        net=ATMS().cuda();net.load_state_dict(initial,strict=True)
        Qdone=K.read(K.checked(q['Qbank_completion']));bank=np.load(K.checked(Qdone['bank']));w={};kv=[]
        with safe_open(K.checked(c['cv']['bq']['h94']),framework='pt',device='cpu') as f:
            for key,name in [('P_weight','image_proj.proj.weight'),('P_bias','image_proj.proj.bias'),('P_gamma','image_proj.norm.weight'),('P_beta','image_proj.norm.bias')]:w[key]=f.get_tensor(name).half().float().cuda()
            for lay in Qdone['layouts']:
                ix=lay['ip_index'];kv.append({k:f.get_tensor(f'ip_adapter.{ix}.{n}.weight').half().float().cuda() for k,n in [('K','to_k_ip'),('V','to_v_ip')]})
        tr=H.Trainer(net,anchor,initial,xall,vall,tc['T'],bank,Qdone['layouts'],w,kv,guard)
        assert np.array_equal(tr.baseline,np.load(K.checked(q['baseline_T']))['metrics'])
        if a.start:
            state,receipt=CK.load(dest,policy_pin);assert state['attempts']==a.start;tr.restore(state)
        else:
            assert not (dest/'checkpoint.json').exists();guard(True);CK.commit(dest,tr.snapshot(),policy_pin)
        for index in range(a.start,a.end):
            assert tr.attempts==index;guard();t0=time.monotonic();event=tr.attempt(a.arm);duration=time.monotonic()-t0
            if a.arm=='margin_constrained' and index<2:
                expected=K.read(K.checked(q['profile_completion']))['first' if index==0 else 'second']
                assert event==expected,'Continuous profile event changed before formal fit'
            guard(True);receipt=CK.commit(dest,tr.snapshot(),policy_pin);guard(True)
            K.write(dest/'progress.json',dict(attempts=tr.attempts,accepted=tr.accepted,exposures=tr.exposures,last_attempt_seconds=duration,checkpoint=receipt,latest=event))
            print(json.dumps(dict(arm=a.arm,attempts=tr.attempts,accepted=tr.accepted,exposures=tr.exposures,last_attempt_seconds=duration,response_loss=event['after']['loss'],scope_margin_decreases=event['after']['decreased'])),flush=True)
            if tr.attempts%83==0:
                epoch=tr.attempts//83;metrics=tr.all_T();guard(True)
                path=dest/f'epoch{epoch:02d}_allT.npz';assert not path.exists();np.savez_compressed(path,metrics=metrics,T_rows=tc['T'])
                events=[K.read(dest/'events'/f'{i:04d}.json') for i in range((epoch-1)*83+1,epoch*83+1)]
                assert sum(e['queries'] for e in events)==13230 and sum(e['constraint_queries'] for e in events)==26460
                K.write(dest/f'epoch{epoch:02d}.json',dict(epoch=epoch,attempts=tr.attempts,accepted_total=tr.accepted,accepted_epoch=sum(e['accepted'] for e in events),exposures=tr.exposures,
                    full_T_metrics=K.ref(path),mean_response_loss_before=sum(e['response_loss_before']*e['queries'] for e in events)/13230,
                    mean_response_loss_after=sum(e['after']['loss']*e['queries'] for e in events)/13230,
                    within_strict_top1=float((metrics[:,1]>0).mean()),global_strict_top1=float((metrics[:,2]>0).mean()),
                    full_T_below_frozen_margin=(metrics[:,1:]-tr.baseline[:,1:]<-1e-6).sum(0).tolist(),
                    scope_only_feasibility_not_all_T_guarantee=True))
        assert tr.attempts==a.end
        result=dict(status='COMPLETE_CONTINUOUS_MARGIN_ATTEMPT_BLOCK',arm=a.arm,start=a.start,end=a.end,attempts=tr.attempts,accepted=tr.accepted,exposures=tr.exposures,checkpoint=receipt['checkpoint'],pointer=K.ref(dest/'checkpoint.json'))
        if a.end==830:
            assert tr.exposures==132300
            K.write(dest/'fit_10_completion.json',dict(result,status='COMPLETE_FIXED10_CONTINUOUS_MARGIN_ARM',epochs=10,all_attempts_committed=True,policy=policy_pin))
    else:
        test=torch.load(K.checked(tc['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True)
        ids=[r['array_row'] for r in tc['queries']];values=np.empty((600,2,1024),np.float32);models=[]
        for j,arm in enumerate(C.ARMS[2:]):
            done=K.read(out/arm/'fit_10_completion.json');assert done['attempts']==830 and done['exposures']==132300
            state,receipt=CK.load(out/arm,policy_pin);assert receipt['checkpoint']==done['checkpoint']
            net=ATMS().cuda().eval();net.load_state_dict({**initial,**state['trainable']},strict=True)
            with torch.inference_mode():
                for lo in range(0,600,50):
                    guard();x=xall[ids[lo:lo+50]] if lo<400 else test[lo-400:lo-350]
                    values[lo:lo+50,j]=net(x.cuda(),torch.ones(len(x),device='cuda',dtype=torch.long)).cpu().numpy()
            models.append(done['checkpoint']);del net
        assert np.isfinite(values).all();guard(True);assert not (out/'conditions.npy').exists();np.save(out/'conditions.npy',values)
        result=dict(status='COMPLETE_CONTINUOUS_MARGIN_EEG_ONLY1200_CONDITIONS',conditions=K.ref(out/'conditions.npy'),models=models,normal_inference_only_EEG=True)
    guard(True);result.update(seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)
    K.write(out/(suffix+'_completion.json'),result)
except BaseException as exc:
    checkpoint=K.ref(dest/'checkpoint.json') if 'dest' in globals() and (dest/'checkpoint.json').exists() else None
    K.write(out/(suffix+'_failure.json'),dict(error=repr(exc),traceback=traceback.format_exc(),last_committed_pointer=checkpoint,uncommitted_parameters_not_promoted=True,seconds=time.monotonic()-started));raise
