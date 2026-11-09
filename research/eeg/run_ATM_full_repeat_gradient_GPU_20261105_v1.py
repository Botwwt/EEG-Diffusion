"""Full T at the frozen positive encoder, plus fixed T8 +/- one small direction."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_full_repeat_gradient_contract_20261105_v1 as C
import atm_full_repeat_gradient_core_20261105_v1 as F
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);args=ap.parse_args()
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(args.policy);q,K,out,pq=c['q'],c['K'],c['out'],c['physical']
assert K.ref(__file__)==q['sources']['GPU'];before=C.resources(c,starting=True)
repo=Path(pq['GPU_wrapper']['path']).parent
job=K.read(repo/'audit'/(q['job_prefix']+'_measure_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)

def guard():
    assert time.monotonic()-started<850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
    assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
    used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
    stat=os.statvfs(q['project_root']);assert stat.f_bavail*stat.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used

try:
    obj=np.load(K.checked(pq['physical_source']),allow_pickle=True)
    if not isinstance(obj,dict):obj=obj.item()
    raw=obj['preprocessed_eeg_data'];assert raw.shape==(16540,4,63,250) and raw.dtype==np.float32
    mean4=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
    targets=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy()
    state=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
    assert state['updates']==830 and state['epoch']==10 and state['arm']=='nce_continued'
    base=state['model'];net=ATMS().cuda().eval();net.load_state_dict(base,strict=True);meta=F.metadata(net)
    T=c['T'];rows=T.reshape(-1,10)[:8,0];block=raw[rows].astype(np.float32)
    y10=targets[(rows//10)[:,None]*10+np.arange(10)]
    assert np.array_equal(block.mean(1),mean4[rows].numpy())
    profile={};saved={};gradient=None;direction=None
    for point,sign in [('base',0),('plus',1),('minus',-1)]:
        if sign:F.perturb(net,meta,base,sign*q['parameter_step_L2']*direction)
        g,result=F.collect(net,meta,block,y10,rows%10,'cuda',torch.float32,record=True,guard=guard)
        ev=F.evidence(g,meta)
        if point=='base':
            direction=g[3,2]/torch.linalg.vector_norm(g[3,2])
            gradient=g.clone()
        profile[point]=dict(comparison=ev['comparison'],losses=result['losses'].tolist(),
            inactive_parameter_tensors=ev['inactive_parameter_tensors'],
            actual_step_L2=float(torch.linalg.vector_norm(torch.cat([(p.detach()-base[n].cuda()).reshape(-1).double() for n,p in zip(meta['names'],meta['parameters'])]))))
        for key,value in result.items():saved[point+'_'+key]=value
        for key in ('norms','tensor_squared_norms','gradient_samples'):saved[point+'_count_'+key]=ev[key]
        K.write(out/'progress.json',dict(phase='T8_FULL_NETWORK_PARAMETER_NEIGHBORHOOD',point=point))
    derivative=torch.einsum('nkp,p->nk',gradient,direction).cpu().numpy()
    plus=np.array([saved['plus_losses'][F.COUNTS==n].mean(0) for n in range(1,5)])
    minus=np.array([saved['minus_losses'][F.COUNTS==n].mean(0) for n in range(1,5)])
    central=(plus-minus)/(2*q['parameter_step_L2'])
    profile['finite_difference']=dict(central=central.tolist(),autograd=derivative.tolist(),absolute_error=abs(central-derivative).tolist(),
        scientific_diagnostic_not_assumed_exact=True,only_one_direction_and_two_parameter_offsets=True)
    F.perturb(net,meta,base,torch.zeros_like(direction))
    for name,value in net.state_dict().items():assert torch.equal(value.cpu(),base[name])
    saved.update(rows=rows,raw=block,targets10=y10,sampled_parameter_indices=meta['indices'])
    np.savez_compressed(out/'fixed_T8_neighborhood_fixture.npz',**saved)
    K.write(out/'profile.json',profile)
    del g,gradient,direction,saved,result,ev
    total=torch.zeros((4,3,meta['dimension']),device='cuda',dtype=torch.float64)
    loss_sums=np.zeros((15,3),float);coverage=np.zeros((13230,15),np.uint8);batches=[]
    for lo in range(0,13230,q['batch']):
        guard();rows=T[lo:lo+q['batch']];block=raw[rows].astype(np.float32)
        assert np.array_equal(block.mean(1),mean4[rows].numpy())
        y10=targets[(rows//10)[:,None]*10+np.arange(10)]
        g,r=F.collect(net,meta,block,y10,rows%10,'cuda',torch.float32,guard=guard)
        total.add_(g,alpha=len(rows)/13230);loss_sums+=r['losses']*len(rows)/13230
        coverage[lo:lo+len(rows)]+=1
        short={}
        for k,key in enumerate(F.OBJECTIVES):
            truth=g[2,k];tn=torch.linalg.vector_norm(truth)
            short[key]={name:dict(relative_L2_error=float(torch.linalg.vector_norm(p-truth)/tn),cosine=float(torch.dot(p,truth)/(torch.linalg.vector_norm(p)*tn)))
                for name,p in zip(F.METHODS,(g[1,k]/3+2*g[3,k]/3,(-g[0,k]+12*g[1,k]+16*g[3,k])/27,g[3,k]))}
        event=dict(first_T_position=lo,rows=len(rows),comparison=short);batches.append(event)
        with (out/'batch_log.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        K.write(out/'progress.json',dict(phase='FULL_T_ALL15_WHOLE_ENCODER_GRADIENT',completed=min(lo+q['batch'],13230),total=13230,seconds=time.monotonic()-started))
    assert np.all(coverage==1);ev=F.evidence(total,meta)
    gram=torch.einsum('ikp,jlp->ikjl',total,total).cpu().numpy()
    np.savez_compressed(out/'full_T_gradient_statistics.npz',gradient_gram=gram,losses=loss_sums,coverage=coverage,T=T,
        norms=ev['norms'],tensor_squared_norms=ev['tensor_squared_norms'],gradient_samples=ev['gradient_samples'],sampled_parameter_indices=meta['indices'])
    K.write(out/'full_T_readout.json',dict(status='COMPLETE_FROZEN_FULL_ATMS_T_GRADIENT_DIAGNOSIS',comparison=ev['comparison'],
        inactive_parameter_tensors=ev['inactive_parameter_tensors'],parameters=meta['dimension'],objectives=list(F.OBJECTIVES),
        parameter_names=meta['names'],parameter_slices=meta['slices'],full_T_training_data_NOT_OOF=True,
        eval_mode_NOT_actual_Adam_or_dropout_training_trajectory=True,R80_gradient_NOT_validated=True,
        batch_count=len(batches),batch_size=q['batch'],no_batch_bootstrap_as_independent_concepts=True))
    for name,value in net.state_dict().items():assert torch.equal(value.cpu(),base[name])
    guard()
    K.write(out/'scientific_completion.json',dict(status='COMPLETE_FULL_T_AND_FIXED_NEIGHBORHOOD_WHOLE_ENCODER_GRADIENTS',policy=K.ref(args.policy),
        fixture=K.ref(out/'fixed_T8_neighborhood_fixture.npz'),statistics=K.ref(out/'full_T_gradient_statistics.npz'),readout=K.ref(out/'full_T_readout.json'),profile=K.ref(out/'profile.json'),
        full_T_forwards=198450,profile_forwards=360,objectives_per_forward=3,temporary_parameter_perturbations=2,original_parameters_and_buffers_restored=True,
        new_models=0,new_updates=0,new_images=0,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
except BaseException as exc:
    K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
