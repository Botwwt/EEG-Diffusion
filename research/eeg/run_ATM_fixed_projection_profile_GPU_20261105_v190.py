"""Finite actual gradient and Adam equivalence profile; all steps discarded."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,copy,fcntl,json,resource,subprocess,sys,time,traceback
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);a=ap.parse_args()
start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
q=json.loads(Path(a.policy).read_bytes());assert q['status']=='LOCKED_FIXED_PROJECTION_ENGINEERING_PROFILE_ONLY'
sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
import atm_CN_image_contract_20261105_v188 as C
c=C.setup(q['prior_policy']['path']);K=c['K'];K.checked(q['prior_policy'])
for pin in q['sources'].values():K.checked(pin)
assert K.ref(__file__)==q['sources']['GPU']
out=Path(q['output_root']);out.mkdir();before=C.resources(c,True)
repo=c['repo'];job=K.read(repo/'audit'/(q['job_name']+'_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==c['physical']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path.insert(0,str(Path(q['training_policy']['path']).parent))
import atm_repeat_count_training_contract_20261105_v1 as T
tc=T.setup(K.checked(q['training_policy']));pq=tc['physical'];sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch,numpy as np
from models.atms import ATMS
from atm_conditional_stimulus_objective_20261104_v1 import torch_loss,numpy_loss_gradient
import atm_actual_NCE_Adam_repeat_core_20261105_v1 as G
import atm_repeat_count_training_core_20261105_v1 as F
import atm_fixed_projection_training_core_20261105_v190 as H
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
def guard():
 assert time.monotonic()-start<850
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
 assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
 assert K.allocated(q['project_root'])<q['maximum_output_bytes']
 C.resources(c,False)
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
try:
 base=torch.load(K.checked(tc['q']['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
 assert base['epoch']==10 and base['updates']==830
 net=ATMS().cuda();net.load_state_dict(base['model'],strict=True);net.train()
 opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01);opt.load_state_dict(copy.deepcopy(base['optimizer']))
 meta=G.F.metadata(net);ids=tc['T'][(np.random.default_rng(1739).permutation(1323)[:16,None]*10+np.arange(10)).reshape(-1)]
 obj=np.load(K.checked(pq['physical_source']),allow_pickle=True);obj=obj if isinstance(obj,dict) else obj.item()
 block=obj['preprocessed_eeg_data'][ids].astype(np.float32)
 mean=torch.load(pq['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
 assert np.array_equal(block.mean(1),mean[ids].numpy())
 targets=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy()[ids]
 x=torch.tensor(block.mean(1),device='cuda');y=torch.tensor(targets,device='cuda')
 oldg,oldloss,active=F.collect_training_gradients(net,meta,block,targets,101729,torch_loss,guard)
 oldbuffers={n:b.detach().clone() for n,b in net.named_buffers()}
 F.combine_and_assign(net,meta,oldg,oldloss,active,'R4_continued');opt.step()
 reference_model={n:v.detach().clone() for n,v in net.state_dict().items()};reference_opt=copy.deepcopy(opt.state_dict())
 net.load_state_dict(base['model'],strict=True);opt.load_state_dict(copy.deepcopy(base['optimizer']))
 loss,gs,logits=H.native_gradients(net,meta,x,y,101729,torch_loss)
 flat=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(meta['parameters'],gs)]).double()
 error=float((flat-oldg[3]).abs().max());assert error==0
 assert all(torch.equal(b,oldbuffers[n]) for n,b in net.named_buffers())
 assert abs(float(loss)-oldloss[3])<1e-7
 ref=numpy_loss_gradient(logits.detach().cpu().numpy(),'nce_continued');assert abs(float(loss)-ref['loss'])<3e-5
 H.assign_clip_mask(net,meta,gs,False);opt.step()
 assert all(torch.equal(v,reference_model[n]) for n,v in net.state_dict().items()) and equal(opt.state_dict(),reference_opt)
 proofs=[]
 for seed in [101729,101730,111729]:
  net.load_state_dict(base['model'],strict=True);opt.load_state_dict(copy.deepcopy(base['optimizer']));net.train()
  loss,gs,logits=H.native_gradients(net,meta,x,y,seed,torch_loss)
  op_before=copy.deepcopy(opt.state_dict());name_to_id=dict(zip([n for n,p in net.named_parameters()],op_before['param_groups'][0]['params']))
  H.assign_clip_mask(net,meta,gs,False)
  opt.step();unrestricted={n:p.detach().clone() for n,p in net.named_parameters()};op_full=copy.deepcopy(opt.state_dict())
  net.load_state_dict(base['model'],strict=True);opt.load_state_dict(copy.deepcopy(base['optimizer']))
  # Same cached gradients; the mask is the sole change in the optimizer step.
  info=H.assign_clip_mask(net,meta,gs,True);opt.step();op_mask=opt.state_dict();changed=[]
  for name,p in net.named_parameters():
   pid=name_to_id[name]
   if name.startswith('proj_eeg.'):
    assert torch.equal(p.cpu(),base['model'][name]) and equal(op_mask['state'].get(pid),op_before['state'].get(pid))
   else:
    assert torch.equal(p,unrestricted[name]) and equal(op_mask['state'].get(pid),op_full['state'].get(pid))
    if not torch.equal(p.cpu(),base['model'][name]):changed.append(name)
  assert any(n.startswith('encoder.') for n in changed) and any(n.startswith('enc_eeg.') for n in changed)
  proofs.append(dict(seed=seed,loss=float(loss),**info,nonprojection_step_and_Adam_exact=True,projection_parameters_and_Adam_unchanged=True,changed_names=changed))
  guard()
 net.load_state_dict(base['model'],strict=True);opt.load_state_dict(copy.deepcopy(base['optimizer']))
 assert all(torch.equal(v.cpu(),base['model'][n]) for n,v in net.state_dict().items())
 assert equal(opt.state_dict(),base['optimizer'])
 np.savez_compressed(out/'fixture.npz',rows=ids,logits=logits.detach().cpu().numpy(),old_losses=oldloss,old_gradient_gram=(oldg@oldg.T).cpu().numpy())
 guard();K.write(out/'completion.json',dict(status='PASS_NATIVE_C_FAST_PATH_BITEXACT_AND_FIXED_HEAD_MASKED_ADAM',policy=K.ref(a.policy),
  native15view_R4_vs_single_forward_gradient_max_error=error,native_C_step_model_and_Adam_bitexact=True,all_BN_buffers_native_once=True,
  fixed_projection_proofs=proofs,temporary_updates_discarded=8,new_final_models=0,new_images=0,all_original_model_restored=True,
  seconds=time.monotonic()-start,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
except BaseException as exc:
 K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
