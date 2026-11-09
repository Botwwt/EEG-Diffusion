"""Actual matched T10 training; no test-target input or architectural change."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,copy,fcntl,hashlib,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_repeat_step_profile_contract_20261105_v1 as C
import atm_repeat_step_training_core_20261105_v1 as F
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',default='profile');ap.add_argument('--arm',default='');ap.add_argument('--end',type=int,default=0);args=ap.parse_args()
start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(args.policy);q,K,out,pq=c['q'],c['K'],c['out'],c['physical'];assert K.ref(__file__)==q['sources']['GPU'];before=C.resources(c,starting=True)
repo=Path(pq['GPU_wrapper']['path']).parent;suffix=args.mode+('_'+args.arm+'_'+str(args.end) if args.mode=='fit' else '')
job=K.read(repo/'audit'/(q['job_prefix']+'_measure_job.json'));assert job['pid']==os.getpid() and job['status']=='RUNNING'
lock=(repo/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from atm_conditional_stimulus_objective_20261104_v1 import torch_loss,numpy_loss_gradient
import atm_actual_NCE_Adam_repeat_core_20261105_v1 as G
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(pq['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory)
def guard():
 assert time.monotonic()-start<850
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<pq['maximum_extra_CPU_bytes']
 assert torch.cuda.max_memory_allocated()<pq['maximum_GPU_bytes']
 used=K.allocated(q['project_root']);assert used<q['maximum_output_bytes']
 st=os.statvfs(q['project_root']);assert st.f_bavail*st.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
def save(path,state):
 assert not path.exists();tmp=path.with_name(path.name+'.pending');assert not tmp.exists();torch.save(state,tmp);tmp.replace(path)
def checkpoint(net,opt,epoch,arm,updates,exposures):
 op=copy.deepcopy(opt.state_dict())
 for state in op['state'].values():
  for key,value in state.items():
   if torch.is_tensor(value):state[key]=value.cpu()
 return dict(model={n:v.detach().cpu().clone() for n,v in net.state_dict().items()},optimizer=op,epoch=epoch,arm=arm,updates=updates,exposures=exposures,
  view_exposures=15*exposures,parent_checkpoint=q['checkpoint'],policy=K.ref(args.policy),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state())
def loadnet(state):
 net=ATMS().cuda();net.load_state_dict(state['model'],strict=True);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
 opt.load_state_dict(copy.deepcopy(state['optimizer']));return net,opt

import atm_repeat_count_training_core_20261105_v1 as OLD
import atm_repeat_update_difference_math_20261105_v1 as REF
base=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
obj=np.load(K.checked(pq['physical_source']),allow_pickle=True);obj=obj if isinstance(obj,dict) else obj.item();raw=obj['preprocessed_eeg_data']
target=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy();T=c['T']
order=np.random.default_rng(1739).permutation(1323);proof=[];fixtures={};math=REF.checks()
for label,groups,bi in [('first160',order[:16],0),('last110',order[-11:],82)]:
 ids=T[(groups[:,None]*10+np.arange(10)).reshape(-1)];block=raw[ids].astype(np.float32);seed=101729+bi
 net,opt=loadnet(base);net.train();meta=G.F.metadata(net);signs=F.signs_for(meta,'cuda')
 oldg,oldloss,active=OLD.collect_training_gradients(net,meta,block,target[ids],seed,torch_loss,guard)
 oldbuffers={n:b.detach().clone() for n,b in net.named_buffers()}
 OLD.combine_and_assign(net,meta,oldg,oldloss,active,'R4_continued');opt.step()
 oldparameters=torch.cat([p.detach().reshape(-1) for p in meta['parameters']]);oldoptimizer=copy.deepcopy(opt.state_dict())
 net.load_state_dict(base['model']);opt.load_state_dict(copy.deepcopy(base['optimizer']))
 g,losses,active2=F.collect(net,meta,block,target[ids],seed,torch_loss,guard)
 assert active==active2 and torch.equal(g[4],oldg[3]) and losses[4]==oldloss[3]
 assert all(torch.equal(b,oldbuffers[n]) for n,b in net.named_buffers())
 provisional=F.provisional_Adam(net,opt,meta,g,active)
 assert torch.equal(torch.cat([p.detach().reshape(-1) for p in meta['parameters']]),oldparameters)
 for key,values in oldoptimizer['state'].items():
  for name,value in values.items():assert torch.equal(value,opt.state_dict()['state'][key][name])
 reference,reference_info=REF.solve(provisional['data'].cpu().numpy(),g[4].cpu().numpy(),g[:4].cpu().numpy(),provisional['D'].cpu().numpy())
 correct,correctproof,_=F.proximal(provisional['data'],g[4],g[:4],provisional['D']);err=float(abs(reference-correct.cpu().numpy()).max());assert err<1e-10
 summaries=[]
 for arm in ('native',)+F.ARMS:
  item,fixture=F.finish(net,meta,g,active,provisional,arm,signs,True);summaries.append(item)
  if arm=='native':assert torch.equal(torch.cat([p.detach().reshape(-1) for p in meta['parameters']]),oldparameters)
  for name,value in fixture.items():fixtures[label+'_'+arm+'_'+name]=value
 net.load_state_dict(base['model']);opt.load_state_dict(copy.deepcopy(base['optimizer']))
 assert all(torch.equal(value.cpu(),base['model'][name]) for name,value in net.state_dict().items())
 fixtures[label+'_rows']=ids;fixtures[label+'_losses']=losses;fixtures[label+'_gradient_gram']=(g@g.T).cpu().numpy()
 proof.append(dict(batch=label,rows=len(ids),R4_gradient_bitexact=True,BN_equals_one_R4=True,native_parameters_bitexact=True,Adam_state_bitexact=True,
   GPU_vs_full_NumPy_step_max_error=err,all_temporary_updates_discarded=True,arms=summaries))
 del oldoptimizer,oldparameters,oldg,net,opt,meta,g,provisional;torch.cuda.empty_cache();guard()
np.savez_compressed(out/'profile_fixture.npz',**fixtures)
K.write(out/'scientific_completion.json',dict(status='PASS_REAL_TRAIN_MODE_REPEAT_STEP_PROFILE',proof=proof,math=math,fixture=K.ref(out/'profile_fixture.npz'),
  seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
  new_final_models=0,new_updates=0,temporary_parameter_assignments=8,not_image_efficacy=True))
