"""Finite engineering of a deployable EEG-backbone repair objective.

Uses only fixed training batches and stored real Q. Four temporary Adam updates
are discarded. This is not a training run, heldout evaluation or image result.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import copy,fcntl,json,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
q=json.loads(Path(sys.argv[1]).read_bytes());P=Path(q['parent_root']);out=Path(q['output_root']);started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
sys.path.insert(0,str(P/'code'));import atm_fixed_projection_contract_20261105_v190 as C
c=C.setup(P/'code/policy.json');K=c['K'];tc=c['train'];pq=tc['physical'];before=C.resources(c,True)
assert K.ref(__file__)==q['sources']['worker'];K.checked(q['sources']['core'])
job=K.read(c['repo']/'audit'/(q['job_name']+'_job.json'));assert job['status']=='RUNNING' and job['pid']==os.getpid()
lock=(c['repo']/'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==pq['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from safetensors import safe_open
import atm_response_contrast_repair_core_20261105_v195 as H
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory)
def guard():
 assert time.monotonic()-started<850 and time.time()<q['absolute_deadline_epoch']
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30 and torch.cuda.max_memory_allocated()<20<<30
 used=K.allocated(P);assert used<384<<20 and K.allocated(out)<q['maximum_output_bytes']
 fs=os.statvfs(P);assert fs.f_bavail*fs.f_frsize>=before['required_including_this_project']+before['own_used_bytes']-used
try:
 for pin in q['models'].values():K.checked(pin)
 K.checked(q['cases']);cases=np.load(q['cases']['path']);K.checked(q['conditions']);conditions=np.load(q['conditions']['path'])['conditions']
 src={k:torch.load(v['path'],map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()}
 state={n:src['NN'][n] if n.startswith('proj_eeg.') else v for n,v in src['CC'].items()}
 mean=torch.load(K.checked(pq['EEG']),map_location='cpu',weights_only=True,mmap=True);visual=torch.load(K.checked(pq['visual_target']),map_location='cpu',weights_only=True,mmap=True)
 ids=tc['T'][(np.random.default_rng(1739).permutation(1323)[:16,None]*10+np.arange(10)).reshape(-1)];x=mean[ids].cuda();y=visual[ids].cuda()
 net=ATMS().cuda();net.load_state_dict(state,strict=True);params=H.prepare_student(net)
 nn=ATMS().cuda().eval().requires_grad_(False);nn.load_state_dict(src['NN'],strict=True)
 with torch.no_grad():a=nn(x,torch.ones(160,dtype=torch.long,device='cuda'));b=net(x,torch.ones(160,dtype=torch.long,device='cuda'))
 cpu=ATMS().double().eval();cpu.load_state_dict({n:v.double() if v.is_floating_point() else v for n,v in state.items()},strict=True);H.prepare_student(cpu)
 with torch.no_grad():cb=cpu(mean[ids].double(),torch.ones(160,dtype=torch.long))
 cond_error=float((b.cpu().double()-cb).abs().max());assert cond_error<3e-5
 h94=c['cv']['bq']['h94'];K.checked(h94);weights={}
 with safe_open(h94['path'],framework='pt',device='cpu') as f:
  for key,name in [('P_weight','image_proj.proj.weight'),('P_bias','image_proj.proj.bias'),('P_gamma','image_proj.norm.weight'),('P_beta','image_proj.norm.bias')]:weights[key]=f.get_tensor(name).half().float().cuda()
  kv={i:{k:f.get_tensor(f'ip_adapter.{i}.{n}.weight').half().float().cuda() for k,n in [('K','to_k_ip'),('V','to_v_ip')]} for i in [1,139]}
 # Known synthetic geometry includes a zero update, deficient rank and no new labels.
 gen=torch.Generator().manual_seed(1729);syn=[]
 for n,d,rank in [(10,19,9),(10,19,3),(4,5,3)]:
  vis=torch.randn(n,rank,generator=gen,dtype=torch.float64)@torch.randn(rank,d,generator=gen,dtype=torch.float64)
  old=torch.randn(7,d,generator=gen,dtype=torch.float64);up=old+torch.randn(7,d,generator=gen,dtype=torch.float64);up[0]=old[0]
  tg,sc,pr=H.targets(old,up,vis,torch.randn(7,d,generator=gen,dtype=torch.float64));assert torch.equal(tg['main'][0],old[0]) and torch.equal(tg['matched_direction'][0],old[0])
  z=up.clone().requires_grad_();ls=H.loss(z,tg['main'],sc);g=torch.autograd.grad(ls,z)[0]
  assert torch.allclose(g,2*(up-tg['main'])/up.numel()/sc,rtol=1e-12,atol=1e-12);syn.append(pr)
 proofs=[];fixture={};gradient_proofs=[];unit_cache=[]
 for qi in [0,79]:
  for ti in [0,3]:
   for li,idx in [(0,1),(69,139)]:
    guard();label=f'q{qi}_t{ti}_l{li}';Q=torch.tensor(cases[label+'__Q'][:,:,[0,-1]],device='cuda').float();w={**weights,**kv[idx]}
    with torch.no_grad():
     old=H.response(a,Q,w);updated=H.response(b,Q,w);teachers=H.response(y,Q,w)
     anchor=H.response(torch.tensor(conditions[qi,0:1],device='cuda').half().float(),Q,w)
    native=cases[label+'__native'][:,:,[0,-1]].reshape(1,-1)
    native_error=float(np.max(abs(anchor.cpu().numpy()-native))/max(1.,float(np.max(abs(native)))));assert native_error<.005
    generator=torch.Generator(device='cuda').manual_seed(1729+len(proofs));random=torch.randn(updated.shape,generator=generator,device='cuda',dtype=torch.float32)
    tg,scale,pr=H.targets(old,updated,teachers,random)
    # Independent CPU NumPy SVD directly on the rectangular teacher matrix.
    D=teachers.cpu().double().numpy();D-=D.mean(0);delta=(updated.double()-old.double()).cpu().numpy();u,s,vt=np.linalg.svd(D,full_matrices=False);keep=s>s.max()*1e-6
    projected=(delta@vt[keep].T)@vt[keep];actual=(tg['main'].double()-old.double()).cpu().numpy()
    svd_error=float(np.linalg.norm(actual-projected)/max(np.linalg.norm(projected),1e-12));assert svd_error<2e-5
    init={name:float(H.loss(updated,t,scale)) for name,t in tg.items()};assert abs(init['main']-init['matched_direction'])<1e-5*max(1.,init['main'])
    proofs.append(dict(label=label,**pr,independent_SVD_relative_error=svd_error,initial_losses=init,Q_positions=2,actual_FP32_response=True,native_FP16_response_relative_error=native_error))
    unit_cache.append((Q,w,tg,scale))
    if not fixture:
     fixture=dict(rows=ids,EEG_first8=mean[ids[:8]].numpy(),NN_condition=a[:8].cpu().numpy(),CN_condition=b[:8].cpu().numpy(),visual_condition=y[:8].cpu().numpy(),Q=Q.cpu().numpy(),old_response=old[:8].cpu().numpy(),updated_response=updated[:8].cpu().numpy(),main_target=tg['main'][:8].cpu().numpy(),control_target=tg['matched_direction'][:8].cpu().numpy(),scale=scale.cpu().numpy())
     # Same full native EEG derivative in independent CPU64 and GPU32 graphs.
     for arm in ['main','matched_direction']:
      net.zero_grad(set_to_none=True);pred=net(x[:8],torch.ones(8,dtype=torch.long,device='cuda'));l=H.loss(H.response(pred,Q,w),tg[arm][:8],scale)
      gp=torch.autograd.grad(l,params,allow_unused=True);cpu.zero_grad(set_to_none=True);cp=cpu(mean[ids[:8]].double(),torch.ones(8,dtype=torch.long));cw={k:v.cpu().double() for k,v in w.items()};ct=tg[arm][:8].cpu().double();cl=H.loss(H.response(cp,Q.cpu().double(),cw),ct,scale.cpu().double());gc=torch.autograd.grad(cl,[p for p in cpu.parameters() if p.requires_grad],allow_unused=True)
      assert [g is None for g in gp]==[g is None for g in gc]
      unused=[n for (n,p),g in zip([(n,p) for n,p in net.named_parameters() if p.requires_grad],gp) if g is None]
      norm=sum(float(g.double().square().sum()) for g in gc if g is not None);error=sum(float((g.cpu().double()-h).square().sum()) for g,h in zip(gp,gc) if g is not None)
      relative=(error/max(norm,1e-30))**.5;assert relative<5e-4
      gradient_proofs.append(dict(arm=arm,CPU64_loss=float(cl),GPU32_loss=float(l),full_trainable_gradient_relative_error=relative,declared_trainable_parameters=sum(p.numel() for p in params),connected_parameters=sum(p.numel() for p,g in zip(params,gp) if g is not None),unused_parameter_names=unused))
      del gp,gc,pred,l,cp,cl
    K.write(out/'progress.json',dict(complete_units=len(proofs),total_units=8,temporary_steps=0,new_fit=False))
 steps=[]
 for arm in ['main','matched_direction']:
  net.load_state_dict(state,strict=True);params=H.prepare_student(net);opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
  for step in range(2):
   guard();Q,w,tg,scale=unit_cache[step];opt.zero_grad(set_to_none=True);pred=net(x,torch.ones(160,dtype=torch.long,device='cuda'));value=H.response(pred,Q,w);loss=H.loss(value,tg[arm],scale);loss.backward()
   norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert np.isfinite(norm) and norm>0;opt.step()
   with torch.no_grad():pafter=net(x,torch.ones(160,dtype=torch.long,device='cuda'));after=H.loss(H.response(pafter,Q,w),tg[arm],scale)
   for n,bf in net.named_buffers():assert torch.equal(bf.cpu(),state[n])
   for n,p in net.named_parameters():
    if n.startswith('proj_eeg.') or n=='logit_scale':assert torch.equal(p.cpu(),state[n]) and p.grad is None
   steps.append(dict(arm=arm,step=step+1,before=float(loss),after=float(after),preclip_norm=norm,condition_change_from_CN=float((pafter-b).norm(dim=1).mean()),BN_and_head_unchanged=True))
  net.load_state_dict(state,strict=True);assert all(torch.equal(v.cpu(),state[n]) for n,v in net.state_dict().items());del opt
 for pin in q['models'].values():K.checked(pin)
 np.savez_compressed(out/'fixture.npz',**fixture);guard()
 K.write(out/'completion.json',dict(status='PASS_TRAINING_ONLY_RESPONSE_CONTRAST_TARGETS_AND_MATCHED_REPAIR_GRADIENTS',synthetic=syn,units=proofs,gradient_proofs=gradient_proofs,temporary_steps=steps,all_steps_discarded=True,all_baselines_unchanged=True,condition_CPU64_max_error=cond_error,fixture=K.ref(out/'fixture.npz'),new_fits=0,new_images=0,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,limits='One training batch160,8stored Q units,2positions/unit; continuousFP32 native weight surrogate. No fullT fit, query-teacher-free inference performance, image efficacy or novelty established. Targets match energy and initial repair distance only, not future optimizer trajectories.'))
except BaseException as exc:K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
