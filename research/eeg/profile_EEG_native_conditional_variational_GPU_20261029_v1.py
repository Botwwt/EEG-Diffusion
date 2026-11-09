"""Real pretrained-prior/real-EEG two-arm engineering; temporary updates reset."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,gc,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(x):p=Path(x['path']);assert ref(p)==x;return p
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def digest(state):
 import torch
 h=hashlib.sha256()
 for key,value in state.items():
  t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def run(sp,job):
 s=json.loads(Path(sp).read_bytes());assert s['status']=='REGISTERED_NATIVE_CONDITIONAL_VARIATIONAL_REAL_ENGINEERING_v1' and s['arm'] in ('EEG_coupled_teacher','visual_only_teacher')
 for pin in s['source_pins']:check(pin)
 out=Path(s['output_root']);assert not out.exists();out.mkdir();started=time.monotonic();base=peak();phase='EXCLUSIVE_WRAPPER_AND_REAL_T_INPUTS'
 try:
  expected='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
  assert expected in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode().replace('\0',' ')
  receipt=Path(s['repository_root'])/'audit'/(job+'_job.json')
  for _ in range(30):
   if receipt.is_file():break
   time.sleep(.1)
  r=json.loads(receipt.read_bytes());assert r['pid']==os.getpid() and r['status']=='RUNNING'
  assert float(r['max_seconds'] if s['host']=='old' else r['maximum_seconds'])==900
  assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  sys.path.insert(0,s['training_code']);import AR_anchored_training_contract_CPU_20261029_v1 as C
  a,_=C.load_admission(s['training_admission']);C.validate_sources(a);C.verify_protected_files(a)
  import numpy as np
  import torch
  from PIL import Image
  from EEG_generative_backend_loader_20261029_v1 import build_and_load
  from eeg_labram_pretrained_20261028_v1 import build_encoder
  from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform
  from EEG_native_conditional_variational_core_20261029_v1 import NativeConditionDecoder,EEGGaussianPrior,ImageVariationalBank,native_moments,diagonal_KL,token_ELBO,reparameterize
  torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  identities=json.loads(check(a['data']['EEG_identities']).read_bytes());tid=json.loads(check(a['data']['token_identities']).read_bytes());C.identity_join(identities['T'],tid)
  raw=np.load(check(s['T_mean63']),mmap_mode='r',allow_pickle=False);targets=np.load(check(a['data']['tokens']),mmap_mode='r',allow_pickle=False);assert raw.shape==(13230,63,250) and targets.shape==(13230,680)
  positions=s['fixed_T_positions'];assert positions==[0,1,2,3];role=s['full_EEG'];stats=json.loads(check(role['normalization']).read_bytes());assert stats['role']=='full' and stats['training_positions']==list(range(13230))
  transform=CacheNativeEEGTransform(tuple(stats['channel_names']),np.array(stats['center']),np.array(stats['scale']));eeg_np=transform(np.array(raw[positions]),channel_names=tuple(stats['channel_names']))
  def guard():
   assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
   assert shutil.disk_usage(out).free>=2<<30
  torch.cuda.reset_peak_memory_stats();phase='UNCHANGED_PUBLIC_VAR_NATIVE_CLASS_COORDINATES'
  vae,d,proof=build_and_load(a['author_source'],a['vendor'],check(a['public_VQ']),check(a['public_VAR']),seed=1729)
  var_state=torch.load(check(a['public_VAR']),weights_only=True,map_location='cpu',mmap=True);classes=var_state['class_emb.weight'].clone();del var_state
  assert classes.shape==(1001,1024);before=digest({k:v for k,v in d.state_dict().items() if not k.startswith('embed_proj.')});VQ_before=digest(vae.state_dict())
  native=NativeConditionDecoder(d,classes);assert digest(d.state_dict())==before
  native.cuda().eval();vae.cuda().eval().requires_grad_(False);d.rng=torch.Generator(device='cuda');cent,scale=native_moments(classes)
  target=torch.tensor(np.array(targets[positions]),device='cuda',dtype=torch.long);widths=[x*x for x in (1,2,3,4,5,6,8,10,13,16)]
  with torch.inference_mode():prefix=vae.quantize.idxBl_to_var_input(list(torch.split(target,widths,dim=1)));z=classes[[0,11,287,1000]].cuda();reference=d(prefix,z);actual=native(prefix,z)
  max_error=float((actual-reference).abs().max());assert torch.allclose(actual,reference,atol=2e-5,rtol=2e-5)
  with torch.inference_mode():image=native.generate(None,seed=1729,guidance=0.,top_k=600);replay=d.autoregressive_infer_cfg(1,classes[-1:].cuda(),g_seed=1729,cfg=0.,top_k=600,top_p=0.,more_smooth=False)
  assert torch.equal(image,replay),'Native null-class function was not preserved'
  pixels=image[0].mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy();Image.fromarray(pixels,mode='RGB').save(out/'native_null_engineering.png');del actual,reference,z,image,replay;guard()
  phase='REAL_FULL_T_EEG_BACKBONE_AND_SINGLE_ELBO_GRADIENTS'
  encoder=build_encoder(a['author_source'],a['vendor'],seed=1729);checkpoint=torch.load(check(role['model']),weights_only=True,map_location='cpu');assert checkpoint['role']=='full' and checkpoint['epoch']==50 and len(checkpoint['model'])==224
  encoder.load_state_dict(checkpoint['model'],strict=True);del checkpoint
  p=EEGGaussianPrior(encoder,cent,scale).cuda().eval();q=ImageVariationalBank(4,cent,scale).cuda();from models.labram import get_input_chans
  channels=get_input_chans([x.upper() for x in stats['channel_names']]);x=torch.from_numpy(eeg_np).cuda();ids=torch.arange(4,device='cuda')
  p_initial={k:v.detach().cpu().clone() for k,v in p.state_dict().items()};q_initial={k:v.detach().cpu().clone() for k,v in q.state_dict().items()};p_digest=digest(p.state_dict());q_digest=digest(q.state_dict())
  op=torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=1e-4,weight_decay=.01);oq=torch.optim.Adam(q.parameters(),lr=.03);rng=torch.Generator(device='cuda').manual_seed(1729)
  teacher_grad=[];EEG_grad=[];backbone_grad=[];events=[];loop=time.monotonic()
  for round_index in range(3):
   for inner in range(2):
    guard();oq.zero_grad(set_to_none=True);p.zero_grad(set_to_none=True);qm,qs=q(ids)
    with torch.no_grad():pm,ps=p(x,input_chans=channels)
    if s['arm']=='visual_only_teacher':pm=cent.cuda().expand_as(qm);ps=scale.cuda().log().expand_as(qs)
    sample=reparameterize(qm,qs,generator=rng);logits=native(prefix,sample);loss,parts=token_ELBO(logits,target,qm,qs,pm.detach(),ps.detach());loss.backward()
    grad=float(q.mean_delta.grad.norm());assert grad>0 and all(v.grad is None for v in p.parameters()) and all(v.grad is None for v in native.parameters());teacher_grad.append(grad)
    torch.nn.utils.clip_grad_norm_(q.parameters(),10.);oq.step();events.append(dict(round=round_index+1,step='image_posterior',inner=inner+1,ELBO_nats_per_token=float(loss.detach()),token_NLL_nats_per_query=float(parts['token_NLL_nats_per_query'].mean()),KL_nats_per_query=float(parts['KL_nats_per_query'].mean()),gradient_norm=grad));del logits,loss,sample,qm,qs,pm,ps
   for inner in range(2):
    guard();op.zero_grad(set_to_none=True);q.zero_grad(set_to_none=True)
    with torch.no_grad():qm,qs=q(ids)
    pm,ps=p(x,input_chans=channels);loss=diagonal_KL(qm.detach(),qs.detach(),pm,ps).mean()/680;loss.backward();gradient=float(p.encoder.head.weight.grad.norm());assert gradient>0 and all(v.grad is None for v in q.parameters())
    backbone=sum(float(v.grad.square().sum()) for n,v in p.encoder.named_parameters() if v.grad is not None and not n.startswith('head.') and not n.startswith('logit_scale'))**.5
    EEG_grad.append(gradient);backbone_grad.append(backbone);torch.nn.utils.clip_grad_norm_(p.parameters(),1.);op.step();events.append(dict(round=round_index+1,step='EEG_conditional_prior',inner=inner+1,KL_nats_per_token=float(loss.detach()),head_gradient_norm=gradient,backbone_gradient_norm=backbone));del loss,qm,qs,pm,ps
  torch.cuda.synchronize();loop_seconds=time.monotonic()-loop;assert max(backbone_grad)>0
  assert digest(d.state_dict())==before and digest(vae.state_dict())==VQ_before and all(v.grad is None for v in native.parameters())
  p.load_state_dict(p_initial,strict=True);q.load_state_dict(q_initial,strict=True);op.state.clear();oq.state.clear();p.zero_grad(set_to_none=True);q.zero_grad(set_to_none=True)
  assert digest(p.state_dict())==p_digest and digest(q.state_dict())==q_digest and all(v.grad is None for v in p.parameters()) and all(v.grad is None for v in q.parameters())
  C.verify_protected_files(a);guard();result=dict(status='PASS_REAL_PRETRAINED_NATIVE_PRIOR_AND_REAL_EEG_CONDITIONAL_ELBO_ENGINEERING_ONLY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(sp),host=s['host'],arm=s['arm'],
   native_public_VAR=proof,all_pretrained_VAR_and_VQ_tensors_preserved=True,random_EEG_adapter_removed=True,native_class_table_shape=list(classes.shape),native_identity_forward_max_error=max_error,native_null_pixels_exact_author=True,
   fixed_T_positions=positions,real_T_queries=4,normalization='Full-T statistics; no Dfit/Dcheck/V EEG or images read',image_posterior_temporary_updates=6,EEG_prior_temporary_updates=6,
   all_temporary_EEG_and_posterior_updates_reset=True,all_optimizer_states_cleared=True,existing_EEG_backbone_gradient_nonzero=True,EEG_predictor_parameters=sum(v.numel() for v in p.parameters()),
   frozen_native_generator_parameters=sum(v.numel() for v in native.parameters()),training_only_posterior_parameters=sum(v.numel() for v in q.parameters()),events=events,
   loop_seconds=loop_seconds,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,
   formal_trained_models=0,new_research_images=0,engineering_images=1,method_benefit_or_SOTA_established=False)
  write(out/'completion.json',result)
 except BaseException as error:
  write(out/'failure.json',dict(status='NEW_ROUTE_NATIVE_VARIATIONAL_ENGINEERING_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,formal_trained_models=0));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--job-name',required=True);v=p.parse_args();run(v.spec,v.job_name)
