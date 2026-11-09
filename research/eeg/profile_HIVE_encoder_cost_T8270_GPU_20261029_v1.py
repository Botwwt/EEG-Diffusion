"""Standalone T-only cost measurement; no official admission or final model.

Reuses public native training/export math, not the gated official entry/token.
The transient head and overlapping engineering exports cannot enter research
generation, official OOF banks, or any subsequent final training.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def check(item):assert ref(item['path'])==item;return Path(item['path'])
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def run(path):
 s=json.loads(Path(path).read_bytes());assert s['status']=='REGISTERED_STANDALONE_T_ONLY_NATIVE_ENCODER_COST_PROFILE' and s['host']=='old'
 for item in s['source_pins']+s['payloads']:check(item)
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 root=Path(s['output_root']);root.mkdir();started=time.monotonic();base=memory();phase='LOAD_RUNTIME'
 sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
 import brainhive_native_models_v1 as N
 import brainhive_official_encoder_training_v1 as T
 import profile_BrainHIVE_official_encoder_GPU_v1 as P
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;T.seed_everything(torch)
 def budget(*_):
  assert time.monotonic()-started<800 and memory()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
  assert shutil.disk_usage(root).free>2<<30
  assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<256<<20
 try:
  split=json.loads(check(s['split']).read_bytes());rows=P.training_rows(split);assert rows==s['training_array_rows'] and len(rows)==8270
  from brainhive_functional_training_contract_v2 import resource_headroom
  headroom=resource_headroom();assert headroom['effective_headroom_bytes']>=8<<30
  torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
  phase='LOAD_REAL_T_PAYLOADS';tick=time.monotonic();scope=N.classes();priors=N.prior_models('cuda');teacher=priors['fusion_encoder'].vision_model
  assert sum(p.numel() for p in teacher.parameters())==11015168;T.frozen_stochastic_teacher(torch,teacher);del priors
  raw=torch.load(check(s['EEG']),weights_only=True,map_location='cpu',mmap=True);visual=torch.load(check(s['visual']),weights_only=True,map_location='cpu',mmap=True)
  eeg,visuals,tensor_report=P.select_T_tensors(torch,raw,visual,rows,dict(teacher.input_meta));eeg=eeg.cuda();visuals={k:v.cuda() for k,v in visuals.items()};del visual;gc.collect();torch.cuda.empty_cache();load_seconds=time.monotonic()-tick
  def state_sha(module):return P.state_sha({k:v.detach().cpu() for k,v in module.state_dict().items()})
  teacher_before=state_sha(teacher);T.seed_everything(torch);model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda();assert sum(p.numel() for p in model.parameters())==6452224
  logit=torch.nn.Parameter(torch.tensor(2.6592,device='cuda'));optimizer,scheduler=T.make_optimizer(torch,model,logit,225);shuffle=torch.Generator(device='cpu').manual_seed(1730);budget();phase='ONE_TRANSIENT_NATIVE_T_EPOCH';tick=time.monotonic()
  event,updates=T.run_epoch(torch,model,teacher,logit,optimizer,scheduler,shuffle,eeg,visuals,epoch=1,updates=0,total_updates=225,batch_size=1024,subject='sub-01',role='T_only_engineering_cost',device=torch.device('cuda'),deadline=started+800,autocast_bfloat16=True,observer=budget)
  torch.cuda.synchronize();train_seconds=time.monotonic()-tick;assert updates==9 and teacher_before==state_sha(teacher)
  state=T.snapshot(torch,model,logit,optimizer,scheduler,shuffle,1,updates,ref(path)['sha256'],s['registration_core_sha256'],ref(__file__)['sha256'],torch.device('cuda'))
  target=root/'transient_engineering_epoch1.pt';torch.save(state,target);assert set(int(v['step']) for v in state['optimizer']['state'].values())=={9}
  del eeg,visuals,teacher;gc.collect();torch.cuda.empty_cache();phase='REAL8270_RAW_LN_GPU_EXPORT';tick=time.monotonic()
  condition,export=P.export_T_conditions(torch,model,raw,rows,state['model'],device=torch.device('cuda'),deadline=started+800);torch.cuda.synchronize();export_seconds=time.monotonic()-tick;budget()
  phase='INDEPENDENT_ALL8270_CPU_FORWARD';tick=time.monotonic();cpu=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);cpu.load_state_dict(state['model'],strict=True);parts=[]
  with torch.inference_mode():
   for lo in range(0,8270,256):parts.append(cpu(raw[rows[lo:lo+256]][:,46:63,:]).float())
  cpu_condition=torch.cat(parts);max_abs=float((cpu_condition-condition).abs().max());relative=float((cpu_condition-condition).norm()/condition.norm());assert max_abs<1e-3 and relative<1e-5
  cpu_seconds=time.monotonic()-tick;assert all(p.grad is None for p in cpu.parameters()) and ref(s['ordinary']['path'])==s['ordinary'];budget()
  result=dict(status='PASS_REAL_T8270_NATIVE_ONE_EPOCH_AND_ALL_ROW_EXPORT_COST_PROFILE',UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),protocol=ref(path),source=ref(__file__),subject='sub-01',unique_T_rows=8270,engineering_epochs=1,engineering_updates=9,scheduler_horizon_updates=225,parameters=6452224,transient_snapshot=ref(target),epoch_event=event,tensor_report=tensor_report,export_report=export,load_seconds=load_seconds,train_seconds=train_seconds,export_seconds=export_seconds,CPU_forward_seconds=cpu_seconds,CPU_GPU_max_absolute=max_abs,CPU_GPU_relative=relative,frozen_teacher_state_unchanged=True,ordinary_checkpoint_unchanged=True,extra_CPU_peak=memory()-base,peak_GPU_allocated=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,resources_before=headroom,output_cap_bytes=256<<20,whole_official_lifecycle_admitted=False,formal_OOF_fit_or_export=False,conditions_saved_to_disk=False,new_research_images=0,new_research_method_configurations=0,temporary_weights_must_not_be_reused=True,old_official_gates_unchanged=True)
  write(root/'completion.json',result);print(json.dumps({k:result[k] for k in ('status','engineering_updates','train_seconds','export_seconds','CPU_GPU_relative','peak_GPU_allocated','seconds')}))
 except BaseException as error:write(root/'failure.json',dict(status='T_ONLY_ENCODER_COST_PROFILE_FAILED_RETAINED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();run(a.protocol)
