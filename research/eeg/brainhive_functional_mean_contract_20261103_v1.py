"""Source-bound OLD-only contract for one closed-form calibration experiment."""
import datetime,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path

GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
STATUS='REGISTERED_FUNCTIONAL_SHARED_MEAN_T_ONLY_STATS'

def read(path):return json.loads(Path(path).read_bytes())
def ref(path):
 p=Path(path)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(pin):
 actual=ref(pin['path']);assert all(actual[k]==v for k,v in pin.items()),pin
 return Path(pin['path'])
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
 temporary=p.with_name(p.name+'.pending');assert not temporary.exists()
 with temporary.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,p)
def rss():return int(next(v.split()[1] for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('VmHWM:')))*1024
def validate(s):
 assert s['status']==STATUS and s['host']=='old' and s['GPU_UUID']==GPU and s['seed']==1729
 assert s['maximum_output_bytes']==1<<30 and s['maximum_extra_CPU_bytes']==10<<30 and s['maximum_GPU_bytes']==20<<30
 assert s['legacy_floor_bytes']==50<<30 and s['disk_buffer_bytes']==8<<30
 assert s['T_queries']==132300 and s['T_images']==13230 and s['OOF_roles']==20 and s['statistic_batch']==256
 assert s['four_future_image_arms']==['native_shared_mean','ordinary_shared_mean','ordinary_full_mean_matched','ordinary_wrong_field']
 assert s['strength']==1. and s['no_strength_or_seed_scan'] and s['ordinary_immutable'] and not s['NEW_contacted']
 for pin in s['source_pins']:checked(pin)
 for pin in s['input_pins'].values():checked(pin)
 assert s['ordinary']['sha256']=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03';checked(s['ordinary'])
 assert Path(s['output_root']).name=='HIVE_functional_mean_calibration_20261103_v1'
 return s
def setup(path):
 s=validate(read(path));sys.path.insert(0,s['repository_root'])
 import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
 return s
def resources(s,start=False):
 protocol=checked(s['resource_protocol']);p=read(protocol)
 pin=next(v for v in p['source_pins'] if Path(v['path']).name=='EEG_complete_native_study_contract_CPU_20261102_v1.py')
 spec=importlib.util.spec_from_file_location('functional_existing_commitments',checked(pin));C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C);C.setup(str(protocol))
 external=C.resources(p);root=Path(s['output_root']);used=0
 if root.exists():
  for child in root.rglob('*'):
   assert not child.is_symlink()
   if child.is_file():used+=child.stat().st_blocks*512
 assert used<=s['maximum_output_bytes']
 own_remaining=max(0,s['maximum_output_bytes']-used)
 assert external['free_bytes']>=external['required_remaining_bytes']+own_remaining
 assert external['effective_CPU_headroom_bytes']>=(24 if start else 10)<<30
 return dict(external=external,own_used_bytes=used,own_remaining_budget=own_remaining,
             required_with_all_prior_commitments=external['required_remaining_bytes']+own_remaining,not_physical_reservation=True)
def wrapper(s,name):
 parent=Path('/proc',str(os.getppid()),'cmdline').read_bytes().decode()
 assert 'run_gpu_job_v2.py' in parent and name in parent
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==GPU
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 return dict(parent_PID=os.getppid(),own_PID=os.getpid(),GPU_UUID=GPU,job_name=name)
def budget(s,started,max_seconds,base_rss,torch=None,refresh=False):
 assert time.monotonic()-started<max_seconds-30 and rss()-base_rss<s['maximum_extra_CPU_bytes']
 if torch is not None:assert torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes']
 if refresh:resources(s)
