"""Run locked evaluators unchanged with finite process CPU/CUDA guards."""
import argparse,hashlib,json,os,runpy,signal,sys,threading,time
from pathlib import Path
def run():
 p=argparse.ArgumentParser();p.add_argument('--execution',required=True);p.add_argument('--audit-file',required=True);p.add_argument('--script',required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
 e=json.loads(Path(a.execution).read_bytes());s=json.loads(Path(e['protocol']['path']).read_bytes());sys.path.insert(0,s['repository_root'])
 for item in s['source_pins']+e['source_pins']:
  with Path(item['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
 import brainhive_native_models_v1
 import torch
 def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
 base=rss();began=time.monotonic();stop=threading.Event();peak_cpu=[0];peak_gpu=[0];violation=[None];output=Path(a.audit_file);output.parent.mkdir(parents=True,exist_ok=True);assert not output.exists()
 def observe():
  while not stop.wait(.5):
   peak_cpu[0]=max(peak_cpu[0],rss()-base)
   if torch.cuda.is_initialized():peak_gpu[0]=max(peak_gpu[0],torch.cuda.max_memory_allocated())
   if peak_cpu[0]>10<<30 or peak_gpu[0]>20<<30:
    violation[0]='Registered10GiB extraCPU or20GiB allocatedCUDA exceeded'
    with output.with_suffix('.limit.json').open('x') as f:json.dump(dict(status='EVALUATOR_RESOURCE_CAP_EXCEEDED',extra_CPU=peak_cpu[0],CUDA_allocated=peak_gpu[0],reason=violation[0]),f)
    os.kill(os.getpid(),signal.SIGTERM);return
 thread=threading.Thread(target=observe,daemon=True);thread.start();sys.argv=[a.script]+(a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments)
 try:runpy.run_path(a.script,run_name='__main__')
 finally:
  stop.set();thread.join(timeout=2);peak_cpu[0]=max(peak_cpu[0],rss()-base)
  if torch.cuda.is_initialized():peak_gpu[0]=max(peak_gpu[0],torch.cuda.max_memory_allocated())
  result=dict(status='EVALUATOR_RESOURCE_OBSERVATION_COMPLETE',script=a.script,seconds=time.monotonic()-began,maximum_extra_CPU=peak_cpu[0],maximum_CUDA_allocated=peak_gpu[0],violation=violation[0],source_metric_math_unchanged=True)
  with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  assert peak_cpu[0]<=10<<30 and peak_gpu[0]<=20<<30 and violation[0] is None
if __name__=='__main__':run()
