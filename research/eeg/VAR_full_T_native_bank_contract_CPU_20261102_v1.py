"""Complete13230x1000x10 bank; unused CLOSED mean budget explicitly released."""
import hashlib,importlib.util,json,os,shutil
from pathlib import Path
B=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
CODE=B/'VAR_full_T_native_bank_code_20261102_v1';OUT=B/'VAR_full_T_native_bank_20261102_v1'
PREVIOUS=dict(path=(B/'VAR_declared_FP16_code_20261102_v1/protocol.json').as_posix(),bytes=6702,sha256='f536a2b087317d1f49c6d4a124c2a55a6f8284f9091e72f24661f8f71f6671d0')
GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p);assert p.is_file() and not p.is_symlink();a=p.stat()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 b=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
 return dict(path=str(p),bytes=b.st_size,sha256=h)
def checked(p):assert ref(p['path'])==p;return Path(p['path'])
def write(p,x,update=False):
 p=Path(p);assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 if not update:
  with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 else:
  t=p.with_name(p.name+'.pending');assert not t.exists()
  with t.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
def previous():
 q=read(checked(PREVIOUS));pin=next(v for v in q['source_pins'] if Path(v['path']).name=='VAR_declared_FP16_contract_CPU_20261102_v1.py');sp=importlib.util.spec_from_file_location('closed_FP16_reference_contract',checked(pin));c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c);c.setup(PREVIOUS['path']);return c,q
def process(pid,ticks):
 try:
  v=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(PID=pid,start_ticks=int(v[19]),state=v[0],exact_live=int(v[19])==ticks and v[0] not in ('Z','X'))
 except FileNotFoundError:return dict(PID=pid,present=False,exact_live=False)
def part_range(index):
 if type(index) is not int or not 0<=index<207:raise ValueError('A fixed complete bank segment required')
 return index*64,min((index+1)*64,13230)
def validate(q):
 assert q['status']=='REVIEWED_COMPLETE_T13230_NATIVE_1000_FULL680_BANK_DECLARED_FP16'
 assert q['host']=='old' and q['GPU_UUID']==GPU and q['code_root']==CODE.as_posix() and q['output_root']==OUT.as_posix()
 assert q['previous_protocol']==PREVIOUS and q['seed']==1729
 assert (q['rows'],q['concepts'],q['classes'],q['tokens'],q['scales'],q['part_rows'],q['parts'],q['class_batch'])==(13230,1323,1000,680,10,64,207,64)
 assert q['precision']=='AMP_FP16_decoder_only_FP32_weights_VQ_logsoftmax_FP64_sums_no_TF32'
 assert q['maximum_new_output_bytes']==1400<<20 and q['maximum_GPU_bytes']==20<<30 and q['maximum_extra_CPU_bytes']==10<<30
 assert q['minimum_start_CPU_headroom_bytes']==24<<30 and q['minimum_runtime_CPU_headroom_bytes']==10<<30
 assert q['GPU_hard_seconds']==900 and q['GPU_soft_seconds']==780 and q['maximum_execution_seconds']==48*3600 and q['CPU_seconds']==300
 assert q['release_only_closed_mean_remaining_budget'] and q['preserve_legacy50GiB_buffer8GiB_and_other_promises']
 assert q['new_optimizer_updates']==q['new_research_images']==0 and not q['EEG_fit_admitted'] and q['no_automatic_retry']
 assert len(q['source_pins'])==4 and all(Path(v['path']).parent==CODE for v in q['source_pins']);return q
def dependency(q,require_sources=True):
 c,prev=previous();n,oldgate=c.dependency(prev);assert not process(524840,119778811)['exact_live']
 done=read(c.OUT/'completion.json');assert done['status']=='COMPLETE_DECLARED_FP16_FULL_REFERENCE_AND_MATCHING_UNTRUNCATED_PROFILE'
 audit=read(checked(done['CPU_audit']));profile=read(checked(done['profile']))
 assert audit['predeclared_precision_fidelity_support'] and audit['all_new_token_terms']==5440000 and profile['all_weights_byte_unchanged']
 if require_sources:
  for pin in q['source_pins']:checked(pin)
 return c,n,prev,dict(FP16_completion=ref(c.OUT/'completion.json'),precision_audit=done['CPU_audit'],measured_full_loop_seconds=profile['loop_seconds'],same_loop_full13230_projection_hours=audit['full13230_same_loop_projection_hours'],closed_mean_dependency=oldgate)
def resources(q,starting=False,require_sources=True):
 # Recompute the old reservation formula; do not temporarily suppress or edit old guards.
 c,n,prev,proof=dependency(q,require_sources=require_sources);m=n.mean_contract();mean=m.validate(read(n.MEAN_PROTOCOL['path']));control=read(n.CONTROL_PROTOCOL['path'])
 meanused=m.allocated(m.OUT)+sum(m.allocated(Path(r)) for r in control['additional_study_roots']);assert meanused<=4<<30
 # Canonical full4000+all16 completion and independent full128k audit are checked by dependency.
 # No further writes are promised for that closed study. Existing bytes remain on disk.
 parent=read(checked(m.PARENT));parentremaining=max(0,(8<<30)-m.allocated(parent['output_root']))
 oldnative=[B/f'VAR_multiscale_native_profile{k}_20261101_v{v}' for v in (1,2,3) for k in ('_code','')]
 oldnative += [B/'VAR_full_native_throughput_top0_code_20261101_v1',B/'VAR_full_native_throughput_top0_20261101_v1']
 nativeused=sum(m.allocated(p) for p in oldnative);assert nativeused<=128<<20
 fp16used=m.allocated(c.CODE)+m.allocated(c.OUT);assert fp16used<=32<<20
 own=m.allocated(CODE)+m.allocated(OUT);assert own<=q['maximum_new_output_bytes']
 required=(50<<30)+(8<<30)+(14<<30)+(64<<20)+(32<<20)+parentremaining+(128<<20)-nativeused+(32<<20)-fp16used+q['maximum_new_output_bytes']-own
 free=shutil.disk_usage(B).free;head=m.headroom();floor=q['minimum_start_CPU_headroom_bytes' if starting else 'minimum_runtime_CPU_headroom_bytes'];assert free>=required and head>=floor
 return dict(free_bytes=free,required_remaining_bytes=required,effective_CPU_headroom_bytes=head,bank_allocated_bytes=own,closed_mean_allocated_bytes=meanused,released_unused_closed_mean_bytes=(4<<30)-meanused,closed_mean_no_future_writes=True,legacy50GiB_buffer8GiB_and_other_promises_preserved=True,physical_reservation=False)
def setup(p):
 assert Path(p)==CODE/'protocol.json';q=validate(read(p));dependency(q);r=read(CODE/'actual_execution_review.json')
 assert r['status']=='PASS_COMPLETE_NATIVE_BANK_SCOPE_SOURCE_RESOURCE_REVIEW' and r['protocol']==ref(p) and r['source_pins']==q['source_pins'];return q
def quick_checks():
 import numpy as np
 ranges=[part_range(k) for k in range(207)];assert ranges[0]==(0,64) and ranges[-1]==(13184,13230)
 assert sum(hi-lo for lo,hi in ranges)==13230 and all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
 assert 13230*1000*10*8+13230*1000<1400<<20
 x=np.arange(680,dtype=np.float32)/-17.;v=0.
 for a in x:v+=float(a)
 assert v==x.astype(np.float64).sum()
 return dict(status='PASS_COMPLETE_BANK_PARTITION_STORAGE_AND_FP64_REDUCTION_CHECKS_ONLY',checks=3)
