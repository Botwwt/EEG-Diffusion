"""Declared AMP decoder probability kernel, separate from the sealed FP32 kernel."""
import hashlib,importlib.util,json,os,shutil
from pathlib import Path
B=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
CODE=B/'VAR_declared_FP16_code_20261102_v1';OUT=B/'VAR_declared_FP16_profile_20261102_v1'
PREVIOUS=dict(path=(B/'VAR_full_native_throughput_top0_code_20261101_v1/protocol.json').as_posix(),bytes=6340,sha256='4ff364435b51c88eee56176c1f5116f6f3e3717965f1bde2aac8d6844bc7120d')
GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p);assert p.is_file() and not p.is_symlink();a=p.stat()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 b=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
 return dict(path=str(p),bytes=b.st_size,sha256=h)
def checked(p):assert ref(p['path'])==p;return Path(p['path'])
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def previous():
 q=read(checked(PREVIOUS));pin=next(p for p in q['source_pins'] if Path(p['path']).name=='VAR_full_native_throughput_contract_CPU_20261101_v1.py')
 sp=importlib.util.spec_from_file_location('sealed_FP32_dependency',checked(pin));c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
 c.setup(Path(PREVIOUS['path']));return c,q
def process(pid,ticks):
 try:
  x=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(PID=pid,start_ticks=int(x[19]),state=x[0],exact_live=int(x[19])==ticks and x[0] not in ('Z','X'))
 except FileNotFoundError:return dict(PID=pid,present=False,exact_live=False)
def validate(q):
 assert q['status']=='DECLARED_AMP_FP16_DECODER_FP32_VQ_SOFTMAX_COMPLETE_REFERENCE_PROFILE'
 assert q['host']=='old' and q['GPU_UUID']==GPU and q['code_root']==CODE.as_posix() and q['output_root']==OUT.as_posix()
 assert q['previous_protocol']==PREVIOUS and q['seed']==1729 and q['class_batch']==64 and q['T_positions']==[0,1,1889,3779,5669,7559,9449,11339]
 assert q['precision']=='AMP_FP16_decoder_only_FP32_weights_VQ_logsoftmax_FP64_sums_no_TF32'
 assert (q['classes'],q['tokens'],q['CFG'],q['top_k'])==(1000,680,0,0)
 assert q['scales']==[1,2,3,4,5,6,8,10,13,16]
 assert q['fidelity_guard']==dict(max_token_logprob_error=.25,max_class_mean_abs_token_error=.02,max_HMM_posterior_TV=.1,all8_global_argmax_same=True,max_full_vs_prefix_logprob_error=.03)
 assert q['maximum_new_output_bytes']==32<<20 and q['maximum_GPU_bytes']==20<<30 and q['maximum_extra_CPU_bytes']==10<<30
 assert q['GPU_hard_seconds']==900 and q['GPU_soft_seconds']==780 and q['CPU_seconds']==300 and q['maximum_execution_seconds']==1800
 assert q['new_optimizer_updates']==q['new_research_images']==0 and not q['full_bank_or_EEG_fit_admitted'] and q['no_automatic_retry']
 assert len(q['source_pins'])==5 and all(Path(v['path']).parent==CODE for v in q['source_pins']);return q
def dependency(q,require_sources=True):
 c,old=previous();n,gate=c.dependency(old)
 for pid,t in [(524025,119622236)]:assert not process(pid,t)['exact_live']
 done=read(c.OUT/'completion.json');assert done['status']=='COMPLETE_FULL_NATIVE_FP32_BATCH64_AND_MATCHING_TOP0_ENGINEERING_ONLY'
 for pin in (done['profile'],done['CPU_audit']):checked(pin)
 if require_sources:
  for pin in q['source_pins']:checked(pin)
 return n,dict(previous_completion=ref(c.OUT/'completion.json'),old_exact_handles_terminal=True)
def resources(q,starting=False):
 c,old=previous();r=c.resources(old,starting=starting);n=c.native_contract();m=n.mean_contract()
 used=m.allocated(CODE)+m.allocated(OUT);assert used<=q['maximum_new_output_bytes']
 required=r['required_remaining_bytes']+q['maximum_new_output_bytes']-used;free=shutil.disk_usage(B).free
 assert free>=required
 return dict(free_bytes=free,required_remaining_bytes=required,new_used_bytes=used,all_old_promises_retained=True,effective_CPU_headroom_bytes=r['effective_CPU_headroom_bytes'])
def setup(p):
 assert Path(p)==CODE/'protocol.json';q=validate(read(p))
 for pin in q['source_pins']:checked(pin)
 r=read(CODE/'actual_execution_review.json');assert r['status']=='PASS_DECLARED_FP16_SCOPE_SOURCE_AND_RESOURCE_REVIEW' and r['protocol']==ref(p) and r['source_pins']==q['source_pins']
 dependency(q);return q
def quick_checks():
 import numpy as np
 # These are arithmetic/source-bound checks, not a pretrained precision test.
 x=np.array([-10001.,-10000.,-9999.]);p=np.exp(x-x.max());p/=p.sum();y=x+333.;r=np.exp(y-y.max());r/=r.sum();assert np.array_equal(p,r)
 assert sum(min(64,1000-a) for a in range(0,1000,64))==1000
 a=np.arange(680,dtype=np.float32)/-17.;s=0.
 for v in a:s+=float(v)
 assert s==a.astype(np.float64).sum()
 return dict(status='PASS_NARROW_FP16_SCOPE_AND_INDEPENDENT_ARITHMETIC_ONLY',checks=3)
