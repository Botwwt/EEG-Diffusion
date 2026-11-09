"""One FP32 execution comparison and matching top0 replay, no EEG admission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

B=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
CODE=B/'VAR_full_native_throughput_top0_code_20261101_v1'
OUT=B/'VAR_full_native_throughput_top0_20261101_v1'
NATIVE=dict(path=(B/'VAR_multiscale_native_profile_code_20261101_v3/protocol.json').as_posix(),bytes=44554,
    sha256='00279f7b4e8a0dee658a2c0c7352fd53692b639d1d9f21615d8b6166340fc891')
GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'


def read(path):return json.loads(Path(path).read_bytes())
def ref(path):
    p=Path(path);assert p.is_file() and not p.is_symlink();a=p.stat()
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    b=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
    return dict(path=str(p),bytes=a.st_size,sha256=sha)
def checked(pin):assert ref(pin['path'])==pin;return Path(pin['path'])
def write(path,value):
    p=Path(path);assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def process(pid,ticks):
    try:
        v=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(PID=pid,start_ticks=int(v[19]),state=v[0],exact_live=int(v[19])==ticks and v[0] not in ('Z','X'))
    except FileNotFoundError:return dict(PID=pid,present=False,exact_live=False)
def native_contract():
    s=read(checked(NATIVE));pin=next(p for p in s['source_pins'] if Path(p['path']).name=='VAR_multiscale_profile_contract_CPU_20261101_v1.py')
    spec=importlib.util.spec_from_file_location('completed_native_before_throughput',checked(pin));n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
    n.setup(Path(NATIVE['path']));return n


def validate(q):
    assert q['status']=='REVIEWED_FULL_T8_K1000_FP32_BATCH64_AND_TOP0_ENGINEERING'
    assert q['host']=='old' and q['GPU_UUID']==GPU and q['code_root']==CODE.as_posix() and q['output_root']==OUT.as_posix()
    assert q['native_protocol']==NATIVE and q['seed']==1729
    assert q['class_batch']==64 and q['reference_class_batch']==8
    assert q['T_positions']==[0,1,1889,3779,5669,7559,9449,11339]
    assert q['classes']==1000 and q['tokens']==680 and q['scales']==[1,2,3,4,5,6,8,10,13,16]
    assert q['precision']=='FP32_no_autocast_no_TF32' and q['CFG']==0 and q['top_k']==0
    assert q['numerical_guard']==dict(atol=1e-4,rtol=2e-5)
    assert q['posterior_max_abs_guard']==1e-3 and q['same_global_argmax_required']
    assert q['maximum_new_output_bytes']==25<<20 and q['shared_native_output_bytes']==128<<20
    assert q['maximum_GPU_bytes']==20<<30 and q['maximum_extra_CPU_bytes']==10<<30
    assert q['minimum_start_CPU_headroom_bytes']==24<<30 and q['minimum_runtime_CPU_headroom_bytes']==10<<30
    assert q['GPU_hard_seconds']==900 and q['GPU_soft_seconds']==780 and q['CPU_seconds']==300
    assert q['maximum_execution_seconds']==1800 and q['new_optimizer_updates']==q['new_research_images']==0
    assert q['full_T_bank_or_EEG_fit_admitted'] is False and q['no_automatic_retry']
    assert len(q['source_pins'])==5 and all(Path(p['path']).parent==CODE for p in q['source_pins'])
    return q


def dependency(q,require_sources=True):
    n=native_contract();s=read(NATIVE['path'])
    done=read(checked(q['reference']['native_completion']))
    assert done['status']=='COMPLETE_T8_K1000_680_NATIVE_ENGINEERING_AND_INDEPENDENT_CPU_ONLY' and done['protocol']==NATIVE
    mean=read(checked(q['reference']['mean_CPU_completion']))
    assert mean['status']=='COMPLETE_WHOLE_MEAN_INDEPENDENT_OLD_SERVER_CPU_AUDIT'
    assert mean['numeric_scope']['actual_defined_paired_cells']==128000 and not mean['CUDA_initialized']
    handles=[(519201,118882540),(518175,118747637),(523375,119484865),(523376,119484873)]
    states=[process(*pair) for pair in handles]
    assert not any(v['exact_live'] for v in states),'Same completed owned handle remains live; no new launch.'
    for pin in q['reference'].values():checked(pin)
    if require_sources:
        for pin in q['source_pins']:checked(pin)
    return n,dict(status='READY_PREVIOUS_FULL_NATIVE_AND_MEAN_AUDITS_CLOSED_EXACT_HANDLES',handles=states)


def storage_requirement(legacy_remaining,original_used,new_used,new_cap,shared_cap):
    assert all(isinstance(v,int) and v>=0 for v in (legacy_remaining,original_used,new_used,new_cap,shared_cap))
    assert new_used<=new_cap and original_used+new_used<=shared_cap
    return legacy_remaining+shared_cap-original_used-new_used


def resources(q,*,starting=False):
    n=native_contract();s=read(NATIVE['path']);m=n.mean_contract();mean=m.validate(read(n.checked(n.MEAN_PROTOCOL)))
    previous=m.resources(mean,starting=False)
    control=read(n.CONTROL_PROTOCOL['path'])
    extra_mean=sum(m.allocated(Path(root)) for root in control['additional_study_roots'])
    assert previous['used_output_bytes']+extra_mean<=mean['maximum_output_bytes']==4<<30
    original_roots=[B/f'VAR_multiscale_native_profile{kind}_20261101_v{version}' for version in (1,2,3) for kind in ('_code','')]
    original=sum(m.allocated(root) for root in original_roots)
    new=m.allocated(CODE)+m.allocated(OUT)
    assert new<=q['maximum_new_output_bytes'] and original+new<=q['shared_native_output_bytes']
    # Count the added outputs inside the unchanged original128MiB promise,
    # rather than charging their already written bytes twice.
    required=storage_requirement(previous['required_remaining_bytes'],original,new,q['maximum_new_output_bytes'],q['shared_native_output_bytes'])
    free=shutil.disk_usage(B).free
    floor=q['minimum_start_CPU_headroom_bytes' if starting else 'minimum_runtime_CPU_headroom_bytes']
    assert free>=required and previous['effective_CPU_headroom_bytes']>=floor
    return dict(free_bytes=free,required_remaining_bytes=required,
        effective_CPU_headroom_bytes=previous['effective_CPU_headroom_bytes'],original_native_allocated_bytes=original,
        new_allocated_bytes=new,shared_native_cap_bytes=q['shared_native_output_bytes'],
        all_previous_mean_and_legacy_promises_retained=True,completed_CPU_audits_have_no_future_output_writes=True,
        physical_reservation=False)


def setup(protocol):
    assert Path(protocol)==CODE/'protocol.json';q=validate(read(protocol))
    for pin in q['source_pins']:checked(pin)
    review=read(CODE/'actual_execution_review.json')
    assert review['status']=='PASS_FULL_NATIVE_THROUGHPUT_TOP0_SOURCE_AND_RESOURCE_REVIEW'
    assert review['protocol']==ref(protocol) and review['source_pins']==q['source_pins']
    dependency(q);return q
