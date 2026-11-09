"""Finite T-only native VAR engineering after all three existing queues exit."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

BASE=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
CODE=BASE/'VAR_multiscale_native_profile_code_20261101_v1'
OUT=BASE/'VAR_multiscale_native_profile_20261101_v1'
MEAN_CODE=BASE/'HIVE_training_mean_R80_code_20261101_v1'
MEAN_CONTRACT=dict(path=(MEAN_CODE/'brainhive_training_mean_R80_contract_CPU_20261101_v1.py').as_posix(),
    bytes=11849,sha256='3110987b3fcc3213a20e602d14f37e8a7ce00adfa3cacde6d99f63edefc41f3d')
MEAN_PROTOCOL=dict(path=(MEAN_CODE/'protocol.json').as_posix(),bytes=16803,
    sha256='56ae9b71794ad3e08fee8379a4491cbe1ff9457d7e433d1172fa570babd6c099')
POSITIONS=[0,1,1889,3779,5669,7559,9449,11339]
CLASSES=[0,11,287,999]
GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
ENTRIES=('VAR_multiscale_profile_contract_CPU_20261101_v1.py',
         'profile_VAR_multiscale_native_GPU_20261101_v1.py',
         'audit_VAR_multiscale_native_profile_CPU_20261101_v1.py',
         'queue_VAR_multiscale_native_profile_CPU_20261101_v1.py',
         'check_VAR_multiscale_profile_contract_CPU_20261101_v1.py',
         'EEG_multiscale_native_kernel_DRAFT_20261101_v1.py',
         'EEG_multiscale_native_class_DRAFT_CPU_20261101_v1.py')


def read(path):return json.loads(Path(path).read_bytes())


def ref(path):
    p=Path(path);assert p.is_file() and not p.is_symlink()
    before=p.stat()
    with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    after=p.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    return dict(path=str(p),bytes=before.st_size,sha256=digest)


def checked(pin):
    assert set(pin)=={'path','bytes','sha256'} and ref(pin['path'])==pin
    return Path(pin['path'])


def write(path,value):
    p=Path(path);assert p.resolve().is_relative_to(OUT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def mean_contract():
    path=checked(MEAN_CONTRACT)
    spec=importlib.util.spec_from_file_location('VAR_profile_exact_existing_mean_contract',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def validate(s):
    assert s['status']=='REVIEWED_CONDITIONAL_T8_ALL1000_FULL680_NATIVE_VAR_ENGINEERING'
    assert s['host']=='old' and s['GPU_UUID']==GPU and s['code_root']==CODE.as_posix() and s['output_root']==OUT.as_posix()
    assert s['existing_control_protocol']==MEAN_PROTOCOL and s['existing_control_contract']==MEAN_CONTRACT
    assert (s['existing_control_PID'],s['existing_control_start_ticks'])==(493907,114931939)
    assert s['fixed_T_positions']==POSITIONS and s['numerical_probe_classes']==CLASSES
    assert s['all_native_classes']==list(range(1000)) and s['class_batch']==8
    assert s['scales']==[1,2,3,4,5,6,8,10,13,16] and s['tokens']==680
    assert s['seed']==1729 and s['top_k']==600 and s['CFG']==0
    assert s['maximum_output_bytes']==128<<20 and s['maximum_GPU_bytes']==20<<30
    assert s['maximum_extra_CPU_bytes']==10<<30 and s['minimum_start_CPU_headroom_bytes']==24<<30
    assert s['minimum_runtime_CPU_headroom_bytes']==10<<30 and s['per_GPU_seconds']==900 and s['soft_GPU_seconds']==780
    assert s['execution_seconds']==1800 and s['waiting_deadline_epoch']==1793699743.411037
    assert s['numerical_guard']==dict(atol=1e-4,rtol=2e-5)
    assert s['constant_class_sampled_tokens_and_pixels_must_match'] is True
    assert s['no_EEG_fit_or_research_images'] is True and s['new_optimizer_updates']==0
    assert s['pretrained_precision']=='FP32_no_autocast_no_TF32' and s['no_automatic_retry'] is True
    assert s['full_training_bank_or_EEG_training_admitted'] is False
    assert {Path(p['path']).name for p in s['source_pins']}==set(ENTRIES) and len(s['source_pins'])==len(ENTRIES)
    assert all(Path(p['path']).parent==CODE for p in s['source_pins'])
    assert s['assets']['author_inventory']['sha256']=='c0d6b113c99f69a96036e95520fe64ccae7507bafd2674bfcb9b38024fe0f8e3'
    assert s['assets']['public_VQ']['sha256']=='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c'
    assert s['assets']['public_VAR']['sha256']=='4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb'
    assert set(s['assets']['data'])=={'tokens','token_identities','token_completion'}
    assert s['native_module_roots']==[(BASE/'EEG_native_conditional_variational_20261029_v1/code').as_posix(),
                                     (BASE/'AR_anchored_training_code_v1').as_posix()]
    return s


def classify_dependency(existing_live,existing_worker_live,upstream_status,complete,failed):
    if existing_live or existing_worker_live or upstream_status.startswith('WAIT_'):return 'WAIT_SAME_THREE_EXISTING_QUEUES'
    if failed or upstream_status.startswith('STOP_'):return 'STOP_EXISTING_DEPENDENCY_FAILED_NO_RETRY'
    if not complete:return 'STOP_EXISTING_CONTROL_INCOMPLETE_NO_RETRY'
    assert upstream_status=='READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS'
    return 'READY_ALL_THREE_COMPLETE_AND_EXITED'


def dependency(s):
    M=mean_contract();mean=M.validate(read(checked(MEAN_PROTOCOL)))
    launch=read(MEAN_CODE/'actual_launch.json')
    assert (launch['PID'],launch['start_ticks'],launch['protocol'])==(493907,114931939,MEAN_PROTOCOL)
    state=M.dependency(mean);process=M.process(493907,114931939)
    root=Path(mean['output_root']);worker=dict(exact_live=False)
    if (root/'progress.json').exists():
        p=read(root/'progress.json')
        if 'active_PID' in p:worker=M.process(p['active_PID'],p['active_start_ticks'])
    completion=(root/'completion.json').exists() and read(root/'completion.json')['status']=='COMPLETE_BOTH_TRAIN_MEAN_R80_4000_IMAGES_ALL16_FIXED4'
    failed=(root/'failure.json').exists()
    return dict(status=classify_dependency(process['exact_live'],worker['exact_live'],state['status'],completion,failed),
        existing_control=process,last_control_worker=worker,prior_two=state,
        existing_control_complete=completion,existing_control_failed=failed,result_payloads_read=0,NEW_contacted=False)


def resources(s,starting=False):
    M=mean_contract();mean=M.validate(read(checked(MEAN_PROTOCOL)))
    existing=M.resources(mean,starting=False)
    used=M.allocated(OUT)+M.allocated(CODE);assert used<=s['maximum_output_bytes']
    free=shutil.disk_usage(OUT if OUT.exists() else BASE).free
    required=existing['required_remaining_bytes']+s['maximum_output_bytes']-used
    floor=s['minimum_start_CPU_headroom_bytes' if starting else 'minimum_runtime_CPU_headroom_bytes']
    assert free>=required and existing['effective_CPU_headroom_bytes']>=floor
    return dict(**{k:v for k,v in existing.items() if k not in ('used_output_bytes','free_bytes','required_remaining_bytes','physical_reservation')},
        free_bytes=free,required_remaining_bytes=required,used_output_bytes=used,
        existing_mean_R80_remainder_and_all_legacy_promises_retained=True,
        profile_full128MiB_remaining_retained=True,physical_reservation=False)


def setup(protocol,require_ready=True):
    assert Path(protocol)==CODE/'protocol.json';s=validate(read(protocol))
    for p in s['source_pins']+s['inherited_source_pins']:checked(p)
    review=read(CODE/'actual_conditional_engineering_review.json')
    assert review['status']=='PASS_SOURCE_BOUND_CONDITIONAL_NATIVE_VAR_ENGINEERING_REVIEW'
    assert review['protocol']==ref(protocol) and review['source_pins']==s['source_pins']
    if require_ready:assert dependency(s)['status']=='READY_ALL_THREE_COMPLETE_AND_EXITED'
    return s
