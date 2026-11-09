"""Separate native R80 control lifecycle; does not alter any source fit or mean."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

BASE=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
CODE=BASE/'HIVE_training_mean_R80_code_20261101_v1'
OUT=BASE/'HIVE_training_mean_R80_20261101_v1'
PARENT=dict(path=(BASE/'HIVE_full1654_original_ordinary_lifecycle_code_20261101_v1/protocol.json').as_posix(),
            bytes=25189,sha256='4fc2c91c01d7f82b6b303181dcdccbe128ef2bfc694ea1c49a9c7bc0359458cd')
MEAN=dict(path=(BASE/'HIVE_training_mean_field_code_20261101_v1/protocol.json').as_posix(),
          bytes=4118,sha256='dd75ff0fb8c34800ad8ecc0dd2ab49652ae689be5dd19009e12f8f89c150180b')
MEAN_ROOT=BASE/'HIVE_training_mean_field_CPU_20261101_v1'
OWNED=dict(path=(Path(PARENT['path']).parent/'brainhive_owned_process_CPU_20261029_v1.py').as_posix(),
           bytes=2214,sha256='1bc79d57a22b2e9acd7402b9cf4bf4ecb4aca724456043ced9340517cee69f51')
GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
STATUS='REVIEWED_TRAINING_MEAN_NATIVE_R80_CONTROL_LIFECYCLE'
ROLES={'mean_bounded_F':'bounded_F','mean_ordinary_full1654':'ordinary_full1654'}
CONTRASTS=(('bounded_F','mean_bounded_F'),('ordinary_full1654','mean_ordinary_full1654'),
           ('mean_bounded_F','parent'),('mean_ordinary_full1654','parent'))
ENTRIES=('brainhive_training_mean_R80_contract_CPU_20261101_v1.py',
         'preflight_HIVE_training_mean_R80_CPU_20261101_v1.py',
         'run_HIVE_training_mean_R80_GPU_20261101_v1.py',
         'audit_HIVE_training_mean_R80_CPU_20261101_v1.py',
         'compare_HIVE_training_mean_R80_CPU_20261101_v1.py',
         'queue_HIVE_training_mean_R80_lifecycle_CPU_20261101_v1.py',
         'check_HIVE_training_mean_R80_CPU_20261101_v1.py')

def read(path):return json.loads(Path(path).read_bytes())

def ref(path):
    path=Path(path);assert path.is_file() and not path.is_symlink()
    before=path.stat()
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    return dict(path=str(path),bytes=before.st_size,sha256=sha)

def checked(pin):
    assert set(pin)=={'path','bytes','sha256'} and ref(pin['path'])==pin
    return Path(pin['path'])

def write(path,value):
    path=Path(path);assert path.resolve().is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def process(pid,ticks):
    try:
        f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(PID=pid,start_ticks=int(f[19]),state=f[0],exact_live=f[0] not in ('Z','X') and int(f[19])==ticks)
    except FileNotFoundError:return dict(PID=pid,present=False,exact_live=False)

def classify_dependency(original_live,mean_live,worker_live,original_complete,mean_complete,failed):
    if original_live or mean_live or worker_live:return 'WAIT_SAME_ORIGINAL_AND_CPU_MEAN_QUEUES'
    if failed:return 'STOP_FAILED_DEPENDENCY_NO_RETRY'
    if not original_complete or not mean_complete:return 'STOP_INCOMPLETE_DEPENDENCY_NO_RETRY'
    return 'READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS'

def validate(s):
    assert s['status']==STATUS and s['host']=='old' and s['GPU_UUID']==GPU
    assert s['code_root']==CODE.as_posix() and s['output_root']==OUT.as_posix()
    assert s['parent_protocol']==PARENT and s['mean_protocol']==MEAN and s['owned_process']==OWNED
    assert (s['parent_PID'],s['parent_start_ticks'])==(487039,113863283)
    assert (s['mean_PID'],s['mean_start_ticks'])==(492474,114699744)
    assert s['roles']==ROLES and s['comparisons']==[list(x) for x in CONTRASTS]
    assert s['queries_per_role']==2000 and s['new_research_images']==4000 and s['new_training_updates']==0
    assert s['training_seed']==1729 and s['source_training_cost_inherited'] is True
    assert s['resolution']==512 and s['UNet_steps']==4 and s['CFG']==0 and s['IP_scale']==1
    assert s['one_image_per_query'] is True and s['batch_queries_per_segment']==200
    assert s['strength']==1.0 and s['radius']==.25
    assert s['profile_indices']==list(range(0,2000,200))
    assert s['fixed_preview_indices']==[0,399,600,999,1400,1999]
    assert s['maximum_output_bytes']==4<<30 and s['maximum_extra_CPU_bytes']==10<<30 and s['maximum_GPU_bytes']==20<<30
    assert s['minimum_start_CPU_headroom_bytes']==24<<30 and s['minimum_runtime_CPU_headroom_bytes']==10<<30
    assert s['legacy_floor_bytes']==50<<30 and s['disk_buffer_bytes']==8<<30
    assert s['per_GPU_seconds']==900 and s['execution_seconds']==21600
    assert s['feature_attempts']==s['author_attempts']==3
    assert s['numerical_guard']==dict(atol=1e-5,rtol=5e-6)
    assert s['official200_previously_exposed'] is True and s['independent_H'] is False
    assert s['mean_uses_R80_inputs_or_scores'] is False and s['no_automatic_scientific_retry'] is True
    assert s['images_require_actual_per_role_GPU_profile'] is True
    assert s['waiting_deadline_epoch']==1793696143.411037
    assert len(s['source_pins'])==len(ENTRIES)
    assert {Path(p['path']).name for p in s['source_pins']}==set(ENTRIES)
    assert all(Path(p['path']).parent==CODE for p in s['source_pins'])
    return s

def dependency(s):
    parent=read(checked(PARENT));mean=read(checked(MEAN))
    original_launch=read(Path(PARENT['path']).parent/'actual_launch.json')
    mean_launch=read(Path(MEAN['path']).parent/'actual_launch.json')
    assert (original_launch['PID'],original_launch['start_ticks'],original_launch['protocol'])==(487039,113863283,PARENT)
    assert (mean_launch['PID'],mean_launch['start_ticks'],mean_launch['protocol'])==(492474,114699744,MEAN)
    roots=[Path(parent['output_root']),MEAN_ROOT]
    handles=[process(487039,113863283),process(492474,114699744)];workers=[]
    for root in roots:
        worker=dict(exact_live=False)
        if (root/'progress.json').exists():
            p=read(root/'progress.json')
            if 'active_PID' in p:worker=process(p['active_PID'],p['active_start_ticks'])
        workers.append(worker)
    completed=[]
    for root,status in zip(roots,('COMPLETE_FULL1654_ORIGINAL_ORDINARY_FIT_2000_R80_ALL16_AND_FIXED5',
                                 'COMPLETE_TRAIN_ONLY_MEAN_FIELD_CPU_PREPARATION_NOT_GPU_EFFICACY')):
        path=root/'completion.json';completed.append(path.exists() and read(path)['status']==status)
    failed=any((root/'failure.json').exists() for root in roots)
    gate=classify_dependency(handles[0]['exact_live'],handles[1]['exact_live'],any(p['exact_live'] for p in workers),*completed,failed)
    return dict(status=gate,controllers=handles,last_workers=workers,completion_flags=completed,
                result_payloads_read=0,NEW_contacted=False)

def allocated(root):
    root=Path(root);total=0;seen=set()
    if not root.exists():return 0
    for path in root.rglob('*'):
        assert not path.is_symlink()
        try:
            st=path.stat()
        except FileNotFoundError:continue  # Concurrent atomic source checkpoint rename; conservative over-reservation.
        if path.is_file() and (st.st_dev,st.st_ino) not in seen:
            seen.add((st.st_dev,st.st_ino));total+=st.st_blocks*512 if hasattr(st,'st_blocks') else st.st_size
    return total

def headroom():
    host=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    group=Path('/sys/fs/cgroup');maximum=(group/'memory.max').read_text().strip()
    return min(host,int(maximum)-int((group/'memory.current').read_text())) if maximum!='max' else host

def resources(s,starting=False):
    used=allocated(OUT);assert used<=s['maximum_output_bytes']
    parent=read(checked(PARENT));remaining_parent=max(0,(8<<30)-allocated(parent['output_root']))
    required=(s['maximum_output_bytes']-used+s['legacy_floor_bytes']+s['disk_buffer_bytes']
              +(14<<30)+(64<<20)+(32<<20)+remaining_parent)
    free=shutil.disk_usage(OUT if OUT.exists() else BASE).free
    available=headroom()
    assert free>=required and available>=s['minimum_start_CPU_headroom_bytes' if starting else 'minimum_runtime_CPU_headroom_bytes']
    return dict(effective_CPU_headroom_bytes=available,used_output_bytes=used,free_bytes=free,
                required_remaining_bytes=required,parent_remaining_budget_retained=remaining_parent,
                CPU_mean_full32MiB_promise_retained=True,physical_reservation=False)

def setup(protocol,need_bindings=True):
    s=validate(read(protocol));assert Path(protocol)==CODE/'protocol.json'
    for pin in s['source_pins']:checked(pin)
    for pin in s['inherited_source_pins']:checked(pin)
    review=read(CODE/'actual_conditional_execution_review.json')
    assert review['status']=='PASS_SOURCE_BOUND_CONDITIONAL_MEAN_CONTROL_LIFECYCLE_REVIEW'
    assert review['protocol']==ref(protocol) and review['source_pins']==s['source_pins']
    assert dependency(s)['status']=='READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS'
    sys.path.insert(0,str(Path(PARENT['path']).parent))
    import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
    parent=F.setup_fit(PARENT['path'])
    sys.path.insert(0,str(CODE))
    if need_bindings:
        closure=read(OUT/'actual_source_data_closure.json')
        assert closure['status']=='PASS_ACTUAL_TRAIN_MEAN_R80_SOURCE_DATA_CLOSURE' and closure['protocol']==ref(protocol)
        for role in ROLES:
            checked(closure['derived_components'][role]);checked(closure['source_components'][ROLES[role]])
    return s,parent,F

def load_control(s,role,torch):
    assert role in ROLES
    closure=read(OUT/'actual_source_data_closure.json')
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    component=closure['derived_components'][role]
    value=torch.load(checked(component),weights_only=True,map_location='cpu')
    assert value['arm']==ROLES[role]+'_training_mean_field' and value['derivation_protocol']==MEAN
    assert value['new_training_updates']==0 and value['source_training_updates']==31014
    assert value['source_component']==closure['source_components'][ROLES[role]]
    assert value['mean_rows']==165400 and value['strength']==1.0 and value['radius']==.25
    assert torch.count_nonzero(value['model']['direction.weight'])==0
    model=BoundedConditionAdapter(predict_risk=False);model.load_state_dict(value['model'],strict=True)
    model.eval().requires_grad_(False)
    return model,component,value

def load_evaluation_inputs(parent,torch):
    import numpy as np
    source=read(checked(parent['evaluation']['input_completion']))
    bank=np.load(checked(source['bank']),allow_pickle=False)
    records=[json.loads(line) for line in checked(parent['evaluation']['baseline_inference']).read_text().splitlines()]
    assert len(records)==2000 and bank['conditions'].shape==(10,200,1024)
    assert bank['gammas'].shape==bank['betas'].shape==(10,1024)
    for sub,item in enumerate(source['reports']):
        state=torch.load(checked(item['model']),weights_only=True,map_location='cpu')['model']
        assert np.array_equal(bank['gammas'][sub],state['block.layers.0.norm.weight'].numpy())
        assert np.array_equal(bank['betas'][sub],state['block.layers.0.norm.bias'].numpy())
    assert len(source['reports'])==10
    for index,r in enumerate(records):
        sub,row=divmod(index,200)
        assert r['subject']==f'sub-{sub+1:02d}' and r['array_row']==row
        assert hashlib.sha256(bank['conditions'][sub,row:row+1].tobytes()).hexdigest()==r['condition_sha256']
    return source,bank,records
