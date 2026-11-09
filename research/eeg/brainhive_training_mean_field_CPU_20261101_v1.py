"""Training-only raw-field ablation contract. CPU only; not GPU admission."""
import hashlib
import json
import os
import datetime
from pathlib import Path
import sys

BASE=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
CODE=BASE/'HIVE_training_mean_field_code_20261101_v1'
OUT=BASE/'HIVE_training_mean_field_CPU_20261101_v1'
REPOSITORY=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
PARENT=dict(path=str(BASE/'HIVE_full1654_original_ordinary_lifecycle_code_20261101_v1/protocol.json'),
            bytes=25189,sha256='4fc2c91c01d7f82b6b303181dcdccbe128ef2bfc694ea1c49a9c7bc0359458cd')
ROLES=('bounded_F','ordinary_full1654')
STATUS='REVIEWED_TRAIN_ONLY_RAW_FIELD_MEAN_CPU_ABLATION_PREPARATION'
FHEAD=dict(path=str(BASE/'HIVE_full1654_F_lifecycle_20261101_v2/bounded_F/component_final.pt'),
           bytes=2104528,sha256='4fc8d0be3f9c3414d49c672a629b42b121b694ab43180c64194d671211fbc228')
OWNED=dict(path=str(Path(PARENT['path']).parent/'brainhive_owned_process_CPU_20261029_v1.py'),
           bytes=2214,sha256='1bc79d57a22b2e9acd7402b9cf4bf4ecb4aca724456043ced9340517cee69f51')

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
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def process(pid,ticks):
    try:
        f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return dict(PID=pid,start_ticks=int(f[19]),state=f[0],exact_live=f[0] not in ('Z','X') and int(f[19])==ticks)
    except FileNotFoundError:return dict(PID=pid,exact_live=False,present=False)

def classify_dependency(controller_live,worker_live,completion_status,failure_present):
    # Never consume incomplete source weights or turn missing handles into a retry.
    if controller_live or worker_live:return 'WAIT_SAME_ORIGINAL_LIFECYCLE'
    if failure_present:return 'STOP_FAILED_DEPENDENCY_NO_RETRY'
    if completion_status!='COMPLETE_FULL1654_ORIGINAL_ORDINARY_FIT_2000_R80_ALL16_AND_FIXED5':
        return 'STOP_MISSING_COMPLETE_DEPENDENCY_NO_RETRY'
    return 'READY_CLOSED_ORIGINAL_LIFECYCLE'

def validate(s):
    assert s['status']==STATUS and s['host']=='old' and s['parent_protocol']==PARENT
    assert s['output_root']==str(OUT) and s['code_root']==str(CODE)
    assert (s['parent_PID'],s['parent_start_ticks'])==(487039,113863283)
    assert tuple(s['roles'])==ROLES
    assert s['queries']==165400 and s['subjects']==10 and s['concepts']==1654 and s['images']==16540
    assert s['chunk']==512 and s['new_training_updates']==0 and s['training_seed']==1729
    assert s['mean_definition']=='FP32_native_raw_MLP_outputs_FP64_sum_over_all165400_training_OOF_rows'
    assert s['raw_R80_inputs_allowed'] is False and s['image_score_payloads_allowed'] is False
    assert s['official200_previously_exposed'] is True and s['independent_H'] is False
    assert s['GPU_admitted'] is False and s['maximum_output_bytes']==32<<20
    assert s['maximum_extra_CPU_bytes']==4<<30 and s['minimum_start_CPU_headroom_bytes']==24<<30
    assert s['minimum_runtime_CPU_headroom_bytes']==16<<30 and s['CPU_seconds_per_stage']==900
    assert s['strength']==1.0 and s['radius']==0.25 and s['no_automatic_retry'] is True
    assert s['legacy_floor_bytes']==50<<30 and s['disk_buffer_bytes']==8<<30
    assert s['numerical_guard']==dict(atol=1e-5,rtol=5e-6)
    assert s['bounded_F_component']==FHEAD
    assert Path(s['bounded_F_fit_completion']['path'])==BASE/'HIVE_full1654_F_lifecycle_20261101_v2/bounded_F/fit_completion.json'
    assert type(s['waiting_deadline_epoch']) in (int,float) and s['waiting_deadline_epoch']>0
    assert s['owned_process']==OWNED
    assert len(s['source_pins'])==5 and {Path(p['path']).name for p in s['source_pins']}=={
        'brainhive_training_mean_field_CPU_20261101_v1.py',
        'compute_HIVE_training_mean_field_CPU_20261101_v1.py',
        'audit_HIVE_training_mean_field_CPU_20261101_v1.py',
        'queue_HIVE_training_mean_field_CPU_20261101_v1.py',
        'check_HIVE_training_mean_field_CPU_20261101_v1.py'}
    assert all(Path(p['path']).parent==CODE for p in s['source_pins'])
    return s

def dependency(s):
    parent=read(checked(s['parent_protocol']))
    launch=read(Path(s['parent_protocol']['path']).parent/'actual_launch.json')
    assert (launch['PID'],launch['start_ticks'],launch['protocol'])==(s['parent_PID'],s['parent_start_ticks'],PARENT)
    assert s['waiting_deadline_epoch']==datetime.datetime.fromisoformat(launch['UTC']).timestamp()+172800
    root=Path(parent['output_root'])
    controller=process(s['parent_PID'],s['parent_start_ticks'])
    worker=dict(exact_live=False)
    if (root/'progress.json').exists():
        value=read(root/'progress.json');worker=process(value['active_PID'],value['active_start_ticks'])
    completion=read(root/'completion.json') if (root/'completion.json').exists() else None
    if completion:
        assert completion['protocol']==PARENT and completion['head_updates']==31014
        assert completion['new_research_images']==2000 and completion['training_seed']==1729
    decision=classify_dependency(controller['exact_live'],worker['exact_live'],completion.get('status') if completion else None,(root/'failure.json').exists())
    return dict(status=decision,controller=controller,worker=worker,
                completion=ref(root/'completion.json') if completion else None,
                result_payloads_read=0,NEW_contacted=False)

def checked_setup(protocol):
    s=validate(read(protocol))
    for pin in s['source_pins']:checked(pin)
    assert Path(protocol)==CODE/'protocol.json'
    gate=dependency(s);assert gate['status']=='READY_CLOSED_ORIGINAL_LIFECYCLE',gate
    review=read(CODE/'actual_CPU_only_preparation_review.json')
    assert review['status']=='PASS_SOURCE_BOUND_TRAIN_ONLY_MEAN_CPU_PREPARATION'
    assert review['protocol']==ref(protocol) and review['source_pins']==s['source_pins'] and review['GPU_admitted'] is False
    sys.path.insert(0,str(Path(PARENT['path']).parent))
    import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
    parent=F.setup_fit(PARENT['path'])
    import torch
    assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    return s,parent,F

def source_head(s,parent,role):
    assert role in ROLES
    if role=='bounded_F':
        done=read(checked(s['bounded_F_fit_completion']))
        assert done['status']=='COMPLETE_FULL1654_F_HEAD_THREE_PASSES'
        assert done['component']==s['bounded_F_component']
    else:
        folder=Path(parent['output_root'])/'ordinary_full1654'
        done=read(folder/'fit_completion.json')
        assert done['status']=='COMPLETE_FULL1654_ORIGINAL_ORDINARY_THREE_PASSES_FIXED_FINAL'
        assert done['protocol']==PARENT
        audit=read(folder/'fit_CPU_audit.json')
        assert audit['status']=='PASS_FULL1654_ORIGINAL_ORDINARY_FINAL_STATE_AND_ALL165400_CPU_FORWARD_AUDIT'
        assert audit['component']==done['component'] and audit['all165400_explicit_CPU64_forwards_verified']
    assert (done['updates'],done['exposures'],done['parameters'])==(31014,496200,525568)
    checked(done['component'])
    return done['component']

def raw_output(c,weight,bias,state,torch,precision='FP32'):
    # Own source affine for every row; never replace it by one subject's affine.
    dtype=torch.float64 if precision=='FP64' else torch.float32
    c,weight,bias=(v.to(dtype=dtype,device='cpu') for v in (c,weight,bias))
    assert (weight.abs()>1e-8).all()
    u=(c-bias)/weight
    h=torch.nn.functional.gelu(torch.nn.functional.linear(u,state['trunk.0.weight'].to(dtype),state['trunk.0.bias'].to(dtype)))
    return torch.nn.functional.linear(h,state['direction.weight'].to(dtype),state['direction.bias'].to(dtype))

def constant_state(original,mean,torch):
    assert mean.shape==(1024,) and mean.dtype==torch.float32 and torch.isfinite(mean).all()
    result={key:value.detach().cpu().clone() for key,value in original.items()}
    result['direction.weight'].zero_();result['direction.bias'].copy_(mean)
    assert torch.equal(result['trunk.0.weight'],original['trunk.0.weight'])
    assert torch.equal(result['trunk.0.bias'],original['trunk.0.bias'])
    assert torch.count_nonzero(result['direction.weight'])==0
    return result

def mean_and_variance(sum_vector,sum_squares,count,torch):
    assert count>0 and sum_vector.dtype==sum_squares.dtype==torch.float64
    mean=sum_vector/count
    variance=sum_squares/count-mean.square()
    assert torch.isfinite(mean).all() and variance.min()>-1e-8*max(1.0,float(mean.square().max()))
    return mean,variance
