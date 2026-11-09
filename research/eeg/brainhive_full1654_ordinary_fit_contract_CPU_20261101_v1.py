"""Future complete original-objective study; requires actual closed engineering first."""
import hashlib
import json
import os
from pathlib import Path
import sys
import brainhive_full1654_ordinary_engine_20261101_v1 as E
K=E.K
STATUS='REVIEWED_FULL1654_ORIGINAL_ORDINARY_COMPLETE_LIFECYCLE'
REQUIRED_ENTRIES=('train_HIVE_full1654_original_ordinary_GPU_20261101_v1.py',
    'audit_HIVE_full1654_original_ordinary_fit_CPU_20261101_v1.py',
    'run_HIVE_full1654_original_ordinary_R80_GPU_20261101_v1.py',
    'audit_HIVE_full1654_original_ordinary_R80_CPU_20261101_v1.py',
    'compare_HIVE_full1654_all4_methods_CPU_20261101_v1.py',
    'queue_HIVE_full1654_original_ordinary_lifecycle_CPU_20261101_v1.py')


def validate_scope(s):
    assert s['status']==STATUS and s['method']==E.METHOD
    assert s['host']=='old' and s['GPU_UUID']==K.GPU
    assert s['official200_previously_exposed'] is True and s['official_results_used_for_previous_direction'] is True
    assert s['independent_H'] is False and s['training_R80_payloads_allowed'] is False
    assert s['maximum_output_bytes']==8<<30 and s['maximum_GPU_bytes']==20<<30 and s['maximum_extra_CPU_bytes']==10<<30
    assert s['minimum_start_CPU_headroom_bytes']==24<<30 and s['minimum_runtime_CPU_headroom_bytes']==10<<30
    assert s['legacy_floor_bytes']==50<<30 and s['disk_buffer_bytes']==8<<30
    assert s['upstream_remaining_budget_reserved_bytes']==(14<<30)+(64<<20)
    assert s['per_GPU_seconds']==900 and s['maximum_training_segments']==128 and s['segment_soft_boundary_seconds']==780
    assert s['numerical_guard']==dict(atol=1e-5,rtol=5e-6)
    assert s['future_GPU_bit_exactness_claimed'] is False and s['fixed_last_only'] is True
    assert s['old_ordinary_immutable'] is True and s['no_new_training_seeds'] is True
    assert s['R80']==dict(queries=2000,subjects=10,candidates=200,resolution=512,UNet_steps=4,CFG=0,IP_scale=1,images_per_query=1)
    assert set(s['reference_CSVs'])==set(s['reference_manifests'])=={'parent','bounded_F','Euclidean_F'}
    assert all(set(value)=={'common','paper'} for value in s['reference_CSVs'].values())
    assert set(REQUIRED_ENTRIES)<=set(Path(pin['path']).name for pin in s['source_pins'])
    assert 'brainhive_gpu_exit_contract_CPU_20261101_v1.py' in {Path(pin['path']).name for pin in s['source_pins']}
    return s


def setup_fit(path):
    # Reject current engineering-only protocols before tensor imports or outputs.
    s=validate_scope(K.read(path))
    for pin in s['source_pins']:K.check(pin)
    K.check(s['old_ordinary'])
    assert s['old_ordinary']['sha256']=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
    profile=K.read(K.check(s['actual_engineering_stage_completion']))
    assert profile['status']=='COMPLETE_ORIGINAL_ORDINARY_FULL1654_ENGINEERING_PROFILE_ONLY'
    evidence=K.read(K.check(profile['CPU_audit']))
    assert evidence['status']=='PASS_ACTUAL_FOUR_STEP_ORDINARY_LOSS_HEAD_VJP_AND_ADAM_CPU_AUDIT'
    assert evidence['temporary_updates_audited']==4 and evidence['formal_training_admitted'] is False
    original=K.read(K.check(s['completed_F_study']))
    assert original['status']=='COMPLETE_FULL1654_TWO_F_FITS_4000_R80_IMAGES_ALL16_FIXED_COMPARISONS'
    review=K.read(K.check(s['actual_complete_lifecycle_review']))
    assert review['status']=='PASS_ACTUAL_COMPLETE_ORDINARY_SOURCE_DATA_RESOURCE_AND_LIFECYCLE_REVIEW'
    assert review['source_pins']==s['source_pins']
    assert review['actual_engineering_stage_completion']==s['actual_engineering_stage_completion']
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(path).parent))
    return s


def prefix_counts(updates):
    assert type(updates) is int and 0<=updates<=31014
    passes,step=divmod(updates,10338)
    return passes,step,passes*165400+step*16


def expected_coverage(updates,orders,torch):
    passes,step,_=prefix_counts(updates)
    coverage=torch.zeros(3,165400,dtype=torch.bool)
    for epoch in range(3):
        if epoch<passes:coverage[epoch]=True
        elif epoch==passes:coverage[epoch,orders[epoch][:step*16]]=True
    return coverage


def atomic_metadata(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.is_symlink()
    pending=path.with_name(path.name+'.pending.'+str(os.getpid()))
    assert not pending.exists()
    with pending.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def save_checkpoint(root,state,torch):
    original=Path(root);assert not original.is_symlink();root=original.resolve(strict=True)
    assert type(state['segment']) is int and 0<=state['segment']<128
    assert prefix_counts(state['updates'])[2]==state['exposures']
    checkpoints=root/'checkpoints';assert not checkpoints.is_symlink();checkpoints.mkdir(exist_ok=True)
    stem=f"step-{state['updates']:06d}_segment-{state['segment']:03d}"
    pending=checkpoints/(stem+'.pending');committed=checkpoints/stem
    assert pending.resolve().is_relative_to(root) and committed.resolve().is_relative_to(root)
    assert not pending.exists() and not committed.exists();pending.mkdir()
    E.save_tensor(pending/'state.pt',state,torch)
    value=K.ref(pending/'state.pt');value['path']=str(committed/'state.pt')
    commit=dict(checkpoint=value,updates=state['updates'],exposures=state['exposures'],segment=state['segment'])
    K.write(pending/'commit.json',commit)
    os.rename(pending,committed)
    descriptor=os.open(checkpoints,os.O_RDONLY|os.O_DIRECTORY) if os.name=='posix' else None
    if descriptor is not None:os.fsync(descriptor);os.close(descriptor)
    pointer=dict(checkpoint=commit['checkpoint'],commit=K.ref(committed/'commit.json'),
                 updates=state['updates'],exposures=state['exposures'],segment=state['segment'])
    atomic_metadata(root/'latest_checkpoint.json',pointer)
    return pointer


def load_checkpoint(root,torch):
    root=Path(root).resolve(strict=True);pointer=K.read(root/'latest_checkpoint.json')
    assert all(Path(pointer[key]['path']).resolve().is_relative_to(root) for key in ('checkpoint','commit'))
    commit=K.read(K.check(pointer['commit']))
    assert commit['checkpoint']==pointer['checkpoint']
    assert all(commit[key]==pointer[key] for key in ('updates','exposures','segment'))
    state=torch.load(K.check(pointer['checkpoint']),weights_only=True,map_location='cpu')
    assert all(state[key]==pointer[key] for key in ('updates','exposures','segment'))
    assert prefix_counts(state['updates'])[2]==state['exposures']
    return state,pointer


def load_final(s,torch):
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    root=Path(s['output_root'])/'ordinary_full1654'
    fit=K.read(root/'fit_completion.json');audit=K.read(root/'fit_CPU_audit.json')
    assert fit['status']=='COMPLETE_FULL1654_ORIGINAL_ORDINARY_THREE_PASSES_FIXED_FINAL'
    assert audit['status']=='PASS_FULL1654_ORIGINAL_ORDINARY_FINAL_STATE_AND_ALL165400_CPU_FORWARD_AUDIT'
    assert fit['component']==audit['component'] and fit['updates']==audit['updates']==31014
    saved=torch.load(K.check(fit['component']),weights_only=True,map_location='cpu')
    assert saved['arm']=='ordinary_full1654' and saved['method']==E.METHOD and saved['protocol']==s['_protocol_pin']
    model=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
    model.load_state_dict(saved['model'],strict=True)
    return model,fit,audit
