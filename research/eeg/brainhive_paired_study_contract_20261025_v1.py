"""Source-bound study contract; independent of the old v2 admission."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
REMOTE=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
WORK=Path('/root/brainhive_reproduction_20261020_v1')
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/paired_effect_20261025_v1')
SCHEMA='BrainHIVE_paired_effect_study_20261025_v1'
ARMS=('O','A','D','U')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def ref(path):return dict(path=str(path),sha256=sha(path))


def bound(item):
    assert sha(item['path'])==item['sha256'],item['path']
    return read(item['path'])


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    assert not temporary.exists(),'Preserve orphan transaction: '+str(temporary)
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def bytes_used(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def core(config):
    payload={k:config[k] for k in ('method','core_sources','input_bindings','authorization')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate(config,*,all_inputs=False):
    assert config['schema']==SCHEMA and config['arms']==list(ARMS)
    assert config['method']==dict(seed=1729,primary_queries=132300,passes=3,updates=24807,batch_size=16,
        hidden=256,radius=.25,lr=.0001,weight_decay=.01,clip_norm=1.,epsilon=1e-6,
        image_weight=1.,token_weight=.1,auxiliary_weight=.1,contrast_component_weight=.25,
        symmetric_absolute_control=True,shared_pair_graphs_in_all_arms=True,
        shared_correct_pair_normalizer_and_joint_valid_mask=True,
        same_subject_same_OOF_role_different_concept=True,parent_Q=True,Q_positions=16,IP_layers=70,
        training_precision='FP32 head outside BF16 autocast over unchanged FP16 UNet_A_P_IP',
        training_times='uniform DDPM integer0..999',last_checkpoint=True,
        strength_policy='Dfit_only_global_strength_calibration')
    authorization=bound(config['authorization'])
    assert authorization['candidate_limit']==13 and authorization['main_candidate_ordinal']==13
    assert authorization['structural_revision_ordinal']==2 and authorization['structural_revision_limit']==2
    assert authorization['fixed_controls']==['O','A','U'] and authorization['new_main']=='D'
    assert authorization['user_instruction_verbatim']=='扩到13，推进这一项及固定对照'
    for name,digest in (config['core_sources']|config.get('pipeline_sources',{})).items():
        assert Path(name).name==name and sha(ROOT/name)==digest,name
    assert config['maximum_training_seconds_per_job']==86400
    assert config['maximum_cumulative_training_seconds_per_arm']==172800
    assert config['maximum_generation_seconds_per_job']==86400
    assert config['maximum_cumulative_generation_seconds_per_arm']==172800
    assert config['maximum_CPU_extra_bytes']==10<<30 and config['maximum_GPU_bytes']==20<<30
    assert config['minimum_training_CPU_headroom_bytes']==24<<30
    assert config['arm_disk_budget_bytes']==24<<30 and config['disk_buffer_bytes']==8<<30
    assert Path(config['output_root'])==DATA and config['automatic_method_search'] is False
    for name,item in config['input_bindings'].items():
        if name!='native_input_sha256':assert sha(item['path'])==item['sha256'],item['path']
        elif all_inputs:
            for path,digest in item.items():assert sha(path)==digest,path
    return core(config)


def resources(config,*,training_start=False):
    from brainhive_functional_training_contract_v2 import resource_headroom
    memory=resource_headroom()
    minimum=config['minimum_training_CPU_headroom_bytes'] if training_start else 10<<30
    assert memory['effective_headroom_bytes']>=minimum,('CPU headroom',memory)
    remaining=sum(max(0,config['arm_disk_budget_bytes']-bytes_used(DATA/'arms'/a)) for a in ARMS)
    assert shutil.disk_usage(DATA.parent).free>=remaining+config['disk_buffer_bytes'],'Full four-arm persistent budget unavailable'
    assert all(bytes_used(DATA/'arms'/a)<=config['arm_disk_budget_bytes'] for a in ARMS)
    return dict(memory=memory,reserved_remaining_bytes=remaining,free_bytes=shutil.disk_usage(DATA.parent).free)


def prepare_plan(config):
    from brainhive_paired_effect_integration_20261025_v1 import build_pair_plan
    bindings=config['input_bindings']
    contract=bound(bindings['input_contract']);protocol=bound(bindings['split'])
    assert contract['status']=='ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    manifest=bindings['image_manifest'];assert sha(manifest['path'])==manifest['sha256']==protocol['image_manifest_sha256']
    with Path(manifest['path']).open(encoding='utf-8',newline='') as stream:
        metadata={int(r['array_row']):r for r in csv.DictReader(stream) if r['dataset_split']=='train'}
    for subject in contract['subjects']:
        for source in subject['sources']:
            if source['role'] not in ('oof0','oof1'):continue
            assert sha(source['parent_registration_path'])==source['parent_registration_sha256']
            reg=read(source['parent_registration_path'])
            trained={metadata[i]['concept_id'] for i in reg['training_array_rows']}
            held={metadata[int(r['array_row'])]['concept_id'] for r in source['groups']['component_train']}
            assert not trained&held and set(reg['training_array_rows'])<=set(protocol['image_rows']['T'])
    plan=build_pair_plan(contract,protocol,metadata)
    canonical=json.dumps(plan,sort_keys=True,separators=(',',':')).encode()
    return plan,hashlib.sha256(canonical).hexdigest()


def setup_runtime(config):
    import sys
    assert 'diffusers' not in sys.modules and 'transformers' not in sys.modules
    runtime=bound(config['runtime_manifest'])
    folder=Path(runtime['path'])
    files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    assert files==runtime['files']
    sys.path.insert(0,str(folder))
    import brainhive_native_models_v1
    import accelerate
    from diffusers.utils import is_accelerate_available
    assert accelerate.__version__=='0.34.2' and is_accelerate_available()


def verify_wrapper(config,job_name,maximum,output):
    import torch
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch=torch
    doc=dict(gpu_job_name=job_name,maximum_seconds=maximum,output_directory=str(output),
        maximum_output_bytes=512<<20,maximum_extra_CPU_bytes=config['maximum_CPU_extra_bytes'],
        maximum_GPU_bytes=config['maximum_GPU_bytes'],minimum_CPU_headroom_bytes=10<<30,
        resource_authorization=config['resource_authorization'])
    return D.verify_wrapper(doc),D.Budget(doc)


def require_admission(config):
    path=DATA/'admission.json';value=read(path)
    assert value['status']=='FULL_PAIRED_STUDY_ADMITTED_AFTER_ACTUAL_ENGINEERING'
    assert value['study_core_sha256']==validate(config) and value['config_sha256']==sha(config['registration_path'])
    profile=bound(value['profile'])
    assert profile['status']=='COMPLETE_REAL_PAIRED_FOUR_ARM_ENGINEERING_PROFILE'
    assert profile['study_core_sha256']==core(config) and profile['all_checks_passed'] is True
    assert value['complete_lifecycle_code_pinned'] and value['full_storage_reserved']
    resources(config)
    return value
