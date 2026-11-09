"""Resolve actual complete lifecycle budgets without selecting on image scores.

The full five-arm objective is retained. Releasing a fully funded arm is
campaign progress; missing controls prohibit a complete mechanism conclusion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import time
import fcntl

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


class Pending(RuntimeError):pass


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def new(path,value):
    assert not path.exists(),('Preserve existing resolved artifact',str(path))
    path.write_text(json.dumps(value,indent=2)+'\n')


def tree_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob('*') if item.is_file())


def remaining():
    ledger=ROOT/'audit/compute_ledger.jsonl'
    used=sum(float(x.get('charged_seconds',0)) for x in map(json.loads,ledger.read_text().splitlines()))
    deadline=load(ROOT/'configs/compute_time_bounds.json')['conservative_occupied_time_deadline_epoch']
    return min(40*3600-used,deadline-time.time())


def storage_reservations(configs):
    image_bytes=tree_bytes(DATA/'images/S1_full_native_batchsafe_v2/V/REF_REG_correct_R4')
    feature_bytes=tree_bytes(DATA/'evaluation/V/REF_REG_correct_R4/features')
    storage=[]
    for name,value in configs.items():
        future_images=0 if phase_done(value,'generate') else max(0,int(image_bytes*1.25)-tree_bytes(value['image_root']))
        future_features=0 if phase_done(value,'features') else max(0,int(feature_bytes*1.25)-tree_bytes(Path(value['evaluation_root'])/'features'))
        training_peak=0 if name=='Frozen' or phase_done(value,'train') else 256*1024**2
        storage.append(dict(arm=name,future_image_bytes=future_images,future_feature_bytes=future_features,
            training_checkpoint_peak_bytes=training_peak,small_evaluation_and_T_bytes=64*1024**2))
    required=sum(x['future_image_bytes']+x['future_feature_bytes']+x['training_checkpoint_peak_bytes']+x['small_evaluation_and_T_bytes'] for x in storage)+1024**3
    return dict(required_free_bytes=required,available_free_bytes=shutil.disk_usage(DATA).free,
        storage_reservations=storage,storage_profile=dict(actual_reference_image_bytes=image_bytes,
            actual_reference_feature_bytes=feature_bytes,prospective_margin=1.25,shared_free_reserve_bytes=1024**3))


def phase_done(config,phase):
    out=Path(config['evaluation_root']);images=Path(config['image_root'])
    paths={'train':Path(config['training_root'])/'completion.json','T_readiness':Path(config['T_readiness_root'])/'completion.json',
        'generate':images/'completion.json','identity_before':out/'identity_before/summary.json',
        'features':out/'features/completion.json','score':out/'metrics/summary.json',
        'lowlevel':out/'low_level/summary.json','identity_after':out/'identity_after/summary.json','report':out/'REPORT_receipt.json'}
    p=paths[phase]
    if not p.exists():return False
    data=load(p)
    if phase=='train':
        return data['status']=='COMPLETED_REGISTERED_MATCHED_FULL_QUERY_PASS' and data['arm']==config['arm'] and data['seen_training_queries']==132300 and sha(data['checkpoint'])==data['checkpoint_sha256']
    if phase in ['generate','T_readiness']:
        return data['status']=='COMPLETED' and data['queries']==(16600 if phase=='generate' else 40) and sha(p.parent/'manifest.jsonl')==data['image_manifest_sha256'] and data['full_training_config_sha256']==config['full_training_config_sha256']
    if phase in ['features','score']:return data['status']=='COMPLETED' and data['queries']==16600
    if phase=='lowlevel':return data['status']=='COMPLETED' and data['verified_and_decoded']==16600 and data['missing_count']==data['corrupt_or_mismatched_count']==0
    if phase=='report':return data['arm']==config['arm'] and data['queries']==16600 and data['source_metric_sha256']==sha(out/'metrics/summary.json')
    return data['status']=='PASSED_ALL16600_IDENTITY_AND_SAVED_FILES' and data['verified_queries']==16600 and data['score_rows_checked']==(16600 if phase=='identity_after' else 0)


def prepare(arm,policy_path):
    policy_path=Path(policy_path);policy=load(policy_path)
    assert policy['version']=='idea_interface_execution_policy_v2' and policy['total_GPU_hour_cap']==40
    assert arm in policy['required_interface_arms']+['Frozen']
    for source,digest in policy['source_sha256'].items():assert sha(ROOT/source)==digest,source
    release=ROOT/'configs'/f'idea_interface_{arm}_resolved_v1.json'
    if release.exists():
        result=load(release);return dict(status='EXISTING_RESOLVED_ARM_INSPECT_STATE_NO_DUPLICATE_RUN',resolved_path=str(release),resolved_sha256=sha(release),arm=arm)
    if (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists():raise Pending('No new development method after final unlock')
    with (ROOT/'audit/GPU_JOB.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise Pending('An actual registered GPU phase holds the single-card lock')
        fcntl.flock(lock,fcntl.LOCK_UN)
    aux_path=ROOT/'audit/idea_image_preparation_queue_state_v1.json';aux=load(aux_path)
    terminal=['PREPARATION_COMPLETE_UNET_NOT_RUN_CPU_PREREQUISITE_FAILED','PREPARATION_COMPLETE_UNET_NOT_RUN_NUMERICAL_STAGE_INCOMPLETE','IMAGE_PREPARATION_TERMINAL_REVIEW_ACTUAL_RECEIPTS']
    if aux['status'] not in terminal:raise Pending('Actual preparation/response queue is not terminal; do not compete for the GPU')
    recipe_path=ROOT/'configs/idea_interface_training_recipe_v1.json';recipe=load(recipe_path)
    for source,digest in recipe['source_sha256'].items():assert sha(ROOT/source)==digest,source
    latent=Path(recipe['latent_bank_root'])/'completion.json'
    if not latent.exists() or load(latent)['status']!='COMPLETED_ALL13230_T_IMAGE_LATENTS':raise Pending('Complete real13230-image T latent bank is required')
    cache_path=ROOT/'configs/idea_first_native_cache_v1.json'
    if not cache_path.exists():raise Pending('Actual native OOF cache configuration missing')
    cache=load(cache_path);complete_path=Path(cache['condition_cache_output_root'])/'completion.json'
    if not complete_path.exists():raise Pending('Actual full native OOF cache is incomplete')
    complete=load(complete_path)
    assert complete['status']=='COMPLETED_ALL10_CONCEPT_OOF_T_AND_MATCHED_D_CHAINS' and complete['subjects']==10
    assert complete['config_sha256']==sha(cache_path) and complete['T_OOF_queries']==132300
    full_path=ROOT/'configs/idea_first_ATMS_training_v1.json';full=load(full_path)
    assert cache['full_reference_root']==str(Path(full['output_root'])/'full')
    validation=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/error_validation/idea_first_ATMS_patience50_v1')
    response=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/error_validation/idea_first_full_UNet_response_v1/summary.json')
    if arm in ['Iso','Moment','Conditional']:
        if not (validation/'summary.json').exists():raise Pending('Actual all10 numerical residual/calibration fit is incomplete')
        numerical=load(validation/'summary.json');assert numerical['subjects']==10 and numerical['Dcheck_queries']==8300
    if arm=='Conditional':
        if not (numerical['gate']['native_energy_CI_upper_below_zero_vs_Iso_and_Moment'] and numerical['gate']['projected_energy_CI_upper_below_zero_vs_Iso_and_Moment']):
            return dict(status='NOT_RUN_FIXED_CONDITIONAL_CPU_PREREQUISITE_FAILED',arm=arm,source=str(validation/'summary.json'),source_sha256=sha(validation/'summary.json'),all_conditional_ideas_rejected=False,scientific_goal_achieved=False)
        if not response.exists():raise Pending('Actual full8300-query UNet response missing')
        if not load(response)['gate_passed']:return dict(status='NOT_RUN_FIXED_CONDITIONAL_UNET_RESPONSE_PREREQUISITE_FAILED',arm=arm,source=str(response),source_sha256=sha(response),all_conditional_ideas_rejected=False,scientific_goal_achieved=False)
    # The one fixed-precision probe found larger batches slower/OOM. Keep the
    # original complete evaluator and sampling batches; no metric protocol switch.
    profile=load(ROOT/'audit/idea_first_batch_cost_profile_v1.json')
    assert profile['status']=='T_ONLY_FIXED_PRECISION_BATCH_COSTS_MEASURED'
    assert profile['plan_sha256']==sha(ROOT/'configs/idea_first_batch_cost_probe_v1.json')
    assert profile['generation']['16']['seconds']>=profile['generation']['4']['seconds']
    assert profile['caption']['64']['seconds']>=profile['caption']['16']['seconds']
    assert profile['detection']['16']['status']=='OOM_NOT_PROMOTED'
    runtime_path=ROOT/'configs/idea_interface_runtime_v1.json'
    runtime_sources=['generate_idea_interface_images_v1.py','generate_reference_images.py','profile_standard_generator.py',
        'extract_image_features.py','score_full_manifest.py','semantic_evaluator.py','image_evaluation.py','locked_metrics.py',
        'audit_idea_image_pairing_v1.py','score_low_level.py','paired_analysis.py','write_idea_interface_report_v1.py']
    runtime=dict(version='idea_interface_fixed_original_runtime_v1',recipe_sha256=sha(recipe_path),resolution=512,steps=20,CFG=5.,IP_scale=1.,
        generation_batch=4,caption_batch=16,detection_batch=4,feature_extractor='extract_image_features.py',
        full_training_config_sha256=sha(full_path),native_cache_config_path=str(cache_path),native_cache_config_sha256=sha(cache_path),
        query_manifest_sha256=load(ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json')['query_manifest_sha256'],
        source_sha256={name:sha(ROOT/name) for name in runtime_sources},batch_profile_sha256=sha(ROOT/'audit/idea_first_batch_cost_profile_v1.json'),
        runtime_upgraded=False,reason='Larger batches slower or OOM in one prospectively fixed T probe')
    if runtime_path.exists():assert load(runtime_path)==runtime
    else:new(runtime_path,runtime)
    gen=load(DATA/'images/S1_full_native_batchsafe_v2/V/REF_REG_correct_R4/completion.json')
    features=load(DATA/'evaluation/V/REF_REG_correct_R4/features/completion.json')
    assert gen['queries']==features['queries']==16600 and gen['status']==features['status']=='COMPLETED'
    training_step=profile['training']['16']['steady_seconds_per_step']
    ceil60=lambda x:60*math.ceil(x/60)
    caps=dict(train=ceil60(training_step*math.ceil(132300/16)*1.2+180),T_readiness=120,
        generate=ceil60(gen['seconds']*1.1+120),identity_before=120,features=ceil60(features['seconds']*1.1+180),
        score=300,lowlevel=600,identity_after=120,report=300)
    caps['train']=min(caps['train'],recipe['total_training_seconds_cap_per_arm'])
    def resolved(name):
        return dict(version='idea_interface_arm_resolved_v1',arm=name,recipe_path=str(recipe_path),recipe_sha256=sha(recipe_path),
            runtime_path=str(runtime_path),runtime_sha256=sha(runtime_path),policy_path=str(policy_path),policy_sha256=sha(policy_path),
            cache_config_path=str(cache_path),cache_config_sha256=sha(cache_path),validation_output=str(validation),
            full_training_config_path=str(full_path),full_training_config_sha256=sha(full_path),
            training_root=str(DATA/'runs'/f'IDEA1_interface_{name}_v1'),
            T_readiness_root=str(DATA/'images/IDEA1_interface_v1/T_readiness'/f'IDEA1_{name}_correct_R4'),
            image_root=str(DATA/'images/IDEA1_interface_v1/V'/f'IDEA1_{name}_correct_R4'),
            evaluation_root=str(DATA/'evaluation/V'/f'IDEA1_{name}_correct_R4'),
            executor_path=str(ROOT/'run_idea_interface_arm_v2.py'),executor_sha256=sha(ROOT/'run_idea_interface_arm_v2.py'),
            phase_caps_seconds={k:v for k,v in caps.items() if k!='train' or name!='Frozen'},
            required_all_five_arms_preserved=True,scientific_goal_achieved=False,formal_scope_queries=16600,
            preparation_implementation_sha256=sha(__file__))
    requested=resolved(arm)
    paired_path=ROOT/'configs/idea_interface_RealPaired_resolved_v1.json'
    if arm!='RealPaired' and not paired_path.exists():raise Pending('Reserve the complete matched RealPaired reference before any other arm')
    frozen=resolved('Frozen')
    outstanding=[]
    for other in sorted((ROOT/'configs').glob('idea_interface_*_resolved_v1.json')):
        if other==release:continue
        value=load(other)
        for phase,seconds in value['phase_caps_seconds'].items():
            if not phase_done(value,phase):outstanding.append(dict(arm=value['arm'],phase=phase,seconds=seconds))
    for phase,seconds in requested['phase_caps_seconds'].items():
        if not phase_done(requested,phase):outstanding.append(dict(arm=arm,phase=phase,seconds=seconds))
    required=sum(x['seconds'] for x in outstanding)+900
    available=remaining();free=shutil.disk_usage(DATA).free
    # Reserve the unfinished complete populations, not a fixed free-space
    # threshold unrelated to the actual number of committed arms.
    configs={requested['arm']:requested}
    for other in sorted((ROOT/'configs').glob('idea_interface_*_resolved_v1.json')):
        value=load(other);configs[value['arm']]=value
    disk=storage_reservations(configs);required_free_bytes=disk['required_free_bytes']
    gate=dict(status='ACTUAL_COMPLETE_ARM_PLUS_SHARED_REFERENCE_AND_COMMITMENTS_BUDGET_CHECK',arm=arm,
        arm_lifecycle_funded=available>=required and free>=required_free_bytes,actual_tighter_remaining_seconds=available,
        required_remaining_seconds=required,outstanding_phase_reservations=outstanding,available_free_bytes=free,
        primary_matched_reference='RealPaired',Frozen_required_but_reservation_deferred=True,
        **{k:v for k,v in disk.items() if k!='available_free_bytes'},
        recipe_sha256=sha(recipe_path),runtime_sha256=sha(runtime_path),cache_config_sha256=sha(cache_path),
        latent_bank_completion_sha256=sha(latent),this_launch_training_cap_seconds=caps['train'],
        formal_image_scope_queries=16600,total_GPU_hour_cap=40,required_all_five_arms_preserved=True,
        scientific_goal_achieved=False,full_five_arm_campaign_funded=False,
        full_five_arm_plus_reference_estimated_cap_seconds=5*sum(caps.values())+sum(frozen['phase_caps_seconds'].values())+900,
        missing_controls_never_mean_hypothesis_rejection=True,
        UNet_response_summary_path=str(response) if response.exists() else None,
        UNet_response_summary_sha256=sha(response) if response.exists() else None,
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    fingerprint=hashlib.sha256(json.dumps(dict(ledger_sha256=sha(ROOT/'audit/compute_ledger.jsonl'),
        reservations=outstanding,free_MiB=int(free//2**20)),sort_keys=True).encode()).hexdigest()
    gate['resource_state_fingerprint']=fingerprint
    previous=sorted((ROOT/'audit').glob(f'idea_interface_{arm}_budget_*.json'))
    if previous and load(previous[-1]).get('resource_state_fingerprint')==fingerprint:
        return dict(status='UNCHANGED_RESOURCE_STATE_NO_NEW_BUDGET_DECISION',arm=arm,decision_path=str(previous[-1]),hypothesis_rejected=False)
    assert len(previous)<4,'At most four actual phase/resource-based budget decisions per arm'
    decision=ROOT/'audit'/f'idea_interface_{arm}_budget_{time.time_ns()}.json';new(decision,gate)
    if not gate['arm_lifecycle_funded']:
        return dict(status='NOT_RUN_CURRENT_COMPLETE_ARM_AND_COMMITMENTS_NOT_FUNDED',arm=arm,decision_path=str(decision),decision_sha256=sha(decision),hypothesis_rejected=False,scientific_goal_achieved=False)
    requested.update(status='RESOLVED_FUNDED_NOT_STARTED',budget_receipt_path=str(decision),budget_receipt_sha256=sha(decision),
        executor_path=str(ROOT/'run_idea_interface_arm_v2.py'),executor_sha256=sha(ROOT/'run_idea_interface_arm_v2.py'))
    new(release,requested)
    return dict(status='RESOLVED_FUNDED_NOT_STARTED',arm=arm,resolved_path=str(release),resolved_sha256=sha(release),
        full_five_arm_goal_complete=False,all_five_arms_required_for_scientific_conclusion=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=['Clean','Iso','Moment','Conditional','RealPaired','Frozen']);parser.add_argument('--policy',required=True);args=parser.parse_args()
    try:result=prepare(args.arm,args.policy)
    except Pending as error:
        print(json.dumps(dict(status='PREREQUISITES_PENDING_NO_ARM_RELEASE_CREATED',reason=str(error))),flush=True);raise SystemExit(2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
