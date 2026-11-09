"""Resolve actual selected-chain OOF configs after complete strong references.

This prepares immutable, bounded jobs only when the ENTIRE S3 stage plus the
remaining full controls/final fit the tighter40h cap. It never starts a GPU job.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
PY='/root/miniconda3/bin/python'


class PrerequisitePending(RuntimeError):pass


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def write_new(path,value):
    assert not path.exists(),('Preserve existing stage/config',str(path))
    path.write_text(json.dumps(value,indent=2)+'\n')


def required_seconds(chain,measured_arm_seconds,prior_seconds_per_query,policy,controls_left=2):
    """Complete-scope reservation, including CPU wall and wrapper grace.

    The prior estimate is from the actual batch4 deployed sampler. The role
    hard caps do not reduce100/400epoch targets: a cap-hit stays incomplete.
    """
    assert chain in ['REF_REG','REF_PRIOR_V'] and measured_arm_seconds>0
    assert controls_left in [0,1,2]
    training=2*(policy['role_training_max_seconds']+policy['GPU_wrapper_margin_seconds'])
    if chain=='REF_PRIOR_V':
        assert prior_seconds_per_query>0
        cache_core=181800*prior_seconds_per_query
        cache=math.ceil((cache_core*policy['cache_sampling_multiplier']+policy['cache_IO_margin_seconds'])/300)*300
        cache_gpu=cache+policy['GPU_wrapper_margin_seconds']
        cache_cpu=0
    else:cache_core=0.;cache=policy['point_cache_CPU_max_seconds'];cache_gpu=0;cache_cpu=cache
    validation=policy['validation_CPU_max_seconds']
    controls=controls_left*measured_arm_seconds*policy['controls_runtime_multiplier']
    final=policy['final_evaluation_reserve_seconds'];margin=policy['global_failure_margin_seconds']
    return dict(training_GPU_reserved_seconds=training,cache_max_seconds=cache,
        prior_sampling_only_estimated_seconds=cache_core,cache_GPU_reserved_seconds=cache_gpu,
        cache_CPU_wall_reserved_seconds=cache_cpu,validation_CPU_wall_reserved_seconds=validation,
        S3_GPU_reserved_seconds=training+cache_gpu,
        S3_complete_wall_reserved_seconds=training+cache_gpu+cache_cpu+validation,
        full_two_control_reserved_seconds=controls,final_evaluation_reserved_seconds=final,
        global_failure_margin_seconds=margin,
        total_required_remaining_seconds=training+cache_gpu+cache_cpu+validation+controls+final+margin)


def prepare(policy_path):
    policy_path=Path(policy_path);policy=load(policy_path)
    assert policy['version']=='native_oof_stage_resource_policy_v1' and policy['total_GPU_hour_cap']==40
    for name,expected in policy['source_sha256'].items():assert sha(ROOT/name)==expected,('Locked stage source changed',name)
    post=ROOT/'audit/S1_postprocessing_state.json'
    if not post.exists() or load(post)['status']!='COMPLETED_BOTH_FULL_REFERENCES_POSTPROCESSED':
        raise PrerequisitePending('Both complete reference image metrics, low-level audits and paired/gallery postprocessing are required')
    probe=ROOT/'audit/projector_cost_probe_queue_state.json'
    terminal=['COST_PROBE_COMPLETED_FORMAL_ARMS_STILL_REQUIRE_BUDGET_GATE','FAILED_OR_BUDGET_GATE_STOPPED_COST_PROBE']
    if not probe.exists() or load(probe)['status'] not in terminal:
        raise PrerequisitePending('The already registered finite projector-cost queue must be terminal before scheduling another GPU stage')
    # Selection is implemented by the existing pre-result rule, never by cost.
    subprocess.run([PY,str(ROOT/'select_reference_for_controls.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    selected_path=ROOT/'audit/reference_control_selection.json';selection=load(selected_path)
    assert selection['status']=='SELECTED_ON_COMPLETE_V_BY_PREDECLARED_RULE'
    assert selection['subjects']==10 and selection['exact_V_images_used_per_method']==16600
    assert selection['repeat_protocol']=='R4' and not selection['final_test_unlocked']
    method=selection['selected_reference'];assert method in ['REF_REG','REF_PRIOR_V']
    for item in selection['inputs'].values():
        assert sha(item['summary_path'])==item['summary_sha256']
        assert sha(item['image_manifest'])==item['image_manifest_sha256']
    subprocess.run([PY,str(ROOT/'refresh_status.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    progress=load(ROOT/'audit/current_progress.json')
    if any(j['alive'] for j in progress['live_GPU_jobs']):raise PrerequisitePending('A registered GPU job remains live')
    assert not progress['final_test_unlocked']
    remaining=progress['remaining_authorized_hours']*3600
    arm=f'{method}_correct_R4'
    completed_controls=[]
    for intervention in ['permuted','missing']:
        folder=DATA/'evaluation/V'/f'{method}_{intervention}_R4'
        summary=folder/'metrics/summary.json';low_level=folder/'low_level/summary.json'
        if summary.exists() and low_level.exists():
            s=load(summary);quality=load(low_level)
            assert s['status']=='COMPLETED' and s['queries']==16600 and s['subjects']==10
            assert s['method']==method and s['intervention']==intervention and s['repeat_protocol']=='R4'
            assert quality['status']=='COMPLETED' and quality['verified_and_decoded']==16600
            assert quality['missing_count']==quality['corrupt_or_mismatched_count']==0
            completed_controls.append(dict(intervention=intervention,summary_path=str(summary),summary_sha256=sha(summary)))
    decisions=sorted((ROOT/'audit').glob('S3_OOF_budget_check_*.json'))
    assert len(decisions)<policy['maximum_budget_decisions'],'No unbounded budget-probing loop'
    if decisions:
        previous=load(decisions[-1])
        assert not previous['full_two_fold_training_and_condition_sampling_funded']
        assert len(completed_controls)>len(previous['completed_full_controls']), 'Recheck only after completing additional full controls'
    gen=DATA/'images/S1_full_native_batchsafe_v2/V'/arm/'completion.json'
    feature=DATA/'evaluation/V'/arm/'features/completion.json'
    g=load(gen);f=load(feature)
    assert g['status']==f['status']=='COMPLETED' and g['queries']==f['queries']==16600
    measured=g['seconds']+f['seconds']
    cost_path=ROOT/'audit/native_prior_cache_cost_profile.json';cost=load(cost_path)
    assert cost['status']=='EXACT_T_NATIVE_PRIOR_CACHE_COST_MEASURED' and cost['batch']==4
    assert cost['counts']['sample_evaluations']==100*cost['queries']
    costs=required_seconds(method,measured,cost['seconds_per_query'],policy,2-len(completed_controls))
    free=shutil.disk_usage(DATA).free
    gate=dict(status='COMPLETE_SCOPE_BUDGET_CHECKED_BEFORE_OOF',UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        selected_reference=method,total_GPU_hour_cap=40,actual_tighter_remaining_seconds=remaining,
        full_two_fold_training_and_condition_sampling_funded=(remaining>=costs['total_required_remaining_seconds']
            and free>=policy['minimum_free_bytes_before_stage']),
        all10_subjects=True,T_OOF_queries=132300,D_queries_per_teacher=16500,D_teachers=3,
        role_epoch_caps=dict(encoder=100,encoder_patience=20,prior=400),role_training_clock_cap_seconds=policy['role_training_max_seconds'],
        resource_cap_hit_is_incomplete_not_hypothesis_rejection=True,
        measured_full_selected_correct_arm_seconds=measured,measured_generation= dict(path=str(gen),sha256=sha(gen)),
        measured_evaluation=dict(path=str(feature),sha256=sha(feature)),
        prior_cache_cost_profile_sha256=sha(cost_path),selection_sha256=sha(selected_path),policy_sha256=sha(policy_path),
        available_free_bytes=free,required_free_bytes=policy['minimum_free_bytes_before_stage'],
        projector_cost_queue_status=load(probe)['status'],**costs)
    gate['completed_full_controls']=completed_controls
    # Empirical image/evaluator cost estimate, before interface training.
    # Wall times can vary; this is not a mathematical runtime lower bound.
    five_images=5*measured
    remaining_for_interface=remaining-costs['S3_complete_wall_reserved_seconds']-costs['full_two_control_reserved_seconds']-costs['final_evaluation_reserved_seconds']-costs['global_failure_margin_seconds']
    gate['S4_full_matched_arms_funded']=False
    gate['S4_five_full_image_evaluation_estimate_seconds']=five_images
    gate['S4_estimate_is_not_mathematical_runtime_lower_bound']=True
    gate['S4_remaining_after_S3_controls_final_margin_seconds']=remaining_for_interface
    gate['S4_status']=('NOT_RUN_ESTIMATED_COMPLETE_IMAGE_COST_EXCEEDS_REMAINING_BUDGET' if five_images>remaining_for_interface
                       else 'FULL_ADAPTER_TRAINING_SCHEDULE_AND_BUDGET_REMAIN_REQUIRED')
    decision_path=ROOT/'audit'/f'S3_OOF_budget_check_{time.time_ns()}.json'
    write_new(decision_path,gate)
    if not gate['full_two_fold_training_and_condition_sampling_funded']:
        return dict(status='NOT_RUN_COMPLETE_S3_PLUS_FULL_CONTROLS_FINAL_NOT_FUNDED',gate_path=str(decision_path),
            gate_sha256=sha(decision_path),hypothesis_rejected=False,training_started=False,costs=costs)
    path=ROOT/'audit/S3_OOF_budget_gate.json'
    write_new(path,gate|dict(budget_decision_record=str(decision_path),budget_decision_record_sha256=sha(decision_path)))
    base_path=ROOT/'configs/full_reference_resolved.json';base=load(base_path)
    assert sha(ROOT/'train_full_reference.py')==base['trainer_source_sha256']
    assert sha(ROOT/'configs/split_manifest.json')==base['split_sha256']
    validation_plan=ROOT/'configs/native_error_channel_validation_plan_v1.json'
    for name,value in load(validation_plan)['source_sha256'].items():assert sha(ROOT/name)==value
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());run_id=f'S3_native_oof_{method}_{stamp}'
    output=DATA/'runs'/run_id;cache=DATA/'conditions'/run_id;validation=DATA/'error_validation'/run_id
    assert not any(p.exists() for p in [output,cache,validation])
    config=base|dict(version='concept_oof_selected_native_chain_v1',condition_chain=method,
        prior_enabled=method=='REF_PRIOR_V',prior_prediction_type='v_prediction' if method=='REF_PRIOR_V' else 'not_used',
        output_root=str(output),oof_trainer_sha256=sha(ROOT/'train_native_oof.py'),
        base_trainer_sha256=sha(ROOT/'train_full_reference.py'),reference_selection_sha256=sha(selected_path),
        budget_gate_sha256=sha(path),role_max_wall_seconds={r:policy['role_training_max_seconds'] for r in ['oof0','oof1']},
        explicit_deviations_from_official_develop_entrypoint=[
            'Same complete ATMS class and unique seed1729; ten separate subject models, concept-held-out T roles.',
            'Full per-role training data every epoch; encoder max100/patience20, matching current full reference (benchmark.sh patience50 differs).',
            'V selects checkpoints; no T-held-out/Dfit/Dcheck gradient or prior-training exposure.',
            'If selected chain includes prior: full400epoch velocity prediction and fixed-V velocity MSE checkpoint selection; equivalent epsilon is not selector.',
            'Role wall cap is a resource limit; a cap hit stays incomplete/resumable and cannot create a valid OOF cache.',
            'No published full multipipeline reproduction or reconstruction success is implied.'])
    training_path=ROOT/'configs'/f'{run_id}_training.json';write_new(training_path,config)
    cache_config={k:config[k] for k in ['seed','condition_chain','output_root','split_sha256','visual_features_path','visual_features_sha256']}
    cache_config.update(cache_builder_sha256=sha(ROOT/'build_native_oof_cache.py'),
        oof_training_config_path=str(training_path),oof_training_config_sha256=sha(training_path),
        reference_selection_sha256=sha(selected_path),budget_gate_sha256=sha(path),
        inference_batch=4,coordinate_space='raw_pooled_CLIP_H_1024_before_native_IP_projection',
        condition_cache_output_root=str(cache),full_reference_root=str(Path(base['output_root'])/'full'),
        condition_cache_max_seconds=costs['cache_max_seconds'],minimum_free_bytes_before_cache=policy['minimum_free_bytes_before_cache'])
    cache_path=ROOT/'configs'/f'{run_id}_cache.json';write_new(cache_path,cache_config)
    resolved=dict(status='RESOLVED_AND_FUNDED_BUT_NOT_STARTED',run_id=run_id,selected_reference=method,
        policy_path=str(policy_path),policy_sha256=sha(policy_path),gate_path=str(path),gate_sha256=sha(path),
        training_config_path=str(training_path),training_config_sha256=sha(training_path),
        cache_config_path=str(cache_path),cache_config_sha256=sha(cache_path),
        validation_plan_path=str(validation_plan),validation_plan_sha256=sha(validation_plan),validation_output=str(validation),
        job_max_seconds=dict(oof0=policy['role_training_max_seconds']+policy['GPU_wrapper_margin_seconds'],
            oof1=policy['role_training_max_seconds']+policy['GPU_wrapper_margin_seconds'],
            cache=costs['cache_max_seconds']+(policy['GPU_wrapper_margin_seconds'] if method=='REF_PRIOR_V' else 0),
            validation=policy['validation_CPU_max_seconds']),
        stage_GPU_flags=dict(oof0=True,oof1=True,cache=method=='REF_PRIOR_V',validation=False),
        retained_full_controls_final_margin_seconds=costs['full_two_control_reserved_seconds']+costs['final_evaluation_reserved_seconds']+costs['global_failure_margin_seconds'],
        S4_may_start=False,final_test_may_unlock=False,source_sha256=sha(__file__))
    result=ROOT/'configs'/f'{run_id}_resolved.json';write_new(result,resolved)
    return resolved|dict(resolved_path=str(result),resolved_sha256=sha(result))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);args=parser.parse_args()
    try:result=prepare(args.policy)
    except PrerequisitePending as exc:
        print(json.dumps(dict(status='PREREQUISITES_PENDING_NO_OOF_CONFIG_OR_JOB_CREATED',reason=str(exc))),flush=True)
        raise SystemExit(2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
