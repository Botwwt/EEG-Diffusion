"""Resolve full EEG controls after actual S3 and matched-interface cost review.

The original1.5h OOF allowance is never double-reserved or treated as funding.
No method selection, interface-training permission or final unlock is invented.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time

from run_native_oof_stage import ROOT,PY,load,sha,fresh_remaining

DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


class Pending(RuntimeError):pass


def check_sources(policy):
    for name,digest in policy['source_sha256'].items():assert sha(ROOT/name)==digest,('Frozen controls source changed',name)


def reserve_seconds(measured_complete_arm_seconds,controls_left,policy):
    assert measured_complete_arm_seconds>0 and controls_left in [0,1,2]
    return (controls_left*measured_complete_arm_seconds*policy['control_runtime_multiplier']
            +policy['final_reserve_seconds']+policy['global_margin_seconds'])


def prepare(policy_path):
    policy_path=Path(policy_path);policy=load(policy_path);check_sources(policy)
    assert policy['total_GPU_hour_cap']==40 and policy['required_queries_per_control']==16600
    output=ROOT/'audit/EEG_controls_stage_release_v2.json'
    if output.exists():
        value=load(output);assert value['policy_sha256']==sha(policy_path)
        assert sha(ROOT/'audit/reference_control_selection.json')==value['reference_selection_sha256']
        return value|dict(release_path=str(output),release_sha256=sha(output))
    post=ROOT/'audit/S1_postprocessing_state.json'
    if not post.exists() or load(post)['status']!='COMPLETED_BOTH_FULL_REFERENCES_POSTPROCESSED':
        raise Pending('Both complete primary reference scores and full CPU audits/paired gallery are required')
    queue=ROOT/'audit/native_oof_stage_queue_state.json'
    if not queue.exists():raise Pending('Actual registered OOF queue state is required')
    native=load(queue)
    if native['status'] not in ['COMPLETE_BUDGET_DECISION_NO_OOF_STARTED','COMPLETE_FUNDED_NATIVE_OOF_DRIVER_REVIEW_REQUIRED']:
        raise Pending('OOF queue is not normally terminal; finish it or inspect its concrete failure before release')
    probe=ROOT/'audit/projector_cost_probe_queue_state.json'
    if not probe.exists() or load(probe)['status'] not in ['COST_PROBE_COMPLETED_FORMAL_ARMS_STILL_REQUIRE_BUDGET_GATE','FAILED_OR_BUDGET_GATE_STOPPED_COST_PROBE']:
        raise Pending('The registered actual UNet projector cost probe must be terminal')
    subprocess.run([PY,str(ROOT/'select_reference_for_controls.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    selection_path=ROOT/'audit/reference_control_selection.json';selection=load(selection_path)
    assert selection['status']=='SELECTED_ON_COMPLETE_V_BY_PREDECLARED_RULE'
    method=selection['selected_reference'];assert method in ['REF_REG','REF_PRIOR_V']
    preparation=load(native['preparation_result']);stage=None;stage_proof=None;validation=None;validation_proof=None
    if native['status']=='COMPLETE_BUDGET_DECISION_NO_OOF_STARTED':
        assert preparation['status']=='NOT_RUN_COMPLETE_S3_PLUS_FULL_CONTROLS_FINAL_NOT_FUNDED'
        assert load(preparation['gate_path'])['selected_reference']==method
        S3_status='NOT_RUN_COMPLETE_SCOPE_BUDGET';S3_complete=False
    else:
        resolved=Path(native['resolved']);r=load(resolved)
        assert preparation['resolved_sha256']==sha(resolved) and r['selected_reference']==method
        path=ROOT/'audit'/f'{r["run_id"]}_state.json';stage=load(path)
        assert stage['resolved_sha256']==sha(resolved)
        stage_proof=dict(path=str(path),sha256=sha(path));S3_status=stage['status']
        accepted=['COMPLETED_ALL10_OOF_CACHE_AND_CPU_DCHECK','STOPPED_BEFORE_NEXT_PHASE_COMPLETE_SCOPE_BUDGET_NOT_AVAILABLE',
                  'FAILED_OR_BUDGET_STOPPED_WITH_SAVED_OUTPUTS','FAILED_STOPPED_DEPENDENT_PHASES']
        if S3_status not in accepted:raise Pending('Actual OOF driver remains active or requires technical inspection')
        S3_complete=S3_status=='COMPLETED_ALL10_OOF_CACHE_AND_CPU_DCHECK'
        if S3_complete:
            path=Path(stage['CPU_validation_summary_path']);assert sha(path)==stage['CPU_validation_summary_sha256']
            validation=load(path);assert validation['subjects']==10 and validation['Dcheck_queries']==8300 and validation['Dfit_queries']==8200
            validation_proof=dict(path=str(path),sha256=sha(path))
    arm=f'{method}_correct_R4';generation_path=DATA/'images/S1_full_native_batchsafe_v2/V'/arm/'completion.json'
    features_path=DATA/'evaluation/V'/arm/'features/completion.json'
    g=load(generation_path);f=load(features_path)
    assert g['status']==f['status']=='COMPLETED' and g['queries']==f['queries']==16600
    measured=g['seconds']+f['seconds'];remaining=fresh_remaining()
    controls_final_margin=reserve_seconds(measured,2,policy)
    available_for_S4=remaining-controls_final_margin
    five_images=policy['matched_interface_arms']*measured
    five_generation_only=policy['matched_interface_arms']*g['seconds']
    CPU_gate_failed=(S3_complete and not (validation['gate']['native_energy_CI_upper_below_zero_vs_Iso_and_Moment']
        and validation['gate']['projected_energy_CI_upper_below_zero_vs_Iso_and_Moment']))
    if not S3_complete:S4_status='NOT_RUN_ACTUAL_S3_PREREQUISITES_INCOMPLETE'
    elif CPU_gate_failed:S4_status='NOT_RUN_PREDECLARED_CPU_CHANNEL_FIDELITY_GATE_FAILED'
    elif five_images>available_for_S4:S4_status='NOT_RUN_MEASURED_COMPLETE_MATCHED_IMAGE_SCOPE_UNFUNDED'
    else:S4_status='POTENTIALLY_FUNDED_REQUIRES_FULL_UNET_RESPONSE_AND_SUFFICIENT_TRAINING_REVIEW'
    may_release=not S4_status.startswith('POTENTIALLY_FUNDED')
    cost_path=ROOT/'audit/native_projector_training_cost.json';cost=load(cost_path) if cost_path.exists() else None
    if cost is not None:
        assert cost['status']=='T_ONLY_PROJECTOR_COST_PROBE_COMPLETED' and cost['projector_trainable_parameters']==8400896
        assert cost['updated_weights_discarded'] and not cost['formal_training']
    result=dict(status='CONTROL_STAGE_RELEASED_AFTER_ACTUAL_S3_S4_REVIEW' if may_release else 'CONTROL_STAGE_NOT_RELEASED_S4_REVIEW_REQUIRED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selected_reference=method,
        policy_path=str(policy_path),policy_sha256=sha(policy_path),source_sha256=sha(__file__),
        reference_selection_sha256=sha(selection_path),S3_queue_state_sha256=sha(queue),
        S3_actual_status=S3_status,S3_complete=S3_complete,S3_stage_proof=stage_proof,CPU_validation_proof=validation_proof,
        S3_remaining_automatic_reservation_seconds=0,S3_accounted_by_actual_GPU_ledger=True,
        S3_second_budget_review_after_completed_controls_not_authorized_by_this_stage=True,
        S4_status=S4_status,S4_interface_training_started=False,S4_image_gain_established=False,
        measured_complete_correct_arm_seconds=measured,
        measured_generation_proof=dict(path=str(generation_path),sha256=sha(generation_path)),
        measured_evaluation_proof=dict(path=str(features_path),sha256=sha(features_path)),
        actual_remaining_seconds=remaining,remaining_controls_final_margin_expected_seconds=controls_final_margin,
        available_for_S4_after_required_controls_final_margin_seconds=available_for_S4,
        matched_interface_arms=policy['matched_interface_arms'],matched_images_and_evaluation_expected_seconds=five_images,
        matched_generation_only_expected_seconds=five_generation_only,
        estimates_are_not_mathematical_runtime_lower_bounds=True,
        projector_probe_queue_status=load(probe)['status'],projector_cost_proof=dict(path=str(cost_path),sha256=sha(cost_path)) if cost is not None else None,
        controls_may_proceed_to_fresh_complete_scope_budget_gate=may_release,
        controls_per_arm_queries=16600,subjects=10,seed=1729,final_reserve_seconds=policy['final_reserve_seconds'],
        final_test_unlocked=False,final_test_may_unlock=False,hypothesis_rejected_by_budget=False)
    target=output if may_release else ROOT/'audit/EEG_controls_stage_S4_review_required_v2.json'
    with target.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    return result|dict(release_path=str(target),release_sha256=sha(target))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);args=parser.parse_args()
    try:value=prepare(args.policy)
    except Pending as error:
        print(json.dumps(dict(status='PREREQUISITES_PENDING_NO_CONTROL_RELEASE_OR_GPU_JOB_CREATED',reason=str(error))),flush=True)
        raise SystemExit(2)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
