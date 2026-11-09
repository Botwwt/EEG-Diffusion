"""One-time actual registration and launch of the finite matched-image queue."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PY='/root/miniconda3/bin/python'


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    plan_path=ROOT/'configs/idea_interface_campaign_plan_v2.json'
    state_path=ROOT/'audit/idea_interface_campaign_state_v2.json'
    assert not plan_path.exists() and not state_path.exists(),'Preserve registered plan/controller history'
    assert not (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists()
    for flag in ['STOP_IDEA_INTERFACE_CAMPAIGN','STOP_IDEA_INTERFACE_ARM','STOP_IDEA_INTERFACE_TRAINING']:
        assert not (ROOT/flag).exists(),flag
    preflight=json.loads((ROOT/'audit/idea_interface_campaign_software_preflight_v2.json').read_text())
    assert preflight['status']=='CLI_IMPORTS_AND_ACTUAL_ACTIVE_GPU_GATE_PASSED_NO_ARM_RELEASED'
    old_state=json.loads((ROOT/'audit/idea_interface_campaign_state_v1.json').read_text())
    assert old_state['status']=='STOPPED_WAITING_BEFORE_ANY_ARM_FOR_USER_IDEA_PRIORITY_REFINEMENT'
    assert not list((ROOT/'configs').glob('idea_interface_*_resolved_v1.json'))
    policy_path=ROOT/'configs/idea_interface_execution_policy_v2.json';policy=json.loads(policy_path.read_text())
    policy.update(registered_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),actual_Dcheck_results_seen_at_registration=True)
    policy_path.write_text(json.dumps(policy,indent=2)+'\n')
    sources=['audit_idea_image_pairing_v1.py','prepare_idea_interface_arm_v2.py','run_idea_interface_arm_v2.py',
        'write_idea_interface_report_v1.py','queue_idea_interface_campaign_v2.py','run_gpu_job.py','run_native_oof_stage.py',
        'configs/idea_interface_execution_policy_v2.json','configs/idea_interface_training_recipe_v1.json',
        'configs/idea_image_preparation_queue_v1.json','prepare_idea_interface_arm_v1.py',*policy['source_sha256']]
    sources=list(dict.fromkeys(sources))
    for source,digest in policy['source_sha256'].items():assert sha(ROOT/source)==digest,source
    now=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    plan=dict(version='idea_interface_finite_campaign_v2',registered_utc=now,
        policy_path=str(policy_path),policy_sha256=sha(policy_path),training_priority=policy['prospective_arm_priority'],
        image_priority=['Conditional','RealPaired','Moment','Iso','Clean'],funding_rounds=2,
        total_GPU_hour_cap=40,training_seed=1729,formal_V_queries_per_arm=16600,images_per_query=1,
        complete_lifecycle_and_committed_reference_funding_required=True,
        no_score_based_resource_decisions=True,no_automatic_retries=True,no_final_unlock=True,
        actual_CPU_Dcheck_results_seen_at_registration=True,actual_full_UNet_results_seen_at_registration=False,
        scientific_goal_complete=False,missing_controls_remain_required=True,full_objective_unchanged=True,
        primary_complete_reference='RealPaired',Frozen_required_deferred_until_after_idea_rounds=True,
        source_sha256={name:sha(ROOT/name) for name in sources})
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    git=lambda args:subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,check=True)
    staged=git(['diff','--cached','--name-only']).stdout.strip();assert not staged,('Do not commit unrelated staged changes',staged)
    files=['audit_idea_image_pairing_v1.py','prepare_idea_interface_arm_v2.py','run_idea_interface_arm_v2.py',
        'write_idea_interface_report_v1.py','queue_idea_interface_campaign_v2.py','register_and_launch_idea_interface_v2.py',
        'configs/idea_interface_execution_policy_v2.json','configs/idea_interface_campaign_plan_v2.json',
        'audit/idea_interface_campaign_software_preflight_v2.json','research/IDEA_FIRST_REFERENCE_RESERVATION_PRIORITY_V2_20261018.md']
    git(['add','--',*files]);git(['-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'])
    git(['-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','-m','Run finite idea-first matched image adapter campaign with complete-arm budgets'])
    commit=git(['rev-parse','HEAD']).stdout.strip()
    log=ROOT/'logs/idea_interface_campaign_controller_v2.log';assert not log.exists()
    with log.open('xb') as file:
        process=subprocess.Popen([PY,str(ROOT/'queue_idea_interface_campaign_v2.py'),'--plan',str(plan_path)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(status='LAUNCHED_WAITING_FOR_REAL_PREREQUISITES',controller_pid=process.pid,
        plan_path=str(plan_path),plan_sha256=sha(plan_path),code_commit=commit,log=str(log),UTC=now,
        scientific_goal_complete=False,formal_image_gain_established=False)
    (ROOT/'audit/idea_interface_campaign_launch_v2.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
