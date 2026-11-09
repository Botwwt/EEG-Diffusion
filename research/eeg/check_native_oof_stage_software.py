"""Resource/proc-control tests; no scientific model, data cache or GPU job."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
from prepare_native_oof_stage import required_seconds
from run_native_oof_stage import stop_group


def main():
    policy=dict(role_training_max_seconds=1800,GPU_wrapper_margin_seconds=60,
        cache_sampling_multiplier=1.2,cache_IO_margin_seconds=180,
        point_cache_CPU_max_seconds=900,validation_CPU_max_seconds=1800,
        controls_runtime_multiplier=1.1,final_evaluation_reserve_seconds=18000,
        global_failure_margin_seconds=900)
    prior=required_seconds('REF_PRIOR_V',3600,.027725307794753462,policy)
    point=required_seconds('REF_REG',3600,.027725307794753462,policy)
    assert prior['cache_max_seconds']==6300 and prior['S3_GPU_reserved_seconds']==10080
    assert prior['total_required_remaining_seconds']==38700
    assert point['cache_GPU_reserved_seconds']==0 and point['prior_sampling_only_estimated_seconds']==0
    assert point['S3_GPU_reserved_seconds']==3720
    assert point['cache_CPU_wall_reserved_seconds']==900
    # Enough for all of S3 is still insufficient for S3+both full controls+final.
    misleading_remaining=prior['S3_complete_wall_reserved_seconds']+1000
    assert misleading_remaining<prior['total_required_remaining_seconds']
    # Only COMPLETE controls release their reservation; no query subsampling.
    after=required_seconds('REF_PRIOR_V',3600,.027725307794753462,policy,controls_left=0)
    assert prior['total_required_remaining_seconds']-after['total_required_remaining_seconds']==7920
    assert after['final_evaluation_reserved_seconds']==18000
    assert after['S3_GPU_reserved_seconds']==prior['S3_GPU_reserved_seconds']
    # A plausible actual runtime saving can permit a second, preserved budget
    # decision after both controls, without changing the scientific protocol.
    before_remaining=38160
    assert before_remaining<prior['total_required_remaining_seconds']
    assert before_remaining-7200>=after['total_required_remaining_seconds']
    process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    try:
        time.sleep(.1);stop_group(process);assert process.poll() is not None
    finally:
        if process.poll() is None:process.kill();process.wait()
    root=Path(__file__).parent
    report=dict(status='RESOURCE_AND_OWNED_PROCESS_CONTROL_SOFTWARE_CHECKS_PASSED',
        tests=['reserve_whole_stage_controls_final','prior_cache_rounding_and_margin',
            'point_cache_has_no_GPU_prior_term','completed_controls_release_only_their_cost',
            'cost_reassessment_after_actual_full_controls','only_owned_new_process_group_stopped'],
        synthetic_resource_example=prior,actual_EEG_used=False,actual_OOF_models_trained=False,
        actual_S3_funding_decision_made=False,GPU_used=False,
        source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ['prepare_native_oof_stage.py','run_native_oof_stage.py']},
        limitations=['No complete real reference selection is available yet; real preparation/run remains gated.',
            'The resource example is synthetic arithmetic, not actual EEG execution or reconstruction evidence.'])
    (root/'audit/native_oof_stage_software_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
