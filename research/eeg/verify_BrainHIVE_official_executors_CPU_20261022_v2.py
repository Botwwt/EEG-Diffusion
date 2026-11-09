"""Retain passing CPU work; correct only the export early-refusal command."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import verify_BrainHIVE_official_executors_CPU_20261022_v1 as V


def main():
    output = V.OUTPUT / 'server_execution_v2.json'
    assert not output.exists()
    prior_path = V.OUTPUT / 'server_execution.json'
    prior = json.loads(prior_path.read_bytes())
    assert V.sha(V.__file__) == prior['source_sha256']
    assert prior['status'] == 'OFFICIAL_EXECUTOR_CPU_CHECKS_FAILED_PARTIAL_EVIDENCE_RETAINED'
    stages = prior['stages']
    assert [s['name'] for s in stages[:2]] == ['training_checks', 'execution_export_checks']
    assert [s['exit_code'] for s in stages] == [0, 0, 1, 2]
    for stage in stages:
        ref = stage['log']; raw = Path(ref['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref['sha256'] and len(raw) == ref['bytes']
    assert b'Ran 8 tests' in Path(stages[0]['log']['path']).read_bytes()
    checked = json.loads((V.OUTPUT / 'execution_export_checks.json').read_bytes())
    assert len(checked['checks']) == 10 and checked['actual_new_fits_or_official_condition_caches'] == 0
    assert b'--execution-review' in Path(stages[3]['log']['path']).read_bytes()
    for name, digest in V.PINS.items(): assert V.sha(V.ROOT / name) == digest
    missing = V.OUTPUT / 'NO_OFFICIAL_EXECUTION_REVIEW_EXISTS.json'
    assert not missing.exists()
    before = V.headroom(); started = time.monotonic()
    command = [sys.executable, '-B', str(V.ROOT / 'export_BrainHIVE_official_OOF_conditions_v1.py'),
        '--plan-directory', str(V.ROOT / 'audit/official_fullfit_metadata_plan_server_20261022_v1'),
        '--task-id', 'sub-01/official_oof0', '--execution-review', str(missing)]
    completed = subprocess.run(command, cwd=V.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                               capture_output=True, timeout=20)
    raw = completed.stdout + completed.stderr
    log = V.OUTPUT / 'export_actual_missing_review_corrected_command.log'
    assert not log.exists(); log.write_bytes(raw)
    assert completed.returncode == 1 and b'Official method/resource execution review does not exist' in raw
    assert not Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/official_fullfit_v1').exists()
    frozen = json.loads((V.ROOT / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json').read_bytes())['source_sha256']
    for name, digest in dict(frozen, **V.PINS).items(): assert V.sha(V.ROOT / name) == digest
    assert V.sha(V.ROOT / 'results.tsv') == '2ee905cc31949671c3068f04c9dbd05046ed4ae9888b05c9ca7cb08d781fb010'
    record = dict(status='OFFICIAL_EXECUTOR_CPU_CHECKS_AND_BOTH_REAL_EARLY_REFUSALS_VERIFIED_NO_ADMISSION',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=V.sha(__file__),
        prior_actual_partial_execution=dict(path=str(prior_path), sha256=V.sha(prior_path)),
        passing_training_and_export_checks_reused_without_rerun=True, training_CPU_checks=8, execution_export_CPU_groups=10,
        actual_entry_missing_review_refusals=2, corrected_command=command, corrected_exit_code=completed.returncode,
        corrected_log=dict(path=str(log), sha256=V.sha(log), bytes=len(raw)),
        seconds=time.monotonic() - started, CPU_headroom_before_bytes=before, CPU_headroom_after_bytes=V.headroom(),
        implementation_source_sha256=V.PINS, unchanged_frozen_sources=29,
        original_command_error='Export CLI required --execution-review; original command reached argparse only, not the admission gate.',
        method_lock_or_execution_admission_issued=False, official_fit_updates=0, actual_official_condition_exports=0,
        actual_dataset_or_pretrained_payloads_read=False, actual_GPU_jobs=0, official_or_new_V_metric_values_read=False,
        actual_CUDA_BF16_or_full_dataset_fitting_verified=False, official_output_namespace_created=False,
        scientific_goal_complete=False)
    output.write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print(json.dumps(dict(status=record['status'], receipt=str(output), sha256=V.sha(output))))


if __name__ == '__main__':
    main()
