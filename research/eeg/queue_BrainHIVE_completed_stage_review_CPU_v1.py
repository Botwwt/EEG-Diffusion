"""Review the complete, prespecified comparison once; never tune on score peeks.

Writes a bounded keep/stop recommendation, not an automatic unregistered fit.
Existing jobs and their immutable recipes are never changed by this observer.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
MAIN = WORK / 'audit/main_storage_continuation_20261021_v1.json'
DIAG = WORK / 'audit/native_function_T32_queue_v2.json'
COMPARISON = WORK / 'comparison/bounded_v1/summary.json'
STATE = WORK / 'audit/completed_stage_review_CPU_v1.json'
RESULT = WORK / 'audit/completed_stage_decision_CPU_v1.json'
RECIPE = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
EXPECTED_RECIPE = '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
PERIOD = 1200
MAX_WAIT = 7 * 86400
TERMINAL_MAIN = {
    'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',
    'FINITE_MAIN_STORAGE_CONTINUATION_PARTIAL_WITH_EXPLICIT_FAILURES',
    'STORAGE_FORECAST_STOPPED_PRESERVED_NO_CASCADING_JOBS',
    'FAILED_MAIN_STORAGE_CONTINUATION_PRESERVED_NEEDS_DIAGNOSIS',
}
TERMINAL_DIAG = {
    'FROZEN_FUNCTIONAL_DIAGNOSTIC_SAVED_NO_FORMAL_COMPONENT_TRAINING',
    'DIAGNOSTIC_FAILED_WITH_PRESERVED_LOG_NO_AUTOMATIC_RETRY',
    'STOPPED_WITH_EXPLICIT_DIAGNOSTIC_FAILURE',
}


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    value = dict(value, UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def process_exists(pid):
    try:
        text = (Path('/proc') / str(int(pid)) / 'stat').read_text()
        return text[text.rfind(')')+2:].split()[0] not in ['Z', 'X']
    except (OSError, TypeError, ValueError):
        return False


def decision(report):
    assert report['status'] == 'ALL_THREE_FULL16600_QUERY_IMAGE_ARMS_PAIRED_COMPARISON_COMPLETE'
    assert {r['arm'] for r in report['table']} == {'risk', 'ordinary', 'parent'}
    assert all(r['queries'] == 16600 and r['repeat'] == 'R4' and r['gallery'] == 1660 for r in report['table'])
    assert report['single_training_seed'] == 1729
    differences = report['paired_differences']
    qualified = []
    negative = []
    for reference in ['parent', 'ordinary']:
        delta = differences['risk_minus_' + reference]
        qualified.append(delta['DINO2AFC']['mean'] >= .01
            and delta['DINO2AFC']['CI95_subject_and_concept_bootstrap'][0] > 0
            and delta['SEED']['mean'] > 0
            and delta['SEED']['CI95_subject_and_concept_bootstrap'][0] > 0
            and delta['ObjectF1']['mean'] >= 0)
        negative.append(delta['DINO2AFC']['CI95_subject_and_concept_bootstrap'][1] < 0
            or delta['SEED']['CI95_subject_and_concept_bootstrap'][1] < 0)
    assert bool(report['practical_numerical_gain_signal']) == all(qualified)
    if all(qualified):
        label = 'PROVISIONAL_IMAGE_GAIN_RUN_PREREGISTERED_MECHANISM_CONTROLS'
        action = 'Retain current checkpoint; gate controls and EEG dependence checks before any risk-mechanism or SOTA claim.'
    elif any(negative):
        label = 'CURRENT_RISK_CONFIGURATION_HAS_NEGATIVE_COMPARATIVE_EVIDENCE'
        action = 'Do not refit or expand this risk configuration. Review T-only functional/radius diagnostics and switch the targeted hypothesis.'
    else:
        label = 'CURRENT_RISK_IMAGE_GAIN_NOT_ESTABLISHED'
        action = 'Do not extend this run or search its radius/checkpoints on V. Use frozen T diagnostics to select the next registered hypothesis.'
    return dict(status=label, next_action=action, original_gain_gate_unchanged=True,
        paired_differences=differences, table=report['table'], mechanism_confirmed=False,
        independent_SOTA_established=False, training_or_regeneration_launched=False,
        no_new_test_based_hyperparameter_search=True)


def main():
    with (WORK / 'audit/completed_stage_review_CPU_v1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not STATE.exists() and not RESULT.exists()
        own_sha = sha(Path(__file__).resolve())
        started = time.monotonic()
        base = dict(pid=os.getpid(), interval_seconds=PERIOD, maximum_seconds=MAX_WAIT,
                    source_sha256=own_sha, GPU_forward_calls=0, model_mutations=False,
                    completed_comparison_read_count=0)
        result = None
        while time.monotonic() - started < MAX_WAIT:
            assert sha(RECIPE) == EXPECTED_RECIPE and sha(Path(__file__).resolve()) == own_sha
            main_state = read(MAIN) or {}
            diag_state = read(DIAG) or {}
            if main_state.get('status') in TERMINAL_MAIN and result is None:
                if main_state['status'] == 'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED':
                    report = read(COMPARISON)
                    assert report is not None
                    result = decision(report)
                    result.update(comparison_path=str(COMPARISON), comparison_sha256=sha(COMPARISON),
                                  completed_comparison_read_count=1)
                else:
                    result = dict(status='COMPARISON_INCOMPLETE_NO_SCIENTIFIC_REJECTION',
                        main_status=main_state['status'], failures=main_state.get('failures'),
                        next_action='Repair documented execution prerequisites; partial or failed jobs do not reject the scientific hypothesis.',
                        completed_comparison_read_count=0, training_or_regeneration_launched=False,
                        mechanism_confirmed=False, independent_SOTA_established=False)
                save(RESULT, result)
                base['completed_comparison_read_count'] = result['completed_comparison_read_count']
            if result is not None and diag_state.get('status') in TERMINAL_DIAG:
                diag_summary = WORK / 'audit/native_function_T32_v2/summary.json'
                result['frozen_T_diagnostic_status'] = diag_state['status']
                result['frozen_T_diagnostic_summary_path'] = str(diag_summary) if diag_summary.exists() else None
                result['frozen_T_diagnostic_summary_sha256'] = sha(diag_summary) if diag_summary.exists() else None
                result['diagnostic_interpretation_required_before_new_fit'] = True
                save(RESULT, result)
                save(STATE, dict(base, status='COMPLETE_STAGE_REVIEW_AND_FROZEN_DIAGNOSTIC_REFERENCE_SAVED', decision=str(RESULT)))
                return
            if not process_exists(main_state.get('pid')) and main_state.get('status') not in TERMINAL_MAIN:
                save(STATE, dict(base, status='STOPPED_ORPHAN_MAIN_NO_SCIENTIFIC_RESULT', main_status=main_state.get('status')))
                return
            save(STATE, dict(base, status='WAITING_FOR_COMPLETED_COMPARISON_AND_FROZEN_DIAGNOSTIC',
                main_status=main_state.get('status'), T32_status=diag_state.get('status'), decision_saved=result is not None))
            time.sleep(min(PERIOD, max(0, MAX_WAIT - (time.monotonic() - started))))
        save(STATE, dict(base, status='STOPPED_FINITE_SEVEN_DAY_WAIT_BOUND', decision_saved=result is not None))


if __name__ == '__main__':
    main()
