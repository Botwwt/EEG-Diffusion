"""Read-only 20-minute observer for the repaired Brain-HIVE main lifecycle.

No model imports, CUDA calls, image decoding, score reads, restarts or kills.
Only this monitor's small registration/latest/history/alert files are written.
The pinned original storage functions are extracted as AST function bodies to
avoid importing their model runtime merely to inspect a disk forecast.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time


ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
MAIN = WORK / 'audit/main_storage_continuation_20261021_v1.json'
DIAG_STATE = WORK / 'audit/native_function_T32_queue_v2.json'
DIAG_RECIPE = ROOT / 'configs/BrainHIVE_native_function_T32_v2.json'
RECIPE = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
GUARD = ROOT / 'brainhive_lifecycle_storage_v1.py'
CONTROLLER = ROOT / 'queue_BrainHIVE_main_storage_continuation_20261021_v1.py'
LATEST = WORK / 'audit/bounded_20min_monitor_latest_v2.json'
HISTORY = WORK / 'audit/bounded_20min_monitor_v2.jsonl'
ALERTS = WORK / 'audit/bounded_20min_monitor_alerts_v2.jsonl'
REGISTRATION = WORK / 'audit/bounded_20min_monitor_registration_v2.json'
LOCK = WORK / 'audit/bounded_20min_monitor_v2.lock'
EXPECTED_RECIPE = '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
EXPECTED_CONTROLLER = '47a9d04edd255f5d08367d93b0cec0aa81a926f87bc232742d71c677069ba806'
EXPECTED_GUARD = '88b5983ce40b2e996698382835aff22dd538c5ecfd84247b4c824bfd34fca083'
INTERVAL = 1200
MAXIMUM = 7 * 24 * 3600
TERMINAL = {
    'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',
    'FINITE_MAIN_STORAGE_CONTINUATION_PARTIAL_WITH_EXPLICIT_FAILURES',
    'STORAGE_FORECAST_STOPPED_PRESERVED_NO_CASCADING_JOBS',
    'FAILED_MAIN_STORAGE_CONTINUATION_PRESERVED_NEEDS_DIAGNOSIS',
}
DIAG_TERMINAL = {
    'FROZEN_FUNCTIONAL_DIAGNOSTIC_SAVED_NO_FORMAL_COMPONENT_TRAINING',
    'DIAGNOSTIC_FAILED_WITH_PRESERVED_LOG_NO_AUTOMATIC_RETRY',
    'STOPPED_WITH_EXPLICIT_DIAGNOSTIC_FAILURE',
}
CPU_PHASES = {
    'audit_BrainHIVE_image_arm_CPU_v1.py': 'CPU_IMAGE_IDENTITY_AUDIT',
    'score_full_manifest.py': 'CPU_PER_QUERY_EVALUATION',
    'score_low_level.py': 'CPU_LOW_LEVEL_AUDIT',
    'compare_BrainHIVE_bounded_V_CPU_v1.py': 'CPU_THREE_ARM_COMPARISON',
}
LOSS_FIELDS = ['last_loss', 'cumulative_mean_loss', 'epsilon_loss', 'token_loss',
               'risk_auxiliary_loss', 'gradient_norm', 'correction_RMS', 'mean_gate']
LINE_CACHE = {}
LOG_CACHE = {}
PROGRESS_CACHE = {}
OWNS_MONITOR = False


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write_latest(value):
    pending = LATEST.with_name(LATEST.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(LATEST)


def process(pid):
    if not isinstance(pid, int) or pid <= 0:
        return dict(pid=pid, live=False)
    try:
        folder = Path('/proc') / str(pid)
        raw = (folder / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        argv = (folder / 'cmdline').read_bytes().decode('utf-8', errors='replace').strip('\0').split('\0')
        return dict(pid=pid, live=fields[0] not in ['Z', 'X'], state=fields[0],
            CPU_ticks=int(fields[11]) + int(fields[12]), start_ticks=int(fields[19]),
            argv=[item[:240] for item in argv[:18]])
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, live=False)
    except PermissionError:
        return dict(pid=pid, live=False, observation_error='PermissionError')


def descendants(pid):
    result, pending, seen = [], [pid], {pid}
    while pending and len(result) < 24:
        parent = pending.pop(0)
        try:
            children = [int(v) for v in (Path('/proc') / str(parent) / 'task' / str(parent) / 'children').read_text().split()]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for child in children:
            if child in seen:
                continue
            seen.add(child); pending.append(child); result.append(process(child))
    return result


def has_entry(proc, name):
    return proc.get('live') and any(Path(arg).name == name for arg in proc.get('argv', []))


def kv(path):
    try:
        return {line.split()[0]: int(line.split()[1]) for line in Path(path).read_text().splitlines()}
    except FileNotFoundError:
        return {}


def memory():
    def integer(path):
        try:
            value = Path(path).read_text().strip()
            return None if value == 'max' else int(value)
        except FileNotFoundError:
            return None
    return dict(current=integer('/sys/fs/cgroup/memory.current'),
        maximum=integer('/sys/fs/cgroup/memory.max'),
        events=kv('/sys/fs/cgroup/memory.events'), shm_free=shutil.disk_usage('/dev/shm').free)


def safe_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: safe_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_numbers(item) for item in value]
    return value


def training(arm):
    folder = WORK / 'components/bounded_v1' / arm
    done = read(folder / 'completion.json')
    progress = read(folder / 'progress.json') or {}
    if done:
        return dict(status='TRAINING_COMPLETE_FIXED_FINAL', updates=done.get('updates'),
            expected_updates=24807, seen_queries=done.get('total_seen_queries'),
            checkpoint_sha256=done.get('checkpoint_sha256'), training_seconds=done.get('training_seconds'))
    path = folder / 'progress.json'
    status = progress.get('status', 'NOT_STARTED')
    if progress.get('updates', 0) >= 24807:
        status = 'OPTIMIZATION_FINISHED_AWAITING_CALIBRATION_AND_FINAL_RECEIPT'
    return dict(status=status, updates=progress.get('updates', 0),
        expected_updates=24807, seen_queries=progress.get('seen_queries', 0),
        progress_age_seconds=time.time() - path.stat().st_mtime if path.exists() else None,
        training_values={key: progress[key] for key in LOSS_FIELDS if key in progress})


def line_progress(path):
    if not path.exists():
        return dict(completed=0, manifest_bytes=0, missing_after_observed=str(path) in LINE_CACHE)
    info = path.stat(); key = str(path)
    old = LINE_CACHE.get(key)
    reset = old is not None and (old['identity'] != (info.st_dev, info.st_ino) or info.st_size < old['offset'])
    offset, count = (old['offset'], old['count']) if old and not reset else (0, 0)
    with path.open('rb') as stream:
        stream.seek(offset)
        while True:
            chunk = stream.read(min(1 << 20, info.st_size - stream.tell()))
            if not chunk:
                break
            count += chunk.count(b'\n')
        offset = stream.tell()
    LINE_CACHE[key] = dict(identity=(info.st_dev, info.st_ino), offset=offset, count=count)
    return dict(completed=count, manifest_bytes=info.st_size, identity_or_size_reset=reset)


def feature_progress(folder):
    done = read(folder / 'completion.json')
    progress = read(folder / 'progress.json') or {}
    return dict(status='COMPLETED' if done else progress.get('status', 'NOT_STARTED'),
        stage='COMPLETE' if done else progress.get('stage'),
        completed=done.get('queries', 0) if done else progress.get('completed', 0),
        expected=done.get('queries') if done else progress.get('expected'))


def log_scan(path):
    if not path.exists():
        return dict(available=False, error_markers=[])
    info = path.stat(); old = LOG_CACHE.get(str(path)); limit = 256 << 10
    reset = old and (old['identity'] != (info.st_dev, info.st_ino) or info.st_size < old['offset'])
    previous = old['offset'] if old and not reset else 0
    start = max(previous, info.st_size - limit)
    with path.open('rb') as stream:
        stream.seek(start); data = stream.read(info.st_size - start)
    LOG_CACHE[str(path)] = dict(identity=(info.st_dev, info.st_ino), offset=start + len(data))
    pattern = re.compile(r'\b(?:nan|inf|infinity|out of memory|outofmemoryerror|oom[_ ]kill(?:ed)?|cuda error)\b', re.I)
    snippets = [line[-500:] for line in data.decode('utf-8', errors='replace').splitlines() if pattern.search(line)]
    return dict(available=True, bytes_scanned=len(data), earlier_unscanned_bytes=max(0, start - previous),
                identity_or_size_reset=bool(reset), error_markers=snippets[:4])


def forecast_function():
    assert sha(GUARD) == EXPECTED_GUARD
    tree = ast.parse(GUARD.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ['bytes_under', 'verify_remaining']]
    assert [node.name for node in nodes] == ['bytes_under', 'verify_remaining']
    namespace = dict(json=json, Path=Path, shutil=shutil, WORK=WORK)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GUARD), 'exec'), namespace)
    return namespace['verify_remaining']


def forecast(function, recipe):
    try:
        return dict(ok=True, **function(recipe))
    except Exception as error:
        result = dict(ok=False, error_type=type(error).__name__, error=str(error)[:2400])
        match = re.search(r'free=(\[[^\]]*\]), required=(\[[^\]]*\])', str(error))
        if match:
            result.update(free=json.loads(match[1]), remaining_required=json.loads(match[2]))
        return result


def phase(state, job, children):
    if state.get('status') in TERMINAL:
        return 'MAIN_TERMINAL', None
    for entry, label in [('train_BrainHIVE_bounded_component_v1.py', 'TRAINING_REAL_OOF'),
                         ('generate_BrainHIVE_bounded_V_v1.py', 'GENERATING_ACTUAL_IMAGES'),
                         ('extract_image_features.py', 'EXTRACTING_IMAGE_FEATURES')]:
        actual = next((child for child in children if has_entry(child, entry)), None)
        if actual:
            if entry == 'extract_image_features.py' and '--real-gallery' in actual.get('argv', []):
                label = 'EXTRACTING_REAL_GALLERY_FEATURES'
            return label, actual
    cpu = state.get('cpu_entry')
    actual = next((child for child in children if cpu and has_entry(child, cpu)), None)
    if actual:
        return CPU_PHASES.get(cpu, 'CPU_OTHER'), actual
    if any(has_entry(child, 'run_gpu_job_v2.py') for child in children):
        return 'STARTING_GPU_WRAPPER_OR_WAITING_FOR_RECEIPT', None
    if cpu:
        return 'BETWEEN_STAGES_WITH_UNCONFIRMED_CPU_ENTRY', None
    return 'QUEUED_OR_BETWEEN_STAGES', None


def auxiliary():
    result = {}
    for name, path in [
        ('deferred_CPU_maintenance', WORK / 'audit/deferred_shared_conditions_maintenance_20261021_v1.json'),
        ('T32_diagnostic_queue', WORK / 'audit/native_function_T32_queue_v2.json'),
        ('shared_cache_relocation', ROOT / 'audit/BrainHIVE_retired_shared_RAM_relocation_20261021_v1.json')]:
        value = read(path)
        if value:
            result[name] = {key: value[key] for key in ['status', 'pid', 'UTC', 'utc', 'helper_attempts', 'generated_images'] if key in value}
            if value.get('pid'):
                proc = process(value['pid'])
                result[name]['process'] = {key: proc[key] for key in ['pid', 'live', 'state'] if key in proc}
    return result


def main():
    global OWNS_MONITOR
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller-pid', type=int, default=205230)
    parser.add_argument('--diagnostic-pid', type=int, default=205294)
    args = parser.parse_args()
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not any(path.exists() for path in [REGISTRATION, LATEST, HISTORY, ALERTS]), 'Preserve all earlier monitor records'
        initial = read(MAIN)
        assert initial and initial['pid'] == args.controller_pid and initial['recipe_sha256'] == EXPECTED_RECIPE
        pins = {str(Path(__file__).resolve()): sha(__file__), str(RECIPE): EXPECTED_RECIPE,
                str(GUARD): EXPECTED_GUARD, str(CONTROLLER): EXPECTED_CONTROLLER}
        for group in ['original_sources_sha256', 'additional_sources_sha256']:
            pins.update({str(ROOT / name): value for name, value in initial[group].items()})
        diagnostic_recipe = read(DIAG_RECIPE)
        assert diagnostic_recipe and diagnostic_recipe['diagnostic'] == 'native_function_T32_v2'
        pins[str(DIAG_RECIPE)] = sha(DIAG_RECIPE)
        for group in ['source_sha256', 'queue_source_sha256']:
            pins.update({str(ROOT / name): value for name, value in diagnostic_recipe[group].items()})
        registration = dict(UTC=utc(), pid=os.getpid(), controller_pid=args.controller_pid,
            diagnostic_pid=args.diagnostic_pid,
            interval_seconds=INTERVAL, maximum_seconds=MAXIMUM, source_pins=pins,
            model_imports=False, CUDA_calls=False, V_or_test_scores_read=False,
            restart_or_kill_actions=False, observed_main_state=str(MAIN))
        with REGISTRATION.open('x') as stream:
            stream.write(json.dumps(registration, indent=2) + '\n')
        OWNS_MONITOR = True
        started = time.monotonic(); previous_events = None; sampled = 0; main_start = None
        function = None
        with HISTORY.open('x', buffering=1) as history, ALERTS.open('x', buffering=1) as alert_file:
            while time.monotonic() - started < MAXIMUM:
                sampled += 1; alerts = []; state = read(MAIN) or {}; now = time.monotonic()
                changed = []
                for path, expected in pins.items():
                    try:
                        actual = sha(path)
                    except OSError as error:
                        actual = type(error).__name__
                    if actual != expected:
                        changed.append(dict(path=path, expected=expected, actual=actual))
                if changed:
                    record = dict(UTC=utc(), monitor_status='STOPPED_PINNED_SOURCE_CHANGED', source_mismatches=changed,
                        pid=os.getpid(), samples=sampled, alerts=['PINNED_SOURCE_OR_RECIPE_CHANGED'], model_or_hyperparameter_mutations=False)
                    history.write(json.dumps(record) + '\n'); alert_file.write(json.dumps(record) + '\n'); write_latest(record)
                    return
                if function is None:
                    function = forecast_function()
                controller = process(args.controller_pid)
                if not has_entry(controller, CONTROLLER.name):
                    if state.get('status') not in TERMINAL:
                        alerts.append('MAIN_CONTROLLER_NOT_LIVE_OR_IDENTITY_MISMATCH')
                    children = []
                else:
                    if main_start is None:
                        main_start = controller['start_ticks']
                    if controller['start_ticks'] != main_start:
                        alerts.append('MAIN_CONTROLLER_PID_REUSED')
                        children = []
                    else:
                        children = descendants(args.controller_pid)
                if state.get('pid') != args.controller_pid:
                    alerts.append('MAIN_STATE_PID_MISMATCH')
                name = state.get('job')
                job = None
                if name:
                    assert Path(name).name == name and name not in ['.', '..']
                    job = read(ROOT / 'audit' / (name + '_job.json'))
                diag = read(DIAG_STATE) or {}
                diag_proc = process(args.diagnostic_pid)
                diagnostic_identity_ok = has_entry(diag_proc, 'queue_BrainHIVE_native_function_T32_v2.py')
                diag_children = descendants(args.diagnostic_pid) if diagnostic_identity_ok else []
                diag_job = read(ROOT / 'audit/BrainHIVE_native_function_T32_v2_job.json')
                if diag.get('pid') != args.diagnostic_pid:
                    alerts.append('T32_QUEUE_STATE_PID_MISMATCH')
                if not diagnostic_identity_ok and diag.get('status') not in DIAG_TERMINAL:
                    alerts.append('T32_QUEUE_NOT_LIVE_OR_IDENTITY_MISMATCH')
                current_phase, active = phase(state, job, children)
                train = {arm: training(arm) for arm in ['risk', 'ordinary']}
                if current_phase == 'TRAINING_REAL_OOF' and train.get(state.get('current_arm'), {}).get('updates', 0) >= 24807:
                    current_phase = 'POST_TRAINING_CALIBRATION_OR_CHECKPOINT_SAVE'
                if state.get('status') in TERMINAL:
                    diag_active = next((child for child in diag_children if has_entry(child, 'diagnose_BrainHIVE_native_function_T32_v2.py')), None)
                    if diag_active:
                        current_phase, active = 'FROZEN_T_ONLY_DIAGNOSTIC', diag_active
                    elif diagnostic_identity_ok and diag.get('status') not in DIAG_TERMINAL:
                        current_phase, active = 'T32_DIAGNOSTIC_QUEUED_OR_BETWEEN_STAGES', None
                diag_output = Path(diagnostic_recipe['output_directory'])
                diag_progress = line_progress(diag_output / 'per_query_timestep.jsonl')
                diag_progress['expected'] = 128
                # Completion is receipt metadata, not diagnostic summary values.
                diag_completion = read(diag_output / 'completion.json')
                if diag_completion:
                    diag_progress.update(completion_status=diag_completion.get('status'), rows=diag_completion.get('rows'))
                images, features = {}, {}
                recipe = read(RECIPE)
                for arm, run in recipe['run_names'].items():
                    images[arm] = line_progress(WORK / 'images' / run / 'inference_manifest.jsonl')
                    images[arm]['expected'] = 16600
                    images[arm]['complete_receipt'] = bool(read(WORK / 'images' / run / 'completion.json'))
                    features[arm] = feature_progress(WORK / 'evaluation' / run / 'features')
                    if images[arm].get('identity_or_size_reset'):
                        alerts.append('IMAGE_MANIFEST_REPLACED_OR_TRUNCATED_' + arm)
                    if images[arm].get('missing_after_observed'):
                        alerts.append('PREVIOUSLY_OBSERVED_IMAGE_MANIFEST_MISSING_' + arm)
                gallery_features = feature_progress(WORK / 'evaluation/bounded_V_gallery_features_v1')
                mem = memory()
                if previous_events is not None:
                    for key in ['oom', 'oom_kill', 'oom_group_kill']:
                        if mem['events'].get(key, 0) > previous_events.get(key, 0):
                            alerts.append('NEW_CGROUP_' + key.upper())
                previous_events = mem['events']
                if mem['maximum'] is not None and mem['current'] is not None and mem['maximum'] - mem['current'] < (2 << 30):
                    alerts.append('CGROUP_HEADROOM_BELOW_2GiB')
                remaining = forecast(function, recipe)
                if not remaining['ok']:
                    alerts.append('REMAINING_MAIN_LIFECYCLE_STORAGE_FORECAST_FAILED')
                log = log_scan(WORK / 'logs' / (name + '.log')) if name else None
                if log and log['error_markers']:
                    alerts.append('NEW_JOB_LOG_NONFINITE_OOM_OR_CUDA_ERROR_MARKER')
                diag_log = log_scan(WORK / 'logs/native_function_T32_v2.log')
                if diag_log['error_markers']:
                    alerts.append('NEW_T32_LOG_NONFINITE_OOM_OR_CUDA_ERROR_MARKER')
                for arm, value in train.items():
                    for key, number in value.get('training_values', {}).items():
                        if isinstance(number, (int, float)) and not math.isfinite(number):
                            alerts.append('NONFINITE_TRAINING_' + arm + '_' + key)
                counter, progress_key = None, None
                arm = state.get('current_arm')
                if current_phase == 'TRAINING_REAL_OOF' and arm in train:
                    counter, progress_key = train[arm]['updates'], 'training:' + arm
                elif current_phase == 'GENERATING_ACTUAL_IMAGES' and arm in images:
                    counter, progress_key = images[arm]['completed'], 'images:' + arm
                elif current_phase.startswith('EXTRACTING_'):
                    value = gallery_features if current_phase == 'EXTRACTING_REAL_GALLERY_FEATURES' else features.get(arm, {})
                    counter = value.get('completed', 0)
                    progress_key = current_phase + ':' + str(arm) + ':' + str(value.get('stage'))
                elif current_phase == 'FROZEN_T_ONLY_DIAGNOSTIC':
                    counter, progress_key = diag_progress['completed'], 'T32_completed_rows'
                elif active:
                    counter = active.get('CPU_ticks', 0)
                    progress_key = current_phase + ':' + str(active['pid']) + ':' + str(active.get('start_ticks'))
                delta = None
                if progress_key is not None:
                    previous = PROGRESS_CACHE.get(progress_key)
                    if previous:
                        delta = counter - previous['value']
                        last_change = now if delta != 0 else previous['last_change']
                        if delta < 0:
                            alerts.append('ACTIVE_PROGRESS_COUNTER_DECREASED')
                        if now - last_change >= 2 * INTERVAL:
                            alerts.append('NO_PROGRESS_FOR_40MIN_WITH_ACTIVE_PROCESS')
                    else:
                        last_change = now
                    PROGRESS_CACHE[progress_key] = dict(value=counter, last_change=last_change)
                if state.get('cpu_entry') and current_phase == 'BETWEEN_STAGES_WITH_UNCONFIRMED_CPU_ENTRY':
                    alerts.append('CPU_ENTRY_METADATA_NOT_CONFIRMED_BY_LIVE_CHILD')
                if job and job.get('status') == 'RUNNING' and not process(job.get('pid')).get('live'):
                    alerts.append('RUNNING_JOB_RECEIPT_HAS_NO_LIVE_WORKER')
                main_worker = process(job.get('pid')) if job and job.get('status') == 'RUNNING' else dict(live=False)
                diag_worker = process(diag_job.get('pid')) if diag_job and diag_job.get('status') == 'RUNNING' else dict(live=False)
                main_finished = state.get('status') in TERMINAL or (not has_entry(controller, CONTROLLER.name) and not main_worker['live'])
                diag_finished = diag.get('status') in DIAG_TERMINAL or (not diagnostic_identity_ok and not diag_worker['live'])
                record = safe_numbers(dict(UTC=utc(), monitor_status='OBSERVING', pid=os.getpid(), samples=sampled,
                    interval_seconds=INTERVAL,
                    main_status=state.get('status'), current_arm=arm, phase=current_phase, controller=controller,
                    child_processes=children, job={key: job[key] for key in ['name', 'status', 'pid', 'wrapper_pid', 'max_seconds'] if key in job} if job else None,
                    T32_diagnostic=dict(queue_status=diag.get('status'), controller=diag_proc, children=diag_children,
                        progress=diag_progress, job={key: diag_job[key] for key in ['name', 'status', 'pid', 'max_seconds'] if key in diag_job} if diag_job else None,
                        scanned_log=diag_log),
                    training=train, images=images, image_features=features, gallery_features=gallery_features,
                    progress=dict(key=progress_key, counter=counter, delta_since_previous_sample=delta),
                    disk_free={str(path): shutil.disk_usage(path).free for path in [WORK, DATA_PATH]},
                    cgroup=mem, remaining_lifecycle=remaining, scanned_job_log=log, auxiliary=auxiliary(),
                    alerts=alerts, model_or_hyperparameter_mutations=False, V_or_test_scores_read=False,
                    GPU_forward_calls=0, restart_or_kill_actions=False))
                if main_finished and diag_finished:
                    record['monitor_status'] = 'STOPPED_MAIN_AND_T32_TERMINAL_OR_NOT_LIVE'
                history.write(json.dumps(record, allow_nan=False) + '\n'); write_latest(record)
                if alerts:
                    alert_file.write(json.dumps(dict(UTC=record['UTC'], phase=current_phase, alerts=alerts,
                        main_status=record['main_status'], job=name), allow_nan=False) + '\n')
                if record['monitor_status'] == 'STOPPED_MAIN_AND_T32_TERMINAL_OR_NOT_LIVE':
                    return
                time.sleep(min(INTERVAL, max(0, MAXIMUM - (time.monotonic() - started))))
            final = dict(UTC=utc(), monitor_status='STOPPED_FINITE_SEVEN_DAY_BOUND', pid=os.getpid(), samples=sampled,
                         model_or_hyperparameter_mutations=False, V_or_test_scores_read=False, restart_or_kill_actions=False)
            history.write(json.dumps(final) + '\n'); write_latest(final)


DATA_PATH = Path('/root/autodl-tmp')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OWNS_MONITOR:
            failure = dict(UTC=utc(), monitor_status='STOPPED_OBSERVER_ERROR_NO_JOB_MUTATION', pid=os.getpid(),
                error_type=type(error).__name__, error=str(error)[:2400], restart_or_kill_actions=False)
            for destination in [HISTORY, ALERTS]:
                with destination.open('a') as stream:
                    stream.write(json.dumps(failure) + '\n')
            write_latest(failure)
        raise
