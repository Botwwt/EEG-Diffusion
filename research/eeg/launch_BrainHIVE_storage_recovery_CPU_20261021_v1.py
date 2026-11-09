"""Preserve failed attempts and resume the exact main recipe plus import repair."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
PYTHON = '/root/miniconda3/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def launch(command, log_path, state_path):
    assert not log_path.exists() and not state_path.exists()
    with log_path.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        assert process.poll() is None, log_path.read_text()[-3000:]
        if state_path.exists():
            state = read(state_path)
            assert state['pid'] == process.pid
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Controller failed to produce live state: ' + str(log_path))
    os.kill(process.pid, 0)
    return dict(pid=process.pid, command=command, state_path=str(state_path), initial_state=state)


def main():
    receipt = ROOT / 'audit/BrainHIVE_storage_recovery_verified_launch_20261021_v1.json'
    assert not receipt.exists()
    recipe = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
    assert sha(recipe) == '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
    assert sha(ROOT / 'queue_BrainHIVE_main_storage_continuation_20261021_v1.py') == '47a9d04edd255f5d08367d93b0cec0aa81a926f87bc232742d71c677069ba806'
    assert sha(ROOT / 'diagnose_BrainHIVE_native_function_T32_v2.py') == 'a1abab29858c660b425f313b504e0fbaadcd077803c87c9a01faa576afaed7b3'
    assert sha(ROOT / 'diagnose_BrainHIVE_native_function_T32_v1.py') == '13c7002716a6ba4f446f6e303806ae1722f18ad5858f155eff36dd21db681960'
    relocation = ROOT / 'audit/BrainHIVE_retired_ATM_RAM_relocation_20261021_v1.json'
    assert read(relocation)['status'] == 'ALL24_BYTE_IDENTICAL_LINKS_AND_EXISTING_MAIN_LIFECYCLE_FUNDED'
    assert len(read(relocation)['moved']) == 24
    from brainhive_lifecycle_storage_v1 import verify_remaining
    storage = verify_remaining(read(recipe))
    with (ROOT / 'audit/GPU_JOB.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = WORK / 'components/bounded_v1/risk/component_final.pt'
    assert sha(checkpoint) == '94dbaf576f7e0a4d095c16e1118a14c12b0990629d578ac15f92b9d907dc544e'
    manifest = WORK / 'images/protected_R4_risk_bounded_v1/inference_manifest.jsonl'
    prefix = manifest.read_bytes()
    rows = [json.loads(x) for x in prefix.splitlines()]
    assert len(rows) == 11000 and [x['global_query_index'] for x in rows] == list(range(11000))
    old_main = WORK / 'audit/bounded_full_lifecycle_queue_v1.json'
    old_diagnostic = WORK / 'audit/native_function_T32_queue_v1.json'
    old_hashes = {str(p): sha(p) for p in [old_main, old_diagnostic]}
    assert read(old_main)['status'] == 'FINITE_MAIN_COMPARISON_PARTIAL_WITH_EXPLICIT_FAILURES'
    assert read(old_diagnostic)['status'] == 'DIAGNOSTIC_FAILED_WITH_PRESERVED_LOG_NO_AUTOMATIC_RETRY'
    old = read(ROOT / 'configs/BrainHIVE_native_function_T32_v1.json')
    new = dict(old)
    new.update(diagnostic='native_function_T32_v2', output_directory=str(WORK / 'audit/native_function_T32_v2'),
        registration_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        repair='Initialize existing native runtime before importing diffusers; scientific settings unchanged.',
        noise_namespace_preserved='native_function_T32_v1', old_failed_registration_sha256=sha(ROOT / 'configs/BrainHIVE_native_function_T32_v1.json'),
        new_component_candidate_configurations=0)
    new['source_sha256'] = dict(old['source_sha256'])
    del new['source_sha256']['diagnose_BrainHIVE_native_function_T32_v1.py']
    new['source_sha256']['diagnose_BrainHIVE_native_function_T32_v2.py'] = sha(ROOT / 'diagnose_BrainHIVE_native_function_T32_v2.py')
    new['queue_source_sha256'] = {name: sha(ROOT / name) for name in
        ['queue_BrainHIVE_native_function_T32_v2.py', 'run_gpu_job_v2.py']}
    from diagnose_BrainHIVE_native_function_T32_v2 import check_registered_input_contract
    contract = check_registered_input_contract(new)
    runtime = subprocess.run([PYTHON, str(ROOT / 'diagnose_BrainHIVE_native_function_T32_v2.py'), '--check-runtime-imports'],
        cwd=ROOT, text=True, capture_output=True, check=True, timeout=45)
    registration = ROOT / 'configs/BrainHIVE_native_function_T32_v2.json'
    assert not registration.exists()
    registration.write_text(json.dumps(new, indent=2) + '\n')
    main_launch = launch([PYTHON, '-u', str(ROOT / 'queue_BrainHIVE_main_storage_continuation_20261021_v1.py'),
        '--recipe', str(recipe)], WORK / 'logs/main_storage_continuation_20261021_v1.log',
        WORK / 'audit/main_storage_continuation_20261021_v1.json')
    diagnostic_launch = launch([PYTHON, '-u', str(ROOT / 'queue_BrainHIVE_native_function_T32_v2.py'),
        '--registration', str(registration)], WORK / 'logs/native_function_T32_queue_v2.log',
        WORK / 'audit/native_function_T32_queue_v2.json')
    assert all(sha(path) == expected for path, expected in old_hashes.items())
    result = dict(status='ORIGINAL_MAIN_CONTINUATION_AND_FIXED_DIAGNOSTIC_QUEUES_VERIFIED_LIVE',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        main=main_launch, diagnostic=diagnostic_launch, relocation_receipt=str(relocation), relocation_sha256=sha(relocation),
        remaining_storage=storage, risk_training_repeated=False, risk_checkpoint_sha256=sha(checkpoint),
        existing_image_manifest_prefix_bytes=len(prefix), existing_image_manifest_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        existing_images_preserved=11000, old_failed_state_sha256=old_hashes,
        diagnostic_input_contract=contract, diagnostic_runtime_check=json.loads(runtime.stdout),
        original_model_recipe_sampling_and_metrics_changed=False, new_candidate_configurations=0,
        new_component_image_gain_established=False)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], main_pid=main_launch['pid'],
        diagnostic_pid=diagnostic_launch['pid'], risk_training_repeated=False, existing_images_preserved=11000)), flush=True)


if __name__ == '__main__':
    main()
