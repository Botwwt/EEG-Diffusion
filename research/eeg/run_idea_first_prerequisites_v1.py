"""Finite idea-first execution; no dependency on a second full image baseline.

Preserves the current nearly complete evaluator, then scores its saved output,
profiles actual interface training, trains aligned full/concept-OOF encoders,
exports all native conditions, and executes the frozen independent Dcheck.
Image adaptation requires the subsequent real-response/training budget review.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time

from run_native_oof_stage import stop_group, recover_gpu_receipt

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
PY = '/root/miniconda3/bin/python'
STATE = ROOT/'audit/idea_first_prerequisite_state_v1.json'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, value, new=False):
    path = Path(path)
    if new:
        assert not path.exists(), ('Preserve existing artifact', str(path))
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def remaining():
    ledger = ROOT/'audit/compute_ledger.jsonl'
    used = sum(float(load_line.get('charged_seconds', 0)) for load_line in map(json.loads, ledger.read_text().splitlines()))
    deadline = load(ROOT/'configs/compute_time_bounds.json')['conservative_occupied_time_deadline_epoch']
    return min(40*3600-used, deadline-time.time())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = load(plan_path)
    assert plan['version'] == 'idea_first_execution_v1' and plan['total_GPU_hour_cap'] == 40
    assert not STATE.exists(), 'Inspect saved outputs before separately named recovery'
    state = dict(status='STARTING', controller_pid=os.getpid(), plan_path=str(plan_path),
                 plan_sha256=sha(plan_path), completed_stages=[], final_test_unlocked=False,
                 mechanism_image_gain_established=False, formal_adapter_training_started=False)

    def update(**fields):
        state.update(fields, UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write(STATE, state)

    def check_sources():
        for name, digest in plan['source_sha256'].items():
            assert sha(ROOT/name) == digest, ('Frozen idea-first source changed', name)
        for name, digest in plan['official_source_sha256'].items():
            assert sha(Path('/root/atm_official_20261016')/name) == digest
        assert not (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists()
        assert not (ROOT/'STOP_IDEA_FIRST_V1').exists(), 'Explicit idea-first stop flag'

    process = None
    active_name = None
    active_gpu = False
    tick = None

    def run(stage, script, arguments, cap, gpu=False):
        nonlocal process, active_name, active_gpu, tick
        check_sources()
        assert remaining() > cap + 900, 'Global cap/margin prevents next phase'
        name = 'IDEA1_' + stage
        command = [PY, str(ROOT/script), *map(str, arguments)]
        if gpu:
            command = [PY, str(ROOT/'run_gpu_job.py'), '--name', name,
                       '--max-seconds', str(cap), '--', *command]
        log = ROOT/'logs'/f'{name}.log'
        assert not log.exists()
        env = os.environ.copy()
        if not gpu:
            env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        tick = time.monotonic()
        active_name, active_gpu = name, gpu
        with log.open('xb') as file:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=file,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            update(status='RUNNING', stage=stage, phase_supervisor_pid=process.pid,
                   command=command, log=str(log), GPU_used=gpu,
                   actual_remaining_seconds_at_start=remaining())
            while process.poll() is None:
                if (ROOT/'STOP_IDEA_FIRST_V1').exists() or time.monotonic()-tick > cap+45:
                    stop_group(process)
                    if gpu:
                        recover_gpu_receipt(name, process.returncode, time.monotonic()-tick)
                    raise RuntimeError('Owned phase stopped; preserve checkpoints and inspect')
                time.sleep(5)
            code = process.wait()
        event = dict(stage=stage, seconds=time.monotonic()-tick, exit_code=code, GPU_used=gpu, log=str(log))
        state.setdefault('phase_receipts', []).append(event)
        if not gpu:
            with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
                file.write(json.dumps(event)+'\n')
        assert code == 0, ('Phase failed; no automatic retry', event)
        state['completed_stages'].append(stage)
        update(status='PHASE_COMPLETED', stage=stage)

    try:
        check_sources()
        update(status='WAITING_ONLY_FOR_CURRENT_EVALUATOR_TO_FINISH', stage='preserve_current_evaluation')
        feature = DATA/'evaluation/V/REF_REG_correct_R4/features/completion.json'
        worker = Path('/proc')/str(plan['current_evaluator_worker'])/'cmdline'
        wrapper_receipt = ROOT/'audit/S1_V_evaluate_REF_REG_job.json'
        waiting_started = time.monotonic()
        while True:
            check_sources()
            receipt = load(wrapper_receipt)
            if receipt['status'] != 'RUNNING':
                assert receipt['status'] == 'COMPLETED' and feature.exists(), receipt
                assert load(feature)['queries'] == 16600
                break
            assert worker.exists() and b'extract_image_features.py' in worker.read_bytes()
            assert time.monotonic()-waiting_started < 1800
            time.sleep(10)
        reg = DATA/'images/S1_full_native_batchsafe_v2/V/REF_REG_correct_R4'
        evaluate = DATA/'evaluation/V/REF_REG_correct_R4'
        gallery = DATA/'evaluation/gallery/V'
        if not (evaluate/'metrics/summary.json').exists():
            run('preserved_REG_score', 'score_full_manifest.py',
                ['--manifest', reg/'manifest.jsonl', '--features', evaluate/'features',
                 '--gallery-manifest', gallery/'manifest.jsonl', '--gallery-features', gallery/'features',
                 '--output', evaluate/'metrics'], 600)
        run('projector_cost', 'profile_native_projector_cost.py', [], 240, True)
        needed = sum(plan['role_max_seconds'].values()) + 900 + 1800 + 900
        assert remaining() >= needed, ('Entire prerequisite stage no longer fits', remaining(), needed)
        gate = dict(status='IDEA_FIRST_PREREQUISITES_FUNDED_NOT_IMAGE_ARMS',
                    full_and_two_fold_training_and_condition_sampling_funded=True,
                    full_two_fold_training_and_condition_sampling_funded=True,
                    actual_tighter_remaining_seconds=remaining(), required_prerequisite_seconds=needed,
                    user_priority_override='Idea first; old complete-reference/control/final reservations deferred',
                    formal_image_arms_funded=False, total_GPU_hour_cap=40,
                    training_seed=1729, final_test_unlocked=False, plan_sha256=sha(plan_path))
        gate_path = ROOT/'audit/idea_first_prerequisite_budget_gate_v1.json'
        write(gate_path, gate, new=True)
        base = load(ROOT/'configs/full_reference_resolved.json')
        training = base | dict(study='idea_first_direct_ATMS_v1', condition_chain='REF_REG', prior_enabled=False,
                    output_root=plan['training_output_root'], encoder_patience=50,
                    role_max_wall_seconds=plan['role_max_seconds'],
                    idea_trainer_sha256=sha(ROOT/'train_idea_first_atms_v1.py'),
                    base_trainer_sha256=sha(ROOT/'train_full_reference.py'),
                    official_benchmark_sha256=plan['official_source_sha256']['Generation/benchmark.sh'],
                    budget_gate_sha256=sha(gate_path),
                    alignment_and_limits=plan['alignment_and_limits'])
        config_path = ROOT/'configs/idea_first_ATMS_training_v1.json'
        write(config_path, training, new=True)
        for role in ['full', 'oof0', 'oof1']:
            run(role, 'train_idea_first_atms_v1.py', ['--config', config_path, '--role', role],
                plan['role_max_seconds'][role]+30, True)
        checkpoints = {}
        for subject in training['subjects']:
            proof = load(Path(training['output_root'])/'full'/subject/'completion.json')
            assert proof['prior'] is None and sha(proof['encoder']['checkpoint']) == proof['encoder']['checkpoint_sha256']
            checkpoints[subject] = dict(encoder_checkpoint=proof['encoder']['checkpoint'],
                                       encoder_sha256=proof['encoder']['checkpoint_sha256'],
                                       prior_checkpoint=None, prior_sha256=None)
        chain_path = ROOT/'audit/idea_first_fixed_chain_v1.json'
        write(chain_path, dict(status='PROSPECTIVELY_FIXED_DIRECT_ENCODER_CHAIN_NOT_PERFORMANCE_SELECTION',
              selected_reference='REF_REG', frozen_reference_checkpoints=checkpoints,
              plan_sha256=sha(plan_path), final_test_unlocked=False,
              official_full_pipeline_reproduction=False, all10_subjects=True), new=True)
        cache = {k:training[k] for k in ['seed','condition_chain','output_root','split_sha256',
                 'visual_features_path','visual_features_sha256']}
        cache.update(cache_builder_sha256=sha(ROOT/'build_idea_first_oof_cache_v1.py'),
                     oof_training_config_path=str(config_path), oof_training_config_sha256=sha(config_path),
                     reference_selection_sha256=sha(chain_path), budget_gate_sha256=sha(gate_path),
                     inference_batch=4, coordinate_space='raw_pooled_CLIP_H_1024_before_native_IP_projection',
                     condition_cache_output_root=plan['cache_output_root'],
                     full_reference_root=str(Path(training['output_root'])/'full'),
                     condition_cache_max_seconds=900, minimum_free_bytes_before_cache=6*1024**3)
        cache_path = ROOT/'configs/idea_first_native_cache_v1.json'
        write(cache_path, cache, new=True)
        run('cache', 'build_idea_first_oof_cache_v1.py', ['--config', cache_path], 900)
        run('Dcheck', 'validate_idea_first_error_channel_v1.py',
            ['--plan', ROOT/'configs/idea_first_error_validation_plan_v1.json', '--cache-config', cache_path,
             '--output', plan['validation_output_root']], 1800)
        summary = load(Path(plan['validation_output_root'])/'summary.json')
        update(status='COMPLETED_REAL_ALL10_OOF_AND_INDEPENDENT_DCHECK_IMAGE_STAGE_REVIEW_REQUIRED',
               CPU_validation_summary=summary, full_UNet_response_validated=False,
               next_action='Use actual cost and Dcheck result to run the smallest informative matched image study; no vector-only success claim')
    except Exception as error:
        if process is not None and process.poll() is None:
            stop_group(process)
            if active_gpu:
                recover_gpu_receipt(active_name, process.returncode, time.monotonic()-tick)
        update(status='FAILED_OR_RESOURCE_STOPPED_PRESERVE_OUTPUTS', error=repr(error),
               hypothesis_rejected=False, automatic_retry=False)
        raise


if __name__ == '__main__':
    main()
