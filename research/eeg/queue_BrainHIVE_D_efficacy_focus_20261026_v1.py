"""User-authorized scope reduction: completed O, one D fit, full V efficacy.

The original four-arm registration is retained for unchanged GPU job contracts.
This queue never calls its four-arm comparison or functional diagnostics.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import brainhive_paired_study_contract_20261025_v1 as C
import queue_BrainHIVE_decisive_T_analysis_20261025_v2 as Q
from brainhive_paired_supervision_20261026_v3 import AdoptedProcess, safe_bytes_used
from queue_BrainHIVE_paired_effect_20261026_v3 import command_spec, close_GPU_receipt


BASE = C.ROOT / 'audit/efficacy_focus_20261026_v1'
SCHEMA = 'BrainHIVE_user_requested_D_only_efficacy_focus_v1'
CORE = '8ccb781afc2a4afe23d951bf0057387246ebd42917fcc2912b097634357c7f18'
STAGES = ('train', 'postfit', 'generate', 'audit', 'features', 'score')
C.bytes_used = safe_bytes_used


def scope(document):
    assert document['schema'] == SCHEMA
    assert document['retained_candidate'] == 'D' and document['completed_baseline'] == 'O'
    assert document['withdrawn_arms'] == ['A', 'U']
    assert document['withdrawn_diagnostics'] == ['original_T80', 'direction_gain_addon']
    assert document['candidate_ordinal'] == document['candidate_limit'] == 13
    assert document['original_scientific_core'] == CORE
    assert document['unchanged_D_training_and_inference'] is True
    assert document['original_four_arm_protocol_completed'] is False
    assert document['automatic_method_search'] is False
    assert document['maximum_metadata_bytes'] == 64 << 20
    assert document['comparison_maximum_seconds'] == 1800
    assert document['comparison_maximum_bytes'] == 64 << 20
    assert document['efficacy_gate'] == dict(DINO_minimum=.01, DINO_CI_lower_positive=True,
        SEED_CI_lower_positive=True, ObjectF1_mean_nonnegative=True,
        SSIM_Y_mean_nonnegative=True, PixCorr_RGB_mean_nonnegative=True)
    assert document['mechanism_claim'] == 'WITHDRAWN_NOT_TESTED'
    assert document['SOTA_established'] is False


def validate(path, *, all_inputs=False, expected_sha256=None):
    if expected_sha256 is not None:
        assert C.sha(path) == expected_sha256, 'The running scope plan changed'
    document = C.read(path); scope(document)
    for name, digest in document['source_sha256'].items():
        assert Path(name).name == name and C.sha(C.ROOT / name) == digest, name
    config = C.bound(document['original_registration'])
    assert C.validate(config, all_inputs=all_inputs) == CORE
    C.require_admission(config)
    assert C.ref(C.DATA / 'admission.json') == document['original_admission']
    withdrawn = C.bound(document['withdrawal_closure'])
    assert withdrawn['status'] == 'USER_WITHDRAWN_A_U_AND_MECHANISM_PROBES_QUIESCENT_CHECKPOINT_PRESERVED'
    assert withdrawn['A_checkpoint_CPU_decoded'] and withdrawn['A_coverage_optimizer_rng_and_log_prefix_verified']
    assert withdrawn['D_U_not_started'] and all(withdrawn['all_registered_processes_dead'].values())
    for pid in withdrawn['all_registered_processes_dead']:
        assert not Q.live(int(pid)), 'Withdrawn process unexpectedly live'
    assert C.ref(C.DATA / 'arms/A/train/resume.pt') == withdrawn['A_checkpoint']
    assert not (C.DATA / 'arms/A/train/completion.json').exists()
    assert not (C.DATA / 'arms/U').exists(), 'Withdrawn U must never start'
    assert set(document['baseline_O_completions']) == set(STAGES)
    for stage, item in document['baseline_O_completions'].items():
        spec = command_spec(config, 'O', stage, int(stage == 'train'))
        assert item['path'] == spec['completion']
        value = C.bound(item)
        assert value['status'] == spec['expected_status']
        if stage == 'train':
            assert value['updates'] == 24807 and value['primary_exposures'] == value['paired_exposures'] == 396900
            assert C.sha(value['checkpoint']['path']) == value['checkpoint']['sha256']
        else:
            assert value['queries'] == (32 if stage == 'postfit' else 16600)
    assert safe_bytes_used(BASE) <= document['maximum_metadata_bytes']
    assert safe_bytes_used(C.DATA / 'efficacy_DO_20261026_v1') <= document['comparison_maximum_bytes']
    return document, config


def specs(config):
    result = [command_spec(config, 'D', stage) | dict(stage=stage, arm='D') for stage in STAGES]
    result.append(dict(name='BrainHIVE_paired_effect_v1_DO_efficacy_compare_20261026',
        maximum_seconds=1800, GPU=False, stage='compare', arm=None,
        command=[sys.executable, '-B', '-u', str(C.ROOT / 'compare_BrainHIVE_DO_efficacy_CPU_20261026_v1.py'),
                 '--registration', config['registration_path']],
        completion=str(C.DATA / 'efficacy_DO_20261026_v1/summary.json'),
        expected_status='FULL16600_DO_EFFICACY_COMPARISON_COMPLETE'))
    assert len(result) == 7 and len({row['name'] for row in result}) == 7
    return result


def owned_after_registration(process, spec, job):
    deadline = time.monotonic() + 3
    while True:
        try:
            record = C.read(job)
        except (FileNotFoundError, json.JSONDecodeError):
            record = None  # The original wrapper writes its receipt non-atomically.
        if record is not None:
            assert record['wrapper_pid'] == process.pid and record['command'] == spec['command']
            assert record['name'] == spec['name']
            if record['status'] == 'RUNNING' and process.poll() is None and Q.live(record['pid']):
                try:
                    return AdoptedProcess(job, spec['name'], spec['command'], process.pid, record['pid'])
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                except AssertionError:
                    # A receipt may close while its running identity is being bound.
                    # Preserve all live identity/command failures as hard errors.
                    try: current = C.read(job)
                    except (FileNotFoundError, json.JSONDecodeError): current = None
                    if current is not None and current['status'] == 'RUNNING' and process.poll() is None:
                        raise
            elif record['status'] != 'RUNNING' and process.poll() is not None:
                return None
        if time.monotonic() >= deadline:
            return None
        time.sleep(.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    document, config = validate(args.plan, all_inputs=True)
    stages = specs(config)
    assert not (C.DATA / 'arms/D').exists(), 'New D queue starts only the unstarted registered candidate'
    assert not (C.DATA / 'efficacy_DO_20261026_v1').exists()
    if args.preflight_only:
        print(__import__('json').dumps(dict(status='D_ONLY_SEVEN_STAGE_EFFICACY_CPU_PREFLIGHT_PASSED',
            original_scientific_core=CORE, stages=[s['stage'] for s in stages], new_O_fits=0,
            starts_GPU=False, reads_V_scores=False)))
        return
    assert C.ROOT == C.REMOTE
    import fcntl
    state_path = BASE / 'queue_state.json'
    assert not state_path.exists(), 'Never overwrite or silently replay this queue'
    state = dict(status='D_ONLY_FULL_T_FULL_V_EFFICACY_RUNNING', pid=os.getpid(), plan=C.ref(args.plan),
        candidate_ordinal=13, original_scientific_core=CORE, completed_baseline='O',
        original_four_arm_protocol_completed=False, withdrawn_arms=['A', 'U'],
        current_stage=None, child_pid=None, completed=[], attempts=[], automatic_new_candidate=False)
    def save():
        state['UTC'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        C.atomic(state_path, state)
    def stop(signum, frame):
        raise InterruptedError('Controller signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    with (C.ROOT / 'audit/functional_full_lifecycle_20261022_v1/lifecycle_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate(args.plan, expected_sha256=state['plan']['sha256']); Q.gpu_free(); save()
        def execute(spec):
            validate(args.plan, expected_sha256=state['plan']['sha256'])
            if spec['GPU']: Q.gpu_free()
            log = BASE / (spec['name'] + '.log')
            job = C.ROOT / 'audit' / (spec['name'] + '_job.json')
            assert not log.exists() and not Path(spec['completion']).exists()
            if spec['GPU']: assert not job.exists()
            argv = ([sys.executable, '-B', str(C.ROOT / 'run_gpu_job_v2.py'), '--name', spec['name'],
                '--max-seconds', str(spec['maximum_seconds']), '--', *spec['command']]
                if spec['GPU'] else spec['command'])
            env = os.environ.copy()
            for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): env[key] = '2'
            if not spec['GPU']: env['CUDA_VISIBLE_DEVICES'] = ''
            entry = spec | dict(log=str(log), wrapper_command=argv)
            state['attempts'].append(entry); state['current_stage'] = spec['name']; save()
            process = owned = None; start = time.monotonic()
            try:
                with log.open('xb') as stream:
                    process = subprocess.Popen(argv, cwd=C.ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                state['child_pid'] = entry['wrapper_pid'] = process.pid; save()
                if spec['GPU']:
                    owned = owned_after_registration(process, spec, job)
                    if process.poll() is None:
                        assert owned is not None, 'Running GPU child lacks stable ownership proof'
                while process.poll() is None:
                    assert time.monotonic() - start < spec['maximum_seconds'] + 120
                    assert log.stat().st_size <= 32 << 20
                    validate(args.plan, expected_sha256=state['plan']['sha256']); time.sleep(10)
                entry.update(exit_code=process.returncode, seconds=time.monotonic() - start, log_sha256=C.sha(log))
                if spec['GPU']:
                    entry.update(close_GPU_receipt(C.read(job), spec, process.pid, process.returncode))
                    entry['GPU_job'] = C.ref(job)
                state['child_pid'] = None; save()
                return process.returncode
            except BaseException:
                if process is not None:
                    try:
                        if spec['GPU']:
                            if owned is None: owned = owned_after_registration(process, spec, job)
                            if owned is not None: owned.validate_owned_identity_for_cleanup()
                            else:
                                assert process.poll() is not None and job.exists() and not Q.live(C.read(job)['pid']), 'GPU ownership unresolved'
                    except BaseException as ownership_error:
                        entry['cleanup'] = dict(signals_sent=False, identity_error=repr(ownership_error),
                            wrapper_live=Q.live(process.pid), reason='Unverified ownership; preserve unresolved')
                        state['child_pid'] = process.pid
                        state['unresolved_live_child'] = True
                        save(); raise
                    entry['cleanup'] = {}
                    terminal = Q.stop_owned(owned or process, spec['GPU'], job, spec['command'], spec['name'], entry['cleanup'])
                    state['child_pid'] = None if terminal else process.pid
                    state['unresolved_live_child'] = not terminal
                save(); raise
        try:
            for spec in stages:
                code = execute(spec)
                if code and spec['stage'] in ('train', 'generate'):
                    root = C.DATA / 'arms/D' / ('train' if spec['stage'] == 'train' else 'images')
                    progress = C.read(root / 'progress.json')
                    allowed = ('PAIRED_TRAIN_FINITE_STOP_EXACT_RESUME' if spec['stage'] == 'train'
                               else 'PAIRED_GENERATION_FINITE_STOP_EXACT_PREFIX')
                    assert progress['status'] == allowed, 'No retry of engineering or scientific failure'
                    if spec['stage'] == 'train': assert (root / 'resume.pt').exists()
                    spec = command_spec(config, 'D', spec['stage'], 1) | dict(stage=spec['stage'], arm='D')
                    code = execute(spec)
                assert code == 0, 'Failure retained; no adaptive replay'
                done = C.read(spec['completion']); assert done['status'] == spec['expected_status']
                if spec['stage'] == 'train':
                    assert done['updates'] == 24807 and done['primary_exposures'] == done['paired_exposures'] == 396900
                elif spec['stage'] != 'compare':
                    assert done['queries'] == (32 if spec['stage'] == 'postfit' else 16600)
                state['completed'].append(dict(stage=spec['stage'], completion=C.ref(spec['completion']))); save()
            state.update(status='D_ONLY_FULL_T_FULL_V_AND_DO_EFFICACY_COMPARISON_COMPLETE', current_stage=None)
            save()
        except BaseException as error:
            state.update(status='D_EFFICACY_QUEUE_FAILED_PRESERVED_NO_AUTOMATIC_RETRY', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
