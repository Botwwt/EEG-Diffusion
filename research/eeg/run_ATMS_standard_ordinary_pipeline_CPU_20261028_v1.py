"""Finite ATMS/standard-SDXL ordinary pipeline, with immutable per-stage receipts.

Wait for the already running public-asset restoration, then engineering checks,
complete genuine OOF training and Dfit-only calibration. No image-effect claim.
This controller never purchases resources, restarts downloads or resumes old D.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PY = '/root/miniconda3/bin/python'
TRAINER = 'train_ATMS_standard_SDXL_ordinary_20261028_v1.py'
IMAGE_PROFILE = 'profile_ATMS_standard_SDXL_ordinary_images_GPU_20261028_v1.py'
LATENT_BUILDER = 'build_ATMS_standard_T_latents_GPU_20261028_v1.py'
IMAGE_GENERATOR = 'generate_ATMS_standard_SDXL_ordinary_20261028_v1.py'
IMAGE_SCORER = 'score_ATMS_standard_SDXL_ordinary_20261028_v1.py'
AUDIT = ROOT / 'audit/cross_framework_ordinary_20261028_v1'
OUTPUT = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1/candidates/X01_ATMS_standard_ordinary')
ACTIVE = None
ACTIVE_JOB = None


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_bytes())


def check(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes']
    return path


def atomic(path, value, overwrite=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    assert not temporary.exists() and (overwrite or not path.exists()), str(path)
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def progress(stage, **values):
    value = dict(status=stage, UTC=utc(), pid=os.getpid(), **values)
    atomic(AUDIT / 'pipeline_progress.json', value, overwrite=True)
    print(json.dumps(value), flush=True)


def check_launch(launch):
    assert launch['status'] == 'ATMS_STANDARD_ORDINARY_FINITE_PIPELINE_REVIEWED'
    for item in launch['source_pins']:
        check(item)
    pins = {item['path']: item['sha256'] for item in launch['source_pins']}
    assert pins[str(Path(__file__).resolve())] == sha(__file__)
    assert pins[str(ROOT / TRAINER)] == launch['trainer_sha256']
    assert launch['trainer_sha256'] == '01df320704f0a4d2f3f8ac56f58868717639ec11abaee582c331c2b190cc6f00'
    assert launch['asset_wait_max_seconds'] == 7500 and launch['maximum_training_segments'] == 128
    auth = read(check(launch['authorization']))
    assert auth['formal_old_D_training_resumed'] is False


def wait_assets(launch):
    started = time.monotonic()
    old = read(check(launch['asset_restoration_launch']))
    assert old['status'] == 'CROSS_FRAMEWORK_SDXL_PUBLIC_ASSET_RESTORE_ACTUALLY_DISPATCHED'
    done = AUDIT / 'public_assets_restored.json'
    while not done.exists():
        assert time.monotonic() - started < launch['asset_wait_max_seconds'], 'Finite asset wait exhausted'
        proc = Path('/proc') / str(old['pid']) / 'cmdline'
        assert proc.exists(), 'Asset restoration ended without a success receipt; preserve and stop'
        command = proc.read_bytes()
        assert b'restore_ATMS_standard_SDXL_assets_CPU_20261028_v1.py' in command
        progress('WAITING_EXISTING_PUBLIC_ASSET_RESTORATION_NO_GPU_JOB', restore_pid=old['pid'], wait_seconds=time.monotonic() - started)
        time.sleep(30)
    value = read(done)
    assert value['status'] == 'PINNED_STANDARD_SDXL_AND_H94_ADAPTER_RESTORED_CPU_SHA_VERIFIED'
    for snapshot in value['snapshots']:
        for item in snapshot['files']:
            check(item)
    return ref(done)


def prepare_latents(launch, assets):
    """Restore absent native inputs as a separate, fully described new bank."""
    historical = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/latents/idea_interface_T_mode_v1/completion.json')
    prior = read(historical)
    assert prior['status'] == 'COMPLETED_ALL13230_T_IMAGE_LATENTS'
    missing = [item['path'] for item in prior['parts'] if not Path(item['path']).is_file()]
    if not missing:
        for item in prior['parts']:
            check(item)
        return ref(historical)
    bank = OUTPUT / 'native_T_latents'
    assert not bank.exists(), 'Preserve failed bank attempts; do not silently restart'
    pins = {item['path']: item for item in launch['source_pins']}
    references = dict(authorization=launch['authorization'], split=ref(ROOT / 'configs/split_manifest.json'),
        image_manifest=ref(ROOT / 'audit/image_manifest.csv'),
        snapshot_completion=ref('/dev/shm/eeg_sota_autoresearch_20261018_v1/sdxl_base_46216598/completion.json'))
    for item in references.values():
        check(item); pins[item['path']] = item
    pins[assets['path']] = assets
    for item in read(references['snapshot_completion']['path'])['files']:
        if Path(item['path']).parent.name == 'vae':
            check(item); pins[item['path']] = item
    spec = dict(status='X01_STANDARD_NATIVE_T_VAE_MODE_REBUILD_LOCKED_BEFORE_GPU', seed=1729,
        source_pins=list(pins.values()), **references, historical_completion=ref(historical),
        historical_missing_parts=missing, old_metadata_overwritten=False,
        model_root='/dev/shm/eeg_sota_autoresearch_20261018_v1/sdxl_base_46216598',
        maximum_GPU_bytes=20 << 30, maximum_extra_CPU_bytes=10 << 30, maximum_output_bytes=2 << 30,
        generator_revision='462165984030d82259a11f4367a4eed129e94a7b', new_model_or_training=False)
    path = ROOT / 'configs/cross_framework_ordinary_20261028_v1/native_T_latent_rebuild_spec.json'
    atomic(path, spec)
    gpu_stage('X01_ATMS_standard_native_T_latents_20261028_v1', LATENT_BUILDER,
        ['--spec', str(path), '--output', str(bank), '--max-seconds', '1800'], 1800)
    completion = bank / 'completion.json'
    assert read(completion)['status'] == 'COMPLETED_ALL13230_T_IMAGE_LATENTS'
    return ref(completion)


def build_spec(launch, assets, native_latents):
    pins = {item['path']: item for item in launch['source_pins']}
    export_path = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1/candidates/D15_cross_encoder/export/completion.json')
    export = read(export_path)
    assert export['status'] == 'D15_ALL10_COMPLETE_ATMS_OOF_T_AND_FULL_D_REEXPORTED'
    references = dict(split=ref(ROOT / 'configs/split_manifest.json'), image_manifest=ref(ROOT / 'audit/image_manifest.csv'),
        ATMS_export_completion=ref(export_path), ATMS_input_completion=export['inputs'],
        CLIP_H_targets=ref('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt'),
        standard_latents_completion=native_latents)
    for item in references.values():
        check(item); pins[item['path']] = item
    pins[assets['path']] = assets
    for key in ('T_ATMS_OOF', 'D_ATMS_full', 'affines'):
        item = export[key]; check(item); pins[item['path']] = item
    latent = read(references['standard_latents_completion']['path'])
    assert latent['status'] == 'COMPLETED_ALL13230_T_IMAGE_LATENTS'
    for item in latent['parts']:
        check(item)
    assert shutil.disk_usage(OUTPUT.parent).free > 6 << 30, 'Preserve full 4GiB output and 2GiB reserve'
    spec = dict(status='ATMS_STANDARD_SDXL_EQUIVALENT_ORDINARY_LOCKED_BEFORE_EFFECTS', UTC=utc(),
        source_pins=list(pins.values()), **references, seed=1729, passes=3, batch_size=16, segment_max_updates=512,
        maximum_GPU_bytes=20 << 30, maximum_extra_CPU_bytes=10 << 30, maximum_output_bytes=4 << 30,
        radius=.25, token_loss_weight=.1, learning_rate=1e-4, weight_decay=.01, gradient_clip=1.,
        queries_per_pass=132300, total_updates=24807, total_exposures=396900, fresh_ordinary_parameters=525568,
        framework='Released frozen ATMS encoder, public native h94 IP-Adapter, standard SDXL-base',
        scope_limit='Direct ATMS branch, not the complete author diffusion-prior/Turbo generation pipeline',
        inference=dict(resolution=512, steps=20, scheduler='EulerDiscreteScheduler', CFG=5., IP_scale=1., images_per_query=1),
        training_objective='Native frozen SDXL epsilon MSE plus0.1 native P token cosine error; no BrainHIVE F/A',
        final_checkpoint_selection='Fixed last24807th update, no epoch/seed selection',
        calibration='Dfit8200 only; strength0,.25,.5,.75,1 inside radius, ties smaller',
        evaluation='Separate matched native ATMS and newordinary images on fixed400 Dcheck gallery,80->400; no HIVE images as native ATMS control',
        old_BrainHIVE_ordinary_changed=False, formal_D_resumed=False, novelty_or_image_gain_established=False)
    path = ROOT / 'configs/cross_framework_ordinary_20261028_v1/candidate_spec.json'
    atomic(path, spec)
    return path


def cleanup_owned():
    global ACTIVE
    if ACTIVE_JOB is not None and ACTIVE_JOB.exists():
        record = read(ACTIVE_JOB)
        pid = record.get('pid'); proc = Path('/proc') / str(pid) / 'cmdline'
        if pid and proc.exists():
            command = proc.read_bytes()
            if any(name.encode() in command for name in (TRAINER, IMAGE_PROFILE, LATENT_BUILDER, IMAGE_GENERATOR, IMAGE_SCORER)):
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                until = time.monotonic() + 30
                while proc.exists() and proc.read_bytes() and time.monotonic() < until:
                    time.sleep(.5)
                if proc.exists() and proc.read_bytes():
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.terminate()
        try:
            ACTIVE.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ACTIVE.kill(); ACTIVE.wait(timeout=10)
    ACTIVE = None


def gpu_stage(name, script, arguments, maximum_seconds):
    global ACTIVE, ACTIVE_JOB
    ACTIVE_JOB = ROOT / 'audit' / (name + '_job.json')
    assert not ACTIVE_JOB.exists()
    command = [PY, '-B', str(ROOT / 'run_gpu_job_v2.py'), '--name', name,
        '--max-seconds', str(maximum_seconds + 45), '--', PY, '-B', str(ROOT / script), *arguments]
    log_path = AUDIT / (name + '.log')
    progress('RUNNING_FINITE_GPU_STAGE', name=name, command=command)
    env = os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with log_path.open('x') as stream:
        ACTIVE = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        code = ACTIVE.wait(timeout=maximum_seconds + 105)
    record = read(ACTIVE_JOB)
    assert code == 0 and record['status'] == 'COMPLETED' and record['exit_code'] == 0, str(log_path)
    ACTIVE = None
    atomic(AUDIT / (name + '_stage.json'), dict(status='FINITE_GPU_STAGE_CLOSED', job=ref(ACTIVE_JOB), log=ref(log_path)))


def audit_checkpoint(spec_path, previous_updates, serial):
    # Original runtime import order is fixed before loading a CPU checkpoint.
    sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
    import torch
    from train_ATMS_standard_SDXL_ordinary_20261028_v1 import make_orders, order_sha, verify_prefix, verify_cursor
    import train_ATMS_standard_SDXL_ordinary_20261028_v1 as trainer
    trainer.torch = torch
    path = OUTPUT / 'train/resume.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['status'] == 'COMMITTED_COMPLETE_UPDATE_BOUNDARY'
    assert previous_updates < saved['updates'] <= min(previous_updates + 512, 24807)
    orders = make_orders()
    assert saved['orders_sha256'] == [order_sha(order) for order in orders]
    assert all(torch.equal(left, right) for left, right in zip(saved['orders'], orders))
    verify_cursor(saved['cursor'], saved['updates'], saved['seen'], saved['coverage'], orders)
    tail = verify_prefix(OUTPUT / 'train/training.jsonl', saved['log_prefix'])
    assert tail['bytes'] == 0, 'Successful segment must end at its committed journal boundary'
    assert saved['input_provenance_sha256'] == sha(OUTPUT / 'train/input_provenance.json')
    assert read(OUTPUT / 'train/input_provenance.json')['spec']['sha256'] == sha(spec_path)
    assert len(saved['optimizer']['state']) == len(saved['model']) == 4
    assert all(torch.isfinite(value).all() for value in saved['model'].values())
    for item in saved['optimizer']['state'].values():
        assert int(item['step']) == saved['updates']
        assert torch.isfinite(item['exp_avg']).all() and torch.isfinite(item['exp_avg_sq']).all()
    assert set(saved['RNG']) == {'python', 'numpy', 'CPU', 'CUDA', 'noise'}
    assert saved['RNG']['CPU'].dtype == saved['RNG']['noise'].dtype == torch.uint8
    assert len(saved['RNG']['CUDA']) == 1 and saved['RNG']['CUDA'][0].dtype == torch.uint8
    assert not torch.cuda.is_initialized(), 'Checkpoint audit must remain CPU-only'
    done = dict(status='REAL_ATMS_ORDINARY_COMMITTED_CHECKPOINT_CPU_VERIFIED', UTC=utc(),
        checkpoint=ref(path), updates=saved['updates'], seen=saved['seen'],
        previous_updates=previous_updates, model_optimizer_RNG_order_coverage_log_prefix_verified=True,
        source='CPU structural/finite and committed-prefix audit, not an independent numerical rerun of GPU training',
        source_sha256=sha(__file__), CUDA_initialized=False)
    atomic(AUDIT / ('checkpoint_audit_' + str(serial).zfill(3) + '.json'), done)
    return saved['updates']


def cpu_stage(name, script, arguments, maximum_seconds):
    """CPU-only producer commands remain finite and leave immutable call logs."""
    global ACTIVE
    log = AUDIT / (name + '.log')
    command = [PY, '-B', str(ROOT / script), *arguments]
    progress('RUNNING_FINITE_CPU_STAGE', name=name, command=command)
    with log.open('x') as stream:
        ACTIVE = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        code = ACTIVE.wait(timeout=maximum_seconds)
    ACTIVE = None
    assert code == 0, str(log)
    atomic(AUDIT / (name + '_stage.json'), dict(status='FINITE_CPU_STAGE_CLOSED', command=command, log=ref(log)))


def create_image_policy(spec):
    policy = ROOT / 'configs/cross_framework_ordinary_20261028_v1/image_policy.json'
    output = OUTPUT / 'paired_images'
    prior = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1/candidates/D09_gain_calibration/stage400_recovered_v3/completion.json')
    assert read(prior)['status'] == 'COMPLETE_D09_STAGE400'
    cpu_stage('X01_ATMS_ordinary_image_policy_before_training', IMAGE_GENERATOR,
        ['--spec', str(policy), '--output', str(output), '--mode', 'policy', '--training-spec', str(spec),
         '--training-output-root', str(OUTPUT), '--D09-completion', str(prior)], 120)
    assert read(policy)['status'] == 'ATMS_EQUIVALENT_ORDINARY_IMAGE_POLICY_FIXED_BEFORE_IMAGES'
    return policy, output


def paired_image_evaluation(policy, output):
    common = ['--spec', str(policy), '--output', str(output)]
    gpu_stage('X01_ATMS_ordinary_prepare_all400_conditions_noise_20261028_v1', IMAGE_GENERATOR,
        common + ['--mode', 'prepare', '--max-seconds', '600'], 600)
    assert read(output / 'prepared/completion.json')['status'] == 'ALL400_ATMS_NATIVE_ORDINARY_CONDITIONS_NOISE_FIXED_BEFORE_IMAGES'
    for stage, maximum_segments in ((80, 16), (400, 64)):
        for serial in range(1, maximum_segments + 1):
            closed = output / ('image_closure' + str(stage) + '.json')
            if closed.exists():
                break
            receipts_before = len(list((output / 'images').glob('*/*.json')))
            arguments = common + ['--mode', 'generate', '--stage', str(stage), '--max-seconds', '1800']
            if receipts_before:
                arguments.append('--resume')
            gpu_stage('X01_ATMS_images_' + str(stage) + '_' + str(serial).zfill(3) + '_20261028_v1',
                IMAGE_GENERATOR, arguments, 1800)
            receipts_after = len(list((output / 'images').glob('*/*.json')))
            assert receipts_before < receipts_after <= 2 * stage, 'Require real image progress, no automatic spin'
            progress('MATCHED_ATMS_IMAGES_COMMITTED', stage=stage, images=receipts_after,
                expected_images=2 * stage, intermediate_effect_scores_read=False)
        closed = output / ('image_closure' + str(stage) + '.json')
        assert read(closed)['status'] == 'COMPLETE_FIXED_ATMS_PAIRED_IMAGE_STAGE'
        cpu_stage('X01_ATMS_images_' + str(stage) + '_CPU_audit', IMAGE_GENERATOR,
            common + ['--mode', 'audit', '--stage', str(stage), '--max-seconds', '600'], 660)
        gpu_stage('X01_ATMS_images_' + str(stage) + '_DINO_features_20261028_v1', IMAGE_SCORER,
            common + ['--mode', 'features', '--stage', str(stage), '--max-seconds', '600'], 600)
        cpu_stage('X01_ATMS_images_' + str(stage) + '_fixed_score', IMAGE_SCORER,
            common + ['--mode', 'score', '--stage', str(stage), '--max-seconds', '600'], 660)
        result_path = output / ('score' + str(stage)) / 'completion.json'
        result = read(result_path)
        assert result['status'] == 'COMPLETE_ATMS_ORDINARY_PAIRED_DINO_SCREEN'
        action = result['decision']['action']
        if stage == 80 and action == 'EXPAND400':
            continue
        assert action in ['STOP_INSTANCE', 'PROMISING_FOR_FULL_EVALUATION']
        atomic(AUDIT / 'actual_image_screen_completion.json', dict(
            status='X01_MATCHED_ATMS_NATIVE_ORDINARY_ACTUAL_IMAGE_SCREEN_CLOSED', UTC=utc(),
            stage=stage, decision=result['decision'], score=ref(result_path),
            image_policy=ref(policy), mean_metrics=result['means'], paired=result['ordinary_minus_native'],
            independent_confirmation=False, SOTA_established=False,
            full_evaluation_required_if_promising=action == 'PROMISING_FOR_FULL_EVALUATION'))
        progress('MATCHED_ATMS_IMAGE_SCREEN_CLOSED', stage=stage, decision=result['decision'])
        return
    raise AssertionError('Both registered stages ended without a final fixed decision')


def run(launch_path):
    launch = read(launch_path); check_launch(launch)
    assets = wait_assets(launch); check_launch(launch)
    native_latents = prepare_latents(launch, assets)
    spec = build_spec(launch, assets, native_latents)
    policy, image_output = create_image_policy(spec)
    args = ['--spec', str(spec), '--output', str(OUTPUT)]
    gpu_stage('X01_ATMS_ordinary_profile_20261028_v1', TRAINER, args + ['--mode', 'profile', '--max-seconds', '900'], 900)
    assert read(OUTPUT / 'profile/completion.json')['status'] == 'ATMS_STANDARD_ORDINARY_ENGINEERING_PROFILE_PASSED'
    gpu_stage('X01_ATMS_ordinary_native_CFG_image_replay_20261028_v1', IMAGE_PROFILE,
        ['--spec', str(spec), '--output', str(OUTPUT / 'engineering_images'), '--max-seconds', '900'], 900)
    # The supplemental entry must prove real native CFG5 images before fitting.
    engineering = read(OUTPUT / 'engineering_images/completion.json')
    assert engineering['status'] == 'ATMS_STANDARD_ORDINARY_ALL20_AFFINES_AND_SIX_CFG5_ENGINEERING_IMAGES_PASSED'
    updates = 0
    for serial in range(1, launch['maximum_training_segments'] + 1):
        check_launch(launch)
        if updates == 24807:
            break
        arguments = args + ['--mode', 'train', '--max-seconds', '1800']
        if updates:
            arguments.append('--resume')
        gpu_stage('X01_ATMS_ordinary_train_' + str(serial).zfill(3) + '_20261028_v1', TRAINER, arguments, 1800)
        updates = audit_checkpoint(spec, updates, serial)
        progress('FULL_OOF_ORDINARY_TRAINING_PROGRESS_CPU_VERIFIED', updates=updates,
            expected_updates=24807, checkpoint_audit=str(AUDIT / ('checkpoint_audit_' + str(serial).zfill(3) + '.json')))
    assert updates == 24807, 'Finite128segment queue ended before full recipe; preserve checkpoint and stop'
    complete = read(OUTPUT / 'train/completion.json')
    assert complete['status'] == 'COMPLETE_ALL10_ATMS_OOF_THREE_PASSES_ORDINARY_TRAINING'
    assert complete['seen'] == 396900 and complete['per_pass_unique_query_coverage'] == [132300] * 3
    gpu_stage('X01_ATMS_ordinary_Dfit_calibration_20261028_v1', TRAINER,
        args + ['--mode', 'calibrate', '--max-seconds', '900'], 900)
    calibration = OUTPUT / 'calibrate/completion.json'
    assert read(calibration)['status'] == 'ATMS_STANDARD_ORDINARY_DFIT_ONLY_FIXED_GRID_CALIBRATION_COMPLETE'
    atomic(AUDIT / 'training_pipeline_completion.json', dict(status='X01_FULL_EQUIVALENT_ORDINARY_TRAINING_AND_DFIT_COMPLETE_READY_FOR_MATCHED_IMAGES',
        UTC=utc(), spec=ref(spec), train=ref(OUTPUT / 'train/completion.json'), calibration=ref(calibration),
        new_research_images=0, image_benefit_established=False, next='Locked nativeATMS versus newordinary paired80->400 actual images'))
    progress('FULL_TRAINING_CLOSED_START_MATCHED_IMAGE_EVALUATION', updates=24807, image_benefit_established=False)
    paired_image_evaluation(policy, image_output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-review', type=Path, required=True)
    args = parser.parse_args()
    assert ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    AUDIT.mkdir(parents=True, exist_ok=True); OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, _frame):
        raise InterruptedError('Controller explicitly stopped by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    with (AUDIT / 'ordinary_pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.launch_review)
        except BaseException as error:
            cleanup_owned()
            atomic(AUDIT / ('pipeline_failure_' + str(time.time_ns()) + '.json'),
                dict(status='X01_PIPELINE_STOPPED_FAILURE_AND_CHECKPOINTS_PRESERVED_NO_AUTOMATIC_RETRY',
                    UTC=utc(), error=repr(error), traceback=traceback.format_exc(), image_gain_established=False))
            raise


if __name__ == '__main__':
    main()
