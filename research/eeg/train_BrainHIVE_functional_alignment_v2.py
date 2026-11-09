"""Independent O/P/PN/S entry point; no existing run or v1 file is modified.

All modes use frozen, unmodified-parent Q at the SAME noisy image/timestep.
They execute two explicit UNet forwards per update. Native checkpoint block
recomputations are additional work, not hidden extra image sampling. This file
does not issue its own admission, reserve disk, or launch a scheduler.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from brainhive_native_models_v1 import *
from brainhive_bounded_component_v1 import BoundedConditionAdapter
from brainhive_component_data_v1 import load_training
from brainhive_component_generator_v1 import load_native, denoise
from brainhive_functional_alignment_v2 import NativeIPFunctionalAlignmentCapture, assert_engineering_signal
from brainhive_functional_training_contract_v2 import (
    ARMS, core_sha, validate_recipe, teacher_mappings, verify_admission, read,
)
from train_BrainHIVE_bounded_component_v1 import development, seed


def verify_exclusive_wrapper(recipe, purpose, admission):
    """Check the trusted, already-locked wrapper before any CUDA allocation."""
    assert str(ROOT) == '/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1'
    parent = Path('/proc') / str(os.getppid())
    command = parent.joinpath('cmdline').read_bytes().split(b'\0')
    assert str(ROOT / 'run_gpu_job_v2.py').encode() in command
    lock_path = ROOT / 'audit/GPU_JOB.lock'
    lock_stat = lock_path.stat()
    held_descriptor = False
    for fd in parent.joinpath('fd').iterdir():
        try:
            details = fd.stat()
            held_descriptor |= (details.st_dev, details.st_ino) == (lock_stat.st_dev, lock_stat.st_ino)
        except FileNotFoundError:
            pass
    assert held_descriptor, 'Parent must retain the actual GPU lock descriptor'
    import fcntl
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise AssertionError('GPU lock is not held by the trusted wrapper')
    assert b'--name' in command
    job_name = command[command.index(b'--name') + 1].decode()
    permitted = [recipe['profile_job_name']] if purpose == 'profile' else recipe['formal_job_names']
    assert Path(job_name).name == job_name and job_name in permitted
    job = read(ROOT / 'audit' / (job_name + '_job.json'))
    assert job['status'] == 'RUNNING' and job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid()
    assert 0 < job['max_seconds'] <= recipe['profile_job_seconds' if purpose == 'profile' else 'training_job_seconds']
    expected_auth = recipe['input_sha256'][str(ROOT / 'configs/resource_authorization_20261020_v3.json')]
    assert job['authorization_sha256'] == expected_auth
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          check=True, capture_output=True, text=True, timeout=10)
    active = [int(value.strip()) for value in apps.stdout.splitlines() if value.strip()]
    assert set(active) <= {os.getpid()}, ('Another GPU process is active', active)
    memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10)
    free = [int(value.strip()) * (1 << 20) for value in memory.stdout.splitlines() if value.strip()]
    assert len(free) == 1, 'The admitted experiment uses the existing single GPU only'
    receipt = read(recipe['admission_receipt'])
    assert free[0] >= receipt['resource_limits']['minimum_free_GPU_bytes']
    return dict(wrapper_pid=os.getppid(), active_GPU_processes=active, free_GPU_bytes=free[0],
                wrapper_job_name=job_name, admission_sha256=admission['admission_sha256'])


def actual_teacher_mappings(inputs):
    protocol = read(ROOT / 'configs/split_manifest.json')
    manifest_path = ROOT / 'audit/image_manifest.csv'
    assert sha(manifest_path) == protocol['image_manifest_sha256']
    rows = {}
    with manifest_path.open() as stream:
        for row in csv.DictReader(stream):
            if row['dataset_split'] == 'train':
                index = int(row['array_row'])
                assert index not in rows
                rows[index] = row['concept_id']
    allowed = inputs['array_rows']
    assert allowed == protocol['image_rows']['T'] and len(allowed) == 13230
    assert set(allowed).isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    result = teacher_mappings(allowed, rows)
    assert result['receipt']['concept_count'] == 1323 and result['receipt']['stimuli_per_concept'] == 10
    receipt = result.pop('receipt')
    receipt.update(image_manifest_sha256=sha(manifest_path), split_manifest_sha256=sha(ROOT / 'configs/split_manifest.json'))
    return {key: torch.tensor(value, dtype=torch.long, device='cuda') for key, value in result.items()}, receipt


def functional_step(model, pipe, adapter, projector, prompts, scheduler, collector,
                    inputs, image_latents, mapping, indices, rng, arm):
    assert arm in ARMS and not collector.active and not collector.queries
    positions = (indices % 13230).cuda()
    query_indices = indices.cuda()
    condition = inputs['condition'][query_indices]
    affine = inputs['affine_ids'][query_indices]
    changed, information = model(condition, inputs['weight'][affine], inputs['bias'][affine], gate_mode='fixed')
    assert model.predict_risk is False and model.radius == .25 and torch.equal(information['gate'], torch.ones_like(information['gate']))
    latent = image_latents[positions]
    noise = torch.randn(latent.shape, device='cuda', generator=rng)
    timesteps = torch.randint(0, 1000, (len(indices),), device='cuda', generator=rng)
    noisy = scheduler.add_noise(latent, noise, timesteps)
    forwards = [0]
    def count_forward(module, args, result):
        forwards[0] += 1
    handle = pipe.unet.register_forward_hook(count_forward)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                base_tokens = projector(adapter(condition))
                real = projector(adapter(inputs['target'][positions]))
                negative = projector(adapter(inputs['target'][mapping['negative'][positions]]))
                if arm == 'S':
                    rank_positive = projector(adapter(inputs['target'][mapping['rank_positive_shuffled'][positions]]))
                    rank_negative = projector(adapter(inputs['target'][mapping['rank_negative_shuffled'][positions]]))
                else:
                    rank_positive = rank_negative = None
                with collector.capture():
                    denoise(pipe, prompts, noisy, timesteps, base_tokens)
            assert not collector.active and len(collector.queries) == 70
            # Version/pointer identity ensures the student does not overwrite
            # the reference bank. No enabled=False context: parent Q is present.
            parent_Q = {key: (value, value.data_ptr(), value._version) for key, value in collector.queries.items()}
            tokens = projector(adapter(changed))
            predicted = denoise(pipe, prompts, noisy, timesteps, tokens)
            assert forwards[0] == 2 and not collector.active
            assert set(parent_Q) == set(collector.queries)
            assert all(collector.queries[key] is value and value.data_ptr() == pointer and value._version == version
                       and not value.requires_grad for key, (value, pointer, version) in parent_Q.items())
            if arm == 'O':
                collector.queries.clear()
                auxiliary = tokens.float().sum() * 0.
                diagnostics = {}
            else:
                mode = 'positive_only' if arm == 'P' else 'full'
                auxiliary, diagnostics = collector.auxiliary(tokens, real, negative, mode=mode,
                    rank_positive_tokens=rank_positive, rank_negative_tokens=rank_negative)
            image_loss = F.mse_loss(predicted.float(), noise)
            token_loss = (1 - F.cosine_similarity(tokens.float(), real.float(), dim=-1)).mean()
            loss = image_loss + .1 * token_loss + .1 * auxiliary
        assert torch.isfinite(loss) and not collector.queries
        return dict(loss=loss, image_loss=image_loss, token_loss=token_loss, auxiliary=auxiliary,
                    diagnostics=diagnostics, information=information, timesteps=timesteps,
                    UNet_forward_count=forwards[0], reference_Q_unchanged=True)
    finally:
        handle.remove()
        if collector.queries:
            collector.queries.clear()


def frozen_gradient_check(pipe, adapter, projector):
    assert all(p.grad is None for module in [pipe.unet, adapter, projector] for p in module.parameters())


def engineering_profile(recipe, model, optimizer, pipe, adapter, projector, prompts,
                        scheduler, collector, inputs, image_latents, mapping, orders, rng, provenance):
    path = Path(recipe['profile_output'])
    assert not path.exists(), 'Preserve existing profiles; no automatic replacement or repeated selection'
    started = time.monotonic()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    fixed_indices = orders[0][:16]
    for update in range(2):
        assert time.monotonic() - started < recipe['profile_job_seconds'] - 30
        torch.cuda.synchronize()
        update_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        result = functional_step(model, pipe, adapter, projector, prompts, scheduler, collector,
            inputs, image_latents, mapping, fixed_indices, rng, recipe['arm'])
        if recipe['arm'] != 'O':
            gradient = torch.autograd.grad(result['auxiliary'], model.direction.weight, retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
            auxiliary_norm = float(gradient.norm())
            signal = assert_engineering_signal(result['diagnostics'], student_gradient_norm=auxiliary_norm,
                minimum_nondegenerate_teacher_fraction=.5)
        else:
            auxiliary_norm, signal = None, None
            assert float(result['auxiliary']) == 0 and not result['diagnostics']
        result['loss'].backward()
        total_gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        assert model.direction.weight.grad is not None and model.direction.weight.grad.norm() > 0
        frozen_gradient_check(pipe, adapter, projector)
        optimizer.step()
        torch.cuda.synchronize()
        rows.append(dict(update=update + 1, loss=float(result['loss'].detach()),
            image_loss=float(result['image_loss'].detach()), token_loss=float(result['token_loss'].detach()),
            auxiliary=float(result['auxiliary'].detach()), auxiliary_only_direction_gradient_norm=auxiliary_norm,
            engineering_signal=signal, total_gradient_norm=float(total_gradient),
            diagnostics={key: float(value) for key, value in result['diagnostics'].items()},
            UNet_forwards=result['UNet_forward_count'], reference_Q_unchanged=result['reference_Q_unchanged'],
            update_seconds=time.monotonic() - update_started))
    torch.cuda.synchronize()
    forecast = 1.25 * max(row['update_seconds'] for row in rows) * 24807
    assert forecast <= recipe['training_job_seconds'], 'Two-step engineering forecast exceeds the prespecified 24h job boundary; no automatic timeout rewrite'
    report = dict(status='ACTUAL_PARENT_Q_TWO_FORWARD_ENGINEERING_PROFILE_COMPLETE',
        candidate=recipe['candidate'], arm=recipe['arm'], actual_GPU=True,
        recipe_core_sha256=core_sha(recipe), source_sha256=recipe['source_sha256'], provenance=provenance,
        updates=2, batch_size=16, distinct_queries=16, query_exposures=32,
        UNet_forwards_per_update=[row['UNet_forwards'] for row in rows], rows=rows,
        actual_parent_Q_engineering_verified=all(row['reference_Q_unchanged'] for row in rows),
        linear_training_time_forecast_seconds=forecast, fixed_training_budget_seconds=86400,
        time_forecast_is_not_runtime_guarantee=True, training_timeout_not_rewritten_after_profile=True,
        seconds=time.monotonic() - started, peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
        engineering_checks_passed=True, threshold_is_engineering_not_scientific_trigger=True,
        formal_training=False, saved_trained_checkpoints=0, generated_images=0,
        V_or_test_score_access=False, image_gain_established=False, risk_gate_mechanism_tested=False)
    encoded = json.dumps(report, indent=2, allow_nan=False).encode()
    admission = read(recipe['admission_receipt'])
    assert len(encoded) <= admission['resource_limits']['profile_metadata_budget_bytes']
    write(path, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ('provenance', 'source_sha256')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert not (args.profile_only and args.resume)
    recipe = read(args.recipe)
    validate_recipe(recipe)
    purpose = 'profile' if args.profile_only else 'formal'
    admission = verify_admission(recipe, purpose)
    if args.preflight_only:
        print(json.dumps(dict(status='CPU_ADMISSION_CHECK_ONLY_NO_GPU_ALLOCATION', purpose=purpose, checks=admission)), flush=True)
        return
    exclusive = verify_exclusive_wrapper(recipe, purpose, admission)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = load_training()
    assert inputs['input_contract_sha256'] == recipe['input_contract_sha256']
    mapping, mapping_receipt = actual_teacher_mappings(inputs)
    latent = read(recipe['latent_receipt'])
    assert latent['status'] == 'COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
    assert sha(latent['path']) == latent['sha256'] and latent['array_rows'] == inputs['array_rows']
    assert latent['format'] == 'numpy_npy_float32_C_contiguous_v2'
    import numpy as np
    array = np.load(latent['path'], mmap_mode='r')
    assert array.shape == (13230, 4, 64, 64) and array.dtype == np.float32
    image_latents = torch.tensor(array, device='cuda')
    del array
    pipe, adapter, projector, prompts, host = load_native(training=True)
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    assert scheduler.config.prediction_type == 'epsilon'
    torch.manual_seed(1729)
    model = BoundedConditionAdapter(predict_risk=False).cuda()
    assert sum(p.numel() for p in model.parameters()) == 525568
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    rng = torch.Generator(device='cuda').manual_seed(seed('shared_training_noise'))
    orders = [torch.randperm(132300, generator=torch.Generator().manual_seed(seed('pass' + str(i)))) for i in range(3)]
    collector = NativeIPFunctionalAlignmentCapture(pipe.unet)
    provenance = dict(candidate=recipe['candidate'], arm=recipe['arm'], recipe_core_sha256=core_sha(recipe),
        recipe_sha256=sha(args.recipe), admission_sha256=admission['admission_sha256'],
        input_contract_sha256=inputs['input_contract_sha256'], latent_receipt_sha256=sha(recipe['latent_receipt']),
        mapping=mapping_receipt, host=host, exclusive_GPU=exclusive,
        orders_sha256=[hashlib.sha256(order.numpy().tobytes()).hexdigest() for order in orders],
        reference_Q_policy=recipe['reference_Q_policy'], explicit_UNet_forwards_per_update=2,
        actual_native_IP_layers=len(collector.layers), maximum_Q_positions=16,
        source_sha256=recipe['source_sha256'], head_parameters=525568,
        risk_head_or_gate=False, all_frozen_host_weights=True,
        shuffled_rank_pair_only=recipe['arm'] == 'S', correct_positive_MSE_unchanged_in_S=True,
        targets_available_at_normal_inference=False,
        loss='epsilon_MSE+.1*token_angular+.1*functional_auxiliary; O0/Ppositive/PNpositive+rank/Spositive+shuffledrank',
        auxiliary_SDP_compute_differs_by_mode_and_must_be_reported=True)
    try:
        if args.profile_only:
            engineering_profile(recipe, model, optimizer, pipe, adapter, projector, prompts,
                scheduler, collector, inputs, image_latents, mapping, orders, rng, provenance)
            return
        out = Path(recipe['output_directory'])
        if args.resume:
            assert out.exists() and (out / 'resume.pt').exists()
        else:
            assert not out.exists(), 'New v2 output only; preserve every prior run'
            out.mkdir(parents=True)
        assert not (out / 'completion.json').exists()
        # The job wrapper PID changes on exact resume; bind invariant training
        # inputs separately from the per-invocation audited launch receipt.
        invariant = {key: value for key, value in provenance.items() if key not in ('exclusive_GPU', 'admission_sha256', 'recipe_sha256')}
        locked = out / 'input_provenance.json'
        if locked.exists():
            assert read(locked) == invariant
        else:
            write(locked, invariant)
        launch_path = out / ('launch_' + str(os.getpid()) + '.json')
        assert not launch_path.exists()
        write(launch_path, provenance)
        cursor = updates = seen = 0
        loss_sum = cumulative_seconds = 0.
        coverage = torch.zeros(3, 132300, dtype=torch.bool)
        if args.resume:
            saved = torch.load(out / 'resume.pt', weights_only=False, map_location='cpu')
            assert saved['input_provenance_sha256'] == sha(locked)
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            cursor, updates, seen, loss_sum = saved['cursor'], saved['updates'], saved['seen'], saved['loss_sum']
            coverage, cumulative_seconds = saved['coverage'], saved['seconds']
            rng.set_state(saved['noise_rng'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        started = last_saved = time.monotonic()
        assert cumulative_seconds < recipe['maximum_cumulative_training_seconds_per_arm'] - 90
        torch.cuda.reset_peak_memory_stats()
        def snapshot():
            verify_admission(recipe, 'formal')
            save_tensor(out / 'resume.pt', dict(model={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                optimizer=optimizer.state_dict(), cursor=cursor, updates=updates, seen=seen, loss_sum=loss_sum,
                coverage=coverage, noise_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), seconds=cumulative_seconds + time.monotonic() - started,
                input_provenance_sha256=sha(locked)))
        with (out / 'training.jsonl').open('a', buffering=1) as log:
            for epoch in range(cursor // 132300, 3):
                start_at = cursor % 132300 if epoch == cursor // 132300 else 0
                for offset in range(start_at, 132300, 16):
                    elapsed = time.monotonic() - started
                    if (elapsed > recipe['training_job_seconds'] - 90 or
                            cumulative_seconds + elapsed > recipe['maximum_cumulative_training_seconds_per_arm'] - 90):
                        snapshot()
                        write(out / 'progress.json', dict(status='FUNCTIONAL_V2_FINITE_RESOURCE_STOP_RESUMABLE', updates=updates, cursor=cursor))
                        raise RuntimeError('Finite v2 job timeout; exact resume retained')
                    indices = orders[epoch][offset:offset + 16]
                    optimizer.zero_grad(set_to_none=True)
                    result = functional_step(model, pipe, adapter, projector, prompts, scheduler, collector,
                        inputs, image_latents, mapping, indices, rng, recipe['arm'])
                    result['loss'].backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    assert model.direction.weight.grad is not None
                    frozen_gradient_check(pipe, adapter, projector)
                    optimizer.step()
                    assert not coverage[epoch, indices].any()
                    coverage[epoch, indices] = True
                    n = len(indices)
                    cursor, updates, seen = epoch * 132300 + offset + n, updates + 1, seen + n
                    loss_sum += float(result['loss'].detach()) * n
                    if updates % 64 == 0 or offset + n == 132300:
                        event = dict(stage='BRAINHIVE_FUNCTIONAL_ALIGNMENT_V2_REAL_OOF_TRAINING', arm=recipe['arm'],
                            pass_number=epoch + 1, updates=updates, expected_updates=24807,
                            seen_queries=seen, expected_queries=396900, cumulative_mean_loss=loss_sum / seen,
                            last_loss=float(result['loss'].detach()), epsilon_loss=float(result['image_loss'].detach()),
                            token_loss=float(result['token_loss'].detach()), functional_auxiliary=float(result['auxiliary'].detach()),
                            functional_diagnostics={key: float(value) for key, value in result['diagnostics'].items()},
                            UNet_forwards=result['UNet_forward_count'], gradient_norm=float(gradient),
                            correction_RMS=float(result['information']['correction'].square().mean().sqrt()),
                            mean_gate=1., seconds=cumulative_seconds + time.monotonic() - started)
                        log.write(json.dumps(event) + '\n')
                        write(out / 'progress.json', dict(status='FUNCTIONAL_ALIGNMENT_V2_RUNNING', **event))
                        print(json.dumps(event), flush=True)
                    if updates % 1024 == 0:
                        verify_admission(recipe, 'formal')
                    if time.monotonic() - last_saved > 120:
                        snapshot()
                        last_saved = time.monotonic()
        assert coverage.all() and seen == 396900 and updates == 24807
        snapshot()
        calibration = development(model.eval(), adapter, projector, 'ordinary')
        assert calibration['kind'] == recipe['strength_policy']
        write(out / 'calibration.json', calibration)
        save_tensor(out / 'component_final.pt', dict(model={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            candidate=recipe['candidate'], arm=recipe['arm'], seed=1729, updates=updates,
            seen_queries=seen, passes=3, calibration=calibration, recipe_core_sha256=core_sha(recipe),
            input_provenance_sha256=sha(locked), trained_from_scratch=True, risk_gate=False))
        write(out / 'completion.json', dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE',
            candidate=recipe['candidate'], arm=recipe['arm'], parameters=525568, updates=updates,
            subjects=10, unique_training_queries=132300, seen_queries=seen, passes=3,
            checkpoint_path=str(out / 'component_final.pt'), checkpoint_sha256=sha(out / 'component_final.pt'),
            recipe_core_sha256=core_sha(recipe), input_provenance_sha256=sha(locked),
            calibration_sha256=sha(out / 'calibration.json'),
            training_seconds=cumulative_seconds + time.monotonic() - started,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(), explicit_UNet_forwards_per_update=2,
            generated_images=0, image_gain_established=False, risk_gate_mechanism_tested=False,
            next_required='Full matched image generation and per-query O/P/PN/S comparisons'))
        print('FUNCTIONAL_ALIGNMENT_V2_FIXED_FINAL_TRAINING_COMPLETE_NO_IMAGE_GAIN_CLAIM', flush=True)
    finally:
        collector.close()


if __name__ == '__main__':
    main()
