"""One fixed response-discrimination experiment on real OOF-T conditions.

The native generator forward is unchanged. Detached Q samples are observed
once and the ranking auxiliary is built outside gradient-checkpoint regions.
There is no second full-UNet teacher forward or inference-time teacher access.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

from brainhive_native_models_v1 import *
from brainhive_bounded_component_v1 import BoundedConditionAdapter
from brainhive_component_data_v1 import load_training
from brainhive_component_generator_v1 import load_native, denoise
from brainhive_response_rank_v1 import NativeIPResponseCapture, concept_derangement
from train_BrainHIVE_bounded_component_v1 import development, seed


CANDIDATE = 'response_rank_v1'
COMPLETE = 'COMPLETE_ALL10_REAL_OOF_THREE_PASSES_AND_FIXED_FINAL_RESPONSE_RANK_COMPONENT'


def recipe_core_sha256(recipe):
    core = {key: value for key, value in recipe.items()
            if key not in {'storage_admission_receipt', 'storage_admission_sha256'}}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_recipe(recipe):
    fixed = dict(candidate=CANDIDATE, training_seed=1729, subjects=10,
                 unique_T_images=13230, unique_OOF_training_queries=132300,
                 passes=3, batch_size=16, learning_rate=.0001,
                 weight_decay=.01, gradient_clip=1.,
                 expected_updates=24807, trainable_parameters=525568,
                 maximum_standardized_direction_radius=.25,
                 response_rank_weight=.1, token_angular_weight=.1,
                 response_maximum_query_positions=16, response_expected_layers=70,
                 response_rank_epsilon=1e-6,
                 response_strength_policy='Dfit_only_global_strength_calibration')
    for key, expected in fixed.items():
        assert recipe[key] == expected, (key, recipe.get(key), expected)
    assert recipe['output_directory'] == str(WORK / 'components/two_ideas_v1' / CANDIDATE)
    assert recipe['generator'] == dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1)
    required = ['brainhive_native_models_v1.py', 'brainhive_bounded_component_v1.py',
                'brainhive_component_data_v1.py', 'brainhive_component_generator_v1.py',
                'train_BrainHIVE_bounded_component_v1.py', 'brainhive_response_rank_v1.py',
                'train_BrainHIVE_response_rank_v1.py', 'brainhive_two_ideas_storage_v1.py']
    for name in required:
        assert recipe['source_sha256'][name] == sha(ROOT / name), name
    mandatory_inputs = [WORK / 'data/component_input_contract_v1.json',
                        ROOT / 'configs/split_manifest.json',
                        ROOT / 'audit/image_manifest.csv', Path(recipe['latent_receipt'])]
    assert all(str(path) in recipe['input_sha256'] for path in mandatory_inputs), mandatory_inputs
    for filename, expected in recipe['input_sha256'].items():
        assert Path(filename).is_absolute() and sha(filename) == expected, filename
    assert 90 < recipe['training_job_seconds'] <= 86400
    return recipe_core_sha256(recipe)


def negative_positions(inputs):
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    manifest_path = ROOT / 'audit/image_manifest.csv'
    assert sha(manifest_path) == protocol['image_manifest_sha256']
    rows = {}
    with manifest_path.open() as stream:
        for row in csv.DictReader(stream):
            if row['dataset_split'] == 'train':
                index = int(row['array_row'])
                assert index not in rows
                rows[index] = row
    allowed = list(inputs['array_rows'])
    assert allowed == protocol['image_rows']['T'] and len(allowed) == 13230
    assert set(allowed).isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    concepts = {index: rows[index]['concept_id'] for index in allowed}
    negative = concept_derangement(allowed, concepts, seed=1729)
    assert all(concepts[allowed[i]] != concepts[allowed[j]] for i, j in enumerate(negative.tolist()))
    return negative, dict(negative_position_sha256=hashlib.sha256(negative.numpy().tobytes()).hexdigest(),
        seed=1729, scope='Fixed bijection of allowed T-image positions, shared across subjects and passes',
        different_concept_verified=True, stimulus_or_array_index_used_as_position=False,
        mapping_array_rows_sha256=hashlib.sha256(torch.tensor(allowed).numpy().tobytes()).hexdigest(),
        image_manifest_sha256=sha(manifest_path), split_manifest_sha256=sha(ROOT / 'configs/split_manifest.json'))


def step(model, pipe, adapter, projector, prompts, scheduler, collector,
         inputs, image_latents, negative, indices, rng):
    positions = (indices % 13230).cuda()
    indices_gpu = indices.cuda()
    affine = inputs['affine_ids'][indices_gpu]
    condition = inputs['condition'][indices_gpu]
    weight, bias = inputs['weight'][affine], inputs['bias'][affine]
    latent = image_latents[positions]
    noise = torch.randn(latent.shape, device='cuda', generator=rng)
    timesteps = torch.randint(0, 1000, (len(indices),), device='cuda', generator=rng)
    noisy = scheduler.add_noise(latent, noise, timesteps)
    changed, information = model(condition, weight, bias)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            real = projector(adapter(inputs['target'][positions]))
            wrong = projector(adapter(inputs['target'][negative[positions]]))
        tokens = projector(adapter(changed))
        with collector.capture():
            predicted = denoise(pipe, prompts, noisy, timesteps, tokens)
        # The capture is CLOSED before either auxiliary backward or checkpoint
        # recomputation. Q is detached; all three responses share that same Q.
        rank, response_diagnostics = collector.auxiliary(tokens, real, wrong)
        image_loss = F.mse_loss(predicted.float(), noise)
        token_loss = (1 - F.cosine_similarity(tokens.float(), real.float(), dim=-1)).mean()
        loss = image_loss + .1 * token_loss + .1 * rank
    assert torch.isfinite(loss)
    return loss, image_loss, token_loss, rank, information, response_diagnostics, timesteps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    assert str(ROOT / 'run_gpu_job_v2.py').encode() in (Path('/proc') / str(os.getppid()) / 'cmdline').read_bytes()
    assert not (args.profile and args.resume)
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recipe = json.loads(args.recipe.read_text())
    core_sha = validate_recipe(recipe)
    from brainhive_two_ideas_storage_v1 import verify_candidate_remaining
    if not args.profile:
        verify_candidate_remaining(recipe, CANDIDATE)
    else:
        assert recipe['engineering_profile_max_updates'] == 2
        assert recipe['engineering_profile_batch_size'] == 16
    inputs = load_training()
    assert inputs['input_contract_sha256'] == recipe['input_contract_sha256']
    negative, negative_receipt = negative_positions(inputs)
    negative = negative.cuda()
    latent_receipt = Path(recipe['latent_receipt'])
    latent = json.loads(latent_receipt.read_text())
    assert latent['status'] == 'COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
    assert sha(latent['path']) == latent['sha256']
    assert latent['array_rows'] == inputs['array_rows']
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
    # These namespaces deliberately match the ordinary control exactly. Adding
    # a deterministic ranking calculation consumes no extra training RNG draws.
    rng = torch.Generator(device='cuda').manual_seed(seed('shared_training_noise'))
    orders = [torch.randperm(132300, generator=torch.Generator().manual_seed(seed('pass' + str(i)))) for i in range(3)]
    collector = NativeIPResponseCapture(pipe.unet)
    provenance = dict(recipe_sha256=sha(args.recipe), recipe_core_sha256=core_sha,
        input_contract_sha256=inputs['input_contract_sha256'], latent_receipt_sha256=sha(latent_receipt),
        candidate=CANDIDATE, parameters=525568, host=host, negative_mapping=negative_receipt,
        orders_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders],
        ordinary_control_noise_and_order_namespaces_identical=True,
        native_processors_replaced=False, actual_native_IP_layers=len(collector.layers),
        maximum_Q_positions_per_layer=16, Q_detached_and_shared=True,
        student_teacher_response_space='Native IP SDP output before shared frozen output projection',
        loss='epsilon_MSE + 0.1*token_angular + 0.1*mean_layer_softplus((Dpos-Dneg)/(teacher_separation+1e-6))',
        true_or_negative_T_visual_conditions_read_at_normal_inference=False,
        model_is_new_and_host_frozen=True, extra_full_UNet_forwards_per_update=0)
    out = Path(recipe['output_directory'])
    if args.profile:
        torch.cuda.reset_peak_memory_stats()
        losses, rank_losses, gradients, rank_gradients = [], [], [], []
        diagnostics = []
        started = time.monotonic()
        try:
            profile_indices = orders[0][:16]
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = step(model, pipe, adapter, projector, prompts, scheduler, collector,
                              inputs, image_latents, negative, profile_indices, rng)
                loss, _, _, rank, _, values, _ = result
                rank_gradient = torch.autograd.grad(rank, model.direction.weight, retain_graph=True)[0]
                assert torch.isfinite(rank_gradient).all(), 'Nonfinite rank-only direction gradient'
                rank_gradients.append(float(rank_gradient.norm()))
                assert rank_gradients[-1] > 0, 'Ranking auxiliary has no actual direction gradient'
                assert float(values['teacher_separation']) > 1e-6, 'Actual IP teacher responses are degenerate'
                assert float(values['near_zero_teacher_fraction']) < 1., 'Every actual teacher pair is degenerate'
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                assert all(p.grad is None for p in pipe.unet.parameters())
                assert all(p.grad is None for module in [adapter, projector] for p in module.parameters())
                assert model.direction.weight.grad is not None
                gradients.append(float(model.direction.weight.grad.norm()))
                assert gradients[-1] > 0
                optimizer.step()
                losses.append(float(loss.detach()))
                rank_losses.append(float(rank.detach()))
                diagnostics.append({key: float(value) for key, value in values.items()})
            torch.cuda.synchronize()
            report = dict(status='ENGINEERING_ONLY_TWO_ACTUAL_OOF_T_RESPONSE_RANK_UPDATES',
                candidate=CANDIDATE, updates=2, batch_size=16, distinct_T_queries=16,
                training_query_exposures=32, losses=losses, rank_losses=rank_losses,
                direction_gradient_norms=gradients, rank_only_direction_gradient_norms=rank_gradients,
                actual_teacher_response_nondegeneracy_checked=True, response_diagnostics=diagnostics,
                seconds_per_update=(time.monotonic()-started)/2,
                peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),
                provenance=provenance, formal_training=False, generated_images=0, heldout_scores_read=False)
            path = WORK / 'audit/response_rank_actual_input_GPU_profile_v1.json'
            assert not path.exists(), 'Preserve earlier profiles'
            write(path, report)
            print(json.dumps(report), flush=True)
        finally:
            collector.close()
        return
    if args.resume:
        assert out.exists() and (out / 'resume.pt').exists()
    else:
        assert not out.exists(), 'Preserve previous runs; use the exact resume option when appropriate'
        out.mkdir(parents=True)
    assert not (out / 'completion.json').exists()
    provenance_path = out / 'input_provenance.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text()) == provenance
    else:
        write(provenance_path, provenance)
    cursor, updates, seen, loss_sum, cumulative_seconds = 0, 0, 0, 0., 0.
    coverage = torch.zeros(3, 132300, dtype=torch.bool)
    if args.resume:
        saved = torch.load(out / 'resume.pt', weights_only=False, map_location='cpu')
        assert saved['input_provenance_sha256'] == sha(provenance_path)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        cursor, updates, seen, loss_sum = saved['cursor'], saved['updates'], saved['seen'], saved['loss_sum']
        coverage = saved['coverage']
        rng.set_state(saved['noise_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        cumulative_seconds = saved['seconds']
    started = time.monotonic()
    last_saved = started
    torch.cuda.reset_peak_memory_stats()

    def snapshot():
        verify_candidate_remaining(recipe, CANDIDATE)
        save_tensor(out / 'resume.pt', dict(model={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            optimizer=optimizer.state_dict(), cursor=cursor, updates=updates, seen=seen, loss_sum=loss_sum,
            coverage=coverage, noise_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), seconds=cumulative_seconds + time.monotonic() - started,
            input_provenance_sha256=sha(provenance_path)))

    try:
        with (out / 'training.jsonl').open('a', buffering=1) as log:
            for epoch in range(cursor // 132300, 3):
                start_at = cursor % 132300 if epoch == cursor // 132300 else 0
                for offset in range(start_at, 132300, 16):
                    if time.monotonic() - started > recipe['training_job_seconds'] - 90:
                        snapshot()
                        write(out / 'progress.json', dict(status='RESOURCE_STOPPED_RESUMABLE', updates=updates, cursor=cursor))
                        raise RuntimeError('Finite training timeout boundary; exact resume retained')
                    indices = orders[epoch][offset:offset + 16]
                    optimizer.zero_grad(set_to_none=True)
                    loss, image_loss, token_loss, rank, info, values, timesteps = step(
                        model, pipe, adapter, projector, prompts, scheduler, collector,
                        inputs, image_latents, negative, indices, rng)
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    assert model.direction.weight.grad is not None
                    assert all(p.grad is None for p in pipe.unet.parameters())
                    assert all(p.grad is None for module in [adapter, projector] for p in module.parameters())
                    optimizer.step()
                    assert not coverage[epoch, indices].any()
                    coverage[epoch, indices] = True
                    n = len(indices)
                    cursor = epoch * 132300 + offset + n
                    updates += 1
                    seen += n
                    loss_sum += float(loss.detach()) * n
                    if updates % 64 == 0 or offset + n == 132300:
                        event = dict(stage='BRAINHIVE_RESPONSE_RANK_REAL_OOF_IMAGE_TRAINING', candidate=CANDIDATE,
                            pass_number=epoch + 1, updates=updates, expected_updates=24807,
                            seen_queries=seen, expected_queries=396900,
                            cumulative_mean_loss=loss_sum / seen, last_loss=float(loss),
                            epsilon_loss=float(image_loss), token_loss=float(token_loss), response_rank_loss=float(rank),
                            response_diagnostics={key: float(value) for key, value in values.items()},
                            timesteps=timesteps.tolist(), gradient_norm=float(gradient),
                            correction_RMS=float(info['correction'].square().mean().sqrt()),
                            mean_gate=float(info['gate'].mean()), seconds=cumulative_seconds + time.monotonic() - started)
                        log.write(json.dumps(event) + '\n')
                        write(out / 'progress.json', dict(status='RUNNING', **event))
                        print(json.dumps(event), flush=True)
                    if updates % 1024 == 0:
                        verify_candidate_remaining(recipe, CANDIDATE)
                    if time.monotonic() - last_saved > 120:
                        snapshot()
                        last_saved = time.monotonic()
        assert coverage.all() and seen == 396900 and updates == 24807
        snapshot()
        # Reuse the exact ordinary control calibration policy and implementation.
        calibration = development(model.eval(), adapter, projector, 'ordinary')
        assert calibration['kind'] == recipe['response_strength_policy']
        write(out / 'calibration.json', calibration)
        save_tensor(out / 'component_final.pt', dict(model={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            candidate=CANDIDATE, seed=1729, updates=updates, seen_queries=seen, passes=3,
            calibration=calibration, input_provenance_sha256=sha(provenance_path), recipe_sha256=sha(args.recipe)))
        write(out / 'completion.json', dict(status=COMPLETE, candidate=CANDIDATE, parameters=525568,
            subjects=10, unique_train_queries=132300, total_seen_queries=seen, updates=updates,
            checkpoint_path=str(out / 'component_final.pt'), checkpoint_sha256=sha(out / 'component_final.pt'),
            input_provenance_sha256=sha(provenance_path), calibration_sha256=sha(out / 'calibration.json'),
            training_seconds=cumulative_seconds + time.monotonic() - started,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),
            checkpoint_selection='Fixed final update; no V or official test target/score use',
            extra_full_UNet_forwards_per_training_update=0, inference_extra_UNet_forwards=0,
            response_teacher_and_negative_available_at_inference=False,
            training_complete_is_not_image_success=True, formal_images_generated=0))
        print('COMPLETE_FULL_RESPONSE_RANK_COMPONENT_TRAINING', flush=True)
    finally:
        collector.close()


if __name__ == '__main__':
    main()
