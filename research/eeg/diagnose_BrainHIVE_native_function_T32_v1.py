"""Frozen, T-only denoising and native attention-response diagnostic.

This probes DDPM-noised real latents at the four native inference time indices.
It is not a sampled four-step trajectory, an image-quality score, or a fit.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
NAME = 'native_function_T32_v1'
ARMS = ('raw_OOF_EEG', 'permuted_different_concept_OOF_EEG', 'correct_F',
        'wrong_different_concept_F', 'missing_zero_raw_condition')
REQUIRED_SOURCES = (
    'diagnose_BrainHIVE_native_function_T32_v1.py', 'brainhive_native_models_v1.py',
    'brainhive_component_data_v1.py', 'brainhive_component_generator_v1.py',
    'brainhive_response_rank_v1.py', 'train_BrainHIVE_response_rank_v1.py',
    'train_BrainHIVE_bounded_component_v1.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_fixed_queries(oracle_rows, allowed, manifest_by_row):
    assert isinstance(oracle_rows, list) and len(oracle_rows) == 32
    assert len(set(allowed)) == len(allowed)
    position = {row: index for index, row in enumerate(allowed)}
    assert len({q['stimulus_id'] for q in oracle_rows}) == 32
    assert len({q['array_row'] for q in oracle_rows}) == 32
    selected = []
    for index, query in enumerate(oracle_rows):
        assert set(query) == {'stimulus_id', 'array_row'}
        row = query['array_row']
        assert isinstance(row, int) and row in position, 'An oracle stimulus is outside allowed T'
        target = manifest_by_row[row]
        assert target['dataset_split'] == 'train'
        assert query['stimulus_id'] == target['stimulus_id'], 'Stimulus/array-row mismatch'
        selected.append(dict(query, oracle_query_index=index, T_position=position[row], concept_id=target['concept_id']))
    return selected


def noise_seed(stimulus_id, timestep):
    key = f'1729|{NAME}|sub-01|{stimulus_id}|R4|DDPM|{timestep}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**63 - 1)


def contract_test():
    allowed = [100 + 3 * i for i in range(40)]
    manifest = {r: dict(stimulus_id=f'image_{r}.jpg', concept_id=f'concept_{i}', dataset_split='train')
                for i, r in enumerate(allowed)}
    oracle = [dict(stimulus_id=manifest[r]['stimulus_id'], array_row=r) for r in allowed[4:36]]
    selected = select_fixed_queries(oracle, allowed, manifest)
    assert selected[0]['array_row'] == 112 and selected[0]['T_position'] == 4
    assert all(q['T_position'] != q['array_row'] for q in selected)
    assert noise_seed('image_112.jpg', 999) == noise_seed('image_112.jpg', 999)
    assert noise_seed('image_112.jpg', 999) != noise_seed('image_112.jpg', 749)
    assert noise_seed('image_112.jpg', 999) != noise_seed('image_115.jpg', 999)
    for change in ('stimulus', 'outside_T', 'duplicate'):
        broken = [dict(q) for q in oracle]
        if change == 'stimulus':
            broken[0]['stimulus_id'] = 'wrong.jpg'
        elif change == 'outside_T':
            broken[0]['array_row'] = 9988
        else:
            broken[1] = dict(broken[0])
        try:
            select_fixed_queries(broken, allowed, manifest)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid T identity mapping accepted')
    return dict(status='CPU_CONTRACT_CHECK_PASSED', array_rows_are_not_positions=True,
                fixed_query_pairing_verified=True, noise_namespace_deterministic=True,
                GPU_executed=False, generated_images=0, fitted_parameters=0)


def validate_registration(registration):
    fixed = dict(diagnostic=NAME, subject='sub-01', queries=32, seed=1729, batch_size=4,
                 output_directory=str(WORK / 'audit' / NAME),
                 oracle_inference_inputs=str(WORK / 'images/oracle_Tdiag32_v1/inference_inputs.json'),
                 latent_receipt=str(WORK / 'audit/component_T_latent_bank_v2.json'))
    for key, expected in fixed.items():
        assert registration[key] == expected, (key, registration.get(key), expected)
    for filename in REQUIRED_SOURCES:
        assert registration['source_sha256'][filename] == sha(ROOT / filename), filename
    mandatory = {registration['oracle_inference_inputs'], registration['latent_receipt'],
                 str(WORK / 'data/component_input_contract_v1.json'),
                 str(ROOT / 'configs/split_manifest.json'), str(ROOT / 'audit/image_manifest.csv')}
    assert mandatory <= set(registration['input_sha256'])
    for filename, expected in registration['input_sha256'].items():
        assert Path(filename).is_absolute() and sha(filename) == expected, filename


def mean(values):
    assert values
    return sum(values) / len(values)


def check_registered_input_contract(registration):
    """Read hashes and mapping metadata only; no tensor import or GPU access."""
    validate_registration(registration)
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    assert sha(ROOT / 'audit/image_manifest.csv') == protocol['image_manifest_sha256']
    allowed = protocol['image_rows']['T']
    assert len(allowed) == 13230
    assert set(allowed).isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    manifest = {}
    with (ROOT / 'audit/image_manifest.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['dataset_split'] == 'train':
                array_row = int(row['array_row'])
                assert array_row not in manifest
                manifest[array_row] = row
    oracle = json.loads(Path(registration['oracle_inference_inputs']).read_text())
    selected = select_fixed_queries(oracle, allowed, manifest)
    return dict(status='ACTUAL_REGISTERED_ORACLE_T32_INPUT_MAPPING_CPU_VERIFIED', queries=32,
        oracle_metadata_sha256=sha(registration['oracle_inference_inputs']),
        first_query=selected[0], last_query=selected[-1], tensor_values_loaded=False,
        GPU_executed=False, generated_images=0, fitted_parameters=0)


def frozen_response_diagnostic(collector, EEG_tokens, correct_tokens, wrong_tokens):
    """Per-query/layer measurements and the existing collector's exact auxiliary."""
    import torch
    import torch.nn.functional as F
    from brainhive_response_rank_v1 import ip_response
    metrics = {key: [] for key in ('d_positive', 'd_negative', 'teacher_separation',
        'normalized_margin', 'J', 'rank_loss', 'near_zero_teacher_fraction')}
    layers = []
    with torch.no_grad():
        for name, (attention, processor) in collector.layers.items():
            query = collector.queries[name]
            assert not query.requires_grad
            EEG = ip_response(query, EEG_tokens, processor, attention.heads).float()
            correct = ip_response(query, correct_tokens, processor, attention.heads).float()
            wrong = ip_response(query, wrong_tokens, processor, attention.heads).float()
            dimensions = tuple(range(1, EEG.ndim))
            d_positive = (EEG - correct).square().mean(dimensions)
            d_negative = (EEG - wrong).square().mean(dimensions)
            separation = (correct - wrong).square().mean(dimensions)
            margin = (d_positive - d_negative) / (separation + 1e-6)
            values = dict(d_positive=d_positive, d_negative=d_negative, teacher_separation=separation,
                normalized_margin=margin, J=-margin, rank_loss=F.softplus(margin),
                near_zero_teacher_fraction=(separation <= 1e-6).float())
            assert all(torch.isfinite(value).all() for value in values.values())
            for key, value in values.items():
                metrics[key].append(value)
            layers.append(dict(layer=name, query_positions=int(query.shape[2]),
                **{key: float(value.mean()) for key, value in values.items()}))
        per_query = {key: torch.stack(values).mean(0).cpu().tolist() for key, values in metrics.items()}
    # The existing training auxiliary requires a gradient-capable student token.
    # This temporary leaf is never backpropagated; all model weights stay frozen.
    with torch.enable_grad():
        leaf = EEG_tokens.detach().requires_grad_(True)
        rank, aggregate = collector.auxiliary(leaf, correct_tokens.detach(), wrong_tokens.detach())
        assert torch.isfinite(rank)
        auxiliary_loss = float(rank.detach())
        aggregate = {key: float(value) for key, value in aggregate.items()}
    assert math.isclose(auxiliary_loss, mean(per_query['rank_loss']), rel_tol=1e-4, abs_tol=3e-5)
    assert all(math.isclose(aggregate[key], mean(per_query[key]), rel_tol=1e-4, abs_tol=3e-5)
               for key in aggregate)
    return per_query, layers, dict(rank_loss=auxiliary_loss, **aggregate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--check-contract', action='store_true')
    parser.add_argument('--check-input-contract', action='store_true')
    args = parser.parse_args()
    if args.check_contract:
        print(json.dumps(contract_test()), flush=True)
        return
    assert args.registration
    registration = json.loads(args.registration.read_text())
    if args.check_input_contract:
        print(json.dumps(check_registered_input_contract(registration)), flush=True)
        return
    assert b'run_gpu_job_v2.py' in (Path('/proc') / str(os.getppid()) / 'cmdline').read_bytes()
    validate_registration(registration)
    out = Path(registration['output_directory'])
    assert not out.exists(), 'Preserve any earlier diagnostic, including incomplete outputs'
    import torch
    import numpy as np
    from diffusers import DDPMScheduler
    from brainhive_component_data_v1 import load_training
    from brainhive_component_generator_v1 import load_native, denoise
    from brainhive_response_rank_v1 import NativeIPResponseCapture
    from train_BrainHIVE_response_rank_v1 import negative_positions
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    assert sha(ROOT / 'audit/image_manifest.csv') == protocol['image_manifest_sha256']
    allowed = protocol['image_rows']['T']
    assert len(allowed) == 13230
    assert set(allowed).isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    image_rows = {}
    with (ROOT / 'audit/image_manifest.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['dataset_split'] == 'train':
                index = int(row['array_row'])
                assert index not in image_rows
                image_rows[index] = row
    oracle = json.loads(Path(registration['oracle_inference_inputs']).read_text())
    selected = select_fixed_queries(oracle, allowed, image_rows)
    inputs = load_training(device='cpu')
    assert inputs['array_rows'] == allowed and inputs['subjects'].count('sub-01') == 1
    subject_index = inputs['subjects'].index('sub-01')
    positions = torch.tensor([q['T_position'] for q in selected], dtype=torch.long)
    indices = positions + subject_index * 13230
    negative, negative_receipt = negative_positions(inputs)
    wrong_positions = negative[positions]
    for query, wrong_position in zip(selected, wrong_positions.tolist()):
        wrong_row = allowed[wrong_position]
        assert image_rows[wrong_row]['concept_id'] != query['concept_id']
        query.update(wrong_T_position=wrong_position, wrong_array_row=wrong_row,
            wrong_stimulus_id=image_rows[wrong_row]['stimulus_id'], wrong_concept_id=image_rows[wrong_row]['concept_id'])
    raw_conditions = inputs['condition'][indices].clone().cuda()
    permuted_conditions = inputs['condition'][wrong_positions + subject_index * 13230].clone().cuda()
    correct_conditions = inputs['target'][positions].clone().cuda()
    wrong_conditions = inputs['target'][wrong_positions].clone().cuda()
    assert raw_conditions.shape == permuted_conditions.shape == correct_conditions.shape == wrong_conditions.shape == (32, 1024)
    contract_path = WORK / 'data/component_input_contract_v1.json'
    contract = json.loads(contract_path.read_text())
    source_subject = next(s for s in contract['subjects'] if s['subject'] == 'sub-01')
    selected_sources = []
    coverage = {row: 0 for q in selected for row in (q['array_row'], q['wrong_array_row'])}
    for source in source_subject['sources']:
        if not source['role'].startswith('oof'):
            continue
        relevant = [r for r in source['groups']['component_train'] if r['array_row'] in coverage]
        if relevant:
            for row in relevant:
                coverage[row['array_row']] += 1
            selected_sources.append(dict(role=source['role'], checkpoint_path=source['checkpoint_path'],
                checkpoint_sha256=source['checkpoint_sha256'], condition_cache_path=source['condition_cache_path'],
                condition_cache_sha256=source['condition_cache_sha256'], selected_mapping=relevant))
    assert all(count == 1 for count in coverage.values())
    del inputs
    latent_receipt_path = Path(registration['latent_receipt'])
    latent_receipt = json.loads(latent_receipt_path.read_text())
    assert latent_receipt['status'] == 'COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
    assert latent_receipt['format'] == 'numpy_npy_float32_C_contiguous_v2'
    assert latent_receipt['array_rows'] == allowed
    assert sha(latent_receipt['path']) == latent_receipt['sha256']
    latent_array = np.load(latent_receipt['path'], mmap_mode='r')
    assert latent_array.shape == (13230, 4, 64, 64) and latent_array.dtype == np.float32
    real_latents = torch.tensor(latent_array[positions.numpy()], device='cuda')
    del latent_array
    pipe, adapter, projector, prompts, host = load_native(training=False)
    pipe.scheduler.set_timesteps(4, device='cuda')
    native_timesteps = pipe.scheduler.timesteps.detach().float()
    assert len(native_timesteps) == 4 and torch.equal(native_timesteps, native_timesteps.round())
    timesteps = native_timesteps.long().cpu().tolist()
    assert len(set(timesteps)) == 4 and all(0 <= t < 1000 for t in timesteps)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    assert scheduler.config.prediction_type == 'epsilon'
    frozen_modules = [pipe.unet, pipe.vae, adapter, projector]
    assert all(not module.training and all(not p.requires_grad for p in module.parameters()) for module in frozen_modules)
    with torch.no_grad():
        token_conditions = {ARMS[0]: projector(adapter(raw_conditions.half())),
            ARMS[1]: projector(adapter(permuted_conditions.half())),
            ARMS[2]: projector(adapter(correct_conditions.half())),
            ARMS[3]: projector(adapter(wrong_conditions.half())),
            ARMS[4]: projector(adapter(torch.zeros_like(raw_conditions).half()))}
    assert all(tokens.shape == (32, 4, 2048) and torch.isfinite(tokens).all() for tokens in token_conditions.values())
    collector = NativeIPResponseCapture(pipe.unet)
    forward_counts = dict(calls=0, samples=0)
    def forward_hook(module, arguments, result):
        forward_counts['calls'] += 1
        forward_counts['samples'] += len(arguments[0])
        assert torch.isfinite(result[0]).all()
    handle = pipe.unet.register_forward_hook(forward_hook)
    out.mkdir(parents=True)
    provenance = dict(registration_path=str(args.registration), registration_sha256=sha(args.registration),
        source_sha256=registration['source_sha256'], input_sha256=registration['input_sha256'],
        oracle_metadata_path=registration['oracle_inference_inputs'], oracle_metadata_sha256=sha(registration['oracle_inference_inputs']),
        input_contract_sha256=sha(contract_path), actual_sub01_OOF_sources=selected_sources,
        training_target_cache_path=contract['target_cache_path'], training_target_cache_sha256=contract['target_cache_sha256'],
        latent_receipt_path=str(latent_receipt_path), latent_receipt_sha256=sha(latent_receipt_path),
        latent_path=latent_receipt['path'], latent_sha256=latent_receipt['sha256'],
        native_asset_verification_sha256=sha(WORK / 'audit/native_asset_verification.json'),
        host=host, negative_mapping=negative_receipt, native_scheduler_class=type(pipe.scheduler).__name__,
        native_scheduler_timesteps=timesteps, diagnostic_noise_scheduler='DDPMScheduler',
        noise_seed_rule=f'SHA256 first16 hex of 1729|{NAME}|sub-01|stimulus_id|R4|DDPM|timestep, modulo2**63-1',
        all_conditions_share_same_real_latent_noise_and_timestep=True,
        Q_reference='Exactly one missing_zero_raw_condition full-UNet pass per query/time; its detached native Q reused for every response arm. Fixed before data, no response-specific Q.',
        functional_J='(Dnegative-Dpositive)/(positive_negative_teacher_separation+1e-6); compare correct EEG J with permuted EEG and missing-condition J at the same Q',
        missing_definition='Exactly zeros_like(raw_OOF_EEG) at input to frozen A; not BrainMLP(zero EEG) and not zero IP tokens',
        generated_images=0, optimizer_created=False, parameter_updates=0, all_native_weights_frozen=True,
        protected_D_V_or_test_tensors_or_scores_loaded=False, split_and_contract_metadata_read_for_T_checks=True,
        diagnostic_scope='T-only frozen denoising training-forward probes at native four time indices; not an actual four-step sampling trajectory or image-quality evaluation')
    write(out / 'provenance.json', provenance)
    write(out / 'fixed_queries.json', selected)
    rows = []
    layer_accumulators = {}
    try:
        with (out / 'per_query_timestep.jsonl').open('x', buffering=1) as log:
            for start in range(0, 32, 4):
                stop = start + 4
                for time_index, timestep in enumerate(timesteps):
                    seeds = [noise_seed(q['stimulus_id'], timestep) for q in selected[start:stop]]
                    noises = [torch.randn((1, 4, 64, 64), device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(seed)) for seed in seeds]
                    noise = torch.cat(noises)
                    time_tensor = torch.full((4,), timestep, device='cuda', dtype=torch.long)
                    noisy = scheduler.add_noise(real_latents[start:stop], noise, time_tensor)
                    epsilon_losses = {}
                    with torch.no_grad():
                        for arm in (ARMS[4],) + ARMS[:4]:
                            if arm == ARMS[4]:
                                with collector.capture():
                                    prediction = denoise(pipe, prompts, noisy, time_tensor, token_conditions[arm][start:stop])
                            else:
                                prediction = denoise(pipe, prompts, noisy, time_tensor, token_conditions[arm][start:stop])
                            losses = (prediction.float() - noise).square().flatten(1).mean(1)
                            assert torch.isfinite(losses).all()
                            epsilon_losses[arm] = losses.cpu().tolist()
                    reference_queries = dict(collector.queries)
                    assert len(reference_queries) == 70
                    functions, aggregates = {}, {}
                    for arm in ARMS:
                        assert not collector.active
                        collector.queries = dict(reference_queries)
                        function, layer_values, aggregate = frozen_response_diagnostic(collector,
                            token_conditions[arm][start:stop], token_conditions[ARMS[2]][start:stop],
                            token_conditions[ARMS[3]][start:stop])
                        functions[arm], aggregates[arm] = function, aggregate
                        assert len(layer_values) == 70 and not collector.queries
                        for layer in layer_values:
                            key = (layer['layer'], timestep, arm)
                            accumulator = layer_accumulators.setdefault(key, dict(layer=layer['layer'], timestep=timestep,
                                arm=arm, queries=0, query_positions=layer['query_positions'], sums={k: 0. for k in function}))
                            accumulator['queries'] += 4
                            for metric in function:
                                accumulator['sums'][metric] += 4 * layer[metric]
                    reference_queries.clear()
                    for local, query in enumerate(selected[start:stop]):
                        item = dict(query, subject='sub-01', repeat_protocol='R4', timestep=timestep,
                            native_time_index=time_index, noise_seed=seeds[local],
                            epsilon_MSE={arm: epsilon_losses[arm][local] for arm in ARMS},
                            EEG_epsilon_MSE_minus_correct_F=epsilon_losses[ARMS[0]][local] - epsilon_losses[ARMS[2]][local],
                            permuted_EEG_epsilon_MSE_minus_EEG=epsilon_losses[ARMS[1]][local] - epsilon_losses[ARMS[0]][local],
                            wrong_F_epsilon_MSE_minus_correct_F=epsilon_losses[ARMS[3]][local] - epsilon_losses[ARMS[2]][local],
                            missing_epsilon_MSE_minus_EEG=epsilon_losses[ARMS[4]][local] - epsilon_losses[ARMS[0]][local],
                            native_IP_response_by_arm={arm: {key: values[local] for key, values in function.items()}
                                for arm, function in functions.items()},
                            EEG_J_minus_permuted_EEG_J=functions[ARMS[0]]['J'][local] - functions[ARMS[1]]['J'][local],
                            EEG_J_minus_missing_J=functions[ARMS[0]]['J'][local] - functions[ARMS[4]]['J'][local],
                            noise_sha256=hashlib.sha256(noises[local].cpu().contiguous().numpy().tobytes()).hexdigest(),
                            noised_real_T_latent_sha256=hashlib.sha256(noisy[local].cpu().contiguous().numpy().tobytes()).hexdigest(),
                            conditional_full_UNet_forwards=5, captured_IP_layers=70,
                            Q_reference_arm=ARMS[4], no_generated_image=True)
                        rows.append(item)
                        log.write(json.dumps(item, allow_nan=False) + '\n')
                    print(json.dumps(dict(stage='FROZEN_NATIVE_T32_FUNCTIONAL_DIAGNOSTIC',
                        queries_finished_through=stop, timestep=timestep, rows_saved=len(rows), expected_rows=128,
                        functional_rank_by_arm=aggregates, seconds=time.monotonic()-started)), flush=True)
        assert len(rows) == len({(r['array_row'], r['timestep']) for r in rows}) == 128
        assert forward_counts == dict(calls=160, samples=640)
        assert all(p.grad is None and not p.requires_grad for module in frozen_modules for p in module.parameters())
        layer_rows = []
        for key in sorted(layer_accumulators):
            value = layer_accumulators[key]
            assert value['queries'] == 32
            layer_rows.append({k: v for k, v in value.items() if k != 'sums'} |
                {k: v / value['queries'] for k, v in value['sums'].items()})
        write(out / 'per_layer_timestep.json', layer_rows)
        summaries = []
        for timestep in timesteps:
            subset = [r for r in rows if r['timestep'] == timestep]
            summaries.append(dict(timestep=timestep, queries=32,
                mean_epsilon_MSE={arm: mean([r['epsilon_MSE'][arm] for r in subset]) for arm in ARMS},
                mean_EEG_epsilon_MSE_minus_correct_F=mean([r['EEG_epsilon_MSE_minus_correct_F'] for r in subset]),
                mean_permuted_EEG_epsilon_MSE_minus_EEG=mean([r['permuted_EEG_epsilon_MSE_minus_EEG'] for r in subset]),
                mean_wrong_F_epsilon_MSE_minus_correct_F=mean([r['wrong_F_epsilon_MSE_minus_correct_F'] for r in subset]),
                mean_missing_epsilon_MSE_minus_EEG=mean([r['missing_epsilon_MSE_minus_EEG'] for r in subset]),
                fraction_correct_F_lower_epsilon_MSE_than_EEG=mean([float(r['EEG_epsilon_MSE_minus_correct_F'] > 0) for r in subset]),
                fraction_EEG_IP_response_closer_to_correct_than_wrong=mean([float(r['native_IP_response_by_arm'][ARMS[0]]['J'] > 0) for r in subset]),
                mean_EEG_J_minus_permuted_EEG_J=mean([r['EEG_J_minus_permuted_EEG_J'] for r in subset]),
                mean_EEG_J_minus_missing_J=mean([r['EEG_J_minus_missing_J'] for r in subset]),
                mean_native_IP_response_by_arm={arm: {key: mean([r['native_IP_response_by_arm'][arm][key] for r in subset])
                    for key in rows[0]['native_IP_response_by_arm'][arm]} for arm in ARMS}))
        write(out / 'summary.json', dict(status='FROZEN_T_ONLY_FUNCTIONAL_MEASUREMENTS_SAVED', by_timestep=summaries,
            interpretation='Positive EEG-minus-correct-F epsilon margin favors correct F; positive permuted/missing-minus-EEG epsilon margins favor correct EEG. Positive J means a response is closer to correct F; correct-EEG minus permuted/missing J deltas measure stimulus-specific functional separation at shared missing-reference Q. Neither measure is image quality or proof of a useful trainable component.',
            teacher_advantage_required_for_engineering_success=False, final_test_inference_performed=False))
        write(out / 'completion.json', dict(status='COMPLETE_FROZEN_NATIVE_T32_FUNCTIONAL_DIAGNOSTIC_NO_IMAGE_QUALITY_CLAIM',
            registration_sha256=sha(args.registration), provenance_sha256=sha(out / 'provenance.json'),
            fixed_queries_sha256=sha(out / 'fixed_queries.json'), per_query_timestep_sha256=sha(out / 'per_query_timestep.jsonl'),
            per_layer_timestep_sha256=sha(out / 'per_layer_timestep.json'), summary_sha256=sha(out / 'summary.json'),
            real_T_queries=32, subject='sub-01', timesteps=timesteps, rows=128, actual_full_UNet_forwards=forward_counts,
            native_IP_layers=70, generated_images=0, fitted_parameters=0, protected_scores_read=False,
            engineering_checks_passed=True, component_effectiveness_established=False,
            native_defect_mechanism_established=False, seconds=time.monotonic()-started))
    except BaseException as error:
        write(out / 'failure.json', dict(status='FAILED_PARTIAL_DIAGNOSTIC_PRESERVED',
            error_type=type(error).__name__, error=str(error)[:1500], rows_saved=len(rows),
            generated_images=0, fitted_parameters=0, seconds=time.monotonic()-started))
        raise
    finally:
        collector.close()
        handle.remove()


if __name__ == '__main__':
    main()
