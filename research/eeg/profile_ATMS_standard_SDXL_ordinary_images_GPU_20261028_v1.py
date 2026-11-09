"""T-only engineering replay of native/zero-head/disabled ordinary with CFG5.

All twenty real subject/fold affine groups are checked.  Two fixed T queries,
one per OOF role, each produce the same three 512/20Euler/CFG5/IP1 images.
No training, calibration, Dfit/Dcheck values, effect selection or new method.
Use the existing finite GPU wrapper and a spec pinning this entry's source.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import train_ATMS_standard_SDXL_ordinary_20261028_v1 as T

torch = np = None
TRAINER_SHA = '01df320704f0a4d2f3f8ac56f58868717639ec11abaee582c331c2b190cc6f00'
CASES = ('native', 'zero_head', 'disabled')


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def pixels_sha(value):
    assert value.dtype == np.uint8 and value.shape == (512, 512, 3)
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def all_affine_replay(data, host, model, budget):
    positions = [int(np.flatnonzero(data['folds'] == fold)[0]) for fold in (0, 1)]
    indices = np.array([si * T.N_IMAGES + position
        for si in range(10) for position in positions], dtype=np.int64)
    condition = torch.from_numpy(np.array(data['conditions'][indices], copy=True)).cuda()
    weight = data['weight'].cuda(); bias = data['bias'].cuda()
    assert condition.shape == weight.shape == bias.shape == (20, 1024)
    assert torch.isfinite(condition).all()
    host['pipe'].unet.eval(); model.eval()
    with torch.no_grad():
        zero, _ = model(condition, weight, bias, gate_mode='fixed')
        disabled, _ = model(condition, weight, bias, enabled=False)
        assert torch.equal(condition, zero) and torch.equal(condition, disabled)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            native_tokens = T.project_tokens(host, condition)
            zero_tokens = T.project_tokens(host, zero)
            disabled_tokens = T.project_tokens(host, disabled)
        assert torch.equal(native_tokens, zero_tokens) and torch.equal(native_tokens, disabled_tokens)
    records = []
    for index, (subject, fold) in enumerate((subject, fold)
            for subject in data['split']['subjects'] for fold in (0, 1)):
        position = positions[fold]; row = data['T_rows'][position]
        assert data['folds'][position] == fold
        records.append(dict(subject=subject, role='oof' + str(fold), T_position=position,
            condition_row=int(indices[index]), array_row=row,
            stimulus_id=data['metadata'][row]['stimulus_id'],
            condition_sha256=tensor_sha(condition[index]),
            zero_head_condition_sha256=tensor_sha(zero[index]),
            disabled_condition_sha256=tensor_sha(disabled[index]),
            native_tokens_sha256=tensor_sha(native_tokens[index]),
            zero_head_tokens_sha256=tensor_sha(zero_tokens[index]),
            disabled_tokens_sha256=tensor_sha(disabled_tokens[index]),
            affine_weight_sha256=tensor_sha(weight[index]), affine_bias_sha256=tensor_sha(bias[index]),
            condition_exact=True, native_P_tokens_exact=True))
    budget.check('ALL20_ACTUAL_AFFINE_GROUPS_NATIVE_ZERO_DISABLED_EXACT')
    # Fixed first T position of each fold; not selected by images or scores.
    queries = [dict(subject='sub-01', subject_index=0, role='oof' + str(fold),
        T_position=positions[fold], array_row=data['T_rows'][positions[fold]],
        condition_row=positions[fold]) for fold in (0, 1)]
    return records, queries


def one_image(host, condition, seed, folder, query, case, budget):
    from diffusers.utils.torch_utils import randn_tensor
    from PIL import Image
    pipe = host['pipe']; prompt, negative, pool, negative_pool = host['prompts']
    deployed = condition.to(device='cuda', dtype=torch.bfloat16)
    assert deployed.shape == (1, 1024) and torch.isfinite(deployed).all()
    prepared = torch.cat([torch.zeros_like(deployed), deployed])[:, None, :]
    generator = torch.Generator(device='cuda').manual_seed(seed)
    noise = randn_tensor((1, 4, 64, 64), generator=[generator], device='cuda', dtype=torch.bfloat16)
    original_noise_sha = tensor_sha(noise)
    before = dict(host['engineering_calls']); projection_inputs = []; projection_tokens = []
    expected_flattened = prepared.reshape(2, 1024)
    def audit_unet(_module, args, kwargs):
        assert len(args[0]) == 2, 'Exactly one negative and one positive CFG query'
        supplied = kwargs['added_cond_kwargs']['image_embeds']
        assert len(supplied) == 1 and torch.equal(supplied[0], prepared)
        host['engineering_calls']['UNet_forward_calls'] += 1
        host['engineering_calls']['UNet_sample_evaluations'] += len(args[0])
    def audit_projection_input(_module, args):
        actual = args[0]
        assert actual.shape == (2, 1024) and torch.equal(actual, expected_flattened), (
            'Actual native P must receive zero negative then correct positive query')
        projection_inputs.append(tensor_sha(actual))
        host['engineering_calls']['native_P_forward_calls'] += 1
    def audit_projection_output(_module, _args, result):
        assert result.shape == (2, 4, 2048)
        projection_tokens.append(tensor_sha(result))
    handles = [pipe.unet.register_forward_pre_hook(audit_unet, with_kwargs=True),
        host['P'].register_forward_pre_hook(audit_projection_input),
        host['P'].register_forward_hook(audit_projection_output)]
    torch.cuda.synchronize(); started = time.monotonic()
    try:
        with torch.inference_mode():
            latent = pipe(prompt_embeds=prompt.clone(), negative_prompt_embeds=negative.clone(),
                pooled_prompt_embeds=pool.clone(), negative_pooled_prompt_embeds=negative_pool.clone(),
                ip_adapter_image_embeds=[prepared], height=512, width=512,
                num_inference_steps=20, guidance_scale=5., num_images_per_prompt=1,
                latents=noise.clone(), output_type='latent').images
            assert latent.shape == (1, 4, 64, 64) and torch.isfinite(latent).all()
            decoded = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor, return_dict=False)[0]
            assert torch.isfinite(decoded).all()
            image = pipe.image_processor.postprocess(decoded, output_type='pil')[0]
            host['engineering_calls']['VAE_decode_calls'] += 1
            host['engineering_calls']['VAE_decode_images'] += 1
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    assert tensor_sha(noise) == original_noise_sha, 'Caller initial noise is preserved'
    calls = {key: value - before[key] for key, value in host['engineering_calls'].items()}
    assert calls == dict(UNet_forward_calls=20, UNet_sample_evaluations=40,
        native_P_forward_calls=20, VAE_decode_calls=1, VAE_decode_images=1), calls
    assert len(projection_inputs) == len(projection_tokens) == 20
    assert len(set(projection_inputs)) == len(set(projection_tokens)) == 1
    pixels = np.array(image.convert('RGB'), copy=True)
    image_path = folder / (query['role'] + '_' + case + '.png')
    assert not image_path.exists(); image.save(image_path, format='PNG')
    with Image.open(image_path) as saved:
        decoded_pixels = np.array(saved.convert('RGB'), copy=True)
    assert np.array_equal(pixels, decoded_pixels), 'PNG must retain all generated pixels'
    budget.check('ENGINEERING_IMAGE_' + query['role'] + '_' + case)
    record = dict(query=query, case=case, path=str(image_path), bytes=image_path.stat().st_size,
        file_sha256=T.sha(image_path), pixel_sha256=pixels_sha(pixels),
        pixel_dimensions=[512, 512, 3], initial_noise_seed=seed,
        initial_noise_sha256=original_noise_sha, raw_condition_sha256=tensor_sha(condition),
        deployed_condition_sha256=tensor_sha(deployed), CFG_packed_condition_sha256=tensor_sha(prepared),
        actual_P_input_sha256=projection_inputs[0], actual_P_token_sha256=projection_tokens[0],
        final_latent_sha256=tensor_sha(latent), actual_calls=calls,
        seconds=time.monotonic() - started, steps=20, CFG=5., IP_scale=1., scheduler='EulerDiscreteScheduler',
        native_negative_branch_is_zero=True, actual_negative_then_positive_P_input_verified=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    assert 120 <= args.max_seconds <= 900
    assert T.sha(T.__file__) == TRAINER_SHA, 'Reviewed trainer must remain unchanged'
    wrapper = T.wrapper_proof(); spec = T.load_spec(args.spec)
    pins = {str(Path(item['path']).resolve()): item['sha256'] for item in spec['source_pins']}
    assert pins.get(str(Path(__file__).resolve())) == T.sha(__file__), 'Engineering entry must be source-pinned'
    global torch, np
    from profile_standard_generator import torch as _torch
    import torch.nn.functional as functional
    import numpy as _numpy
    torch, np = _torch, _numpy
    T.torch, T.F = torch, functional
    torch.set_num_threads(2); torch.cuda.set_device(0); assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True); torch.cuda.reset_peak_memory_stats()
    budget = T.Resources(args.max_seconds, spec)
    assert not args.output.exists(), 'Use a new output for a failed engineering attempt; preserve all six prefixes'
    args.output.mkdir(parents=True)
    records = []; active = 'INPUTS'
    try:
        data = T.validate_inputs(spec); host = T.setup_host(spec, budget)
        pipe = host['pipe']; pipe.set_progress_bar_config(disable=True)
        assert type(pipe.scheduler).__name__ == 'EulerDiscreteScheduler'
        assert all(not parameter.requires_grad for component in (
            pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2) for parameter in component.parameters())
        model = T.new_adapter().eval()
        affine_records, queries = all_affine_replay(data, host, model, budget)
        T.atomic_json(args.output / 'all20_affine_replay.json', dict(
            status='ALL20_TRUE_ATMS_AFFINE_GROUPS_NATIVE_ZERO_DISABLED_EXACT', groups=affine_records))
        host['engineering_calls'] = dict(UNet_forward_calls=0, UNet_sample_evaluations=0,
            native_P_forward_calls=0, VAE_decode_calls=0, VAE_decode_images=0)
        for query in queries:
            assert not budget.stopping(), 'Finite image-profile time protection'
            position = query['T_position']; row = query['array_row']; fold = int(query['role'][-1])
            assert row in data['T_rows'] and data['folds'][position] == fold
            assert data['metadata'][row]['concept_id'] in set(data['split']['train_concepts']['T'])
            query['stimulus_id'] = data['metadata'][row]['stimulus_id']
            c = torch.from_numpy(np.array(data['conditions'][query['condition_row']:query['condition_row'] + 1], copy=True)).cuda()
            weight = data['weight'][fold:fold + 1].cuda(); bias = data['bias'][fold:fold + 1].cuda()
            with torch.no_grad():
                zero = model(c, weight, bias, gate_mode='fixed')[0]
                off = model(c, weight, bias, enabled=False)[0]
            assert torch.equal(c, zero) and torch.equal(c, off)
            seed = T.seed('engineering_CFG5_image|' + query['subject'] + '|' + str(row))
            query_records = []
            for case, condition in zip(CASES, (c, zero, off)):
                assert not budget.stopping(), 'Finite image-profile time protection'
                active = query['role'] + '_' + case
                record = one_image(host, condition, seed, args.output, query, case, budget)
                records.append(record); query_records.append(record)
                T.append_event(args.output / 'images.jsonl', record)
            for key in ('initial_noise_sha256', 'pixel_sha256', 'final_latent_sha256',
                        'raw_condition_sha256', 'deployed_condition_sha256',
                        'actual_P_input_sha256', 'actual_P_token_sha256'):
                assert len({record[key] for record in query_records}) == 1, (query, key)
        active = 'FINAL_FROZEN_STATE_AND_SOURCE_RECHECK'
        assert T.host_digests(pipe) == host['original'], 'Frozen native host tensor bytes changed'
        T.source_reverify(spec, args.output); budget.check('ENGINEERING_SIX_IMAGES_CLOSED')
        assert len(records) == 6
        done = dict(status='ATMS_STANDARD_ORDINARY_ALL20_AFFINES_AND_SIX_CFG5_ENGINEERING_IMAGES_PASSED',
            spec=T.reference(args.spec), source=T.reference(__file__), trainer=T.reference(T.__file__),
            wrapper=wrapper, provenance=T.provenance(args.spec, spec, data, host),
            affine_groups=20, queries=queries, images=records, actual_calls=host['engineering_calls'],
            all20_affine_replay_additional_calls=dict(native_P_forward_calls=3, native_P_sample_evaluations=60,
                UNet_forward_calls=0, VAE_decode_calls=0),
            native_zero_head_disabled_pixels_exact=True, same_query_initial_noise_exact=True,
            actual_CFG_negative_zero_positive_correct_P_inputs_verified=True,
            all_frozen_host_tensor_digests_unchanged=True, T_only=True,
            Dfit_Dcheck_tensor_values_selected=0, V_or_official_test_images_read=0,
            original_query_images_opened=0, engineering_images=6, research_effect_images=0,
            engineering_updates=0, scientific_updates=0, scientific_effect_established=False,
            actual_GPU_resume_proof_performed=False, **budget.result())
        T.atomic_json(args.output / 'completion.json', done)
        print(json.dumps(dict(status=done['status'], completion=T.reference(args.output / 'completion.json'))), flush=True)
    except BaseException as exc:
        T.atomic_json(args.output / ('failure_' + str(time.time_ns()) + '.json'), dict(
            status='ATMS_STANDARD_ORDINARY_ENGINEERING_IMAGES_FAILED_PREFIX_PRESERVED',
            error=repr(exc), traceback=traceback.format_exc(), active=active,
            actual_images_saved=len(records), source=T.reference(__file__), wrapper=wrapper,
            automatic_retry=False, scientific_updates=0, **budget.result()))
        raise


if __name__ == '__main__':
    main()
