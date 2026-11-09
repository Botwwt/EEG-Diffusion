"""Actual full-V O/P/PN/S images, native HIVE chain, strict own-arm resume.

The generator consumes only cached EEG conditions. Target metadata/files are
joined AFTER all 16600 inference records exist. No teacher, reference-Q pass,
retrieval gallery, negative or image-score selection enters sampling.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from brainhive_functional_training_contract_v2 import (
    ROOT, WORK, ARMS, core_sha, sha, read, validate_recipe, verify_admission,
    private_path, run_name, query_identity, verified_manifest_prefix,
)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    assert not temporary.exists(), 'Preserve orphan metadata for explicit repair'
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def same_or_create(path, text):
    path = Path(path)
    if path.exists():
        assert path.read_text() == text, 'Never overwrite a differing saved join'
    else:
        with path.open('x') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())


def generation_contract(recipe, *, check_files=True):
    validate_recipe(recipe, check_files=check_files)
    filename = Path(__file__).name
    assert filename in recipe['source_sha256']
    if check_files:
        assert sha(__file__) == recipe['source_sha256'][filename]
    assert isinstance(recipe['generation_job_names'], list) and 1 <= len(recipe['generation_job_names']) <= 2
    assert len(set(recipe['generation_job_names'])) == len(recipe['generation_job_names'])
    assert all(Path(name).name == name for name in recipe['generation_job_names'])
    assert 90 < recipe['generation_job_seconds'] <= 86400
    assert recipe['generation_job_seconds'] <= recipe['maximum_cumulative_generation_seconds'] <= 172800
    assert recipe['generation_gallery_manifest'] == str(WORK / 'gallery/V/manifest.jsonl')
    assert recipe['generation_gallery_manifest'] in recipe['input_sha256']
    return core_sha(recipe)


def exclusive_generation_wrapper(recipe):
    assert str(ROOT) == '/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1'
    parent = Path('/proc') / str(os.getppid())
    command = parent.joinpath('cmdline').read_bytes().split(b'\0')
    assert str(ROOT / 'run_gpu_job_v2.py').encode() in command and b'--name' in command
    job_name = command[command.index(b'--name') + 1].decode()
    assert job_name in recipe['generation_job_names']
    lock_path = ROOT / 'audit/GPU_JOB.lock'
    details = lock_path.stat()
    same = False
    for fd in parent.joinpath('fd').iterdir():
        try:
            item = fd.stat()
            same |= (item.st_dev, item.st_ino) == (details.st_dev, details.st_ino)
        except FileNotFoundError:
            pass
    assert same
    import fcntl
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise AssertionError('Generation wrapper does not hold the single-GPU lock')
    job = read(ROOT / 'audit' / (job_name + '_job.json'))
    assert job['status'] == 'RUNNING' and job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid()
    assert 0 < job['max_seconds'] <= recipe['generation_job_seconds']
    assert job['authorization_sha256'] == recipe['input_sha256'][str(ROOT / 'configs/resource_authorization_20261020_v3.json')]
    listed = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10)
    assert {int(value.strip()) for value in listed.stdout.splitlines() if value.strip()} <= {os.getpid()}
    capacity = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             check=True, capture_output=True, text=True, timeout=10)
    free = [int(value.strip()) * (1 << 20) for value in capacity.stdout.splitlines() if value.strip()]
    assert len(free) == 1 and free[0] >= read(recipe['admission_receipt'])['resource_limits']['minimum_free_GPU_bytes']
    return job_name


def own_storage_entry(recipe):
    receipt = read(recipe['admission_receipt'])
    assert set(receipt['scope']) >= {'training', 'actual_images', 'per_query_evaluation'}
    binding = receipt['four_arm_lifecycle_plan']
    assert sha(binding['path']) == binding['sha256']
    plan = read(binding['path'])
    entry = next(item for item in plan['arms'] if item['arm'] == recipe['arm'])
    assert entry['candidate'] == recipe['candidate'] and entry['recipe_core_sha256'] == core_sha(recipe)
    assert entry['run_name'] == run_name(recipe['arm'])
    assert entry['metadata_root'] == str(WORK / 'images' / run_name(recipe['arm']))
    assert entry['inference_manifest'] == str(WORK / 'images' / run_name(recipe['arm']) / 'inference_manifest.jsonl')
    assert entry['lifecycle_source_sha256'][Path(__file__).name] == sha(__file__)
    return entry


def validate_blind_inputs(inputs, protocol):
    assert inputs['expected_queries'] == 16600 and len(inputs['subjects']) == len(set(inputs['subjects'])) == 10
    assert inputs['repeat_protocol'] == 'R4' and not inputs['target_image_or_class_fields_present']
    queries = inputs['inputs']
    assert len(queries) == len({row['stimulus_id'] for row in queries}) == 1660
    assert len({row['array_row'] for row in queries}) == 1660
    assert len({Path(row['stimulus_id']).stem for row in queries}) == 1660
    assert all(set(row) == {'stimulus_id', 'array_row'} for row in queries)
    assert {row['array_row'] for row in queries} == set(protocol['image_rows']['V'])
    assert all(Path(row['stimulus_id']).name == row['stimulus_id'] for row in queries)


def row_identity(row, subject, query, index, recipe, version):
    identity, image_seed = query_identity(subject, query, version)
    assert row['query_id'] == identity and row['image_seed'] == image_seed
    assert row['subject'] == subject and row['global_query_index'] == index
    assert row['stimulus_id'] == row['EEG_source_stimulus_id'] == query['stimulus_id']
    assert row['array_row'] == query['array_row'] and row['repeat_protocol'] == 'R4'
    assert row['candidate'] == recipe['candidate'] and row['arm'] == recipe['arm']
    assert row['recipe_core_sha256'] == core_sha(recipe) and row['run_name'] == run_name(recipe['arm'])
    assert not row['target_or_visual_features_read_for_condition'] and not row['oracle_diagnostic_only']


def parent_bindings(recipe, inputs):
    contract_path = Path(str(WORK / 'data/component_input_contract_v1.json'))
    assert sha(contract_path) == recipe['input_contract_sha256']
    contract = read(contract_path)
    assert contract['status'] == 'ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    by_subject = {item['subject']: item for item in contract['subjects']}
    assert set(by_subject) == set(inputs['subjects']) and len(by_subject) == 10
    protocol = read(ROOT / 'configs/split_manifest.json')
    result = {}
    for subject in inputs['subjects']:
        source = next(item for item in by_subject[subject]['sources'] if item['role'] == 'full')
        assert sha(source['condition_cache_path']) == source['condition_cache_sha256']
        assert sha(source['checkpoint_path']) == source['checkpoint_sha256']
        assert sha(source['parent_registration_path']) == source['parent_registration_sha256']
        registration = read(source['parent_registration_path'])
        assert set(registration['training_array_rows']) == set(protocol['image_rows']['T'])
        assert set(registration['training_array_rows']).isdisjoint(protocol['image_rows']['V'])
        result[subject] = {key: source[key] for key in ('condition_cache_path', 'condition_cache_sha256',
            'checkpoint_path', 'checkpoint_sha256', 'parent_registration_path', 'parent_registration_sha256')}
    return result


def attach_targets_after_all_inference(recipe, entry, inputs, inference_rows, inference_completion):
    """CPU-only identity/file join, called only after the entire arm is saved."""
    assert len(inference_rows) == 16600 and inference_completion['queries'] == 16600
    assert inference_completion['normal_inference_target_access'] is False
    out = Path(entry['metadata_root'])
    assert sha(entry['inference_manifest']) == inference_completion['inference_manifest_sha256']
    targets_path = Path(str(WORK / 'data/generation_inputs/V_targets.jsonl'))
    assert sha(targets_path) == recipe['input_sha256'][str(targets_path)]
    target_rows = [json.loads(line) for line in targets_path.read_text().splitlines()]
    targets = {row['stimulus_id']: row for row in target_rows}
    gallery_path = Path(recipe['generation_gallery_manifest'])
    assert sha(gallery_path) == recipe['input_sha256'][str(gallery_path)]
    gallery_rows = [json.loads(line) for line in gallery_path.read_text().splitlines()]
    gallery = {row['stimulus_id']: row for row in gallery_rows}
    assert len(targets) == len(target_rows) == len(gallery) == len(gallery_rows) == 1660
    assert set(targets) == set(gallery) == {row['stimulus_id'] for row in inputs['inputs']}
    for stimulus, target in targets.items():
        actual = gallery[stimulus]
        assert target['image_sha256'] == actual['image_sha256'] and target['concept_id'] == actual['concept_id']
        assert sha(actual['image_path']) == target['image_sha256']
    joined = []
    for row in inference_rows:
        target = targets[row['stimulus_id']]
        assert int(target['array_row']) == row['array_row']
        image_path = gallery[row['stimulus_id']]['image_path']
        joined.append(row | dict(concept_id=target['concept_id'], target_image_sha256=target['image_sha256'],
            target_image_path=image_path, score_target_path=image_path, target_image_zip=target['image_zip'],
            target_image_member=target['image_member'], target_mapping_attached_after_inference=True))
    same_or_create(out / 'manifest.jsonl', ''.join(json.dumps(row, allow_nan=False) + '\n' for row in joined))
    hashes = {}
    for row in inference_rows:
        hashes.setdefault(row['generated_sha256'], []).append(row['query_id'])
    proof = dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_TARGET_AND_QUERY_IDENTITIES_JOINED',
        candidate=recipe['candidate'], arm=recipe['arm'], queries=16600,
        manifest_sha256=sha(out / 'manifest.jsonl'), gallery_manifest=str(gallery_path), gallery_manifest_sha256=sha(gallery_path),
        target_metadata_sha256=sha(targets_path),
        target_metadata_semantically_parsed_and_joined_only_after_all_inference=True,
        target_metadata_byte_hash_checked_before_inference=True,
        duplicate_image_SHA_groups=[values for values in hashes.values() if len(values) > 1],
        image_paths_unique=len({row['generated_path'] for row in inference_rows}) == 16600,
        raw_event_or_EEG_encoder_forward_replayed=False, generated_new_images_during_audit=False,
        scientific_image_gain_established=False)
    path = out / 'pairing_audit.json'
    if path.exists():
        assert read(path) == proof
    else:
        write(path, proof)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    args = parser.parse_args()
    recipe = read(args.recipe)
    generation_contract(recipe)
    verify_admission(recipe, 'formal')
    job_name = exclusive_generation_wrapper(recipe)
    entry = own_storage_entry(recipe)
    inputs = read(str(WORK / 'data/generation_inputs/V.json'))
    protocol = read(ROOT / 'configs/split_manifest.json')
    validate_blind_inputs(inputs, protocol)
    parents = parent_bindings(recipe, inputs)
    out = Path(entry['metadata_root'])
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'completion.json').exists(), 'Completed arms are immutable'
    saved = verified_manifest_prefix(entry, inputs, protocol['version'])
    import torch
    from PIL import Image
    import numpy as np
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = Path(recipe['output_directory'])
    trained = read(folder / 'completion.json')
    assert trained['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE'
    assert trained['candidate'] == recipe['candidate'] and trained['arm'] == recipe['arm']
    assert trained['recipe_core_sha256'] == core_sha(recipe) and trained['updates'] == 24807
    assert trained['seen_queries'] == 396900 and trained['passes'] == 3 and trained['subjects'] == 10
    assert trained['parameters'] == 525568 and trained['explicit_UNet_forwards_per_update'] == 2
    checkpoint = Path(trained['checkpoint_path'])
    assert checkpoint == folder / 'component_final.pt' and sha(checkpoint) == trained['checkpoint_sha256']
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert state['candidate'] == recipe['candidate'] and state['arm'] == recipe['arm']
    assert state['recipe_core_sha256'] == core_sha(recipe) and state['seed'] == 1729 and not state['risk_gate']
    assert state['updates'] == 24807 and state['seen_queries'] == 396900 and state['passes'] == 3
    assert sha(folder / 'input_provenance.json') == state['input_provenance_sha256'] == trained['input_provenance_sha256']
    provenance = read(folder / 'input_provenance.json')
    assert provenance['source_sha256'] == recipe['source_sha256'] and provenance['input_contract_sha256'] == recipe['input_contract_sha256']
    calibration = state['calibration']
    assert read(folder / 'calibration.json') == calibration and sha(folder / 'calibration.json') == trained['calibration_sha256']
    assert calibration['kind'] == recipe['strength_policy'] and calibration['strengths'] == [0., .25, .5, .75, 1.]
    assert calibration['selected_strength'] in calibration['strengths']
    component = BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
    component.load_state_dict(state['model'], strict=True)
    component.radius = .25 * calibration['selected_strength']
    previous_seconds = 0.
    for prior_name in recipe['generation_job_names']:
        if prior_name == job_name:
            continue
        prior_path = ROOT / 'audit' / (prior_name + '_job.json')
        if prior_path.exists():
            prior_job = read(prior_path)
            assert prior_job['status'] != 'RUNNING' and not prior_job.get('exit_code') is None
            previous_seconds += prior_job['charged_seconds']
    assert previous_seconds < recipe['maximum_cumulative_generation_seconds'] - 90
    started = time.monotonic()
    pipe = adapter = projector = prompts = host = None
    if len(saved) < 16600:
        assert torch.cuda.is_available()
        component.cuda()
        pipe, adapter, projector, prompts, host = load_native(training=False)
    else:
        host = read(out / 'model_receipt.json')['generator']
    configuration = dict(candidate=recipe['candidate'], arm=recipe['arm'], run_name=entry['run_name'],
        recipe_core_sha256=core_sha(recipe), source_sha256=recipe['source_sha256'],
        generation_source_sha256=sha(__file__), generator=host, component_checkpoint_path=str(checkpoint),
        component_checkpoint_sha256=trained['checkpoint_sha256'], parent_bindings=parents,
        parent_role='full_source_dropout_v2', calibration=calibration, training_seed=1729,
        resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1,
        normal_inference_target_access=False, teacher_or_rank_pair_used_at_inference=False)
    receipt_path = out / 'model_receipt.json'
    if receipt_path.exists():
        assert read(receipt_path) == configuration
    else:
        write(receipt_path, configuration)
    progress, handles = {}, []
    if pipe is not None:
        def unet_hook(module, arguments, output):
            assert arguments[0].shape[0] == 1 and torch.isfinite(output[0]).all()
            if progress['UNet_NFE'] == 0:
                progress['initial_UNet_input_sha256'] = hashlib.sha256(arguments[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            progress['UNet_NFE'] += 1
        def key_hook(module, arguments):
            assert arguments[0].shape == (1, 4, 2048) and arguments[0].data_ptr() == progress['token_pointer']
            progress['IP_conditioned_attention_calls'] += 1
        def value_hook(module, arguments):
            assert arguments[0].shape == (1, 4, 2048) and arguments[0].data_ptr() == progress['token_pointer']
            progress['IP_value_calls'] += 1
        def vae_hook(module, arguments, output):
            assert torch.isfinite(output).all()
            progress['VAE_decoder_calls'] += 1
            progress['VAE_float_min'], progress['VAE_float_max'] = float(output.min()), float(output.max())
        layers = [processor for processor in pipe.unet.attn_processors.values() if type(processor).__name__ == 'IPAttnProcessor2_0']
        assert len(layers) == 70
        handles = [pipe.unet.register_forward_hook(unet_hook), pipe.vae.decoder.register_forward_hook(vae_hook)]
        for processor in layers:
            handles.extend([processor.to_k_ip.register_forward_pre_hook(key_hook), processor.to_v_ip.register_forward_pre_hook(value_hook)])
    manifest = Path(entry['inference_manifest'])
    records = list(saved)
    index = 0
    stopped = False
    try:
        with manifest.open('a', buffering=1) as log:
            for subject in inputs['subjects']:
                source = parents[subject]
                data = torch.load(source['condition_cache_path'], weights_only=True, map_location='cpu', mmap=True)
                assert data['model_sha256'] == source['checkpoint_sha256']
                assert data['condition'].shape == (len(data['array_rows']), 1024)
                lookup = {int(row): i for i, row in enumerate(data['array_rows'])}
                assert len(lookup) == len(data['array_rows'])
                brain = torch.load(source['checkpoint_path'], weights_only=True, map_location='cpu')['model']
                weight, bias = brain['block.layers.0.norm.weight'][None], brain['block.layers.0.norm.bias'][None]
                if pipe is not None:
                    weight, bias = weight.cuda(), bias.cuda()
                for query in inputs['inputs']:
                    current = index
                    index += 1
                    position = lookup[query['array_row']]
                    condition = data['condition'][position:position + 1]
                    assert condition.shape == (1, 1024) and torch.isfinite(condition).all()
                    condition_sha = hashlib.sha256(condition.contiguous().numpy().tobytes()).hexdigest()
                    if current < len(saved):
                        row = saved[current]
                        row_identity(row, subject, query, current, recipe, protocol['version'])
                        assert row['condition_sha256'] == condition_sha
                        assert row['parent_checkpoint_sha256'] == source['checkpoint_sha256']
                        assert row['condition_cache_sha256'] == source['condition_cache_sha256']
                        assert row['component_checkpoint_sha256'] == trained['checkpoint_sha256']
                        assert row['generation_source_sha256'] == sha(__file__)
                        assert row['model_receipt_sha256'] == sha(receipt_path)
                        continue
                    elapsed = time.monotonic() - started
                    if (elapsed > recipe['generation_job_seconds'] - 90 or
                            previous_seconds + elapsed > recipe['maximum_cumulative_generation_seconds'] - 90):
                        stopped = True
                        break
                    if current % 500 == 0:
                        verify_admission(recipe, 'formal')
                    segment = next(item for item in entry['image_segments'] if item['start'] <= current < item['stop'])
                    path = private_path(str(Path(segment['root']) / entry['run_name'] / subject / (Path(query['stimulus_id']).stem + '.webp')))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + '.pending')
                    assert not path.exists() and not temporary.exists(), 'Preserve orphan image; never regenerate over it'
                    identity, image_seed = query_identity(subject, query, protocol['version'])
                    progress.clear()
                    progress.update(UNet_NFE=0, IP_conditioned_attention_calls=0, IP_value_calls=0, VAE_decoder_calls=0)
                    with torch.inference_mode():
                        raw, info = component(condition.cuda(), weight, bias, gate_mode='fixed')
                        native = raw.to(torch.float16)
                        tokens = projector(adapter(native))
                        assert tokens.shape == (1, 4, 2048) and torch.isfinite(tokens).all()
                        progress['token_pointer'] = tokens.data_ptr()
                        image = pipe(height=512, width=512,
                            generator=torch.Generator(device='cuda').manual_seed(image_seed),
                            prompt_embeds=prompts[0], pooled_prompt_embeds=prompts[1],
                            cross_attention_kwargs={'ip_hidden_states': tokens}, num_inference_steps=4,
                            guidance_scale=0., output_type='pil').images[0].convert('RGB')
                    assert progress['UNet_NFE'] == 4 and progress['IP_conditioned_attention_calls'] == progress['IP_value_calls'] == 280
                    assert progress['VAE_decoder_calls'] == 1
                    progress.pop('token_pointer')
                    pixels = np.asarray(image)
                    image.save(temporary, format='WEBP', lossless=True, quality=90, method=4)
                    with Image.open(temporary) as decoded:
                        assert decoded.size == (512, 512) and np.array_equal(pixels, np.asarray(decoded.convert('RGB')))
                    with temporary.open('rb') as stream:
                        os.fsync(stream.fileno())
                    temporary.rename(path)
                    row = dict(query_id=identity, subject=subject, stimulus_id=query['stimulus_id'], array_row=query['array_row'],
                        global_query_index=current, EEG_source_stimulus_id=query['stimulus_id'], image_seed=image_seed,
                        candidate=recipe['candidate'], arm=recipe['arm'], run_name=entry['run_name'],
                        method='BrainHIVE_' + entry['run_name'], repeat_protocol='R4', intervention='correct',
                        recipe_core_sha256=core_sha(recipe), generation_source_sha256=sha(__file__),
                        model_receipt_sha256=sha(receipt_path), generated_path=str(path), generated_sha256=sha(path),
                        parent_checkpoint_path=source['checkpoint_path'], parent_checkpoint_sha256=source['checkpoint_sha256'],
                        condition_cache_path=source['condition_cache_path'], condition_cache_sha256=source['condition_cache_sha256'],
                        condition_sha256=condition_sha, condition_norm=float(condition.norm()),
                        component_checkpoint_path=str(checkpoint), component_checkpoint_sha256=trained['checkpoint_sha256'],
                        component_correction_RMS=float((raw - condition.cuda()).square().mean().sqrt()),
                        component_effective_radius=component.radius, component_gate=float(info['gate'][0]),
                        actual_A_input_fp16_sha256=hashlib.sha256(native.cpu().contiguous().numpy().tobytes()).hexdigest(),
                        actual_A_input_fp16_norm=float(native.float().norm()),
                        target_or_visual_features_read_for_condition=False, oracle_diagnostic_only=False,
                        teacher_or_negative_used_at_inference=False, selected_best_seed=False,
                        lossless_pixel_roundtrip=True, **progress)
                    assert all(math.isfinite(row[key]) for key in ('condition_norm', 'component_correction_RMS', 'VAE_float_min', 'VAE_float_max'))
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                    records.append(row)
                    if len(records) % 100 == 0 or len(records) == 16600:
                        print(json.dumps(dict(stage='FUNCTIONAL_ALIGNMENT_V2_ACTUAL_V_IMAGES', arm=recipe['arm'],
                            completed=len(records), expected=16600, seconds=time.monotonic() - started)), flush=True)
                if stopped:
                    break
    finally:
        for handle in handles:
            handle.remove()
        write(out / ('generation_invocation_' + job_name + '_complete.json'),
            dict(job_name=job_name, seconds=time.monotonic() - started, records=len(records), resource_stopped=stopped))
    if stopped:
        write(out / 'progress.json', dict(status='FUNCTIONAL_V2_GENERATION_FINITE_STOP_EXACT_PREFIX_PRESERVED', completed=len(records), expected=16600))
        raise RuntimeError('Finite generation boundary; complete query prefix preserved without regeneration')
    assert index == len(records) == 16600
    completion = dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_ACTUAL_IMAGES_SAVED_NO_TARGET_ACCESS',
        candidate=recipe['candidate'], arm=recipe['arm'], run_name=entry['run_name'], queries=16600,
        inference_manifest_sha256=sha(manifest), model_receipt_sha256=sha(receipt_path),
        recipe_core_sha256=core_sha(recipe), normal_inference_target_access=False,
        images_per_query=1, UNet_NFE_per_image=4, IP_calls_per_image=280,
        selected_best_seed=False, generated_images_selected_or_discarded=False, image_gain_established=False)
    inference_path = out / 'inference_completion.json'
    if inference_path.exists():
        assert read(inference_path) == completion
    else:
        write(inference_path, completion)
    proof = attach_targets_after_all_inference(recipe, entry, inputs, records, completion)
    write(out / 'completion.json', dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_IMAGES_AND_TARGET_JOIN_COMPLETE',
        candidate=recipe['candidate'], arm=recipe['arm'], role='full', split='V', intervention='correct',
        queries=16600, images_per_query=1, selected_best_seed=False,
        inference_completion_sha256=sha(inference_path), manifest_sha256=proof['manifest_sha256'],
        pairing_audit_sha256=sha(out / 'pairing_audit.json'), recipe_core_sha256=core_sha(recipe),
        image_gain_established=False, next_required='Frozen full-query feature extraction and matched evaluation'))


if __name__ == '__main__':
    main()
