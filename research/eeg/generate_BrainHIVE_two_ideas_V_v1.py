"""Two registered candidates, unchanged native HIVE sampling and identity joins.

The KAN arm uses its own raw-EEG encoder exports. It never borrows a BrainMLP
condition while claiming to evaluate KAN. Target images are joined after sampling.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
CANDIDATES = ('brainkan_full_v1', 'response_rank_v1')
KAN_STATUS = 'ALL10_FIXED_EPOCH_BRAINKAN_FULL_T_ALIGNMENTS_AND_CONDITION_EXPORTS_COMPLETE'
RESPONSE_STATUS = 'COMPLETE_ALL10_REAL_OOF_THREE_PASSES_AND_FIXED_FINAL_RESPONSE_RANK_COMPONENT'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def recipe_core_sha256(recipe):
    body = {k: v for k, v in recipe.items() if k not in ('storage_admission_receipt', 'storage_admission_sha256')}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_placement(recipe, candidate):
    assert candidate in CANDIDATES and recipe['candidate'] == candidate
    run = recipe['run_name']
    assert isinstance(run, str) and run and Path(run).name == run and '/' not in run and '\\' not in run
    segments = recipe['image_segments']
    assert segments and segments[0]['start'] == 0 and segments[-1]['stop'] == 16600
    assert all(s['start'] < s['stop'] for s in segments)
    assert all(a['stop'] == b['start'] for a, b in zip(segments, segments[1:]))
    allowed = [PurePosixPath(p) for p in recipe['allowed_private_roots']]
    assert all(PurePosixPath(s['root']).is_absolute() and PurePosixPath(s['root']) in allowed
               and '..' not in PurePosixPath(s['root']).parts for s in segments)
    return segments


def query_identity(subject, query, protocol_version):
    q = f"{subject}/{query['stimulus_id']}/R4"
    seed = int(hashlib.sha256(f"1729|{subject}|{query['stimulus_id']}|R4|{protocol_version}".encode()).hexdigest()[:16], 16) % (2**63 - 1)
    return q, seed


def check_contract():
    recipe = dict(candidate='brainkan_full_v1', run_name='test_only',
        image_segments=[dict(start=0, stop=12000, root='/root/private_a'), dict(start=12000, stop=16600, root='/root/private_b')],
        allowed_private_roots=['/root/private_a', '/root/private_b'])
    validate_placement(recipe, 'brainkan_full_v1')
    base = recipe_core_sha256(recipe)
    recipe.update(storage_admission_receipt='/tmp/admission', storage_admission_sha256='x')
    assert recipe_core_sha256(recipe) == base
    q = dict(stimulus_id='sample_3.jpg', array_row=841)
    assert query_identity('sub-01', q, 'v1') == query_identity('sub-01', q, 'v1')
    assert query_identity('sub-01', q, 'v1') != query_identity('sub-02', q, 'v1')
    assert query_identity('sub-01', q, 'v1') != query_identity('sub-01', dict(q, stimulus_id='sample_4.jpg'), 'v1')
    for broken in (dict(recipe, candidate='response_rank_v1'), dict(recipe, run_name='../bad'),
                   dict(recipe, image_segments=[dict(start=1, stop=16600, root='/root/private_a')])):
        try:
            validate_placement(broken, 'brainkan_full_v1')
        except AssertionError:
            pass
        else:
            raise AssertionError('Malformed candidate placement accepted')
    return dict(status='CPU_CONTRACT_CHECK_PASSED', candidate_identity_separate=True,
        same_query_noise_rule_as_parent=True, no_images_generated=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe')
    parser.add_argument('--candidate', choices=CANDIDATES)
    parser.add_argument('--check-contract', action='store_true')
    args = parser.parse_args()
    if args.check_contract:
        print(json.dumps(check_contract()), flush=True)
        return
    assert args.recipe and args.candidate
    assert b'run_gpu_job_v2.py' in (Path('/proc') / str(os.getppid()) / 'cmdline').read_bytes()
    recipe_path = Path(args.recipe)
    recipe = json.loads(recipe_path.read_text())
    segments = validate_placement(recipe, args.candidate)
    for name, value in recipe['source_sha256'].items():
        assert sha(ROOT / name) == value, name
    from brainhive_two_ideas_storage_v1 import verify_candidate_remaining
    verify_candidate_remaining(recipe, args.candidate)
    input_path = WORK / 'data/generation_inputs/V.json'
    assert sha(input_path) == recipe['generation_inputs_sha256'][str(input_path)]
    import torch
    from PIL import Image
    import numpy as np
    from brainhive_component_generator_v1 import load_native
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = recipe['run_name']
    out = WORK / 'images' / run
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'completion.json').exists()
    encoder_root = Path(recipe.get('encoder_output_directory', str(WORK / 'runs' / ('brainkan_full_v1' if args.candidate == 'brainkan_full_v1' else 'full_source_dropout_v2'))))
    encoder_role = 'brainkan_full_v1' if args.candidate == 'brainkan_full_v1' else 'full_source_dropout_v2'
    component = None
    calibration = {}
    component_provenance = None
    if args.candidate == 'brainkan_full_v1':
        done = json.loads((encoder_root / 'completion.json').read_text())
        assert done['status'] == KAN_STATUS and done['subjects'] == 10
        assert done['recipe_core_sha256'] == recipe_core_sha256(recipe)
        assert done['recipe_sha256'] == sha(recipe_path)
    else:
        folder = Path(recipe.get('component_output_directory', str(WORK / 'components/two_ideas_v1/response_rank_v1')))
        done = json.loads((folder / 'completion.json').read_text())
        assert done['status'] == RESPONSE_STATUS
        checkpoint = Path(done['checkpoint_path'])
        assert sha(checkpoint) == done['checkpoint_sha256']
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        assert state['candidate'] == args.candidate and state['recipe_sha256'] == sha(recipe_path)
        assert state['updates'] == 24807 and state['seen_queries'] == 396900 and state['passes'] == 3
        calibration = state['calibration']
        assert calibration['kind'] == 'Dfit_only_global_strength_calibration'
        assert calibration['selected_strength'] in [0, .25, .5, .75, 1]
        component = BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
        component.load_state_dict(state['model'], strict=True)
        component.radius = .25 * calibration['selected_strength']
        component_provenance = dict(path=str(checkpoint), sha256=done['checkpoint_sha256'])
    pipe, adapter, projector, prompts, host = load_native(training=False)
    configuration = dict(generator=host, candidate=args.candidate, run_name=run,
        recipe_path=str(recipe_path), recipe_sha256=sha(recipe_path), recipe_core_sha256=recipe_core_sha256(recipe),
        source_sha256=sha(__file__), encoder_role=encoder_role, encoder_root=str(encoder_root),
        component_checkpoint=component_provenance, calibration=calibration,
        resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1, training_seed=1729,
        normal_inference_target_access=False, component_gain_established=False)
    receipt = out / 'model_receipt.json'
    if receipt.exists():
        assert json.loads(receipt.read_text()) == configuration
    else:
        write(receipt, configuration)
    inputs = json.loads((WORK / 'data/generation_inputs/V.json').read_text())
    assert inputs['expected_queries'] == 16600 and len(inputs['subjects']) == 10 and len(inputs['inputs']) == 1660
    assert len(set(inputs['subjects'])) == 10
    assert len({q['stimulus_id'] for q in inputs['inputs']}) == 1660
    assert len({Path(q['stimulus_id']).stem for q in inputs['inputs']}) == 1660
    assert not inputs['target_image_or_class_fields_present'] and inputs['repeat_protocol'] == 'R4'
    assert all(set(r) == {'stimulus_id', 'array_row'} for r in inputs['inputs'])
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    progress = {}
    cls = type(next(p for n, p in pipe.unet.attn_processors.items() if n.endswith('attn2.processor')))
    original = cls.__call__
    cross_count = sum(n.endswith('attn2.processor') for n in pipe.unet.attn_processors)
    @functools.wraps(original)
    def checked(self, *a, **kw):
        value = kw.get('ip_hidden_states')
        assert value is not None and value.shape == (1, 4, 2048)
        assert value.data_ptr() == progress['token_pointer']
        progress['IP_conditioned_attention_calls'] += 1
        return original(self, *a, **kw)
    cls.__call__ = checked
    def unet_hook(module, args, result):
        progress['UNet_NFE'] += args[0].shape[0]
        assert torch.isfinite(result[0]).all()
    def vae_hook(module, args, result):
        assert torch.isfinite(result).all()
        progress['VAE_decoder_calls'] += 1
        progress['VAE_float_min'] = float(result.min())
        progress['VAE_float_max'] = float(result.max())
    handles = [pipe.unet.register_forward_hook(unet_hook), pipe.vae.decoder.register_forward_hook(vae_hook)]
    manifest = out / 'inference_manifest.jsonl'
    saved = [json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    done = {r['query_id']: r for r in saved}
    assert len(done) == len(saved)
    for row in saved:
        assert sha(row['generated_path']) == row['generated_sha256']
        assert row['candidate'] == args.candidate and row['encoder_role'] == encoder_role
    global_index = 0
    segment_index = 0
    started = time.monotonic()
    try:
        with manifest.open('a', buffering=1) as log:
            for subject in inputs['subjects']:
                folder = encoder_root / subject
                checkpoint = folder / 'brain_final.pt'
                cache = folder / 'conditions.pt'
                checkpoint_sha = sha(checkpoint)
                cache_sha = sha(cache)
                data = torch.load(cache, weights_only=True, map_location='cpu', mmap=True)
                assert data['model_sha256'] == checkpoint_sha
                state = torch.load(checkpoint, weights_only=True, map_location='cpu')
                brain = state['model']
                if args.candidate == 'brainkan_full_v1':
                    assert state['class_name'] == 'brainhive_brainkan_v1.BrainKAN'
                    assert state['recipe_core_sha256'] == recipe_core_sha256(recipe)
                    assert data['source_class'] == 'brainhive_brainkan_v1.BrainKAN'
                    norm_keys = ('norm.weight', 'norm.bias')
                else:
                    norm_keys = ('block.layers.0.norm.weight', 'block.layers.0.norm.bias')
                weight = brain[norm_keys[0]].cuda()[None]
                bias = brain[norm_keys[1]].cuda()[None]
                lookup = {int(row): i for i, row in enumerate(data['array_rows'])}
                assert len(lookup) == len(data['array_rows'])
                assert data['condition'].shape == (16540, 1024) and data['condition'].dtype == torch.float32
                for query in inputs['inputs']:
                    index = global_index
                    global_index += 1
                    while index >= segments[segment_index]['stop']:
                        segment_index += 1
                    segment = segments[segment_index]
                    assert segment['start'] <= index < segment['stop']
                    q, image_seed = query_identity(subject, query, protocol['version'])
                    if q in done:
                        prior = done[q]
                        assert prior['global_query_index'] == index and prior['encoder_checkpoint_sha256'] == checkpoint_sha
                        assert prior['condition_cache_sha256'] == cache_sha
                        continue
                    if index % 500 == 0:
                        verify_candidate_remaining(recipe, args.candidate)
                    p = Path(segment['root']) / run / subject / (Path(query['stimulus_id']).stem + '.webp')
                    p.parent.mkdir(parents=True, exist_ok=True)
                    assert not p.exists() and not p.is_symlink(), 'Preserve orphan output for explicit repair'
                    c = data['condition'][lookup[query['array_row']]:lookup[query['array_row']] + 1]
                    assert c.shape == (1, 1024) and torch.isfinite(c).all()
                    progress.clear()
                    progress.update(UNet_NFE=0, IP_conditioned_attention_calls=0, VAE_decoder_calls=0)
                    with torch.inference_mode():
                        raw = c.cuda()
                        if component is not None:
                            raw, _ = component(raw, weight, bias)
                        native = raw.to(torch.float16)
                        tokens = projector(adapter(native))
                        progress['token_pointer'] = tokens.data_ptr()
                        assert torch.isfinite(tokens).all() and tokens.shape == (1, 4, 2048)
                        image = pipe(height=512, width=512,
                            generator=torch.Generator(device='cuda').manual_seed(image_seed),
                            prompt_embeds=prompts[0], pooled_prompt_embeds=prompts[1],
                            cross_attention_kwargs={'ip_hidden_states': tokens}, num_inference_steps=4,
                            guidance_scale=0., output_type='pil').images[0].convert('RGB')
                    assert progress['UNet_NFE'] == 4 and progress['IP_conditioned_attention_calls'] == 4 * cross_count
                    assert progress['VAE_decoder_calls'] == 1
                    progress.pop('token_pointer')
                    pixels = np.asarray(image)
                    image.save(p, format='WEBP', lossless=True, quality=90, method=4)
                    with Image.open(p) as decoded:
                        assert np.array_equal(pixels, np.asarray(decoded.convert('RGB')))
                    row = dict(query_id=q, subject=subject, stimulus_id=query['stimulus_id'], array_row=query['array_row'],
                        candidate=args.candidate, global_query_index=index, repeat_protocol='R4', intervention='correct',
                        method='BrainHIVE_' + run, EEG_source_stimulus_id=query['stimulus_id'], image_seed=image_seed,
                        generated_path=str(p), generated_sha256=sha(p), encoder_role=encoder_role,
                        encoder_checkpoint_path=str(checkpoint), encoder_checkpoint_sha256=checkpoint_sha,
                        condition_cache_path=str(cache), condition_cache_sha256=cache_sha,
                        condition_sha256=hashlib.sha256(c.contiguous().numpy().tobytes()).hexdigest(), condition_norm=float(c.norm()),
                        actual_A_input_fp16_sha256=hashlib.sha256(native.cpu().contiguous().numpy().tobytes()).hexdigest(),
                        actual_A_input_fp16_norm=float(native.float().norm()), component_checkpoint=component_provenance,
                        component_correction_RMS=float((raw - c.cuda()).square().mean().sqrt()),
                        component_effective_radius=component.radius if component is not None else 0.,
                        target_or_visual_features_read_for_condition=False, oracle_diagnostic_only=False,
                        lossless_pixel_roundtrip=True, **progress)
                    log.write(json.dumps(row) + '\n')
                    done[q] = row
                    if len(done) % 100 == 0 or len(done) == 16600:
                        print(json.dumps(dict(stage='BRAINHIVE_TWO_IDEAS_ACTUAL_IMAGES', candidate=args.candidate,
                            completed=len(done), expected=16600, seconds=time.monotonic() - started)), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        cls.__call__ = original
    assert global_index == len(done) == 16600
    write(out / 'completion.json', dict(status='ALL_ACTUAL_LOSSLESS_IMAGES_SAVED_INFERENCE_ONLY', queries=16600,
        role='brainkan_full_v1' if args.candidate == 'brainkan_full_v1' else 'full', encoder_role=encoder_role,
        candidate=args.candidate, split='V', intervention='correct', seconds=time.monotonic() - started,
        inference_manifest_sha256=sha(manifest), model_receipt_sha256=sha(receipt),
        images_per_query=1, selected_best_seed=False, component_gain_established=False))


if __name__ == '__main__':
    main()
