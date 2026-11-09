"""Independent CPU join and encoder/branch provenance audit for both new arms."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

from generate_BrainHIVE_two_ideas_V_v1 import (CANDIDATES, KAN_STATUS, RESPONSE_STATUS,
    ROOT, WORK, check_contract, query_identity, recipe_core_sha256, sha, validate_placement, write)


def assert_same_or_create(path, data):
    path = Path(path)
    if path.exists():
        assert path.read_text() == data, 'Preserve an existing differing audit/manifest'
    else:
        path.write_text(data)


def verify_query_row(row, subject, query, index, protocol_version, candidate):
    identity, seed = query_identity(subject, query, protocol_version)
    assert row['query_id'] == identity and row['image_seed'] == seed
    assert row['global_query_index'] == index and row['subject'] == subject
    assert row['candidate'] == candidate and row['repeat_protocol'] == 'R4'
    assert row['stimulus_id'] == row['EEG_source_stimulus_id'] == query['stimulus_id']
    assert row['array_row'] == query['array_row'] and row['intervention'] == 'correct'
    assert not row['target_or_visual_features_read_for_condition'] and not row['oracle_diagnostic_only']


def contract_test():
    result = check_contract()
    query = dict(stimulus_id='image_06.jpg', array_row=985)
    q, seed = query_identity('sub-03', query, 'protocol_test')
    row = dict(query_id=q, image_seed=seed, global_query_index=99, subject='sub-03',
        candidate='brainkan_full_v1', repeat_protocol='R4', stimulus_id=query['stimulus_id'],
        EEG_source_stimulus_id=query['stimulus_id'], array_row=985, intervention='correct',
        target_or_visual_features_read_for_condition=False, oracle_diagnostic_only=False)
    verify_query_row(row, 'sub-03', query, 99, 'protocol_test', 'brainkan_full_v1')
    for key, value in [('array_row', 984), ('EEG_source_stimulus_id', 'other.jpg'),
                       ('candidate', 'response_rank_v1'), ('image_seed', seed + 1), ('global_query_index', 100)]:
        try:
            verify_query_row(dict(row, **{key: value}), 'sub-03', query, 99, 'protocol_test', 'brainkan_full_v1')
        except AssertionError:
            pass
        else:
            raise AssertionError('Query/index/encoder mismatch accepted')
    return dict(result, deliberate_pairing_mismatches_rejected=True, no_images_scored=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe')
    parser.add_argument('--candidate', choices=CANDIDATES)
    parser.add_argument('--run-name')
    parser.add_argument('--check-contract', action='store_true')
    args = parser.parse_args()
    if args.check_contract:
        print(json.dumps(contract_test()), flush=True)
        return
    if args.recipe:
        recipe_path = Path(args.recipe)
    else:
        assert args.run_name and Path(args.run_name).name == args.run_name
        prior = json.loads((WORK / 'images' / args.run_name / 'model_receipt.json').read_text())
        recipe_path = Path(prior['recipe_path'])
    recipe = json.loads(recipe_path.read_text())
    for filename,digest in recipe['generation_inputs_sha256'].items():
        assert sha(filename) == digest, filename
    candidate = args.candidate or recipe['candidate']
    segments = validate_placement(recipe, candidate)
    run = recipe['run_name']
    if args.run_name:
        assert args.run_name == run
    out = WORK / 'images' / run
    completion = json.loads((out / 'completion.json').read_text())
    receipt = json.loads((out / 'model_receipt.json').read_text())
    assert completion['status'] == 'ALL_ACTUAL_LOSSLESS_IMAGES_SAVED_INFERENCE_ONLY'
    assert completion['candidate'] == receipt['candidate'] == candidate
    assert completion['split'] == 'V' and completion['intervention'] == 'correct' and completion['queries'] == 16600
    assert completion['images_per_query'] == 1 and not completion['selected_best_seed']
    assert receipt['recipe_sha256'] == sha(recipe_path) and receipt['recipe_core_sha256'] == recipe_core_sha256(recipe)
    assert completion['model_receipt_sha256'] == sha(out / 'model_receipt.json')
    assert sha(out / 'inference_manifest.jsonl') == completion['inference_manifest_sha256']
    rows = list(map(json.loads, (out / 'inference_manifest.jsonl').read_text().splitlines()))
    assert len(rows) == len({r['query_id'] for r in rows}) == 16600
    assert len({r['generated_path'] for r in rows}) == len(rows)
    assert len({r['generated_sha256'] for r in rows}) == len(rows), 'Duplicate actual outputs need investigation'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_two_ideas_storage_v1 import verify_candidate_remaining
    verify_candidate_remaining(recipe, candidate)
    inputs = json.loads((WORK / 'data/generation_inputs/V.json').read_text())
    assert inputs['repeat_protocol'] == 'R4' and len(inputs['subjects']) == 10 and len(inputs['inputs']) == 1660
    assert not inputs['target_image_or_class_fields_present']
    targets_list = list(map(json.loads, (WORK / 'data/generation_inputs/V_targets.jsonl').read_text().splitlines()))
    targets = {r['stimulus_id']: r for r in targets_list}
    assert len(targets) == len(targets_list) == 1660
    assert set(targets) == {r['stimulus_id'] for r in inputs['inputs']}
    assert len({Path(stimulus).name for stimulus in targets}) == len(targets)
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    cfg = json.loads((WORK / 'assets/prior_H14/attn_adapter/config.json').read_text())
    expected_IP_calls = 4 * sum(n.endswith('attn2.processor') for n in cfg['attn_processor_keys'])
    assert expected_IP_calls == 280
    expected_encoder_role = 'brainkan_full_v1' if candidate == 'brainkan_full_v1' else 'full_source_dropout_v2'
    assert receipt['encoder_role'] == completion['encoder_role'] == expected_encoder_role
    assert completion['role'] == ('brainkan_full_v1' if candidate == 'brainkan_full_v1' else 'full')
    encoder_root = Path(recipe.get('encoder_output_directory', str(WORK / 'runs' / expected_encoder_role)))
    assert Path(receipt['encoder_root']) == encoder_root
    component = None
    if candidate == 'brainkan_full_v1':
        done = json.loads((encoder_root / 'completion.json').read_text())
        assert done['status'] == KAN_STATUS and done['subjects'] == 10
        assert done['recipe_core_sha256'] == recipe_core_sha256(recipe)
        assert receipt['component_checkpoint'] is None
    else:
        folder = Path(recipe.get('component_output_directory', str(WORK / 'components/two_ideas_v1/response_rank_v1')))
        done = json.loads((folder / 'completion.json').read_text())
        assert done['status'] == RESPONSE_STATUS
        assert sha(done['checkpoint_path']) == done['checkpoint_sha256']
        assert receipt['component_checkpoint'] == dict(path=done['checkpoint_path'], sha256=done['checkpoint_sha256'])
        state = torch.load(done['checkpoint_path'], weights_only=True, map_location='cpu')
        assert state['candidate'] == candidate and state['recipe_sha256'] == sha(recipe_path)
        assert state['calibration'] == receipt['calibration']
        assert receipt['calibration']['kind'] == 'Dfit_only_global_strength_calibration'
        component = BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
        component.load_state_dict(state['model'], strict=True)
        component.radius = .25 * receipt['calibration']['selected_strength']
    gallery = WORK / 'gallery/V'
    gallery.mkdir(parents=True, exist_ok=True)
    real = []
    for stimulus, target in targets.items():
        path = gallery / Path(stimulus).name
        if not path.exists():
            with zipfile.ZipFile(target['image_zip']) as z:
                data = z.read(target['image_member'])
            assert hashlib.sha256(data).hexdigest() == target['image_sha256']
            path.write_bytes(data)
        assert sha(path) == target['image_sha256']
        real.append(dict(stimulus_id=stimulus, image_path=str(path), image_sha256=target['image_sha256'], concept_id=target['concept_id']))
    gallery_lookup = {r['stimulus_id']: r for r in real}
    joined = []
    index = 0
    segment_index = 0
    maximum_branch_replay_RMS_error = 0.
    maximum_branch_replay_A_input_norm_error = 0.
    encoder_provenance = []
    with torch.inference_mode():
        for subject in inputs['subjects']:
            checkpoint = encoder_root / subject / 'brain_final.pt'
            cache = encoder_root / subject / 'conditions.pt'
            checkpoint_sha = sha(checkpoint)
            cache_sha = sha(cache)
            exported = torch.load(cache, weights_only=True, map_location='cpu', mmap=True)
            assert exported['model_sha256'] == checkpoint_sha
            state = torch.load(checkpoint, weights_only=True, map_location='cpu')
            if candidate == 'brainkan_full_v1':
                assert state['class_name'] == exported['source_class'] == 'brainhive_brainkan_v1.BrainKAN'
                assert state['recipe_core_sha256'] == recipe_core_sha256(recipe)
                norm_keys = ('norm.weight', 'norm.bias')
            else:
                norm_keys = ('block.layers.0.norm.weight', 'block.layers.0.norm.bias')
            weight = state['model'][norm_keys[0]][None]
            bias = state['model'][norm_keys[1]][None]
            lookup = {int(q): i for i, q in enumerate(exported['array_rows'])}
            assert len(lookup) == len(exported['array_rows'])
            encoder_provenance.append(dict(subject=subject, checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                condition_cache_path=str(cache), condition_cache_sha256=cache_sha, encoder_role=expected_encoder_role))
            for query in inputs['inputs']:
                row = rows[index]
                verify_query_row(row, subject, query, index, protocol['version'], candidate)
                while index >= segments[segment_index]['stop']:
                    segment_index += 1
                expected_path = Path(segments[segment_index]['root']) / run / subject / (Path(query['stimulus_id']).stem + '.webp')
                assert Path(row['generated_path']) == expected_path
                assert row['encoder_role'] == expected_encoder_role
                assert row['encoder_checkpoint_path'] == str(checkpoint) and row['encoder_checkpoint_sha256'] == checkpoint_sha
                assert row['condition_cache_path'] == str(cache) and row['condition_cache_sha256'] == cache_sha
                assert row['component_checkpoint'] == receipt['component_checkpoint']
                c = exported['condition'][lookup[query['array_row']]:lookup[query['array_row']] + 1]
                assert hashlib.sha256(c.contiguous().numpy().tobytes()).hexdigest() == row['condition_sha256']
                assert abs(float(c.norm()) - row['condition_norm']) < 1e-5
                corrected = component(c, weight, bias)[0] if component is not None else c
                expected_RMS = float((corrected - c).square().mean().sqrt())
                RMS_error = abs(expected_RMS - row['component_correction_RMS'])
                maximum_branch_replay_RMS_error = max(maximum_branch_replay_RMS_error, RMS_error)
                assert RMS_error < 2e-5, 'Loaded branch differs from independent CPU checkpoint replay'
                norm_error = abs(float(corrected.half().float().norm()) - row['actual_A_input_fp16_norm'])
                maximum_branch_replay_A_input_norm_error = max(maximum_branch_replay_A_input_norm_error, norm_error)
                # CPU and CUDA GEMMs may cross a few FP16 rounding boundaries.
                assert norm_error < .005
                if component is None:
                    assert row['component_correction_RMS'] == row['component_effective_radius'] == 0
                    assert hashlib.sha256(c.half().contiguous().numpy().tobytes()).hexdigest() == row['actual_A_input_fp16_sha256']
                else:
                    assert row['component_effective_radius'] == component.radius
                target = targets[row['stimulus_id']]
                assert int(target['array_row']) == row['array_row']
                assert sha(expected_path) == row['generated_sha256']
                with Image.open(expected_path) as im:
                    im.load()
                    assert im.size == (512, 512) and im.mode == 'RGB'
                assert row['UNet_NFE'] == 4 and row['IP_conditioned_attention_calls'] == expected_IP_calls
                assert row['VAE_decoder_calls'] == 1 and row['lossless_pixel_roundtrip']
                assert all(math.isfinite(row[k]) for k in ('VAE_float_min', 'VAE_float_max', 'condition_norm', 'component_correction_RMS'))
                joined.append(row | dict(concept_id=target['concept_id'], target_image_sha256=target['image_sha256'],
                    score_target_path=gallery_lookup[row['stimulus_id']]['image_path'], target_mapping_attached_after_inference=True))
                index += 1
                if index % 500 == 0:
                    verify_candidate_remaining(recipe, candidate)
    assert index == 16600
    manifest = out / 'manifest.jsonl'
    assert_same_or_create(manifest, ''.join(json.dumps(r) + '\n' for r in joined))
    gallery_manifest = gallery / 'manifest.jsonl'
    assert_same_or_create(gallery_manifest, ''.join(json.dumps(r) + '\n' for r in real))
    proof = dict(status='ALL_SAVED_IMAGES_TARGETS_AND_STIMULUS_ROWS_JOIN_VERIFIED', candidate=candidate, queries=index,
        manifest_sha256=sha(manifest), gallery_manifest=str(gallery_manifest), gallery_manifest_sha256=sha(gallery_manifest),
        independent_raw_event_replay=False, independent_raw_EEG_encoder_forward_replay=False,
        sampled_new_images=False, scientific_image_gain_claimed=False, exact_cached_condition_row_SHA_verified=True,
        actual_encoder_checkpoint_and_cache_SHA_verified=True, encoder_provenance=encoder_provenance,
        fixed_query_identity_order_and_sampling_seed_verified=True, required_IP_calls_per_image=expected_IP_calls,
        response_component_independent_CPU_replay_verified=component is not None,
        maximum_branch_replay_RMS_error=maximum_branch_replay_RMS_error,
        maximum_branch_replay_A_input_norm_error=maximum_branch_replay_A_input_norm_error)
    write(out / 'pairing_audit.json', proof)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
