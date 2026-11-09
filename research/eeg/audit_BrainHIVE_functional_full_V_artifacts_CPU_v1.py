"""Bounded, score-free engineering audit of one completed functional-v2 arm.

This adapter reuses the already checked risk audit's byte reader, decoder,
spatial measurements and bounded gzip writer. It adds the actual v2 completion,
own-arm path and inference/target-join contracts. No image/model is generated or
changed. A constant/duplicate flag is not a perceptual-quality or object verdict.
"""
import argparse
import collections
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time

import audit_BrainHIVE_spatial_image_integrity_CPU_v1 as spatial

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
SCHEMA = 'BrainHIVE_functional_full_V_artifacts_CPU_v1'
MEASUREMENT_SOURCE_SHA256 = 'dce7846742391a5b33ce5a0b1755e14ef24711e024381ce0ea07bb9ef89fa570'
EXPECTED = 16600
MAX_SECONDS = 1800
MAX_OUTPUT_BYTES = 16 << 20
FLAGS = ('exact_spatial_constant', 'near_spatial_constant', 'legacy_global_blank_proxy')
EXTRA_FIELDS = ['expected_query_id', 'global_query_index', 'arm', 'candidate',
                'expected_generated_path', 'identity_verified', 'canonical_path_verified']
FIELDS = spatial.FIELDS + EXTRA_FIELDS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def core_sha(recipe):
    raw = {k: v for k, v in recipe.items() if k not in ('admission_receipt', 'admission_sha256')}
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path, pins, expected=None):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= spatial.MAX_MANIFEST_BYTES, 'Missing/oversized metadata: ' + str(path))
    actual = spatial.sha_file(path)
    require(expected is None or actual == expected, 'Metadata SHA mismatch: ' + str(path))
    pins[str(path)] = actual
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def read_jsonl(path, pins, expected=None):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= spatial.MAX_MANIFEST_BYTES, 'Missing/oversized metadata: ' + str(path))
    actual = spatial.sha_file(path)
    require(expected is None or actual == expected, 'Metadata SHA mismatch: ' + str(path))
    pins[str(path)] = actual
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            require(len(line.encode()) <= spatial.MAX_MANIFEST_LINE_BYTES, 'Oversized metadata row')
            rows.append(json.loads(line))
    return rows


def prepare(recipe_path, deadline, *, _fixture=None):
    """Metadata-only admission; production CLI never accepts a smaller count."""
    prepared_at = time.monotonic()
    require(spatial.sha_file(spatial.__file__) == MEASUREMENT_SOURCE_SHA256, 'Pinned spatial implementation changed')
    work, repo, durable = (WORK, ROOT, (WORK, DATA)) if _fixture is None else (
        _fixture['work'], _fixture['repo'], (_fixture['root'],))
    count = EXPECTED if _fixture is None else _fixture['count']
    subjects_expected = 10 if _fixture is None else _fixture['subjects']
    pins = {str(Path(spatial.__file__)): MEASUREMENT_SOURCE_SHA256, str(Path(__file__)): spatial.sha_file(__file__)}
    recipe = read_json(recipe_path, pins)
    arm = recipe['arm']
    require(arm in ('O', 'P', 'PN', 'S'), 'Unknown functional arm')
    require(recipe['schema'] == 'BrainHIVE_functional_alignment_v2_recipe', 'Not a v2 recipe')
    require(recipe['candidate'] == 'functional_alignment_v2_' + arm and recipe['training_seed'] == 1729, 'Candidate/seed mismatch')
    require(recipe['generator'] == dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1), 'Generator contract mismatch')
    core = core_sha(recipe)
    name = 'protected_R4_functional_alignment_v2_' + arm
    admission = read_json(recipe['admission_receipt'], pins, recipe['admission_sha256'])
    require(admission['candidate'] == recipe['candidate'] and admission['recipe_core_sha256'] == core, 'Admission belongs to another arm/recipe')
    require(admission['status'] == 'FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMITTED', 'No formal lifecycle admission')
    plan = read_json(admission['four_arm_lifecycle_plan']['path'], pins, admission['four_arm_lifecycle_plan']['sha256'])
    matches = [e for e in plan['arms'] if e['arm'] == arm]
    require(len(matches) == 1, 'Missing/duplicate own storage arm')
    entry = matches[0]
    expected_sources = entry['lifecycle_source_sha256']
    for source in (Path(__file__), Path(spatial.__file__)):
        require(expected_sources.get(source.name) == spatial.sha_file(source), 'Sealed lifecycle audit source mismatch: ' + source.name)
    folder = work / 'images' / name
    require(entry['candidate'] == recipe['candidate'] and entry['recipe_core_sha256'] == core, 'Storage plan identity mismatch')
    require(entry['run_name'] == name and Path(entry['metadata_root']) == folder, 'Noncanonical metadata root')
    manifest, inference = folder / 'manifest.jsonl', folder / 'inference_manifest.jsonl'
    require(Path(entry['inference_manifest']) == inference, 'Another arm\'s inference manifest')
    complete = read_json(folder / 'completion.json', pins)
    require(complete['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL16600_IMAGES_AND_TARGET_JOIN_COMPLETE', 'Full generation and target join are not complete')
    for key, expected in dict(candidate=recipe['candidate'], arm=arm, queries=count, role='full', split='V', intervention='correct',
                              images_per_query=1, selected_best_seed=False, recipe_core_sha256=core).items():
        require(complete.get(key) == expected, 'Generation completion mismatch: ' + key)
    inf_complete = read_json(folder / 'inference_completion.json', pins, complete['inference_completion_sha256'])
    require(inf_complete['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL16600_ACTUAL_IMAGES_SAVED_NO_TARGET_ACCESS', 'Inference incomplete')
    for key, expected in dict(candidate=recipe['candidate'], arm=arm, run_name=name, queries=count, recipe_core_sha256=core,
                              normal_inference_target_access=False, images_per_query=1, UNet_NFE_per_image=4,
                              IP_calls_per_image=280, selected_best_seed=False, generated_images_selected_or_discarded=False).items():
        require(inf_complete.get(key) == expected, 'Inference completion mismatch: ' + key)
    proof = read_json(folder / 'pairing_audit.json', pins, complete['pairing_audit_sha256'])
    require(proof['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL16600_TARGET_AND_QUERY_IDENTITIES_JOINED', 'Target join audit incomplete')
    require(all(proof.get(k) == v for k, v in dict(candidate=recipe['candidate'], arm=arm, queries=count,
            manifest_sha256=complete['manifest_sha256']).items()), 'Target join belongs to another arm')
    for path, expected in [(manifest, complete['manifest_sha256']), (inference, inf_complete['inference_manifest_sha256'])]:
        n, actual = spatial.describe_manifest(path, deadline)
        require(n == count and actual == expected, 'Full manifest count/SHA mismatch: ' + str(path))
        pins[str(path)] = actual
    model = read_json(folder / 'model_receipt.json', pins, inf_complete['model_receipt_sha256'])
    require(all(model.get(k) == v for k, v in dict(candidate=recipe['candidate'], arm=arm, run_name=name,
            recipe_core_sha256=core, resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1,
            normal_inference_target_access=False, teacher_or_rank_pair_used_at_inference=False).items()), 'Model receipt identity/interface mismatch')
    require(model['source_sha256'] == recipe['source_sha256'] and digest(model['component_checkpoint_sha256']), 'Model source/checkpoint mismatch')
    require(model['generation_source_sha256'] == recipe['source_sha256']['generate_BrainHIVE_functional_alignment_V_v2.py'], 'Generation source mismatch')
    training = read_json(Path(recipe['output_directory']) / 'completion.json', pins)
    require(training['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE', 'Training incomplete')
    for key, expected in dict(candidate=recipe['candidate'], arm=arm, updates=24807, seen_queries=396900,
                              passes=3, parameters=525568, recipe_core_sha256=core,
                              checkpoint_sha256=model['component_checkpoint_sha256']).items():
        require(training.get(key) == expected, 'Training completion mismatch: ' + key)
    calibration_path = Path(recipe['output_directory']) / 'calibration.json'
    calibration = read_json(calibration_path, pins, training['calibration_sha256'])
    require(model['calibration'] == calibration and calibration['kind'] == 'Dfit_only_global_strength_calibration'
            and calibration['strengths'] == [0., .25, .5, .75, 1.] and calibration['selected_strength'] in calibration['strengths'], 'Calibration differs from the completed fixed arm')
    inputs_path, protocol_path = work / 'data/generation_inputs/V.json', repo / 'configs/split_manifest.json'
    inputs = read_json(inputs_path, pins, recipe['input_sha256'][str(inputs_path)])
    protocol = read_json(protocol_path, pins, recipe['input_sha256'][str(protocol_path)])
    require(inputs['expected_queries'] == count and len(inputs['subjects']) == len(set(inputs['subjects'])) == subjects_expected, 'Input denominator/subjects mismatch')
    require(inputs['repeat_protocol'] == 'R4' and inputs['target_image_or_class_fields_present'] is False, 'Input permission mismatch')
    queries = inputs['inputs']
    require(len(queries) * subjects_expected == count and len({q['stimulus_id'] for q in queries}) == len(queries), 'Stimulus coverage mismatch')
    require(all(set(q) == {'stimulus_id', 'array_row'} and Path(q['stimulus_id']).name == q['stimulus_id'] for q in queries), 'Invalid blind input')
    require({q['array_row'] for q in queries} == set(protocol['image_rows']['V']), 'Protected V array rows mismatch')
    targets_path = work / 'data/generation_inputs/V_targets.jsonl'
    targets_list = read_jsonl(targets_path, pins, recipe['input_sha256'][str(targets_path)])
    gallery_path = Path(recipe['generation_gallery_manifest'])
    gallery_list = read_jsonl(gallery_path, pins, recipe['input_sha256'][str(gallery_path)])
    targets, gallery = ({r['stimulus_id']: r for r in items} for items in (targets_list, gallery_list))
    require(len(targets) == len(targets_list) == len(gallery) == len(gallery_list) == len(queries), 'Target metadata duplicate/missing rows')
    require(set(targets) == set(gallery) == {q['stimulus_id'] for q in queries}, 'Target metadata identity mismatch')
    require(proof['gallery_manifest_sha256'] == pins[str(gallery_path)] and proof['target_metadata_sha256'] == pins[str(targets_path)], 'Pairing inputs differ')
    require(entry['image_segments'][0]['start'] == 0 and entry['image_segments'][-1]['stop'] == count, 'Image storage denominator mismatch')
    require(all(a['stop'] == b['start'] for a, b in zip(entry['image_segments'], entry['image_segments'][1:])), 'Image segment gap/overlap')
    for segment in entry['image_segments']:
        image_root = Path(segment['root'])
        require(image_root.is_absolute() and 0 <= segment['start'] < segment['stop'] <= count, 'Bad image segment')
        require(any(image_root.resolve().is_relative_to(root.resolve()) for root in durable), 'Image segment is not private persistent storage')
    if _fixture is None:
        for filename, expected in recipe['source_sha256'].items():
            require(Path(filename).name == filename and spatial.sha_file(repo / filename) == expected, 'Pinned source changed: ' + filename)
            pins[str(repo / filename)] = expected
    return dict(recipe=recipe, core=core, arm=arm, run_name=name, count=count, inputs=inputs, protocol=protocol,
                entry=entry, model=model, manifest=manifest, inference=inference, targets=targets, gallery=gallery,
                pins=pins, work=work, durable=durable, fixture=_fixture is not None,
                preflight_seconds=time.monotonic() - prepared_at)


def expected_identity(context, index):
    subjects, queries = context['inputs']['subjects'], context['inputs']['inputs']
    subject, query = subjects[index // len(queries)], queries[index % len(queries)]
    qid = subject + '/' + query['stimulus_id'] + '/R4'
    seed = int(hashlib.sha256(('1729|' + subject + '|' + query['stimulus_id'] + '|R4|' + context['protocol']['version']).encode()).hexdigest()[:16], 16) % (2**63 - 1)
    segment = next(s for s in context['entry']['image_segments'] if s['start'] <= index < s['stop'])
    path = Path(segment['root']) / context['run_name'] / subject / (Path(query['stimulus_id']).stem + '.webp')
    return subject, query, qid, seed, path


def identity_issues(context, inference, joined, index):
    subject, query, qid, seed, path = expected_identity(context, index)
    issues = []
    fixed = dict(query_id=qid, subject=subject, stimulus_id=query['stimulus_id'], array_row=query['array_row'],
        global_query_index=index, EEG_source_stimulus_id=query['stimulus_id'], image_seed=seed,
        candidate=context['recipe']['candidate'], arm=context['arm'], run_name=context['run_name'],
        repeat_protocol='R4', intervention='correct', recipe_core_sha256=context['core'],
        component_checkpoint_sha256=context['model']['component_checkpoint_sha256'],
        component_checkpoint_path=context['model']['component_checkpoint_path'],
        model_receipt_sha256=context['pins'][str(context['manifest'].parent / 'model_receipt.json')],
        generation_source_sha256=context['model']['generation_source_sha256'],
        target_or_visual_features_read_for_condition=False, oracle_diagnostic_only=False,
        teacher_or_negative_used_at_inference=False, selected_best_seed=False,
        lossless_pixel_roundtrip=True, UNet_NFE=4, IP_conditioned_attention_calls=280,
        IP_value_calls=280, VAE_decoder_calls=1)
    if any(inference.get(k) != v for k, v in fixed.items()):
        issues.append('inference_identity_failure')
    if any(joined.get(k) != v for k, v in inference.items()):
        issues.append('joined_inference_row_mismatch')
    target, gallery = context['targets'][query['stimulus_id']], context['gallery'][query['stimulus_id']]
    expected_target = dict(concept_id=target['concept_id'], target_image_sha256=target['image_sha256'],
        target_image_path=gallery['image_path'], score_target_path=gallery['image_path'],
        target_image_zip=target['image_zip'], target_image_member=target['image_member'], target_mapping_attached_after_inference=True)
    if int(target['array_row']) != query['array_row'] or target['image_sha256'] != gallery['image_sha256'] or target['concept_id'] != gallery['concept_id'] or any(joined.get(k) != v for k, v in expected_target.items()):
        issues.append('target_metadata_identity_failure')
    parent = context['model']['parent_bindings'][subject]
    if any(inference.get(out) != parent[src] for out, src in [('parent_checkpoint_path', 'checkpoint_path'),
            ('parent_checkpoint_sha256', 'checkpoint_sha256'), ('condition_cache_path', 'condition_cache_path'),
            ('condition_cache_sha256', 'condition_cache_sha256')]):
        issues.append('parent_condition_provenance_failure')
    path_ok = (inference.get('generated_path') == str(path) and joined.get('generated_path') == str(path)
               and not path.is_symlink() and any(path.resolve().is_relative_to(root.resolve()) for root in context['durable']))
    if not path_ok:
        issues.append('canonical_generated_path_failure')
    for name in ('generated_sha256', 'condition_sha256', 'initial_UNet_input_sha256', 'actual_A_input_fp16_sha256'):
        if not digest(inference.get(name)):
            issues.append('invalid_' + name)
    numeric = ('condition_norm', 'component_correction_RMS', 'component_effective_radius', 'component_gate',
               'actual_A_input_fp16_norm', 'VAE_float_min', 'VAE_float_max')
    if any(type(inference.get(k)) not in (int, float) or not math.isfinite(inference[k]) for k in numeric):
        issues.append('nonfinite_generation_numeric_metadata')
    elif (inference['VAE_float_min'] > inference['VAE_float_max'] or inference['component_gate'] != 1
          or inference['component_effective_radius'] != .25 * context['model']['calibration']['selected_strength']):
        issues.append('invalid_generation_numeric_metadata')
    return issues, path_ok, qid, path


def audit_context(context, output, *, max_seconds=MAX_SECONDS, max_output_bytes=MAX_OUTPUT_BYTES):
    require(0 < max_seconds <= MAX_SECONDS and (1 << 20) <= max_output_bytes <= MAX_OUTPUT_BYTES, 'Invalid finite audit budget')
    output = Path(output)
    require(not output.exists(), 'Preserve completed and partial audits; use a fresh output directory')
    metadata_root = Path(context['entry']['metadata_root'])
    output_is_own_metadata = output.resolve() != metadata_root.resolve() and output.resolve().is_relative_to(metadata_root.resolve())
    require(output_is_own_metadata or (context['fixture'] and output.resolve().is_relative_to(context['durable'][0].resolve())),
            'Output must be a fresh child of its sealed own-arm metadata_root')
    used_metadata = sum(p.stat().st_size for p in metadata_root.rglob('*') if p.is_file() and not p.is_symlink())
    require(context['entry']['metadata_budget_bytes'] - used_metadata >= max_output_bytes, 'Sealed remaining own-arm metadata budget does not cover full audit cap')
    require(shutil.disk_usage(metadata_root).free >= max_output_bytes + (64 << 20), 'Insufficient persistent space for bounded audit plus64MiB buffer')
    for name in spatial.THREAD_VARIABLES:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    memory = spatial.memory_preflight()
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 512 * 512
    started = time.monotonic()
    deadline = started + max_seconds
    output.mkdir(parents=True, exist_ok=False)
    budget = spatial.OutputBudget(output, max_output_bytes)
    table = spatial.ChunkedCSV(budget, 'per_query.csv.gz', FIELDS)
    counts, failures, flags, trusted_flags = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    pixels, files = {}, collections.Counter()
    exhausted = output_exhausted = False
    source_errors = []
    with context['inference'].open(encoding='utf-8') as a, context['manifest'].open(encoding='utf-8') as b:
        try:
            for index, pair in enumerate(itertools.zip_longest(a, b)):
                require(index < context['count'] and all(line is not None for line in pair), 'Manifest count changed after preflight')
                row = dict(source_line_number=index + 1, source_manifest_sha256=context['pins'][str(context['manifest'])],
                    hash_verified=False, decoded_512_RGB=False, arm=context['arm'], candidate=context['recipe']['candidate'],
                    global_query_index=index, identity_verified=False, canonical_path_verified=False)
                issues, error = [], ''
                subject, query, qid, seed, path = expected_identity(context, index)
                row.update(expected_query_id=qid, expected_generated_path=str(path))
                try:
                    inference, joined = (json.loads(line) for line in pair)
                    require(isinstance(inference, dict) and isinstance(joined, dict), 'Manifest row must be an object')
                    issues, path_ok, _, _ = identity_issues(context, inference, joined, index)
                    row.update({k: joined.get(k, '') for k in ('query_id', 'subject', 'stimulus_id', 'generated_path')})
                    row['manifest_generated_sha256'] = joined.get('generated_sha256', '')
                    row['identity_verified'], row['canonical_path_verified'] = not issues, path_ok
                except Exception as exception:
                    issues = ['invalid_manifest_identity']
                    error = str(exception)[:200]
                counts['identity_verified'] += int(row['identity_verified'])
                counts['canonical_path_verified'] += int(row['canonical_path_verified'])
                if time.monotonic() >= deadline:
                    exhausted = True
                if issues:
                    pass  # Never read a path whose own-arm identity is invalid.
                elif exhausted:
                    issues.append('not_processed_time_limit')
                else:
                    try:
                        measurements, extra, error = spatial.inspect_generated(joined, deadline, np, Image)
                        row.update(measurements)
                        issues.extend(extra)
                        counts['images_attempted'] += 1
                    except spatial.TimeBudgetExceeded as exception:
                        exhausted = True
                        issues.append('not_processed_time_limit')
                        error = str(exception)
                if row.get('actual_file_sha256'):
                    files[row['actual_file_sha256']] += 1
                if row.get('decoded_512_RGB'):
                    counts['decoded'] += 1
                    if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                        issues.append('nonfinite_pixel_statistic')
                    trusted = bool(row.get('hash_verified')) and not issues
                    counts['trusted'] += int(trusted)
                    for flag in FLAGS:
                        flags[flag] += int(row[flag])
                        trusted_flags[flag] += int(row[flag] and trusted)
                    group = pixels.setdefault(row['decoded_RGB_pixel_sha256'], [0, set()])
                    group[0] += 1
                    group[1].add(row.get('actual_file_sha256', ''))
                failures.update(issues)
                row.update(status='|'.join(issues) if issues else 'OK', error=error)
                table.add(row)
                counts['visited'] += 1
                if counts['visited'] % 1000 == 0:
                    table.flush()
                    print(json.dumps(dict(stage='FUNCTIONAL_FULL_V_ARTIFACTS_CPU', arm=context['arm'],
                        visited=counts['visited'], expected=context['count'], decoded=counts['decoded'])), flush=True)
            table.flush()
        except spatial.OutputBudgetExceeded:
            output_exhausted = True
        except Exception as exception:
            source_errors.append(type(exception).__name__ + ': ' + str(exception)[:200])
            try:
                table.flush()
            except spatial.OutputBudgetExceeded:
                output_exhausted = True
    duplicate_groups = [dict(decoded_RGB_pixel_sha256=k, query_count=v[0], distinct_file_sha256_count=len(v[1]))
                        for k, v in sorted(pixels.items()) if v[0] > 1]
    if not output_exhausted:
        groups = spatial.ChunkedCSV(budget, 'pixel_duplicate_groups.csv.gz',
            ['decoded_RGB_pixel_sha256', 'query_count', 'distinct_file_sha256_count'])
        try:
            for row in duplicate_groups:
                groups.add(row)
            groups.flush()
        except spatial.OutputBudgetExceeded:
            output_exhausted = True
    for path, expected in context['pins'].items():
        if not Path(path).is_file() or spatial.sha_file(path) != expected:
            source_errors.append('Immutable metadata/source changed: ' + path)
    complete = (table.rows_written == context['count'] == counts['trusted'] and not failures
                and not source_errors and not exhausted and not output_exhausted)
    summary = dict(schema=SCHEMA, utc=spatial.utc_now(), status=('CPU_TINY_FIXTURE_ONLY' if context['fixture'] else
        'COMPLETE_FULL_V_ARTIFACT_AUDIT' if complete else 'FULL_V_ARTIFACT_AUDIT_RECORDED_WITH_FAILURES_OR_UNKNOWN'),
        arm=context['arm'], candidate=context['recipe']['candidate'], recipe_core_sha256=context['core'],
        complete_technical_coverage=complete, actual_full_V16600_verified=complete and not context['fixture'],
        expected_queries=context['count'], full_denominator=context['count'], visited_records=counts['visited'],
        identity_verified=counts['identity_verified'], canonical_path_verified=counts['canonical_path_verified'],
        generated_files_attempted=counts['images_attempted'], decoded_512_RGB=counts['decoded'],
        hash_verified_decoded_without_other_failures=counts['trusted'], spatial_flags_unknown=context['count'] - counts['decoded'],
        per_query_rows_written=table.rows_written, per_query_rows_unwritten=context['count'] - table.rows_written,
        failures=dict(failures), source_errors=source_errors, decoded_flag_counts={k: flags[k] for k in FLAGS},
        hash_verified_flag_counts={k: trusted_flags[k] for k in FLAGS},
        decoded_flag_fractions_full_denominator={k: flags[k] / context['count'] for k in FLAGS},
        decoded_pixel_duplicate_groups=len(duplicate_groups), duplicate_group_queries=sum(r['query_count'] for r in duplicate_groups),
        unique_actual_file_SHA_count=len(files), duplicate_file_SHA_groups=sum(v > 1 for v in files.values()),
        fixed_flags=spatial.REGISTRATION_FLAGS, fixed_continuous_diagnostics=spatial.REGISTRATION_DIAGNOSTICS,
        near_spatial_constant_max_std=spatial.NEAR_THRESHOLD, source_sha256=spatial.sha_file(__file__),
        measurement_source_sha256=MEASUREMENT_SOURCE_SHA256, immutable_metadata_sha256=context['pins'],
        table_sha256=spatial.sha_file(output / 'per_query.csv.gz') if (output / 'per_query.csv.gz').exists() else None,
        budget=dict(maximum_seconds=max_seconds, seconds=time.monotonic() - started, maximum_output_bytes=max_output_bytes,
            fixed_unmeasured_full_V_runtime_bound=True, metadata_preflight_seconds=context['preflight_seconds'],
            output_bytes_before_summary=budget.used, time_budget_exhausted=exhausted,
            output_budget_exhausted=output_exhausted, memory_preflight=memory,
            own_arm_metadata_budget_bytes=context['entry']['metadata_budget_bytes'],
            metadata_bytes_before_audit=used_metadata, full_output_cap_charged_to_sealed_metadata_budget=True),
        read_policy=dict(target_pixels_read=False, semantic_scores_read=False, model_or_EEG_tensors_read=False,
            generated_files_only_for_pixel_measurements=True, changed_input_files=False, GPU_used=False,
            automatic_retry=False, image_resampling=False),
        interpretation=dict(engineering_flags_only=True, failed_queries_remain_in_full_denominator=True,
            unprocessed_queries_remain_unknown=True, perceptual_noninferiority_established=False,
            purple_noise_rate_identified=False, correct_objects_established=False, component_gain_established=False,
            pixel_duplicates_do_not_by_themselves_prove_file_reuse_or_bad_quality=True))
    budget.json('summary.json', summary)
    require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= max_output_bytes, 'Audit output exceeded fixed budget')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--max-output-bytes', type=int, default=MAX_OUTPUT_BYTES)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    context = prepare(args.recipe, time.monotonic() + 120)
    if args.preflight_only:
        print(json.dumps(dict(status='COMPLETE_V2_ARM_METADATA_GUARD_PASSED_NO_PIXELS_READ',
                             arm=context['arm'], queries=context['count'], recipe_core_sha256=context['core'])))
        return
    result = audit_context(context, args.output, max_seconds=args.max_seconds, max_output_bytes=args.max_output_bytes)
    print(json.dumps({k: v for k, v in result.items() if k not in ('immutable_metadata_sha256', 'fixed_flags', 'fixed_continuous_diagnostics')}), flush=True)


if __name__ == '__main__':
    main()
