"""Tiny temporary PIL fixtures for the functional-v2 artifact adapter only."""
import argparse
import gzip
import csv
import hashlib
import json
import random
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import audit_BrainHIVE_functional_full_V_artifacts_CPU_v1 as audit


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True, indent=2) + '\n', encoding='utf-8')


def jsonl(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(v, allow_nan=True) + '\n' for v in values), encoding='utf-8')


def pattern(seed):
    y, x = np.indices((512, 512))
    return np.stack(((x + seed) % 256, (y * 3 + seed * 11) % 256, (x + y * 7 + seed * 13) % 256), -1).astype('uint8')


def fixture(root, specifications, arm='PN'):
    root, n = Path(root), len(specifications)
    repo, work = root / 'repo', root / 'work'
    folder = work / 'images' / ('protected_R4_functional_alignment_v2_' + arm)
    image_root = root / 'images'
    training_root = work / 'components/functional_alignment_v2' / arm
    inputs_path, protocol_path = work / 'data/generation_inputs/V.json', repo / 'configs/split_manifest.json'
    target_path, gallery_path = work / 'data/generation_inputs/V_targets.jsonl', work / 'gallery/V/manifest.jsonl'
    queries = [dict(stimulus_id='stimulus_' + str(i) + '.jpg', array_row=100 + i) for i in range(n)]
    inputs = dict(expected_queries=n, subjects=['sub-01'], repeat_protocol='R4', target_image_or_class_fields_present=False, inputs=queries)
    protocol = dict(version='tiny_fixture_v1', image_rows=dict(V=[q['array_row'] for q in queries]))
    targets = [dict(**q, concept_id='concept_' + str(i), image_sha256=hashlib.sha256(('target' + str(i)).encode()).hexdigest(),
                    image_zip=str(root / 'not_opened_targets.zip'), image_member='member_' + str(i)) for i, q in enumerate(queries)]
    gallery = [dict(stimulus_id=t['stimulus_id'], concept_id=t['concept_id'], image_sha256=t['image_sha256'],
                    image_path=str(root / 'never_read_target' / t['stimulus_id'])) for t in targets]
    write(inputs_path, inputs)
    write(protocol_path, protocol)
    jsonl(target_path, targets)
    jsonl(gallery_path, gallery)
    recipe_path, admission_path, plan_path = root / 'recipe.json', root / 'admission.json', root / 'plan.json'
    name = folder.name
    recipe = dict(schema='BrainHIVE_functional_alignment_v2_recipe', arm=arm, candidate='functional_alignment_v2_' + arm,
        training_seed=1729, generator=dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1),
        admission_receipt=str(admission_path), admission_sha256=None, output_directory=str(training_root),
        generation_gallery_manifest=str(gallery_path), input_sha256={str(p): audit.spatial.sha_file(p)
            for p in (inputs_path, protocol_path, target_path, gallery_path)},
        source_sha256={'generate_BrainHIVE_functional_alignment_V_v2.py': 'c' * 64})
    core = audit.core_sha(recipe)
    entry = dict(arm=arm, candidate=recipe['candidate'], recipe_core_sha256=core, run_name=name,
        metadata_root=str(folder), inference_manifest=str(folder / 'inference_manifest.jsonl'),
        metadata_budget_bytes=150000000,
        lifecycle_source_sha256={Path(audit.__file__).name: audit.spatial.sha_file(audit.__file__),
                                Path(audit.spatial.__file__).name: audit.spatial.sha_file(audit.spatial.__file__)},
        image_segments=[dict(start=0, stop=n, root=str(image_root))])
    image_root.mkdir(parents=True)
    write(plan_path, dict(arms=[entry]))
    admission = dict(candidate=recipe['candidate'], recipe_core_sha256=core,
        status='FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMITTED',
        four_arm_lifecycle_plan=dict(path=str(plan_path), sha256=audit.spatial.sha_file(plan_path)))
    write(admission_path, admission)
    recipe['admission_sha256'] = audit.spatial.sha_file(admission_path)
    write(recipe_path, recipe)
    assert core == audit.core_sha(recipe)
    parent = dict(checkpoint_path=str(root / 'never_read_parent.pt'), checkpoint_sha256='a' * 64,
        condition_cache_path=str(root / 'never_read_conditions.pt'), condition_cache_sha256='b' * 64)
    model = dict(candidate=recipe['candidate'], arm=arm, run_name=name, recipe_core_sha256=core,
        source_sha256=recipe['source_sha256'], component_checkpoint_sha256='d' * 64,
        component_checkpoint_path=str(training_root / 'component_final.pt'),
        generation_source_sha256='c' * 64, parent_bindings={'sub-01': parent}, resolution=512, steps=4,
        CFG=0, IP_scale=1, images_per_query=1, normal_inference_target_access=False,
        teacher_or_rank_pair_used_at_inference=False)
    calibration = dict(kind='Dfit_only_global_strength_calibration', strengths=[0., .25, .5, .75, 1.], selected_strength=1.)
    model['calibration'] = calibration
    write(training_root / 'calibration.json', calibration)
    write(folder / 'model_receipt.json', model)
    write(training_root / 'completion.json', dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE',
        candidate=recipe['candidate'], arm=arm, updates=24807, seen_queries=396900, passes=3,
        parameters=525568, recipe_core_sha256=core, checkpoint_sha256='d' * 64,
        calibration_sha256=audit.spatial.sha_file(training_root / 'calibration.json')))
    rows, joined = [], []
    for index, (query, specification) in enumerate(zip(queries, specifications)):
        path = image_root / name / 'sub-01' / (Path(query['stimulus_id']).stem + '.webp')
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = pattern(index + 1)
        if specification in ('normal', 'wrong_sha', 'wrong_identity', 'wrong_path', 'wrong_target', 'nonfinite'):
            pass
        elif specification == 'constant':
            pixels[:] = [255, 0, 255]  # Colored constant: global RGB std alone misses it.
        elif specification in ('near', 'above_near'):
            pixels[:] = 127
            pixels[:, 1::2] = 129 if specification == 'near' else 130
        elif specification == 'duplicate':
            pixels = pattern(1)
        elif specification not in ('corrupt', 'missing', 'grayscale', 'dimensions'):
            raise ValueError(specification)
        if specification == 'missing':
            file_sha = 'e' * 64
        elif specification == 'corrupt':
            path.write_bytes(b'tiny intentionally invalid image')
            file_sha = audit.spatial.sha_file(path)
        else:
            image = Image.fromarray(pixels)
            if specification == 'grayscale':
                image = image.convert('L')
            elif specification == 'dimensions':
                image = image.resize((32, 32))
            # Different encoded bytes with identical decoded RGB test pixel hashing.
            image.save(path, format='PNG' if specification in ('duplicate', 'grayscale') else 'WEBP', lossless=True)
            file_sha = audit.spatial.sha_file(path)
        seed = int(hashlib.sha256(('1729|sub-01|' + query['stimulus_id'] + '|R4|' + protocol['version']).encode()).hexdigest()[:16], 16) % (2**63 - 1)
        row = dict(query_id='sub-01/' + query['stimulus_id'] + '/R4', subject='sub-01', stimulus_id=query['stimulus_id'],
            array_row=query['array_row'], global_query_index=index, EEG_source_stimulus_id=query['stimulus_id'], image_seed=seed,
            candidate=recipe['candidate'], arm=arm, run_name=name, repeat_protocol='R4', intervention='correct',
            recipe_core_sha256=core, component_checkpoint_sha256='d' * 64,
            component_checkpoint_path=str(training_root / 'component_final.pt'),
            model_receipt_sha256=audit.spatial.sha_file(folder / 'model_receipt.json'), generation_source_sha256='c' * 64,
            target_or_visual_features_read_for_condition=False, oracle_diagnostic_only=False,
            teacher_or_negative_used_at_inference=False, selected_best_seed=False, lossless_pixel_roundtrip=True,
            UNet_NFE=4, IP_conditioned_attention_calls=280, IP_value_calls=280, VAE_decoder_calls=1,
            generated_path=str(path), generated_sha256=file_sha, parent_checkpoint_path=parent['checkpoint_path'],
            parent_checkpoint_sha256=parent['checkpoint_sha256'], condition_cache_path=parent['condition_cache_path'],
            condition_cache_sha256=parent['condition_cache_sha256'], condition_sha256='f' * 64,
            initial_UNet_input_sha256='1' * 64, actual_A_input_fp16_sha256='2' * 64,
            condition_norm=32., component_correction_RMS=.1, component_effective_radius=.25, component_gate=1.,
            actual_A_input_fp16_norm=32., VAE_float_min=-1., VAE_float_max=1.)
        if specification == 'wrong_sha':
            row['generated_sha256'] = '0' * 64
        elif specification == 'wrong_identity':
            row['query_id'] = 'sub-01/incorrect_stimulus.jpg/R4'
        elif specification == 'wrong_path':
            row['generated_path'] = str(root / 'another_arm' / 'not_read.webp')
        elif specification == 'nonfinite':
            row['VAE_float_min'] = float('nan')
        target, actual = targets[index], gallery[index]
        join = row | dict(concept_id=target['concept_id'], target_image_sha256=target['image_sha256'],
            target_image_path=actual['image_path'], score_target_path=actual['image_path'], target_image_zip=target['image_zip'],
            target_image_member=target['image_member'], target_mapping_attached_after_inference=True)
        if specification == 'wrong_target':
            join['score_target_path'] = str(root / 'wrong_target.jpg')
        rows.append(row)
        joined.append(join)
    jsonl(folder / 'inference_manifest.jsonl', rows)
    jsonl(folder / 'manifest.jsonl', joined)
    inf_complete = dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_ACTUAL_IMAGES_SAVED_NO_TARGET_ACCESS',
        candidate=recipe['candidate'], arm=arm, run_name=name, queries=n, recipe_core_sha256=core,
        normal_inference_target_access=False, images_per_query=1, UNet_NFE_per_image=4, IP_calls_per_image=280,
        selected_best_seed=False, generated_images_selected_or_discarded=False,
        inference_manifest_sha256=audit.spatial.sha_file(folder / 'inference_manifest.jsonl'),
        model_receipt_sha256=audit.spatial.sha_file(folder / 'model_receipt.json'))
    write(folder / 'inference_completion.json', inf_complete)
    proof = dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_TARGET_AND_QUERY_IDENTITIES_JOINED', candidate=recipe['candidate'],
        arm=arm, queries=n, manifest_sha256=audit.spatial.sha_file(folder / 'manifest.jsonl'),
        gallery_manifest_sha256=audit.spatial.sha_file(gallery_path), target_metadata_sha256=audit.spatial.sha_file(target_path))
    write(folder / 'pairing_audit.json', proof)
    complete = dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_IMAGES_AND_TARGET_JOIN_COMPLETE', candidate=recipe['candidate'],
        arm=arm, queries=n, role='full', split='V', intervention='correct', images_per_query=1, selected_best_seed=False,
        recipe_core_sha256=core, manifest_sha256=proof['manifest_sha256'],
        inference_completion_sha256=audit.spatial.sha_file(folder / 'inference_completion.json'),
        pairing_audit_sha256=audit.spatial.sha_file(folder / 'pairing_audit.json'))
    write(folder / 'completion.json', complete)
    options = dict(root=root, repo=repo, work=work, count=n, subjects=1)
    return recipe_path, options, folder


def load_context(fixture_result):
    recipe, options, _ = fixture_result
    return audit.prepare(recipe, time.monotonic() + 20, _fixture=options)


def estimate_full_csv():
    """Real writer/schema, full cardinality, high-entropy synthetic metadata only.

    No actual V rows or images and no large local output file are read/written.
    Long255-byte filename components and extra200-byte errors overestimate normal
    rows. This is a forecast, not a proof of all future compression ratios.
    """
    class CountingBudget:
        def __init__(self):
            self.used = 0
            self.raw_bytes = 0
        def append(self, name, data, summary=False):
            self.used += len(data)
            self.raw_bytes += len(gzip.decompress(data))
    sink = CountingBudget()
    table = audit.spatial.ChunkedCSV(sink, 'not_written.csv.gz', audit.FIELDS)
    generator = random.Random(1729)
    max_path = 0
    for index in range(16600):
        subject = 'sub-' + str(index // 1660 + 1).zfill(2)
        stem = hashlib.shake_256(str(index % 1660).encode()).hexdigest(125)  # 250 + '.webp' =255 bytes.
        stimulus = stem + '.jpg'
        path = ('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/images/'
                'BrainHIVE_functional_alignment_v2_PN/protected_R4_functional_alignment_v2_PN/' + subject + '/' + stem + '.webp')
        max_path = max(max_path, len(path))
        qid = subject + '/' + stimulus + '/R4'
        row = {key: generator.random() for key in audit.FIELDS}
        row.update(source_line_number=index + 1, source_manifest_sha256='f' * 64,
            query_id=qid, expected_query_id=qid, subject=subject, stimulus_id=stimulus, generated_path=path,
            expected_generated_path=path, manifest_generated_sha256=hashlib.sha256(('manifest' + str(index)).encode()).hexdigest(),
            actual_file_sha256=hashlib.sha256(('actual' + str(index)).encode()).hexdigest(),
            decoded_RGB_pixel_sha256=hashlib.sha256(('pixels' + str(index)).encode()).hexdigest(),
            status='hash_mismatch', error=hashlib.shake_256(('error' + str(index)).encode()).hexdigest(100),
            hash_verified=False, decoded_512_RGB=True, width=512, height=512, original_mode='RGB',
            exact_spatial_constant=False, near_spatial_constant=False, legacy_global_blank_proxy=False,
            global_query_index=index, arm='PN', candidate='functional_alignment_v2_PN',
            identity_verified=True, canonical_path_verified=True)
        table.add(row)
    table.flush()
    require_bytes = sink.used + (1 << 20)  # Dup-group table, provenance summary and writer reserve allowance.
    assert require_bytes < audit.MAX_OUTPUT_BYTES
    return dict(rows=16600, fields=len(audit.FIELDS), maximum_synthetic_generated_path_bytes=max_path,
        filename_component_bytes=255, per_query_uncompressed_CSV_bytes=sink.raw_bytes, per_query_gzip_bytes=sink.used,
        additional_summary_and_duplicate_table_allowance_bytes=1 << 20,
        total_forecast_bytes=require_bytes, fixed_maximum_output_bytes=audit.MAX_OUTPUT_BYTES,
        actual_V_metadata_or_pixels_read=False, local_large_file_written=False,
        compression_ratio_is_a_forecast_not_guarantee=True,
        formal_metadata_budget_must_include_full_fixed_output_cap=True)


def run_checks():
    tests = {}
    with tempfile.TemporaryDirectory(prefix='brainhive_artifact_tiny_') as temporary:
        base = Path(temporary)
        good = fixture(base / 'good', ['normal', 'constant', 'near', 'above_near', 'duplicate'])
        context = load_context(good)
        output = good[1]['work'] / 'audit/good_audit'
        before = {p: audit.spatial.sha_file(p) for p in good[2].rglob('*') if p.is_file()}
        result = audit.audit_context(context, output)
        with gzip.open(output / 'per_query.csv.gz', 'rt', newline='') as stream:
            records = list(csv.DictReader(stream))
        assert result['complete_technical_coverage'] and result['full_denominator'] == len(records) == 5
        assert not result['actual_full_V16600_verified'] and result['status'] == 'CPU_TINY_FIXTURE_ONLY'
        assert records[1]['exact_spatial_constant'] == records[1]['near_spatial_constant'] == 'True'
        assert records[1]['legacy_global_blank_proxy'] == 'False'
        assert records[2]['near_spatial_constant'] == 'True' and records[2]['exact_spatial_constant'] == 'False'
        assert records[3]['near_spatial_constant'] == 'False'
        assert result['decoded_pixel_duplicate_groups'] == 1 and result['duplicate_group_queries'] == 2
        assert result['duplicate_file_SHA_groups'] == 0
        assert all(audit.spatial.sha_file(p) == digest for p, digest in before.items())
        tests['normal_colored_constant_exact_threshold_above_threshold_and_pixel_duplicate'] = True
        tests['different_encoded_file_SHA_same_decoded_pixels_detected'] = True
        tests['input_metadata_and_images_not_changed'] = True
        tests['fixture_cannot_claim_formal_16600_coverage'] = True
        bad_specs = ['corrupt', 'missing', 'wrong_sha', 'wrong_identity', 'wrong_path', 'wrong_target', 'grayscale', 'dimensions', 'nonfinite']
        bad = fixture(base / 'bad', bad_specs)
        calls = []
        original = audit.spatial.inspect_generated
        def observed(record, *arguments):
            calls.append(record['query_id'])
            return original(record, *arguments)
        audit.spatial.inspect_generated = observed
        try:
            result = audit.audit_context(load_context(bad), bad[1]['work'] / 'audit/bad_audit')
        finally:
            audit.spatial.inspect_generated = original
        assert result['full_denominator'] == result['per_query_rows_written'] == 9
        assert not result['complete_technical_coverage'] and result['spatial_flags_unknown'] == 8
        for key in ('corrupt_or_decode_failure', 'missing', 'hash_mismatch', 'inference_identity_failure',
                    'canonical_generated_path_failure', 'target_metadata_identity_failure', 'RGB_mode_failure',
                    'dimension_failure', 'nonfinite_generation_numeric_metadata'):
            assert result['failures'].get(key) == 1, (key, result['failures'])
        assert len(calls) == 5  # Four invalid metadata/path rows never reach the file reader.
        tests['corrupt_missing_wrong_SHA_and_no_colour_or_dimension_repair'] = True
        tests['wrong_identity_path_target_and_nonfinite_metadata_not_read_as_valid_images'] = True
        tests['all_failure_queries_kept_in_denominator_and_table'] = True
        malformed = fixture(base / 'malformed', ['normal', 'normal', 'normal'])
        folder = malformed[2]
        for filename in ('manifest.jsonl', 'inference_manifest.jsonl'):
            lines = (folder / filename).read_text(encoding='utf-8').splitlines()
            lines[1] = '{invalid_JSON'
            (folder / filename).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        inf_path, proof_path, complete_path = (folder / n for n in ('inference_completion.json', 'pairing_audit.json', 'completion.json'))
        value = json.loads(inf_path.read_text(encoding='utf-8'))
        value['inference_manifest_sha256'] = audit.spatial.sha_file(folder / 'inference_manifest.jsonl')
        write(inf_path, value)
        value = json.loads(proof_path.read_text(encoding='utf-8'))
        value['manifest_sha256'] = audit.spatial.sha_file(folder / 'manifest.jsonl')
        write(proof_path, value)
        value = json.loads(complete_path.read_text(encoding='utf-8'))
        value.update(manifest_sha256=audit.spatial.sha_file(folder / 'manifest.jsonl'),
                     inference_completion_sha256=audit.spatial.sha_file(inf_path), pairing_audit_sha256=audit.spatial.sha_file(proof_path))
        write(complete_path, value)
        result = audit.audit_context(load_context(malformed), malformed[1]['work'] / 'audit/malformed_audit')
        assert result['full_denominator'] == result['per_query_rows_written'] == 3 and result['spatial_flags_unknown'] == 1
        assert result['failures']['invalid_manifest_identity'] == 1 and result['decoded_512_RGB'] == 2
        tests['malformed_identity_row_kept_between_other_rows_without_changing_denominator'] = True
        early = fixture(base / 'early', ['normal'])
        path = early[2] / 'completion.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['queries'] = 0
        write(path, value)
        try:
            load_context(early)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete generation incorrectly admitted')
        tests['incomplete_generation_rejected_before_pixel_import_or_output'] = True
        for source_name in (Path(audit.__file__).name, Path(audit.spatial.__file__).name):
            altered = fixture(base / ('altered_source_' + source_name), ['normal'])
            recipe_path = altered[0]
            recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
            admission_path = Path(recipe['admission_receipt'])
            admission = json.loads(admission_path.read_text(encoding='utf-8'))
            plan_path = Path(admission['four_arm_lifecycle_plan']['path'])
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            plan['arms'][0]['lifecycle_source_sha256'][source_name] = '0' * 64
            write(plan_path, plan)
            admission['four_arm_lifecycle_plan']['sha256'] = audit.spatial.sha_file(plan_path)
            write(admission_path, admission)
            recipe['admission_sha256'] = audit.spatial.sha_file(admission_path)
            write(recipe_path, recipe)
            try:
                load_context(altered)
            except ValueError as error:
                assert 'audit source mismatch' in str(error)
            else:
                raise AssertionError('Unbound/tampered audit source was admitted')
        tests['sealed_lifecycle_expected_auditor_and_measurement_SHA_required'] = True
        short = fixture(base / 'short', ['normal', 'constant'])
        result = audit.audit_context(load_context(short), short[1]['work'] / 'audit/timeout_audit', max_seconds=1e-12)
        assert result['full_denominator'] == result['per_query_rows_written'] == result['spatial_flags_unknown'] == 2
        assert result['failures']['not_processed_time_limit'] == 2 and result['generated_files_attempted'] == 0
        tests['time_limit_keeps_unknown_rows_and_full_denominator'] = True
        for arm in ('O', 'P', 'S'):
            arm_fixture = fixture(base / ('arm_' + arm), ['normal'], arm)
            assert load_context(arm_fixture)['arm'] == arm
        tests['all_four_actual_candidate_arm_names_admitted_in_fixture_schema'] = True
        assert 'torch' not in sys.modules
        tests['no_torch_or_GPU_module_imported'] = True
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise FileExistsError('Preserve previous CPU receipt')
    tests = run_checks()
    size_forecast = estimate_full_csv()
    root = Path(__file__).resolve().parent
    report = dict(status='FUNCTIONAL_FULL_V_ARTIFACT_CPU_TINY_CONTRACTS_PASSED_NO_ACTUAL_ARM_SCAN',
        utc=audit.spatial.utc_now(), tests=tests, full16600_synthetic_schema_output_forecast=size_forecast,
        source_sha256={n: audit.spatial.sha_file(root / n) for n in (
            Path(__file__).name, 'audit_BrainHIVE_functional_full_V_artifacts_CPU_v1.py',
            'audit_BrainHIVE_spatial_image_integrity_CPU_v1.py')},
        real_EEG_models_or_V_images_read=False, real_semantic_scores_read=False,
        GPU_used=False, actual_arm_audit_completed=False, new_training=False,
        local_fixture_directory_removed=True, interpretation='Software contracts only; no perceptual NI, object success or component gain evidence.')
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
