"""Full2000 SYNTHETIC metadata fixtures, never actual benchmark/image evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import time

import brainhive_official_benchmark_contract_v1 as C

ROOT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def named_sha(name):
    return sha(('SYNTHETIC_METADATA_ONLY|' + name).encode())


def ref(name):
    return dict(path='/root/synthetic_official_R80_contract_ONLY/' + name, sha256=named_sha(name))


def protocol_fixture():
    return dict(schema=C.PROTOCOL_SCHEMA, training_seed=1729, subjects=list(C.SUBJECTS), repeat_protocol='R80',
        physical_repeat_indices=list(range(80)), permutation_policy=C.PERMUTATION_POLICY,
        images_per_query=1, best_of_N=False, official200_previously_exposed=True, independent_H=False,
        split_version='eeg_full10_concept_protocol_v1', inputs=[dict(stimulus_id=f'synthetic_{i:03d}.jpg', array_row=i) for i in range(200)],
        generator=dict(model='SDXL-Turbo', resolution=512, steps=4, CFG=0, IP_scale=1,
            scheduler='EulerAncestralDiscreteScheduler', timestep_spacing='trailing'),
        generator_receipt=ref('NOT_REAL_generator_receipt.json'), identity_contract_sha256=sha(Path(C.__file__).read_bytes()))


def arm_fixture(protocol, intervention='correct', candidate='synthetic_PN', native=False):
    bindings = {}
    for number, subject in enumerate(C.SUBJECTS):
        bindings[subject] = {key: ref(subject + '/NOT_REAL_' + key + '.json') for key in
            ('EEG_R80_source', 'parent_checkpoint', 'parent_registration', 'parent_affine', 'condition_cache', 'condition_export')}
        bindings[subject]['condition_cache_model_sha256'] = bindings[subject]['parent_checkpoint']['sha256']
        bindings[subject].update(condition_shape=[200, 1024], condition_dtype='float32', normalization=C.NORMALIZATION,
            additional_L2_normalization=False, rows=[dict(query, cache_row=(i + 19) % 200,
                condition_sha256=C.observe_condition_bytes(struct.pack('<f', float(number * 200 + i + 1)) + bytes(4092)))
                for i, query in enumerate(protocol['inputs'])])
    return dict(schema=C.ARM_SCHEMA, protocol_sha256=C.core_sha(protocol), intervention=intervention,
        run_name=candidate + '_' + intervention, candidate=candidate, image_root='/root/synthetic_official_R80_contract_ONLY/images',
        producer_source_sha256=named_sha('NOT_REAL_future_generator.py'), subject_bindings=bindings,
        component=dict(kind='native_parent' if native else 'bounded_condition_adapter',
            checkpoint=None if native else ref(candidate + '/NOT_REAL_component.pt'),
            calibration=None if native else ref(candidate + '/NOT_REAL_calibration.json')))


def inference_fixture(protocol, arm):
    C.validate_arm(protocol, arm)
    cached = C.core_sha(protocol), C.core_sha(arm), C.permutation(protocol)
    rows = []
    for subject in C.SUBJECTS:
        for i in range(200):
            row = C.expected_identity(protocol, arm, subject, i, _cached=cached)
            # These buffers/hashes are SYNTHETIC. No real image or UNet exists.
            noisy = struct.pack('<e', float(i + 1)) + bytes(32766)
            row.update(actual_A_input_fp16_sha256=C.ZERO_A_FP16_SHA if arm['intervention'] == 'missing_raw_condition' else named_sha('synthetic_A|' + row['query_id']),
                initial_UNet_input_sha256=sha(noisy), initial_UNet_input_dtype='float16', initial_UNet_input_shape=[1, 4, 64, 64],
                initial_UNet_input_capture='first_native_UNet_forward_pre_hook', generated_sha256=named_sha('not_an_image|' + arm['run_name'] + row['query_id']))
            rows.append(row)
    assert len(rows) == 2000 and set(rows[0]) == C.INFERENCE_KEYS
    return rows


def targets_fixture(protocol):
    targets, gallery = [], []
    for i, query in enumerate(protocol['inputs']):
        concept = 'synthetic_concept_' + str(i).zfill(3)
        targets.append(dict(dataset_split='test', array_row=str(i), stimulus_id=query['stimulus_id'], concept_id=concept,
            local_concept_id='local_' + str(i), image_zip='/root/synthetic_official_R80_contract_ONLY/NOT_REAL_testing_images.zip',
            image_member='testing_images/local_' + str(i) + '/' + query['stimulus_id'], image_bytes='123', image_sha256=named_sha('not_real_target_' + str(i)),
            alignment='author_release_order_plus_metadata_and_zip_receipt', independent_event_label_replay='False'))
        gallery.append(dict(stimulus_id=query['stimulus_id'], image_path='/root/synthetic_official_R80_contract_ONLY/gallery/' + query['stimulus_id'],
            image_sha256=targets[-1]['image_sha256'], concept_id=concept))
    assert all(set(r) == {'stimulus_id', 'image_path', 'image_sha256', 'concept_id'} for r in gallery)
    return targets, gallery


def write_manifest(directory, name, rows):
    filename = directory / (name + '.jsonl')
    raw = b''.join(C.canonical(row) + b'\n' for row in rows)
    assert not filename.exists() and len(raw) < C.MAX_METADATA_BYTES
    filename.write_bytes(raw)
    return dict(path=str(filename.resolve()), sha256=sha(raw)), len(raw)


def score_fixture(joined, family):
    registry = C.metric_registry(family)
    result = []
    for row in joined:
        score = {key: row[key] for key in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'generated_path', 'score_target_path', 'EEG_source_stimulus_id')}
        score.update(image_seed=str(row['image_seed']), target_image_path=row['score_target_path'])
        score.update({key: '0.5' for key in registry['primary'] | registry['extra_code_metric']})
        if family == 'paper': score.update(generated_sha256=row['generated_sha256'], target_sha256=row['target_image_sha256'])
        result.append(score)
    return result


def reject(call, message):
    try:
        call()
    except (ValueError, KeyError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError('Invalid identity accepted: ' + message)


def actual_schema_keys():
    """Inspect saved OLD schema keys only; no metric values enter tests."""
    filename = ROOT / 'audit/gallery_index_actual_20261022_v1/actual_old_schema_samples.json'
    samples = json.loads(filename.read_bytes())
    gallery = next(value for name, value in samples.items() if name.endswith('/gallery/V/manifest.jsonl'))
    origins = next(value for name, value in samples.items() if name.endswith('/data/generation_inputs/V_targets.jsonl'))
    native = next(value for name, value in samples.items() if '/images/protected_R4_parent_bounded_v1/' in name)
    require_keys = {'query_id', 'subject', 'stimulus_id', 'array_row', 'image_seed', 'condition_sha256', 'generated_path', 'generated_sha256'}
    assert require_keys <= set(native) and 'initial_UNet_input_sha256' not in native
    assert set(gallery) == {'stimulus_id', 'image_path', 'image_sha256', 'concept_id'} and 'array_row' not in gallery
    assert 'array_row' in origins and type(origins['array_row']) is str
    return native, dict(path=str(filename), sha256=sha(filename.read_bytes()),
        gallery_four_field_schema=True, array_row_owned_by_target_origin=True, legacy_missing_actual_first_UNet_observation=True,
        actual_metric_values_used=False)


def check():
    started = time.monotonic(); checks = []
    protocol = protocol_fixture(); targets, gallery = targets_fixture(protocol)
    arms = {key: arm_fixture(protocol, key) for key in C.INTERVENTIONS}
    inference = {key: inference_fixture(protocol, arm) for key, arm in arms.items()}
    for key in C.INTERVENTIONS:
        result = C.validate_inference(protocol, arms[key], inference[key])
        assert result['queries'] == 2000 and result['payloads_verified'] is False and result['execution_admitted'] is False
    checks.append('full2000_each_of_correct_permuted_missing_raw_condition_metadata_positive')
    observations = C.observe_tensor_bytes(bytes(2048), 'float16', [1, 1024], 'native_A_forward_pre_hook')
    assert observations['sha256'] == C.ZERO_A_FP16_SHA and C.observe_condition_bytes(bytes(4096)) == C.ZERO_RAW_SHA
    noise = C.observe_tensor_bytes(bytes(32768), 'float16', [1, 4, 64, 64], 'first_native_UNet_forward_pre_hook')
    assert noise['sha256'] == sha(bytes(32768)) != named_sha('1729')
    reject(lambda: C.observe_tensor_bytes(bytes(2), 'float16', [1, 1024], 'native_A_forward_pre_hook'), 'byte count')
    reject(lambda: C.observe_condition_bytes(struct.pack('<f', float('nan')) + bytes(4092)), 'Nonfinite')
    checks.append('actual_supplied_tensor_byte_hash_helpers_shape_dtype_length_and_finite_guards_no_hook_certification')
    legacy, evidence = actual_schema_keys()
    reject(lambda: C.validate_inference(protocol, arms['correct'], [legacy] * 2000), 'missing evidence or forbidden target')
    fake_upgrade = dict(legacy, initial_UNet_input_sha256=named_sha(str(legacy['image_seed'])))
    reject(lambda: C.validate_inference(protocol, arms['correct'], [fake_upgrade] * 2000), 'missing evidence or forbidden target')
    checks.append('actual_old_schema_missing_noise_and_seed_hash_fake_upgrade_refused')
    joined, sizes = {}, {}
    with tempfile.TemporaryDirectory(prefix='HIVE_R80_metadata_contract_') as temporary:
        directory = Path(temporary)
        for key in C.INTERVENTIONS:
            binding, sizes[key] = write_manifest(directory, key, inference[key])
            joined[key], report = C.join_targets(protocol, arms[key], binding, targets, gallery)
            assert report['target_join_after_full_inference'] and report['metadata_only']
            assert C.validate_joined(protocol, arms[key], joined[key], targets, gallery)['queries'] == 2000
        checks.append('all2000_saved_blind_JSONL_SHA_before_separate_target_join_real_four_field_gallery_schema')
        class ForbiddenTargets:
            def __len__(self): raise AssertionError('Targets accessed before inference closure')
        partial, _ = write_manifest(directory, 'partial', inference['correct'][:-1])
        reject(lambda: C.join_targets(protocol, arms['correct'], partial, ForbiddenTargets(), ForbiddenTargets()), 'denominator required before target access')
        reject(lambda: C.join_targets(protocol, arms['correct'], dict(partial, sha256='0' * 64), targets, gallery), 'manifest SHA differs')
        checks.append('partial1999_and_manifest_SHA_drift_refused_before_any_target_iteration')
    for key in ('permuted', 'missing_raw_condition'):
        result = C.validate_pair(protocol, arms['correct'], joined['correct'], arms[key], joined[key], targets, gallery, 'EEG_intervention')
        assert result['metadata_pairing_complete'] and result['per_condition_costs_must_be_reported_separately']
    other = arm_fixture(protocol, candidate='synthetic_O')
    other_blind = inference_fixture(protocol, other)
    other_joined = C._joined_rows(protocol, other, other_blind, targets, gallery)
    C.validate_pair(protocol, other, other_joined, arms['correct'], joined['correct'], targets, gallery, 'matched_methods')
    native = arm_fixture(protocol, candidate='synthetic_parent', native=True)
    C.validate_inference(protocol, native, inference_fixture(protocol, native))
    checks.append('matched_methods_own_different_component_calibration_and_native_None_bindings_plus_same_method_interventions')
    bad = copy.deepcopy(inference['correct']); bad[-1] = copy.deepcopy(bad[0])
    reject(lambda: C.validate_inference(protocol, arms['correct'], bad), 'binding differs')
    for field in ('EEG_source_stimulus_id', 'EEG_R80_source_sha256', 'condition_cache_row', 'condition_sha256', 'condition_cache_sha256',
            'parent_checkpoint_sha256', 'parent_registration_sha256', 'parent_affine_sha256', 'component_checkpoint_sha256', 'calibration_sha256', 'image_seed'):
        bad = copy.deepcopy(inference['correct'])
        bad[0][field] = bad[1][field] if field in ('EEG_source_stimulus_id', 'condition_cache_row', 'condition_sha256', 'image_seed') else '0' * 64
        reject(lambda bad=bad: C.validate_inference(protocol, arms['correct'], bad), 'binding differs')
    checks.append('same_count_wrong_subject_query_cache_row_or_SHA_parent_component_calibration_or_seed_refused')
    stale = copy.deepcopy(arms['correct']); stale['subject_bindings']['sub-01']['condition_cache_model_sha256'] = '0' * 64
    reject(lambda: C.validate_arm(protocol, stale), 'different parent model')
    checks.append('cache_export_model_SHA_must_equal_its_bound_parent_checkpoint')
    bad = copy.deepcopy(inference['permuted']); bad[0]['EEG_source_stimulus_id'] = bad[0]['stimulus_id']
    reject(lambda: C.validate_inference(protocol, arms['permuted'], bad), 'binding differs')
    bad = copy.deepcopy(inference['missing_raw_condition']); bad[0]['actual_A_input_fp16_sha256'] = named_sha('encoder_zero_EEG_is_not_raw_zero')
    reject(lambda: C.validate_inference(protocol, arms['missing_raw_condition'], bad), 'exactly zero at native A')
    bad = copy.deepcopy(inference['missing_raw_condition']); bad[0]['component_applied'] = True
    reject(lambda: C.validate_inference(protocol, arms['missing_raw_condition'], bad), 'binding differs')
    checks.append('permuted_self_source_and_missing_encoder_zeroEEG_or_nonbypassed_component_refused')
    for field, value in (('concept_id', 'answer'), ('target_image_sha256', '0' * 64), ('score_target_path', '/root/answer.jpg')):
        bad = copy.deepcopy(inference['correct']); bad[0][field] = value
        reject(lambda bad=bad: C.validate_inference(protocol, arms['correct'], bad), 'forbidden target/score')
    bad_protocol = copy.deepcopy(protocol); bad_protocol['target_answers'] = []
    reject(lambda: C.validate_protocol(bad_protocol), 'unregistered/target fields')
    checks.append('no_target_label_path_or_semantic_answer_in_blind_input_and_inference')
    for field, value in (('array_row', '199'), ('image_sha256', '0' * 64), ('concept_id', 'incorrect')):
        bad_targets = copy.deepcopy(targets); bad_targets[0][field] = value
        reject(lambda bad_targets=bad_targets: C.validate_joined(protocol, arms['correct'], joined['correct'], bad_targets, gallery),
            'array/stimulus' if field == 'array_row' else 'concept/SHA')
    checks.append('origin_array_vs_stimulus_and_target_SHA_or_concept_join_errors_refused')
    bad_joined = copy.deepcopy(joined['correct']); bad_joined[0]['initial_UNet_input_sha256'] = named_sha('different_actual_buffer_same_seed')
    reject(lambda: C.validate_pair(protocol, other, other_joined, arms['correct'], bad_joined, targets, gallery, 'matched_methods'), 'ACTUAL first-UNet observation differs')
    aliased = copy.deepcopy(arms['correct']); aliased['run_name'] = other['run_name']
    aliased_joined = C._joined_rows(protocol, aliased, inference_fixture(protocol, aliased), targets, gallery)
    reject(lambda: C.validate_pair(protocol, other, other_joined, aliased, aliased_joined, targets, gallery, 'matched_methods'), 'Cross-arm generated path reuse')
    reject(lambda: C.validate_pair(protocol, other, other_joined, arms['permuted'], joined['permuted'], targets, gallery, 'matched_methods'), 'across EEG interventions')
    checks.append('same_seed_different_observed_UNet_input_crossarm_path_reuse_and_crossintervention_main_comparison_refused')
    for family in ('paper', 'common'):
        scores = score_fixture(joined['correct'], family)
        first_metric = next(iter(C.metric_registry(family)['primary'])); scores[0][first_metric] = ''
        result = C.validate_score_rows(protocol, arms['correct'], joined['correct'], targets, gallery, scores, family)
        assert result['queries'] == 2000 and result['defined_query_counts'][first_metric] == 1999 and result['full_denominator_preserved']
        bad = copy.deepcopy(scores); bad[0]['DINO2AFC' if family == 'paper' else 'eval_swav'] = '0.5'
        reject(lambda bad=bad, family=family: C.validate_score_rows(protocol, arms['correct'], joined['correct'], targets, gallery, bad, family), 'cannot be mixed')
        bad = copy.deepcopy(scores); bad[0]['score_target_path'] = '/root/wrong.jpg'
        reject(lambda bad=bad, family=family: C.validate_score_rows(protocol, arms['correct'], joined['correct'], targets, gallery, bad, family), 'Score identity differs')
        missing_scores = score_fixture(joined['missing_raw_condition'], family)
        missing_scores[0]['EEG_source_stimulus_id'] = ''  # csv.DictWriter encodes None as an empty field.
        C.validate_score_rows(protocol, arms['missing_raw_condition'], joined['missing_raw_condition'], targets, gallery, missing_scores, family)
    assert C.metric_registry('paper')['primary']['eval_swav'] == 'lower'
    assert C.metric_registry('paper')['extra_code_metric']['eval_effnet'] == 'lower' and C.metric_registry('common')['primary']['EffNet'] == 'higher'
    checks.append('full2000_paper7_plus_extra_and_common8_separate_directions_undefined_retained_no_threshold_or_selection')
    assert 'torch' not in sys.modules and 'PIL' not in sys.modules
    require_time = time.monotonic() - started
    assert require_time < 180, 'Finite CPU software-check bound reached'
    return dict(status='PASS_FULL2000_SYNTHETIC_METADATA_CONTRACT_ONLY', checks=checks, seconds=require_time,
        source_sha256={Path(C.__file__).name: sha(Path(C.__file__).read_bytes()), Path(__file__).name: sha(Path(__file__).read_bytes())},
        actual_saved_old_schema_evidence=evidence, synthetic_queries_per_condition=2000, synthetic_conditions=3,
        synthetic_saved_inference_bytes=sizes, maximum_input_manifest_bytes=C.MAX_METADATA_BYTES,
        actual_benchmark_or_new_V_score_files_read=False, real_metric_values_used=False, actual_images_or_tensor_payloads_read=False,
        actual_GPU_or_model_forward=False, method_lock_issued=False, actual_training_generation_or_scoring_complete=False,
        actual_hook_execution_or_payload_authenticity_certified=False,
        limits=['2000 rows are complete synthetic metadata, not actual EEG/cache/image or scientific success.',
            'Byte observation helpers check supplied finite buffers; pinned future producer/hook and payload audits remain required.',
            'No final method/calibration rule is chosen or admitted; current four-arm closure is still required.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--receipt', type=Path); args = parser.parse_args()
    if args.receipt: assert not args.receipt.exists()
    report = check(); raw = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    assert len(raw) < 64 << 10
    if args.receipt:
        with args.receipt.open('xb') as stream: stream.write(raw)
    print(raw.decode(), end='')


if __name__ == '__main__':
    main()
