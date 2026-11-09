"""Synthetic-only software contracts; no actual V rows, tensors or images."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import compare_BrainHIVE_functional_V_CPU_v2 as comparison


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def csv_rows(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else Path.open
    with opener(path, 'wt' if path.suffix == '.gz' else 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def reseal_plan(root, plan):
    plan_path = root / 'plan.json'
    write(plan_path, plan)
    for arm in comparison.ARMS:
        recipe_path, admission_path = root / ('recipe_' + arm + '.json'), root / ('admission_' + arm + '.json')
        recipe = json.loads(recipe_path.read_text())
        write(admission_path, dict(status='FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMITTED', candidate=recipe['candidate'],
            recipe_core_sha256=comparison.artifacts.core_sha(recipe),
            four_arm_lifecycle_plan=dict(path=str(plan_path), sha256=comparison.sha(plan_path))))
        recipe['admission_sha256'] = comparison.sha(admission_path)
        write(recipe_path, recipe)


def make_closures(root, plan):
    gallery_features = Path(plan['comparison']['gallery_features_root'])
    for entry in plan['arms']:
        recipe = json.loads((root / ('recipe_' + entry['arm'] + '.json')).read_text())
        metadata, features = Path(entry['metadata_root']), Path(entry['feature_root'])
        score, low, artifact = Path(entry['score_root']), Path(entry['low_level_root']), Path(entry['artifact_audit_root'])
        refs = dict(manifest=metadata / 'manifest.jsonl', gallery_manifest=Path(recipe['generation_gallery_manifest']),
            features_config=features / 'config.json', features_completion=features / 'completion.json',
            gallery_features_config=gallery_features / 'config.json', gallery_features_completion=gallery_features / 'completion.json',
            score_summary=score / 'summary.json', score_per_query=score / 'per_query.csv', low_summary=low / 'summary.json',
            low_per_query=low / 'per_query.csv', artifact_summary=artifact / 'summary.json', artifact_per_query=artifact / 'per_query.csv.gz')
        shards = {}
        for key, directory in (('feature_shards', features), ('gallery_feature_shards', gallery_features)):
            pieces = []
            for stage in ('DINO', 'EffNet', 'CapSim', 'ObjectF1'):
                path = directory / stage / '00000000.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(('synthetic feature placeholder ' + str(path)).encode())
                pieces.append(dict(path=str(path), sha256=comparison.sha(path)))
            shards[key] = pieces
        calls = []
        for script, flags in (
            ('score_full_manifest.py', ['--manifest', str(refs['manifest']), '--features', str(features),
                '--gallery-manifest', str(refs['gallery_manifest']), '--gallery-features', str(gallery_features), '--output', str(score)]),
            ('score_low_level.py', ['--manifest', str(refs['manifest']), '--gallery-manifest', str(refs['gallery_manifest']), '--output', str(low)])):
            path = metadata / 'synthetic_invocation_logs' / (script + '.log')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Synthetic fixture invocation; not actually run.\n')
            calls.append(dict(argv=[sys.executable, str(comparison.ROOT / script)] + flags, exit_code=0, seconds=.01,
                stdout_log=dict(path=str(path), sha256=comparison.sha(path))))
        closure = dict(schema='BrainHIVE_functional_evaluation_CPU_v1_closure',
            status='CPU_TINY_FIXTURE_ONLY', arm=entry['arm'], candidate=recipe['candidate'],
            recipe_core_sha256=comparison.artifacts.core_sha(recipe), queries=8,
            actual_full_V16600_evaluation_completed=False, full_identity_target_seed_noise_chain_verified=True,
            complete_low_level_mechanical_coverage=True, GPU_used=False, image_sampling=False,
            **{key: dict(path=str(path), sha256=comparison.sha(path)) for key, path in refs.items()},
            **shards, source_sha256=plan['comparison']['lifecycle_source_sha256'], actual_invocations=calls,
            immutable_input_sha256={str(path): comparison.sha(path) for path in refs.values()})
        write(metadata / 'evaluation_closure.json', closure)


def fixture(root, *, increments=None):
    """Tiny shared native metadata schema; audit receipts are synthetic evidence.

    Placeholder generated files are never decoded here; real production must
    present the full auditor's actual16600 receipt. Fixture summaries are never
    reported as real audit/evaluation evidence.
    """
    root = Path(root)
    work, repo = root / 'work', root / 'repo'
    subjects = ['sub-02', 'sub-01']  # Generation order differs from sorted bootstrap axes.
    queries = [dict(stimulus_id='stim_' + str(i) + '.jpg', array_row=100 + i) for i in range(4)]
    count = len(subjects) * len(queries)
    inputs_path, protocol_path = work / 'data/generation_inputs/V.json', repo / 'configs/split_manifest.json'
    target_path, gallery_path = work / 'data/generation_inputs/V_targets.jsonl', work / 'gallery/V/manifest.jsonl'
    inputs = dict(expected_queries=count, subjects=subjects, repeat_protocol='R4',
        target_image_or_class_fields_present=False, inputs=queries)
    protocol = dict(version='tiny_four_arm_contract_v2', image_rows=dict(V=[q['array_row'] for q in queries]))
    targets = [dict(**q, concept_id='concept_' + str(i // 2), image_sha256=hashlib.sha256(('target' + str(i)).encode()).hexdigest(),
        image_zip=str(root / 'never_opened_targets.zip'), image_member='member_' + str(i)) for i, q in enumerate(queries)]
    gallery = [dict(stimulus_id=t['stimulus_id'], concept_id=t['concept_id'], image_sha256=t['image_sha256'],
        image_path=str(root / 'never_read_target' / t['stimulus_id'])) for t in targets]
    write(inputs_path, inputs)
    write(protocol_path, protocol)
    jsonl(target_path, targets)
    jsonl(gallery_path, gallery)
    sources = {name: comparison.sha(comparison.ROOT / name) for name in comparison.DEPENDENCIES}
    generator_hash = comparison.sha(comparison.ROOT / 'generate_BrainHIVE_functional_alignment_V_v2.py')
    common_source = {'generate_BrainHIVE_functional_alignment_V_v2.py': generator_hash}
    source_artifacts = {Path(comparison.artifacts.__file__).name: comparison.sha(comparison.artifacts.__file__),
        Path(comparison.artifacts.spatial.__file__).name: comparison.sha(comparison.artifacts.spatial.__file__)}
    input_pins = {str(p): comparison.sha(p) for p in (inputs_path, protocol_path, target_path, gallery_path)}
    plan = dict(schema='BrainHIVE_functional_alignment_v2_four_arm_storage_plan',
        all_four_train_generate_score_lifecycles_implemented=True, recipes=[], arms=[],
        comparison=dict(schema=comparison.SCHEMA + '_sealed_plan', maximum_seconds=comparison.MAX_SECONDS,
            maximum_output_bytes=comparison.MAX_OUTPUT_BYTES, lifecycle_source_sha256=sources,
            paired_statistics_plan=dict(path=str(comparison.ROOT / 'configs/paired_statistics_plan.json'),
                                        sha256=comparison.STATISTICS_SHA),
            gallery_features_root=str(work / 'evaluation/shared_gallery_features'), gallery_features_reuse_only=True,
            output_root=str(work / 'images/protected_R4_functional_alignment_v2_PN/comparison_CPU_v2')))
    parent = {s: dict(checkpoint_path=str(root / (s + '_never_read_parent.pt')), checkpoint_sha256='a' * 64,
        condition_cache_path=str(root / (s + '_never_read_conditions.pt')), condition_cache_sha256='b' * 64) for s in subjects}
    feature_common = dict(DINO_weight_sha256='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d',
        dino_preprocessing='Historical routes CPU resize real/CUDA resize generated; 256 bilinear antialias; layer11 CLS+mean patch concatenated and normalized',
        SEED_source_commit='5327a134b90b04fb1b37066d98b309c9cc533a02',
        ObjectF1_grid='public code np.arange(.05,.95,.05); area50; preserve empty detections convention', semantic_model_receipts=['e' * 64, 'f' * 64])
    increments = increments or dict(O=0., P=.02, PN=.05, S=0.)
    for arm in comparison.ARMS:
        recipe_path = root / ('recipe_' + arm + '.json')
        name = 'protected_R4_functional_alignment_v2_' + arm
        metadata = work / 'images' / name
        train = work / 'components/functional_alignment_v2' / arm
        image_root = root / ('private_images_' + arm)
        feature_root = work / 'evaluation' / name / 'features'
        recipe = dict(schema='BrainHIVE_functional_alignment_v2_recipe', arm=arm, candidate='functional_alignment_v2_' + arm,
            training_seed=1729, generator=dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1),
            admission_receipt=str(root / ('admission_' + arm + '.json')), admission_sha256=None,
            output_directory=str(train), generation_gallery_manifest=str(gallery_path), input_sha256=input_pins,
            source_sha256=common_source, input_contract_sha256='1' * 64, latent_receipt=dict(path='not_loaded', sha256='2' * 64),
            generator_asset_root='/not_loaded', reference_Q_policy='fixed_parent', timestep_policy='uniform_0_999', strength_policy='fixed_original_Dfit')
        core = comparison.artifacts.core_sha(recipe)
        entry = dict(arm=arm, candidate=recipe['candidate'], recipe_core_sha256=core, run_name=name,
            metadata_root=str(metadata), metadata_budget_bytes=200000000,
            inference_manifest=str(metadata / 'inference_manifest.jsonl'), feature_root=str(feature_root), training_root=str(train),
            score_root=str(metadata / 'scores'), low_level_root=str(metadata / 'low_level'),
            artifact_audit_root=str(metadata / 'artifact_audit_CPU_v1'), lifecycle_source_sha256=source_artifacts,
            image_segments=[dict(start=0, stop=count, root=str(image_root))])
        plan['arms'].append(entry)
        plan['recipes'].append(dict(arm=arm, path=str(recipe_path)))
        write(recipe_path, recipe)
        train.mkdir(parents=True)
        checkpoint = train / 'component_final.pt'
        checkpoint.write_bytes(('tiny placeholder never loaded ' + arm).encode())
        checkpoint_sha = comparison.sha(checkpoint)
        calibration = dict(kind='Dfit_only_global_strength_calibration', strengths=[0., .25, .5, .75, 1.], selected_strength=1.)
        write(train / 'calibration.json', calibration)
        write(train / 'completion.json', dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE',
            candidate=recipe['candidate'], arm=arm, updates=24807, seen_queries=396900, passes=3, parameters=525568,
            recipe_core_sha256=core, checkpoint_sha256=checkpoint_sha, calibration_sha256=comparison.sha(train / 'calibration.json')))
        model = dict(candidate=recipe['candidate'], arm=arm, run_name=name, recipe_core_sha256=core,
            source_sha256=common_source, component_checkpoint_sha256=checkpoint_sha, component_checkpoint_path=str(checkpoint),
            generation_source_sha256=generator_hash, parent_bindings=parent, resolution=512, steps=4, CFG=0, IP_scale=1,
            images_per_query=1, normal_inference_target_access=False, teacher_or_rank_pair_used_at_inference=False, calibration=calibration)
        write(metadata / 'model_receipt.json', model)
        rows, joined, audit_rows, score_rows, low_rows = [], [], [], [], []
        for index, (subject, query) in enumerate((s, q) for s in subjects for q in queries):
            target, actual = targets[index % 4], gallery[index % 4]
            qid = subject + '/' + query['stimulus_id'] + '/R4'
            seed = int(hashlib.sha256(('1729|' + subject + '|' + query['stimulus_id'] + '|R4|' + protocol['version']).encode()).hexdigest()[:16], 16) % (2**63 - 1)
            path = image_root / name / subject / (Path(query['stimulus_id']).stem + '.webp')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('fixture placeholder ' + arm + qid).encode())
            row = dict(query_id=qid, subject=subject, stimulus_id=query['stimulus_id'], array_row=query['array_row'],
                global_query_index=index, EEG_source_stimulus_id=query['stimulus_id'], image_seed=seed, candidate=recipe['candidate'],
                arm=arm, run_name=name, method='BrainHIVE_' + name, repeat_protocol='R4', intervention='correct', recipe_core_sha256=core,
                component_checkpoint_sha256=checkpoint_sha, component_checkpoint_path=str(checkpoint),
                model_receipt_sha256=comparison.sha(metadata / 'model_receipt.json'), generation_source_sha256=generator_hash,
                target_or_visual_features_read_for_condition=False, oracle_diagnostic_only=False, teacher_or_negative_used_at_inference=False,
                selected_best_seed=False, lossless_pixel_roundtrip=True, UNet_NFE=4, IP_conditioned_attention_calls=280,
                IP_value_calls=280, VAE_decoder_calls=1, generated_path=str(path), generated_sha256=comparison.sha(path),
                parent_checkpoint_path=parent[subject]['checkpoint_path'], parent_checkpoint_sha256='a' * 64,
                condition_cache_path=parent[subject]['condition_cache_path'], condition_cache_sha256='b' * 64,
                condition_sha256=hashlib.sha256(qid.encode()).hexdigest(), initial_UNet_input_sha256=hashlib.sha256(('noise' + qid).encode()).hexdigest(),
                actual_A_input_fp16_sha256='2' * 64, condition_norm=32., component_correction_RMS=.1, component_effective_radius=.25,
                component_gate=1., actual_A_input_fp16_norm=32., VAE_float_min=-1., VAE_float_max=1.)
            join = row | dict(concept_id=target['concept_id'], target_image_sha256=target['image_sha256'],
                target_image_path=actual['image_path'], score_target_path=actual['image_path'], target_image_zip=target['image_zip'],
                target_image_member=target['image_member'], target_mapping_attached_after_inference=True)
            rows.append(row)
            joined.append(join)
            audit_rows.append(dict(query_id=qid, expected_query_id=qid, generated_path=str(path), expected_generated_path=str(path),
                manifest_generated_sha256=row['generated_sha256'], actual_file_sha256=row['generated_sha256'],
                source_manifest_sha256='', status='OK', hash_verified=True, decoded_512_RGB=True, identity_verified=True,
                canonical_path_verified=True, decoded_RGB_pixel_sha256=hashlib.sha256(('pixel' + arm + qid).encode()).hexdigest(),
                **{k: False for k in comparison.artifacts.FLAGS}))
            variation = .002 * (index % 4) + .001 * (index // 4)
            increment = increments[arm]
            semantic = dict(DINO2AFC=.4 + variation + increment, DINO_cosine=.2 + increment,
                DINO_Top1=0, DINO_Top5=1, EffNet=.2 + variation + increment,
                CapSim=.2 + variation + increment, ObjectF1=.2 + variation + increment,
                SEED=.2 + variation + increment)
            score_rows.append({k: join[k] for k in ('query_id', 'subject', 'stimulus_id', 'concept_id',
                'EEG_source_stimulus_id', 'target_image_path', 'generated_path', 'score_target_path', 'image_seed')} | semantic)
            low_rows.append({k: join[k] for k in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'generated_path')} |
                dict(status='OK', scoring_target_path=actual['image_path'], PixCorr_RGB=.2 + increment,
                     SSIM_Y=.4 + increment, mean_RGB_absolute_error=.1 - increment))
        jsonl(metadata / 'inference_manifest.jsonl', rows)
        jsonl(metadata / 'manifest.jsonl', joined)
        manifest_sha = comparison.sha(metadata / 'manifest.jsonl')
        inf_done = dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_ACTUAL_IMAGES_SAVED_NO_TARGET_ACCESS',
            candidate=recipe['candidate'], arm=arm, run_name=name, queries=count, recipe_core_sha256=core,
            normal_inference_target_access=False, images_per_query=1, UNet_NFE_per_image=4, IP_calls_per_image=280,
            selected_best_seed=False, generated_images_selected_or_discarded=False,
            inference_manifest_sha256=comparison.sha(metadata / 'inference_manifest.jsonl'), model_receipt_sha256=comparison.sha(metadata / 'model_receipt.json'))
        write(metadata / 'inference_completion.json', inf_done)
        proof = dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_TARGET_AND_QUERY_IDENTITIES_JOINED', candidate=recipe['candidate'],
            arm=arm, queries=count, manifest_sha256=manifest_sha, gallery_manifest_sha256=comparison.sha(gallery_path), target_metadata_sha256=comparison.sha(target_path))
        write(metadata / 'pairing_audit.json', proof)
        write(metadata / 'completion.json', dict(status='FUNCTIONAL_ALIGNMENT_V2_ALL16600_IMAGES_AND_TARGET_JOIN_COMPLETE', candidate=recipe['candidate'],
            arm=arm, queries=count, role='full', split='V', intervention='correct', images_per_query=1, selected_best_seed=False,
            recipe_core_sha256=core, manifest_sha256=manifest_sha, inference_completion_sha256=comparison.sha(metadata / 'inference_completion.json'),
            pairing_audit_sha256=comparison.sha(metadata / 'pairing_audit.json')))
        feature_cfg = dict(manifest=str(metadata / 'manifest.jsonl'), manifest_sha256=manifest_sha,
            keys=[r['query_id'] for r in rows], real_gallery=False, **feature_common)
        write(feature_root / 'config.json', feature_cfg)
        write(feature_root / 'completion.json', dict(status='COMPLETED', queries=count, manifest_sha256=manifest_sha, all_queries_used=True, query_subsampling=False))
        audit_root = Path(entry['artifact_audit_root'])
        for row in audit_rows:
            row['source_manifest_sha256'] = manifest_sha
        csv_rows(audit_root / 'per_query.csv.gz', audit_rows)
        csv_rows(audit_root / 'pixel_duplicate_groups.csv.gz', [], ['decoded_RGB_pixel_sha256', 'query_count', 'distinct_file_sha256_count'])
        write(audit_root / 'summary.json', dict(status='CPU_TINY_FIXTURE_ONLY', complete_technical_coverage=True,
            actual_full_V16600_verified=False, arm=arm, candidate=recipe['candidate'], recipe_core_sha256=core,
            **{k: count for k in ('expected_queries', 'full_denominator', 'identity_verified', 'canonical_path_verified',
                'decoded_512_RGB', 'hash_verified_decoded_without_other_failures', 'per_query_rows_written')},
            per_query_rows_unwritten=0, spatial_flags_unknown=0, failures={}, source_errors=[], source_sha256=source_artifacts[Path(comparison.artifacts.__file__).name],
            measurement_source_sha256=comparison.artifacts.MEASUREMENT_SOURCE_SHA256,
            immutable_metadata_sha256={str(metadata / 'manifest.jsonl'): manifest_sha}, table_sha256=comparison.sha(audit_root / 'per_query.csv.gz')))
        score_root, low_root = Path(entry['score_root']), Path(entry['low_level_root'])
        csv_rows(score_root / 'per_query.csv', score_rows)
        csv_rows(low_root / 'per_query.csv', low_rows)
        write(score_root / 'summary.json', dict(status='COMPLETED', queries=count, subjects=len(subjects), gallery_images=len(queries),
            method='BrainHIVE_' + name, intervention='correct', repeat_protocol='R4', metric_version='fixed_real_gallery_explicit_stimulus_v1',
            metric_source_sha256=sources['locked_metrics.py'], summary={'synthetic_only': 42.}, per_subject={'synthetic_only': 99.}))
        write(low_root / 'summary.json', dict(status='COMPLETED', expected_images=count, actual_present=count, verified_and_decoded=count,
            missing_count=0, corrupt_or_mismatched_count=0, duplicate_hash_groups=0, duplicate_images_beyond_first=0,
            summary_over_valid_images={'synthetic_only': 7.}, manifest_sha256=manifest_sha,
            gallery_manifest_sha256=comparison.sha(gallery_path), source_sha256=sources['score_low_level.py'], GPU_used=False))
    gallery_root = Path(plan['comparison']['gallery_features_root'])
    write(gallery_root / 'config.json', dict(manifest=str(gallery_path), manifest_sha256=comparison.sha(gallery_path),
        keys=[g['stimulus_id'] for g in gallery], real_gallery=True, **feature_common))
    write(gallery_root / 'completion.json', dict(status='COMPLETED', queries=len(queries), manifest_sha256=comparison.sha(gallery_path),
        all_queries_used=True, query_subsampling=False))
    make_closures(root, plan)
    reseal_plan(root, plan)
    return root / 'recipe_O.json', dict(root=root, work=work, repo=repo, count=count,
        subjects=len(subjects), concepts=2, stimuli_per_concept=2), plan


def prepare(data):
    return comparison.prepare(data[0], time.monotonic() + 60, _fixture=data[1])


def expect_refusal(call, contains):
    try:
        call()
    except (ValueError, KeyError, OSError) as error:
        assert contains in str(error), str(error)
    else:
        raise AssertionError('Expected refusal: ' + contains)


def test_bootstrap():
    rng = np.random.default_rng(123)
    subjects, concepts = ['s1', 's2', 's3'], ['c1', 'c2', 'c3', 'c4']
    rows = [dict(subject=s, concept_id=c) for s in subjects for c in concepts for _ in range(2)]
    values = [dict(zip(comparison.METRICS, rng.normal(size=len(comparison.METRICS)))) for _ in rows]
    effects, _ = comparison.crossed_statistics(values, rows, subjects, concepts, deadline=time.monotonic() + 60)
    for metric in comparison.METRICS:
        matrix = np.array([[np.mean([v[metric] for r, v in zip(rows, values) if r['subject'] == s and r['concept_id'] == c])
                            for c in concepts] for s in subjects])
        old_rng, draws = np.random.default_rng(1729), []
        for _ in range(1000):
            si = old_rng.integers(len(subjects), size=len(subjects))
            ci = old_rng.integers(len(concepts), size=len(concepts))
            draws.append(matrix[si[:, None], ci[None, :]].mean())
        np.testing.assert_allclose(effects[metric]['mean'], matrix.mean(), rtol=0, atol=1e-14)
        np.testing.assert_allclose(effects[metric]['CI95_subject_and_concept_bootstrap'], np.quantile(draws, [.025, .975]), rtol=0, atol=1e-14)


def estimate_full_output():
    class CountingBudget:
        def __init__(self):
            self.used = 0
            self.raw = 0
        def append(self, name, data, summary=False):
            self.used += len(data)
            self.raw += len(gzip.decompress(data))
    sink, rng = CountingBudget(), random.Random(1729)
    identity = ['query_id', 'subject', 'stimulus_id', 'array_row', 'concept_id', 'image_seed',
        'initial_UNet_input_sha256', 'target_image_path', 'score_target_path', 'target_image_sha256',
        'generated_path', 'reference_generated_path', 'generated_sha256', 'reference_generated_sha256']
    fields = identity + [prefix + k for k in comparison.METRICS for prefix in ('candidate_', 'reference_', 'delta_')]
    for arm, reference in comparison.CONTRASTS:
        table = comparison.FullPrecisionCSV(sink, arm + reference, fields)
        for i in range(16600):
            stem = hashlib.shake_256(str(i % 1660).encode()).hexdigest(125)
            subject = 'sub-' + str(i // 1660 + 1).zfill(2)
            base = '/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/images/BrainHIVE_functional_alignment_v2_'
            row = {k: rng.uniform(-1, 1) for k in fields}
            target = '/root/brainhive_reproduction_20261020_v1/gallery/V/' + stem + '.jpg'
            row.update(query_id=subject + '/' + stem + '.jpg/R4', subject=subject, stimulus_id=stem + '.jpg',
                array_row=i % 1660, concept_id='concept_' + str(i % 166), image_seed=rng.randrange(2**63),
                target_image_path=target, score_target_path=target,
                generated_path=base + arm + '/protected_R4_functional_alignment_v2_' + arm + '/' + subject + '/' + stem + '.webp',
                reference_generated_path=base + reference + '/protected_R4_functional_alignment_v2_' + reference + '/' + subject + '/' + stem + '.webp')
            for key in ('initial_UNet_input_sha256', 'target_image_sha256', 'generated_sha256', 'reference_generated_sha256'):
                row[key] = hashlib.sha256(str(rng.getrandbits(256)).encode()).hexdigest()
            table.add(row)
        table.flush()
    allowance = 4 << 20  # Complete strata table plus summary/source/failure metadata.
    assert sink.used + allowance < comparison.MAX_OUTPUT_BYTES
    return dict(query_delta_rows=66400, fields_per_delta_row=len(fields), uncompressed_delta_CSV_bytes=sink.raw,
        gzip_delta_CSV_bytes=sink.used, additional_strata_summary_allowance_bytes=allowance,
        total_forecast_bytes=sink.used + allowance, fixed_output_cap_bytes=comparison.MAX_OUTPUT_BYTES,
        actual_V_metadata_or_scores_read=False, local_large_file_written=False, compression_forecast_not_guarantee=True,
        full_cap_must_be_charged_to_PN_metadata_budget=True)


def run_checks():
    tests = {}
    tiny_memory = dict(effective_available_bytes=2 << 30)
    test_bootstrap()
    tests['vectorized_crossed_bootstrap_matches_original_interleaved_draws_with_tolerance'] = True
    with tempfile.TemporaryDirectory(prefix='brainhive_compare_CPU_') as temporary:
        root = Path(temporary)
        # Consume the other entry's actual tiny closure writer, not just our
        # independently constructed four-arm metadata. Scorer child outputs are
        # injected and explicitly marked tiny; no subprocess or real V score.
        import check_BrainHIVE_functional_evaluation_CPU_v1 as producer_check
        producer = producer_check.entry
        recipe, options = producer_check.fixture(root / 'producer_consumer_schema')
        producer_context = producer.prepare(recipe, time.monotonic() + 60, _fixture=options)
        runner, calls = producer_check.fake_runner('normal')
        tiny = producer.evaluate(producer_context, time.monotonic() + 60, _runner=runner,
            _memory=dict(effective_available_bytes=producer.MIN_MEMORY))
        registration = dict(gallery_features_root=str(producer_context['gallery_feature_root']),
            lifecycle_source_sha256={name: comparison.sha(comparison.ROOT / name) for name in comparison.DEPENDENCIES})
        accepted = comparison.verify_evaluation_closure(producer_context,
            dict(score=producer_context['score'], low=producer_context['low'], artifact=producer_context['artifact']),
            registration, {}, time.monotonic() + 60)
        assert accepted == tiny and len(calls) == 2 and not accepted['actual_full_V16600_evaluation_completed']
        tests['actual_new_producer_tiny_closure_consumed_with_exact_argv_refs_and_shard_SHA'] = True
        data = fixture(root / 'positive')
        context = prepare(data)
        assert context['score_table_read_count'] == dict.fromkeys(comparison.ARMS, 0)
        result = comparison.compare(context, _memory=tiny_memory)
        assert result['status'] == 'CPU_TINY_FIXTURE_ONLY' and not result['actual_full_V16600_comparison_completed'], result
        assert result['provisional_PN_signal'] and result['both_ranking_contrasts_descriptively_consistent']
        assert not result['ranking_mechanism_confirmed'] and not result['winner_selection_performed']
        assert result['complete_query_delta_rows'] == dict.fromkeys(['PN_minus_O', 'PN_minus_P', 'PN_minus_S', 'P_minus_O'], 8)
        assert result['score_table_read_count'] == dict.fromkeys(comparison.ARMS, 1)
        tests['all_four_native_metadata_to_complete_query_delta_and_strata_outputs'] = True
        tests['fixed_PN_primary_and_two_incremental_controls_no_winner_or_mechanism_proof'] = True
        tests['score_tables_read_once_after_four_arm_gate_and_fixture_never_claims_fullV'] = True
        # Header access must not parse a metric dictionary, even invalid JSON.
        path = root / 'header_only.json'
        path.write_text('{"status":"COMPLETED", "summary":THIS_IS_NOT_JSON}', encoding='utf-8')
        assert comparison.scalar_header(path, 'summary', ('status',)) == {'status': 'COMPLETED'}
        tests['strict_scalar_header_never_decodes_metric_values'] = True
        for label, remove in [('missing_arm', 'score'), ('missing_artifact', 'artifact')]:
            item = fixture(root / label)
            entry = next(e for e in item[2]['arms'] if e['arm'] == 'S')
            path = Path(entry['score_root']) / 'per_query.csv' if remove == 'score' else Path(entry['artifact_audit_root']) / 'summary.json'
            path.unlink()
            expect_refusal(lambda: prepare(item), 'Missing/incomplete')
        tests['missing_arm_or_artifact_rejected_before_metric_access'] = True
        item = fixture(root / 'missing_evaluation_closure')
        entry = next(e for e in item[2]['arms'] if e['arm'] == 'S')
        (Path(entry['metadata_root']) / 'evaluation_closure.json').unlink()
        expect_refusal(lambda: prepare(item), 'Missing/incomplete')
        tests['missing_actual_evaluation_closure_rejected_before_metric_access'] = True
        item = fixture(root / 'bad_invocation')
        entry = next(e for e in item[2]['arms'] if e['arm'] == 'S')
        path = Path(entry['metadata_root']) / 'evaluation_closure.json'
        closure = json.loads(path.read_text())
        closure['actual_invocations'][0]['exit_code'] = 1
        write(path, closure)
        expect_refusal(lambda: prepare(item), 'invocation failed')
        tests['failed_real_invocation_cannot_close_comparison'] = True
        item = fixture(root / 'cross_arm_hardlink')
        generated = {}
        for entry in item[2]['arms']:
            generated[entry['arm']] = json.loads((Path(entry['metadata_root']) / 'manifest.jsonl').read_text().splitlines()[0])['generated_path']
        Path(generated['PN']).unlink()
        os.link(generated['O'], generated['PN'])
        expect_refusal(lambda: prepare(item), 'Cross-arm reused generated file')
        tests['physically_reused_generated_file_rejected_even_when_paths_differ'] = True
        for key, value in [('score_target_path', 'wrong_target'), ('image_seed', '123'), ('generated_path', 'wrong_arm_image')]:
            item = fixture(root / key)
            context = prepare(item)
            file = context['paths']['PN']['score'] / 'per_query.csv'
            with file.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            rows[0][key] = value
            csv_rows(file, rows)
            expect_refusal(lambda: comparison.read_values(context, time.monotonic() + 60), 'differs: ' + key)
        item = fixture(root / 'wrong_sort')
        context = prepare(item)
        file = context['paths']['PN']['score'] / 'per_query.csv'
        with file.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        rows[0], rows[1] = rows[1], rows[0]
        csv_rows(file, rows)
        expect_refusal(lambda: comparison.read_values(context, time.monotonic() + 60), 'differs: query_id')
        tests['wrong_target_seed_generated_path_or_query_sort_rejected'] = True
        item = fixture(root / 'source_tamper')
        item[2]['comparison']['lifecycle_source_sha256'][Path(comparison.__file__).name] = '0' * 64
        reseal_plan(root / 'source_tamper', item[2])
        expect_refusal(lambda: prepare(item), 'source mismatch')
        tests['comparison_and_dependencies_require_presealed_expected_SHA'] = True
        item = fixture(root / 'null_PN', increments=dict(O=0., P=.02, PN=0., S=-.02))
        result = comparison.compare(prepare(item), _memory=tiny_memory)
        assert result['provisional_PN_signal'] is False and result['positive_only_P_observation'] is True
        assert result['P_replaces_failed_PN'] is False and not result['both_ranking_contrasts_descriptively_consistent']
        assert not result['ranking_mechanism_confirmed']
        tests['P_positive_PN_null_S_negative_cannot_rename_P_or_prove_rank_from_S_drop'] = True
        item = fixture(root / 'negative_PN', increments=dict(O=0., P=0., PN=-.02, S=-.04))
        result = comparison.compare(prepare(item), _memory=tiny_memory)
        assert result['provisional_PN_signal'] is False and result['paired_differences']['PN_minus_O']['DINO2AFC']['mean'] < 0
        tests['negative_primary_retained_despite_shuffled_arm_worse'] = True
        effects = {k: dict(mean=0., CI95_subject_and_concept_bootstrap=[0., 0.]) for k in comparison.METRICS}
        effects['DINO2AFC'] = dict(mean=.01, CI95_subject_and_concept_bootstrap=[.001, .02])
        effects['SEED'] = dict(mean=.001, CI95_subject_and_concept_bootstrap=[.0001, .002])
        assert comparison.positive_signal(effects)
        effects['SEED']['CI95_subject_and_concept_bootstrap'][0] = 0.
        assert not comparison.positive_signal(effects)
        effects['SEED']['CI95_subject_and_concept_bootstrap'][0] = .0001
        effects['ObjectF1']['mean'] = -1e-12
        assert not comparison.positive_signal(effects)
        tests['prespecified_practical_and_strict_CI_boundary_directions'] = True
        item = fixture(root / 'timeout')
        result = comparison.compare(prepare(item), max_seconds=1e-12, _memory=tiny_memory)
        assert result['status'] == 'FULL_V_COMPARISON_FAILED_NO_DECISION' and result['provisional_PN_signal'] is None
        assert result['expected_queries_per_arm'] == 8 and result['full_denominator_preserved']
        assert result['score_table_read_count'] == dict.fromkeys(comparison.ARMS, 0) and not result['actual_full_V16600_comparison_completed']
        tests['finite_timeout_keeps_full_denominator_unknown_no_metric_access_or_retry'] = True
        item = fixture(root / 'low_memory')
        context = prepare(item)
        expect_refusal(lambda: comparison.compare(context, _memory=dict(effective_available_bytes=(2 << 30) - 1)), 'fresh2GiB')
        assert not context['output'].exists() and context['score_table_read_count'] == dict.fromkeys(comparison.ARMS, 0)
        context['fixture'] = False
        expect_refusal(lambda: comparison.compare(context, _memory=tiny_memory), 'forbidden for formal')
        tests['memory_failure_precedes_scores_and_fixture_injection_forbidden_in_formal'] = True
        draft = comparison.ROOT / 'configs/BrainHIVE_functional_alignment_v2_preregistration_DRAFT.json'
        expect_refusal(lambda: comparison.prepare(draft, time.monotonic() + 60), 'DRAFT/unknown recipe')
        tests['current_real_DRAFT_rejected_before_any_metric_table_or_output'] = True
        assert 'torch' not in sys.modules
        tests['no_GPU_models_tensors_or_real_scores_accessed'] = True
    return tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise FileExistsError('Preserve previous receipt')
    tests, forecast = run_checks(), estimate_full_output()
    report = dict(status='FUNCTIONAL_FOUR_ARM_COMPARISON_SYNTHETIC_CPU_CONTRACTS_PASSED_NO_ACTUAL_V_READ',
        utc=comparison.artifacts.spatial.utc_now(), tests=tests, full16600_output_forecast=forecast,
        source_sha256={name: comparison.sha(comparison.ROOT / name) for name in (Path(__file__).name,
            'check_BrainHIVE_functional_evaluation_CPU_v1.py', 'check_BrainHIVE_functional_full_V_artifacts_CPU_v1.py') + comparison.DEPENDENCIES},
        statistics_plan_sha256=comparison.STATISTICS_SHA, actual_full_V_compared=False, actual_V_scores_read=False,
        GPU_used=False, new_training=False, temporary_fixture_directory_removed=True,
        interpretation='Software/interface and locked-statistics checks only; no evidence for actual component gain, quality NI or SOTA.')
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
