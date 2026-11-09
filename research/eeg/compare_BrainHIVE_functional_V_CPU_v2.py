"""One closed, complete O/P/PN/S comparison; no selection or GPU/model imports.

The real evaluators predate v2: their CSV does not contain array rows, target
SHA, gallery SHA or initial-noise SHA. Those identities are joined back through
the pinned generation manifests, feature configs/completions and full artifact
audit. Scalar completion headers are admitted for ALL arms before metric values
are parsed. Production has no small-query switch and never retries a failure.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time

import audit_BrainHIVE_functional_full_V_artifacts_CPU_v1 as artifacts

ROOT, WORK = artifacts.ROOT, artifacts.WORK
SCHEMA = 'BrainHIVE_functional_V_comparison_CPU_v2'
ARMS = ('O', 'P', 'PN', 'S')
CONTRASTS = (('PN', 'O'), ('PN', 'P'), ('PN', 'S'), ('P', 'O'))
SEMANTIC = ('DINO2AFC', 'DINO_cosine', 'DINO_Top1', 'DINO_Top5', 'EffNet', 'CapSim', 'ObjectF1', 'SEED')
LOW = ('PixCorr_RGB', 'SSIM_Y', 'mean_RGB_absolute_error')
METRICS = SEMANTIC + LOW
MAX_SECONDS, MAX_OUTPUT_BYTES = 1800, 64 << 20
STATISTICS_SHA = '028a67fb746d171efad15e163cf4016d0846c843db87ea87ee54ef8e3619e788'
DEPENDENCIES = ('compare_BrainHIVE_functional_V_CPU_v2.py',
    'audit_BrainHIVE_functional_full_V_artifacts_CPU_v1.py',
    'audit_BrainHIVE_spatial_image_integrity_CPU_v1.py', 'extract_image_features.py',
    'score_full_manifest.py', 'score_low_level.py', 'image_evaluation.py',
    'locked_metrics.py', 'semantic_evaluator.py', 'run_BrainHIVE_functional_evaluation_CPU_v1.py')
require, sha, read_json = artifacts.require, artifacts.spatial.sha_file, artifacts.read_json


class FullPrecisionCSV(artifacts.spatial.ChunkedCSV):
    """Retain round-trip float64 deltas; the pixel-audit writer uses10digits."""
    def add(self, row):
        super().add({key: repr(value) if isinstance(value, float) else value for key, value in row.items()})


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise artifacts.spatial.TimeBudgetExceeded('Fixed comparison time bound reached; no automatic retry')


def child(path, root):
    return path.resolve() != root.resolve() and path.resolve().is_relative_to(root.resolve())


def nonalias(paths, label):
    resolved, inodes = set(), set()
    for path in paths:
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), 'Missing/symlink ' + label + ': ' + str(path))
        actual = str(path.resolve())
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        require(actual not in resolved and identity not in inodes, 'Cross-arm reused ' + label)
        resolved.add(actual)
        inodes.add(identity)


def scalar_header(path, stop, allowed):
    """Parse only preceding scalar metadata; never decode the metric object.

    The frozen producers put the first metric dictionary at the given stop key.
    Even NaN or invalid text in that dictionary is not interpreted by this
    function. Missing/reordered/extra headers fail closed, rather than falling
    back to json.load and accidentally reading a partial arm's metric values.
    """
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        text = stream.read(16384)
    position, result = 0, {}
    def ws(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i
    position = ws(position)
    require(position < len(text) and text[position] == '{', 'Completion header is not JSON object')
    position += 1
    while True:
        position = ws(position)
        key, end = decoder.raw_decode(text, position)
        require(isinstance(key, str), 'Invalid completion header key')
        position = ws(end)
        require(position < len(text) and text[position] == ':', 'Invalid completion header delimiter')
        if key == stop:
            require(set(result) == set(allowed), 'Missing/reordered completion header fields')
            return result
        require(key in allowed and key not in result, 'Unregistered completion header field: ' + key)
        value, position = decoder.raw_decode(text, ws(position + 1))
        require(type(value) in (str, int, bool, type(None)), 'Non-scalar/numeric-metric completion header')
        result[key] = value
        position = ws(position)
        require(position < len(text) and text[position] == ',', 'Completion metric boundary missing')
        position += 1


def verify_evaluation_closure(context, meta, registration, pins, deadline):
    """Consume the real CPU wrapper's output, never issue an inferred receipt."""
    closure_path = Path(context['entry']['metadata_root']) / 'evaluation_closure.json'
    closure = read_json(closure_path, pins)
    require(closure['schema'] == 'BrainHIVE_functional_evaluation_CPU_v1_closure'
            and closure['status'] == ('CPU_TINY_FIXTURE_ONLY' if context['fixture'] else 'FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED')
            and closure['arm'] == context['arm'] and closure['candidate'] == context['recipe']['candidate']
            and closure['recipe_core_sha256'] == context['core'] and closure['queries'] == context['count']
            and closure['actual_full_V16600_evaluation_completed'] is (not context['fixture'])
            and closure['full_identity_target_seed_noise_chain_verified'] is True
            and closure['complete_low_level_mechanical_coverage'] is True
            and closure['GPU_used'] is False and closure['image_sampling'] is False,
            'Evaluation closure is not complete own-arm execution evidence')
    features = Path(context['entry']['feature_root'])
    gallery_features = Path(registration['gallery_features_root'])
    expected_refs = dict(manifest=context['manifest'], gallery_manifest=Path(context['recipe']['generation_gallery_manifest']),
        features_config=features / 'config.json', features_completion=features / 'completion.json',
        gallery_features_config=gallery_features / 'config.json', gallery_features_completion=gallery_features / 'completion.json',
        score_summary=meta['score'] / 'summary.json', score_per_query=meta['score'] / 'per_query.csv',
        low_summary=meta['low'] / 'summary.json', low_per_query=meta['low'] / 'per_query.csv',
        artifact_summary=meta['artifact'] / 'summary.json', artifact_per_query=meta['artifact'] / 'per_query.csv.gz')
    for key, path in expected_refs.items():
        check_time(deadline)
        ref = closure[key]
        require(Path(ref['path']) == path and artifacts.digest(ref['sha256']) and sha(path) == ref['sha256'],
                'Evaluation closure path/SHA differs: ' + key)
        pins[str(path)] = ref['sha256']
    source_pins = registration['lifecycle_source_sha256']
    for name in DEPENDENCIES:
        if name == Path(__file__).name:
            continue
        require(closure['source_sha256'].get(name) == source_pins[name], 'Closure evaluator/dependency source differs: ' + name)
    for key, root, count in (('feature_shards', features, context['count']),
                             ('gallery_feature_shards', gallery_features, len(context['inputs']['inputs']))):
        expected_paths = {str(root / stage / (str(start).zfill(8) + '.pt'))
            for stage in ('DINO', 'EffNet', 'CapSim', 'ObjectF1') for start in range(0, count, 256)}
        refs = closure[key]
        require(len(refs) == len(expected_paths) and {ref['path'] for ref in refs} == expected_paths,
                'Closure feature-shard coverage differs')
        for ref in refs:
            check_time(deadline)
            path = Path(ref['path'])
            require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= (64 << 20)
                    and sha(path) == ref['sha256'], 'Frozen feature-shard byte identity differs')
            pins[str(path)] = ref['sha256']
    calls = closure['actual_invocations']
    require(len(calls) == 2, 'Exactly one real scoring and one real low-level invocation required')
    expected_calls = {
        'score_full_manifest.py': {'--manifest': str(context['manifest']), '--features': str(features),
            '--gallery-manifest': str(context['recipe']['generation_gallery_manifest']),
            '--gallery-features': str(gallery_features), '--output': str(meta['score'])},
        'score_low_level.py': {'--manifest': str(context['manifest']),
            '--gallery-manifest': str(context['recipe']['generation_gallery_manifest']), '--output': str(meta['low'])}}
    seen = set()
    for call in calls:
        argv = call['argv']
        require(isinstance(argv, list) and all(isinstance(v, str) for v in argv), 'No actual invocation argv')
        scripts = [(i, Path(token).name) for i, token in enumerate(argv) if Path(token).name in expected_calls]
        require(len(scripts) == 1, 'Invocation is not one of the real frozen evaluator scripts')
        index, name = scripts[0]
        require(name not in seen and Path(argv[index]).resolve() == (ROOT / name).resolve() and call['exit_code'] == 0,
                'Evaluation invocation failed, duplicated or used another script')
        tail = argv[index + 1:]
        require(len(tail) % 2 == 0 and len(set(tail[::2])) == len(tail[::2])
                and dict(zip(tail[::2], tail[1::2])) == expected_calls[name], 'Actual evaluator CLI/input/output differs')
        require(type(call['seconds']) in (int, float) and math.isfinite(call['seconds']) and call['seconds'] >= 0,
                'Invalid evaluator runtime receipt')
        log = call['stdout_log']
        path = Path(log['path'])
        require(child(path, Path(context['entry']['metadata_root'])) and path.is_file() and not path.is_symlink()
                and sha(path) == log['sha256'], 'Actual evaluator output log changed/missing')
        pins[str(path)] = log['sha256']
        seen.add(name)
    require(seen == set(expected_calls), 'Missing actual evaluator invocation')
    for key in expected_refs:
        path = str(expected_refs[key])
        if key.startswith(('score_', 'low_')):
            continue
        require(closure['immutable_input_sha256'].get(path) == pins[path], 'Closure immutable input chain differs: ' + key)
    return closure


def prepare(recipe_path, deadline, *, _fixture=None):
    """All-four metadata gates; hashes are not metric-value access."""
    started, pins = time.monotonic(), {}
    fixture = _fixture is not None
    root_recipe = read_json(recipe_path, pins)
    require(root_recipe.get('schema') == 'BrainHIVE_functional_alignment_v2_recipe',
            'DRAFT/unknown recipe cannot open comparison scores')
    require(root_recipe.get('arm') in ARMS, 'Unknown arm recipe')
    admission = read_json(root_recipe['admission_receipt'], pins, root_recipe['admission_sha256'])
    require(admission['status'] == 'FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMITTED', 'No sealed formal admission')
    plan_ref = admission['four_arm_lifecycle_plan']
    plan = read_json(plan_ref['path'], pins, plan_ref['sha256'])
    require(plan['schema'] == 'BrainHIVE_functional_alignment_v2_four_arm_storage_plan', 'Unknown lifecycle plan')
    require(plan['all_four_train_generate_score_lifecycles_implemented'] is True, 'Four-arm lifecycle incomplete')
    registration = plan['comparison']  # Required before fitting, not invented after results.
    require(registration['schema'] == SCHEMA + '_sealed_plan', 'Comparison not registered in sealed plan')
    require(registration['maximum_seconds'] == MAX_SECONDS and registration['maximum_output_bytes'] == MAX_OUTPUT_BYTES,
            'Fixed comparison resource limits differ')
    source_pins = registration['lifecycle_source_sha256']
    for name in DEPENDENCIES:
        require(source_pins.get(name) == sha(ROOT / name), 'Sealed comparison/dependency source mismatch: ' + name)
        pins[str(ROOT / name)] = source_pins[name]
    stats_ref = registration['paired_statistics_plan']
    stats_path = ROOT / 'configs/paired_statistics_plan.json'
    require(Path(stats_ref['path']).resolve() == stats_path.resolve() and stats_ref['sha256'] == STATISTICS_SHA,
            'Statistics registration differs from locked original')
    stats = read_json(stats_path, pins, STATISTICS_SHA)
    require(stats['bootstrap_draws'] == 1000 and stats['bootstrap_seed'] == stats['training_seed'] == 1729
            and stats['practical_DINO_gain'] == .01 and stats['primary_metric'] == 'DINO2AFC'
            and stats['quality_noninferiority_margin'] is None, 'Locked decision/statistics contract differs')
    refs = plan['recipes']
    require(len(refs) == 4 and {r['arm'] for r in refs} == set(ARMS), 'Exactly four recipe references required')
    require(len(plan['arms']) == 4 and {e['arm'] for e in plan['arms']} == set(ARMS), 'Exactly four storage arms required')
    # First existence pass is independent of score-file content.
    entries = {entry['arm']: entry for entry in plan['arms']}
    paths, recipes = {}, {}
    work = WORK if not fixture else _fixture['work']
    for ref in refs:
        arm, entry = ref['arm'], entries[ref['arm']]
        recipe = read_json(ref['path'], pins, ref.get('sha256'))
        recipes[arm] = recipe
        require(recipe['arm'] == arm and recipe['candidate'] == 'functional_alignment_v2_' + arm
                and entry['recipe_core_sha256'] == artifacts.core_sha(recipe), 'Wrong arm recipe/core')
        metadata = Path(entry['metadata_root'])
        require(metadata == work / 'images' / ('protected_R4_functional_alignment_v2_' + arm), 'Noncanonical own metadata root')
        score_root, low_root = Path(entry['score_root']), Path(entry['low_level_root'])
        audit_root = Path(entry['artifact_audit_root'])
        for path in (score_root, low_root, audit_root):
            require(child(path, metadata), 'Evaluation/artifact output is outside charged own-arm metadata budget')
        require(len({str(p.resolve()) for p in (score_root, low_root, audit_root)}) == 3,
                'Different evaluation stages share a directory')
        paths[arm] = dict(score=score_root, low=low_root, artifact=audit_root)
        required = [metadata / name for name in ('completion.json', 'inference_completion.json',
            'pairing_audit.json', 'manifest.jsonl', 'inference_manifest.jsonl', 'model_receipt.json', 'evaluation_closure.json')]
        required += [Path(entry['feature_root']) / name for name in ('completion.json', 'config.json')]
        required += [directory / name for directory in (score_root, low_root) for name in ('summary.json', 'per_query.csv')]
        required += [audit_root / name for name in ('summary.json', 'per_query.csv.gz', 'pixel_duplicate_groups.csv.gz')]
        require(all(p.is_file() and 0 < p.stat().st_size <= artifacts.spatial.MAX_MANIFEST_BYTES for p in required),
                'Missing/incomplete/oversized arm artifacts; no metric tables opened: ' + arm)
    nonalias([paths[a]['score'] / 'per_query.csv' for a in ARMS], 'score CSV')
    nonalias([paths[a]['low'] / 'per_query.csv' for a in ARMS], 'low-level CSV')
    nonalias([paths[a]['artifact'] / 'per_query.csv.gz' for a in ARMS], 'artifact CSV')
    contexts, flags, manifests, closures = {}, {}, {}, {}
    for arm in ARMS:
        check_time(deadline)
        ref = next(r for r in refs if r['arm'] == arm)
        context = artifacts.prepare(ref['path'], deadline, _fixture=_fixture)
        require(context['arm'] == arm, 'Wrong prepared arm')
        contexts[arm] = context
        pins.update(context['pins'])
        sibling_admission = read_json(recipes[arm]['admission_receipt'], pins, recipes[arm]['admission_sha256'])
        require(sibling_admission['four_arm_lifecycle_plan'] == plan_ref, 'Four arms use different sealed plans')
        for key in ('source_sha256', 'input_sha256', 'input_contract_sha256', 'latent_receipt',
                    'generator_asset_root', 'generator', 'reference_Q_policy', 'timestep_policy', 'strength_policy'):
            require(recipes[arm][key] == recipes['O'][key], 'Four-arm frozen host/data mismatch: ' + key)
        entry, meta = context['entry'], paths[arm]
        feature_root = Path(entry['feature_root'])
        require(feature_root == work / 'evaluation' / context['run_name'] / 'features', 'Wrong own-arm feature root')
        feature_done = read_json(feature_root / 'completion.json', pins)
        config = read_json(feature_root / 'config.json', pins)
        manifest_hash = context['pins'][str(context['manifest'])]
        require(feature_done['status'] == 'COMPLETED' and feature_done['queries'] == context['count']
                and feature_done['manifest_sha256'] == config['manifest_sha256'] == manifest_hash
                and feature_done['all_queries_used'] is True and feature_done['query_subsampling'] is False
                and config['manifest'] == str(context['manifest']) and config['real_gallery'] is False,
                'Feature completion is not the actual full arm manifest')
        audit = read_json(meta['artifact'] / 'summary.json', pins)
        require(audit['status'] == ('CPU_TINY_FIXTURE_ONLY' if fixture else 'COMPLETE_FULL_V_ARTIFACT_AUDIT')
                and audit['complete_technical_coverage'] is True
                and audit['actual_full_V16600_verified'] is (not fixture) and audit['arm'] == arm
                and audit['candidate'] == recipes[arm]['candidate'] and audit['recipe_core_sha256'] == context['core'],
                'Full artifact audit is not mechanically complete for own arm')
        require(all(audit[k] == context['count'] for k in ('expected_queries', 'full_denominator', 'identity_verified',
            'canonical_path_verified', 'decoded_512_RGB', 'hash_verified_decoded_without_other_failures', 'per_query_rows_written'))
            and audit['per_query_rows_unwritten'] == audit['spatial_flags_unknown'] == 0
            and not audit['failures'] and not audit['source_errors'], 'Artifact audit retains failures/unknown')
        require(audit['source_sha256'] == source_pins[Path(artifacts.__file__).name]
                and audit['measurement_source_sha256'] == source_pins[Path(artifacts.spatial.__file__).name]
                and audit['immutable_metadata_sha256'][str(context['manifest'])] == manifest_hash,
                'Artifact source/manifest identity differs')
        table = meta['artifact'] / 'per_query.csv.gz'
        require(sha(table) == audit['table_sha256'], 'Artifact table changed after complete audit')
        pins[str(table)] = audit['table_sha256']
        rows = artifacts.read_jsonl(context['manifest'], pins, manifest_hash)
        inf = artifacts.read_jsonl(context['inference'], pins, context['pins'][str(context['inference'])])
        require(len(rows) == len(inf) == context['count'], 'Full manifest denominator differs')
        with gzip.open(table, 'rt', newline='', encoding='utf-8') as stream:
            audit_rows = list(csv.DictReader(stream))
        require(len(audit_rows) == context['count'], 'Incomplete artifact query table')
        for index, (row, inference, audited) in enumerate(zip(rows, inf, audit_rows)):
            issues, path_ok, identity, path = artifacts.identity_issues(context, inference, row, index)
            require(not issues and path_ok, 'Manifest identity/path failure before scoring: ' + arm)
            require(audited['query_id'] == audited['expected_query_id'] == row['query_id'] == identity
                    and audited['generated_path'] == audited['expected_generated_path'] == str(path)
                    and audited['manifest_generated_sha256'] == audited['actual_file_sha256'] == row['generated_sha256']
                    and audited['source_manifest_sha256'] == manifest_hash and audited['status'] == 'OK'
                    and all(audited[k] == 'True' for k in ('hash_verified', 'decoded_512_RGB', 'identity_verified', 'canonical_path_verified')),
                    'Artifact query/file/identity proof does not match manifest')
            require(artifacts.digest(audited['decoded_RGB_pixel_sha256']), 'Missing decoded-pixel identity')
        require(config['keys'] == [r['query_id'] for r in rows], 'Feature query order differs')
        flags[arm], manifests[arm] = audit_rows, rows
        contexts[arm]['feature_config'] = config
    # Balanced identities, common gallery and truly equal query/noise streams.
    baseline = contexts['O']
    queries, subjects = baseline['inputs']['inputs'], sorted(baseline['inputs']['subjects'])
    concepts = sorted({str(t['concept_id']) for t in baseline['targets'].values()})
    expected_subjects, expected_concepts = (10, 166) if not fixture else (_fixture['subjects'], _fixture['concepts'])
    require(len(subjects) == expected_subjects and len(concepts) == expected_concepts, 'Wrong full subject/concept coverage')
    concept_counts = collections.Counter(str(t['concept_id']) for t in baseline['targets'].values())
    require(set(concept_counts.values()) == {10 if not fixture else _fixture['stimuli_per_concept']}, 'Incomplete concept stimuli')
    for arm in ARMS:
        require(contexts[arm]['inputs'] == baseline['inputs'] and contexts[arm]['gallery'] == baseline['gallery']
                and contexts[arm]['targets'] == baseline['targets'] and contexts[arm]['protocol'] == baseline['protocol'],
                'Four-arm input/gallery/protocol differs')
        for index, row in enumerate(manifests[arm]):
            base = manifests['O'][index]
            for key in ('query_id', 'subject', 'stimulus_id', 'array_row', 'global_query_index', 'concept_id',
                        'target_image_path', 'score_target_path', 'target_image_sha256', 'target_image_zip', 'target_image_member',
                        'EEG_source_stimulus_id', 'image_seed', 'initial_UNet_input_sha256', 'parent_checkpoint_path',
                        'parent_checkpoint_sha256', 'condition_cache_path', 'condition_cache_sha256', 'condition_sha256'):
                require(row[key] == base[key], 'Pair/query/noise/target mismatch: ' + key)
        common_features = {k: v for k, v in baseline['feature_config'].items()
                           if k not in ('manifest', 'manifest_sha256', 'keys')}
        require({k: v for k, v in contexts[arm]['feature_config'].items()
                 if k not in ('manifest', 'manifest_sha256', 'keys')} == common_features, 'Evaluator model/preprocessing differs')
    # All 66400 generated paths must be physically separate; equal hashes are
    # reported, never confused with reused files or silently discarded images.
    nonalias((r['generated_path'] for arm in ARMS for r in manifests[arm]), 'generated file')
    nonalias([contexts[a]['model']['component_checkpoint_path'] for a in ARMS], 'component checkpoint')
    gallery_root = Path(registration['gallery_features_root'])
    require(registration['gallery_features_reuse_only'] is True, 'Shared gallery features must already be complete, not an unbudgeted future job')
    gallery_done = read_json(gallery_root / 'completion.json', pins)
    gallery_cfg = read_json(gallery_root / 'config.json', pins)
    gallery_path = Path(recipes['O']['generation_gallery_manifest'])
    gallery_hash = baseline['pins'][str(gallery_path)]
    require(gallery_done['status'] == 'COMPLETED' and gallery_done['queries'] == len(queries)
            and gallery_done['manifest_sha256'] == gallery_cfg['manifest_sha256'] == gallery_hash
            and gallery_done['all_queries_used'] is True and gallery_done['query_subsampling'] is False
            and gallery_cfg['manifest'] == str(gallery_path) and gallery_cfg['real_gallery'] is True
            and gallery_cfg['keys'] == list(baseline['gallery']), 'Shared real-gallery feature identity/order differs')
    for key in ('DINO_weight_sha256', 'dino_preprocessing', 'SEED_source_commit', 'ObjectF1_grid', 'semantic_model_receipts'):
        require(gallery_cfg[key] == baseline['feature_config'][key], 'Real/query evaluator differs: ' + key)
    for arm in ARMS:
        closures[arm] = verify_evaluation_closure(contexts[arm], paths[arm], registration, pins, deadline)
    nonalias((piece['path'] for arm in ARMS for piece in closures[arm]['feature_shards']), 'query feature shard')
    # Only now inspect scalar score/low-level completion headers for all arms.
    score_fields = ('status', 'queries', 'subjects', 'gallery_images', 'method', 'intervention',
                    'repeat_protocol', 'metric_version', 'metric_source_sha256')
    low_fields = ('status', 'expected_images', 'actual_present', 'verified_and_decoded', 'missing_count',
                  'corrupt_or_mismatched_count', 'duplicate_hash_groups', 'duplicate_images_beyond_first')
    for arm in ARMS:
        score, low = paths[arm]['score'], paths[arm]['low']
        for file in (score / 'summary.json', score / 'per_query.csv', low / 'summary.json', low / 'per_query.csv'):
            pins[str(file)] = sha(file)
        header = scalar_header(score / 'summary.json', 'summary', score_fields)
        require(header == dict(status='COMPLETED', queries=baseline['count'], subjects=expected_subjects,
            gallery_images=len(queries), method='BrainHIVE_' + contexts[arm]['run_name'], intervention='correct',
            repeat_protocol='R4', metric_version='fixed_real_gallery_explicit_stimulus_v1',
            metric_source_sha256=source_pins['locked_metrics.py']), 'Full score metadata header is not complete own arm')
        header = scalar_header(low / 'summary.json', 'summary_over_valid_images', low_fields)
        require(header['status'] == 'COMPLETED' and all(header[k] == baseline['count'] for k in
            ('expected_images', 'actual_present', 'verified_and_decoded')) and header['missing_count'] == header['corrupt_or_mismatched_count'] == 0,
            'Full low-level metadata header is incomplete')
    output = Path(registration['output_root'])
    pn_metadata = Path(entries['PN']['metadata_root'])
    require(output == pn_metadata / 'comparison_CPU_v2' and child(output, pn_metadata), 'Comparison output must be charged within own PN metadata root')
    require(entries['PN']['metadata_budget_bytes'] - sum(p.stat().st_size for p in pn_metadata.rglob('*') if p.is_file() and not p.is_symlink())
            >= MAX_OUTPUT_BYTES, 'PN remaining metadata budget cannot cover full comparison cap')
    require(shutil.disk_usage(pn_metadata).free >= MAX_OUTPUT_BYTES + (64 << 20), 'Insufficient bounded comparison disk space')
    require(not output.exists(), 'Preserve existing/partial comparison; no automatic re-analysis')
    for path, expected in pins.items():
        check_time(deadline)
        require(sha(path) == expected, 'Metadata/source changed during admission')
    return dict(contexts=contexts, manifests=manifests, flags=flags, paths=paths, pins=pins, output=output,
        plan=plan, plan_ref=plan_ref, subjects=subjects, concepts=concepts, count=baseline['count'],
        fixture=fixture, score_table_read_count={a: 0 for a in ARMS}, all_four_metadata_gates_closed=True,
        closures=closures, started=started, preflight_seconds=time.monotonic() - started)


def read_values(context, deadline):
    """Read each actual score CSV exactly once, only after all-four closure."""
    require(context['all_four_metadata_gates_closed'] is True, 'Score-value access before all-four closure')
    values = {}
    for arm in ARMS:
        check_time(deadline)
        meta, expected = context['paths'][arm], context['manifests'][arm]
        require(context['score_table_read_count'][arm] == 0, 'Repeated score peek forbidden')
        with (meta['score'] / 'per_query.csv').open(newline='', encoding='utf-8') as stream:
            scores = list(csv.DictReader(stream))
        context['score_table_read_count'][arm] += 1
        with (meta['low'] / 'per_query.csv').open(newline='', encoding='utf-8') as stream:
            low = list(csv.DictReader(stream))
        require(len(scores) == len(low) == context['count'], 'Full score/low query denominator differs')
        require(len({r['query_id'] for r in scores}) == context['count'], 'Duplicate score query')
        data = []
        for index, (row, details, manifest) in enumerate(zip(scores, low, expected)):
            if index % 1000 == 0:
                check_time(deadline)
            for key in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'EEG_source_stimulus_id',
                        'generated_path', 'target_image_path', 'score_target_path', 'image_seed'):
                require(row[key] == str(manifest[key]), 'Score query/order/target/seed/path differs: ' + key)
            for key in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'generated_path'):
                require(details[key] == str(manifest[key]), 'Low-level query/order/path differs: ' + key)
            require(details['status'] == 'OK' and details['scoring_target_path'] == manifest['score_target_path'], 'Low-level target or valid denominator differs')
            numbers = {k: float(row[k]) for k in SEMANTIC}
            numbers.update({k: float(details[k]) for k in LOW})
            require(all(math.isfinite(v) for v in numbers.values()), 'Null/nonfinite metric retained as failure; no dropped query')
            require(all(0 <= numbers[k] <= 1 for k in ('DINO2AFC', 'ObjectF1', 'DINO_Top1', 'DINO_Top5'))
                    and all(numbers[k] in (0., 1.) for k in ('DINO_Top1', 'DINO_Top5')), 'Invalid metric range')
            require(abs(numbers['SEED'] - sum(numbers[k] for k in ('EffNet', 'CapSim', 'ObjectF1')) / 3) < 1e-10,
                    'SEED no longer matches frozen public-code combination')
            data.append(numbers)
        values[arm] = data
        # The low-level producer includes manifest/gallery/source pins after its
        # metric dictionaries. They can be parsed now that all arms are closed.
        low_summary = read_json(meta['low'] / 'summary.json', {})
        own = context['contexts'][arm]
        require(low_summary['manifest_sha256'] == own['pins'][str(own['manifest'])]
                and low_summary['gallery_manifest_sha256'] == own['pins'][own['recipe']['generation_gallery_manifest']]
                and low_summary['source_sha256'] == context['plan']['comparison']['lifecycle_source_sha256']['score_low_level.py']
                and low_summary['GPU_used'] is False, 'Low-level source/manifest/gallery provenance differs')
    return values


def crossed_statistics(values, manifests, subjects, concepts, *, deadline):
    """Original equal-subject/equal-concept mean and SAME1000 crossed draws."""
    import numpy as np
    matrix = np.empty((len(subjects), len(concepts), len(METRICS)), dtype=np.float64)
    groups = collections.defaultdict(list)
    for row, value in zip(manifests, values):
        groups[(row['subject'], str(row['concept_id']))].append([value[k] for k in METRICS])
    require(len(groups) == len(subjects) * len(concepts) and len({len(g) for g in groups.values()}) == 1,
            'Crossed bootstrap requires complete balanced subject/concept cells')
    for i, subject in enumerate(subjects):
        for j, concept in enumerate(concepts):
            matrix[i, j] = np.mean(groups[(subject, concept)], axis=0)
    require(np.isfinite(matrix).all(), 'Nonfinite full bootstrap matrix')
    rng, draws = np.random.default_rng(1729), []
    for index in range(1000):
        if index % 25 == 0:
            check_time(deadline)
        si = rng.integers(len(subjects), size=len(subjects))
        ci = rng.integers(len(concepts), size=len(concepts))
        draws.append(matrix[si[:, None], ci[None, :]].mean(axis=(0, 1)))
    draws = np.asarray(draws)
    effects = {key: dict(mean=float(matrix[:, :, i].mean()),
        CI95_subject_and_concept_bootstrap=np.quantile(draws[:, i], [.025, .975]).tolist(),
        training_seed_variability_covered=False) for i, key in enumerate(METRICS)}
    return effects, matrix


def positive_signal(effects):
    dino, seed, obj = (effects[k] for k in ('DINO2AFC', 'SEED', 'ObjectF1'))
    return bool(dino['mean'] >= .01 and dino['CI95_subject_and_concept_bootstrap'][0] > 0
        and seed['mean'] > 0 and seed['CI95_subject_and_concept_bootstrap'][0] > 0 and obj['mean'] >= 0)


def describe_increment(effects):
    """Natural-zero descriptions, not a new practical threshold/decision test."""
    return dict(DINO_mean_positive=effects['DINO2AFC']['mean'] > 0,
        DINO_CI_entirely_positive=effects['DINO2AFC']['CI95_subject_and_concept_bootstrap'][0] > 0,
        SEED_mean_positive=effects['SEED']['mean'] > 0,
        SEED_CI_entirely_positive=effects['SEED']['CI95_subject_and_concept_bootstrap'][0] > 0,
        ObjectF1_mean_nonnegative=effects['ObjectF1']['mean'] >= 0,
        scope='Descriptive natural-zero directions only; no added practical cutoff or confirmatory mechanism test.')


def compare(context, *, max_seconds=MAX_SECONDS, _memory=None):
    require(max_seconds > 0 and max_seconds <= MAX_SECONDS, 'Invalid finite comparison bound')
    for key in artifacts.spatial.THREAD_VARIABLES:
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    memory = artifacts.spatial.memory_preflight()
    require(_memory is None or context['fixture'], 'Synthetic memory injection is forbidden for formal comparison')
    if _memory is not None:
        memory = dict(_memory, actual_tiny_memory_preflight=memory, synthetic_fixture_injection=True)
    require(memory['effective_available_bytes'] >= (2 << 30), 'Full four-arm CPU comparison requires fresh2GiB raw headroom')
    output = context['output']
    require(not output.exists(), 'Preserve existing comparison')
    output.mkdir(parents=True, exist_ok=False)
    budget = artifacts.spatial.OutputBudget(output, MAX_OUTPUT_BYTES)
    started, deadline = time.monotonic(), context['started'] + max_seconds
    written, report = {}, None
    try:
        values = read_values(context, deadline)
        stats, matrices = {}, {}
        for arm in ARMS:
            stats[arm], matrices[arm] = crossed_statistics(values[arm], context['manifests'][arm],
                context['subjects'], context['concepts'], deadline=deadline)
        differences = {}
        identity_fields = ['query_id', 'subject', 'stimulus_id', 'array_row', 'concept_id', 'image_seed',
            'initial_UNet_input_sha256', 'target_image_path', 'score_target_path', 'target_image_sha256',
            'generated_path', 'reference_generated_path', 'generated_sha256', 'reference_generated_sha256']
        delta_fields = identity_fields + [prefix + k for k in METRICS for prefix in ('candidate_', 'reference_', 'delta_')]
        for arm, reference in CONTRASTS:
            name = arm + '_minus_' + reference
            delta = [{k: a[k] - b[k] for k in METRICS} for a, b in zip(values[arm], values[reference])]
            differences[name], matrices[name] = crossed_statistics(delta, context['manifests'][arm],
                context['subjects'], context['concepts'], deadline=deadline)
            table = FullPrecisionCSV(budget, name + '_per_query.csv.gz', delta_fields)
            for index, (row, base, diff) in enumerate(zip(context['manifests'][arm], context['manifests'][reference], delta)):
                if index % 1000 == 0:
                    check_time(deadline)
                record = {k: row[k] for k in identity_fields if not k.startswith('reference_')}
                record.update(reference_generated_path=base['generated_path'], reference_generated_sha256=base['generated_sha256'])
                for key in METRICS:
                    record.update({f'candidate_{key}': values[arm][index][key], f'reference_{key}': values[reference][index][key],
                                   f'delta_{key}': diff[key]})
                table.add(record)
            table.flush()
            written[name] = table.rows_written
        strata = FullPrecisionCSV(budget, 'all_arm_and_contrast_strata.csv.gz',
            ['entry', 'stratum', 'subject', 'concept_id', 'queries'] + list(METRICS))
        cell_n = context['count'] // (len(context['subjects']) * len(context['concepts']))
        for name, matrix in matrices.items():
            for i, subject in enumerate(context['subjects']):
                strata.add(dict(entry=name, stratum='subject', subject=subject, concept_id='',
                    queries=cell_n * len(context['concepts']), **{k: float(matrix[i, :, j].mean()) for j, k in enumerate(METRICS)}))
            for i, concept in enumerate(context['concepts']):
                strata.add(dict(entry=name, stratum='concept', subject='', concept_id=concept,
                    queries=cell_n * len(context['subjects']), **{k: float(matrix[:, i, j].mean()) for j, k in enumerate(METRICS)}))
            for i, subject in enumerate(context['subjects']):
                for j, concept in enumerate(context['concepts']):
                    strata.add(dict(entry=name, stratum='subject_concept', subject=subject, concept_id=concept,
                        queries=cell_n, **{k: float(matrix[i, j, n]) for n, k in enumerate(METRICS)}))
        strata.flush()
        for path, expected in context['pins'].items():
            check_time(deadline)
            require(sha(path) == expected, 'Input/source changed after comparison; no valid decision')
        pn_o = differences['PN_minus_O']
        incremental = {name: describe_increment(differences[name]) for name in ('PN_minus_P', 'PN_minus_S')}
        consistent = all(all(value for key, value in increment.items() if key != 'scope') for increment in incremental.values())
        overlap = {arm: sum(a['generated_sha256'] == b['generated_sha256'] for a, b in
            zip(context['manifests'][arm], context['manifests']['O'])) for arm in ('P', 'PN', 'S')}
        report = dict(schema=SCHEMA, status='CPU_TINY_FIXTURE_ONLY' if context['fixture'] else 'ALL_FOUR_FULL16600_V_PAIRED_COMPARISONS_COMPLETE',
            actual_full_V16600_comparison_completed=not context['fixture'], queries_per_arm=context['count'], arms=list(ARMS),
            subjects=len(context['subjects']), concepts=len(context['concepts']), repeat_protocol='R4', gallery_images=len(context['contexts']['O']['gallery']),
            fixed_training_seed=1729, bootstrap_seed=1729, bootstrap_draws=1000, bootstrap='Equal subjects/concepts; crossed resampling; same paired draws',
            arm_statistics=stats, paired_differences=differences,
            primary_comparison='PN_minus_O', provisional_PN_signal=positive_signal(pn_o),
            primary_gate=dict(DINO_delta_at_least=.01, DINO_CI_lower_strictly_above=0, SEED_delta_strictly_above=0,
                              SEED_CI_lower_strictly_above=0, ObjectF1_delta_at_least=0),
            ranking_descriptive_increments=incremental, both_ranking_contrasts_descriptively_consistent=consistent,
            ranking_mechanism_confirmed=False, mechanism_confirmation_requires_controlled_increment_over_both_P_and_S=True,
            shuffled_S_drop_alone_is_mechanism_proof=False, positive_only_P_observation=positive_signal(differences['P_minus_O']),
            P_replaces_failed_PN=False, winner_selection_performed=False, fitted_parameters=0, image_generation_calls=0,
            score_table_read_count=context['score_table_read_count'], complete_query_delta_rows=written,
            same_encoded_bytes_as_O_count=overlap, artifact_flag_counts={a: {k: sum(r[k] == 'True' for r in context['flags'][a])
                for k in artifacts.FLAGS} for a in ARMS},
            immutable_input_sha256=context['pins'], source_sha256=sha(__file__), plan_identity=context['plan_ref'],
            identity_evidence='Score CSV fields joined to complete generator, target metadata, real gallery, feature config/completion, actual evaluator invocation/shard/output closure and full artifact SHA proof; array/target/noise SHA are not claimed to be native score CSV columns. Initial_UNet_input_sha256 is the first UNet input, not a claim of hashing every ancestral random draw.',
            independent_H_available=False, independent_SOTA_established=False, cross_architecture_generalization_tested=False,
            quality_noninferiority_established=False, quality_noninferiority_margin=None, human_quality_ratings_collected=False,
            official_test_used_to_tune=False, automatic_retry=False, no_new_p_values=True,
            limits=['Single training seed; bootstrap does not cover training-seed variability.',
                'V is exposed development evidence; intervals are descriptive and not independent confirmation.',
                'ObjectF1 is a COCO+gender proxy; DINO2AFC is not correct-object accuracy.',
                'Engineering constant/duplicate flags are not perceptual quality or object success judgments.',
                'A null/negative PN result is retained; P cannot be renamed to erase it or reopen candidate search.',
                'No independent H or cross-architecture evidence; no SOTA or universal-adapter claim.'])
    except Exception as error:
        report = dict(schema=SCHEMA, status='FULL_V_COMPARISON_FAILED_NO_DECISION',
            actual_full_V16600_comparison_completed=False, expected_queries_per_arm=context['count'],
            full_denominator_preserved=True, no_query_exclusion=True, partial_delta_rows=written,
            failure=type(error).__name__ + ': ' + str(error)[:300], score_table_read_count=context['score_table_read_count'],
            provisional_PN_signal=None, ranking_mechanism_confirmed=False, automatic_retry=False)
    report['budget'] = dict(maximum_seconds=max_seconds, seconds=time.monotonic() - started,
        fixed_full_V_runtime_bound_unmeasured=True, preflight_seconds=context['preflight_seconds'],
        maximum_output_bytes=MAX_OUTPUT_BYTES, output_bytes_before_summary=budget.used, memory_preflight=memory,
        full_cap_charged_to_PN_metadata_budget=True)
    budget.json('summary.json', report)
    require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= MAX_OUTPUT_BYTES, 'Comparison exceeded fixed output cap')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True, help='One formally sealed own-arm recipe; its plan binds all four.')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    try:
        context = prepare(args.recipe, time.monotonic() + MAX_SECONDS)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps(dict(status='REFUSED_BEFORE_METRIC_VALUE_ACCESS', error=str(error)[:300], score_table_read_count=0)))
        raise SystemExit(2)
    if args.preflight_only:
        print(json.dumps(dict(status='ALL_FOUR_METADATA_GATES_CLOSED_NO_METRIC_VALUES_READ', queries_per_arm=context['count'],
            output=str(context['output']), score_table_read_count=0)))
        return
    report = compare(context)
    print(json.dumps({k: report[k] for k in ('status', 'actual_full_V16600_comparison_completed', 'provisional_PN_signal', 'ranking_mechanism_confirmed')}))
    if report['status'] == 'FULL_V_COMPARISON_FAILED_NO_DECISION':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
