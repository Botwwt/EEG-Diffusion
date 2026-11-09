"""One D-minus-O efficacy readout after both complete lifecycles; CPU only.

This does not consume or replace the withdrawn four-arm mechanism comparison.
All completion and identity evidence is admitted before score CSVs are parsed.
The CLI has no partial-query, alternate contrast, threshold or retry option.
"""
import argparse
import csv
import gzip
import math
import os
from pathlib import Path
import time

import brainhive_paired_study_contract_20261025_v1 as C
from evaluate_BrainHIVE_paired_effect_CPU_20261025_v2 import own_rows, join_scores, stratum_rows
from compare_BrainHIVE_functional_V_CPU_v2 import METRICS, crossed_statistics

ARMS = ('O', 'D')
MAX_SECONDS, MAX_BYTES = 1800, 64 << 20
OUTPUT_NAME = 'efficacy_DO_20261026_v1'
IDENTITY = ('query_id', 'subject', 'stimulus_id', 'array_row', 'concept_id', 'image_seed',
            'initial_UNet_input_sha256', 'target_image_sha256', 'target_image_path', 'score_target_path')


def check_time(deadline):
    assert time.monotonic() < deadline, 'Finite D/O comparison deadline reached; no automatic retry'


def pin(item, expected, pins, deadline):
    check_time(deadline)
    path = Path(item['path'])
    assert path == Path(expected) and path.is_file() and not path.is_symlink(), ('Unexpected/missing reference', path)
    assert C.sha(path) == item['sha256'], ('SHA mismatch', path)
    previous = pins.setdefault(str(path), item['sha256'])
    assert previous == item['sha256']
    return path


def unique_file(path, inodes, deadline):
    check_time(deadline)
    path = Path(path)
    assert path.is_file() and not path.is_symlink()
    stat = path.stat()
    inode = (stat.st_dev, stat.st_ino)
    assert inode not in inodes, ('Reused image or feature inode', path)
    inodes.add(inode)


def verify_arm(config, arm, pins, inodes, deadline, *, count=16600, post_count=32, rows_reader=own_rows):
    """Metadata verifier; nondefault sizes are for explicit synthetic unit fixtures only."""
    assert arm in ARMS
    root = C.DATA / 'arms' / arm
    core = C.core(config)
    train_path = root / 'train/completion.json'
    train = C.read(train_path)
    assert train['status'] == 'PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE'
    assert train['arm'] == arm and train['study_core_sha256'] == core
    assert train['updates'] == 24807 and train['passes'] == 3 and train['parameters'] == 525568
    assert train['primary_exposures'] == train['paired_exposures'] == 396900
    for key, name in (('checkpoint', 'component_final.pt'), ('calibration', 'calibration.json'), ('provenance', 'input_provenance.json')):
        pin(train[key], root / 'train' / name, pins, deadline)
    pins[str(train_path)] = C.sha(train_path)
    post_path = root / 'postfit_T32/completion.json'
    post = C.read(post_path)
    assert post['status'] == 'PAIRED_GENERATION_COMPLETE' and post['queries'] == post_count
    assert post['arm'] == arm and post['split'] == 'T32' and post['study_core_sha256'] == core
    assert post['checkpoint'] == train['checkpoint'] and post['normal_inference_target_access'] is False
    post_manifest = pin(post['manifest'], root / 'postfit_T32/inference_manifest.jsonl', pins, deadline)
    import json
    with post_manifest.open() as stream:
        post_rows = [json.loads(line) for line in stream]
    assert len(post_rows) == len({row['query_id'] for row in post_rows}) == post_count
    for index, row in enumerate(post_rows):
        assert row['arm'] == arm and row['global_query_index'] == index
        assert row['study_core_sha256'] == core and row['component_checkpoint_sha256'] == train['checkpoint']['sha256']
        expected = root / 'postfit_T32/pixels' / row['subject'] / (Path(row['stimulus_id']).stem + '.webp')
        pin(dict(path=row['generated_path'], sha256=row['generated_sha256']), expected, pins, deadline)
        unique_file(expected, inodes, deadline)
    pins[str(post_path)] = C.sha(post_path)
    # The real helper replays every V query against blind input metadata.
    rows, generation = rows_reader(config, arm)
    assert len(rows) == count and len({row['query_id'] for row in rows}) == count
    assert generation['checkpoint'] == train['checkpoint']
    assert generation['normal_inference_target_access'] is False
    manifest = pin(generation['manifest'], root / 'images/manifest.jsonl', pins, deadline)
    pins[str(root / 'images/completion.json')] = C.sha(root / 'images/completion.json')
    for row in rows:
        assert row['study_core_sha256'] == core and row['component_checkpoint_sha256'] == train['checkpoint']['sha256']
        expected = root / 'images/pixels' / row['subject'] / (Path(row['stimulus_id']).stem + '.webp')
        pin(dict(path=row['generated_path'], sha256=row['generated_sha256']), expected, pins, deadline)
        unique_file(expected, inodes, deadline)
        assert row['target_mapping_attached_after_inference'] is True
        assert row['target_image_path'] == row['score_target_path']
        pin(dict(path=row['target_image_path'], sha256=row['target_image_sha256']), row['score_target_path'], pins, deadline)
    artifact_path = root / 'artifact/completion.json'
    artifact = C.read(artifact_path)
    assert artifact['status'] == 'FULL16600_PAIRED_ARTIFACT_AND_EEG_IDENTITY_AUDIT_COMPLETE'
    assert artifact['arm'] == arm and artifact['queries'] == artifact['full_denominator'] == count
    assert artifact['counts']['verified'] == count and not artifact['failures']
    assert artifact['manifest'] == generation['manifest']
    pin(artifact['table'], root / 'artifact/per_query.csv.gz', pins, deadline)
    feature_path, config_path = root / 'features/completion.json', root / 'features/config.json'
    feature, feature_config = C.read(feature_path), C.read(config_path)
    assert feature['status'] == 'COMPLETED' and feature['queries'] == count
    assert feature['all_queries_used'] is True and feature['query_subsampling'] is False
    assert feature['manifest_sha256'] == feature_config['manifest_sha256'] == generation['manifest']['sha256']
    gallery = C.bound(config['input_bindings']['gallery_feature_config'])
    for key in ('DINO_weight_sha256', 'dino_preprocessing', 'SEED_source_commit', 'ObjectF1_grid', 'semantic_model_receipts'):
        assert feature_config[key] == gallery[key]
    guard_path = root / 'features_guard/completion.json'
    guard = C.read(guard_path)
    assert guard['status'] == 'UNCHANGED_EXTRACTOR_WITH_REAL_PEAK_GUARD_COMPLETE' and guard['queries'] == count
    assert guard['engineering_only'] is False and guard['feature_completion'] == C.ref(feature_path)
    assert guard['peak_extra_CPU_bytes'] <= config['maximum_CPU_extra_bytes']
    assert guard['peak_GPU_bytes'] <= config['maximum_GPU_bytes']
    assert guard['extractor_sha256'] == C.sha(C.ROOT / 'extract_image_features.py')
    closure_path = root / 'evaluation/completion.json'
    closure = C.read(closure_path)
    assert closure['status'] == 'FULL16600_REAL_EVALUATOR_INVOCATIONS_CLOSED'
    assert closure['arm'] == arm and closure['queries'] == count
    assert closure['manifest'] == generation['manifest']
    for key, path in (('artifact', artifact_path), ('feature_completion', feature_path), ('feature_config', config_path), ('feature_guard', guard_path)):
        pin(closure[key], path, pins, deadline)
    shards = closure['feature_shards']
    expected_shards = {str(root / 'features' / stage / (str(start).zfill(8) + '.pt'))
                       for stage in ('DINO', 'EffNet', 'CapSim', 'ObjectF1') for start in range(0, count, 256)}
    assert set(shards) == expected_shards == {str(path) for path in (root / 'features').rglob('*.pt')}
    for filename, digest in shards.items():
        path = pin(dict(path=filename, sha256=digest), filename, pins, deadline)
        assert path.resolve().is_relative_to((root / 'features').resolve())
        unique_file(path, inodes, deadline)
    assert set(closure['outputs']) == {'semantic', 'low'}
    for branch, items in closure['outputs'].items():
        assert set(items) == {'table', 'summary'}
        for key, filename in (('table', 'per_query.csv'), ('summary', 'summary.json')):
            pin(items[key], root / 'evaluation' / branch / filename, pins, deadline)
    assert len(closure['invocations']) == 2
    expected_scripts = ('score_full_manifest.py', 'score_low_level.py')
    for index, invocation in enumerate(closure['invocations']):
        assert invocation['exit_code'] == 0
        assert invocation['manifest_before'] == invocation['manifest_after'] == generation['manifest']
        command = invocation['command']
        assert command[1:3] == ['-B', str(C.ROOT / expected_scripts[index])]
        flags = command[3:]
        assert len(flags) % 2 == 0
        actual = dict(zip(flags[::2], flags[1::2]))
        expected = {'--manifest': str(manifest), '--gallery-manifest': config['input_bindings']['gallery_manifest']['path'],
                    '--output': str(root / 'evaluation' / ('semantic' if index == 0 else 'low'))}
        if index == 0:
            expected.update({'--features': str(root / 'features'), '--gallery-features': config['gallery_features_root']})
        assert actual == expected and len(actual) * 2 == len(flags)
        pin(invocation['log'], root / 'evaluation' / ('invocation_' + str(index) + '.log'), pins, deadline)
    pins[str(closure_path)] = C.sha(closure_path)
    return dict(rows=rows, closure=closure, closure_ref=C.ref(closure_path))


def match_arms(contexts):
    assert set(contexts) == set(ARMS)
    reference = contexts['O']['rows']
    assert len(reference) == len(contexts['D']['rows'])
    for left, right in zip(reference, contexts['D']['rows']):
        for key in IDENTITY:
            assert left[key] == right[key], ('D/O identity mismatch', key)
        assert left['generated_path'] != right['generated_path']
    return reference


def verify_targets(config, rows, pins, deadline, *, stimuli=1660):
    """Replay target labels/paths against the locked blind-to-target join inputs."""
    import json
    mappings = {}
    for key in ('gallery_manifest', 'V_targets'):
        item = config['input_bindings'][key]
        path = pin(item, item['path'], pins, deadline)
        with path.open() as stream:
            records = [json.loads(line) for line in stream]
        mapping = {row['stimulus_id']: row for row in records}
        assert len(mapping) == len(records) == stimuli
        mappings[key] = mapping
    gallery, targets = mappings['gallery_manifest'], mappings['V_targets']
    assert set(gallery) == set(targets) == {row['stimulus_id'] for row in rows}
    for row in rows:
        target, actual = targets[row['stimulus_id']], gallery[row['stimulus_id']]
        assert row['array_row'] == int(target['array_row'])
        assert row['concept_id'] == target['concept_id'] == actual['concept_id']
        assert row['target_image_sha256'] == target['image_sha256'] == actual['image_sha256']
        assert row['target_image_path'] == row['score_target_path'] == actual['image_path']


def read_values(context, deadline):
    """Called only after both complete closures and all cross-arm identities pass."""
    tables = {}
    for branch, item in context['closure']['outputs'].items():
        check_time(deadline)
        with Path(item['table']['path']).open(newline='') as stream:
            tables[branch] = list(csv.DictReader(stream))
        if branch == 'low':
            assert all(row['status'] == 'OK' for row in tables[branch])
    return join_scores(context['rows'], tables['semantic'], tables['low'], METRICS)


def gates(effects):
    for metric in METRICS:
        assert math.isfinite(effects[metric]['mean'])
        assert len(effects[metric]['CI95_subject_and_concept_bootstrap']) == 2
        assert all(math.isfinite(value) for value in effects[metric]['CI95_subject_and_concept_bootstrap'])
    semantic = (effects['DINO2AFC']['mean'] >= .01
        and effects['DINO2AFC']['CI95_subject_and_concept_bootstrap'][0] > 0
        and effects['SEED']['CI95_subject_and_concept_bootstrap'][0] > 0
        and effects['ObjectF1']['mean'] >= 0)
    low = effects['SSIM_Y']['mean'] >= 0 and effects['PixCorr_RGB']['mean'] >= 0
    return dict(semantic_gate=bool(semantic), mean_low_level_no_regression_guard=bool(low),
                provisional_efficacy_signal=bool(semantic and low))


def statistics(values, reference, deadline):
    subjects = sorted({row['subject'] for row in reference})
    concepts = sorted({row['concept_id'] for row in reference})
    delta = [{metric: right[metric] - left[metric] for metric in METRICS}
             for left, right in zip(values['O'], values['D'])]
    stats, matrices = {}, {}
    for name, rows in (('O', values['O']), ('D', values['D']), ('D_minus_O', delta)):
        stats[name], matrices[name] = crossed_statistics(rows, reference, subjects, concepts, deadline=deadline)
    return stats, matrices, delta, subjects, concepts


def compare(config):
    deadline = time.monotonic() + MAX_SECONDS
    pins, inodes, contexts = {}, set(), {}
    for arm in ARMS:
        for directory in ('train', 'postfit_T32', 'images', 'artifact', 'features', 'features_guard', 'evaluation'):
            assert (C.DATA / 'arms' / arm / directory / 'completion.json').is_file(), ('Missing arm completion', arm, directory)
    # No score parser is invoked until BOTH arms, every SHA and every identity pass.
    for arm in ARMS:
        contexts[arm] = verify_arm(config, arm, pins, inodes, deadline)
    reference = match_arms(contexts)
    assert len(reference) == 16600
    assert len({row['subject'] for row in reference}) == 10 and len({row['concept_id'] for row in reference}) == 166
    verify_targets(config, reference, pins, deadline)
    values = {arm: read_values(contexts[arm], deadline) for arm in ARMS}
    stats, matrices, delta, subjects, concepts = statistics(values, reference, deadline)
    output = C.DATA / OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=False)
    def budget():
        check_time(deadline)
        assert C.bytes_used(output) <= MAX_BYTES, 'D/O metadata exceeds fixed 64MiB budget'
    differences = output / 'all16600_D_minus_O.csv.gz'
    identity_fields = [*IDENTITY, 'generated_path', 'reference_generated_path', 'generated_sha256', 'reference_generated_sha256']
    with gzip.open(differences, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*identity_fields, *[prefix + metric for metric in METRICS for prefix in ('candidate_', 'reference_', 'delta_')]])
        writer.writeheader()
        for index, (row, other, difference) in enumerate(zip(contexts['D']['rows'], reference, delta)):
            record = {key: row[key] for key in IDENTITY}
            record.update(generated_path=row['generated_path'], reference_generated_path=other['generated_path'],
                          generated_sha256=row['generated_sha256'], reference_generated_sha256=other['generated_sha256'])
            for metric in METRICS:
                record.update({prefix + metric: value for prefix, value in (('candidate_', values['D'][index][metric]),
                    ('reference_', values['O'][index][metric]), ('delta_', difference[metric]))})
            writer.writerow(record)
            if index % 256 == 0:
                budget()
    strata = output / 'all_O_D_and_difference_strata.csv.gz'
    n = 0
    with gzip.open(strata, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['entry', 'stratum', 'subject', 'concept_id', 'queries', *METRICS])
        writer.writeheader()
        for name, matrix in matrices.items():
            for row in stratum_rows(name, matrix, subjects, concepts, 10, METRICS):
                writer.writerow(row)
                n += 1
                if n % 256 == 0:
                    budget()
    assert n == 3 * (10 + 166 + 10 * 166)
    for path, digest in pins.items():
        check_time(deadline)
        assert C.sha(path) == digest, ('Input changed during comparison', path)
    budget()
    summary = dict(status='FULL16600_DO_EFFICACY_COMPARISON_COMPLETE', schema=OUTPUT_NAME,
        study_core_sha256=C.core(config), comparison_source=C.ref(__file__),
        queries_per_arm=16600, subjects=10, concepts=166, metrics=list(METRICS),
        arm_statistics={arm: stats[arm] for arm in ARMS}, paired_differences={'D_minus_O': stats['D_minus_O']},
        **gates(stats['D_minus_O']), evaluation_closures={arm: contexts[arm]['closure_ref'] for arm in ARMS},
        all_differences=C.ref(differences), strata=C.ref(strata), stratum_rows=n,
        fixed_seed=1729, bootstrap_draws=1000, maximum_seconds=MAX_SECONDS, maximum_output_bytes=MAX_BYTES,
        mechanism_controls_status='WITHDRAWN_BY_USER', mechanism_result='UNKNOWN',
        functional_mechanism_probe_status='WITHDRAWN_BY_USER', mechanism_confirmed=False,
        original_four_arm_summary_modified=False, scope='D/O exposed-R4 development efficacy only',
        low_level_guard_is_not_human_quality_noninferiority=True, training_seed_variability_covered=False,
        independent_confirmation=False, SOTA_established=False, automatic_new_candidate=False, GPU_used=False)
    C.atomic(output / 'summary.json', summary)
    budget()
    return summary


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    config = C.read(args.registration)
    C.validate(config)
    # Original study admission/source pins remain evidence; no A/U closure is required.
    C.require_admission(config)
    compare(config)
    print('FULL16600_DO_EFFICACY_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
