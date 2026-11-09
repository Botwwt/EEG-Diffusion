"""Synthetic CPU-only evidence checks; never a real 16600-query completion."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import compare_BrainHIVE_DO_efficacy_CPU_20261026_v1 as M


def rejected(fn):
    try:
        fn()
    except (AssertionError, ValueError, KeyError, OSError):
        return
    raise AssertionError('Invalid fixture accepted')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return M.C.ref(path)


def binary(path, payload=b'fixture only'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return M.C.ref(path)


def fixture(root):
    """Four real CSV joins and files per arm, with synthetic closure metadata."""
    gallery = write(root / 'gallery/config.json', dict(DINO_weight_sha256='fixture', dino_preprocessing={},
        SEED_source_commit='fixture', ObjectF1_grid=[], semantic_model_receipts={}))
    config = dict(input_bindings={'gallery_feature_config': gallery, 'gallery_manifest': {'path': str(root / 'gallery.jsonl')}},
                  maximum_CPU_extra_bytes=10 << 30, maximum_GPU_bytes=20 << 30, gallery_features_root=str(root / 'gallery'))
    target = binary(root / 'target.bin')
    manifests = {}
    for arm in M.ARMS:
        own = root / 'arms' / arm
        refs = {key: binary(own / 'train' / name) for key, name in
                (('checkpoint', 'component_final.pt'), ('calibration', 'calibration.json'), ('provenance', 'input_provenance.json'))}
        write(own / 'train/completion.json', dict(status='PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE', arm=arm,
            study_core_sha256='fixture_core', updates=24807, passes=3, parameters=525568,
            primary_exposures=396900, paired_exposures=396900, **refs))
        post = []
        for index in range(2):
            image = binary(own / 'postfit_T32/pixels/sub-01' / ('t' + str(index) + '.webp'), bytes([index]))
            post.append(dict(query_id='t' + str(index), subject='sub-01', stimulus_id='t' + str(index) + '.jpg',
                global_query_index=index, arm=arm, study_core_sha256='fixture_core',
                component_checkpoint_sha256=refs['checkpoint']['sha256'], generated_path=image['path'], generated_sha256=image['sha256']))
        post_path = own / 'postfit_T32/inference_manifest.jsonl'
        post_path.write_text(''.join(json.dumps(row) + '\n' for row in post))
        write(own / 'postfit_T32/completion.json', dict(status='PAIRED_GENERATION_COMPLETE', queries=2,
            arm=arm, split='T32', study_core_sha256='fixture_core', checkpoint=refs['checkpoint'],
            manifest=M.C.ref(post_path), normal_inference_target_access=False))
        rows = []
        for index, (subject, concept) in enumerate((('s1', 'c1'), ('s1', 'c2'), ('s2', 'c1'), ('s2', 'c2'))):
            image = binary(own / 'images/pixels' / subject / ('x' + str(index) + '.webp'), bytes([index + 2]))
            rows.append(dict(query_id='q' + str(index), subject=subject, concept_id=concept, stimulus_id='x' + str(index) + '.jpg',
                array_row=index, image_seed=1729 + index, initial_UNet_input_sha256='noise' + str(index),
                target_image_sha256=target['sha256'], target_image_path=target['path'], score_target_path=target['path'],
                target_mapping_attached_after_inference=True, generated_path=image['path'], generated_sha256=image['sha256'],
                study_core_sha256='fixture_core', component_checkpoint_sha256=refs['checkpoint']['sha256']))
        manifest_path = own / 'images/manifest.jsonl'
        manifest_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        generation = dict(checkpoint=refs['checkpoint'], manifest=M.C.ref(manifest_path), normal_inference_target_access=False)
        write(own / 'images/completion.json', generation)
        manifests[arm] = (rows, generation)
        artifact = write(own / 'artifact/completion.json', dict(status='FULL16600_PAIRED_ARTIFACT_AND_EEG_IDENTITY_AUDIT_COMPLETE',
            arm=arm, queries=4, full_denominator=4, counts={'verified': 4}, failures=[], manifest=generation['manifest'],
            table=binary(own / 'artifact/per_query.csv.gz')))
        feature_config = write(own / 'features/config.json', M.C.read(gallery['path']) | {'manifest_sha256': generation['manifest']['sha256']})
        feature = write(own / 'features/completion.json', dict(status='COMPLETED', queries=4, all_queries_used=True,
            query_subsampling=False, manifest_sha256=generation['manifest']['sha256']))
        guard = write(own / 'features_guard/completion.json', dict(status='UNCHANGED_EXTRACTOR_WITH_REAL_PEAK_GUARD_COMPLETE',
            queries=4, engineering_only=False, feature_completion=feature, peak_extra_CPU_bytes=1, peak_GPU_bytes=1,
            extractor_sha256=M.C.sha(M.C.ROOT / 'extract_image_features.py')))
        outputs = {}
        for branch, metrics in (('semantic', M.METRICS[:8]), ('low', M.METRICS[8:])):
            folder = own / 'evaluation' / branch
            folder.mkdir(parents=True)
            path = folder / 'per_query.csv'
            fields = ['query_id', 'subject', 'stimulus_id', 'generated_path', 'score_target_path', 'status', *metrics]
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    delta = .02 if arm == 'D' else 0.
                    writer.writerow({key: row[key] for key in fields[:5]} | {'status': 'OK'} | {key: .5 + delta for key in metrics})
            outputs[branch] = dict(table=M.C.ref(path), summary=write(folder / 'summary.json', {'fixture_only': True}))
        shards = {}
        for stage in ('DINO', 'EffNet', 'CapSim', 'ObjectF1'):
            shard = binary(own / 'features' / stage / '00000000.pt')
            shards[shard['path']] = shard['sha256']
        invocations = []
        for index, script in enumerate(('score_full_manifest.py', 'score_low_level.py')):
            flags = {'--manifest': str(manifest_path), '--gallery-manifest': config['input_bindings']['gallery_manifest']['path'],
                '--output': str(own / 'evaluation' / ('semantic' if index == 0 else 'low'))}
            if index == 0:
                flags.update({'--features': str(own / 'features'), '--gallery-features': config['gallery_features_root']})
            command = ['python', '-B', str(M.C.ROOT / script)] + [item for pair in flags.items() for item in pair]
            invocations.append(dict(exit_code=0, command=command, manifest_before=generation['manifest'],
                manifest_after=generation['manifest'], log=binary(own / 'evaluation' / ('invocation_' + str(index) + '.log'))))
        write(own / 'evaluation/completion.json', dict(status='FULL16600_REAL_EVALUATOR_INVOCATIONS_CLOSED', arm=arm, queries=4,
            manifest=generation['manifest'], artifact=artifact, feature_completion=feature, feature_config=feature_config,
            feature_guard=guard, feature_shards=shards, outputs=outputs, invocations=invocations))
    for name in ('gallery_manifest', 'V_targets'):
        path = root / (name + '.jsonl')
        with path.open('w') as stream:
            for row in manifests['O'][0]:
                record = dict(stimulus_id=row['stimulus_id'], concept_id=row['concept_id'],
                              image_sha256=row['target_image_sha256'], image_path=row['target_image_path'], array_row=row['array_row'])
                stream.write(json.dumps(record) + '\n')
        config['input_bindings'][name] = M.C.ref(path)
    # Invocation metadata above must bind the final gallery path.
    for arm in M.ARMS:
        path = root / 'arms' / arm / 'evaluation/completion.json'
        closure = M.C.read(path)
        for invocation in closure['invocations']:
            command = invocation['command']
            command[command.index('--gallery-manifest') + 1] = config['input_bindings']['gallery_manifest']['path']
        write(path, closure)
    return config, manifests


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='hive_DO_efficacy_CPU_fixture_') as directory:
        root = Path(directory)
        config, manifests = fixture(root)
        with patch.object(M.C, 'DATA', root), patch.object(M.C, 'core', return_value='fixture_core'):
            def verify(arm, inodes=None):
                return M.verify_arm(config, arm, {}, set() if inodes is None else inodes,
                    time.monotonic() + 30, count=4, post_count=2, rows_reader=lambda c, a: manifests[a])
            inodes = set()
            contexts = {arm: verify(arm, inodes) for arm in M.ARMS}
            reference = M.match_arms(contexts)
            M.verify_targets(config, reference, {}, time.monotonic() + 30, stimuli=4)
            bad_targets = copy.deepcopy(reference)
            bad_targets[0]['concept_id'] = 'wrong'
            rejected(lambda: M.verify_targets(config, bad_targets, {}, time.monotonic() + 30, stimuli=4))
            checks.append('locked_gallery_target_concept_array_row_path_SHA_join_and_wrong_concept_rejection')
            values = {arm: M.read_values(contexts[arm], time.monotonic() + 30) for arm in M.ARMS}
            stats, matrices, delta, subjects, concepts = M.statistics(values, reference, time.monotonic() + 30)
            assert abs(stats['D_minus_O']['DINO2AFC']['mean'] - .02) < 1e-12
            assert M.gates(stats['D_minus_O'])['provisional_efficacy_signal']
            strata = [row for name, matrix in matrices.items() for row in M.stratum_rows(name, matrix, subjects, concepts, 1, M.METRICS)]
            assert len(delta) == 4 and len(strata) == 3 * (2 + 2 + 4)
            checks.append('synthetic_two_arm_files_closures_real_CSV_join_11metrics_1000crossed_draws_and_all_strata')
            bad = copy.deepcopy(contexts)
            bad['D']['rows'][0]['image_seed'] += 1
            rejected(lambda: M.match_arms(bad))
            bad = copy.deepcopy(contexts)
            bad['D']['rows'][0]['initial_UNet_input_sha256'] = 'wrong'
            rejected(lambda: M.match_arms(bad))
            checks.append('wrong_query_seed_and_initial_noise_cross_arm_join_rejected')
            image = Path(manifests['D'][0][0]['generated_path'])
            saved = image.read_bytes()
            image.write_bytes(b'wrong')
            rejected(lambda: verify('D'))
            image.write_bytes(saved)
            checks.append('changed_generated_image_SHA_rejected')
            low = contexts['D']['closure']['outputs']['low']['table']['path']
            payload = Path(low).read_text()
            rows = list(csv.DictReader(payload.splitlines()))
            rows[0]['score_target_path'] = '/wrong'
            rejected(lambda: M.join_scores(manifests['D'][0], list(csv.DictReader(Path(contexts['D']['closure']['outputs']['semantic']['table']['path']).read_text().splitlines())), rows, M.METRICS))
            checks.append('wrong_score_target_join_rejected')
            Path(low).write_text(payload)
            closure_path = root / 'arms/D/evaluation/completion.json'
            closure = closure_path.read_bytes()
            closure_path.unlink()
            with patch.object(M, 'read_values', side_effect=AssertionError('SCORE_READ_BEFORE_BOTH_CLOSURES')) as score_reader:
                rejected(lambda: M.compare(config))
                assert not score_reader.called
            closure_path.write_bytes(closure)
            assert not (root / M.OUTPUT_NAME).exists()
            checks.append('missing_D_closure_rejected_before_any_score_read_or_output_creation')
            rejected(lambda: M.unique_file(image, {(image.stat().st_dev, image.stat().st_ino)}, time.monotonic() + 30))
            checks.append('cross_arm_or_repeated_inode_rejected')
    def effect(value):
        return {metric: {'mean': value, 'CI95_subject_and_concept_bootstrap': [value, value]} for metric in M.METRICS}
    assert M.gates(effect(.02))['provisional_efficacy_signal']
    assert not M.gates(effect(0.))['semantic_gate']
    assert not M.gates(effect(-.02))['provisional_efficacy_signal']
    below = effect(.02)
    below['DINO2AFC']['mean'] = .009999999
    assert not M.gates(below)['semantic_gate']
    low = effect(.02)
    low['SSIM_Y']['mean'] = -.00001
    assert M.gates(low)['semantic_gate'] and not M.gates(low)['provisional_efficacy_signal']
    checks.append('positive_zero_negative_exact_DINO_threshold_and_low_level_guard')
    rejected(lambda: M.check_time(time.monotonic() - 1))
    checks.append('finite_deadline_rejected')
    print(json.dumps(dict(status='DO_EFFICACY_SYNTHETIC_CPU_CHECKS_PASSED', checks=checks,
        fixture_queries_per_arm=4, real_16600_run=False, production_scores_read=False,
        GPU_used=False, new_fit=False, mechanism_established=False, SOTA_established=False)))


if __name__ == '__main__':
    main()
