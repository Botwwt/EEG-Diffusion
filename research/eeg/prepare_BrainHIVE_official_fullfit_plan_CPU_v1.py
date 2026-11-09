"""Metadata-only official full-fit planning; no method lock, fit or launch.

Strictly reads four pinned metadata inputs. It never imports model libraries,
reads EEG/image/weight payloads, opens a score table, launches a child process,
or creates an executable recipe/admission. New OOF checkpoints and conditions
are pending artifacts, never inferred from the existing protected-T models.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import time

ROOT = Path(__file__).resolve().parent
SEED = 1729
STATUS = 'PLAN_ONLY_NO_METHOD_LOCK_NO_FIT_NO_AUTO_LAUNCH'
MAX_SECONDS = 600
MAX_BYTES = 32 << 20
SUBJECTS = tuple('sub-' + str(i).zfill(2) for i in range(1, 11))
CHANNELS = ('P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2')
COLUMNS = ('dataset_split', 'array_row', 'stimulus_id', 'concept_id', 'local_concept_id',
    'image_zip', 'image_member', 'image_bytes', 'image_sha256', 'alignment', 'independent_event_label_replay')
PINS = {
    'image_manifest': ('91e75aa52fbb0addc004eeb38a87aa237f48e0d6bdbd5f5cc3d0d0e839d1b145', 5360147),
    'bindings': ('7acfb0bdc3f6627150f8f320e21ca6cdc7f04ff9114ba498496ba4f8c458fa66', 5535),
    'source_split': ('1969d664555d08399e77a2d76c011f19eadbda1ce251f9c72d72c94de3910ad2', 281314),
    'master_policy': ('707cdd48fed6c3359e0d862fecee2b92b5f437c62bf8e65a3765a6d0f39348ed', 54504),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size)


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def private_metadata_path(value):
    path = PurePosixPath(value)
    assert path.is_absolute() and '..' not in path.parts
    return path


def validate_metadata(rows, bindings, split):
    """Full actual-set invariants, independent of fixed whole-file SHA checks."""
    assert split['version'] == 'eeg_full10_concept_protocol_v1' and split['training_seed'] == SEED
    assert split['number_of_training_seeds'] == 1 and tuple(split['subjects']) == SUBJECTS
    assert split['final_independent_H'] == []
    assert split['official_test_status'] == 'previously exposed benchmark; not independent confirmation'
    assert bindings['train_stimuli'] == 16540 and bindings['test_stimuli'] == 200
    assert bindings['image_manifest_sha256'] == PINS['image_manifest'][0]
    assert bindings['split_manifest_sha256'] == PINS['source_split'][0]
    assert tuple(sorted(bindings['subjects'])) == SUBJECTS
    assert tuple(bindings['selected_channels']) == CHANNELS
    assert bindings['selected_indices'] == list(range(46, 63))
    assert bindings['additional_per_channel_zscore'] is False
    assert digest(bindings['channels_source_sha256'])
    assert split['official_test_protocols']['R80'] == list(range(80))
    for subject, item in bindings['subjects'].items():
        private_metadata_path(item['train_path']); private_metadata_path(item['test_R80']['path'])
        assert digest(item['train_sha256']) and digest(item['test_R80']['sha256'])
        assert item['train_repeats'] == 4 and item['test_repeats'] == 80
    assert len(rows) == 16740 and all(set(row) == set(COLUMNS) for row in rows)
    assert {row['dataset_split'] for row in rows} == {'train', 'test'}
    by_split = {}
    for name, count, concepts, per_concept in (('train', 16540, 1654, 10), ('test', 200, 200, 1)):
        selected = [dict(row, array_row=int(row['array_row']), image_bytes=int(row['image_bytes']))
            for row in rows if row['dataset_split'] == name]
        selected.sort(key=lambda row: row['array_row'])
        assert len(selected) == count and [r['array_row'] for r in selected] == list(range(count))
        assert len({r['stimulus_id'] for r in selected}) == count
        counts = Counter(r['concept_id'] for r in selected)
        assert len(counts) == concepts and set(counts.values()) == {per_concept}
        local_by_concept = {}
        for row in selected:
            assert row['stimulus_id'] == PurePosixPath(row['stimulus_id']).name
            assert row['stimulus_id'] and row['concept_id'] and row['local_concept_id']
            assert row['image_bytes'] > 0 and digest(row['image_sha256'])
            private_metadata_path(row['image_zip'])
            member = PurePosixPath(row['image_member'])
            assert not member.is_absolute() and '..' not in member.parts
            # The official test archive numbers its 200 local concepts
            # independently; concept_id remains the global split identifier.
            assert member.name == row['stimulus_id'] and row['local_concept_id'] in member.parts
            assert row['alignment'] == 'author_release_order_plus_metadata_and_zip_receipt'
            assert row['independent_event_label_replay'] == 'False'
            assert row['concept_id'] not in local_by_concept or local_by_concept[row['concept_id']] == row['local_concept_id']
            local_by_concept[row['concept_id']] = row['local_concept_id']
        assert len(set(local_by_concept.values())) == concepts
        by_split[name] = selected
    train, test = by_split['train'], by_split['test']
    assert {r['concept_id'] for r in train}.isdisjoint(r['concept_id'] for r in test)
    assert {r['stimulus_id'] for r in train}.isdisjoint(r['stimulus_id'] for r in test)
    old_groups = split['image_rows']
    assert set(old_groups) == {'T', 'D', 'V'}
    assert [len(old_groups[k]) for k in ('T', 'D', 'V')] == [13230, 1650, 1660]
    all_old = [row for group in old_groups.values() for row in group]
    assert len(set(all_old)) == len(all_old) == 16540 and set(all_old) == set(range(16540))
    group_by_row = {}
    for group in ('T', 'D', 'V'):
        concepts = set(split['train_concepts'][group])
        assert {train[i]['concept_id'] for i in old_groups[group]} == concepts
        for index in old_groups[group]:
            group_by_row[index] = group
    assert len(split['oof_fold_by_concept']) == 1323
    assert set(split['oof_fold_by_concept']) == set(split['train_concepts']['T'])
    assert set(split['oof_fold_by_concept'].values()) == {0, 1}
    assert set(split['d_fit_concepts']) | set(split['d_check_concepts']) == set(split['train_concepts']['D'])
    assert set(split['d_fit_concepts']).isdisjoint(split['d_check_concepts'])
    assert len(split['d_fit_concepts']) == 82 and len(split['d_check_concepts']) == 83
    assert len(set(r['image_sha256'] for r in train)) == 16540
    return train, test, group_by_row


def concept_folds(train):
    concepts = sorted({r['concept_id'] for r in train})
    assert len(concepts) == 1654
    # Independent final-phase partition over ALL1654 concepts. It does not
    # extend the incomplete protected-T map or move single images across folds.
    random.Random(SEED).shuffle(concepts)
    fold_by_concept = {concept: index % 2 for index, concept in enumerate(concepts)}
    folds = []
    for held in (0, 1):
        held_concepts = sorted(c for c, fold in fold_by_concept.items() if fold == held)
        fit_concepts = sorted(c for c, fold in fold_by_concept.items() if fold != held)
        held_rows = [r['array_row'] for r in train if fold_by_concept[r['concept_id']] == held]
        fit_rows = [r['array_row'] for r in train if fold_by_concept[r['concept_id']] != held]
        assert len(held_concepts) == len(fit_concepts) == 827
        assert len(held_rows) == len(fit_rows) == 8270
        assert set(held_concepts).isdisjoint(fit_concepts) and set(held_rows).isdisjoint(fit_rows)
        assert set(held_rows) | set(fit_rows) == set(range(16540))
        folds.append(dict(fold=held, role='official_oof' + str(held), held_concepts=held_concepts,
            fit_concepts=fit_concepts, training_array_rows=fit_rows, exported_OOF_array_rows=held_rows,
            unique_fit_concepts=827, unique_fit_images=8270, unique_exported_OOF_images=8270))
    return fold_by_concept, folds


def tasks_for(folds, bindings):
    tasks = []
    for subject in SUBJECTS:
        for fold in folds:
            tasks.append(dict(task_id=subject + '/' + fold['role'], subject=subject, role=fold['role'],
                held_fold=fold['fold'], training_seed=SEED, independent_training_replicate=False,
                training_array_rows=fold['training_array_rows'], exported_OOF_array_rows=fold['exported_OOF_array_rows'],
                training_concepts=fold['fit_concepts'], held_concepts=fold['held_concepts'],
                train_EEG_metadata=dict(path=bindings['subjects'][subject]['train_path'],
                    sha256=bindings['subjects'][subject]['train_sha256'], physical_repeat_count=4,
                    expected_full_tensor_shape=[16540, 63, 250], selected_indices=list(range(46, 63)),
                    expected_model_input_shape=[8270, 4250], payload_reverified_here=False),
                epochs=25, batch_size=1024, expected_updates=225, expected_training_exposures=206750,
                final_batch_size=78, checkpoint_selection='fixed_final25epochs_no_official_benchmark_selection',
                status='PENDING_NEW_OFFICIAL_CONCEPT_OOF_FIT_AND_EXPORT', checkpoint_path=None,
                checkpoint_sha256=None, condition_cache_path=None, condition_cache_sha256=None,
                existing_protected_T_OOF_reused=False, optimizer_created=False, parameter_updates=0))
    assert len(tasks) == len({task['task_id'] for task in tasks}) == 20
    return tasks


def query_records(train, fold_by_concept, folds, old_groups):
    positions = {fold['fold']: {row: position for position, row in enumerate(fold['exported_OOF_array_rows'])}
        for fold in folds}
    for subject_index, subject in enumerate(SUBJECTS):
        for row in train:
            index = row['array_row']; held = fold_by_concept[row['concept_id']]
            yield dict(query_index=subject_index * 16540 + index, subject=subject,
                dataset_split='official_train', array_row=index, stimulus_id=row['stimulus_id'], concept_id=row['concept_id'],
                EEG_repeat_protocol='R4', physical_repeat_count=4, held_fold=held,
                encoder_task_id=subject + '/official_oof' + str(held),
                future_OOF_cache_row=positions[held][index], previous_development_group=old_groups[index],
                actual_condition_available=False, encoder_checkpoint_path=None, encoder_checkpoint_sha256=None,
                condition_cache_path=None, condition_cache_sha256=None,
                image_target_permission='official_training_supervision_only_after_method_lock',
                official_benchmark_query=False, training_seed=SEED)


def build_metadata(rows, bindings, split, input_refs):
    train, test, old_groups = validate_metadata(rows, bindings, split)
    fold_by_concept, folds = concept_folds(train)
    tasks = tasks_for(folds, bindings)
    native_training = dict(training_seed=SEED, training_replicates=1,
        class_name='main.models_brain.BrainMLP', input_dimensions=4250, output_dimensions=1024,
        num_layers=1, dropout=.1, model_parameters=6452224, epochs=25, batch_size=1024,
        optimizer='AdamW', learning_rate=.0005, betas=[.9, .995], weight_decay=.05,
        exclude_bias_and_LayerNorm_from_decay=True, scheduler='cosine; optimizer before scheduler',
        warmup_steps=0, gradient_clip=1., seed_fixed_before_model_initialization=True,
        shuffle_policy='existing deterministic training-seed-plus-one derivation; not another replicate',
        objective='symmetric cosine InfoNCE against frozen F in original training mode',
        frozen_visual_teacher_training_mode='train', frozen_visual_teacher_dropout=.1,
        frozen_visual_weights=True, deterministic_F_cache_substituted_for_encoder_training=False,
        reference_implementation='train_BrainHIVE_parent_v2.py; new final-phase role/output support still required')
    plan = dict(schema='BrainHIVE_official_fullfit_metadata_plan_v1', status=STATUS,
        method_locked=False, fitting_admitted=False, formal_training_admitted=False,
        automatic_launch_enabled=False, executable_commands_generated=False,
        new_parent_fits_started=0, new_component_fits_started=0, GPU_jobs_started=0,
        tensor_or_image_or_weight_payloads_read=0, score_tables_read=0, input_metadata=input_refs,
        authoritative_policy_section='EEG_SOTA_AUTORESEARCH_MASTER_PROMPT.md#4.3',
        official_training=dict(concepts=1654, images=16540, subjects=list(SUBJECTS), unique_OOF_queries=165400,
            physical_EEG_repeat_count=4, selected_channels=list(CHANNELS), selected_indices=list(range(46, 63)),
            concept_OOF_folds=2, OOF_encoder_tasks=20,
            all_previous_T_Dfit_Dcheck_V_rows_included=True,
            previous_groups=dict(T_images=13230, Dfit_images=820, Dcheck_images=830, V_images=1660),
            existing_protected_T132300_queries_are_not_official_full165400=True),
        parent_training=native_training,
        component_training=dict(training_seed=SEED, subjects=10, unique_training_queries=165400,
            unique_training_images=16540, passes=3, batch_size=16, final_batch_size=8,
            expected_exposures=496200, expected_updates=31014, completed_updates=0,
            selection='future_fixed_final_update; no official-benchmark checkpoint selection',
            final_phase_latents_and_OOF_inputs='PENDING_NEW_FULL16540_IMAGE_AND_165400_QUERY_CONTRACTS',
            actual_native_objective_interface='method-lock pending; no new recipe/admission is issued here'),
        baseline_opportunity=dict(minimum_native_roles=['parent', 'O', 'P', 'PN', 'S'],
            native_heads_each_receive_full165400_queries_three_passes31014_updates=True,
            parent_full16540_published_models_reuse='PENDING_COMPLETE_AUDIT; existing full-T models are insufficient',
            parent_checkpoint_sha256=None, additional_baseline_list_locked=False,
            additional_required_review=['old bounded_v1 ordinary', 'old bounded_v1 risk',
                'strongest retained internal baselines', 'public and cross-architecture strong references with protocol/cost audit'],
            additional_strong_baselines_may_not_be_ignored=True, no_winner_chosen_by_this_plan=True,
            failed_PN_cannot_be_relabelled_as_P=True, extra_main_candidate_search_authorized=False),
        calibration_permission=dict(status='PENDING_PRE_BENCHMARK_METHOD_LOCK',
            old_Dfit_is_independent_after_fullfit=False, old_Dcheck_is_independent_after_fullfit=False,
            old_V_is_independent_after_fullfit=False,
            final_strength_and_risk_calibration_policy_must_be_registered_before_official_readout=True,
            official_test_or_gallery_targets_for_calibration=False,
            current_development_alpha_automatically_reused=False,
            any_new_nested_train_only_calibration_requires_equal_baseline_permissions=True),
        official_benchmark=dict(repeat_protocol='R80', physical_repeat_indices=list(range(80)),
            subjects=list(SUBJECTS), unique_gallery_images=200, queries=2000, images_per_query=1,
            best_of_N=False, test_tensor_expected_shape_per_subject=[200, 63, 250],
            status='PREVIOUSLY_EXPOSED_BENCHMARK_NOT_INDEPENDENT_H', independent_H=False,
            confirmation_level_SOTA_established=False, per_query_input='EEG only; no target/concept/retrieval answer at inference',
            paper_seven_metrics_and_known_evaluator_differences_must_be_reported=True,
            author_ten_reconstruction_average_must_be_separate_from_single_image_protocol=True,
            official_test_concepts=sorted(r['concept_id'] for r in test),
            no_official_test_query_used_in_any_encoder_fit_or_component_OOF=True),
        pending_method_lock_items=['Close current full O/P/PN/S development comparison without selecting incomplete scores',
            'Lock method, training hyperparameters, final-head rule and all required strong baseline roles',
            'Lock final-phase strength/calibration permissions before any official result access',
            'Implement and resource-audit new official OOF encoder/export/component/latent/inference/evaluation entries',
            'Audit reusable published full16540 parents; never reuse protected-T OOF as official OOF',
            'Preserve fixed single seed and exposed-benchmark limitations'],
        interpretation=dict(metadata_coverage_only=True, successful_fit_or_image_quality_established=False,
            component_independent_increment_established=False, SOTA_established=False,
            raw_event_identity_replay_independently_verified=False,
            alignment_limit=split['alignment_limit']))
    assert plan['component_training']['expected_updates'] == 3 * math.ceil(165400 / 16)
    assert all(task['expected_updates'] == 25 * math.ceil(len(task['training_array_rows']) / 1024) for task in tasks)
    folds_document = dict(schema='BrainHIVE_official_fullfit_concept_folds_plan_v1', status=STATUS,
        training_seed=SEED, method_locked=False, official_train_only=True,
        mapping_rule='sort1654concepts; Python Random1729 shuffle; alternating positions0/1',
        concept_fold_sha256=hashlib.sha256(canonical(fold_by_concept)).hexdigest(),
        fold_by_concept=fold_by_concept, folds=folds,
        old_T_only_fold_map_reused=False, official_test_concepts_in_fold_map=False)
    tasks_document = dict(schema='BrainHIVE_official_fullfit_encoder_tasks_plan_v1', status=STATUS,
        method_locked=False, executable_commands=[], task_count=20, tasks=tasks,
        actual_completed_models=0, future_model_SHA_is_pending_not_fabricated=True)
    return plan, folds_document, tasks_document, query_records(train, fold_by_concept, folds, old_groups)


def pinned_inputs(paths):
    refs = {}
    for key, expected in PINS.items():
        path = Path(paths[key])
        assert path.is_file() and path.stat().st_size == expected[1], (key, 'pinned metadata size differs')
        item = ref(path)
        assert item['sha256'] == expected[0], (key, 'pinned metadata SHA differs')
        refs[key] = item
    with Path(paths['image_manifest']).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == COLUMNS
        rows = list(reader)
    bindings = json.loads(Path(paths['bindings']).read_text(encoding='utf-8'))
    split = json.loads(Path(paths['source_split']).read_text(encoding='utf-8'))
    master = Path(paths['master_policy']).read_text(encoding='utf-8')
    assert '### 4.3 划分与最终测试' in master
    assert '重新生成该阶段需要的折外条件/误差缓存' in master
    return rows, bindings, split, refs


def check_budget(output, deadline):
    assert time.monotonic() < deadline, 'Finite metadata-only planning time expired'
    used = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    assert used <= MAX_BYTES, 'Metadata-only output cap exceeded; partial proof retained'
    return used


def json_write(path, value):
    with path.open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def write_query_map(path, records, output, deadline):
    count = 0; subjects = Counter(); folds = Counter(); old = Counter(); identity = hashlib.sha256()
    with path.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
        for row in records:
            assert row['query_index'] == count
            encoded = canonical(row) + b'\n'; compressed.write(encoded); identity.update(encoded)
            count += 1; subjects[row['subject']] += 1; folds[row['held_fold']] += 1; old[row['previous_development_group']] += 1
            if count % 4096 == 0:
                check_budget(output, deadline)
    assert count == 165400 and dict(subjects) == {s: 16540 for s in SUBJECTS}
    assert dict(folds) == {0: 82700, 1: 82700}
    assert dict(old) == {'T': 132300, 'D': 16500, 'V': 16600}
    # Decode every gzip row again, enforcing CRC and exact subject/stimulus row
    # order. This proves a complete plan table, never real condition tensors.
    decoded = hashlib.sha256(); read_count = 0
    with gzip.open(path, 'rb') as stream:
        for line in stream:
            record = json.loads(line)
            assert record['query_index'] == read_count
            assert record['subject'] == SUBJECTS[read_count // 16540]
            assert record['array_row'] == read_count % 16540
            assert record['condition_cache_sha256'] is None and record['encoder_checkpoint_sha256'] is None
            decoded.update(line); read_count += 1
            if read_count % 4096 == 0:
                check_budget(output, deadline)
    assert read_count == count and decoded.digest() == identity.digest()
    return dict(rows=count, per_subject=dict(subjects), per_fold=dict(folds), previous_group_queries=dict(old),
        uncompressed_rows_sha256=identity.hexdigest(), gzip_all_rows_and_CRC_redecoded=True,
        actual_cached_conditions=0)


def prepare(paths, output):
    started = time.monotonic(); deadline = started + MAX_SECONDS
    output = Path(output)
    assert not output.exists(), 'Keep any prior plan or partial result; use a new explicit output directory'
    assert not any(Path(path).resolve().is_relative_to(output.resolve()) for path in paths.values()), 'Output contains an input path'
    rows, bindings, split, input_refs = pinned_inputs(paths)
    plan, folds, tasks, records = build_metadata(rows, bindings, split, input_refs)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for name, value in (('official_fullfit_plan.json', plan), ('official_fullfit_two_folds.json', folds),
            ('official_fullfit_encoder_tasks.json', tasks)):
        path = output / name; json_write(path, value); artifacts.append(ref(path))
    query_path = output / 'official_fullfit_OOF_query_map.jsonl.gz'
    coverage = write_query_map(query_path, records, output, deadline); artifacts.append(ref(query_path))
    for key, old in input_refs.items():
        assert ref(paths[key]) == old, 'Input changed during metadata planning'
    result = dict(schema='BrainHIVE_official_fullfit_metadata_preparation_v1', status=STATUS,
        metadata_plan_complete=True, official_fullfit_completed=False, method_locked=False,
        fitting_admitted=False, formal_training_admitted=False, automatic_launch_enabled=False,
        executable_commands_generated=False, actual_GPU_jobs=0, actual_parameter_updates=0,
        actual_new_OOF_models=0, actual_new_OOF_condition_tensors=0, score_tables_read=0,
        tensor_image_weight_payloads_opened=0, input_metadata=input_refs, artifacts=artifacts,
        source=ref(Path(__file__)), complete_plan_coverage=coverage,
        expected_future_encoder_tasks=20, expected_future_component_updates=31014,
        expected_future_R80_queries=2000, independent_H=False, SOTA_established=False,
        budget=dict(maximum_seconds=MAX_SECONDS, maximum_output_bytes=MAX_BYTES,
            seconds=time.monotonic() - started, output_bytes_before_completion=check_budget(output, deadline)))
    json_write(output / 'preparation.json', result)
    check_budget(output, deadline)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-manifest', required=True, type=Path)
    parser.add_argument('--bindings', required=True, type=Path)
    parser.add_argument('--source-split', required=True, type=Path)
    parser.add_argument('--master-policy', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = prepare({key: getattr(args, key) for key in PINS}, args.output)
    print(json.dumps(dict(status=result['status'], output=str(args.output), planned_OOF_queries=165400,
        planned_encoder_tasks=20, actual_fits=0, actual_GPU_jobs=0, method_locked=False)))


if __name__ == '__main__':
    main()
