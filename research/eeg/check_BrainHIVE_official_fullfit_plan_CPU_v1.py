"""CPU checks against the real pinned metadata; no model or data payload reads."""
from collections import Counter
import builtins
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

import prepare_BrainHIVE_official_fullfit_plan_CPU_v1 as P


class OfficialFullfitMetadataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = dict(image_manifest=P.ROOT / 'audit/image_manifest.csv',
            bindings=P.ROOT / 'audit/official_fullfit_inputs_20261022_v1/actual_input_bindings.json',
            source_split=P.ROOT / 'configs/split_manifest.json',
            master_policy=P.ROOT.parent.parent / 'EEG_SOTA_AUTORESEARCH_MASTER_PROMPT.md')
        cls.rows, cls.bindings, cls.split, cls.refs = P.pinned_inputs(cls.paths)
        cls.train, cls.test, cls.groups = P.validate_metadata(cls.rows, cls.bindings, cls.split)
        cls.plan, cls.folds, cls.tasks, _ = P.build_metadata(cls.rows, cls.bindings, cls.split, cls.refs)

    def test_actual_pins_and_full_official_counts(self):
        self.assertEqual(len(self.rows), 16740)
        self.assertEqual(len(self.train), 16540)
        self.assertEqual(len({r['concept_id'] for r in self.train}), 1654)
        self.assertEqual(len(self.test), 200)
        for key, item in self.refs.items():
            self.assertEqual((item['sha256'], item['bytes']), P.PINS[key])
            self.assertEqual(len(item['sha256']), 64)

    def test_actual_test_archive_uses_local_concept_numbers(self):
        test = self.test[0]
        self.assertNotEqual(test['local_concept_id'], test['concept_id'])
        self.assertIn(test['local_concept_id'], test['image_member'])
        self.assertNotIn(test['concept_id'], test['image_member'])
        P.validate_metadata(self.rows, self.bindings, self.split)

    def test_new_balanced_concept_folds_include_previous_D_and_V(self):
        mapping = self.folds['fold_by_concept']
        self.assertEqual(len(mapping), 1654)
        self.assertEqual(Counter(mapping.values()), {0: 827, 1: 827})
        self.assertEqual(set(mapping), {r['concept_id'] for r in self.train})
        self.assertEqual(len(set(mapping) - set(self.split['oof_fold_by_concept'])), 331)
        self.assertTrue(set(self.split['train_concepts']['D']).issubset(mapping))
        self.assertTrue(set(self.split['train_concepts']['V']).issubset(mapping))
        self.assertFalse(self.folds['old_T_only_fold_map_reused'])
        for fold in self.folds['folds']:
            self.assertEqual(len(fold['training_array_rows']), 8270)
            self.assertEqual(len(fold['exported_OOF_array_rows']), 8270)
            self.assertTrue(set(fold['training_array_rows']).isdisjoint(fold['exported_OOF_array_rows']))
            self.assertEqual(set(fold['training_array_rows']) | set(fold['exported_OOF_array_rows']), set(range(16540)))

    def test_fold_reproduction_does_not_change_global_RNG(self):
        state = random.getstate()
        first, first_rows = P.concept_folds(self.train)
        self.assertEqual(state, random.getstate())
        second, second_rows = P.concept_folds(self.train)
        self.assertEqual(first, second)
        self.assertEqual(first_rows, second_rows)

    def test_twenty_tasks_cover_every_subject_and_exclude_held_concepts(self):
        tasks = self.tasks['tasks']
        self.assertEqual(len(tasks), 20)
        self.assertEqual(Counter(t['subject'] for t in tasks), {s: 2 for s in P.SUBJECTS})
        for task in tasks:
            fit = {self.train[i]['concept_id'] for i in task['training_array_rows']}
            held = {self.train[i]['concept_id'] for i in task['exported_OOF_array_rows']}
            self.assertEqual(fit, set(task['training_concepts']))
            self.assertEqual(held, set(task['held_concepts']))
            self.assertEqual(len(fit), 827)
            self.assertTrue(fit.isdisjoint(held))
            self.assertEqual(task['expected_updates'], 225)
            self.assertEqual(task['expected_training_exposures'], 8270 * 25)
            self.assertEqual(task['final_batch_size'], 78)
            self.assertEqual(task['training_seed'], 1729)
            self.assertFalse(task['existing_protected_T_OOF_reused'])
            self.assertEqual(task['parameter_updates'], 0)
            for field in ('checkpoint_path', 'checkpoint_sha256', 'condition_cache_path', 'condition_cache_sha256'):
                self.assertIsNone(task[field])

    def test_every_query_maps_to_a_new_nonfitting_encoder(self):
        mapping = self.folds['fold_by_concept']; folds = self.folds['folds']
        all_tasks = {t['task_id']: t for t in self.tasks['tasks']}
        query_count = Counter(); old_count = Counter()
        for count, record in enumerate(P.query_records(self.train, mapping, folds, self.groups)):
            self.assertEqual(record['query_index'], count)
            task = all_tasks[record['encoder_task_id']]
            row = task['exported_OOF_array_rows'][record['future_OOF_cache_row']]
            self.assertEqual(row, record['array_row'])
            self.assertEqual(record['subject'], task['subject'])
            self.assertEqual(record['held_fold'], mapping[record['concept_id']])
            self.assertEqual(self.train[row]['stimulus_id'], record['stimulus_id'])
            self.assertEqual(record['physical_repeat_count'], 4)
            self.assertFalse(record['actual_condition_available'])
            self.assertFalse(record['official_benchmark_query'])
            self.assertIsNone(record['encoder_checkpoint_sha256'])
            self.assertIsNone(record['condition_cache_sha256'])
            query_count[record['subject']] += 1
            old_count[record['previous_development_group']] += 1
        self.assertEqual(count + 1, 165400)
        self.assertEqual(query_count, {s: 16540 for s in P.SUBJECTS})
        self.assertEqual(old_count, {'T': 132300, 'D': 16500, 'V': 16600})

    def test_original_parent_training_and_full_component_budget(self):
        parent = self.plan['parent_training']; component = self.plan['component_training']
        self.assertEqual((parent['epochs'], parent['batch_size']), (25, 1024))
        self.assertEqual((parent['learning_rate'], parent['betas'], parent['weight_decay']), (.0005, [.9, .995], .05))
        self.assertEqual(parent['frozen_visual_teacher_training_mode'], 'train')
        self.assertEqual(parent['frozen_visual_teacher_dropout'], .1)
        self.assertTrue(parent['frozen_visual_weights'])
        self.assertFalse(parent['deterministic_F_cache_substituted_for_encoder_training'])
        self.assertTrue(parent['seed_fixed_before_model_initialization'])
        self.assertEqual((component['unique_training_queries'], component['passes'], component['expected_exposures']), (165400, 3, 496200))
        self.assertEqual((component['expected_updates'], component['final_batch_size']), (31014, 8))
        self.assertEqual(component['completed_updates'], 0)

    def test_no_method_lock_admission_or_executable_is_issued(self):
        self.assertEqual(self.plan['status'], P.STATUS)
        for key in ('method_locked', 'fitting_admitted', 'formal_training_admitted', 'automatic_launch_enabled', 'executable_commands_generated'):
            self.assertIs(self.plan[key], False)
        self.assertEqual(self.tasks['executable_commands'], [])
        self.assertEqual(self.tasks['actual_completed_models'], 0)
        self.assertEqual(self.plan['new_parent_fits_started'], 0)
        self.assertEqual(self.plan['GPU_jobs_started'], 0)

    def test_all_baseline_and_calibration_permissions_remain_explicit(self):
        baseline = self.plan['baseline_opportunity']; calibration = self.plan['calibration_permission']
        self.assertEqual(baseline['minimum_native_roles'], ['parent', 'O', 'P', 'PN', 'S'])
        self.assertFalse(baseline['additional_baseline_list_locked'])
        self.assertIn('old bounded_v1 ordinary', baseline['additional_required_review'])
        self.assertIn('old bounded_v1 risk', baseline['additional_required_review'])
        self.assertTrue(baseline['additional_strong_baselines_may_not_be_ignored'])
        self.assertTrue(baseline['failed_PN_cannot_be_relabelled_as_P'])
        self.assertFalse(calibration['old_Dfit_is_independent_after_fullfit'])
        self.assertFalse(calibration['old_Dcheck_is_independent_after_fullfit'])
        self.assertFalse(calibration['old_V_is_independent_after_fullfit'])
        self.assertFalse(calibration['official_test_or_gallery_targets_for_calibration'])
        self.assertFalse(calibration['current_development_alpha_automatically_reused'])

    def test_official_R80_is_exposed_single_image_benchmark(self):
        benchmark = self.plan['official_benchmark']
        self.assertEqual(benchmark['physical_repeat_indices'], list(range(80)))
        self.assertEqual((benchmark['unique_gallery_images'], benchmark['queries'], benchmark['images_per_query']), (200, 2000, 1))
        self.assertFalse(benchmark['best_of_N'])
        self.assertFalse(benchmark['independent_H'])
        self.assertFalse(benchmark['confirmation_level_SOTA_established'])
        self.assertTrue(benchmark['no_official_test_query_used_in_any_encoder_fit_or_component_OOF'])
        self.assertTrue(set(benchmark['official_test_concepts']).isdisjoint(self.folds['fold_by_concept']))
        self.assertFalse(self.plan['interpretation']['raw_event_identity_replay_independently_verified'])

    def reject_rows(self, mutator):
        rows = [dict(r) for r in self.rows]
        mutator(rows)
        with self.assertRaises((AssertionError, ValueError, KeyError)):
            P.validate_metadata(rows, self.bindings, self.split)

    def test_missing_or_duplicated_full_training_query_is_rejected(self):
        self.reject_rows(lambda rows: rows.pop(100))
        self.reject_rows(lambda rows: rows[0].update(array_row=rows[1]['array_row']))

    def test_mispaired_archive_or_claimed_raw_event_replay_is_rejected(self):
        self.reject_rows(lambda rows: rows[0].update(image_member='training_images/wrong/' + rows[0]['stimulus_id']))
        self.reject_rows(lambda rows: rows[0].update(independent_event_label_replay='True'))

    def test_missing_subject_R80_mixup_and_wrong_channel_order_are_rejected(self):
        for mutate in (lambda b: b['subjects'].pop('sub-10'),
                lambda b: b['subjects']['sub-01'].update(train_repeats=80),
                lambda b: b.update(selected_indices=list(range(45, 62)))):
            bindings = copy.deepcopy(self.bindings); mutate(bindings)
            with self.assertRaises(AssertionError):
                P.validate_metadata(self.rows, bindings, self.split)

    def test_old_development_exclusions_cannot_hide_final_training_rows(self):
        split = copy.deepcopy(self.split); split['image_rows']['V'].pop()
        with self.assertRaises(AssertionError):
            P.validate_metadata(self.rows, self.bindings, split)
        split = copy.deepcopy(self.split); split['final_independent_H'] = [1]
        with self.assertRaises(AssertionError):
            P.validate_metadata(self.rows, self.bindings, split)

    def test_whole_metadata_tampering_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory(prefix='BrainHIVE_official_pin_') as tmp:
            tmp = Path(tmp); changed = tmp / 'bindings.json'
            payload = self.paths['bindings'].read_bytes()
            changed.write_bytes(payload.replace(b'false', b'False', 1))
            self.assertEqual(changed.stat().st_size, len(payload))
            paths = dict(self.paths, bindings=changed); output = tmp / 'must_not_exist'
            with self.assertRaises(AssertionError):
                P.prepare(paths, output)
            self.assertFalse(output.exists())

    def test_budget_and_existing_partial_output_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='BrainHIVE_official_guard_') as tmp:
            output = Path(tmp)
            with self.assertRaises(AssertionError):
                P.check_budget(output, time.monotonic() - 1)
            marker = output / 'keep_partial'; marker.write_bytes(b'KEEP')
            with self.assertRaises(AssertionError):
                P.prepare(self.paths, output)
            self.assertEqual(marker.read_bytes(), b'KEEP')
            with patch.object(P, 'MAX_BYTES', 3), self.assertRaises(AssertionError):
                P.check_budget(output, time.monotonic() + 10)

    def test_real_complete_output_redecodes_every_gzip_row_without_payload_reads(self):
        # Guard every Path.open in the actual producer. The only allowed reads
        # are the four pinned metadata files, this producer source, and its own
        # newly generated output. In particular /root EEG/ZIP/weights fail here.
        original_open = Path.open; original_builtin_open = builtins.open
        with tempfile.TemporaryDirectory(prefix='BrainHIVE_official_positive_') as tmp:
            output = Path(tmp) / 'plan'
            allowed = {p.resolve() for p in self.paths.values()} | {Path(P.__file__).resolve()}
            opened = []

            def guarded_open(path, mode='r', *args, **kwargs):
                absolute = path.resolve()
                self.assertTrue(absolute in allowed or absolute.is_relative_to(output.resolve()), str(path))
                opened.append(absolute)
                return original_open(path, mode, *args, **kwargs)

            def guarded_builtin_open(path, mode='r', *args, **kwargs):
                absolute = Path(path).resolve()
                self.assertTrue(absolute in allowed or absolute.is_relative_to(output.resolve()), str(path))
                opened.append(absolute)
                return original_builtin_open(path, mode, *args, **kwargs)

            with patch.object(Path, 'open', guarded_open), patch.object(builtins, 'open', guarded_builtin_open):
                result = P.prepare(self.paths, output)
            self.assertTrue(result['metadata_plan_complete'])
            self.assertFalse(result['official_fullfit_completed'])
            self.assertEqual(result['complete_plan_coverage']['rows'], 165400)
            self.assertTrue(result['complete_plan_coverage']['gzip_all_rows_and_CRC_redecoded'])
            self.assertEqual(result['actual_new_OOF_models'], 0)
            self.assertEqual(result['tensor_image_weight_payloads_opened'], 0)
            self.assertLessEqual(sum(p.stat().st_size for p in output.iterdir()), P.MAX_BYTES)
            saved = json.loads((output / 'preparation.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, json.loads(json.dumps(result)))
            for artifact in saved['artifacts']:
                self.assertEqual(P.ref(artifact['path']), artifact)
            decoded = hashlib.sha256(); count = 0
            with gzip.open(output / 'official_fullfit_OOF_query_map.jsonl.gz', 'rb') as stream:
                for line in stream:
                    decoded.update(line); count += 1
            self.assertEqual(count, 165400)
            self.assertEqual(json.loads(line)['query_index'], 165399)
            self.assertEqual(decoded.hexdigest(), result['complete_plan_coverage']['uncompressed_rows_sha256'])
            self.assertTrue(allowed.issubset(opened))
            with self.assertRaises(AssertionError):
                P.prepare(self.paths, output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
