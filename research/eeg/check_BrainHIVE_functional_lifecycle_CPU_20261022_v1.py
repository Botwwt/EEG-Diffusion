"""CPU contract/integration checks for the NEW lifecycle helpers, never GPU proof.

Uses existing real four-arm recipe schema and tiny temporary metadata. No source
mutation, SSH, dataset access, actual profile, model import, training or images.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import brainhive_functional_training_contract_v2 as C
import prepare_BrainHIVE_functional_lifecycle_CPU_20261022_v1 as P
import queue_BrainHIVE_functional_lifecycle_20261022_v1 as Q


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.original = {a: P.read(P.OLD / (a + '.json')) for a in C.ARMS}
        self.sources = {name: C.sha(P.ROOT / name) for name in P.SOURCES}
        self.recipes = {a: P.new_recipe(self.original[a], a, self.sources, 'a' * 64) for a in C.ARMS}

    def reject(self, function, *arguments):
        with self.assertRaises((AssertionError, KeyError, FileNotFoundError)):
            function(*arguments)

    def test_complete_all_four_schema_and_original_not_mutated(self):
        for arm in C.ARMS:
            original = P.read(P.OLD / (arm + '.json'))
            self.assertEqual(original, self.original[arm])
            value = self.recipes[arm]
            self.assertEqual(C.validate_recipe(value, check_files=False), C.core_sha(value))
            self.assertEqual(value['training_seed'], 1729)
            self.assertEqual(value['expected_updates'], 24807)
            self.assertEqual(value['total_training_query_exposures'], 396900)
            self.assertEqual(value['generator'], dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1))
            self.assertEqual(value['input_sha256'][P.GALLERY.as_posix()], 'a' * 64)
            self.assertTrue(set(P.SOURCES) <= set(value['source_sha256']))
            self.assertNotIn('admission_receipt', value)
            for field in P.GENERATION_KEYS:
                self.assertIn(field, value)

    def test_old_profile_cannot_recertify_full_recipe(self):
        for arm in C.ARMS:
            self.assertNotEqual(C.core_sha(self.original[arm]), C.core_sha(self.recipes[arm]))
            before = C.core_sha(self.recipes[arm])
            value = copy.deepcopy(self.recipes[arm])
            value.update(admission_receipt='/absolute/new_receipt.json', admission_sha256='f' * 64)
            self.assertEqual(C.core_sha(value), before)
            value['generation_job_seconds'] += 1
            self.assertNotEqual(C.core_sha(value), before)

    def test_actual_frozen_generation_contract_accepts_complete_recipe(self):
        import generate_BrainHIVE_functional_alignment_V_v2 as generator
        for recipe in self.recipes.values():
            self.assertEqual(generator.generation_contract(recipe, check_files=False), C.core_sha(recipe))
            incomplete = copy.deepcopy(recipe); incomplete.pop('generation_job_names')
            self.reject(lambda: generator.generation_contract(incomplete, check_files=False))
            unpinned = copy.deepcopy(recipe); unpinned['input_sha256'].pop(P.GALLERY.as_posix())
            self.reject(lambda: generator.generation_contract(unpinned, check_files=False))

    def test_foreign_arm_recipe_is_rejected(self):
        self.reject(P.new_recipe, self.original['P'], 'PN', self.sources, 'a' * 64)

    def test_job_names_unique_and_exact_resume_is_only_second_slot(self):
        names = []
        for arm, recipe in self.recipes.items():
            for stage in ('profile', 'train', 'generate', 'features'):
                name, seconds, entry, args = Q.command(recipe, stage)
                names.append(name)
                self.assertGreater(seconds, 90)
                self.assertIn(entry, P.SOURCES)
                self.assertNotIn('--resume', args)
            name, seconds, entry, args = Q.command(recipe, 'train', index=1)
            names.append(name)
            self.assertEqual(args[-1], '--resume')
            name, seconds, entry, args = Q.command(recipe, 'generate', index=1)
            names.append(name)
            self.assertNotIn('--resume', args)  # Existing generator verifies exact manifest prefix itself.
            self.reject(Q.command, recipe, 'arbitrary-command')
            self.reject(lambda: Q.command(recipe, 'train', index=2))
        self.assertEqual(len(names), len(set(names)))

    def test_formal_recipe_mutation_seed_and_budget_rejected(self):
        for key, value in [('training_seed', 42), ('expected_updates', 2), ('unique_OOF_training_queries', 1323), ('gate_policy', 'risk')]:
            changed = copy.deepcopy(self.recipes['PN']); changed[key] = value
            self.reject(lambda: C.validate_recipe(changed, check_files=False))

    def test_group_identity_detects_cross_arm_host_input_mutation(self):
        entries = [dict(arm=a, candidate=self.recipes[a]['candidate'], run_name=C.run_name(a),
            recipe_core_sha256=C.core_sha(self.recipes[a]), training_root=self.recipes[a]['output_directory'],
            metadata_root=str(C.WORK / 'images' / C.run_name(a)),
            inference_manifest=str(C.WORK / 'images' / C.run_name(a) / 'inference_manifest.jsonl'),
            feature_root=str(C.WORK / 'evaluation' / C.run_name(a) / 'features'),
            image_segments=[dict(start=0, stop=16600, root=str(C.DATA / 'test' / a))]) for a in C.ARMS]
        plan = dict(arms=entries)
        self.assertTrue(C.validate_group_identity(plan, self.recipes))
        changed = copy.deepcopy(self.recipes); changed['S']['input_sha256'][P.GALLERY.as_posix()] = 'b' * 64
        self.reject(C.validate_group_identity, plan, changed)
        entries[-1]['image_segments'][0]['root'] = entries[0]['image_segments'][0]['root']
        self.reject(C.validate_group_identity, plan, self.recipes)

    def test_complete_metadata_identity_no_scalar_false_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'completion.json'
            P.write(path, dict(status=Q.GENERATE_DONE, arm='PN', recipe_core_sha256='a' * 64, queries=16600))
            self.assertEqual(Q.exact_completion(path, Q.GENERATE_DONE, arm='PN', core='a' * 64, count=16600)['sha256'], C.sha(path))
            self.reject(lambda: Q.exact_completion(path, Q.GENERATE_DONE, arm='S', count=16600))
            self.reject(lambda: Q.exact_completion(path, Q.GENERATE_DONE, arm='PN', core='b' * 64))
            self.reject(lambda: Q.exact_completion(path, Q.GENERATE_DONE, arm='PN', count=1660))
            self.reject(Q.exact_completion, path, 'COMPLETED')

    def test_training_resume_requires_all_concrete_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = dict(self.recipes['PN'], output_directory=directory)
            base = Path(directory)
            self.assertFalse(Q.training_can_resume(recipe, 1))
            (base / 'resume.pt').write_bytes(b'fixture-only-never-model')
            P.write(base / 'progress.json', dict(status='FUNCTIONAL_V2_FINITE_RESOURCE_STOP_RESUMABLE'))
            self.assertTrue(Q.training_can_resume(recipe, 1))
            self.assertFalse(Q.training_can_resume(recipe, 0))
            (base / 'progress.json').write_text(json.dumps(dict(status='NUMERICAL_FAILURE')))
            self.assertFalse(Q.training_can_resume(recipe, 1))

    def test_generation_resume_requires_resource_status_and_existing_prefix(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Q, 'WORK', Path(directory)):
            recipe = self.recipes['PN']
            folder = Q.WORK / 'images' / C.run_name('PN'); folder.mkdir(parents=True)
            P.write(folder / 'progress.json', dict(status='FUNCTIONAL_V2_GENERATION_FINITE_STOP_EXACT_PREFIX_PRESERVED'))
            self.assertFalse(Q.generation_can_resume(recipe, 1))
            (folder / 'inference_manifest.jsonl').write_bytes(b'fixture-only')
            self.assertTrue(Q.generation_can_resume(recipe, 1))
            self.assertFalse(Q.generation_can_resume(recipe, 0))

    def test_bound_receipt_refuses_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'; P.write(path, dict(status='TINY_FIXTURE_ONLY'))
            bound = P.ref(path)
            self.assertEqual(P.load_ref(bound)['status'], 'TINY_FIXTURE_ONLY')
            path.write_bytes(b'{"status":"tampered"}\n')
            self.reject(P.load_ref, bound)

    def test_draft_unknown_evaluation_is_refused_before_asset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'readiness.json'
            P.write(path, dict(status='UNKNOWN', metric_selection_or_tuning=False))
            self.reject(P.evaluation_ready, path, {})

    def test_readiness_positive_join_keeps_readiness_reference_after_nonempty_input_loop(self):
        import run_BrainHIVE_functional_evaluation_CPU_v1 as evaluation
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            gallery = base / 'gallery.jsonl'
            gallery.write_text(''.join(json.dumps(dict(stimulus_id=str(i))) + '\n' for i in range(1660)))
            asset = base / 'SYNTHETIC_PUBLIC_ASSET.txt'; asset.write_bytes(b'fixture bytes, not actual model weights')
            input_file = base / 'SYNTHETIC_LAST_INPUT.txt'; input_file.write_bytes(b'last input is deliberately different from readiness')
            probe_source = P.ROOT / 'probe_BrainHIVE_evaluation_environment_GPU_20261022_v2.py'
            job = base / 'job.json'
            P.write(job, dict(status='COMPLETED', exit_code=0, pid=999999999, command=[str(probe_source)]))
            stages = ['DINO', 'EffNet', 'CapSim', 'ObjectF1']
            probe = base / 'SYNTHETIC_PROBE.json'
            P.write(probe, dict(status='FROZEN_FOUR_METRIC_MODELS_RUNTIME_PROBE_COMPLETE', actual_GPU=True,
                stages=stages, metric_selection_or_tuning=False, source_sha256=C.sha(probe_source),
                wrapper_receipt=str(job), pid=999999999, trained_parameters=0, new_generated_images=0,
                paired_metrics_computed=0, score_tables_read=0, feature_tensors_saved=0, optimizer_created=False,
                training_seed=1729, input_sha256={str(input_file): C.sha(input_file)},
                stage_records=[dict(stage=s, original_route_completed=True, optimizer_created=False,
                    parameter_gradients_absent=True, frozen_parameters=1) for s in stages]))
            readiness = base / 'SYNTHETIC_READINESS.json'
            source_pins = {name: C.sha(P.ROOT / name) for name in P.EVALUATION_SOURCES}
            P.write(readiness, dict(status='FROZEN_EVALUATION_ASSETS_READY', metric_selection_or_tuning=False,
                source_sha256=source_pins, assets=[P.ref(asset)], actual_GPU_probe=P.ref(probe)))
            plan = dict(evaluation=dict(lifecycle_source_sha256=source_pins))
            calls = []
            def gallery_check(directory, manifest, keys, real, pins, repo, deadline):
                self.assertEqual(len(keys), 1660)
                self.assertTrue(real)
                self.assertEqual(pins[str(gallery)], C.sha(gallery))
                calls.append(keys)
                return []
            with patch.object(P, 'GALLERY', gallery), patch.object(C, 'process_live', lambda pid: False), patch.object(evaluation, 'feature_inputs', gallery_check):
                result = P.evaluation_ready(readiness, plan)
            self.assertEqual(result['readiness'], P.ref(readiness))
            self.assertNotEqual(result['readiness'], P.ref(input_file))
            self.assertEqual(len(calls), 1)

    def test_postfit_builder_exercises_actual_frozen_shape_and_current_final(self):
        import generate_BrainHIVE_functional_postfit_T32_v1 as T
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            base = Path(directory); repo = base / 'repo'; work = base / 'work'; config = repo / 'configs'; evidence = repo / 'audit'
            config.mkdir(parents=True); evidence.mkdir(); work.mkdir()
            for name in T.SOURCES:
                (repo / name).write_bytes((P.ROOT / name).read_bytes())
            oracle = base / 'SYNTHETIC_ORACLE.json'; P.write(oracle, [])
            doc = P.read(P.ROOT / 'configs/BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json')
            doc['source_selection']['oracle_inference_inputs_path'] = str(oracle)
            doc['source_selection']['oracle_inference_inputs_sha256'] = C.sha(oracle)
            role_files = {}
            for role in ('oof0', 'oof1'):
                role_files[role] = {}
                for name in ('condition_cache', 'parent_checkpoint', 'parent_registration'):
                    path = base / (role + '_' + name + '.SYNTHETIC_BYTES')
                    path.write_bytes(('CPU fixture only: ' + role + name).encode())
                    role_files[role][name] = P.ref(path)
            for row in doc['rows']:
                for name, bound in role_files[row['source_role']].items():
                    row[name + '_path'], row[name + '_sha256'] = bound['path'], bound['sha256']
            query = config / 'BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json'; P.write(query, doc)
            recipes = copy.deepcopy(self.recipes)
            calibration = dict(kind='Dfit_only_global_strength_calibration', strengths=[0., .25, .5, .75, 1.],
                Dfit_token_errors=[3., 2., 1., .5, .6], selected_strength=.75, selected_without_V_or_test_images=True)
            for arm, recipe in recipes.items():
                folder = work / 'components' / arm; folder.mkdir(parents=True)
                recipe['output_directory'] = folder.as_posix()
                recipe['input_sha256'] = {}
                P.write(config / (arm + '.json'), recipe)
                if arm != 'PN':
                    continue  # Future arms deliberately have no final checkpoint.
                checkpoint = folder / 'component_final.pt'; checkpoint.write_bytes(b'SYNTHETIC_FINAL_NOT_A_TENSOR')
                provenance = dict(recipe_core_sha256=C.core_sha(recipe), arm=arm,
                    source_sha256=recipe['source_sha256'], input_contract_sha256=recipe['input_contract_sha256'],
                    risk_head_or_gate=False, all_frozen_host_weights=True)
                P.write(folder / 'calibration.json', calibration); P.write(folder / 'input_provenance.json', provenance)
                P.write(folder / 'completion.json', dict(status=Q.TRAIN_DONE, arm=arm, candidate=recipe['candidate'],
                    updates=24807, seen_queries=396900, passes=3, subjects=10, parameters=525568,
                    explicit_UNet_forwards_per_update=2, recipe_core_sha256=C.core_sha(recipe),
                    checkpoint_path=checkpoint.as_posix(), checkpoint_sha256=C.sha(checkpoint),
                    calibration_sha256=C.sha(folder / 'calibration.json'), input_provenance_sha256=C.sha(folder / 'input_provenance.json')))
            before = {a: C.sha(config / (a + '.json')) for a in C.ARMS}
            seen = []
            def actual_postfit_metadata_check(plan):
                T.shape(plan)
                admitted = P.load_ref(plan['postfit_admission'])
                # Native production is Linux. Windows turns a valid Posix
                # checkpoint into backslashes when frozen current_final calls
                # str(Path(...)); expose that local fixture alias only here.
                native_plan = copy.deepcopy(plan)
                if os.name != 'posix':
                    native_plan['input_sha256'].update({str(Path(k)): v for k, v in plan['input_sha256'].items()})
                T.current_final(native_plan, recipes['PN'], admitted)
                self.assertEqual({a: C.sha(config / (a + '.json')) for a in C.ARMS}, before)
                self.assertEqual(len(plan['formal_recipes']), 4)
                self.assertEqual(plan['queries'], 32)
                seen.append(plan)
                return {}
            for target, name, value in ((P, 'ROOT', repo), (P, 'WORK', work), (P, 'CONFIG', config), (P, 'BASE', evidence),
                    (T, 'REPO', repo), (T, 'BASE', work / 'audit/functional_postfit_T32_v1'), (T, 'QUERY_SHA', C.sha(query)),
                    (T, 'terminal_check', lambda: {}), (C, 'verify_admission', lambda recipe, purpose: {}),
                    (T, 'prepare', actual_postfit_metadata_check), (P, 'assert_host', lambda: None)):
                stack.enter_context(patch.object(target, name, value))
            result = P.postfit('PN')
            self.assertEqual(result['status'], 'CURRENT_REAL_FINAL_POSTFIT_WRAPPER_VERIFIED_NO_IMAGES')
            self.assertEqual(len(seen), 1)
            self.assertFalse((work / 'components' / 'O' / 'component_final.pt').exists())

    def test_sealed_feature_training_and_metadata_overages_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = dict(arm='PN', training_root=str(root / 'training'), feature_root=str(root / 'features'),
                metadata_root=str(root / 'metadata'), training_budget_bytes=10, feature_budget_bytes=10, metadata_budget_bytes=10)
            plan = dict(arms=[entry])
            Q.check_output_budgets(plan)
            for key in ('training_root', 'feature_root', 'metadata_root'):
                path = Path(entry[key]); path.mkdir(); (path / 'fixture.bin').write_bytes(b'12345678901')
                self.reject(Q.check_output_budgets, plan)
                (path / 'fixture.bin').write_bytes(b'1234567890')
                Q.check_output_budgets(plan)

    def test_wrapper_handle_is_waited_even_when_registered_child_has_already_exited(self):
        class Handle:
            pid = 123456789
            waits = 0
            done = False
            def wait(self, timeout):
                self.waits += 1; self.done = True
                return 0
            def poll(self):
                return 0 if self.done else None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'job.json'
            P.write(path, dict(wrapper_pid=123456789, pid=11111111))
            handle = Handle()
            with patch.object(C, 'process_live', lambda pid: False):
                self.assertTrue(Q.stop_owned_process(handle, True, path))
            self.assertEqual(handle.waits, 1)
            missing = Handle()
            self.assertTrue(Q.stop_owned_process(missing, True, Path(directory) / 'missing.json'))
            self.assertEqual(missing.waits, 1)

    def test_observation_timeouts_do_not_become_terminal_wrapper_claims(self):
        class StillRunning:
            pid = 123456789
            def wait(self, timeout):
                raise Q.subprocess.TimeoutExpired('CPU synthetic handle', timeout)
            def poll(self):
                return None
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(Q.stop_owned_process(StillRunning(), True, Path(directory) / 'not_yet_published.json'))

    def test_cpu_helpers_import_no_torch_and_keep_stage_sources_bound(self):
        import sys
        self.assertNotIn('torch', sys.modules)
        self.assertEqual(self.sources['train_BrainHIVE_functional_alignment_v2.py'],
            '9a8d2672d31a5a54e9fdc7c714228a575cf20789e6a8fb13c5f1f582d8323b6f')
        self.assertEqual(self.sources['generate_BrainHIVE_functional_alignment_V_v2.py'],
            '03d89fda08aff0f8ff1821ab8c5a90a96122eca528544c4d6ae36971ec52bceb')


if __name__ == '__main__':
    unittest.main(verbosity=2)
