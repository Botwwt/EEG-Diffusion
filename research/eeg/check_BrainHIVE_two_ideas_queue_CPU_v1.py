"""Isolated accounting/control-flow regressions, no real queue, EEG, or GPU.

Temporary files are tiny. POSIX allocated-block metadata and disk free space are
simulated explicitly; this cannot serve as an actual storage admission receipt.
"""
import argparse
import ast
import copy
from contextlib import ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


storage = load_module('brainhive_two_ideas_storage_v1', 'brainhive_two_ideas_storage_v1.py')
try:
    import fcntl
except ImportError:
    # Windows CPU validation emulates only the queue lock's interface.
    fcntl = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *_: None)
    sys.modules['fcntl'] = fcntl
queue = load_module('_two_ideas_queue_under_test', 'queue_BrainHIVE_two_ideas_v1.py')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')


class AccountingAndQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='brainhive_queue_CPU_')
        self.base = Path(self.temp.name)
        self.root, self.work, self.data = [self.base / n for n in ['code', 'work', 'data']]
        for p in [self.root, self.work / 'audit', self.data]:
            p.mkdir(parents=True)
        self.stack = ExitStack()
        for module in [storage, queue]:
            for key, value in [('ROOT', self.root), ('WORK', self.work), ('DATA', self.data)]:
                self.stack.enter_context(mock.patch.object(module, key, value))
        self.stack.enter_context(mock.patch.object(queue, 'STATE', self.work / 'audit/two_ideas_queue_v1.json'))
        queue.OWNS_STATE = False
        queue.QUEUE_LOCK = None
        self.free = 100 << 30
        self.main_need = 512 << 20
        self.dev = self.work.stat().st_dev
        self.stack.enter_context(mock.patch.object(storage, 'protected_main', side_effect=self.protected_main))
        self.stack.enter_context(mock.patch.object(storage.shutil, 'disk_usage',
            side_effect=lambda _: types.SimpleNamespace(free=self.free)))
        self.reserve_paths = set()
        real_stat = Path.stat

        def block_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if path in self.reserve_paths:
                return types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600,
                    st_blocks=(16 << 20) // 512, st_dev=result.st_dev,
                    st_size=result.st_size)
            return result

        self.stack.enter_context(mock.patch.object(Path, 'stat', block_stat))

    def tearDown(self):
        if queue.QUEUE_LOCK is not None:
            queue.QUEUE_LOCK.close()
        self.stack.close()
        self.temp.cleanup()

    def protected_main(self):
        return {self.dev: self.main_need}, dict(
            free=[self.free, self.free], remaining_required=[self.main_need, 0])

    def recipes(self):
        roots = [self.work / 'allocated_images', self.data / 'allocated_images']
        for root in roots:
            root.mkdir(exist_ok=True)
        return [dict(candidate=candidate, run_name='test_' + candidate,
            formal_V_queries_per_arm=16600, reserved_bytes_per_image=330000,
            allowed_private_roots=[str(p) for p in roots],
            image_segments=[dict(start=0, stop=16600, root=str(roots[1]))],
            output_directory=str(self.work / 'runs' / candidate),
            training_artifact_budget_bytes=1 << 30, feature_budget_bytes=2 << 20,
            metadata_budget_bytes=4 << 20, source_sha256={})
            for candidate in ['brainkan_full_v1', 'response_rank_v1']]

    def admitted_fixture(self):
        recipes = self.recipes()
        folder = self.work / 'two_ideas_v1/recipes'
        paths = [folder / (r['candidate'] + '.json') for r in recipes]
        plan_path = folder / 'group_plan.json'
        plan = dict(recipes=[dict(path=str(p), core_sha256=storage.core_sha(r))
                             for p, r in zip(paths, recipes)])
        put(plan_path, plan)
        for path, recipe in zip(paths, recipes):
            reserve = folder / (recipe['candidate'] + '.metadata_reserve')
            reserve.write_bytes(b'MOCK_ALLOCATION_ONLY')
            self.reserve_paths.add(reserve)
            receipt_path = path.with_suffix('.admission.json')
            receipt = dict(status='FULL_TRAIN_GENERATE_SCORE_ADMITTED',
                candidate=recipe['candidate'], recipe_core_sha256=storage.core_sha(recipe),
                output_directory=recipe['output_directory'],
                scope=['training', 'actual_images', 'per_query_evaluation'],
                max_training_artifact_bytes=recipe['training_artifact_budget_bytes'],
                reservation_paths=[dict(path=str(reserve), minimum_allocated_bytes=16 << 20)],
                group_plan=str(plan_path), group_plan_sha256=storage.sha(plan_path))
            put(receipt_path, receipt)
            recipe['storage_admission_receipt'] = str(receipt_path)
            recipe['storage_admission_sha256'] = storage.sha(receipt_path)
            put(path, recipe)
        return recipes, paths, plan_path

    def test_receipt_cycle_excludes_only_two_fields(self):
        base = self.recipes()[0]
        digest = storage.core_sha(base)
        admitted = dict(base, storage_admission_receipt='/some/receipt', storage_admission_sha256='a' * 64)
        self.assertEqual(digest, storage.core_sha(admitted))
        for key, value in [('training_artifact_budget_bytes', 1), ('image_segments', []),
                           ('source_sha256', {'owned.py': 'b' * 64}),
                           ('storage_admission_other', 'must be bound')]:
            changed = copy.deepcopy(admitted)
            changed[key] = value
            self.assertNotEqual(digest, storage.core_sha(changed), key)

    def test_remaining_forecast_protects_main_and_all_new_images(self):
        recipes = self.recipes()
        one_arm_images = 16600 * 330000
        non_image = sum(r['training_artifact_budget_bytes'] + r['feature_budget_bytes'] +
                        r['metadata_budget_bytes'] for r in recipes)
        # Enough for main, both models/evaluation, and only ONE arm's images.
        self.free = self.main_need + non_image + one_arm_images
        report = storage.forecast(recipes)
        self.assertFalse(report['funded'])
        self.assertEqual(report['total_deficit'], one_arm_images)
        self.assertEqual(sum(v['protected_main_bytes'] for v in report['volumes']), self.main_need)
        self.assertEqual(sum(r['images_remaining'] for r in report['candidates']), 33200)

    def test_valid_admission_is_denied_after_fresh_space_loss(self):
        recipes, _, _ = self.admitted_fixture()
        candidate = recipes[0]['candidate']
        self.assertTrue(storage.verify_candidate_remaining(recipes[0], candidate)['funded'])
        self.free = self.main_need + (1 << 30)
        with self.assertRaisesRegex(AssertionError, 'storage no longer funded'):
            storage.verify_candidate_remaining(recipes[0], candidate)

    def test_changed_pinned_source_blocks_use_of_old_admission(self):
        source = self.root / 'owned_training_entry.py'
        source.write_text('reviewed_source = 1\n', encoding='utf-8')
        recipes, paths, plan_path = self.admitted_fixture()
        caller = recipes[0]
        caller['source_sha256'] = {source.name: storage.sha(source)}
        # Reissue this fixture honestly with the updated core, then mutate the
        # executable only. A storage receipt must not bless changed execution.
        plan = storage.read(plan_path)
        plan['recipes'][0]['core_sha256'] = storage.core_sha(caller)
        put(plan_path, plan)
        receipt_path = Path(caller['storage_admission_receipt'])
        receipt = storage.read(receipt_path)
        receipt['recipe_core_sha256'] = storage.core_sha(caller)
        receipt['group_plan_sha256'] = storage.sha(plan_path)
        put(receipt_path, receipt)
        caller['storage_admission_sha256'] = storage.sha(receipt_path)
        put(paths[0], caller)
        self.assertTrue(storage.verify_candidate_remaining(caller, caller['candidate'])['funded'])
        source.write_text('unreviewed_source = 2\n', encoding='utf-8')
        with self.assertRaisesRegex(AssertionError, source.name):
            storage.verify_candidate_remaining(caller, caller['candidate'])

    def test_group_plan_cannot_omit_own_or_other_candidate(self):
        recipes, _, plan_path = self.admitted_fixture()
        original = storage.read(plan_path)
        for entries in [[], [original['recipes'][1]], [original['recipes'][1]] * 2]:
            put(plan_path, dict(recipes=entries))
            receipt_path = Path(recipes[0]['storage_admission_receipt'])
            receipt = storage.read(receipt_path)
            receipt['group_plan_sha256'] = storage.sha(plan_path)
            put(receipt_path, receipt)
            caller = dict(recipes[0], storage_admission_sha256=storage.sha(receipt_path))
            with self.assertRaises(AssertionError):
                storage.verify_candidate_remaining(caller, caller['candidate'])

    def test_bad_candidate_query_bounds_and_segments_are_rejected(self):
        recipes = self.recipes()
        bad_variants = []
        r = copy.deepcopy(recipes[0]); r['candidate'] = 'third_unregistered_idea'; bad_variants.append(r)
        r = copy.deepcopy(recipes[0]); r['formal_V_queries_per_arm'] = 16601; bad_variants.append(r)
        for parts in [[(0, 8000), (7999, 16600)], [(0, 8000), (8001, 16600)], [(0, 16601)]]:
            r = copy.deepcopy(recipes[0])
            r['image_segments'] = [dict(start=a, stop=b, root=r['allowed_private_roots'][1]) for a, b in parts]
            bad_variants.append(r)
        for recipe in bad_variants:
            with self.assertRaises(AssertionError):
                storage.forecast([recipe, recipes[1]])
        external = self.base / 'public_or_unrelated'
        external.mkdir()
        r = copy.deepcopy(recipes[0]); r['allowed_private_roots'] = [str(external)]
        r['image_segments'][0]['root'] = str(external)
        with self.assertRaises(AssertionError):
            storage.forecast([r, recipes[1]])

    def sequence(self):
        path = self.base / 'sequence.json'
        put(path, dict(fixed_order=['brainkan_full_v1', 'response_rank_v1'],
            source_sha256={}, template_sha256={}, maximum_wait_seconds=60))
        return path

    def call_main(self, sequence):
        with mock.patch.object(sys, 'argv', ['queue.py', '--sequence', str(sequence)]):
            return queue.main()

    def test_prior_queue_survives_relaunch_exception_handler(self):
        prior = b'{"status":"PRIOR_HISTORY_MUST_SURVIVE","pid":12345}\n'
        queue.STATE.write_bytes(prior)
        sequence = self.sequence()
        # Execute the actual CLI exception handler as well as main's guard.
        tree = ast.parse((HERE / 'queue_BrainHIVE_two_ideas_v1.py').read_text())
        cli = tree.body[-1]
        self.assertIsInstance(cli, ast.If)
        namespace = dict(queue.__dict__, __name__='__main__')
        with mock.patch.object(sys, 'argv', ['queue.py', '--sequence', str(sequence)]):
            with self.assertRaisesRegex(AssertionError, 'prior queue'):
                exec(compile(ast.Module(body=[cli], type_ignores=[]), '<actual queue CLI>', 'exec'), namespace)
        self.assertFalse(queue.OWNS_STATE)
        self.assertEqual(queue.STATE.read_bytes(), prior)

    def test_existing_main_lifecycle_is_not_bypassed(self):
        sequence = self.sequence()
        put(self.work / 'audit/bounded_full_lifecycle_queue_v1.json', dict(status='RUNNING_REGISTERED_JOB'))
        with mock.patch.object(queue, 'allocate_recipes') as allocate, \
             mock.patch.object(queue, 'run_candidate') as run, \
             mock.patch.object(queue.time, 'sleep', side_effect=RuntimeError('STOP_TEST_WAIT')):
            with self.assertRaisesRegex(RuntimeError, 'STOP_TEST_WAIT'):
                self.call_main(sequence)
        allocate.assert_not_called(); run.assert_not_called()
        self.assertEqual(storage.read(queue.STATE)['status'],
                         'WAITING_FOR_EXISTING_MAIN_LIFECYCLE_AND_EXCLUSIVE_GPU')

    def test_failed_first_candidate_does_not_skip_independent_second(self):
        sequence = self.sequence()
        put(self.work / 'audit/bounded_full_lifecycle_queue_v1.json',
            dict(status='FINITE_MAIN_COMPARISON_PARTIAL_WITH_EXPLICIT_FAILURES'))
        paths = []
        for candidate in ['brainkan_full_v1', 'response_rank_v1']:
            path = self.base / (candidate + '.json'); put(path, dict(candidate=candidate)); paths.append(path)
        with mock.patch.object(queue, 'gpu_idle', return_value=True), \
             mock.patch.object(queue, 'allocate_recipes', return_value=(paths, {})), \
             mock.patch.object(queue, 'run_candidate', side_effect=[RuntimeError('fixed prerequisite failed'), None]) as run:
            self.call_main(sequence)
        self.assertEqual(run.call_args_list, [mock.call(p) for p in paths])
        final = storage.read(queue.STATE)
        self.assertEqual(final['status'], 'FINITE_PARTIAL_WITH_PRESERVED_FAILURES')
        self.assertEqual(final['completed'], ['response_rank_v1'])
        self.assertIn('fixed prerequisite failed', final['failures']['brainkan_full_v1'])

    def test_failed_engineering_does_not_start_formal_training(self):
        path = self.base / 'candidate.json'
        put(path, dict(candidate='brainkan_full_v1', training_entry='train_BrainHIVE_BrainKAN_v1.py'))
        with mock.patch.object(queue, 'gpu', return_value=1) as gpu, \
             mock.patch.object(queue, 'verify_candidate_remaining') as verify, \
             mock.patch.object(queue, 'cpu') as cpu:
            error = queue.run_candidate(path)
        self.assertIn('no formal training started', error)
        self.assertEqual(gpu.call_count, 1)
        verify.assert_not_called(); cpu.assert_not_called()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AccountingAndQueueTests))
    print(stream.getvalue(), end='')
    report = dict(status='ISOLATED_QUEUE_ACCOUNTING_CPU_CHECKS_PASSED' if result.wasSuccessful() else 'FAILED',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        mocked_free_space_and_POSIX_allocation=True, actual_storage_admission=False,
        actual_queue_or_subprocess_started=False, real_EEG_or_weights_loaded=False, GPU_work=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
