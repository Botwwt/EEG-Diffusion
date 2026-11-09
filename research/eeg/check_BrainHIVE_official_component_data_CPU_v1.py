"""Actual165400 METADATA and private tiny CPU tensor assembly, no real fit/gate."""
import argparse
import ast
import copy
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import brainhive_official_component_data_v1 as D

ROOT = Path(__file__).resolve().parent
PLAN_DIRECTORY = ROOT / 'audit/official_fullfit_metadata_plan_20261022_v1'


def fixture():
    """Two subjects, two disjoint held folds, six images and four raw coordinates."""
    import torch
    subjects = ['sub-01', 'sub-02']; held = [[4, 0, 2], [3, 5, 1]]
    jobs = []; sources = []; mapping = [None] * 12
    for subject_index, subject in enumerate(subjects):
        for fold in range(2):
            index = len(jobs); role = 'official_oof' + str(fold)
            job = dict(task_id=subject + '/' + role, subject=subject, role=role,
                expected_export_rows=held[fold], registration_core_sha256=str(index + 1) * 64,
                registration=dict(EEG_input=dict(sha256='e' * 64)))
            fit = dict(model_sha256=str(index + 2) * 64, registration_sha256=str(index + 3) * 64,
                producer_source_sha256='a' * 64)
            weight = torch.arange(1, 5, dtype=torch.float32) + index
            bias = torch.arange(4, dtype=torch.float32) / 10 + index
            state = {D.E.AFFINE_KEYS[0]: weight, D.E.AFFINE_KEYS[1]: bias}
            condition = torch.stack([torch.arange(4, dtype=torch.float32) + subject_index * 100 + row * 10 for row in held[fold]])
            tensors = dict(condition=condition, array_rows=torch.tensor(held[fold], dtype=torch.long), affine_weight=weight, affine_bias=bias)
            cache = D.X.make_cache(tensors, job, fit, 'b' * 64)
            jobs.append(job); sources.append(dict(job=job, cache=cache, model=state, fit=fit, exporter_sha256='b' * 64))
            for local, row in enumerate(held[fold]): mapping[subject_index * 6 + row] = (index, local)
    layout = dict(jobs=jobs, subjects=subjects, source_by_query=mapping,
        task_index={job['task_id']: index for index, job in enumerate(jobs)})
    target = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    return layout, sources, target, dict(images=6, width=4, subjects=2)


class EarlyGateChecks(unittest.TestCase):
    def test_actual_latent_consumer_context_keys_match_gate_return_without_token(self):
        # Inspect the actual callee return and actual consumer accesses. No
        # successful preflight/context is fabricated to test the pending gate.
        module = ast.parse(Path(D.__file__).read_text(encoding='utf-8'))
        preflight = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'preflight')
        returned = next(node for node in ast.walk(preflight) if isinstance(node, ast.Return))
        keys = {item.arg for item in returned.value.keywords}
        consumer_path = ROOT / 'build_BrainHIVE_official_train_latents_v1.py'
        consumer = ast.parse(consumer_path.read_text(encoding='utf-8'))
        accessed = {node.slice.value for node in ast.walk(consumer) if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == 'context' and isinstance(node.slice, ast.Constant)}
        self.assertTrue(accessed <= keys, 'Latent consumer missing context keys: ' + str(accessed - keys))
        self.assertIn('runtime', keys); self.assertIn('stage_spec', keys)

    def test_missing_review_before_plan_torch_payload_output_in_fresh_interpreter(self):
        with tempfile.TemporaryDirectory(prefix='HIVE_official_component_negative_') as tmp:
            missing = Path(tmp) / 'NO_SEALED_REVIEW.json'; plan = Path(tmp) / 'NO_PLAN_READ'
            code = r'''
import importlib.abc, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='torch' or name.startswith('torch.') or name=='brainhive_native_models_v1':
   raise AssertionError('FORBIDDEN_MODEL_IMPORT')
sys.meta_path.insert(0,Block())
import brainhive_official_component_data_v1 as D
def no(*a,**k): raise AssertionError('FORBIDDEN_PLAN_PAYLOAD_OUTPUT')
D.E.read_plan_bundle=no; pathlib.Path.mkdir=no
for purpose in ('component_inputs','latents'):
 try: D.preflight(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]),purpose=purpose,entry_name='NOT_RUN.py')
 except ValueError as error: assert 'review does not exist' in str(error)
 else: raise AssertionError('MISSING_REVIEW_ACCEPTED')
try: D.load_training({'_stage_token':True})
except ValueError as error: assert 'actual successful' in str(error)
else: raise AssertionError('JSON_TOKEN_ACCEPTED')
assert 'torch' not in sys.modules and 'brainhive_native_models_v1' not in sys.modules
print('EARLY_REFUSAL_PASS')
'''
            result = subprocess.run([sys.executable, '-B', '-c', code, str(plan), str(missing)], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('EARLY_REFUSAL_PASS', result.stdout); self.assertFalse(plan.exists())

    def test_handset_review_flag_cannot_grant_stage_or_latent_permission(self):
        with tempfile.TemporaryDirectory(prefix='HIVE_official_handset_negative_') as tmp:
            review = Path(tmp) / 'NOT_A_REVIEW.json'
            review.write_text(json.dumps(dict(schema='BrainHIVE_official_fullfit_metadata_plan_v1', status='passed', fitting_admitted=True)))
            with patch.object(D.E, 'read_plan_bundle', side_effect=AssertionError('PLAN_MUST_NOT_BE_READ')):
                for purpose in ('component_inputs', 'latents'):
                    with self.assertRaisesRegex(ValueError, 'planning flags are not admission'):
                        D.preflight(Path(tmp) / 'NO_PLAN', review, purpose=purpose, entry_name='NOT_RUN.py')

    def test_file_identity_sha_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory(prefix='HIVE_official_SHA_') as tmp:
            path = Path(tmp) / 'ONLY_TINY_BYTES.dat'; path.write_bytes(b'PRIVATE_CPU_SYNTHETIC_ONLY')
            reference = D.file_ref(path)
            self.assertEqual(D.bound_file(reference), path)
            path.write_bytes(b'CORRUPT_TINY_FIXTURE')
            with self.assertRaisesRegex(ValueError, 'payload/source SHA differs'): D.bound_file(reference)

    def test_closed_export_job_replaces_mutated_running_reference_without_adopting_old_sha(self):
        with tempfile.TemporaryDirectory(prefix='HIVE_closed_wrapper_CPU_only_') as tmp:
            directory = Path(tmp); review_path = directory / 'TINY_NO_ADMISSION_REVIEW.json'
            task_id = 'sub-01/official_oof0'
            task = dict(plan_directory='/PRIVATE_CPU_ONLY_PLAN', train_job_names=['TINY_FIT'], export_job_name='TINY_EXPORT')
            review_path.write_text(json.dumps(dict(runtime=dict(python_executable=sys.executable), tasks={task_id: task},
                resource_authorization=dict(sha256='a' * 64))))
            review = D.G.ref(review_path); path = directory / 'TINY_EXPORT_job.json'
            command = [sys.executable, str(ROOT / Path(D.X.__file__).name), '--plan-directory', task['plan_directory'],
                '--task-id', task_id, '--execution-review', str(review_path)]
            record = dict(status='RUNNING', command=command, pid=99999991, wrapper_pid=99999992,
                charged_seconds=1.5, exit_code=None, authorization_sha256='a' * 64)
            path.write_text(json.dumps(record)); historical_running_ref = D.G.ref(path)
            record.update(status='COMPLETED', exit_code=0, charged_seconds=2.)
            path.write_text(json.dumps(record)); closed_ref = D.G.ref(path)
            self.assertNotEqual(historical_running_ref['sha256'], closed_ref['sha256'])
            self.assertEqual(historical_running_ref['path'], closed_ref['path'])
            actual = D._closed_wrapper(closed_ref, entry_name=Path(D.X.__file__).name, task_id=task_id, execution_review=review)
            self.assertEqual(actual, record)
            with self.assertRaisesRegex(ValueError, 'Missing/changed/symlink'): D.G.bound(historical_running_ref)
            wrong = dict(record, command=command + ['--unexpected'])
            path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'independent successful'):
                D._closed_wrapper(D.G.ref(path), entry_name=Path(D.X.__file__).name, task_id=task_id, execution_review=review)


class ActualMetadataChecks(unittest.TestCase):
    def test_all_real_twenty_tasks_165400_queries_without_payloads(self):
        with patch.object(D, 'fit_export_chain', side_effect=AssertionError('NO_ACTUAL_PAYLOAD_CHAIN')):
            bundle = D.E.read_plan_bundle(PLAN_DIRECTORY)
            value = D.metadata_layout(bundle)
        self.assertEqual((value['queries'], value['images'], value['concepts']), (165400, 16540, 1654))
        self.assertEqual(len(value['jobs']), 20)
        self.assertEqual(len(value['source_by_query']), 165400)
        for subject_index in range(10):
            actual = value['source_by_query'][subject_index * 16540:(subject_index + 1) * 16540]
            counts = {}
            for task_index, cache_row in actual: counts.setdefault(task_index, set()).add(cache_row)
            self.assertEqual(sorted(map(len, counts.values())), [8270, 8270])
            self.assertTrue(all(value['jobs'][index]['subject'] == value['subjects'][subject_index] for index in counts))
        self.assertEqual({r['array_row'] for r in value['train_identity']}, set(range(16540)))
        self.assertEqual(value['train_identity_sha256'], D.E.core_sha(value['train_identity']))


class TinyActualTensorChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2); cls.torch = torch

    def test_actual_raw_values_global_query_order_distinct_own_affine_and_teacher(self):
        torch = self.torch; layout, sources, target, dimensions = fixture()
        result = D._assemble(layout, iter(reversed(sources)), target, _fixture=dimensions)
        expected = torch.stack([torch.arange(4, dtype=torch.float32) + subject * 100 + row * 10 for subject in range(2) for row in range(6)])
        self.assertTrue(torch.equal(result['condition'], expected))
        self.assertTrue(torch.equal(result['target'], target)); self.assertNotEqual(result['target'].data_ptr(), target.data_ptr())
        self.assertTrue(torch.equal(result['weight'], torch.stack([s['cache']['affine_weight'] for s in sources])))
        self.assertTrue(torch.equal(result['bias'], torch.stack([s['cache']['affine_bias'] for s in sources])))
        for query, (source, cache_row) in enumerate(layout['source_by_query']):
            self.assertEqual(result['affine_ids'][query], source)
            self.assertEqual(result['query_cache_rows'][query], cache_row)
            self.assertTrue(torch.equal(result['condition'][query], sources[source]['cache']['condition'][cache_row]))
        self.assertFalse(result['actual_full165400_verified']); self.assertTrue(result['fixture_only'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_local_cache_global_row_and_cross_fold_join_rejected(self):
        layout, sources, target, dimensions = fixture()
        wrong = copy.deepcopy(layout); wrong['source_by_query'][0] = (0, 0)
        with self.assertRaisesRegex(ValueError, 'planned query/cache-row identity'): D._assemble(wrong, sources, target, _fixture=dimensions)
        wrong_sources = copy.deepcopy(sources); wrong_sources[0]['cache']['array_rows'] = wrong_sources[1]['cache']['array_rows'].clone()
        with self.assertRaisesRegex(ValueError, 'global array_rows/order'): D._assemble(layout, wrong_sources, target, _fixture=dimensions)
        wrong_sources = copy.deepcopy(sources); wrong_sources[0]['job']['role'] = 'oof0'
        with self.assertRaisesRegex(ValueError, 'OldT/full/published'): D._assemble(layout, wrong_sources, target, _fixture=dimensions)

    def test_missing_duplicate_and_cross_subject_source_rejected(self):
        layout, sources, target, dimensions = fixture()
        with self.assertRaisesRegex(ValueError, 'Missing/repeated'): D._assemble(layout, sources[:-1], target, _fixture=dimensions)
        with self.assertRaisesRegex(ValueError, 'duplicate task'): D._assemble(layout, sources + sources[:1], target, _fixture=dimensions)
        wrong = copy.deepcopy(sources); wrong[0]['cache']['subject'] = 'sub-02'
        with self.assertRaisesRegex(ValueError, 'subject/fold/EEG'): D._assemble(layout, wrong, target, _fixture=dimensions)

    def test_all_value_finite_dtype_source_and_own_affine_negatives(self):
        torch = self.torch; layout, sources, target, dimensions = fixture()
        for change, expected in (('NaN', 'Nonfinite cached'), ('dtype', 'shape/dtype'), ('affine', 'own final-LN'),
                ('checkpoint', 'model/source chain'), ('norm', 'model/source chain'), ('near_zero_affine', 'near-zero scale')):
            wrong = copy.deepcopy(sources)
            if change == 'NaN': wrong[2]['cache']['condition'][-1, -1] = float('nan')
            elif change == 'dtype': wrong[2]['cache']['condition'] = wrong[2]['cache']['condition'].double()
            elif change == 'affine':
                wrong[2]['cache']['affine_weight'] = wrong[2]['cache']['affine_weight'].clone()
                wrong[2]['cache']['affine_weight'][0] += 100
            elif change == 'checkpoint': wrong[2]['cache']['model_sha256'] = '0' * 64
            elif change == 'norm': wrong[2]['cache']['additional_L2_normalization'] = True
            else:
                wrong[2]['cache']['affine_weight'][0] = 0
                wrong[2]['model'][D.E.AFFINE_KEYS[0]][0] = 0
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, expected): D._assemble(layout, wrong, target, _fixture=dimensions)
        for value in (target[:-1], target.double(), target.clone()):
            if value.shape == target.shape and value.dtype == torch.float32: value[-1, -1] = float('nan')
            with self.assertRaisesRegex(ValueError, 'teacher bank'): D._assemble(layout, sources, value, _fixture=dimensions)


def main(argv=None):
    global PLAN_DIRECTORY
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument('--plan-directory', type=Path, default=PLAN_DIRECTORY)
    args, remaining = parser.parse_known_args(argv); PLAN_DIRECTORY = args.plan_directory
    unittest.main(argv=[sys.argv[0], *remaining])


if __name__ == '__main__': main()
