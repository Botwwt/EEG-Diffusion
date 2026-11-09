"""Real CPU bank transactions and pinned full-train metadata; no actual VAE/GPU."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import build_BrainHIVE_official_train_latents_v1 as B


def registration(count=513):
    return dict(shape=[count, 1, 2, 2], array_rows=list(range(count)), dtype='float32',
        batch_size=4, chunk_size=256, actual_official_full_train_scope=False,
        source='CPU deterministic tiny encoder; no VAE/model/EEG/image payload')


def encoder(observed):
    import numpy as np
    def run(ids):
        observed.extend(ids)
        return np.stack([np.full((1, 2, 2), float(i) + .125, dtype=np.float32) for i in ids])
    return run


class Checks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='BrainHIVE_latent_CPU_fixture_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.guard_calls = 0

    def guard(self):
        self.guard_calls += 1

    def build(self, name='bank', count=513):
        observed = []
        result = B._build_bank(self.root / name, registration(count), encoder(observed), self.guard)
        return result, observed

    def test_01_complete_real_bank_rows_last_batch_and_no_exists_skip(self):
        import numpy as np
        result, observed = self.build()
        self.assertEqual(observed, list(range(513)))
        self.assertEqual(result['VAE_encode_calls_this_attempt'], 129)
        self.assertEqual([(r['start'], r['stop']) for r in result['committed_chunks']], [(0, 256), (256, 512), (512, 513)])
        self.assertEqual(result['status'], 'CPU_TINY_LATENT_BANK_FIXTURE_COMPLETE')
        self.assertFalse(result['official_test_image_values_used'])
        root = self.root / 'bank'
        self.assertTrue(os.path.samefile(root / 'native_VAE_train_modes.npy', root / 'native_VAE_train_modes.npy.pending'))
        values = np.load(root / 'native_VAE_train_modes.npy', allow_pickle=False)
        self.assertTrue(np.array_equal(values[:, 0, 0, 0], np.arange(513, dtype=np.float32) + .125))
        self.assertEqual(B.allocated_bytes(root), sum(p.stat().st_blocks * 512 if hasattr(p.stat(), 'st_blocks') else p.stat().st_size
            for p in [root / 'native_VAE_train_modes.npy', root / 'registration.json', root / 'completion.json', *sorted((root / 'chunks').glob('chunk-*.json'))]))
        before = {str(p.relative_to(root)): B.sha(p) for p in root.rglob('*') if p.is_file()}
        def forbidden(_): raise AssertionError('Already complete bank must verify without encoding')
        self.assertEqual(B._build_bank(root, registration(), forbidden, self.guard), result)
        self.assertEqual(before, {str(p.relative_to(root)): B.sha(p) for p in root.rglob('*') if p.is_file()})
        self.assertGreater(self.guard_calls, 130)

    def test_02_post_commit_interrupt_reuses_only_valid_committed_chunks(self):
        import numpy as np
        expected, _ = self.build('continuous')
        root = self.root / 'interrupted'; visited = []
        def fault(stage, number):
            if stage == 'after_chunk_commit' and number == 0: raise RuntimeError('Injected after commit')
        with self.assertRaisesRegex(RuntimeError, 'after commit'):
            B._build_bank(root, registration(), encoder(visited), self.guard, fault=fault)
        self.assertEqual(visited, list(range(256)))
        self.assertEqual(len(list((root / 'chunks').glob('chunk-*.json'))), 1)
        old_commit = (root / 'chunks/chunk-000.json').read_bytes()
        array = np.load(root / 'native_VAE_train_modes.npy.pending', mmap_mode='r+')
        array[256:] = -999.; array.flush(); del array
        rest = []
        resumed = B._build_bank(root, registration(), encoder(rest), self.guard)
        self.assertEqual(rest, list(range(256, 513)))
        self.assertEqual(resumed['committed_chunks_reused'], 1)
        self.assertEqual(resumed['bank']['sha256'], expected['bank']['sha256'])
        self.assertEqual((root / 'chunks/chunk-000.json').read_bytes(), old_commit)

    def test_03_pre_commit_interrupt_recomputes_uncommitted_data(self):
        root = self.root / 'precommit'; visited = []
        def fault(stage, number):
            if stage == 'before_chunk_commit' and number == 0: raise RuntimeError('Injected precommit')
        with self.assertRaisesRegex(RuntimeError, 'precommit'):
            B._build_bank(root, registration(), encoder(visited), self.guard, fault=fault)
        self.assertEqual(visited, list(range(256)))
        self.assertFalse(list((root / 'chunks').glob('chunk-*.json')))
        all_rows = []
        B._build_bank(root, registration(), encoder(all_rows), self.guard)
        self.assertEqual(all_rows, list(range(513)))

    def test_04_corrupt_or_missing_committed_chunk_refuses_before_more_encoding(self):
        import numpy as np
        root = self.root / 'corrupt'
        def fault(stage, number):
            if stage == 'after_chunk_commit' and number == 0: raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):
            B._build_bank(root, registration(), encoder([]), self.guard, fault=fault)
        array = np.load(root / 'native_VAE_train_modes.npy.pending', mmap_mode='r+')
        array[0] = 42.; array.flush(); del array
        calls = []
        with self.assertRaisesRegex(ValueError, 'chunk bytes changed'):
            B._build_bank(root, registration(), encoder(calls), self.guard)
        self.assertFalse(calls)
        # A valid finished bank is also checked in full, not accepted by exists.
        self.build('complete_corrupt', 5)
        completed = self.root / 'complete_corrupt'
        array = np.load(completed / 'native_VAE_train_modes.npy', mmap_mode='r+')
        array[4] = 99.; array.flush(); del array
        with self.assertRaisesRegex(ValueError, 'bank SHA differs'):
            B._build_bank(completed, registration(5), encoder([]), self.guard)

    def test_05_no_nonfinite_shape_dtype_or_source_change_adoption(self):
        import numpy as np
        for name, bad in [('nonfinite', np.full((4, 1, 2, 2), np.nan, dtype=np.float32)),
                          ('dtype', np.zeros((4, 1, 2, 2), dtype=np.float64)),
                          ('shape', np.zeros((4, 2), dtype=np.float32))]:
            root = self.root / name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'wrong shape/dtype/nonfinite'):
                B._build_bank(root, registration(5), lambda _ids: bad, self.guard)
            self.assertFalse((root / 'completion.json').exists())
            self.assertFalse(list((root / 'chunks').glob('chunk-*.json')))
        self.build('source', 5)
        changed = dict(registration(5), source='changed source')
        with self.assertRaisesRegex(ValueError, 'Changed source/input/shape'):
            B._build_bank(self.root / 'source', changed, encoder([]), self.guard)

    def test_06_commit_gaps_or_orphan_canonical_are_retained_and_refused(self):
        root = self.root / 'gap'
        def fault(stage, number):
            if stage == 'after_chunk_commit' and number == 1: raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):
            B._build_bank(root, registration(), encoder([]), self.guard, fault=fault)
        # Renaming two fixture files stays inside this resolved temporary root.
        original = root / 'chunks/chunk-000.json'; renamed = root / 'chunks/chunk-003.json'
        self.assertTrue(original.resolve().is_relative_to(self.root) and renamed.resolve().is_relative_to(self.root))
        original.rename(renamed)
        with self.assertRaisesRegex(ValueError, 'Nonconsecutive'):
            B._build_bank(root, registration(), encoder([]), self.guard)
        self.assertTrue(renamed.exists())
        orphan = self.root / 'orphan'; real_write = B.write_once
        def stop_completion(path, value, **kwargs):
            if Path(path).name == 'completion.json': raise RuntimeError('Injected completion write failure')
            return real_write(path, value, **kwargs)
        with patch.object(B, 'write_once', stop_completion), self.assertRaisesRegex(RuntimeError, 'completion write failure'):
            B._build_bank(orphan, registration(5), encoder([]), self.guard)
        self.assertTrue((orphan / 'native_VAE_train_modes.npy').exists())
        with self.assertRaisesRegex(ValueError, 'Orphan canonical'):
            B._build_bank(orphan, registration(5), encoder([]), self.guard)

    def test_07_guard_fails_before_directory_and_aliases_cannot_bypass(self):
        root = self.root / 'forbidden'
        def refusal(): raise ValueError('No admitted stage context')
        with self.assertRaisesRegex(ValueError, 'No admitted'):
            B._build_bank(root, registration(5), encoder([]), refusal)
        self.assertFalse(root.exists())
        if os.name == 'posix':
            target = self.root / 'target'; target.mkdir()
            alias_root = self.root / 'alias'; alias_root.mkdir()
            (alias_root / 'chunks').symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'aliased chunk'):
                B._build_bank(alias_root, registration(5), encoder([]), self.guard)
            self.assertFalse(list(target.iterdir()))

    def test_08_actual16540_metadata_all_rows_and_old_T_subset_refusal(self):
        path = B.ROOT / 'audit/image_manifest.csv'
        self.assertEqual(B.sha(path), '91e75aa52fbb0addc004eeb38a87aa237f48e0d6bdbd5f5cc3d0d0e839d1b145')
        with path.open(encoding='utf-8', newline='') as stream: raw = list(csv.DictReader(stream))
        rows = B.train_rows(dict(image_rows=raw))
        self.assertEqual(len(rows), 16540)
        self.assertEqual([r['array_row'] for r in rows], list(range(16540)))
        self.assertTrue(all(r['dataset_split'] == 'train' for r in rows))
        identity = B.image_identity(rows)
        self.assertEqual(len(identity), 16540)
        self.assertEqual(len(B.chunk_bounds(16540, 256)), 65)
        self.assertEqual(B.chunk_bounds(16540, 256)[-1], (16384, 16540))
        self.assertEqual(16540 * 4 * 64 * 64 * 4, 1083965440)
        with self.assertRaisesRegex(ValueError, 'Full official training'):
            B.train_rows(dict(image_rows=raw[:13230]))

    def test_09_original_native_preprocess_and_mode_scaling_math_CPU(self):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        original = B.ROOT / 'build_BrainHIVE_component_T_latents_v2.py'
        tree = ast.parse(original.read_bytes())
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'preprocess' for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        native = eval(compile(ast.Expression(assignments[0].value), str(original), 'eval'), {'transforms': transforms})
        pixels = np.arange(37 * 71 * 3, dtype=np.uint8).reshape(37, 71, 3)
        image = Image.fromarray(pixels, 'RGB')
        self.assertTrue(torch.equal(B.native_preprocess()(image), native(image)))
        calls = []
        class Distribution:
            def mode(self):
                calls.append(('mode', torch.is_grad_enabled()))
                return torch.arange(8, dtype=torch.float16).reshape(2, 1, 2, 2)
            def sample(self): raise AssertionError('Must never stochastic sample')
        class VAE:
            config = type('Config', (), dict(scaling_factor=.13025))()
            def encode(self, value): return type('Encoded', (), dict(latent_dist=Distribution()))()
        value = B.encode_native_modes(VAE(), torch.zeros(2, 3, 4, 4), torch)
        self.assertEqual(calls, [('mode', False)])
        self.assertTrue(torch.equal(value, torch.arange(8).reshape(2, 1, 2, 2).float() * .13025))
        self.assertEqual(value.dtype, torch.float32)
        self.assertFalse(torch.cuda.is_initialized())

    def test_10_actual_missing_review_rejects_before_Torch_payload_and_outputs(self):
        # Real CLI with an import hook; any tensor/model import would fail with
        # a distinct assertion instead of the expected missing-review error.
        review = self.root / 'missing-review.json'; absent = self.root / 'absent-plan'
        entry = B.ROOT / 'build_BrainHIVE_official_train_latents_v1.py'
        code = """import builtins,runpy,sys
original=builtins.__import__
def blocked(name,*a,**k):
 if name.split('.')[0] in ('torch','numpy','diffusers','torchvision','brainhive_native_models_v1'):
  raise AssertionError('MODEL_PAYLOAD_IMPORT_BEFORE_REVIEW')
 return original(name,*a,**k)
builtins.__import__=blocked
sys.argv=[sys.argv[1],'--plan-directory',sys.argv[2],'--execution-review',sys.argv[3]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(entry), str(absent), str(review)],
                                cwd=B.ROOT, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('MODEL_PAYLOAD_IMPORT_BEFORE_REVIEW', result.stderr)
        self.assertIn('review', result.stderr.lower())
        self.assertNotIn('ModuleNotFoundError', result.stderr)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_11_actual_low_cap_refuses_before_bank_preallocation_or_metadata_write(self):
        root = self.root / 'lowcap'; cap = 8192
        def reserve(additional):
            if B.allocated_bytes(root) + additional > cap: raise ValueError('Actual full write would exceed cap')
        with self.assertRaisesRegex(ValueError, 'would exceed cap'):
            B._build_bank(root, registration(), encoder([]), self.guard, reserve=reserve)
        self.assertFalse((root / 'native_VAE_train_modes.npy.pending').exists())
        self.assertLessEqual(B.allocated_bytes(root), cap)
        metadata = root / 'too_large.json'
        before = B.allocated_bytes(root)
        with self.assertRaisesRegex(ValueError, 'would exceed cap'):
            B.write_once(metadata, dict(large='x' * 20000), reserve=reserve)
        self.assertFalse(metadata.exists())
        self.assertFalse(metadata.with_name(metadata.name + '.pending.' + str(os.getpid())).exists())
        self.assertEqual(B.allocated_bytes(root), before)
        with self.assertRaisesRegex(ValueError, 'Production bank requires'):
            B._build_bank(self.root / 'production_no_budget', dict(registration(5), actual_official_full_train_scope=True), encoder([]), self.guard)

    def test_12_native_VAE_config_SHA_is_bound_by_original_decoder_not_new_review(self):
        path = B.ROOT / 'configs/ATM_public_decoder_registration_20261020_v1.json'
        self.assertEqual(B.sha(path), B.DECODER_REGISTRATION_SHA)
        decoder = json.loads(path.read_bytes())
        payload = next(x for x in decoder['generator_assets'] if 'sdxl_turbo_fp16' in x['path'])
        config = PurePosixPath(decoder['generator_asset_root']) / 'sdxl_config/vae/config.json'
        inputs = dict(generator=dict(path=payload['path'], sha256=payload['sha256']),
                      VAE_config=dict(path=str(config), sha256=decoder['source_sha256'][str(config)]))
        B.validate_decoder_bindings(inputs, decoder)
        self.assertEqual(inputs['VAE_config']['sha256'], '85a181424a5fd0cc292e44e467be66a5263976eee1baf36d208133f9d49d4269')
        wrong = dict(inputs, VAE_config=dict(inputs['VAE_config'], sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'Original native VAE config SHA/path'):
            B.validate_decoder_bindings(wrong, decoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(status='OFFICIAL_LATENT_BANK_CPU_CHECKS_PASSED_NO_REAL_VAE_OR_GPU' if result.wasSuccessful() else 'CPU_CHECK_FAILURE_PRESERVED',
        checks=result.testsRun, failures=len(result.failures), errors=len(result.errors), seconds=time.monotonic() - started,
        producer=B.ref(B.__file__), checker=B.ref(__file__), actual_VAE_models_loaded=0, actual_EEG_or_images_loaded=0,
        actual_full16540_bank_built=False, actual_full_training_metadata_rows=16540, GPU_jobs=0, official_execution_admitted=False)
    if args.receipt:
        require_parent = args.receipt.parent
        require_parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open('xb') as stream: stream.write(json.dumps(record, indent=2).encode() + b'\n')
    print(json.dumps(record))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__': main()
