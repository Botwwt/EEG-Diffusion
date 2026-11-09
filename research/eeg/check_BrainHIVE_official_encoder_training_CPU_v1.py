"""Synthetic native-class CPU training/transaction checks, never an official fit.

The tiny worker is private to this checker and has no real input/admission API.
It only constructs fixed 8-dimensional random native classes and CPU arrays.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Tuple, Union
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import brainhive_official_encoder_training_v1 as T

ROOT = Path(__file__).resolve().parent


def native_classes(torch):
    import torch.nn.functional as F
    scope = dict(torch=torch, nn=torch.nn, F=F, Dict=Dict, List=List, Tuple=Tuple, Union=Union)
    source = ROOT / 'reproduction_sources/BrainHIVE_f06db69f'
    pins = {row['path']: row['sha256'] for row in json.loads((source / '_fetch_receipt.json').read_text())['artifacts']}
    for filename, names in [('main/layers.py', ['BaseMLP', 'ResNet1DLayer', 'ResNet1DBlock', 'FusionEncoder']),
            ('main/models_brain.py', ['BrainMLP'])]:
        path = source / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins[filename]
        nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name in names]
        assert [node.name for node in nodes] == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def fixture(torch, samples=37, batch=8, epochs=4):
    scope = native_classes(torch)
    T.seed_everything(torch)
    teacher = scope['FusionEncoder'](hidden_size=8, intermediate_size=8, num_layers=1,
        proj_meta={'a': 4, 'b': 5, 'c': 6}, dropout=.1)
    T.frozen_stochastic_teacher(torch, teacher)
    T.seed_everything(torch)
    model = scope['BrainMLP'](6, 8, num_layers=1, dropout=.1)
    logit = torch.nn.Parameter(torch.tensor(2.6592))
    optimizer, scheduler = T.make_optimizer(torch, model, logit, math.ceil(samples / batch) * epochs)
    shuffle = torch.Generator(device='cpu').manual_seed(1730)
    data_rng = torch.Generator(device='cpu').manual_seed(1729)
    eeg = torch.randn(samples, 2, 3, generator=data_rng)
    visual = {name: torch.randn(samples, width, generator=data_rng) for name, width in {'a': 4, 'b': 5, 'c': 6}.items()}
    return dict(model=model, teacher=teacher, logit=logit, optimizer=optimizer, scheduler=scheduler,
        shuffle=shuffle, eeg=eeg, visual=visual, samples=samples, batch=batch, epochs=epochs,
        total_updates=math.ceil(samples / batch) * epochs)


def one_epoch(torch, value, epoch, updates):
    return T.run_epoch(torch, value['model'], value['teacher'], value['logit'], value['optimizer'],
        value['scheduler'], value['shuffle'], value['eeg'], value['visual'], epoch=epoch, updates=updates,
        total_updates=value['total_updates'], batch_size=value['batch'], subject='SYNTHETIC_CPU_ONLY',
        role='SYNTHETIC_NATIVE_8D_NO_OFFICIAL_FIT', device='cpu', deadline=time.monotonic() + 60)


def fixture_registration(root):
    raw = T.json_bytes(dict(status='SYNTHETIC_CPU_ONLY_NO_OFFICIAL_INPUTS_OR_FIT', samples=37, width=8,
        epochs=4, updates=20, training_seed=1729))
    path = root / 'fixture_registration.json'
    if not path.exists():
        T.write_once(path, raw)
    assert path.read_bytes() == raw
    return T.sha(path), hashlib.sha256(raw.rstrip(b'\n')).hexdigest(), T.sha(Path(__file__))


def scan(root, registration):
    return T.scan_committed_epochs(root, registration[0], registration[1], producer_source_sha256=registration[2],
        subject='SYNTHETIC_CPU_ONLY', role='SYNTHETIC_NATIVE_8D_NO_OFFICIAL_FIT', updates_per_epoch=5, total_epochs=4, full_epoch_rows=37)


def save_epoch(torch, root, value, row, updates, registration, fault=None):
    state = T.snapshot(torch, value['model'], value['logit'], value['optimizer'], value['scheduler'], value['shuffle'],
        row['epoch'], updates, *registration, 'cpu')
    return T.publish_epoch(torch, root, state, row, fault=fault)


def worker(mode, root):
    import torch
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    root = Path(root).resolve()
    assert root.is_relative_to(Path(tempfile.gettempdir()).resolve())
    root.mkdir(parents=True, exist_ok=True)
    reg = fixture_registration(root); value = fixture(torch)
    epochs, pending = scan(root, reg); start = updates = 0
    if mode == 'resume':
        assert len(epochs) == 2
        state = torch.load(Path(epochs[-1]['directory']) / 'resume.pt', weights_only=True, map_location='cpu')
        start, updates = T.restore(torch, state, value['model'], value['logit'], value['optimizer'], value['scheduler'],
            value['shuffle'], *reg, 'cpu', 5, 20)
    else:
        assert not epochs and not pending
    until = 2 if mode == 'first' else 4
    for number in range(start + 1, until + 1):
        row, updates = one_epoch(torch, value, number, updates)
        save_epoch(torch, root, value, row, updates, reg)
    epochs, pending = scan(root, reg)
    if until == 4:
        T.publish_final(torch, root, value['model'], value['logit'], registration_sha256=reg[0],
            registration_core_sha256=reg[1], producer_source_sha256=reg[2], committed=epochs, total_epochs=4,
            total_updates=20, subject='SYNTHETIC_CPU_ONLY', role='SYNTHETIC_NATIVE_8D_NO_OFFICIAL_FIT')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='SYNTHETIC_CPU_TRANSACTION_ONLY', committed_epochs=len(epochs), actual_official_fits=0)))


def reject_worker(root):
    """Fresh interpreter: missing review must beat every model/payload/write."""
    import builtins
    from unittest.mock import patch
    import train_BrainHIVE_official_encoder_v1 as entry
    assert 'torch' not in sys.modules
    original = builtins.__import__
    def no_models(name, *args, **kwargs):
        assert name.split('.')[0] not in ('torch', 'brainhive_native_models_v1'), 'Model imported before admission'
        return original(name, *args, **kwargs)
    with patch.object(builtins, '__import__', no_models), patch.object(entry.G.E, 'read_plan_bundle', side_effect=AssertionError('plan read before review')), \
            patch.object(entry.G, 'verify_payload_files', side_effect=AssertionError('payload read before review')), \
            patch.object(Path, 'mkdir', side_effect=AssertionError('training directory created before review')):
        try:
            entry.main(['--plan-directory', str(root / 'never_read_plan'), '--task-id', 'sub-01/official_oof0',
                '--execution-review', str(root / 'missing_sealed_review.json')])
        except ValueError as error:
            assert 'does not exist' in str(error) and 'NOT admitted' in str(error)
        else:
            raise AssertionError('Missing official review was admitted')
        # Negative-only API boundary fixture: even a separately successful
        # export context must not dispatch into training or load a payload.
        with patch.object(entry.G, 'require_context', return_value=None):
            try:
                entry.run_admitted({'action': 'export'})
            except ValueError as error:
                assert 'Only the train action' in str(error)
            else:
                raise AssertionError('Export action entered the training engine')
    assert 'torch' not in sys.modules and not (root / 'never_read_plan').exists()
    print('MISSING_REVIEW_REJECTED_BEFORE_MODEL_PAYLOAD_AND_WRITES')


def same_tree(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return set(left) == set(right) and all(same_tree(torch, left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) == type(right) and len(left) == len(right) and all(same_tree(torch, a, b) for a, b in zip(left, right))
    return left == right


class EarlyGateChecks(unittest.TestCase):
    def test_missing_sealed_review_rejects_in_fresh_interpreter(self):
        with tempfile.TemporaryDirectory(prefix='BrainHIVE_gate_cpu_') as temporary:
            result = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--_fixture-worker', 'reject',
                '--fixture-root', temporary], capture_output=True, text=True, timeout=30,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('MISSING_REVIEW_REJECTED_BEFORE_MODEL_PAYLOAD_AND_WRITES', result.stdout)


class TrainingCoreChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        assert not torch.cuda.is_initialized()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='BrainHIVE_epoch_cpu_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.reg = fixture_registration(self.root)

    def test_native_optimizer_and_full8270_synthetic_last78_batch(self):
        torch = self.torch; value = fixture(torch, samples=8270, batch=1024, epochs=1)
        before = {key: item.clone() for key, item in value['teacher'].state_dict().items()}
        row, updates = one_epoch(torch, value, 1, 0)
        self.assertEqual((row['full_epoch_rows'], updates, 8270 % 1024), (8270, 9, 78))
        self.assertTrue(all(torch.equal(before[key], item) for key, item in value['teacher'].state_dict().items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in value['teacher'].parameters()))
        groups = value['optimizer'].param_groups
        self.assertEqual([g['weight_decay'] for g in groups], [.05, 0.])
        self.assertTrue(any(parameter is value['logit'] for parameter in groups[0]['params']))
        norm_ids = {id(p) for module in value['model'].modules() if isinstance(module, torch.nn.LayerNorm) for p in module.parameters()}
        self.assertTrue(norm_ids.issubset({id(p) for p in groups[1]['params']}))

    def test_fresh_process_continuous_vs_complete_epoch_resume_bit_equal(self):
        torch = self.torch; continuous = self.root / 'continuous'; resumed = self.root / 'resumed'
        for mode, root in [('continuous', continuous), ('first', resumed), ('resume', resumed)]:
            result = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--_fixture-worker', mode,
                '--fixture-root', str(root)], capture_output=True, text=True, timeout=60,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
            self.assertEqual(result.returncode, 0, result.stderr)
        a = torch.load(continuous / 'epochs/epoch-04/resume.pt', weights_only=True, map_location='cpu')
        b = torch.load(resumed / 'epochs/epoch-04/resume.pt', weights_only=True, map_location='cpu')
        self.assertTrue(same_tree(torch, a, b), 'model/logit/optimizer/scheduler/CPU+shuffle RNG must match bit-for-bit')
        self.assertEqual((continuous / 'epochs.jsonl').read_bytes(), (resumed / 'epochs.jsonl').read_bytes())
        self.assertFalse(torch.cuda.is_initialized())

    def test_payload_before_commit_is_retained_and_not_adopted(self):
        torch = self.torch; value = fixture(torch); row, updates = one_epoch(torch, value, 1, 0)
        def crash(stage):
            if stage == 'payloads_written_before_commit': raise RuntimeError('injected precommit crash')
        with self.assertRaisesRegex(RuntimeError, 'precommit'):
            save_epoch(torch, self.root, value, row, updates, self.reg, fault=crash)
        committed, pending = scan(self.root, self.reg)
        self.assertEqual(committed, []); self.assertEqual(len(pending), 1)
        self.assertTrue((Path(pending[0]) / 'resume.pt').is_file())
        self.assertFalse((Path(pending[0]) / 'commit.json').exists())

    def test_rename_is_authoritative_without_canonical_resume(self):
        torch = self.torch; value = fixture(torch); row, updates = one_epoch(torch, value, 1, 0)
        before_rng = torch.get_rng_state().clone()
        def crash(stage):
            if stage == 'epoch_rename_complete_before_canonical_publication': raise RuntimeError('injected postrename crash')
        with self.assertRaisesRegex(RuntimeError, 'postrename'):
            save_epoch(torch, self.root, value, row, updates, self.reg, fault=crash)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        committed, pending = scan(self.root, self.reg)
        self.assertEqual(len(committed), 1); self.assertEqual(pending, [])
        self.assertFalse((self.root / 'resume.pt').exists())
        restored = fixture(torch)
        state = torch.load(Path(committed[0]['directory']) / 'resume.pt', weights_only=True, map_location='cpu')
        self.assertEqual(T.restore(torch, state, restored['model'], restored['logit'], restored['optimizer'],
            restored['scheduler'], restored['shuffle'], *self.reg, 'cpu', 5, 20), (1, 5))

    def test_changed_resume_registration_or_log_is_rejected(self):
        torch = self.torch; value = fixture(torch); row, updates = one_epoch(torch, value, 1, 0)
        save_epoch(torch, self.root, value, row, updates, self.reg)
        path = self.root / 'epochs/epoch-01/epoch.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'Changed epoch'):
            scan(self.root, self.reg)

    def test_exact_resume_rejects_incomplete_rng_optimizer_or_scheduler(self):
        torch = self.torch; value = fixture(torch); row, updates = one_epoch(torch, value, 1, 0)
        state = T.snapshot(torch, value['model'], value['logit'], value['optimizer'], value['scheduler'], value['shuffle'],
            1, updates, *self.reg, 'cpu')
        bad_states = []
        bad = copy.deepcopy(state); del bad['torch_rng']; bad_states.append(bad)
        bad = copy.deepcopy(state); bad['optimizer']['state'] = {}; bad_states.append(bad)
        bad = copy.deepcopy(state); bad['optimizer']['param_groups'][0]['lr'] = .03; bad_states.append(bad)
        bad = copy.deepcopy(state); bad['scheduler']['last_epoch'] = 0; bad_states.append(bad)
        bad = copy.deepcopy(state); bad['registration_core_sha256'] = 'f' * 64; bad_states.append(bad)
        for number, bad in enumerate(bad_states):
            with self.subTest(number=number), self.assertRaises(ValueError):
                T.restore(torch, bad, value['model'], value['logit'], value['optimizer'], value['scheduler'],
                    value['shuffle'], *self.reg, 'cpu', 5, 20)

    def test_final_and_all_prior_resume_payloads_are_never_overwritten(self):
        torch = self.torch; value = fixture(torch); updates = 0
        for number in range(1, 5):
            row, updates = one_epoch(torch, value, number, updates)
            save_epoch(torch, self.root, value, row, updates, self.reg)
        committed, _ = scan(self.root, self.reg)
        kwargs = dict(registration_sha256=self.reg[0], registration_core_sha256=self.reg[1], producer_source_sha256=self.reg[2],
            committed=committed, total_epochs=4, total_updates=20, subject='SYNTHETIC_CPU_ONLY', role='SYNTHETIC_NATIVE_8D_NO_OFFICIAL_FIT')
        first = T.publish_final(torch, self.root, value['model'], value['logit'], **kwargs)
        paths = [self.root / 'brain_final.pt', self.root / 'resume.pt'] + [Path(item['directory']) / 'resume.pt' for item in committed]
        facts = [(p.stat().st_ino, p.stat().st_mtime_ns, T.sha(p)) for p in paths]
        second = T.publish_final(torch, self.root, value['model'], value['logit'], **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(facts, [(p.stat().st_ino, p.stat().st_mtime_ns, T.sha(p)) for p in paths])
        with torch.no_grad(): value['logit'].add_(1.)
        with self.assertRaisesRegex(ValueError, 'Existing final differs'):
            T.publish_final(torch, self.root, value['model'], value['logit'], **kwargs)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--_fixture-worker', choices=('continuous', 'first', 'resume', 'reject'))
    parser.add_argument('--fixture-root', type=Path)
    args, rest = parser.parse_known_args()
    if args._fixture_worker:
        assert args.fixture_root is not None and not rest
        return reject_worker(args.fixture_root) if args._fixture_worker == 'reject' else worker(args._fixture_worker, args.fixture_root)
    assert args.fixture_root is None
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)


if __name__ == '__main__':
    main()
