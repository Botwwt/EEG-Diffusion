"""Read-only current published-parent payload audit; no benchmark readout.

Strict-loads all ten historical full16540 parents and checks the existing EEG
condition caches. It does not recompute EEG exports, read targets, run a model
forward, fit, admit model reuse or substitute for new official-full OOF models.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from typing import Dict, List, Tuple, Union

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
UPSTREAM = ROOT / 'reproduction_sources/BrainHIVE_f06db69f'
SUBJECTS = tuple('sub-' + str(i).zfill(2) for i in range(1, 11))
PROOF_SHA = '942d530b9c78ec9b5d68c72382dbfd4f281c8d2f13969848e1201076a05a1352'
REG_SHA = '4bc6f4929654ca67a09468c875239abaf30ed440527ea9854dd4a096c1b268cc'
FETCH_SHA = 'e53b1da7d72b636a6be7f3048c1f22a414f7708f463c9d00b4e15ba090a03d4f'
SOURCE_PINS = {'main/layers.py': '560b038708652fac9baa98a1f8f642812eacb7bbb172a56e2c8199ebbdc5b559',
    'main/models_brain.py': 'fc06ed93c1f1f8659605ec561d87e85f91ca72856fe034f4cf8416c2b60b39b1'}
AFFINE = ('block.layers.0.norm.weight', 'block.layers.0.norm.bias')
MAX_SECONDS = 300


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resource_headroom():
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    maximum = int(Path('/sys/fs/cgroup/memory.max').read_text())
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    value = min(available, maximum - current)
    require(value >= 12 << 30, 'Leave current formal training its fixed memory guard')
    return value


def file_identity(path, expected):
    path = Path(path)
    before = path.stat()
    require(path.is_file() and not path.is_symlink(), 'Explicit regular historical payload required')
    require(sha(path) == expected, 'Historical payload SHA changed')
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Payload changed while reading')
    return dict(path=str(path), sha256=expected, bytes=after.st_size, device=after.st_dev,
                inode=after.st_ino, mtime_ns=after.st_mtime_ns)


def release_clean_cache(path):
    # Only this verified read-only file's clean page cache; no file deletion.
    with Path(path).open('rb') as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def source_class(torch):
    require(sha(UPSTREAM / '_fetch_receipt.json') == FETCH_SHA, 'Original fetch receipt changed')
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, Dict=Dict, List=List, Tuple=Tuple, Union=Union)
    for filename, names in [('main/layers.py', ['BaseMLP', 'ResNet1DLayer', 'ResNet1DBlock', 'FusionEncoder']),
                            ('main/models_brain.py', ['BrainMLP'])]:
        path = UPSTREAM / filename
        require(sha(path) == SOURCE_PINS[filename], 'Original class source changed')
        nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name in names]
        require([node.name for node in nodes] == names, 'Wrong native class extraction')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['BrainMLP']


def validate_checkpoint(torch, model, state):
    require((state['seed'], state['epochs'], state['updates'], state['registration_sha256']) ==
            (1729, 25, 425, REG_SHA), 'Wrong historical full-fit checkpoint budget/registration')
    require(sum(p.numel() for p in model.parameters()) == 6452224, 'Actual native model size differs')
    model.load_state_dict(state['model'], strict=True)
    model.eval().requires_grad_(False)
    require(all(t.device.type == 'cpu' and (not t.is_floating_point() or bool(torch.isfinite(t).all()))
                for t in model.state_dict().values()), 'Nonfinite or non-CPU state')
    require(state['logit_scale'].numel() == 1 and bool(torch.isfinite(state['logit_scale']).all()), 'Invalid trained logit scale')
    for key in AFFINE:
        require(state['model'][key].shape == (1024,) and state['model'][key].dtype == torch.float32,
                'Own native output affine differs')
    affine = {key: hashlib.sha256(state['model'][key].contiguous().numpy().tobytes()).hexdigest() for key in AFFINE}
    return dict(strict_native_state_load=True, parameters=6452224, finite_CPU_state=True,
        own_affine_raw_float32_byte_sha256=affine, own_gamma_minimum_absolute=float(state['model'][AFFINE[0]].abs().min()))


def validate_cache(torch, cache, expected_model_sha):
    require(cache['role'] == 'published' and cache['model_sha256'] == expected_model_sha
            and cache['normalization'] == 'raw LayerNorm output', 'Existing cache points to another model/normalization')
    require(cache['array_rows'].dtype == torch.int64 and torch.equal(cache['array_rows'], torch.arange(200)),
            'Existing R80 cache must retain every array row in native order')
    condition = cache['condition']
    require(condition.device.type == 'cpu' and condition.dtype == torch.float32 and tuple(condition.shape) == (200, 1024)
            and bool(torch.isfinite(condition).all()), 'Existing complete raw condition cache invalid')
    return dict(existing_R80_cache_shape=[200, 1024], existing_R80_cache_dtype='float32',
        existing_R80_cache_model_SHA_link_valid=True, existing_cache_array_order_complete=True,
        original_R80_EEG_or_export_recomputed=False, forwards=0, images_read=0, scores_read=0)


def validate_loaded(torch, model, state, cache, expected_model_sha):
    return dict(validate_checkpoint(torch, model, state), **validate_cache(torch, cache, expected_model_sha))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    expected_output = ROOT / f'audit/published_parent_payloads_CPU_20261022_v1/current_payloads_attempt{args.attempt}.json'
    require(os.name == 'posix' and args.output == expected_output and not args.output.exists(), 'One new explicit read-only receipt; preserve prior runs')
    resource_headroom()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU-only audit')
    args.output.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    pending_cache_release = set()
    result = dict(status='PUBLISHED_PARENT_PAYLOAD_CPU_AUDIT_RUNNING', UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_sha256=sha(__file__), models=[], GPU_initialized=False, actual_new_fits=0, forwards=0,
        benchmark_target_or_visual_payloads_read=0, score_tables_read=0, official_method_locked=False,
        model_reuse_admitted=False, benchmark_readout_admitted=False, may_replace_new_official_OOF=False,
        existing_R80_cache_exports_recomputed=False, old_files_deleted=0)
    def timeout(_signum, _frame):
        raise TimeoutError('Finite read-only published parent audit; retain partial evidence')
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        proof_path = ROOT / 'audit/BrainHIVE_all40_parent_training_CPU_verification_v1.json'
        require(sha(proof_path) == PROOF_SHA, 'Original historical CPU evidence changed')
        proof = json.loads(proof_path.read_text())
        require(proof['status'] == 'ALL40_CHECKPOINTS_LOGS_AND_CONDITION_IDENTITIES_VERIFIED_CPU', 'Wrong original evidence')
        rows = [row for row in proof['models'] if row['role'] == 'published']
        require(len(rows) == 10 and {r['subject'] for r in rows} == set(SUBJECTS), 'All ten distinct subjects required')
        registration = WORK / 'runs/published_source_dropout_v2/registration.json'
        require(sha(registration) == REG_SHA, 'Current registration differs from original checkpoint registration')
        reg = json.loads(registration.read_text())
        require(reg['training_array_rows'] == list(range(16540)) and reg['excluded_oof_rows'] == []
                and reg['subjects'] == list(SUBJECTS), 'Published full official train scope differs')
        require(reg['training_seed'] == 1729 and reg['frozen_visual_training_dropout'] == .1
                and reg['deterministic_F_cache_used_as_training_target'] is False, 'Wrong original stochastic teacher metadata')
        BrainMLP = source_class(torch)
        minimum = resource_headroom(); verified_caches = 0; unavailable_caches = []
        for subject in SUBJECTS:
            minimum = min(minimum, resource_headroom())
            row = next(r for r in rows if r['subject'] == subject)
            folder = WORK / 'runs/published_source_dropout_v2' / subject
            checkpoint, cachefile = folder / 'brain_final.pt', folder / 'conditions.pt'
            require(str(checkpoint) == row['checkpoint'], 'Wrong historical checkpoint location')
            before_model = file_identity(checkpoint, row['checkpoint_sha256'])
            pending_cache_release.add(checkpoint)
            state = torch.load(checkpoint, weights_only=True, map_location='cpu')
            model = BrainMLP(4250, 1024, num_layers=1, dropout=.1)
            checked = validate_checkpoint(torch, model, state)
            cache_record = dict(logical_path=str(cachefile), expected_original_sha256=row['condition_cache_sha256'])
            if cachefile.exists():
                physical_cache = cachefile.resolve(strict=True)
                require(physical_cache.is_absolute() and any(physical_cache.is_relative_to(root) for root in
                    (WORK, Path('/dev/shm'), Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1'))), 'Cache target outside known private runtime')
                before_cache = file_identity(physical_cache, row['condition_cache_sha256'])
                pending_cache_release.add(physical_cache)
                cache = torch.load(physical_cache, weights_only=True, map_location='cpu')
                cache_record.update(status='EXISTING_CACHE_PAYLOAD_VERIFIED', physical_file=before_cache,
                    **validate_cache(torch, cache, row['checkpoint_sha256']))
                del cache; gc.collect()
                require(file_identity(physical_cache, row['condition_cache_sha256']) == before_cache, 'Cache mutated during audit')
                release_clean_cache(physical_cache); pending_cache_release.discard(physical_cache); verified_caches += 1
            else:
                cache_record.update(status='MISSING_DANGLING_SYMLINK' if cachefile.is_symlink() else 'MISSING',
                    symlink_target=os.readlink(cachefile) if cachefile.is_symlink() else None,
                    cache_payload_verified=False, export_replay_attempted=False)
                unavailable_caches.append(subject)
            complete_path, epochs_path = folder / 'training_complete.json', folder / 'epochs.jsonl'
            complete = json.loads(complete_path.read_text())
            require(complete['subject'] == subject and complete['role'] == 'published'
                    and complete['model_sha256'] == row['checkpoint_sha256']
                    and (complete['epochs'], complete['updates'], complete['unique_train_stimuli'], complete['training_exposures']) ==
                    (25, 425, 16540, 413500), 'Current training completion differs')
            epochs = [json.loads(line) for line in epochs_path.read_text().splitlines()]
            require(len(epochs) == 25, 'All original epochs required')
            for epoch, record in enumerate(epochs, 1):
                require((record['subject'], record['role'], record['epoch'], record['updates'], record['expected_updates'], record['full_epoch_rows']) ==
                        (subject, 'published', epoch, epoch * 17, 425, 16540), 'Current epoch coverage differs')
            del state, model; gc.collect()
            require(file_identity(checkpoint, row['checkpoint_sha256']) == before_model, 'Historical checkpoint mutated during audit')
            release_clean_cache(checkpoint)
            pending_cache_release.discard(checkpoint)
            result['models'].append(dict(subject=subject, checkpoint=before_model, existing_R80_cache=cache_record,
                training_completion_sha256=sha(complete_path), full25epoch_log_sha256=sha(epochs_path), **checked))
            minimum = min(minimum, resource_headroom())
        require(not torch.cuda.is_initialized() and sha(registration) == REG_SHA and sha(proof_path) == PROOF_SHA, 'CPU/source state changed')
        result.update(status=('ALL10_PUBLISHED_PARENT_CURRENT_PAYLOADS_STRICT_LOAD_VERIFIED_CPU_REUSE_ADMISSION_PENDING' if verified_caches == 10
            else 'ALL10_PUBLISHED_CHECKPOINTS_STRICT_LOAD_VERIFIED_CPU_EXISTING_R80_CACHES_UNAVAILABLE_REUSE_PENDING'),
            current_models_verified=10, current_existing_R80_caches_verified=verified_caches,
            unavailable_existing_R80_cache_subjects=unavailable_caches,
            registration_sha256=REG_SHA, source_artifact_sha256=SOURCE_PINS,
            observed_minimum_CPU_headroom_bytes=minimum,
            limits=['No EEG export replay or native CUDA/BF16 inference verified here.',
                    'Not new official OOF models, not a method lock, reuse admission or benchmark readout.',
                    'Previously exposed official200 remains exposed; no independent H restored.'])
    except BaseException as error:
        result.update(status='PUBLISHED_PARENT_PAYLOAD_CPU_AUDIT_FAILED_PARTIAL_RETAINED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['failure_path_verified_file_cache_release_errors'] = []
        for path in pending_cache_release:
            try:
                release_clean_cache(path)
            except OSError as error:
                result['failure_path_verified_file_cache_release_errors'].append(dict(path=str(path), error=str(error)))
        result['seconds'] = time.monotonic() - started
        raw = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
        require(len(raw) <= 128 << 10, 'Finite metadata budget exceeded')
        args.output.write_bytes(raw)
    print(json.dumps(dict(status=result['status'], receipt=str(args.output), sha256=sha(args.output))))


if __name__ == '__main__':
    main()
