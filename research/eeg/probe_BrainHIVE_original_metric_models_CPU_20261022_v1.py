"""Finite sequential CPU strict-load probe of the five restored paper assets.

The coordinator imports no model library and observes one owned CPU worker at a
time. There are no image inputs, forwards, metric scores, downloads or fits.
Workers exit between models so their Torch allocations are actually released.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import restore_BrainHIVE_original_metric_assets_CPU_20261022_v1 as A

ROOT = Path(__file__).resolve().parent
ASSET_RECEIPT = ROOT / 'audit/original_metrics_restoration_20261022_v1/persistent_asset_recovery_attempt1.json'
ASSET_RECEIPT_SHA = '7e05d482f8ee77ecf20154019478c8e634557f929128fc2817ee0f58356a175f'
RESTORER_SHA = '83c2efe2c196e7613ec65f650f1d6276a202d26d55a7f6b7480aee2b5f027a18'
MAX_SECONDS, MAX_BYTES, MIN_MEMORY = 1800, 256 << 10, 12 << 30
STAGES = ('AlexNet', 'Inception', 'EffNet', 'SwAV', 'CLIP_L14')
PARAMETERS = dict(AlexNet=61100840, Inception=27161264, EffNet=7794184, SwAV=25557032, CLIP_L14=427616513)
SWAV_MISSING = ('fc.weight', 'fc.bias')
SWAV_UNUSED = ('projection_head.0.weight', 'projection_head.0.bias',
    'projection_head.1.weight', 'projection_head.1.bias', 'projection_head.1.running_mean',
    'projection_head.1.running_var', 'projection_head.1.num_batches_tracked',
    'projection_head.3.weight', 'projection_head.3.bias', 'prototypes.weight')
CLIP_COMMIT = 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
PACKAGE = ROOT / 'reproduction_sources/OpenAI_CLIP_d05afc436d78'


def memory(extra=0):
    value = A.C.resource_headroom()
    assert value['effective_headroom_bytes'] >= MIN_MEMORY + extra, \
        'Preserve12GiB actual CPU headroom plus this worker preflight margin'
    return value


def payload_rows(receipt):
    assert receipt['status'] == 'ALL5_PERSISTENT_ASSET_SHA_READY_CPU_MODEL_PROBE_PENDING'
    assert receipt['GPU_used'] is False and receipt['image_values_read'] == receipt['score_values_read'] == 0
    assert receipt['original29_and_original_metric_sources_and_metadata_unchanged'] is True
    assert len(receipt['assets']) == 5 and {row['name'] for row in receipt['assets']} == set(STAGES)
    result = {}
    for row in receipt['assets']:
        expected = A.ASSETS[row['name']]; actual = row['persistent_asset']
        assert row['logical_path'] == expected['path']
        assert actual['sha256'] == expected['sha256'] and actual['bytes'] == expected['bytes']
        path = Path(actual['path'])
        if row['name'] in ('AlexNet', 'CLIP_L14'):
            assert path == A.DURABLE / Path(expected['path']).name
        elif row['name'] in ('Inception', 'SwAV'):
            assert path == Path(expected['path'])
        else:
            assert path.is_relative_to(A.SHARED_DURABLE) and path != A.SHARED_DURABLE
        assert actual['clean_pagecache_advised'] is True
        result[row['name']] = dict(expected, durable_path=str(path))
    return result


def swav_coverage(missing, unexpected):
    assert len(missing) == 2 and set(missing) == set(SWAV_MISSING)
    assert len(unexpected) == len(SWAV_UNUSED) and set(unexpected) == set(SWAV_UNUSED)


def source_checks():
    """Hash only small source/metadata/package files, never core payloads."""
    assert A.C.sha(Path(A.__file__)) == RESTORER_SHA
    assert A.C.sha(ASSET_RECEIPT) == ASSET_RECEIPT_SHA
    receipt = json.loads(ASSET_RECEIPT.read_text()); rows = payload_rows(receipt)
    pins = {str(ROOT / name): digest for name, digest in A.SOURCE_PINS.items()}
    pins[str(ASSET_RECEIPT)] = ASSET_RECEIPT_SHA
    pins[str(Path(A.__file__))] = RESTORER_SHA
    manifest = ROOT / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json'
    expected = receipt['source_and_metadata_sha256'][str(manifest)]
    assert A.C.sha(manifest) == expected
    pins[str(manifest)] = expected
    sources = json.loads(manifest.read_text())['source_sha256']
    assert len(sources) == 29 and all(Path(name).name == name for name in sources)
    pins.update({str(ROOT / name): digest for name, digest in sources.items()})
    # Reuse only the restorer's small package-provenance check. No model import.
    packages = A.package_presence(pins)
    old = A.OLD_RECEIPT
    assert A.C.sha(old) == receipt['source_and_metadata_sha256'][str(old)]
    pins[str(old)] = receipt['source_and_metadata_sha256'][str(old)]
    original = json.loads(old.read_text()); A.validate_old_receipt(original)
    assert all(A.C.sha(path) == digest for path, digest in pins.items())
    clip_pin = ROOT / 'audit/BrainHIVE_CLIP_metric_source_pin_20261020_v1.json'
    proof = json.loads(clip_pin.read_text())
    assert proof['commit'] == CLIP_COMMIT and len(proof['files']) == 3
    assert {row['path'] for row in proof['files']} == {'clip/model.py', 'clip/clip.py', 'LICENSE'}
    for item in proof['files']:
        path = PACKAGE / item['path']; raw = path.read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
        assert item['sha256'] == A.SOURCE_PINS[str(path.relative_to(ROOT)).replace('\\', '/')]
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == item['git_blob']
    pins[str(clip_pin)] = A.C.sha(clip_pin)
    return rows, pins, packages


def release_payload(path):
    """Drop only this durable payload's clean cache after the child read it."""
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        assert os.fstat(fd).st_dev != Path('/dev/shm').stat().st_dev
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def validate_model_record(value, stage):
    assert value['status'] == 'ORIGINAL_METRIC_MODEL_CPU_STRICT_LOAD_COMPLETE'
    assert value['stage'] == stage and value['parameters'] == PARAMETERS[stage]
    assert value['strict_original_load'] is True and value['all_floating_parameters_and_buffers_finite'] is True
    assert value['all_parameter_and_buffer_devices'] == ['cpu']
    assert value['GPU_initialized'] is False and value['images_read'] == value['forwards'] == value['scores_read'] == 0
    if stage == 'SwAV':
        swav_coverage(value['missing_unused_head'], value['unused_pretraining_heads'])
    if stage == 'CLIP_L14':
        assert value['source_commit'] == CLIP_COMMIT
        assert value['visual_resolution'] == 224 and value['visual_layers'] == 24
        assert value['patch_kernel'] == [14, 14] and value['embedding_dimensions'] == 768


def worker(stage, output, parent_pid):
    assert stage in STAGES and os.getppid() == parent_pid and parent_pid > 1
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules
    started = time.monotonic(); memory()
    rows, pins, packages = source_checks()
    asset = rows[stage]; target = Path(asset['durable_path'])
    assert target.is_file() and target.resolve(strict=True) == target and not target.is_symlink()
    assert target.stat().st_size == asset['bytes']
    sys.path.insert(0, str(A.PACKAGE_ROOT))
    os.nice(10)
    import torch
    import torchvision
    import skimage
    import lazy_loader
    import numpy
    import scipy
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity
    assert torch.__version__ == '2.3.0+cu121' and torchvision.__version__ == '0.18.0+cu121'
    assert skimage.__version__ == '0.24.0' and lazy_loader.__version__ == '0.4'
    assert Path(skimage.__file__).resolve().is_relative_to(A.PACKAGE_ROOT)
    assert Path(lazy_loader.__file__).resolve().is_relative_to(A.PACKAGE_ROOT)
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    from torchvision.models import alexnet, inception_v3, efficientnet_b1, resnet50
    memory()
    details = {}
    if stage == 'CLIP_L14':
        spec = importlib.util.spec_from_file_location('pinned_original_CLIP_CPU_model', PACKAGE / 'clip/model.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        scripted = torch.jit.load(str(target), map_location='cpu').eval()
        state = scripted.state_dict()
        assert all(torch.isfinite(t).all().item() for t in state.values() if t.is_floating_point())
        memory()
        model = module.build_model(state).eval()
        del state, scripted; gc.collect()
        details.update(source_commit=CLIP_COMMIT, visual_resolution=model.visual.input_resolution,
            visual_layers=len(model.visual.transformer.resblocks), patch_kernel=list(model.visual.conv1.kernel_size),
            embedding_dimensions=model.text_projection.shape[1], original_mixed_weight_dtype_preserved=True)
    else:
        constructor = dict(AlexNet=lambda: alexnet(weights=None),
            Inception=lambda: inception_v3(weights=None, init_weights=False),
            EffNet=lambda: efficientnet_b1(weights=None), SwAV=lambda: resnet50(weights=None))[stage]
        model = constructor()
        state = torch.load(str(target), weights_only=True, map_location='cpu')
        assert isinstance(state, dict) and state
        assert all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items())
        assert all(torch.isfinite(t).all().item() for t in state.values() if t.is_floating_point())
        memory()
        if stage == 'SwAV':
            renamed = {key.replace('module.', ''): value for key, value in state.items()}
            assert len(renamed) == len(state), 'No parameter-name collision after original prefix mapping'
            coverage = model.load_state_dict(renamed, strict=False)
            swav_coverage(coverage.missing_keys, coverage.unexpected_keys)
            details.update(missing_unused_head=coverage.missing_keys, unused_pretraining_heads=coverage.unexpected_keys)
            del renamed
        else:
            coverage = model.load_state_dict(state, strict=True)
            assert not coverage.missing_keys and not coverage.unexpected_keys
        del state; gc.collect()
        model.eval()
    model.requires_grad_(False)
    tensors = list(model.parameters()) + list(model.buffers())
    assert tensors and all(t.device.type == 'cpu' for t in tensors)
    assert all(torch.isfinite(t).all().item() for t in tensors if t.is_floating_point())
    value = dict(status='ORIGINAL_METRIC_MODEL_CPU_STRICT_LOAD_COMPLETE', stage=stage,
        parameters=sum(t.numel() for t in model.parameters()), strict_original_load=True,
        all_floating_parameters_and_buffers_finite=True, all_parameter_and_buffer_devices=['cpu'],
        GPU_initialized=torch.cuda.is_initialized(), images_read=0, forwards=0, scores_read=0,
        asset=dict(path=str(target), sha256=asset['sha256'], bytes=asset['bytes']),
        CPU_execution_only=True, native_GPU_evaluation_or_precision_verified=False,
        package_versions=dict(torch=torch.__version__, torchvision=torchvision.__version__, skimage=skimage.__version__,
            lazy_loader=lazy_loader.__version__, numpy=numpy.__version__, scipy=scipy.__version__),
        original_low_level_package_functions_importable=callable(rgb2gray) and callable(structural_similarity),
        memory_after_load=memory(), source_and_metadata_sha256=pins, package_files_present=packages,
        seconds=time.monotonic() - started, **details)
    validate_model_record(value, stage)
    assert all(A.C.sha(path) == digest for path, digest in pins.items())
    del tensors, model; gc.collect()
    assert not torch.cuda.is_initialized()
    A.write(output, value)


def output_bytes(directory):
    used = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    assert used <= MAX_BYTES, 'Preserve partial metadata/logs after256KiB cap'
    return used


def finish_child(process):
    if process.poll() is None:
        process.terminate()
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=10)


def cleanup_after_failure(status, active):
    # A failed /proc/cgroup read must never prevent owned-worker cleanup.
    try:
        status['failure_resource_snapshot_before_owned_child_cleanup'] = A.C.resource_headroom()
    except BaseException as error:
        status['failure_resource_snapshot_error'] = dict(type=type(error).__name__, detail=str(error)[:300])
    if active is not None:
        try:
            status['failure_owned_child_exit_code'] = finish_child(active)
            status['failure_owned_child_waited'] = True
        except BaseException as error:
            status['failure_owned_child_waited'] = False
            status['failure_owned_child_cleanup_unresolved'] = dict(type=type(error).__name__, detail=str(error)[:300])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=STAGES)
    parser.add_argument('--worker-output', type=Path)
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT == A.SERVER_ROOT and sys.version_info[:2] == (3, 12)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.worker:
        expected = A.AUDIT / f'CPU_model_probe_attempt{args.attempt}' / (args.worker + '.json')
        assert 1 <= args.attempt <= 3 and args.worker_output == expected
        return worker(args.worker, args.worker_output, args.parent_pid)
    assert not args.worker_output and not args.parent_pid and 1 <= args.attempt <= 3
    output = A.AUDIT / f'CPU_model_probe_attempt{args.attempt}'
    assert not output.exists(), 'Preserve prior success/failure and partial logs; no automatic retries'
    output.mkdir(mode=0o700)
    assert output.resolve(strict=True) == output
    completion = output / 'completion.json'
    # Retain20 seconds inside the total bound for owned-worker terminate/kill.
    started = time.monotonic(); deadline = started + MAX_SECONDS - 20
    status = dict(status='ORIGINAL_FIVE_METRIC_CPU_MODEL_PROBE_RUNNING', UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_sha256=A.C.sha(__file__), maximum_seconds=MAX_SECONDS, minimum_CPU_headroom_bytes=MIN_MEMORY,
        headroom_sampling_seconds=.25, one_owned_worker_at_a_time=True, automatic_retry=False,
        images_read=0, benchmark_inputs_read=0, forwards=0, scores_read=0, model_fitting=False,
        GPU_initialized=False, original_sources_or_recipes_changed=False, stages=[], worker_processes=[],
        GPU_metric_probe_pending=True, native_GPU_evaluation_or_precision_verified=False)
    A.write(completion, status)
    active = None
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout(_signum, _frame):
        raise TimeoutError('One finite1800-second CPU original-model probe; retain partial evidence')
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS - 20)
    try:
        rows, pins, packages = source_checks()
        status.update(asset_readiness=dict(path=str(ASSET_RECEIPT), sha256=ASSET_RECEIPT_SHA),
            source_and_metadata_sha256=pins, package_files_present=packages, memory_before=memory())
        minimum = status['memory_before']['effective_headroom_bytes']
        for stage in STAGES:
            assert time.monotonic() < deadline
            asset = rows[stage]; target = Path(asset['durable_path'])
            verified = A.hash_and_advise(target, asset)
            # Conservative transient construction margins, separate from the
            # unchanged formal GPU process's own8GiB admission threshold.
            reserve = (6 if stage == 'CLIP_L14' else 2) << 30
            before = memory(reserve); minimum = min(minimum, before['effective_headroom_bytes'])
            record_path = output / (stage + '.json'); log_path = output / (stage + '.log')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            command = [sys.executable, '-B', str(Path(__file__)), '--worker', stage,
                '--worker-output', str(record_path), '--parent-pid', str(os.getpid()), '--attempt', str(args.attempt)]
            with log_path.open('xb') as log:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=False)
                child_row = dict(stage=stage, pid=active.pid, command=command, memory_before=before,
                    transient_preflight_margin_bytes=reserve, finished=False)
                status['worker_processes'].append(child_row); A.write(completion, status)
                try:
                    while active.poll() is None:
                        assert time.monotonic() < deadline
                        observation = memory(); minimum = min(minimum, observation['effective_headroom_bytes'])
                        output_bytes(output)
                        time.sleep(.25)
                    code = active.wait(); child_row.update(exit_code=code, finished=True)
                finally:
                    if active.poll() is None:
                        child_row.update(exit_code=finish_child(active), finished=True, stopped_by_coordinator=True)
                    # It is now safe to release the worker's durable file cache.
                    release_payload(target)
                active = None
            assert code == 0, ('CPU strict-load child failed; no automatic retry', stage, code)
            value = json.loads(record_path.read_text()); validate_model_record(value, stage)
            assert value['asset'] == dict(path=str(target), sha256=asset['sha256'], bytes=asset['bytes'])
            assert value['source_and_metadata_sha256'] == pins
            after = A.hash_and_advise(target, asset)
            assert all(verified[key] == after[key] for key in ('device', 'inode', 'mtime_ns', 'bytes', 'sha256'))
            status['stages'].append(dict(stage=stage, record=dict(path=str(record_path), sha256=A.C.sha(record_path)),
                log=dict(path=str(log_path), sha256=A.C.sha(log_path)), asset=after, process_exited=True))
            status.update(memory_latest=memory(), observed_minimum_headroom_bytes=minimum)
            output_bytes(output); A.write(completion, status)
        assert tuple(row['stage'] for row in status['stages']) == STAGES
        assert all(row['finished'] and row['exit_code'] == 0 for row in status['worker_processes'])
        assert all(A.C.sha(path) == digest for path, digest in pins.items())
        assert A.C.sha(__file__) == status['source_sha256'] and 'torch' not in sys.modules
        status.update(status='ALL5_ORIGINAL_METRIC_MODELS_STRICT_LOAD_VERIFIED_CPU_GPU_PROBE_PENDING',
            CPU_strict_model_load_pending=False, GPU_model_probe_pending=True,
            full_image_evaluation_verified=False, observed_minimum_headroom_bytes=minimum,
            memory_after=memory(), seconds=time.monotonic() - started, output_bytes_before_completion=output_bytes(output))
        A.write(completion, status); output_bytes(output)
        print(json.dumps(dict(status=status['status'], receipt=str(completion), sha256=A.C.sha(completion))), flush=True)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_after_failure(status, active)
        status.update(status='ORIGINAL_METRIC_CPU_MODEL_PROBE_FAILED_PARTIAL_EVIDENCE_RETAINED',
            error_type=type(error).__name__, error=str(error)[:600], seconds=time.monotonic() - started)
        A.write(completion, status)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    main()
