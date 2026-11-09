"""Bounded evaluator load/finite probe on one predetermined original T image.

Runs the four unchanged feature routes. No pair scores, fitted parameters,
generated images, captions, boxes, or feature tensors are saved. This is an
environment check and cannot establish full-manifest evaluation readiness.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SERVER_ROOT = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
ORIGINAL_IMAGES = Path('/root/autodl-tmp/avde_single_task_data_20261016/image_set/training_images')
DINO_PATH = Path('/root/mced_brain_dinov3_s_v1/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
DINO_SHA = '08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'
PINS = {
    'image_evaluation.py': 'e14172655a156040b625694bad1ffbd441ad0b648c876e9a6b150d03e8d27f37',
    'semantic_evaluator.py': 'a57e9624102f0856634866c507e272166cf96ad344c3ad0d628bb20cded3e8eb',
    'locked_metrics.py': 'c882fe6a2158cf04b320d22280e289c7e53945a989f60a2c08d27662272d4393',
    'extract_image_features.py': '3b24d22c4ac62f23efa8d4cf80bef148b0d075775ba54c6be46699d3bccd32ef',
    'run_gpu_job_v2.py': 'a6e4b652cd58134a1b9c260b084392955b06ed506ae63bbcdc9c75b55e8c8114',
    'configs/split_manifest.json': '1969d664555d08399e77a2d76c011f19eadbda1ce251f9c72d72c94de3910ad2',
    'configs/BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json': '5019b618f570d9454be0744ffeb5985a939227a77835f7f990c4f755b45a29e5',
    'audit/image_manifest.csv': '91e75aa52fbb0addc004eeb38a87aa237f48e0d6bdbd5f5cc3d0d0e839d1b145',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    assert len(raw) <= 64 << 10, 'Environment probe metadata cap'
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def selected_T(protocol, registration, rows, base=ORIGINAL_IMAGES):
    """Identity check independent of torch and all result/score tables."""
    first = registration['rows'][0]
    assert first['array_row'] == 9024 and first['stimulus_id'] == 'moose_05s.jpg'
    assert first['array_row'] in protocol['image_rows']['T']
    assert first['array_row'] not in protocol['image_rows']['D'] + protocol['image_rows']['V']
    selected = [r for r in rows if int(r['array_row']) == 9024 and r['dataset_split'] == 'train']
    assert len(selected) == 1, 'Original training stimulus must join exactly once'
    row = selected[0]
    assert row['stimulus_id'] == first['stimulus_id']
    member = PurePosixPath(row['image_member'])
    assert not member.is_absolute() and '..' not in member.parts
    relative = member.relative_to('training_images')
    assert relative.name == first['stimulus_id']
    path = base / relative
    assert path.resolve().is_relative_to(base.resolve()), 'Original image remains in its input tree'
    return path, row


def require_asset_records(expected, records):
    assert len(expected) == len(records) == 25
    assert len({r['path'] for r in expected}) == 25
    assert len({r['path'] for r in records}) == 25
    got = {r['path']: r for r in records}
    for item in expected:
        record = got[item['path']]
        assert record['sha256'] == item['sha256']
        assert record['bytes'] == item['bytes']


def require_job_record(job, attempt, pid, parent_pid):
    assert job['name'] == f'BrainHIVE_evaluation_environment_probe_20261022_v2_attempt{attempt}'
    assert job['status'] == 'RUNNING'
    assert job['pid'] == pid and job['wrapper_pid'] == parent_pid
    assert job['max_seconds'] == 900


def wrapper_guard(attempt):
    import fcntl
    name = f'BrainHIVE_evaluation_environment_probe_20261022_v2_attempt{attempt}'
    path = ROOT / 'audit' / (name + '_job.json')
    assert b'run_gpu_job_v2.py' in (Path('/proc') / str(os.getppid()) / 'cmdline').read_bytes()
    job = None
    for _ in range(100):
        try:
            job = json.loads(path.read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.05)
    assert job is not None, 'Wrapper must publish its live job receipt'
    require_job_record(job, attempt, os.getpid(), os.getppid())
    authorization = Path(job['authorization'])
    assert sha(authorization) == job['authorization_sha256']
    with (ROOT / 'audit/GPU_JOB.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
            raise AssertionError('GPU wrapper must own its exclusive lock')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    assert not processes.strip(), 'Preserve an existing GPU computation'
    return path


def assert_frozen(model):
    parameters = list(model.parameters())
    assert parameters and not model.training
    assert all(not p.requires_grad and p.grad is None for p in parameters)
    return sum(p.numel() for p in parameters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-receipt', required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT == SERVER_ROOT
    assert 1 <= args.attempt <= 3
    output = ROOT / 'audit' / f'new_instance_evaluation_environment_probe_20261022_v2_attempt{args.attempt}.json'
    readiness = ROOT / 'audit' / f'new_instance_evaluation_readiness_20261022_v2_attempt{args.attempt}.json'
    assert not output.exists() and not output.with_name(output.name + '.pending').exists()
    assert not readiness.exists() and not readiness.with_name(readiness.name + '.pending').exists()
    started = time.monotonic()
    initial = dict(status='EVALUATION_ENVIRONMENT_PROBE_PREFLIGHT',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(),
        maximum_seconds=900, maximum_receipt_bytes=64 << 10, source_sha256=sha(__file__),
        training_seed=1729, predetermined_T_array_row=9024, predetermined_stimulus_id='moose_05s.jpg',
        optimizer_created=False, trained_parameters=0, new_generated_images=0,
        paired_metrics_computed=0, score_tables_read=0, feature_tensors_saved=0,
        metric_selection_or_tuning=False,
        full_manifest_evaluation_verified=False, formal_training_admitted=False, actual_GPU=False, stages=[])
    write(output, initial)
    try:
        job = wrapper_guard(args.attempt)
        for name, digest in PINS.items():
            assert sha(ROOT / name) == digest, name
        import restore_BrainHIVE_evaluation_runtime_CPU_20261022_v1 as recovery
        manifests_path = ROOT / 'audit/seed_hf_selected_assets.json'
        archive_path = ROOT / 'seed_small_configs.zip'
        direct_path = ROOT / 'audit/seed_direct_assets_receipt.json'
        small, large = recovery.inspect_small(json.loads(manifests_path.read_text()), archive_path)
        direct = json.loads(direct_path.read_text())
        assert direct['status'] == 'OFFICIAL_PUBLIC_EVALUATOR_ASSETS_VERIFIED'
        assert len(direct['assets']) == 2
        expected = small + large + direct['assets']
        asset_receipt = Path(args.asset_receipt)
        assert asset_receipt.resolve().is_relative_to(ROOT / 'audit')
        restored = json.loads(asset_receipt.read_text())
        assert restored['status'] == 'ALL25_ORIGINAL_PUBLIC_EVALUATOR_ASSETS_EXACT_BYTES_RESTORED'
        require_asset_records(expected, restored['assets'])
        for asset in expected:
            assert sha(asset['path']) == asset['sha256'], asset['path']
        assert sha(DINO_PATH) == DINO_SHA
        protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
        queries = json.loads((ROOT / 'configs/BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json').read_text())
        with (ROOT / 'audit/image_manifest.csv').open() as stream:
            path, row = selected_T(protocol, queries, list(csv.DictReader(stream)))
        assert sha(path) == row['image_sha256']
        inputs = {str(ROOT / n): d for n, d in PINS.items()}
        inputs.update({str(p): sha(p) for p in [asset_receipt, manifests_path, archive_path, direct_path,
            ROOT / 'audit/seed_hf_acquisition_completion.json', DINO_PATH, path]})
        initial.update(status='EVALUATION_ENVIRONMENT_PROBE_RUNNING', wrapper_receipt=str(job),
            input_sha256=inputs, original_T_image=dict(path=str(path), sha256=row['image_sha256'],
            array_row=9024, stimulus_id=row['stimulus_id']),
            preprocessing='Original evaluator routes unchanged; original source image; no added resize/crop or image output')
        write(output, initial)
        import image_evaluation as evaluator
        torch = evaluator.torch
        import numpy as np
        torch.set_num_threads(2); torch.manual_seed(1729); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        initial['actual_GPU'] = True
        torch.cuda.reset_peak_memory_stats()
        stages = initial['stages']
        for stage in ['DINO', 'EffNet', 'CapSim', 'ObjectF1']:
            assert time.monotonic() - started < 840
            tick = time.monotonic()
            with torch.inference_mode():
                if stage == 'DINO':
                    model = evaluator.dino_model()
                    count = assert_frozen(model)
                    values = evaluator.dino_features([path], model, real_images=True)
                    assert values.ndim == 2 and values.shape[0] == 1 and torch.isfinite(values).all()
                    detail = dict(feature_shape=list(values.shape), values_finite=True)
                    del values
                elif stage == 'EffNet':
                    model, preprocess = evaluator.effnet_model()
                    count = assert_frozen(model)
                    values = evaluator.effnet_features([path], model, preprocess, batch_size=1)
                    assert values.ndim == 2 and values.shape[0] == 1 and np.isfinite(values).all()
                    detail = dict(feature_shape=list(values.shape), values_finite=True)
                    del values, preprocess
                elif stage == 'CapSim':
                    processor, model, text, info = evaluator.caption_models()
                    count = assert_frozen(model) + assert_frozen(text)
                    captions, values = evaluator.caption_features([path], processor, model, text, batch_size=1)
                    assert len(captions) == 1 and isinstance(captions[0], str)
                    assert values.ndim == 2 and values.shape[0] == 1 and np.isfinite(values).all()
                    detail = dict(feature_shape=list(values.shape), values_finite=True,
                        text_loading_discrepancies_absent=not any(info.get(k) for k in
                        ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']),
                        captions_saved=False)
                    assert detail['text_loading_discrepancies_absent']
                    del captions, values, processor, text, info
                else:
                    model, classes, cfg = evaluator.detection_model()
                    count = assert_frozen(model.model)
                    values = evaluator.detection_features([path], model, classes, batch_size=1)
                    assert len(values) == 1 and len(classes) == 82
                    found = values[0]
                    assert len(found['labels']) == len(found['scores']) == len(found['bboxes'])
                    assert np.isfinite(np.asarray(found['scores'])).all()
                    assert np.isfinite(np.asarray(found['bboxes'])).all()
                    detail = dict(predictions_finite=True, classes=82, predictions_saved=False)
                    del values, found, cfg, classes
                assert_frozen(model.model if stage == 'ObjectF1' else model)
            torch.cuda.synchronize()
            stages.append(dict(stage=stage, original_route_completed=True, frozen_parameters=count,
                optimizer_created=False, parameter_gradients_absent=True, seconds=time.monotonic()-tick, **detail))
            del model
            gc.collect(); torch.cuda.empty_cache()
            write(output, initial)
            print(json.dumps(dict(stage=stage, status='ORIGINAL_EVALUATOR_ROUTE_LOAD_AND_FINITE_CHECK_PASSED')), flush=True)
        assert [s['stage'] for s in stages] == ['DINO', 'EffNet', 'CapSim', 'ObjectF1']
        assert sha(path) == row['image_sha256']
        assert all(sha(p) == digest for p, digest in inputs.items())
        assert all(sha(asset['path']) == asset['sha256'] for asset in expected)
        assert sha(__file__) == initial['source_sha256']
        initial.update(status='FROZEN_FOUR_METRIC_MODELS_RUNTIME_PROBE_COMPLETE',
            stage_records=list(stages), stages=['DINO', 'EffNet', 'CapSim', 'ObjectF1'],
            end_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seconds=time.monotonic()-started,
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(), GPU_name=torch.cuda.get_device_name(0),
            original_input_and_asset_and_scientific_source_SHA_unchanged=True,
            full_manifest_evaluation_verified=False, formal_training_admitted=False)
        write(output, initial)
        write(readiness, dict(status='FROZEN_EVALUATION_ASSETS_READY',
            UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            metric_selection_or_tuning=False,
            source_sha256={name: PINS[name] for name in ['extract_image_features.py',
                'image_evaluation.py', 'locked_metrics.py', 'semantic_evaluator.py']},
            assets=[dict(path=item['path'], sha256=item['sha256']) for item in expected] +
                [dict(path=str(DINO_PATH), sha256=DINO_SHA)],
            actual_GPU_probe=dict(path=str(output), sha256=sha(output)),
            full_manifest_evaluation_verified=False, formal_training_admitted=False,
            note='All four original evaluator routes load and give finite output on one fixed T input; full gallery and formal manifests require separate verification.'))
        print(json.dumps(dict(status=initial['status'], receipt=str(output), readiness=str(readiness))), flush=True)
    except Exception as error:
        initial.update(status='EVALUATION_ENVIRONMENT_PROBE_FAILED_PRIOR_RESULTS_UNCHANGED',
            error_type=type(error).__name__, error=str(error)[:600], seconds=time.monotonic()-started)
        write(output, initial)
        raise


if __name__ == '__main__':
    main()
