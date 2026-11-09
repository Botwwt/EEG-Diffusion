"""Explicit recovery of the unchanged v2 probe with isolated meta-weight loading.

No v2 source, learned weight, diagnostic threshold, sample, or precision changes.
Accelerate only activates Diffusers' existing empty-initialization loader branch.
Every converted tensor loaded by that branch is compared byte-for-byte.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_frozen_function_memory_recovery_20261025_v3'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    assert Path(path).stat().st_size < 32 << 20
    return json.loads(Path(path).read_bytes())


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def bound(item):
    assert sha(item['path']) == item['sha256'], item['path']
    return read(item['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def memory():
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            result[line.split(':')[0] + '_bytes'] = int(line.split()[1]) * 1024
    assert set(result) == {'VmRSS_bytes', 'VmHWM_bytes'}
    return result


def verify_config(path):
    config = read(path)
    assert config['schema'] == SCHEMA and config['automatic_new_fit'] is False
    assert config['source_sha256'][Path(__file__).name] == sha(__file__)
    for name, digest in config['source_sha256'].items():
        assert Path(name).name == name and sha(ROOT / name) == digest, name
    runtime = bound(config['isolated_runtime_manifest'])
    runtime_path = Path(runtime['path'])
    assert runtime_path.is_absolute() and 'analysis_runtime' in runtime_path.parts
    assert runtime['wheel_sha256'] == 'd69159e2c4e4a473d14443b27d2d732929254e826b3ab4813b3785b5ac616c7c'
    files = {p.relative_to(runtime_path).as_posix(): sha(p) for p in runtime_path.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert files == runtime['files'], 'Isolated loader dependency changed'
    diagnostic = bound(config['diagnostic_registration'])
    # The failed 10GiB constraint remains exactly in force.
    assert diagnostic['maximum_extra_CPU_bytes'] == 10 << 30
    assert diagnostic['maximum_GPU_bytes'] == 20 << 30
    assert diagnostic['batch_size'] == 2 and diagnostic['maximum_seconds'] == 7200
    failure = bound(config['previous_failure'])
    assert failure['observed_condition_cells'] == 0 and failure['completed_stages'] == []
    assert failure['optimizer_created'] is False and failure['parameter_updates'] == 0
    job = bound(config['previous_GPU_job'])
    assert job['exit_code'] == 1 and job['status'] == 'FAILED_OR_JOB_TIMEOUT_RESUMABLE'
    queue = bound(config['previous_analysis_queue'])
    assert queue['status'] == 'ANALYSIS_FAILED_PRESERVED_NO_AUTOMATIC_RETRY'
    assert [item['stage'] for item in queue['completed']] == ['logsCPU', 'layersCPU']
    main = bound(config['upstream_closed_queue'])
    assert main['status'] == 'ALL_FOUR_FULL_LIFECYCLES_COMPLETE_AFTER_DOCUMENTED_SINGLE_O_RESTART'
    assert [item['arm'] for item in main['completed']] == ['O', 'P', 'PN', 'S']
    return config, runtime, diagnostic


class ConvertedWeightAudit:
    def __init__(self, torch, directory):
        self.torch, self.directory = torch, Path(directory)
        self.records, self.originals = [], []

    def inspect(self, model, checkpoint, original, args, kwargs):
        torch = self.torch
        before = memory()
        assert all(p.is_meta for p in model.parameters()), 'Empty-weight branch was not used'
        result = original(model, checkpoint, *args, **kwargs)
        assert not result, ('Unexpected converted weights', result)
        actual = model.state_dict()
        assert set(actual) == set(checkpoint), (type(model).__name__,
            sorted(set(actual) - set(checkpoint)), sorted(set(checkpoint) - set(actual)))
        expected_hash, actual_hash = hashlib.sha256(), hashlib.sha256()
        count = total = 0
        for name in sorted(actual):
            expected = checkpoint[name].detach().to(dtype=actual[name].dtype, device='cpu').contiguous()
            value = actual[name].detach().cpu().contiguous()
            assert value.shape == expected.shape and torch.equal(value, expected), name
            for digest, tensor in ((expected_hash, expected), (actual_hash, value)):
                digest.update(name.encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
            count += 1
            total += value.numel()
            del expected, value
        assert actual_hash.hexdigest() == expected_hash.hexdigest()
        record = dict(component=type(model).__name__, tensor_count=count, elements=total,
            expected_converted_state_sha256=expected_hash.hexdigest(),
            actual_loaded_state_sha256=actual_hash.hexdigest(), all_tensors_exact=True,
            before=before, after=memory())
        self.records.append(record)
        write(self.directory / ('component_' + str(len(self.records)) + '.json'), record)
        print(json.dumps(dict(stage='EXACT_META_LOADER_COMPONENT', **record)), flush=True)
        return result

    def install(self):
        from diffusers.loaders import single_file_model, single_file_utils
        for module in (single_file_model, single_file_utils):
            original = module.load_model_dict_into_meta
            def observed(model, checkpoint, *args, _original=original, **kwargs):
                return self.inspect(model, checkpoint, _original, args, kwargs)
            self.originals.append((module, original))
            module.load_model_dict_into_meta = observed

    def restore(self):
        for module, original in self.originals:
            module.load_model_dict_into_meta = original

    def complete(self):
        classes = sorted(item['component'] for item in self.records)
        assert classes == ['AutoencoderKL', 'CLIPTextModel', 'CLIPTextModelWithProjection', 'UNet2DConditionModel'], classes
        return dict(status='ALL_FOUR_NATIVE_COMPONENTS_EXACT_CONVERTED_TENSOR_LOAD_VERIFIED',
                    components=self.records, loading_uses_existing_official_conversion=True,
                    frozen_FP16_weights_changed=False, metadata_only_observer=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--profile-load', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config, runtime, document = verify_config(args.registration)
    # Must precede importing Diffusers: its availability flag is cached at import.
    assert 'diffusers' not in sys.modules and 'transformers' not in sys.modules
    sys.path.insert(0, runtime['path'])
    import brainhive_native_models_v1
    import accelerate
    import torch
    from diffusers.utils import is_accelerate_available
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    assert accelerate.__version__ == '0.34.2' and is_accelerate_available()
    D.validate_registration(document)
    if args.preflight_only:
        D.inspect_finals(document)
        assert not torch.cuda.is_initialized()
        print('LOW_MEMORY_RECOVERY_CPU_PREFLIGHT_PASSED_NO_GPU')
        return
    import fcntl
    own = ROOT / 'audit/diagnostic_memory_recovery_20261025_v3'
    own.mkdir(parents=True, exist_ok=True)
    with (own / 'recovery_stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovery_output = Path(config['profile_output'] if args.profile_load else config['recovery_output'])
        recovery_output.mkdir(parents=True, exist_ok=False)
        audit = ConvertedWeightAudit(torch, recovery_output)
        audit.install()
        started = time.monotonic()
        try:
            if args.profile_load:
                # A finite engineering load under exactly the same CPU/GPU caps.
                profile_document = dict(document, gpu_job_name=config['profile_job_name'])
                D.torch = torch
                D.verify_wrapper(profile_document)
                budget = D.Budget(profile_document)
                budget.check()
                torch.set_num_threads(2)
                torch.manual_seed(1729)
                from brainhive_component_generator_v1 import load_native
                pipe, adapter, projector, prompts, host = load_native(training=False)
                budget.check()
                proof = audit.complete()
                write(recovery_output / 'profile.json', dict(status='EXACT_LOW_MEMORY_NATIVE_LOAD_PROFILE_PASSED',
                    source=ref(__file__), registration=ref(args.registration), runtime_manifest=config['isolated_runtime_manifest'],
                    proof=proof, elapsed_seconds=time.monotonic() - started,
                    peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes, peak_GPU_bytes=budget.peak_GPU_bytes,
                    CPU_cap_bytes=10 << 30, GPU_cap_bytes=20 << 30, host=host,
                    generated_images=0, parameter_updates=0, actual_functional_diagnostic_executed=False))
                del pipe, adapter, projector, prompts
                gc.collect()
                print('EXACT_LOW_MEMORY_NATIVE_LOAD_PROFILE_PASSED', flush=True)
            else:
                previous = bound(config['profile_success'])
                assert previous['status'] == 'EXACT_LOW_MEMORY_NATIVE_LOAD_PROFILE_PASSED'
                assert previous['peak_extra_CPU_bytes'] <= 10 << 30
                D.execute(document, Path(config['diagnostic_registration']['path']))
                proof = audit.complete()
                write(recovery_output / 'completion.json', dict(status='UNCHANGED_V2_FROZEN_FUNCTION_RECOVERED_WITH_EXACT_META_LOADING',
                    proof=proof, source=ref(__file__), registration=ref(args.registration),
                    diagnostic_completion=ref(Path(document['output_directory']) / 'completion.json'),
                    runtime_manifest=config['isolated_runtime_manifest'], thresholds_changed=False,
                    queries_changed=False, precision_changed=False, parameter_updates=0))
        except BaseException as error:
            write(recovery_output / 'failure.json', dict(status='LOW_MEMORY_RECOVERY_FAILED_PRESERVED',
                error_type=type(error).__name__, error=str(error), memory=memory(),
                elapsed_seconds=time.monotonic() - started, component_proofs=audit.records,
                thresholds_changed=False, parameter_updates=0))
            raise
        finally:
            audit.restore()


if __name__ == '__main__':
    main()
