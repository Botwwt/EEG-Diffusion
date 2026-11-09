"""One fixed existing T query, three actual native image paths, no training.

Parent / zero-head / explicitly-disabled are independently sampled with the
same fixed query seed. Engineering source is not a completed pixel replay.
No automatic queue, formal-candidate admission, teacher or quality score exists
in this probe. A future reviewed, hash-complete registration is required.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from brainhive_functional_training_contract_v2 import (
    ROOT, REPO, WORK, FIXED_INPUTS, MAIN, DIAG, MAIN_TERMINAL, DIAG_TERMINAL,
    sha, read, query_identity, resource_headroom, private_path,
)
from diagnose_BrainHIVE_native_function_T32_v2 import select_fixed_queries


NAME = 'functional_T_pixel_replay_v2'
OUT = WORK / 'audit' / NAME
CASES = ('parent', 'zero_head', 'explicitly_disabled')
T32_REGISTRATION = str(REPO / 'configs/BrainHIVE_native_function_T32_v2.json')
REQUIRED_SOURCES = (
    'diagnose_BrainHIVE_functional_T_pixel_replay_v2.py',
    'brainhive_functional_training_contract_v2.py', 'brainhive_native_models_v1.py',
    'brainhive_bounded_component_v1.py', 'brainhive_component_generator_v1.py',
    'generate_BrainHIVE_functional_alignment_V_v2.py',
    'diagnose_BrainHIVE_native_function_T32_v2.py', 'run_gpu_job_v2.py',
)
# Only the native loader's assets; no F feature cache, noisy image latents,
# V generation metadata or official test tensors are required/read here.
NATIVE_INPUTS = tuple(path for path in FIXED_INPUTS if any(part in path for part in (
    '/reproduction_sources/BrainHIVE_f06db69f/', '/assets/prior_H14/', '/sdxl_config/',
    '/sdxl_turbo_fp16.safetensors', '/audit/native_asset_verification.json',
    '/configs/ATM_public_decoder_registration_20261020_v1.json',
    '/configs/resource_authorization_20261020_v3.json')))
CONTROLLERS = (
    str(REPO / 'queue_BrainHIVE_bounded_full_lifecycle_v1.py'),
    str(REPO / 'queue_BrainHIVE_main_storage_continuation_20261021_v1.py'),
    str(REPO / 'queue_BrainHIVE_native_function_T32_v2.py'),
)


def write_new(path, value):
    path = Path(path)
    assert not path.exists()
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_config(config, *, check_files=True):
    fixed = dict(schema='BrainHIVE_functional_T_pixel_replay_v2_registration',
        status='REVIEWED_ENGINEERING_PROBE_READY', diagnostic=NAME,
        seed=1729, fixed_T32_query_index=0, cases=list(CASES),
        T32_registration_path=T32_REGISTRATION, output_directory=str(OUT),
        generator=dict(resolution=512, steps=4, CFG=0, IP_scale=1),
        formal_training_enabled=False, optimizer_created=False,
        visual_targets_or_scores_enabled=False, automatic_launch_enabled=False)
    for key, expected in fixed.items():
        assert config[key] == expected, (key, config.get(key), expected)
    assert Path(config['job_name']).name == config['job_name']
    assert 90 < config['maximum_seconds'] <= 1800
    assert 4 << 20 <= config['maximum_probe_bytes'] <= 64 << 20
    assert config['minimum_CPU_headroom_bytes'] >= 2 << 30
    assert config['minimum_free_GPU_bytes'] > 0
    assert config['minimum_retained_disk_bytes'] >= 64 << 20
    assert set(REQUIRED_SOURCES) <= set(config['source_sha256'])
    assert set(NATIVE_INPUTS) | {T32_REGISTRATION} <= set(config['input_sha256'])
    if check_files:
        for name, expected in config['source_sha256'].items():
            assert Path(name).name == name and sha(ROOT / name) == expected
        for path, expected in config['input_sha256'].items():
            assert Path(path).is_absolute() and sha(path) == expected, path
        registration = read(str(REPO / 'configs/ATM_public_decoder_registration_20261020_v1.json'))
        from brainhive_functional_training_contract_v2 import GENERATOR_ASSET_ROOT, CONFIG_FILES
        assert registration['generator_asset_root'] == str(GENERATOR_ASSET_ROOT)
        selected = [item for item in registration['generator_assets'] if 'sdxl_turbo_fp16' in item['path']]
        assert len(selected) == 1 and config['input_sha256'][selected[0]['path']] == selected[0]['sha256']
        tree = Path(str(GENERATOR_ASSET_ROOT / 'sdxl_config'))
        actual = {str(path) for path in tree.rglob('*') if path.is_file()}
        assert {str(GENERATOR_ASSET_ROOT / 'sdxl_config' / name) for name in CONFIG_FILES} <= actual <= set(config['input_sha256'])
    return True


def reject_active_terminal(state, *, expected_states, snapshot):
    """Pure check; snapshots come from actual /proc, never from a RUNNING label."""
    assert state['status'] in expected_states, 'Original controller has not reached a terminal state'
    assert int(state['pid']) > 0
    assert not snapshot['live'], ('Controller PID remains active or has been reused; preserve it', snapshot)
    return dict(pid=state['pid'], status=state['status'], actual_proc_snapshot=snapshot)


def proc_snapshot(pid):
    folder = Path('/proc') / str(int(pid))
    try:
        stat_text = (folder / 'stat').read_text()
        fields = stat_text[stat_text.rfind(')') + 2:].split()
        command = (folder / 'cmdline').read_bytes()
        return dict(exists=True, state=fields[0], live=fields[0] not in ('Z', 'X'),
            start_ticks=fields[19], cmdline_sha256=hashlib.sha256(command).hexdigest(),
            command=command.decode(errors='replace').replace('\0', ' '))
    except FileNotFoundError:
        return dict(exists=False, live=False, state=None, cmdline_sha256=None, command=None)


def terminal_preflight(config):
    bindings = config['terminal_state_bindings']
    assert {item['path'] for item in bindings} == {str(MAIN), str(DIAG)} and len(bindings) == 2
    evidence = []
    for item in bindings:
        assert sha(item['path']) == item['sha256']
        state = read(item['path'])
        expected = MAIN_TERMINAL if item['path'] == str(MAIN) else DIAG_TERMINAL
        evidence.append(reject_active_terminal(state, expected_states=expected, snapshot=proc_snapshot(state['pid'])))
    # A stale terminal receipt must not hide another live instance of either
    # actual controller script. No process is killed or altered by this check.
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            command = (folder / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(script.encode() in command for script in CONTROLLERS):
            snapshot = proc_snapshot(int(folder.name))
            assert not snapshot['live'], ('An original controller is still active', snapshot)
    return evidence


def fixed_input_binding(config):
    """Metadata/hashes only; the eventual tensor row comes from this binding."""
    registration = read(config['T32_registration_path'])
    assert registration['subject'] == 'sub-01' and registration['queries'] == 32 and registration['seed'] == 1729
    input_path = registration['oracle_inference_inputs']
    def registered_path(suffix):
        paths = [path for path in registration['input_sha256'] if path.endswith(suffix)]
        assert len(paths) == 1
        return paths[0]
    protocol_path = registered_path('/configs/split_manifest.json')
    manifest_path = registered_path('/audit/image_manifest.csv')
    contract_path = registered_path('/data/component_input_contract_v1.json')
    for path in (input_path, protocol_path, manifest_path, contract_path):
        assert config['input_sha256'][path] == registration['input_sha256'][path] == sha(path)
    protocol = read(protocol_path)
    assert sha(manifest_path) == protocol['image_manifest_sha256']
    with Path(manifest_path).open() as stream:
        manifest = {int(row['array_row']): row for row in csv.DictReader(stream) if row['dataset_split'] == 'train'}
    selected = select_fixed_queries(read(input_path), protocol['image_rows']['T'], manifest)
    query = selected[0]
    assert query['oracle_query_index'] == config['fixed_T32_query_index'] == 0
    assert query['array_row'] not in protocol['image_rows']['D'] + protocol['image_rows']['V']
    contract = read(contract_path)
    subject = registration['subject']
    person = next(item for item in contract['subjects'] if item['subject'] == subject)
    matches = []
    for source in person['sources']:
        if source['role'].startswith('oof'):
            entries = [entry for entry in source['groups']['component_train'] if entry['array_row'] == query['array_row']]
            if entries:
                assert len(entries) == 1
                matches.append((source, entries[0]['cache_row']))
    assert len(matches) == 1, 'The fixed T query must have exactly one actual OOF source'
    source, cache_row = matches[0]
    for path_key, hash_key in (('condition_cache_path', 'condition_cache_sha256'),
                              ('checkpoint_path', 'checkpoint_sha256'),
                              ('parent_registration_path', 'parent_registration_sha256')):
        assert config['input_sha256'][source[path_key]] == source[hash_key] == sha(source[path_key])
    parent_registration = read(source['parent_registration_path'])
    trained_concepts = {manifest[row]['concept_id'] for row in parent_registration['training_array_rows']}
    assert query['concept_id'] not in trained_concepts
    identity, image_seed = query_identity(subject, query, protocol['version'])
    return dict(subject=subject, query=query, query_id=identity, image_seed=image_seed,
        source=source, cache_row=cache_row, protocol_version=protocol['version'],
        T32_registration_sha256=sha(config['T32_registration_path']),
        oracle_metadata_path=input_path, oracle_metadata_sha256=sha(input_path),
        condition_contract_path=contract_path, condition_contract_sha256=sha(contract_path),
        raw_D_V_or_test_tensors_read=False, target_images_or_visual_feature_values_read=False)


def independent_preflight(config):
    # Reject current live main/T32 before hashing multi-GB native assets.
    terminals = terminal_preflight(config)
    validate_config(config)
    out = private_path(config['output_directory'])
    assert not out.exists(), 'A previous attempt exists; no automatic overwrite or retry'
    memory = resource_headroom()
    assert memory['effective_headroom_bytes'] >= config['minimum_CPU_headroom_bytes']
    assert shutil.disk_usage(str(WORK)).free >= config['maximum_probe_bytes'] + config['minimum_retained_disk_bytes']
    binding = fixed_input_binding(config)
    return dict(terminals=terminals, memory=memory, binding=binding,
        complete_four_arm_storage_required=False, positive_functional_effect_required=False)


def exclusive_wrapper(config):
    parent = Path('/proc') / str(os.getppid())
    command = (parent / 'cmdline').read_bytes().split(b'\0')
    assert str(ROOT / 'run_gpu_job_v2.py').encode() in command and b'--name' in command
    assert command[command.index(b'--name') + 1].decode() == config['job_name']
    lock = ROOT / 'audit/GPU_JOB.lock'
    expected = lock.stat()
    descriptors = []
    for fd in (parent / 'fd').iterdir():
        try:
            current = fd.stat()
            descriptors.append((current.st_dev, current.st_ino))
        except FileNotFoundError:
            pass
    assert (expected.st_dev, expected.st_ino) in descriptors
    import fcntl
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
            raise AssertionError('The trusted wrapper must already hold the exclusive GPU lock')
    job = read(ROOT / 'audit' / (config['job_name'] + '_job.json'))
    assert job['status'] == 'RUNNING' and job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid()
    assert 0 < job['max_seconds'] <= config['maximum_seconds']
    auth_path = str(REPO / 'configs/resource_authorization_20261020_v3.json')
    assert job['authorization_sha256'] == config['input_sha256'][auth_path]
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          check=True, capture_output=True, text=True, timeout=10)
    assert {int(value.strip()) for value in apps.stdout.splitlines() if value.strip()} <= {os.getpid()}
    capacity = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                              check=True, capture_output=True, text=True, timeout=10)
    free = [int(value.strip()) * (1 << 20) for value in capacity.stdout.splitlines() if value.strip()]
    assert len(free) == 1 and free[0] >= config['minimum_free_GPU_bytes']
    return dict(wrapper_pid=os.getppid(), job_name=config['job_name'], free_GPU_bytes=free[0])


def run_probe(config, evidence):
    started = time.monotonic()
    import torch
    import numpy as np
    from PIL import Image
    from brainhive_component_generator_v1 import load_native
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    binding = evidence['binding']
    source = binding['source']
    exported = torch.load(source['condition_cache_path'], weights_only=True, map_location='cpu', mmap=True)
    position = binding['cache_row']
    assert int(exported['array_rows'][position]) == binding['query']['array_row']
    assert exported['model_sha256'] == source['checkpoint_sha256']
    condition = exported['condition'][position:position + 1].clone()
    assert condition.shape == (1, 1024) and torch.isfinite(condition).all()
    model_state = torch.load(source['checkpoint_path'], weights_only=True, map_location='cpu')['model']
    weight = model_state['block.layers.0.norm.weight'][None].cuda()
    bias = model_state['block.layers.0.norm.bias'][None].cuda()
    condition = condition.cuda()
    pipe, adapter, projector, prompts, host = load_native(training=False)
    zero_head = BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
    disabled_head = BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
    assert sum(parameter.numel() for parameter in zero_head.parameters()) == 525568
    with torch.no_grad():
        disabled_head.direction.weight.fill_(.01)
        disabled_head.direction.bias.copy_(torch.linspace(-10, 10, 1024, device='cuda'))
        enabled_probe = disabled_head(condition, weight, bias, gate_mode='fixed')[0]
        assert not torch.equal(enabled_probe, condition), 'Disabled path must exercise an otherwise active head'
    out = Path(config['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    images = out / 'images'
    images.mkdir()
    write_new(out / 'input_provenance.json', dict(evidence=evidence, source_sha256=config['source_sha256'],
        input_sha256=config['input_sha256'], host=host, query_id=binding['query_id'], image_seed=binding['image_seed'],
        condition_sha256=hashlib.sha256(condition.cpu().contiguous().numpy().tobytes()).hexdigest(),
        active_head_countercheck_correction_RMS=float((enabled_probe - condition).square().mean().sqrt()),
        teacher_or_real_image_latent_used=False, first_UNet_input_is_scheduler_scaled_noise_not_raw_randn=True))
    progress = {}
    def unet_hook(module, args, result):
        assert args[0].shape[0] == 1 and torch.isfinite(result[0]).all()
        if progress['UNet_NFE'] == 0:
            progress['initial_UNet_input_sha256'] = hashlib.sha256(args[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        progress['UNet_NFE'] += 1
    def key_hook(module, args):
        assert args[0].shape == (1, 4, 2048) and args[0].data_ptr() == progress['token_pointer']
        progress['IP_key_calls'] += 1
    def value_hook(module, args):
        assert args[0].shape == (1, 4, 2048) and args[0].data_ptr() == progress['token_pointer']
        progress['IP_value_calls'] += 1
    def vae_hook(module, args, result):
        assert torch.isfinite(result).all()
        progress['VAE_decoder_calls'] += 1
        progress['VAE_float_min'], progress['VAE_float_max'] = float(result.min()), float(result.max())
    processors = [item for item in pipe.unet.attn_processors.values() if type(item).__name__ == 'IPAttnProcessor2_0']
    assert len(processors) == 70
    handles = [pipe.unet.register_forward_hook(unet_hook), pipe.vae.decoder.register_forward_hook(vae_hook)]
    for processor in processors:
        handles.extend([processor.to_k_ip.register_forward_pre_hook(key_hook), processor.to_v_ip.register_forward_pre_hook(value_hook)])
    rows, pixels = [], {}
    try:
        for case in CASES:
            assert time.monotonic() - started < config['maximum_seconds'] - 30
            progress.clear()
            progress.update(UNet_NFE=0, IP_key_calls=0, IP_value_calls=0, VAE_decoder_calls=0)
            with torch.inference_mode():
                if case == 'parent':
                    actual_condition = condition
                elif case == 'zero_head':
                    actual_condition = zero_head(condition, weight, bias, gate_mode='fixed')[0]
                else:
                    actual_condition = disabled_head(condition, weight, bias, enabled=False)[0]
                assert torch.equal(actual_condition, condition)
                native = actual_condition.half()
                tokens = projector(adapter(native))
                assert tokens.shape == (1, 4, 2048) and torch.isfinite(tokens).all()
                progress['token_pointer'] = tokens.data_ptr()
                image = pipe(height=512, width=512,
                    generator=torch.Generator(device='cuda').manual_seed(binding['image_seed']),
                    prompt_embeds=prompts[0], pooled_prompt_embeds=prompts[1],
                    cross_attention_kwargs={'ip_hidden_states': tokens}, num_inference_steps=4,
                    guidance_scale=0., output_type='pil').images[0].convert('RGB')
            assert progress['UNet_NFE'] == 4 and progress['IP_key_calls'] == progress['IP_value_calls'] == 280
            assert progress['VAE_decoder_calls'] == 1
            progress.pop('token_pointer')
            actual = np.asarray(image).copy()
            assert actual.shape == (512, 512, 3) and actual.dtype == np.uint8
            path = images / (case + '.webp')
            assert not path.exists()
            image.save(path, format='WEBP', lossless=True, quality=90, method=4)
            with Image.open(path) as decoded:
                assert decoded.size == (512, 512) and np.array_equal(actual, np.asarray(decoded.convert('RGB')))
            pixels[case] = actual
            row = dict(case=case, query_id=binding['query_id'], image_seed=binding['image_seed'],
                generated_path=str(path), generated_sha256=sha(path), RGB_pixel_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                raw_condition_byteexact=True, native_A_input_sha256=hashlib.sha256(native.cpu().contiguous().numpy().tobytes()).hexdigest(),
                IP_token_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
                lossless_WebP_roundtrip=True, teacher_or_target_used=False, **progress)
            rows.append(row)
            write_new(out / (case + '_receipt.json'), row)
        reference = rows[0]
        comparisons = {}
        for row in rows[1:]:
            case = row['case']
            comparisons[case] = dict(pixel_byteexact=np.array_equal(pixels['parent'], pixels[case]),
                maximum_absolute_RGB_difference=int(np.abs(pixels['parent'].astype('int16') - pixels[case].astype('int16')).max()),
                initial_UNet_input_byteexact=row['initial_UNet_input_sha256'] == reference['initial_UNet_input_sha256'],
                native_A_input_byteexact=row['native_A_input_sha256'] == reference['native_A_input_sha256'],
                IP_tokens_byteexact=row['IP_token_sha256'] == reference['IP_token_sha256'])
        passed = all(item['pixel_byteexact'] and item['initial_UNet_input_byteexact'] and item['native_A_input_byteexact']
                     and item['IP_tokens_byteexact'] for item in comparisons.values())
        result = dict(status='FUNCTIONAL_T_PIXEL_REPLAY_V2_EXACT_THREE_PATHS_PASSED' if passed else 'FUNCTIONAL_T_PIXEL_REPLAY_V2_DIFFERENCE_PRESERVED',
            actual_GPU_used=True, cases=rows, comparisons=comparisons, fixed_T_query=binding['query'], subject=binding['subject'],
            total_actual_UNet_NFE=12, native_IP_layers=70, total_IP_calls=840, actual_images_generated=3,
            component_parameters_trained=0, optimizer_created=False, D_V_test_tensors_or_scores_read=False,
            actual_image_gain_or_functional_bottleneck_established=False, engineering_identity_only=True,
            input_provenance_sha256=sha(out / 'input_provenance.json'), seconds=time.monotonic() - started)
        report_bytes = len((json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
        assert sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) + report_bytes <= config['maximum_probe_bytes']
        write_new(out / 'summary.json', result)
        assert passed, 'Actual replay differs; preserve all three images and diagnostics without seed changes'
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write_new(out / 'failure.json', dict(status='PIXEL_PROBE_PARTIAL_PRESERVED_NO_AUTOMATIC_RETRY',
            error=repr(error), completed_cases=rows, actual_images_generated=len(rows), no_model_updates=True))
        raise
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    evidence = independent_preflight(config)
    if args.preflight_only:
        print(json.dumps(dict(status='CPU_PIXEL_PROBE_PREFLIGHT_ONLY_NO_MODEL_OR_GPU_ACCESS', evidence=evidence)), flush=True)
        return
    evidence['exclusive_GPU_wrapper'] = exclusive_wrapper(config)
    run_probe(config, evidence)


if __name__ == '__main__':
    main()
