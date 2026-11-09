"""Separately permitted T-only engineering costs, never an official fit/export.

The absent engineering permission rejects before model imports, payloads or
output writes. A future source-bound permission uses the existing exclusive
GPU wrapper and funds only one temporary native epoch plus one full8270-row
raw-LN export. No method lock, official admission, trained final checkpoint,
official condition cache, image, score, scheduler or permission is issued here.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import brainhive_official_execution_contract_CPU_v1 as G
import brainhive_official_encoder_training_v1 as T

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
OUTPUT_BASE = ROOT / 'audit/official_encoder_engineering_profile_v1'
STATUS = 'T_ONLY_OFFICIAL_ENCODER_ENGINEERING_PROFILE_SOURCE_BOUND_PERMISSION'
SOURCE_REQUIRED = set(G.CORE_SOURCES) | set(G.AUTHOR_SOURCES) | {Path(__file__).name}
SPLIT_SHA = '1969d664555d08399e77a2d76c011f19eadbda1ce251f9c72d72c94de3910ad2'
BINDINGS_SHA = '7acfb0bdc3f6627150f8f320e21ca6cdc7f04ff9114ba498496ba4f8c458fa66'
_TOKEN = object()
require = T.require


def check_sources(permission):
    sources = permission['source_sha256']
    require(SOURCE_REQUIRED <= set(sources), 'Pin every actual profile, frozen encoder/native/resource and author dependency')
    for name, digest in sources.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and G.E.P.digest(digest)
            and T.sha(ROOT / path) == digest, 'Changed or unbound engineering source: ' + name)
    require(sources[Path(__file__).name] == T.sha(__file__), 'Profile entry itself must be bound before execution')


def training_rows(protocol, count=8270, *, _fixture=False):
    """Use global T row IDs in their registered order, never range(count)."""
    parts = protocol['image_rows']; allowed = parts['T']
    require(len(allowed) == len(set(allowed)) and set(allowed).isdisjoint(parts['D'] + parts['V']), 'T/D/V identity overlap')
    require((len(allowed) == 13230 and count == 8270) or _fixture, 'Full engineering T8270 scope is fixed')
    rows = allowed[:count]
    require(len(rows) == count and all(type(i) is int and 0 <= i < (16540 if not _fixture else 100000) for i in rows), 'Complete fixed unique T rows required')
    return rows


def wrapper_evidence(permission, permission_path):
    """No subprocess launch: validate the already running trusted wrapper."""
    require(os.name == 'posix' and sys.executable == permission['runtime']['python_executable'], 'Actual reviewed Linux runtime required')
    parent = Path('/proc') / str(os.getppid())
    argv = [v.decode() for v in (parent / 'cmdline').read_bytes().split(b'\0') if v]
    require(str(ROOT / 'run_gpu_job_v2.py') in argv and '--name' in argv and '--' in argv, 'Existing exclusive GPU wrapper required')
    name = argv[argv.index('--name') + 1]
    require(name == permission['job_name'] and Path(name).name == name, 'Only the single preregistered engineering job is allowed')
    job_path = ROOT / 'audit' / (name + '_job.json'); job = G.read(job_path)
    expected = [sys.executable, '-B', str(Path(__file__).resolve()), '--permission', str(permission_path)]
    require(job['status'] == 'RUNNING' and job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid()
        and job['command'] == expected == argv[argv.index('--') + 1:], 'Actual profile command/PID/wrapper identity differs')
    require(job['max_seconds'] == permission['maximum_seconds'] and 0 < job['max_seconds'] <= 3600, 'Finite one-hour-or-less engineering job required')
    auth = G.ref(ROOT / 'configs/resource_authorization_20261020_v3.json')
    require(job['authorization_sha256'] == auth['sha256'] and permission['resource_authorization'] == auth, 'Existing resource authority changed')
    lock_path = ROOT / 'audit/GPU_JOB.lock'; info = lock_path.stat()
    held = False
    for fd in (parent / 'fd').iterdir():
        try:
            stat = fd.stat(); held |= (stat.st_dev, stat.st_ino) == (info.st_dev, info.st_ino)
        except FileNotFoundError:
            pass
    require(held, 'Trusted parent must retain the actual GPU lock descriptor')
    import fcntl
    with lock_path.open('r') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
            raise ValueError('The existing GPU lock is not held')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], check=True,
        capture_output=True, text=True, timeout=10)
    require({int(v.strip()) for v in apps.stdout.splitlines() if v.strip()} <= {os.getpid()}, 'Another GPU process is active')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], check=True,
        capture_output=True, text=True, timeout=10)
    rows = [v.split(',') for v in gpu.stdout.splitlines() if v.strip()]
    require(len(rows) == 1 and rows[0][0].strip() == permission['runtime']['GPU_uuid']
        and int(rows[0][1]) * (1 << 20) >= permission['minimum_free_GPU_bytes'], 'Actual permitted single GPU/free memory differs')
    import datetime
    elapsed = time.time() - datetime.datetime.fromisoformat(job['start_utc'].replace('Z', '+00:00')).timestamp()
    require(0 <= elapsed < job['max_seconds'], 'Finite actual profile deadline expired')
    return dict(job_name=name, job_ref_at_start=G.ref(job_path), wrapper_pid=os.getppid(), pid=os.getpid(),
        deadline=time.monotonic() + job['max_seconds'] - elapsed, initial_GPU_free_bytes=int(rows[0][1]) * (1 << 20))


def resources(context):
    require(context.get('_token') is _TOKEN, 'Actual independent engineering permission required')
    T.tick(context['deadline']); permission = context['permission']
    require(G.ref(context['permission_ref']['path']) == context['permission_ref'], 'Engineering permission changed')
    from brainhive_functional_training_contract_v2 import formal_remaining, resource_headroom
    memory = resource_headroom()
    require(memory['effective_headroom_bytes'] >= permission['minimum_effective_CPU_headroom_bytes'], 'Fresh permitted CPU headroom unavailable')
    plan = G.bound(permission['remaining_lifecycle_plan'])
    volumes = formal_remaining(plan)
    output = Path(context['output_root']); ancestor = output
    while not ancestor.exists(): ancestor = ancestor.parent
    require(all(not p.is_symlink() for p in [output, *output.parents] if p.exists()), 'Engineering output path/ancestor cannot alias other storage')
    output_device = ancestor.stat().st_dev
    matching = [v for v in volumes if Path(v['root']).stat().st_dev == output_device]
    require(len(matching) == 1, 'Engineering output must be budgeted on an actual lifecycle forecast filesystem')
    used = directory_allocated(output)
    cap = permission['maximum_output_bytes']
    require(used <= cap and matching[0]['free'] >= matching[0]['required'] + cap - used,
        'Engineering output would spend storage reserved for the current full lifecycle')
    return dict(memory=memory, remaining_lifecycle=volumes, engineering_allocated_bytes=used, engineering_output_cap_bytes=cap)


def directory_allocated(output):
    output = Path(output); used = 0; inodes = set()
    if output.exists():
        for file in output.rglob('*'):
            require(not file.is_symlink(), 'Engineering namespace cannot contain aliases')
            if file.is_file():
                stat = file.stat(); inode = stat.st_dev, stat.st_ino
                if inode not in inodes:
                    used += stat.st_blocks * 512 if hasattr(stat, 'st_blocks') else stat.st_size; inodes.add(inode)
    return used


def write_engineering_receipt(root, raw, maximum_bytes):
    """Account for the actual receipt before its first payload byte is written.

    The engine's pending/canonical publication uses one hardlinked inode, so
    it is charged once; round the prospective regular file to filesystem blocks.
    """
    root = Path(root)
    require(root.is_dir() and not root.is_symlink() and isinstance(raw, bytes) and len(raw) <= 16 << 20,
        'Bounded private engineering receipt bytes required')
    block = max(4096, os.statvfs(root).f_frsize) if hasattr(os, 'statvfs') else 4096
    prospective = ((len(raw) + block - 1) // block) * block
    require(type(maximum_bytes) is int and directory_allocated(root) + prospective <= maximum_bytes,
        'Engineering receipt would exceed the permitted output cap before publication')
    T.write_once(root / 'engineering_profile.json', raw)
    require(directory_allocated(root) <= maximum_bytes, 'Actual engineering receipt allocation exceeds permitted cap; preserve evidence')


def preflight(permission_path):
    """The first public CLI operation. No positive fixture/admission bypass."""
    path = Path(permission_path)
    require(path.is_absolute() and path.is_file(), 'Engineering GPU permission does not exist; no Torch, payload or output access')
    permission = G.read(path)
    require(permission['schema'] == 'BrainHIVE_official_encoder_engineering_permission_v1' and permission['status'] == STATUS,
        'A formal-method Boolean or metadata plan is not an engineering permission')
    require(permission['purpose'] == 'T_only_native_first_epoch_and_full8270_raw_LN_export_costs'
        and permission['formal_fit_or_export_admitted'] is False and permission['V_test_or_scores_allowed'] is False
        and permission['new_candidate_configurations'] == 0 and permission['training_seed'] == 1729
        and permission['training_epochs'] == 1 and permission['training_updates'] == 9
        and permission['scheduler_horizon_updates'] == 225 and permission['export_rows'] == 8270
        and type(permission['write_single_epoch_snapshot']) is bool, 'Only the fixed engineering scope is implemented')
    require(type(permission['maximum_output_bytes']) is int and 0 < permission['maximum_output_bytes'] <= 1 << 30
        and type(permission['minimum_effective_CPU_headroom_bytes']) is int and permission['minimum_effective_CPU_headroom_bytes'] >= 8 << 30
        and type(permission['minimum_free_GPU_bytes']) is int and permission['minimum_free_GPU_bytes'] > 0,
        'Concrete bounded engineering output/CPU/GPU resources are mandatory')
    check_sources(permission)
    cpu = G.bound(permission['CPU_contract'])
    require(cpu['status'] == 'PASS_TINY_CPU_ENGINEERING_ONLY_NO_PERMISSION_OR_GPU'
        and cpu['source_sha256'][Path(__file__).name] == T.sha(__file__)
        and cpu['actual_GPU_used'] is False, 'Actual current-source CPU contract must precede engineering GPU permission')
    for name, digest in cpu['source_sha256'].items():
        require(Path(name).name == name and G.E.P.digest(digest) and T.sha(ROOT / name) == digest,
            'Engineering CPU-check dependency changed after its actual receipt')
    require(permission['split_manifest'] == dict(path=str(ROOT / 'configs/split_manifest.json'), sha256=SPLIT_SHA), 'Use the actual protected T split')
    protocol = G.bound(permission['split_manifest']); rows = training_rows(protocol)
    require(permission['actual_input_bindings'] == dict(path=str(WORK / 'data/actual_input_bindings.json'), sha256=BINDINGS_SHA), 'Use actual physical-mean4 input bindings')
    bindings = G.bound(permission['actual_input_bindings']); subject = permission['subject']
    require(subject in G.E.P.SUBJECTS and bindings['selected_indices'] == list(range(46, 63))
        and bindings['subjects'][subject]['train_repeats'] == 4 and bindings['additional_per_channel_zscore'] is False, 'Native17channel/R4 input contract changed')
    payload = permission['payloads']; item = bindings['subjects'][subject]
    require(payload['EEG'] == dict(path=item['train_path'], sha256=item['train_sha256'])
        and payload['visual']['path'] == str(WORK / 'data/train_visual_embeddings.pt')
        and set(payload['native_prior_assets']) == {str(WORK / 'assets/prior_H14' / module / name)
            for module in ('fusion_encoder', 'proj') for name in ('config.json', 'diffusion_pytorch_model.safetensors')}
        and payload['native_asset_verification']['path'] == str(WORK / 'audit/native_asset_verification.json'), 'Only native training EEG/visual/F/A/P assets are permitted')
    require(permission['runtime']['python_executable'] == sys.executable
        and permission['runtime']['python_version'] == sys.version.split()[0]
        and permission['runtime']['package_source_sha256'], 'Concrete actual Python/Torch runtime source pins required')
    for filename, digest in permission['runtime']['package_source_sha256'].items():
        require(Path(filename).is_absolute() and G.E.P.digest(digest) and T.sha(filename) == digest, 'Actual package source changed')
    wrapper = wrapper_evidence(permission, path)
    output = Path(permission['output_root'])
    require(output == OUTPUT_BASE / permission['job_name'] and not output.exists() and not output.is_symlink(), 'A fresh exclusive engineering namespace is mandatory; prior attempts are preserved')
    context = dict(_token=_TOKEN, permission=permission, permission_ref=G.ref(path), wrapper=wrapper,
        output_root=str(output), rows=rows, subject=subject, deadline=wrapper['deadline'])
    context['resources_before_payload'] = resources(context)
    return context


def verify_payloads(context):
    resources(context); permission = context['permission']; payload = permission['payloads']
    refs = [payload['EEG'], payload['visual'], payload['native_asset_verification']]
    refs += [dict(path=p, sha256=s) for p, s in payload['native_prior_assets'].items()]
    started = time.monotonic()
    for item in refs:
        require(set(item) == {'path', 'sha256'} and G.E.P.digest(item['sha256']) and T.sha(item['path']) == item['sha256'], 'Actual engineering input bytes changed')
        T.tick(context['deadline'])
    verified = G.read(payload['native_asset_verification']['path'])
    for filename, digest in payload['native_prior_assets'].items():
        if filename.endswith('.safetensors'):
            require(verified['files'][filename]['sha256'] == digest, 'Native receipt/engineering asset SHA differs')
    return refs, time.monotonic() - started


def select_T_tensors(torch, raw_eeg, raw_visuals, rows, proj_meta, *, _fixture=None):
    """Only chosen T numeric values are inspected. Other rows stay mmap data."""
    shape = (16540, 63, 250) if _fixture is None else tuple(_fixture['EEG_shape'])
    channels = list(range(46, 63)) if _fixture is None else _fixture['selected_indices']
    count = 8270 if _fixture is None else _fixture['rows']
    require(isinstance(raw_eeg, torch.Tensor) and raw_eeg.dtype == torch.float32 and tuple(raw_eeg.shape) == shape,
        'Full EEG storage metadata shape/dtype differs')
    require(len(rows) == len(set(rows)) == count and all(type(i) is int and 0 <= i < shape[0] for i in rows), 'Exact unique ordered engineering T rows required')
    require(set(raw_visuals) == set(proj_meta), 'Native F visual keys differ')
    ids = torch.tensor(rows, dtype=torch.long); selected_channels = torch.tensor(channels, dtype=torch.long)
    eeg = raw_eeg[ids[:, None], selected_channels[None, :], :].contiguous()
    visuals = {}
    for name, width in proj_meta.items():
        value = raw_visuals[name]
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and len(value) == shape[0]
            and math.prod(value.shape[1:]) == width, 'Full visual storage metadata differs')
        visuals[name] = value.index_select(0, ids).contiguous()
    require(bool(torch.isfinite(eeg).all()) and float(eeg.std(unbiased=False)) > 0
        and all(bool(torch.isfinite(v).all()) and float(v.std(unbiased=False)) > 0 for v in visuals.values()), 'Nonfinite/degenerate selected T payload')
    return eeg, visuals, dict(EEG_storage_shape=list(shape), selected_shape=list(eeg.shape), rows=rows,
        selected_channels=channels, visual_shapes={k:list(v.shape) for k,v in visuals.items()},
        D_or_V_numeric_values_inspected=False, fixture_only=_fixture is not None)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(state):
    pieces = [name + ':' + str(value.dtype) + ':' + str(list(value.shape)) + ':' + tensor_sha(value) for name, value in sorted(state.items())]
    return hashlib.sha256('\n'.join(pieces).encode()).hexdigest()


def export_T_conditions(torch, model, raw_eeg, rows, checkpoint_model, *, device, deadline, _fixture=None):
    """Separate engineering loop; same native math, not the gated official exporter."""
    count, width = (8270, 1024) if _fixture is None else (_fixture['rows'], _fixture['width'])
    shape = (16540, 63, 250) if _fixture is None else tuple(_fixture['EEG_shape'])
    channels = list(range(46, 63)) if _fixture is None else _fixture['selected_indices']
    require(tuple(raw_eeg.shape) == shape and raw_eeg.dtype == torch.float32 and len(rows) == len(set(rows)) == count,
        'Engineering full export storage metadata/row scope differs')
    if _fixture is None: require(sum(p.numel() for p in model.parameters()) == 6452224, 'Actual unscaled native BrainMLP required')
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    output = []; batches = []
    with torch.inference_mode():
        for lo in range(0, count, 256):
            T.tick(deadline); ids = rows[lo:lo + 256]
            result = model(raw_eeg[ids][:, channels, :].to(device=device, dtype=torch.float32))
            require(result.dtype == torch.float32 and tuple(result.shape) == (len(ids), width)
                and bool(torch.isfinite(result).all()), 'Engineering raw-LN export shape/dtype/finite failure')
            output.append(result.detach().cpu()); batches.append(len(ids))
    condition = torch.cat(output).contiguous()
    require(float(condition.std(unbiased=False)) > 0, 'Degenerate engineering export')
    affine = [model.state_dict()[key].detach().cpu().clone() for key in G.E.AFFINE_KEYS]
    require(all(value.dtype == torch.float32 and tuple(value.shape) == (width,) and torch.equal(value, checkpoint_model[key].cpu())
        for key, value in zip(G.E.AFFINE_KEYS, affine)), 'Engineering affine is not the same transient head own final-LN affine')
    report = dict(rows=count, shape=list(condition.shape), dtype='float32', array_rows=rows, batch_rows=batches,
        condition_tensor_sha256=tensor_sha(condition), affine_weight_sha256=tensor_sha(affine[0]), affine_bias_sha256=tensor_sha(affine[1]),
        normalization='raw LayerNorm output', additional_L2_normalization=False,
        model_training_mode=model.training, gradients_enabled=False, parameters_frozen=all(not p.requires_grad for p in model.parameters()),
        actual_full8270_GPU_engineering_export=_fixture is None and str(device).startswith('cuda'),
        official_OOF_export=False, exported_rows_overlap_engineering_training_rows=True, conditions_saved_to_disk=False)
    verify_export_observation(torch, condition, report, rows, checkpoint_model, _fixture=_fixture)
    return condition, report


def verify_export_observation(torch, condition, report, rows, checkpoint_model, *, _fixture=None):
    count, width = (8270, 1024) if _fixture is None else (_fixture['rows'], _fixture['width'])
    require(condition.dtype == torch.float32 and tuple(condition.shape) == (count, width)
        and bool(torch.isfinite(condition).all()) and report['array_rows'] == rows and report['rows'] == count,
        'Observed engineering export tensor/global row order differs')
    require(report['condition_tensor_sha256'] == tensor_sha(condition) and report['normalization'] == 'raw LayerNorm output'
        and report['additional_L2_normalization'] is False and report['model_training_mode'] is False
        and report['gradients_enabled'] is False and report['parameters_frozen'] is True,
        'Observed engineering export raw-LN/value/mode differs')
    require(report['affine_weight_sha256'] == tensor_sha(checkpoint_model[G.E.AFFINE_KEYS[0]])
        and report['affine_bias_sha256'] == tensor_sha(checkpoint_model[G.E.AFFINE_KEYS[1]]),
        'Observed engineering export own checkpoint affine differs')


def measure(torch, operation):
    torch.cuda.synchronize(); start = time.monotonic(); result = operation(); torch.cuda.synchronize()
    return result, time.monotonic() - start


def run_profile(context):
    require(context.get('_token') is _TOKEN, 'Only an actual engineering permission can load models or run a profile')
    started = time.monotonic(); permission = context['permission']; before = resources(context)
    files, hashing_seconds = verify_payloads(context)
    import brainhive_native_models_v1 as N
    torch = N.torch
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.__version__ == permission['runtime']['torch_version'], 'Actual permitted CUDA/BF16/Torch runtime required')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(); T.seed_everything(torch)
    scope = N.classes()
    priors, native_load_seconds = measure(torch, lambda: N.prior_models('cuda'))
    teacher = priors['fusion_encoder'].vision_model
    require(sum(p.numel() for p in teacher.parameters()) == 11015168, 'Use actual H14 F.vision_model before A')
    T.frozen_stochastic_teacher(torch, teacher)
    proj_meta = dict(teacher.input_meta)
    del priors; gc.collect(); torch.cuda.empty_cache()
    cpu_start = time.monotonic(); payload = permission['payloads']
    raw_eeg = torch.load(payload['EEG']['path'], weights_only=True, map_location='cpu', mmap=True)
    raw_visuals = torch.load(payload['visual']['path'], weights_only=True, map_location='cpu', mmap=True)
    selected_eeg, selected_visuals, selection = select_T_tensors(torch, raw_eeg, raw_visuals, context['rows'], proj_meta)
    selection_seconds = time.monotonic() - cpu_start
    (eeg, visuals), transfer_seconds = measure(torch, lambda: (selected_eeg.to('cuda'), {k:v.to('cuda') for k,v in selected_visuals.items()}))
    del raw_visuals, selected_eeg, selected_visuals; gc.collect()
    teacher_sha_before = state_sha(teacher.state_dict())
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        probe = {k:v[:4] for k,v in visuals.items()}
        first, second = teacher(**probe), teacher(**probe)
        require(bool(torch.isfinite(first).all()) and float(first.float().std(unbiased=False)) > 0
            and not torch.equal(first, second), 'Native frozen train/dropout teacher is nonfinite/degenerate/not stochastic')
    # The probe cannot change the actual first-epoch RNG policy.
    T.seed_everything(torch)
    model = scope['BrainMLP'](4250, 1024, num_layers=1, dropout=.1).to('cuda')
    require(sum(p.numel() for p in model.parameters()) == 6452224, 'Full native student size changed')
    logit = torch.nn.Parameter(torch.tensor(2.6592, device='cuda'))
    optimizer, scheduler = T.make_optimizer(torch, model, logit, 225)
    shuffle = torch.Generator(device='cpu').manual_seed(1730)
    initial_model_sha = state_sha(model.state_dict())
    row, train_seconds = measure(torch, lambda: T.run_epoch(torch, model, teacher, logit, optimizer, scheduler, shuffle,
        eeg, visuals, epoch=1, updates=0, total_updates=225, batch_size=1024, subject=context['subject'],
        role='ENGINEERING_T8270_ONE_EPOCH_NOT_OFFICIAL_OOF', device='cuda', deadline=context['deadline'], autocast_bfloat16=True)[0])
    require(row['updates'] == 9 and row['full_epoch_rows'] == 8270 and row['expected_updates'] == 225
        and row['grad_norm_last'] > 0 and initial_model_sha != state_sha(model.state_dict())
        and teacher_sha_before == state_sha(teacher.state_dict()) and all(p.grad is None for p in teacher.parameters()), 'No actual finite student update or frozen-teacher contract failed')
    after_train = resources(context)
    # Engineering snapshot is never named brain_final.pt/training_complete.json.
    cpu_start = time.monotonic()
    snapshot = T.snapshot(torch, model, logit, optimizer, scheduler, shuffle, 1, 9,
        context['permission_ref']['sha256'], G.E.core_sha(permission), T.sha(__file__), 'cuda')
    snapshot_seconds = time.monotonic() - cpu_start
    root = Path(context['output_root']); root.mkdir(parents=True, mode=0o700)
    transaction = None; transaction_seconds = None; snapshot_bytes = 0
    if permission['write_single_epoch_snapshot']:
        require(sum(v.numel() * v.element_size() for v in snapshot['model'].values()) * 4 < permission['maximum_output_bytes'], 'Single engineering snapshot estimate exceeds permitted output cap')
        cpu_start = time.monotonic(); transaction = T.publish_epoch(torch, root, snapshot, row)
        transaction_seconds = time.monotonic() - cpu_start
        snapshot_path = Path(transaction['path']).parent / 'resume.pt'; snapshot_bytes = snapshot_path.stat().st_size
        snapshot = torch.load(snapshot_path, weights_only=True, map_location='cpu')
        require(snapshot['producer_source_sha256'] == T.sha(__file__) and snapshot['updates'] == 9, 'Engineering snapshot reload/source differs')
    transient_checkpoint_sha = state_sha(snapshot['model'])
    del model, teacher, optimizer, scheduler, eeg, visuals, logit, first, second, probe
    gc.collect(); torch.cuda.empty_cache()
    (export_model, _), reconstruction_seconds = measure(torch, lambda: (scope['BrainMLP'](4250, 1024, num_layers=1, dropout=.1), None))
    export_model.load_state_dict(snapshot['model'], strict=True)
    (condition, export_report), export_seconds = measure(torch, lambda: export_T_conditions(torch, export_model, raw_eeg, context['rows'],
        snapshot['model'], device='cuda', deadline=context['deadline']))
    require(export_report['rows'] == 8270 and len(export_report['batch_rows']) == 33 and export_report['batch_rows'][-1] == 78,
        'Actual full8270 export batches were truncated')
    after = resources(context); check_sources(permission)
    report = dict(schema='BrainHIVE_official_encoder_actual_engineering_profile_v1',
        status='ACTUAL_T_ONLY_NATIVE_FIRST_EPOCH_AND_FULL8270_ENGINEERING_EXPORT_MEASURED_NOT_FORMAL',
        purpose_segments=['train', 'export'], actual_GPU_used=True, permission=context['permission_ref'], wrapper=context['wrapper'],
        source_sha256=permission['source_sha256'], runtime=dict(torch=torch.__version__, CUDA=torch.version.cuda,
            GPU_uuid=permission['runtime']['GPU_uuid'], device_name=torch.cuda.get_device_name()),
        inputs=files, subject=context['subject'], physical_repeat_count=4, selection=selection,
        train=dict(actual_epoch=row, global_batch=1024, actual_batch_rows=[1024]*8+[78], scheduler_horizon_updates=225,
            native_parameters=6452224, teacher_parameters=11015168, teacher_mode='frozen weights, train/dropout .1',
            teacher_sha_before=teacher_sha_before, teacher_sha_after=teacher_sha_before,
            student_initial_sha256=initial_model_sha, student_profiled_sha256=transient_checkpoint_sha,
            first_epoch_LR_horizon_matches_official=True, BF16_forward=True, last_batch_retained=True),
        export=export_report, engineering_epoch_transaction=transaction, engineering_snapshot_bytes=snapshot_bytes,
        seconds=dict(payload_file_hashing=hashing_seconds, native_F_A_P_load=native_load_seconds,
            CPU_mmap_and_T_selection=selection_seconds, selected_T_to_GPU=transfer_seconds, one_native_epoch=train_seconds,
            snapshot_CPU_materialization=snapshot_seconds, single_epoch_transaction_IO=transaction_seconds,
            export_model_CPU_reconstruction=reconstruction_seconds, full8270_raw_LN_export=export_seconds,
            total=time.monotonic()-started),
        actual_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), actual_peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
        resource_before=before, resource_after_train=after_train, resource_after=after,
        conditions_tensor_bytes=condition.numel()*condition.element_size(), conditions_saved_to_disk=False,
        official_models_or_caches_produced=0, generated_images=0, scores_read=0, new_candidate_configurations=0,
        method_lock_or_formal_admission_issued=False,
        limits=['Engineering head trained one T-only epoch; exported rows overlap its training rows. No held-OOF effectiveness evidence.',
            'One warm-process train+export job has two measured segments, not two independent completed GPU jobs.',
            'No25epoch convergence/20task cost or all25 optimizer snapshots measured; optional one-epoch I/O is only one transaction.',
            'Export reconstructs the transient same-source head; it is not the protected official executor or a new official cache.',
            'File hashes cover full training-container bytes; numeric EEG/visual access is limited to registered T rows. No D/V/test target or score is read.',
            'Cold separate-process startup, full original-task transfer patterns, cumulative allocator effects and formal retries need explicit future scope-compatibility review.',
            'This output is measured engineering evidence, not a method lock, official execution permission or image-quality result.'])
    raw = T.json_bytes(report)
    require(len(raw) <= 16 << 20, 'Bounded engineering receipt size exceeded')
    resources(context)
    write_engineering_receipt(root, raw, permission['maximum_output_bytes'])
    resources(context)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--permission', type=Path, required=True)
    args = parser.parse_args(argv); context = preflight(args.permission)
    old = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame): raise TimeoutError('Finite independent engineering profile deadline; preserve evidence')
    signal.signal(signal.SIGALRM, expire); signal.setitimer(signal.ITIMER_REAL, max(.001, context['deadline']-time.monotonic()))
    try:
        result = run_profile(context)
        print(json.dumps(dict(status=result['status'], output_root=context['output_root'], official_fit_or_export=False)), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


if __name__ == '__main__': main()
