"""Future admitted full16540 TRAIN native VAE-mode bank, never test images.

The CLI first requires the separate source/method/resource/wrapper gate. This
file grants no permission and imports no tensor/model library at module import.
The existing13230 protected-T bank is neither read, replaced nor extended.
Immutable committed chunks permit deterministic continuation of the new bank;
uncommitted data may be recomputed, while all old files remain present.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
DECODER_REGISTRATION_SHA = 'c3cc49533d09e6f35f134142414516065e3861f1150df5cb1157fa5337fe3cd5'
SHAPE = (16540, 4, 64, 64)
PREPROCESSING = 'Unchanged author Resize512 BILINEAR, CenterCrop512, RGB[-1,1]'
COMPLETE = 'OFFICIAL_ALL16540_TRAIN_NATIVE_FP32_VAE_MODE_LATENTS_COMPLETE'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    with Path(path).open('rb') as stream:
        result = hashlib.file_digest(stream, 'sha256').hexdigest()
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return result


def ref(path):
    return dict(path=str(Path(path)), sha256=sha(path))


def sync_directory(path):
    if os.name == 'posix':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def allocation_bound(path, bytes_to_write):
    ancestor = Path(path).parent
    while not ancestor.exists(): ancestor = ancestor.parent
    unit = max(4096, os.statvfs(ancestor).f_frsize) if hasattr(os, 'statvfs') else 4096
    return ((bytes_to_write + unit - 1) // unit) * unit


def write_once(path, value, *, reserve=None):
    """Publish only complete metadata. Pending evidence is deliberately retained."""
    path = Path(path)
    require(not path.exists(), 'Immutable metadata already exists: ' + str(path))
    raw = canonical(value) + b'\n'
    if reserve is not None: reserve(allocation_bound(path, len(raw)))
    pending = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with pending.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path)
    sync_directory(path.parent)


def train_rows(bundle):
    """Actual full training metadata only, in canonical global array-row order."""
    rows = [dict(row, array_row=int(row['array_row']), image_bytes=int(row['image_bytes']))
            for row in bundle['image_rows'] if row['dataset_split'] == 'train']
    rows.sort(key=lambda row: row['array_row'])
    require(len(rows) == 16540 and [r['array_row'] for r in rows] == list(range(16540)),
            'Full official training rows0..16539 required, never the old T subset')
    require(len({r['concept_id'] for r in rows}) == 1654
            and len({r['stimulus_id'] for r in rows}) == 16540
            and len({r['image_sha256'] for r in rows}) == 16540, 'Training identities differ')
    return rows


def image_identity(rows):
    return [dict(array_row=r['array_row'], stimulus_id=r['stimulus_id'],
                 image_sha256=r['image_sha256'], image_bytes=r['image_bytes'],
                 image_zip=r['image_zip'], image_member=r['image_member']) for r in rows]


def allocated_bytes(root):
    seen, total = set(), 0
    for path in Path(root).rglob('*'):
        require(not path.is_symlink(), 'Output aliases are forbidden')
        if path.is_file():
            info = path.stat(); key = info.st_dev, info.st_ino
            if key not in seen:
                total += info.st_blocks * 512 if hasattr(info, 'st_blocks') else info.st_size
                seen.add(key)
    return total


def chunk_bounds(count, chunk_size):
    return [(start, min(start + chunk_size, count)) for start in range(0, count, chunk_size)]


def _scan_chunks(root, array, registration_sha, bounds):
    import numpy as np
    records = []
    paths = sorted((root / 'chunks').glob('chunk-*.json'))
    require(len(paths) <= len(bounds), 'More committed chunks than full bank')
    for number, path in enumerate(paths):
        require(path.name == 'chunk-' + str(number).zfill(3) + '.json'
                and not path.is_symlink(), 'Nonconsecutive or aliased committed chunks')
        record = json.loads(path.read_bytes()); start, stop = bounds[number]
        require(set(record) == {'start', 'stop', 'registration_sha256', 'latent_bytes_sha256'}
                and record['start'] == start and record['stop'] == stop
                and record['registration_sha256'] == registration_sha, 'Committed chunk identity differs')
        values = array[start:stop]
        require(bool(np.isfinite(values).all()) and hashlib.sha256(values.tobytes()).hexdigest()
                == record['latent_bytes_sha256'], 'Committed latent chunk bytes changed')
        records.append(record)
    return records


def _verify_bank(root, registration):
    """Full actual array/chunk verification; no sampled or exists-only adoption."""
    import numpy as np
    root = Path(root)
    reg_path, done_path = root / 'registration.json', root / 'completion.json'
    require(not reg_path.is_symlink() and not done_path.is_symlink()
            and not (root / 'chunks').is_symlink(), 'No aliased bank metadata/chunk directory')
    require(json.loads(reg_path.read_bytes()) == registration, 'Bank registration differs')
    done = json.loads(done_path.read_bytes())
    expected_status = COMPLETE if registration['actual_official_full_train_scope'] else 'CPU_TINY_LATENT_BANK_FIXTURE_COMPLETE'
    require(done['status'] == expected_status and done['registration'] == ref(reg_path),
            'Completion/registration source chain differs')
    path = root / 'native_VAE_train_modes.npy'
    require(not path.is_symlink() and done['bank'] == ref(path), 'Full completed bank SHA differs')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    require(array.shape == tuple(registration['shape']) and array.dtype == np.float32
            and array.flags.c_contiguous, 'Bank shape/dtype/layout differs')
    bounds = chunk_bounds(len(array), registration['chunk_size'])
    chunks = _scan_chunks(root, array, sha(reg_path), bounds)
    require(len(chunks) == len(bounds) and chunks == done['committed_chunks']
            and done['array_rows'] == registration['array_rows'], 'Incomplete canonical full bank coverage')
    require(done['official_test_image_values_used'] is False and done['parameters_fitted'] == 0,
            'Bank is not a no-fit training-only artifact')
    pending = root / 'native_VAE_train_modes.npy.pending'
    require(pending.is_file() and not pending.is_symlink() and os.path.samefile(pending, path),
            'Canonical bank must be the actual committed pending inode')
    del array
    return done


def _build_bank(root, registration, encode_batch, guard, *, reserve=None, fault=None):
    """Shared real bank writer. Callers must supply a production gate or CPU fixture.

    encode_batch consumes the precise requested global rows and returns float32
    ndarray values. The production caller below fixes the full shape and VAE.
    The CPU checker injects a tiny deterministic encoder, never an admission.
    """
    import numpy as np
    root = Path(root); shape = tuple(registration['shape'])
    ids, batch, chunk = registration['array_rows'], registration['batch_size'], registration['chunk_size']
    require(len(ids) == shape[0] and len(set(ids)) == len(ids)
            and all(type(i) is int and i >= 0 for i in ids), 'Unique complete ordered rows required')
    require(batch == 4 and chunk == 256 and registration['dtype'] == 'float32', 'Fixed bank batch/chunk/dtype contract')
    require(not registration['actual_official_full_train_scope'] or callable(reserve),
            'Production bank requires concrete write-before-allocation budget checks')
    guard(); require(not root.is_symlink(), 'No aliased output root')
    require(all(not p.is_symlink() for p in root.parents if p.exists()), 'No aliased output ancestor')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not (root / 'chunks').is_symlink(), 'No aliased chunk directory')
    (root / 'chunks').mkdir(exist_ok=True, mode=0o700)
    reg_path = root / 'registration.json'
    if reg_path.exists():
        require(not reg_path.is_symlink() and json.loads(reg_path.read_bytes()) == registration,
                'Changed source/input/shape registration: retain and refuse')
    else:
        require(not (root / 'native_VAE_train_modes.npy.pending').exists(), 'Orphan bank is retained, never adopted')
        write_once(reg_path, registration, reserve=reserve)
    if (root / 'completion.json').exists():
        guard(); return _verify_bank(root, registration)
    require(not (root / 'native_VAE_train_modes.npy').exists(), 'Orphan canonical bank retained; no overwrite')
    pending = root / 'native_VAE_train_modes.npy.pending'
    if pending.exists():
        require(not pending.is_symlink() and stat.S_ISREG(pending.stat().st_mode), 'Regular private pending bank required')
        array = np.load(pending, mmap_mode='r+', allow_pickle=False)
        require(array.shape == shape and array.dtype == np.float32 and array.flags.c_contiguous,
                'Pending bank shape/dtype differs')
    else:
        require(not list((root / 'chunks').glob('chunk-*.json')), 'Committed chunks lack their bank; retain and stop')
        if reserve is not None:
            # Fixed4D npy header is below512 bytes; reserve full allocated file
            # blocks before open_memmap can preallocate or write any array.
            reserve(allocation_bound(pending, int(np.prod(shape)) * 4 + 512))
        array = np.lib.format.open_memmap(pending, mode='w+', dtype=np.float32, shape=shape)
    bounds = chunk_bounds(len(ids), chunk)
    committed = _scan_chunks(root, array, sha(reg_path), bounds)
    resumed = len(committed); calls = 0; started = time.monotonic()
    for number in range(resumed, len(bounds)):
        start, stop = bounds[number]
        for offset in range(start, stop, batch):
            guard(); selected = ids[offset:min(offset + batch, stop)]
            value = encode_batch(selected); calls += 1
            require(isinstance(value, np.ndarray) and value.dtype == np.float32
                    and value.shape == (len(selected), *shape[1:]) and bool(np.isfinite(value).all()),
                    'Native VAE mode returned wrong shape/dtype/nonfinite values')
            array[offset:offset + len(selected)] = value
        array.flush()
        with pending.open('r+b') as stream: os.fsync(stream.fileno())
        record = dict(start=start, stop=stop, registration_sha256=sha(reg_path),
                      latent_bytes_sha256=hashlib.sha256(array[start:stop].tobytes()).hexdigest())
        if fault is not None: fault('before_chunk_commit', number)
        guard(); write_once(root / 'chunks' / ('chunk-' + str(number).zfill(3) + '.json'), record, reserve=reserve)
        committed.append(record)
        if fault is not None: fault('after_chunk_commit', number)
    array.flush(); del array
    guard(); canonical_path = root / 'native_VAE_train_modes.npy'
    os.link(pending, canonical_path); sync_directory(root)
    done = dict(status=COMPLETE if registration['actual_official_full_train_scope'] else 'CPU_TINY_LATENT_BANK_FIXTURE_COMPLETE',
        registration=ref(reg_path), bank=ref(canonical_path), committed_chunks=committed, array_rows=ids,
        shape=list(shape), dtype='float32', deterministic_VAE_mode=True, RAM_is_volatile=False,
        official_test_image_values_used=False, parameters_fitted=0, images_reconstructed=0,
        committed_chunks_reused=resumed, VAE_encode_calls_this_attempt=calls,
        seconds_this_attempt=time.monotonic() - started,
        whole_bank_recomputed_in_this_attempt=resumed == 0,
        interpretation='Latent/input engineering artifact only; no fit, reconstruction or score evidence.')
    guard(); write_once(root / 'completion.json', done, reserve=reserve)
    return _verify_bank(root, registration)


def validate_decoder_bindings(inputs, decoder):
    """The original decoder fixes both checkpoint and config, including scale."""
    payload = next(x for x in decoder['generator_assets'] if 'sdxl_turbo_fp16' in x['path'])
    require(inputs['generator'] == dict(path=payload['path'], sha256=payload['sha256']), 'Use the exact native generator weight bytes')
    config_path = PurePosixPath(decoder['generator_asset_root']) / 'sdxl_config/vae/config.json'
    require(inputs['VAE_config'] == dict(path=str(config_path), sha256=decoder['source_sha256'][str(config_path)]),
            'Original native VAE config SHA/path, including scaling factor, must be retained')


def verify_inputs(context, gate):
    gate.require_context(context, purpose='latents')
    inputs = context['stage_spec']['latent_inputs']; rows = train_rows(context['bundle'])
    decoder_path = ROOT / 'configs/ATM_public_decoder_registration_20261020_v1.json'
    require(inputs['decoder_registration'] == dict(path=str(decoder_path), sha256=DECODER_REGISTRATION_SHA),
            'Original native decoder registration required')
    require(sha(decoder_path) == DECODER_REGISTRATION_SHA, 'Original decoder registration changed')
    decoder = json.loads(decoder_path.read_bytes())
    validate_decoder_bindings(inputs, decoder)
    archive_refs = inputs['archives']
    require(len(archive_refs) == len({x['path'] for x in archive_refs})
            and {x['path'] for x in archive_refs} == {row['image_zip'] for row in rows},
            'Pin exactly all full-training image containers')
    for item in [inputs['generator'], inputs['VAE_config'], *archive_refs]:
        require(set(item) == {'path', 'sha256'} and Path(item['path']).is_file()
                and len(item['sha256']) == 64 and sha(item['path']) == item['sha256'],
                'Actual native/image container payload differs: ' + str(item.get('path')))
        gate.require_context(context, purpose='latents')
    return rows, inputs


def native_preprocess():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])


def encode_native_modes(vae, pixels, torch):
    with torch.no_grad():
        value = vae.encode(pixels).latent_dist.mode().float() * vae.config.scaling_factor
    require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'Nonfinite native VAE mode')
    return value


def execute(context, gate):
    gate.require_context(context, purpose='latents')
    gate.check_sources(context)
    require(context['source_sha256'][Path(__file__).name] == sha(__file__), 'Latent producer source changed')
    rows, inputs = verify_inputs(context, gate)
    output = Path(context['stage_output_root'])
    require(context['maximum_output_bytes'] >= 2 << 30, 'Full persistent bank plus retained metadata requires reviewed2GiB cap')
    require(not output.exists() or allocated_bytes(output) <= context['maximum_output_bytes'], 'Stage output cap exceeded')
    import numpy as np
    from brainhive_native_models_v1 import torch
    require(torch.__version__ == context['runtime']['torch_version'] and torch.cuda.is_available(), 'Actual pinned CUDA runtime required')
    torch.set_num_threads(2); torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors import safe_open
    config = json.loads(Path(inputs['VAE_config']['path']).read_bytes())
    with safe_open(inputs['generator']['path'], framework='pt', device='cpu') as source:
        weights = {key: source.get_tensor(key) for key in source.keys() if key.startswith('first_stage_model.')}
    require(bool(weights), 'Native checkpoint has no first-stage VAE')
    vae = AutoencoderKL.from_config(config)
    converted = convert_ldm_vae_checkpoint(weights, config)
    result = vae.load_state_dict(converted, strict=True)
    require(not result.missing_keys and not result.unexpected_keys
            and sum(p.numel() for p in vae.parameters()) == 83653863, 'Strict unchanged native VAE required')
    del converted, weights
    vae = vae.cuda().float().eval().requires_grad_(False); vae.enable_slicing()
    require(all(p.dtype == torch.float32 and not p.requires_grad for p in vae.parameters()), 'Native frozen FP32 VAE required')
    from PIL import Image
    preprocess = native_preprocess()
    registration = dict(schema='BrainHIVE_official_train_native_VAE_bank_registration_v1',
        actual_official_full_train_scope=True, array_rows=list(range(16540)), shape=list(SHAPE), dtype='float32',
        batch_size=4, chunk_size=256, preprocessing=PREPROCESSING,
        VAE_parameters=83653863, VAE_precision='FP32 from exact native checkpoint', VAE_strict_load=True,
        VAE_scaling_factor=float(vae.config.scaling_factor), stochastic_sampling=False,
        image_identity_sha256=hashlib.sha256(canonical(image_identity(rows))).hexdigest(), inputs=inputs,
        plan_preparation=context['bundle']['preparation'], method_lock=context['method_lock_ref'],
        admission=context['admission_ref'], source_sha256=context['source_sha256'],
        official_test_image_values_used=False, old_T_bank_reused_or_changed=False)
    archives = {}; row_by_id = {row['array_row']: row for row in rows}

    def guard():
        gate.require_context(context, purpose='latents')
        if output.exists(): require(allocated_bytes(output) <= context['maximum_output_bytes'], 'Retain output and stop at cap')
        from brainhive_functional_training_contract_v2 import resource_headroom
        require(resource_headroom()['effective_headroom_bytes'] >= context['resource_limits']['minimum_effective_CPU_headroom_bytes'],
                'Fresh actual effective CPU headroom below reviewed stage floor')

    def reserve(additional_bytes):
        guard()
        require(allocated_bytes(output) + additional_bytes <= context['maximum_output_bytes'],
                'Full actual unique-inode output plus pending write exceeds reviewed cap; no write')

    def encode(selected):
        pixels = []
        for index in selected:
            row = row_by_id[index]; archive_path = row['image_zip']
            if archive_path not in archives: archives[archive_path] = zipfile.ZipFile(archive_path)
            raw = archives[archive_path].read(row['image_member'])
            require(len(raw) == row['image_bytes'] and hashlib.sha256(raw).hexdigest() == row['image_sha256'],
                    'Actual training image member SHA/byte count differs')
            with Image.open(io.BytesIO(raw)) as image: pixels.append(preprocess(image.convert('RGB')))
        value = encode_native_modes(vae, torch.stack(pixels).cuda(), torch)
        return value.cpu().numpy()

    torch.cuda.reset_peak_memory_stats()
    try:
        result = _build_bank(output, registration, encode, guard, reserve=reserve)
        guard(); gate.check_sources(context)
        require(sha(__file__) == context['source_sha256'][Path(__file__).name], 'Producer changed during execution')
        executions = output / 'executions'
        require(not executions.is_symlink(), 'No aliased execution evidence directory')
        executions.mkdir(exist_ok=True, mode=0o700)
        actual = dict(status='ACTUAL_FULL16540_VAE_BANK_STAGE_EXECUTION_RECORDED',
            bank_completion=ref(output / 'completion.json'), producer=ref(__file__),
            runtime=context['runtime'], actual_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
            stage_execution_review=context['stage_review_ref'], wrapper_job_name=context['stage_spec']['wrapper_job_name'],
            producer_pid=os.getpid(), wrapper_pid=os.getppid(), actual_stage_complete=True,
            actual_official_fit_performed=False, image_reconstructions=0, official_scores_read=False)
        write_once(executions / ('execution-' + str(os.getpid()) + '.json'), actual, reserve=reserve)
        guard()
        return dict(result, actual_execution=actual, actual_stage_complete=True)
    finally:
        for archive in archives.values(): archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-directory', required=True, type=Path)
    parser.add_argument('--execution-review', required=True, type=Path)
    args = parser.parse_args()
    import brainhive_official_component_data_v1 as gate
    context = gate.preflight(args.plan_directory, args.execution_review, purpose='latents', entry_name=Path(__file__).name)
    result = execute(context, gate)
    print(json.dumps(dict(status=result['status'], shape=result['shape'], actual_stage_complete=True)), flush=True)


if __name__ == '__main__':
    main()
