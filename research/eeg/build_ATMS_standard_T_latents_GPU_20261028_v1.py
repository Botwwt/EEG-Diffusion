"""Rebuild a NEW, pinned standard-SDXL FP32 VAE-mode bank for all T images.

No UNet/text model is loaded. Existing bank receipts/links are never modified.
Run only as a child of run_gpu_job_v2.py; the parent owns the GPU lease.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import signal as signal_control
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
REVISION = '462165984030d82259a11f4367a4eed129e94a7b'
N = 13230
BATCH = 4
PART = 256
MAX_GPU = 20 << 30
MAX_CPU = 10 << 30
MAX_OUTPUT = 2 << 30


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256'], str(path)
    return path


def ref(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def rss_peak():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Linux VmHWM is required')


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.maximum_seconds = seconds
        self.base_cpu = rss_peak()
        self.peak_cpu = self.base_cpu
        self.peak_gpu = 0
        self.stopped = False
        self.torch = None
        self.output = None

    def check(self, stage):
        assert not self.stopped, 'Termination requested at ' + stage
        assert time.monotonic() - self.started < self.maximum_seconds, 'Time cap at ' + stage
        self.peak_cpu = max(self.peak_cpu, rss_peak())
        assert self.peak_cpu - self.base_cpu <= MAX_CPU, 'Extra CPU cap at ' + stage
        if self.torch is not None and self.torch.cuda.is_initialized():
            self.peak_gpu = max(self.peak_gpu, self.torch.cuda.max_memory_allocated())
            assert self.peak_gpu <= MAX_GPU, 'GPU allocated cap at ' + stage
        if self.output is not None:
            size = sum(max(p.stat().st_size, getattr(p.stat(), 'st_blocks', 0) * 512)
                       for p in self.output.iterdir() if p.is_file())
            assert size <= MAX_OUTPUT, 'Output cap at ' + stage
        return dict(seconds=time.monotonic() - self.started,
                    peak_extra_CPU_bytes=self.peak_cpu - self.base_cpu,
                    peak_GPU_bytes=self.peak_gpu)


def wrapper_proof():
    raw = (Path('/proc') / str(os.getppid()) / 'cmdline').read_bytes().split(b'\0')
    argv = [item.decode() for item in raw if item]
    assert any(Path(item).name == 'run_gpu_job_v2.py' for item in argv), 'Use the unchanged finite GPU wrapper'
    return dict(wrapper_PID=os.getppid(), parent_argv=argv)


def validate_spec(path):
    spec = read(path)
    assert spec['status'] == 'X01_STANDARD_NATIVE_T_VAE_MODE_REBUILD_LOCKED_BEFORE_GPU'
    assert spec.get('seed', 1729) == 1729
    pins = {}
    for item in spec['source_pins']:
        resolved = str(checked(item).resolve())
        assert resolved not in pins
        pins[resolved] = item['sha256']
    for key in ('authorization', 'split', 'image_manifest', 'snapshot_completion'):
        item = spec[key]
        assert pins.get(str(checked(item).resolve())) == item['sha256'], key
    for filename in (Path(__file__).name, 'profile_standard_generator.py', 'build_idea_training_latents_v1.py'):
        assert pins.get(str((ROOT / filename).resolve())) == sha(ROOT / filename), filename
    if 'historical_completion' in spec:
        checked(spec['historical_completion'])
    for key, expected in [('maximum_GPU_bytes', MAX_GPU), ('maximum_extra_CPU_bytes', MAX_CPU),
                          ('maximum_output_bytes', MAX_OUTPUT), ('batch_size', BATCH), ('part_size', PART)]:
        assert spec.get(key, expected) == expected, key
    return spec


def inputs(spec):
    split = read(checked(spec['split']))
    rows = split['image_rows']['T']
    assert len(rows) == N and len(set(rows)) == N and all(type(row) is int for row in rows)
    denied = set(split['image_rows']['D']) | set(split['image_rows']['V'])
    assert not set(rows) & denied
    with checked(spec['image_manifest']).open(encoding='utf-8', newline='') as stream:
        entries = [item for item in csv.DictReader(stream) if item['dataset_split'] == 'train']
    metadata = {int(item['array_row']): item for item in entries}
    assert len(entries) == len(metadata) == 16540
    assert {metadata[row]['concept_id'] for row in rows} == set(split['train_concepts']['T'])
    for row in rows:
        item = metadata[row]
        assert len(item['image_sha256']) == 64 and item['image_member']
    return rows, metadata


def load_exact_vae(spec, native, budget):
    torch = native.torch
    snapshot = read(checked(spec['snapshot_completion']))
    assert snapshot['status'] == 'COMPLETE_OFFICIAL_SNAPSHOT' and snapshot['revision'] == REVISION
    model_root = Path(spec.get('model_root', str(native.MODEL))).resolve()
    assert model_root == native.MODEL.resolve(), 'Match the original standard-generator runtime model root'
    vae_root = model_root / 'vae'
    required = {'config.json', 'diffusion_pytorch_model.fp16.safetensors'}
    selected = {}
    for item in snapshot['files']:
        candidate = Path(item['path'])
        if candidate.parent.resolve() == vae_root and candidate.name in required:
            assert candidate.name not in selected
            checked(item)
            selected[candidate.name] = item
            budget.check('VERIFY_VAE_FILE_' + candidate.name)
    assert set(selected) == required, 'Exact pinned standard VAE config and fp16 variant required'
    # The original runtime explicitly loads this fp16-file variant into FP32.
    vae, info = native.AutoencoderKL.from_pretrained(
        vae_root, local_files_only=True, variant='fp16', torch_dtype=torch.float32,
        output_loading_info=True)
    assert not any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
    vae.eval().requires_grad_(False)
    assert all(parameter.dtype == torch.float32 and not parameter.requires_grad for parameter in vae.parameters())
    assert float(vae.config.scaling_factor) > 0
    vae.to(device='cuda', dtype=torch.float32)
    vae.enable_slicing()
    assert vae.use_slicing
    torch.cuda.synchronize()
    budget.check('EXACT_STANDARD_FP32_VAE_LOADED')
    return vae, dict(generator='stabilityai/stable-diffusion-xl-base-1.0', revision=REVISION,
        snapshot_completion=spec['snapshot_completion'], loaded_subset='vae only; UNet/text/IP not loaded',
        original_profile_source=ref(ROOT / 'profile_standard_generator.py'), variant='fp16',
        actual_parameter_dtype='float32', load_infos=info, VAE_files=selected,
        VAE_parameters=sum(parameter.numel() for parameter in vae.parameters()),
        VAE_scaling_factor=float(vae.config.scaling_factor), slicing=True)


def pixels_for(selected, metadata, archives, image_records, native, budget):
    result = []
    for row in selected:
        item = metadata[row]
        if item['image_zip'] not in archives:
            archives[item['image_zip']] = zipfile.ZipFile(item['image_zip'])
        raw = archives[item['image_zip']].read(item['image_member'])
        assert hashlib.sha256(raw).hexdigest() == item['image_sha256'], row
        with native.Image.open(io.BytesIO(raw)) as image:
            array = native.np.array(image.convert('RGB').resize((512, 512), native.Image.Resampling.BICUBIC), copy=True)
        assert array.shape == (512, 512, 3) and array.dtype == native.np.uint8
        result.append(native.torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1)
        if image_records is not None:
            image_records.append(dict(array_row=row, stimulus_id=item['stimulus_id'],
                image_zip=item['image_zip'], image_member=item['image_member'],
                image_sha256=item['image_sha256'], original_bytes=len(raw)))
        budget.check('READ_SHA_AND_PREPROCESS_T_ONLY')
    return native.torch.stack(result)


def execute(args, budget):
    spec_path = Path(args.spec).resolve()
    spec = validate_spec(spec_path)
    parent = wrapper_proof()
    rows, metadata = inputs(spec)
    output = Path(args.output).absolute()
    assert not output.exists() and not output.is_symlink(), 'Use a separate new output; never replace old receipts/links'
    assert '/dev/shm/' not in str(output.resolve()), 'Persistent output required'
    if 'historical_completion' in spec:
        assert output.resolve() != Path(spec['historical_completion']['path']).resolve().parent
    output.parent.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(output.parent).free >= MAX_OUTPUT + (1 << 30), 'Need bank cap plus 1GiB free margin'
    output.mkdir()
    budget.output = output
    write(output / 'input_provenance.json', dict(spec=ref(spec_path), source_pins=spec['source_pins'],
        wrapper=parent, authorization=spec['authorization'], partition='T', array_rows=rows,
        image_manifest=spec['image_manifest'], split=spec['split'],
        historical_completion=spec.get('historical_completion'),
        historical_missing_parts=spec.get('historical_missing_parts'),
        old_bank_receipt_or_links_modified=False, new_bank_claims_old_part_SHA_identity=False,
        preprocessing='PIL RGB bicubic resize512; pixels /127.5 -1; FP32 VAE posterior MODE times scaling_factor',
        maximum_seconds=args.max_seconds, maximum_GPU_bytes=MAX_GPU,
        maximum_extra_CPU_bytes=MAX_CPU, maximum_output_bytes=MAX_OUTPUT))
    # Importing this module supplies the identical isolated runtime and class.
    # It does not call its full pipeline loader.
    import profile_standard_generator as native
    torch = native.torch
    budget.torch = torch
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    vae, model_info = load_exact_vae(spec, native, budget)
    write(output / 'model_load.json', model_info)
    archives, parts, image_records = {}, [], []
    saved_replay = None
    try:
        with torch.inference_mode():
            for start in range(0, N, PART):
                stop = min(start + PART, N)
                latents = []
                for offset in range(start, stop, BATCH):
                    chosen = rows[offset:min(offset + BATCH, stop)]
                    pixels = pixels_for(chosen, metadata, archives, image_records, native, budget)
                    value = vae.encode(pixels.cuda()).latent_dist.mode() * vae.config.scaling_factor
                    assert value.dtype == torch.float32 and value.shape == (len(chosen), 4, 64, 64)
                    assert torch.isfinite(value).all()
                    latents.append(value.cpu())
                    del value, pixels
                    torch.cuda.synchronize()
                    budget.check('FP32_MODE_T_BATCH4')
                latent = torch.cat(latents)
                if saved_replay is None:
                    saved_replay = latent[:2].clone()
                path = output / f'{start:06d}.pt'
                temporary = path.with_name(path.name + '.tmp')
                torch.save(dict(array_rows=rows[start:stop], latents=latent, recipe_sha256=sha(spec_path)), temporary)
                os.replace(temporary, path)
                parts.append(dict(path=str(path.resolve()), sha256=sha(path), start=start, stop=stop,
                                  array_rows=rows[start:stop]))
                event = dict(status='RUNNING_STANDARD_VAE_T_BANK', completed=stop, expected=N, **budget.check('PART_WRITTEN'))
                write(output / 'progress.json', event)
                with (output / 'progress.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
                del latent, latents
            assert len(image_records) == N and [item['array_row'] for item in image_records] == rows
            write(output / 'all_T_image_SHA_verification.json', dict(status='ALL13230_T_IMAGE_BYTES_SHA_VERIFIED',
                images=image_records, V_or_D_or_official_test_images_read=False))
            # Independent device replay of the first two predetermined T images;
            # re-read the actual image bytes and verify their original SHA again.
            vae.to(device='cpu', dtype=torch.float32)
            torch.cuda.empty_cache()
            pixels = pixels_for(rows[:2], metadata, archives, None, native, budget)
            replay = (vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor).float()
            assert torch.isfinite(replay).all()
            maximum = float((replay - saved_replay).abs().max())
            relative = float((replay - saved_replay).norm() / saved_replay.norm().clamp_min(1e-12))
            assert maximum <= 2e-3 and relative <= 2e-4, (maximum, relative)
            torch.save(dict(array_rows=rows[:2], GPU_latents=saved_replay, CPU_latents=replay), output / 'two_T_CPU_replay.pt')
            replay_info = dict(array_rows=rows[:2], source_SHA_rechecked=True, CPU_actual_FP32_VAE_forward=True,
                maximum_absolute_error=maximum, relative_L2_error=relative,
                maximum_absolute_tolerance=2e-3, relative_L2_tolerance=2e-4,
                comparison_is_tolerance_based_not_bitwise=True, saved=ref(output / 'two_T_CPU_replay.pt'))
            write(output / 'two_T_CPU_replay.json', replay_info)
            assert all(not parameter.requires_grad and parameter.dtype == torch.float32 for parameter in vae.parameters())
        coverage = []
        for part in parts:
            assert sha(part['path']) == part['sha256']
            bank = torch.load(part['path'], map_location='cpu', weights_only=True, mmap=True)
            assert bank['array_rows'] == part['array_rows'] == rows[part['start']:part['stop']]
            assert bank['recipe_sha256'] == sha(spec_path)
            assert bank['latents'].dtype == torch.float32 and bank['latents'].shape == (part['stop'] - part['start'], 4, 64, 64)
            assert torch.isfinite(bank['latents']).all()
            coverage.extend(bank['array_rows'])
            del bank
            budget.check('FINAL_ALL_PARTS_REOPEN_COVERAGE')
        assert coverage == rows and len(parts) == 52
        for item in spec['source_pins']:
            checked(item)
        for item in model_info['VAE_files'].values():
            checked(item)
        assert sha(spec_path) == read(output / 'input_provenance.json')['spec']['sha256']
        report = dict(status='COMPLETED_ALL13230_T_IMAGE_LATENTS', partition='T', unique_images=N,
            array_rows=rows, dtype='float32', shape=[N, 4, 64, 64], parts=parts,
            recipe_sha256=sha(spec_path), input_provenance_sha256=sha(output / 'input_provenance.json'),
            generator_revision=REVISION, VAE_scaling_factor=model_info['VAE_scaling_factor'],
            V_or_D_or_official_test_images_read=False, model_load=ref(output / 'model_load.json'),
            all_T_image_SHA_verification=ref(output / 'all_T_image_SHA_verification.json'),
            two_T_CPU_replay=replay_info, old_bank_receipt_or_links_modified=False,
            old_part_SHA_identity_claimed=False, batch_size=BATCH, part_size=PART,
            images_generated=0, adapter_trained=False, training_updates=0,
            UNet_loaded=False, UNet_forward_calls=0, VAE_decode_calls=0,
            **budget.check('BEFORE_COMPLETION'))
        write(output / 'completion.json', report)
        budget.check('FINAL_COMPLETION_WRITTEN')
        print(json.dumps({key: value for key, value in report.items() if key not in ('array_rows', 'parts', 'two_T_CPU_replay')}), flush=True)
    finally:
        for archive in archives.values():
            archive.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    args = parser.parse_args()
    assert 60 <= args.max_seconds <= 1800
    budget = Budget(args.max_seconds)
    signal_control.signal(signal_control.SIGTERM, lambda *_: setattr(budget, 'stopped', True))
    signal_control.signal(signal_control.SIGINT, lambda *_: setattr(budget, 'stopped', True))
    try:
        execute(args, budget)
    except BaseException as exc:
        if budget.output is not None:
            write(budget.output / 'failure.json', dict(status='FAILED_NEW_STANDARD_T_LATENT_BANK_PRESERVED',
                error=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic() - budget.started,
                completion_is_valid=False, old_bank_receipt_or_links_modified=False,
                images_generated=0, training_updates=0))
            # A failure after final-write must never leave a success marker.
            done = budget.output / 'completion.json'
            if done.exists():
                os.replace(done, budget.output / 'failed_completion_preserved.json')
        raise


if __name__ == '__main__':
    main()
