"""Rebuild the original T-only latent cache and require every old chunk/file SHA.

No new model, supervision, image reconstruction, V scoring, or recipe changes.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from brainhive_native_models_v1 import ROOT, WORK, sha, write, torch
import numpy as np

OUT = ROOT / 'audit/runtime_T_latents_exact_restored_20261022_v1.json'
PROGRESS = ROOT / 'audit/runtime_T_latent_restore_progress_20261022_v1.json'


def main():
    assert b'run_gpu_job_v2.py' in (Path('/proc') / str(os.getppid()) / 'cmdline').read_bytes()
    assert not OUT.exists()
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip(), 'Preserve any existing external GPU job'
    torch.set_num_threads(2); torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    historical_path = WORK / 'audit/component_T_latent_bank_v2.json'
    historical_sha = sha(historical_path)
    historical = json.loads(historical_path.read_text())
    original_source = ROOT / 'build_BrainHIVE_component_T_latents_v2.py'
    assert sha(original_source) == historical['source_sha256']
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    assert recipe['input_sha256'][str(historical_path)] == historical_sha
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    ids = protocol['image_rows']['T']
    assert ids == historical['array_rows'] and len(ids) == 13230
    assert set(ids).isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    target = Path(historical['path'])
    assert target.is_relative_to('/dev/shm') and recipe['input_sha256'][str(target)] == historical['sha256']
    assert historical['shape'] == [13230, 4, 64, 64] and historical['dtype'] == 'float32'
    public = json.loads((ROOT / 'audit/runtime_public_restore_20261022_v1.json').read_text())
    assert public['status'] == 'ALL19_PINNED_PUBLIC_RUNTIME_ASSETS_NATIVE_SHA_VERIFIED'
    for file in public['files']:
        assert sha(file['path']) == file['sha256']
    if target.exists():
        assert sha(target) == historical['sha256']
        write(OUT, dict(status='ORIGINAL_T_LATENT_CACHE_ALREADY_PRESENT_EXACT_SHA', path=str(target),
            sha256=historical['sha256'], historical_receipt_sha256=historical_sha, GPU_calls=0))
        return
    assert shutil.disk_usage('/dev/shm').free > 13230 * 4 * 64 * 64 * 4 + (512 << 20)
    registration = json.loads((ROOT / 'configs/ATM_public_decoder_registration_20261020_v1.json').read_text())
    payload = next(x for x in registration['generator_assets'] if 'sdxl_turbo_fp16' in x['path'])
    assert sha(payload['path']) == historical['registration']['generator_sha256']
    vae_config_path = Path(registration['generator_asset_root']) / 'sdxl_config/vae/config.json'
    assert sha(vae_config_path) == historical['registration']['VAE_config_sha256']
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors import safe_open
    config = json.loads(vae_config_path.read_text())
    with safe_open(payload['path'], framework='pt', device='cpu') as source:
        original = {key: source.get_tensor(key) for key in source.keys() if key.startswith('first_stage_model.')}
    converted = convert_ldm_vae_checkpoint(original, config)
    vae = AutoencoderKL.from_config(config)
    result = vae.load_state_dict(converted, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert sum(p.numel() for p in vae.parameters()) == 83653863
    del converted, original
    vae = vae.cuda().float().eval().requires_grad_(False); vae.enable_slicing()
    assert float(vae.config.scaling_factor) == historical['registration']['VAE_scaling_factor']
    from PIL import Image
    from torchvision import transforms
    preprocess = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
    rows = {int(row['array_row']): row for row in csv.DictReader((ROOT / 'audit/image_manifest.csv').open())
            if row['dataset_split'] == 'train'}
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged = target.with_name(target.name + '.restore.partial')
    state = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else dict(completed=0, chunks=[])
    expected_chunks = historical['exact_chunk_coverage']
    assert [(p['start'], p['stop']) for p in expected_chunks] == [
        (start, min(start + 256, 13230)) for start in range(0, 13230, 256)]
    if staged.exists():
        array = np.lib.format.open_memmap(staged, mode='r+')
        assert tuple(array.shape) == (13230, 4, 64, 64) and array.dtype == np.float32
        for saved, expected in zip(state['chunks'], expected_chunks):
            assert saved == expected
            assert hashlib.sha256(array[saved['start']:saved['stop']].tobytes()).hexdigest() == saved['sha256']
        assert state['completed'] == (state['chunks'][-1]['stop'] if state['chunks'] else 0)
    else:
        assert state['completed'] == 0
        array = np.lib.format.open_memmap(staged, mode='w+', dtype=np.float32, shape=(13230, 4, 64, 64))
    archives = {}; forwards = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for expected in expected_chunks[len(state['chunks']):]:
            start, stop = expected['start'], expected['stop']
            for offset in range(start, stop, 4):
                pixels = []
                for index in ids[offset:min(offset + 4, stop)]:
                    row = rows[index]
                    if row['image_zip'] not in archives:
                        archives[row['image_zip']] = zipfile.ZipFile(row['image_zip'])
                    raw = archives[row['image_zip']].read(row['image_member'])
                    assert hashlib.sha256(raw).hexdigest() == row['image_sha256']
                    with Image.open(io.BytesIO(raw)) as image:
                        pixels.append(preprocess(image.convert('RGB')))
                value = vae.encode(torch.stack(pixels).cuda()).latent_dist.mode().float() * vae.config.scaling_factor
                forwards += 1
                assert torch.isfinite(value).all()
                array[offset:offset + len(pixels)] = value.cpu().numpy()
            array.flush()
            assert hashlib.sha256(array[start:stop].tobytes()).hexdigest() == expected['sha256'], 'Preserve mismatch; never relax the old chunk identity'
            state['chunks'].append(expected); state['completed'] = stop
            state['seconds_this_attempt'] = time.monotonic() - started
            write(PROGRESS, state)
            print(json.dumps(dict(completed=stop, expected=13230, seconds=state['seconds_this_attempt'])), flush=True)
    for archive in archives.values():
        archive.close()
    array.flush(); del array
    assert state['chunks'] == expected_chunks
    assert sha(staged) == historical['sha256']
    assert not target.exists()
    staged.rename(target)
    assert sha(historical_path) == historical_sha
    write(OUT, dict(status='ALL13230_T_LATENTS_ORIGINAL_CHUNKS_AND_FILE_SHA_RESTORED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), path=str(target), sha256=sha(target),
        historical_receipt_sha256=historical_sha, historical_receipt_unchanged=True,
        source_sha256=sha(__file__), original_builder_source_sha256=sha(original_source),
        T_images=13230, V_or_official_image_values_used=False, VAE_encode_forwards_this_attempt=forwards,
        UNet_forwards=0, training_updates=0, images_generated=0,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic() - started,
        RAM_is_volatile=True, formal_functional_training_started=False))


if __name__ == '__main__':
    main()
