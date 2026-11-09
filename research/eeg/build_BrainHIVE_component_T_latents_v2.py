"""Complete T, VAE only, fixed native preprocessing, resumable mmap blocks."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from brainhive_native_models_v1 import *
import numpy as np

RAM=Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2/component_T_latents_v2')


def main():
    assert b'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes()
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    done=WORK/'audit/component_T_latent_bank_v2.json';assert not done.exists()
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text());ids=protocol['image_rows']['T'];assert len(ids)==13230
    assert set(ids).isdisjoint(protocol['image_rows']['D']+protocol['image_rows']['V'])
    rows={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    cfg=json.loads((ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json').read_text())
    payload=next(x for x in cfg['generator_assets'] if 'sdxl_turbo_fp16' in x['path']);assert sha(payload['path'])==payload['sha256']
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors import safe_open
    vae_config_path=Path(cfg['generator_asset_root'])/'sdxl_config/vae/config.json'
    vae_config=json.loads(vae_config_path.read_text())
    with safe_open(payload['path'],framework='pt',device='cpu') as f:
        keys=[k for k in f.keys() if k.startswith('first_stage_model.')];assert keys
        original={k:f.get_tensor(k) for k in keys}
    converted=convert_ldm_vae_checkpoint(original,vae_config)
    vae=AutoencoderKL.from_config(vae_config)
    coverage=vae.load_state_dict(converted,strict=True)
    assert not coverage.missing_keys and not coverage.unexpected_keys
    assert sum(p.numel() for p in vae.parameters())==83653863
    del converted,original
    vae=vae.cuda().float().eval().requires_grad_(False);vae.enable_slicing()
    from PIL import Image
    from torchvision import transforms
    preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    RAM.mkdir(mode=0o700,exist_ok=True);path=RAM/'native_VAE_T_modes.npy'
    registration=dict(array_rows=ids,shape=[13230,4,64,64],dtype='float32',batch_size=4,
        preprocessing='Unchanged author Resize512 BILINEAR, CenterCrop512, RGB[-1,1]',
        VAE_scaling_factor=float(vae.config.scaling_factor),generator_sha256=payload['sha256'],
        VAE_config_sha256=sha(vae_config_path),VAE_parameters=83653863,VAE_strict_load=True,
        VAE_precision='FP32 from exact native checkpoint',source_sha256=sha(__file__),D_V_official_test_values_used=False)
    reg=RAM/'registration.json'
    if reg.exists():assert json.loads(reg.read_text())==registration
    else:write(reg,registration)
    progress=WORK/'audit/component_T_latent_progress_v2.json'
    state=json.loads(progress.read_text()) if progress.exists() else dict(completed=0,chunks=[])
    if path.exists():
        array=np.lib.format.open_memmap(path,mode='r+');assert array.shape==(13230,4,64,64) and array.dtype==np.float32
        for part in state['chunks']:
            assert hashlib.sha256(array[part['start']:part['stop']].tobytes()).hexdigest()==part['sha256']
    else:
        assert state['completed']==0 and shutil.disk_usage('/dev/shm').free>13230*4*64*64*4+(512<<20)
        array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(13230,4,64,64))
    archives={};started=time.monotonic();start_at=state['completed'];torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for chunk_start in range(start_at,len(ids),256):
            chunk_stop=min(chunk_start+256,len(ids))
            for start in range(chunk_start,chunk_stop,4):
                selected=ids[start:min(start+4,chunk_stop)];pixels=[]
                for i in selected:
                    row=rows[i]
                    if row['image_zip'] not in archives:archives[row['image_zip']]=zipfile.ZipFile(row['image_zip'])
                    raw=archives[row['image_zip']].read(row['image_member']);assert hashlib.sha256(raw).hexdigest()==row['image_sha256']
                    with Image.open(io.BytesIO(raw)) as image:pixels.append(preprocess(image.convert('RGB')))
                value=vae.encode(torch.stack(pixels).cuda()).latent_dist.mode().float()*vae.config.scaling_factor
                assert torch.isfinite(value).all();array[start:start+len(selected)]=value.cpu().numpy()
            array.flush();state['chunks'].append(dict(start=chunk_start,stop=chunk_stop,
                sha256=hashlib.sha256(array[chunk_start:chunk_stop].tobytes()).hexdigest()))
            state.update(completed=chunk_stop,expected=13230,source_sha256=sha(__file__),seconds=time.monotonic()-started,
                peak_GPU_bytes=torch.cuda.max_memory_allocated())
            write(progress,state);print(json.dumps({k:v for k,v in state.items() if k!='chunks'}),flush=True)
    for archive in archives.values():archive.close()
    array.flush();del array
    write(done,dict(status='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS',path=str(path),sha256=sha(path),
        format='numpy_npy_float32_C_contiguous_v2',array_rows=ids,shape=[13230,4,64,64],dtype='float32',
        registration_sha256=sha(reg),registration=registration,source_sha256=sha(__file__),
        exact_chunk_coverage=state['chunks'],RAM_is_volatile=True,D_V_official_test_values_used=False,
        VAE_only_no_UNet_or_text_encoder_loaded=True,VAE_strict_load=True,
        prior_interruption_preserved='v1 was SIGKILL without Python traceback; cgroup recorded no oom_kill; cause unconfirmed.',
        seconds_this_attempt=time.monotonic()-started))
    print('COMPLETE_ALL_ALLOWED_T_NATIVE_VAE_MMAP_BANK',flush=True)


if __name__=='__main__':main()
