"""Complete allowed T latents in the native frozen Turbo VAE, never D/V/test."""
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
from brainhive_component_generator_v1 import load_native

RAM=Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2/component_T_latents_v1')


def main():
    assert b'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes()
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    done=WORK/'audit/component_T_latent_bank_v1.json';assert not done.exists()
    contract=WORK/'data/component_input_contract_v1.json';assert json.loads(contract.read_text())['training_queries']==132300
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text());ids=protocol['image_rows']['T'];assert len(ids)==13230
    assert set(ids).isdisjoint(protocol['image_rows']['D']+protocol['image_rows']['V'])
    rows={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    assert shutil.disk_usage('/dev/shm').free>len(ids)*4*64*64*4+(512<<20)
    RAM.mkdir(mode=0o700,exist_ok=False)
    pipe,adapter,projector,prompts,receipt=load_native(training=True)
    from PIL import Image
    from torchvision import transforms
    preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    write(RAM/'registration.json',dict(array_rows=ids,partition='T',unique_images=13230,
        preprocessing='Author train_prior Resize512 BILINEAR, CenterCrop512, RGB[-1,1]',
        latent='Frozen native Turbo VAE FP32 posterior MODE * actual scaling_factor; FP32 stored',
        source_sha256=sha(__file__),generator=receipt,D_V_official_test_values_used=False))
    latent=[];images=[];archives={};started=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(ids),4):
            selected=ids[start:start+4];pixels=[]
            for i in selected:
                row=rows[i]
                if row['image_zip'] not in archives:archives[row['image_zip']]=zipfile.ZipFile(row['image_zip'])
                raw=archives[row['image_zip']].read(row['image_member'])
                assert hashlib.sha256(raw).hexdigest()==row['image_sha256']
                with Image.open(io.BytesIO(raw)) as image:pixels.append(preprocess(image.convert('RGB')))
                images.append(dict(array_row=i,stimulus_id=row['stimulus_id'],image_sha256=row['image_sha256']))
            value=pipe.vae.encode(torch.stack(pixels).cuda()).latent_dist.mode().float()*pipe.vae.config.scaling_factor
            assert torch.isfinite(value).all();latent.append(value.cpu())
            if (start+len(selected))%512==0 or start+len(selected)==len(ids):
                event=dict(completed=start+len(selected),expected=len(ids),seconds=time.monotonic()-started)
                write(WORK/'audit/component_T_latent_progress_v1.json',event);print(json.dumps(event),flush=True)
    for archive in archives.values():archive.close()
    tensor=torch.cat(latent);assert tensor.shape==(13230,4,64,64) and tensor.dtype==torch.float32
    path=RAM/'native_VAE_T_modes.pt';save_tensor(path,dict(array_rows=torch.tensor(ids),latents=tensor,partition='T',
        preprocessing_registration_sha256=sha(RAM/'registration.json')))
    write(done,dict(status='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS',path=str(path),sha256=sha(path),
        shape=list(tensor.shape),dtype='torch.float32',source_sha256=sha(__file__),
        registration_sha256=sha(RAM/'registration.json'),source_images=images,
        VAE_scaling_factor=float(pipe.vae.config.scaling_factor),generator=receipt,
        RAM_is_volatile=True,restoration='Rebuild with this frozen source and verify SHA against this receipt before resuming training.',
        D_V_official_test_values_used=False,training_candidate_parameters=False,seconds=time.monotonic()-started))
    print('COMPLETE_ALLOWED_T_NATIVE_LATENT_BANK',flush=True)


if __name__=='__main__':main()
