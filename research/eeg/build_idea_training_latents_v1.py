"""Actual complete T image latent bank; no V, D or benchmark image reads."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from profile_standard_generator import load_pipeline,torch,ROOT
from PIL import Image
import numpy as np


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());started=time.monotonic()
    assert plan['version']=='idea_interface_training_recipe_v1' and plan['seed']==1729
    for source,digest in plan['source_sha256'].items():assert sha(ROOT/source)==digest,source
    parent=Path('/proc')/str(os.getppid())/'cmdline'
    assert parent.exists() and str(ROOT/'run_gpu_job.py').encode() in parent.read_bytes()
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text());rows=protocol['image_rows']['T']
    assert len(rows)==13230 and not set(rows)&set(protocol['image_rows']['D']+protocol['image_rows']['V'])
    images={int(x['array_row']):x for x in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    out=Path(plan['latent_bank_root']);out.mkdir(parents=True,exist_ok=True)
    lock=dict(recipe_sha256=sha(plan_path),array_rows=rows,partition='T',resolution=512,
        preprocessing='PIL RGB bicubic resize512, pixels /127.5 -1; frozen FP32 VAE posterior MODE times scaling_factor',
        source_images=[dict(array_row=i,stimulus_id=images[i]['stimulus_id'],image_sha256=images[i]['image_sha256']) for i in rows])
    if (out/'input_provenance.json').exists():assert json.loads((out/'input_provenance.json').read_text())==lock
    else:write(out/'input_provenance.json',lock)
    assert not (out/'completion.json').exists(),'Preserve completed bank; reuse its verified receipt'
    pipe,infos,model_receipt,prompts=load_pipeline();archives={};parts=[]
    with torch.inference_mode():
        for start in range(0,len(rows),256):
            stop=min(start+256,len(rows));path=out/f'{start:06d}.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                assert saved['array_rows']==rows[start:stop] and saved['recipe_sha256']==sha(plan_path)
                latent=saved['latents'];assert latent.shape==(stop-start,4,64,64)
            else:
                latents=[]
                for offset in range(start,stop,4):
                    selected=rows[offset:min(offset+4,stop)];pixels=[]
                    for i in selected:
                        item=images[i]
                        if item['image_zip'] not in archives:archives[item['image_zip']]=zipfile.ZipFile(item['image_zip'])
                        raw=archives[item['image_zip']].read(item['image_member']);assert hashlib.sha256(raw).hexdigest()==item['image_sha256']
                        with Image.open(io.BytesIO(raw)) as image:a=np.array(image.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),copy=True)
                        pixels.append(torch.from_numpy(a).permute(2,0,1).float()/127.5-1)
                    value=pipe.vae.encode(torch.stack(pixels).cuda()).latent_dist.mode()*pipe.vae.config.scaling_factor
                    assert torch.isfinite(value).all();latents.append(value.float().cpu())
                latent=torch.cat(latents);temporary=path.with_name(path.name+'.tmp')
                torch.save(dict(array_rows=rows[start:stop],latents=latent,recipe_sha256=sha(plan_path)),temporary);os.replace(temporary,path)
            assert latent.dtype==torch.float32 and torch.isfinite(latent).all()
            parts.append(dict(path=str(path),sha256=sha(path),start=start,stop=stop,array_rows=rows[start:stop]))
            write(out/'progress.json',dict(status='RUNNING',completed=stop,expected=13230,seconds=time.monotonic()-started))
            print(json.dumps(dict(stage='complete_T_VAE_latents',completed=stop,expected=13230,seconds=time.monotonic()-started)),flush=True)
    for archive in archives.values():archive.close()
    report=dict(status='COMPLETED_ALL13230_T_IMAGE_LATENTS',partition='T',unique_images=13230,array_rows=rows,
        dtype='float32',shape=[13230,4,64,64],parts=parts,recipe_sha256=sha(plan_path),
        input_provenance_sha256=sha(out/'input_provenance.json'),generator_revision=model_receipt['revision'],
        VAE_scaling_factor=float(pipe.vae.config.scaling_factor),V_or_D_or_official_test_images_read=False,
        seconds=time.monotonic()-started,images_generated=0,adapter_trained=False)
    write(out/'completion.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['parts','array_rows']}),flush=True)


if __name__=='__main__':main()
