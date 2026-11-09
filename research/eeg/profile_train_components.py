"""S0 real-shape prior/native IP training profile; updated weights are discarded."""
from pathlib import Path
import hashlib
import io
import json
import sys
import time
import traceback
import zipfile
from profile_standard_generator import load_pipeline, torch, ROOT

SOURCE=Path('/root/atm_official_20261016')
sys.path.insert(0,str(SOURCE/'Generation'))
from diffusion_prior import DiffusionPriorUNet
from diffusers import DDPMScheduler
import torch.nn.functional as F
from PIL import Image
import numpy as np


def main():
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1729)
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    rows=sorted(protocol['image_rows']['T'])
    f=torch.load('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt',weights_only=True,mmap=True)
    target=f[rows[:1024]].clone().cuda();condition=target.clone()
    prior=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().train()
    opt=torch.optim.Adam(prior.parameters(),lr=1e-3)
    schedule=DDPMScheduler();times=[]
    for step in range(25):
        torch.cuda.synchronize();tick=time.monotonic()
        noise=torch.randn_like(target);t=torch.randint(0,1000,(len(target),),device='cuda')
        z=schedule.add_noise(target,noise,t)
        loss=F.mse_loss(prior(z,t,condition),noise)
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(prior.parameters(),1.,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize();times.append(time.monotonic()-tick)
    prior_record=dict(batch=1024,updates=25,steady_seconds_per_step=sum(times[5:])/20,
        parameters=sum(p.numel() for p in prior.parameters()),loss=float(loss),gradient_norm_preclip=float(norm),
        caveat='S0 condition uses true visual vector solely for shape/timing; no EEG result')
    del prior,opt,target,condition,z,noise;torch.cuda.empty_cache()
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline()
    # Train the original native image projector and every IP key/value projection.
    selected=[]
    for name,p in pipe.unet.named_parameters():
        if name.startswith('encoder_hid_proj.') or '.to_k_ip.' in name or '.to_v_ip.' in name:
            p.requires_grad_(True);p.data=p.data.float();selected.append((name,p))
    assert selected and any('to_k_ip' in n for n,p in selected)
    pipe.unet.train();pipe.unet.enable_gradient_checkpointing()
    opt=torch.optim.AdamW([p for n,p in selected],lr=1e-4,weight_decay=.01)
    import csv
    manifest={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    print('MANIFEST_KEYS',next(iter(manifest.values())).keys(),flush=True)
    zpath=Path('/root/autodl-tmp/mced_research_bundle/external/things_eeg2_direct_v1/training_images.zip')
    # Exact audited ZIP member; original resolution is converted uniformly for training.
    with zipfile.ZipFile(zpath) as archive:
        images=[]
        for row in rows[:4]:
            m=manifest[row];member=m['image_member']
            raw=archive.read(member);assert hashlib.sha256(raw).hexdigest()==m['image_sha256']
            im=Image.open(io.BytesIO(raw)).convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
            images.append(torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/127.5-1)
    pixels=torch.stack(images).cuda()
    with torch.no_grad():
        latent=pipe.vae.encode(pixels).latent_dist.mode()*pipe.vae.config.scaling_factor
    del pixels
    schedule=DDPMScheduler.from_config(pipe.scheduler.config)
    time_ids=pipe._get_add_time_ids((512,512),(0,0),(512,512),dtype=prompt.dtype,
        text_encoder_projection_dim=pool.shape[-1]).cuda()
    records=[]
    for batch in (2,4):
        torch.cuda.reset_peak_memory_stats();measure=[]
        try:
            for step in range(6):
                torch.cuda.synchronize();tick=time.monotonic()
                noise=torch.randn_like(latent[:batch]);t=torch.randint(0,1000,(batch,),device='cuda')
                x=schedule.add_noise(latent[:batch],noise,t)
                cond=f[rows[:batch]].clone().cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=pipe.unet(x.to(torch.bfloat16),t,encoder_hidden_states=prompt.repeat(batch,1,1),
                        added_cond_kwargs=dict(text_embeds=pool.repeat(batch,1),time_ids=time_ids.repeat(batch,1),
                                              image_embeds=[cond[:,None,:]]),return_dict=False)[0]
                    loss=F.mse_loss(prediction.float(),noise)
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for n,p in selected],1.,error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize();measure.append(time.monotonic()-tick)
            record=dict(batch=batch,status='PASSED',updates=6,steady_seconds_per_step=sum(measure[2:])/4,
                peak_cuda_allocated=torch.cuda.max_memory_allocated(),loss=float(loss),
                gradient_norm_preclip=float(norm),gradient_parameter_names=[n for n,p in selected if p.grad is not None])
        except torch.cuda.OutOfMemoryError as exc:
            opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            record=dict(batch=batch,status='OUT_OF_MEMORY',message=str(exc))
        records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='gradient_parameter_names'}),flush=True)
    report=dict(status='S0_PROFILE_COMPLETED',formal_training=False,discard_updated_weights=True,
        prior=prior_record,adapter_cases=records,adapter_trainable_parameters=sum(p.numel() for n,p in selected),
        adapter_trainable_names=[n for n,p in selected],total_seconds=time.monotonic()-start)
    (ROOT/'audit/training_components_profile.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'audit/training_components_failure.json').write_text(traceback.format_exc());raise
