"""Check the single budget-selected 20-step setting before any formal EEG run."""
from profile_standard_generator import *
import numpy as np
from PIL import Image

OUT_BUDGET=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S0_budget_sampler20')


def main():
    started=time.monotonic();assert not OUT_BUDGET.exists();OUT_BUDGET.mkdir(parents=True)
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline()
    source=json.loads((ROOT/'audit/generator_throughput_profile.json').read_text())['cases'][0]
    rows=source['array_rows'][:4]
    f=torch.load('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt',weights_only=True,mmap=True)
    c=f[rows].clone().to('cuda',dtype=torch.bfloat16)
    seeds=source['seeds'][:4]
    generators=[torch.Generator(device='cuda').manual_seed(s) for s in seeds]
    pipe.set_progress_bar_config(disable=True);torch.cuda.synchronize();tick=time.monotonic()
    with torch.inference_mode():
        latent=pipe(prompt_embeds=prompt.repeat(4,1,1),negative_prompt_embeds=negative.repeat(4,1,1),
            pooled_prompt_embeds=pool.repeat(4,1),negative_pooled_prompt_embeds=negative_pool.repeat(4,1),
            ip_adapter_image_embeds=[torch.cat([torch.zeros_like(c),c])[:,None,:]],height=512,width=512,
            num_inference_steps=20,guidance_scale=5.,generator=generators,output_type='latent').images
        assert torch.isfinite(latent).all()
        pixels=pipe.vae.decode(latent.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
        assert torch.isfinite(pixels).all();images=pipe.image_processor.postprocess(pixels,output_type='pil')
    torch.cuda.synchronize();seconds=time.monotonic()-tick
    files=[]
    for row,img in zip(rows,images):
        p=OUT_BUDGET/f'{row:08d}.png';j=OUT_BUDGET/f'{row:08d}.jpg'
        img.save(p);img.save(j,quality=90,subsampling=0)
        original=np.array(img).astype(np.float32);decoded=np.array(Image.open(j).convert('RGB')).astype(np.float32)
        mse=float(np.mean((original-decoded)**2))
        files.append(dict(array_row=row,png=str(p),png_sha256=sha(p),png_bytes=p.stat().st_size,
            jpeg=str(j),jpeg_sha256=sha(j),jpeg_bytes=j.stat().st_size,
            jpeg_PSNR_dB=10*math.log10(255**2/mse) if mse else 1000.))
    d=dict(status='PASSED',role='S0 T-oracle check; budget-based fixed setting, not a checkpoint/seed search',
        resolution=512,steps=20,cfg=5.,IP_scale=1.,batch=4,seeds=seeds,files=files,
        measured_generation_seconds_per_image=seconds/4,total_seconds=time.monotonic()-started,
        canonical_formal_image_encoding='JPEG quality90 subsampling0 uniformly; no best seed or postprocess',
        formal_EEG_performance=False)
    (ROOT/'audit/budget_sampler_profile.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({k:v for k,v in d.items() if k not in ['files','seeds']}),flush=True)


if __name__=='__main__':main()
