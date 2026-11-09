"""S0 full-size batch timing only; same 30-step standard SDXL, T oracle conditions."""
from pathlib import Path
import hashlib
import json
import time
import traceback
from profile_standard_generator import load_pipeline, torch, ROOT

OUT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S0_generator_throughput')


def main():
    started=time.monotonic()
    assert not OUT.exists();OUT.mkdir(parents=True)
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline()
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    ordered=sorted(protocol['image_rows']['T'])
    rows=[ordered[i*len(ordered)//32] for i in range(32)]
    features=torch.load('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt',weights_only=True,mmap=True)
    pipe.set_progress_bar_config(disable=True)
    records=[]
    for batch in (16,32):
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
        case=dict(resolution=512,batch=batch,steps=30,cfg=5.,IP_scale=1.,
                  array_rows=rows[:batch],role='T-only throughput; not EEG or formal method performance')
        try:
            c=features[rows[:batch]].clone().to('cuda',dtype=torch.bfloat16)
            seeds=[int(hashlib.sha256(f'1729|throughput|{i}'.encode()).hexdigest()[:16],16)%(2**63-1) for i in rows[:batch]]
            generators=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
            with torch.inference_mode():
                latents=pipe(prompt_embeds=prompt.repeat(batch,1,1),negative_prompt_embeds=negative.repeat(batch,1,1),
                    pooled_prompt_embeds=pool.repeat(batch,1),negative_pooled_prompt_embeds=negative_pool.repeat(batch,1),
                    ip_adapter_image_embeds=[torch.cat([torch.zeros_like(c),c])[:,None,:]],
                    height=512,width=512,num_inference_steps=30,guidance_scale=5.,
                    generator=generators,output_type='latent').images
                assert torch.isfinite(latents).all()
                pixels=pipe.vae.decode(latents.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
                assert torch.isfinite(pixels).all()
                images=pipe.image_processor.postprocess(pixels,output_type='pil')
            torch.cuda.synchronize();seconds=time.monotonic()-tick
            folder=OUT/f'batch{batch}';folder.mkdir()
            files=[]
            for row,img in zip(rows[:batch],images):
                p=folder/f'{row:08d}.png';img.save(p)
                files.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
            case.update(status='PASSED',seconds=seconds,seconds_per_image=seconds/batch,
                        seconds_with_saving=time.monotonic()-tick,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                        seeds=seeds,files=files)
            del c,latents,pixels,images
        except torch.cuda.OutOfMemoryError as exc:
            case.update(status='OUT_OF_MEMORY',message=str(exc));torch.cuda.empty_cache()
        records.append(case)
        (ROOT/'audit/generator_throughput_profile.json').write_text(json.dumps(dict(
            status='IN_PROGRESS',formal_evaluation=False,cases=records),indent=2)+'\n')
        print(json.dumps({k:v for k,v in case.items() if k not in ['files','seeds','array_rows']}),flush=True)
    (ROOT/'audit/generator_throughput_profile.json').write_text(json.dumps(dict(
        status='COMPLETED',formal_evaluation=False,training_updates=0,cases=records,
        total_seconds=time.monotonic()-started),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'audit/generator_throughput_failure.json').write_text(traceback.format_exc());raise
