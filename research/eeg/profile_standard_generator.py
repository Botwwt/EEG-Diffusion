"""S0 actual-image profiling of pinned standard SDXL; no benchmark images read."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL = Path('/dev/shm/eeg_sota_autoresearch_20261018_v1/sdxl_base_46216598')
ASSETS = Path('/dev/shm/eeg_diffusion_next_20261018_v1')
OUT = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S0_standard_generator_attempt2')
sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class EEGStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """One explicitly batched native pooled condition per EEG query.

    The base helper repeats supplied embeddings by prompt batch size, which is
    appropriate for its one-image-prompt convention, but mixes prebatched EEG
    query conditions. Our caller already supplies the complete CFG query batch.
    """
    def prepare_ip_adapter_image_embeds(self,ip_adapter_image,ip_adapter_image_embeds,
                                       device,num_images_per_prompt,do_classifier_free_guidance):
        assert ip_adapter_image is None and isinstance(ip_adapter_image_embeds,list)
        assert len(ip_adapter_image_embeds)==1
        batch=int(num_images_per_prompt)*(2 if do_classifier_free_guidance else 1)
        value=ip_adapter_image_embeds[0]
        assert value.ndim==3 and value.shape==(batch,1,1024), (value.shape,batch)
        # Negative queries precede positive queries, exactly as packed by caller.
        return [value.to(device=device)]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for b in iter(lambda:file.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def load_pipeline():
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads((MODEL/'completion.json').read_text())
    assert receipt['status']=='COMPLETE_OFFICIAL_SNAPSHOT'
    assert receipt['revision']=='462165984030d82259a11f4367a4eed129e94a7b'
    for row in receipt['files']:assert sha(Path(row['path']))==row['sha256']
    load_infos={}
    def load(cls,folder,dtype):
        model,info=cls.from_pretrained(MODEL/folder,local_files_only=True,variant='fp16',
                  torch_dtype=dtype,output_loading_info=True)
        load_infos[folder]=info
        assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']), (folder,info)
        return model.eval().requires_grad_(False)
    unet=load(UNet2DConditionModel,'unet',torch.bfloat16)
    vae=load(AutoencoderKL,'vae',torch.float32)
    text=load(CLIPTextModel,'text_encoder',torch.bfloat16)
    text2=load(CLIPTextModelWithProjection,'text_encoder_2',torch.bfloat16)
    tokenizer=CLIPTokenizer.from_pretrained(MODEL/'tokenizer',local_files_only=True)
    tokenizer2=CLIPTokenizer.from_pretrained(MODEL/'tokenizer_2',local_files_only=True)
    scheduler=EulerDiscreteScheduler.from_pretrained(MODEL/'scheduler',local_files_only=True)
    assert scheduler.config.prediction_type=='epsilon'
    pipe=EEGStableDiffusionXLPipeline(vae=vae,text_encoder=text,text_encoder_2=text2,
         tokenizer=tokenizer,tokenizer_2=tokenizer2,unet=unet,scheduler=scheduler,
         image_encoder=None,feature_extractor=None,add_watermarker=False)
    pipe.load_ip_adapter(str(ASSETS),subfolder='',weight_name='adapter.safetensors',
         image_encoder_folder=None,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.set_ip_adapter_scale(1.)
    pipe.unet.eval().requires_grad_(False);pipe.unet.to(device='cuda',dtype=torch.bfloat16)
    pipe.vae.to(device='cuda',dtype=torch.float32);pipe.vae.enable_slicing()
    pipe.text_encoder.to('cuda');pipe.text_encoder_2.to('cuda')
    with torch.inference_mode():
        prompt,negative,pool,negative_pool=pipe.encode_prompt('',device='cuda',
            num_images_per_prompt=1,do_classifier_free_guidance=True,negative_prompt=None)
    # Diffusers derives its execution device from registered components. Keep
    # both text encoders on CUDA so latent initialization also stays on CUDA.
    assert pipe._execution_device.type=='cuda'
    torch.cuda.empty_cache()
    assert pipe.unet.encoder_hid_proj.image_projection_layers[0].image_embeds.in_features==1024
    expected_batch={}
    def guard_unet(module,args,kwargs):
        count=len(args[0]);expected_batch['count']=count
        values=kwargs['added_cond_kwargs']['image_embeds']
        assert len(values)==1 and values[0].shape==(count,1,1024), (values[0].shape,count)
    def guard_ip_keys(module,args):
        value=args[0];count=expected_batch['count']
        assert value.shape[0]==count and value.numel()//(count*value.shape[-1])==4, (value.shape,count)
    pipe.unet.register_forward_pre_hook(guard_unet,with_kwargs=True)
    key_modules=[module for name,module in pipe.unet.named_modules() if name.endswith('to_k_ip.0')]
    assert key_modules
    for module in key_modules:module.register_forward_pre_hook(guard_ip_keys)
    return pipe, load_infos, receipt, (prompt,negative,pool,negative_pool)


def main():
    started=time.monotonic()
    assert not OUT.exists();OUT.mkdir(parents=True)
    pipe,load_infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline()
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    indices=sorted(protocol['image_rows']['T'])[:4]
    features=torch.load('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt',weights_only=True,mmap=True)
    # The author cache was independently checked against the actual CLIP-H image encoder.
    c=features[indices].clone().to(device='cuda',dtype=torch.bfloat16)
    counts=dict(UNet_forward_calls=0,UNet_sample_evaluations=0,VAE_decode_images=0)
    def record(module,args):
        counts['UNet_forward_calls']+=1;counts['UNet_sample_evaluations']+=len(args[0])
    handle=pipe.unet.register_forward_pre_hook(record)
    pipe.set_progress_bar_config(disable=True)
    cases=[(512,1,'oracle'),(512,4,'oracle'),(1024,1,'oracle'),(1024,4,'oracle'),(1024,4,'zero_condition')]
    results=[]
    for resolution,batch,condition in cases:
        label=f'{resolution}_batch{batch}_{condition}';folder=OUT/label;folder.mkdir()
        before=dict(counts);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
        cond=c[:batch] if condition=='oracle' else torch.zeros_like(c[:batch])
        prepared=torch.cat([torch.zeros_like(cond),cond])[:,None,:]
        seeds=[int(hashlib.sha256(f'1729|S0-standard|train-row-{i}'.encode()).hexdigest()[:16],16) % (2**63-1) for i in indices[:batch]]
        generators=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
        row=dict(case=label,resolution=resolution,batch=batch,condition=condition,
                 array_rows=indices[:batch],seeds=seeds,steps=30,cfg=5.,ip_adapter_scale=1.,
                 formal_evaluation=False,role='T only; load/oracle/cost diagnostic')
        try:
            with torch.inference_mode():
                latent=pipe(prompt_embeds=prompt.repeat(batch,1,1),negative_prompt_embeds=negative.repeat(batch,1,1),
                    pooled_prompt_embeds=pool.repeat(batch,1),negative_pooled_prompt_embeds=negative_pool.repeat(batch,1),
                    ip_adapter_image_embeds=[prepared],height=resolution,width=resolution,
                    num_inference_steps=30,guidance_scale=5.,num_images_per_prompt=1,
                    generator=generators,output_type='latent').images
                assert latent.shape==(batch,4,resolution//8,resolution//8) and torch.isfinite(latent).all()
                images=pipe.vae.decode(latent.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
                assert torch.isfinite(images).all()
                images=pipe.image_processor.postprocess(images,output_type='pil')
            torch.cuda.synchronize();elapsed=time.monotonic()-tick
            records=[]
            for index,img in zip(indices[:batch],images):
                png=folder/f'row_{index}.png';jpg=folder/f'row_{index}_q95.jpg'
                img.save(png);img.save(jpg,quality=95,subsampling=0)
                original=np.asarray(img).astype(np.float32);decoded=np.asarray(Image.open(jpg).convert('RGB')).astype(np.float32)
                mse=float(np.mean((original-decoded)**2));psnr=10*math.log10(255**2/mse) if mse else 1000.
                records.append(dict(array_row=index,png=str(png),png_sha256=sha(png),png_bytes=png.stat().st_size,
                                    jpeg95=str(jpg),jpeg95_sha256=sha(jpg),jpeg95_bytes=jpg.stat().st_size,
                                    jpeg95_psnr_db=psnr))
            counts['VAE_decode_images']+=batch
            row.update(status='COMPLETED',seconds=elapsed,seconds_per_image=elapsed/batch,
                       peak_cuda_allocated=torch.cuda.max_memory_allocated(),files=records,
                       latent_rms=float(latent.float().square().mean().sqrt()),
                       condition_norms=cond.float().norm(dim=-1).tolist())
            del latent,images
        except torch.cuda.OutOfMemoryError as exc:
            row.update(status='OUT_OF_MEMORY',error=str(exc));torch.cuda.empty_cache()
        row['counts']={k:counts[k]-before[k] for k in counts}
        results.append(row)
        (OUT/'progress.json').write_text(json.dumps(dict(last_case=row,results=results),indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='files'}),flush=True)
    handle.remove()
    report=dict(status='S0_PROFILE_COMPLETED',formal_evaluation=False,training_updates=0,
         generator='stabilityai/stable-diffusion-xl-base-1.0',revision=receipt['revision'],
         scheduler=dict(pipe.scheduler.config),seed=1729,cases=results,counts=counts,
         load_infos=load_infos,UNet_with_adapter_parameters=sum(p.numel() for p in pipe.unet.parameters()),
         VAE_parameters=sum(p.numel() for p in pipe.vae.parameters()),
         trainable_parameters=sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad),
         precision=dict(UNet='bfloat16',VAE='float32',native_conditions='float32 cached; projection receives bfloat16'),
         caveat='Timing includes first-use overhead; repetition/batch scaling and actual full-dataset I/O still need allowance. These T oracle images do not establish EEG reconstruction quality.',
         seconds=time.monotonic()-started)
    (ROOT/'audit/standard_generator_profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'],output=str(OUT))))


if __name__=='__main__':
    main()
