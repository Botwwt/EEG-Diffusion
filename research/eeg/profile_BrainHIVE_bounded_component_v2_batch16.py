"""Finite T-only engineering profile, four updates per arm, no model selection.

This is not formal training or a reconstruction result. It tests actual image
denoising gradients and fixed saved images before a complete lifecycle is funded.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

from brainhive_native_models_v1 import *
from brainhive_bounded_component_v1 import BoundedConditionAdapter,risk_target
from brainhive_component_generator_v1 import load_native,denoise


def main():
    assert b'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes()
    for name,status in [('native_benchmark_queue_v2','ALL3_NATIVE_R80_IMAGE_ARMS_AND_LOCKED_PER_QUERY_METRICS_COMPLETE'),
        ('original_metric_queue_v1','ALL3_NATIVE_R80_AUTHOR_EIGHT_METRICS_AND_QUERY_RECORDS_COMPLETE')]:
        assert json.loads((WORK/'audit'/(name+'.json')).read_text())['status']==status
    assert json.loads((WORK/'audit/bounded_component_CPU_v1.json').read_text())['status']=='BOUNDED_COMPONENT_EXACT_FALLBACK_BOUND_AND_GRADIENT_CHECKS_PASSED_CPU'
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=WORK/'engineering/bounded_component_profile_v2_batch16';assert not out.exists();out.mkdir(parents=True)
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text());ids=protocol['image_rows']['T'][:32]
    image_rows={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    contract_path=WORK/'data/component_input_contract_v1.json';contract=json.loads(contract_path.read_text())
    subject=contract['subjects'][0];assert subject['subject']=='sub-01'
    conditions={};affine={};source_rows=[]
    for source in subject['sources']:
        if not source['role'].startswith('oof'):continue
        assert sha(source['condition_cache_path'])==source['condition_cache_sha256']
        data=torch.load(source['condition_cache_path'],weights_only=True,map_location='cpu',mmap=True)
        weights=torch.load(source['checkpoint_path'],weights_only=True,map_location='cpu')['model']
        for row in source['groups']['component_train']:
            i=row['array_row']
            if i in ids:
                assert i not in conditions
                conditions[i]=data['condition'][row['cache_row']].clone()
                affine[i]=(weights['block.layers.0.norm.weight'].clone(),weights['block.layers.0.norm.bias'].clone())
                source_rows.append(dict(array_row=i,role=source['role'],checkpoint_sha256=source['checkpoint_sha256']))
    assert set(conditions)==set(ids)
    c=torch.stack([conditions[i] for i in ids]).cuda()
    gamma=torch.stack([affine[i][0] for i in ids]).cuda();beta=torch.stack([affine[i][1] for i in ids]).cuda()
    registration=dict(status='ENGINEERING_PROFILE_ONLY',subject='sub-01',array_rows=ids,
        source_rows=source_rows,updates_per_arm=4,batch_size=16,training_seed=1729,
        input_contract_sha256=sha(contract_path),source_sha256=sha(__file__),
        module_sha256=sha(ROOT/'brainhive_bounded_component_v1.py'),
        loader_sha256=sha(ROOT/'brainhive_component_generator_v1.py'),
        target_preprocessing='Author Resize512 BILINEAR / CenterCrop512 / RGB[-1,1], VAE FP32 posterior MODE',
        objective='epsilon MSE + .1 token angular error; risk arm additionally .1 base-risk MSE',
        fixed_radius=.25,learning_rate=1e-4,weight_decay=.01,
        no_candidate_checkpoint_selection=True,no_V_or_benchmark_images_read=True,
        no_scientific_image_success_claim=True,diagnostic_saved_images_per_arm=4)
    write(out/'registration.json',registration)
    pipe,adapter,projector,prompts,generator_receipt=load_native(training=True)
    from diffusers import DDPMScheduler
    from PIL import Image
    import numpy as np
    from torchvision import transforms
    transform=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    pixels=[];image_proofs=[]
    for i in ids:
        row=image_rows[i]
        with zipfile.ZipFile(row['image_zip']) as z:data=z.read(row['image_member'])
        assert hashlib.sha256(data).hexdigest()==row['image_sha256']
        with Image.open(io.BytesIO(data)) as image:pixels.append(transform(image.convert('RGB')))
        image_proofs.append(dict(array_row=i,stimulus_id=row['stimulus_id'],image_sha256=row['image_sha256']))
    with torch.no_grad():
        latents=torch.cat([pipe.vae.encode(x.cuda()).latent_dist.mode().float()*pipe.vae.config.scaling_factor for x in torch.stack(pixels).split(2)])
    assert latents.shape==(32,4,64,64) and torch.isfinite(latents).all()
    targets=torch.load(contract['target_cache_path'],weights_only=True,map_location='cpu',mmap=True)[ids].cuda()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        real_tokens=projector(adapter(targets));base_tokens=projector(adapter(c))
    target_risk=risk_target(base_tokens,real_tokens)
    scheduler=DDPMScheduler.from_config(pipe.scheduler.config)
    assert scheduler.config.prediction_type=='epsilon'
    arms={};profile={};torch.cuda.reset_peak_memory_stats()
    for name,with_risk in [('ordinary',False),('risk',True)]:
        torch.manual_seed(1729);model=BoundedConditionAdapter(predict_risk=with_risk).cuda()
        corrected,_=model(c,gamma,beta);assert torch.equal(corrected,c)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        rng=torch.Generator(device='cuda').manual_seed(1729)
        elapsed=[];events=[]
        for step in range(4):
            indices=torch.arange((step%2)*16,(step%2)*16+16,device='cuda')
            noise=torch.randn(latents[indices].shape,generator=rng,device='cuda')
            t=torch.randint(0,1000,(16,),device='cuda',generator=rng)
            noisy=scheduler.add_noise(latents[indices],noise,t)
            torch.cuda.synchronize();started=time.monotonic();optimizer.zero_grad(set_to_none=True)
            corrected,info=model(c[indices],gamma[indices],beta[indices])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tokens=projector(adapter(corrected));predicted=denoise(pipe,prompts,noisy,t,tokens)
                epsilon_loss=F.mse_loss(predicted.float(),noise)
                token_loss=(1-F.cosine_similarity(tokens.float(),real_tokens[indices].float(),dim=-1)).mean()
                risk_loss=F.mse_loss(info['risk'],target_risk[indices]) if with_risk else epsilon_loss.new_zeros(())
                loss=epsilon_loss+.1*token_loss+.1*risk_loss
            assert torch.isfinite(loss);loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert float(gradient)>0 and model.direction.weight.grad is not None
            assert all(p.grad is None for p in pipe.unet.parameters())
            assert all(p.grad is None for p in adapter.parameters()) and all(p.grad is None for p in projector.parameters())
            optimizer.step();torch.cuda.synchronize();seconds=time.monotonic()-started;elapsed.append(seconds)
            event=dict(arm=name,update=step+1,epsilon_loss=float(epsilon_loss),token_loss=float(token_loss),
                risk_loss=float(risk_loss),total_loss=float(loss),gradient_norm=float(gradient),
                timesteps=t.tolist(),seconds=seconds)
            events.append(event);print(json.dumps(event),flush=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            corrected,info=model(c,gamma,beta);changed=projector(adapter(corrected))
            token_change=float((changed.float()-base_tokens.float()).square().mean().sqrt())
            assert token_change>0
            t=torch.full((32,),749,device='cuda',dtype=torch.long)
            test_rng=torch.Generator(device='cuda').manual_seed(1731)
            noise=torch.randn(latents.shape,device='cuda',generator=test_rng)
            noisy=scheduler.add_noise(latents,noise,t)
            old=denoise(pipe,prompts,noisy,t,base_tokens)
            new=denoise(pipe,prompts,noisy,t,changed)
            response=float((new.float()-old.float()).square().mean().sqrt());assert response>0
        profile[name]=dict(parameters=sum(p.numel() for p in model.parameters()),updates=4,
            seconds_per_update=elapsed,steady_seconds_per_query=float(np.mean(elapsed[1:]))/16,
            final_token_RMS_change=token_change,actual_UNet_RMS_change=response,events=events,
            final_raw_correction_RMS=float(info['correction'].square().mean().sqrt()))
        arms[name]=model.eval();del optimizer
    peak=torch.cuda.max_memory_allocated()
    # Fixed T images exercise the actual inference path; they are not scored or
    # used to choose a checkpoint, strength, seed, or formal training schedule.
    del pipe,adapter,projector,prompts
    __import__('gc').collect();torch.cuda.empty_cache()
    pipe,adapter,projector,prompts,inference_receipt=load_native(training=False)
    rows=[]
    for name in ['parent','ordinary','risk']:
        for position,i in enumerate(ids[:4]):
            seed=int(hashlib.sha256(f"1729|sub-01|{image_rows[i]['stimulus_id']}|R4|{protocol['version']}".encode()).hexdigest()[:16],16)%(2**63-1)
            with torch.inference_mode():
                condition=c[position:position+1]
                if name!='parent':condition=arms[name](condition,gamma[position:position+1],beta[position:position+1])[0]
                tokens=projector(adapter(condition.to(torch.float16)))
                image=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),
                    prompt_embeds=prompts[0].half(),pooled_prompt_embeds=prompts[1].half(),
                    cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,
                    guidance_scale=0.,output_type='pil').images[0].convert('RGB')
            p=out/'images'/name/(Path(image_rows[i]['stimulus_id']).stem+'.webp');p.parent.mkdir(parents=True,exist_ok=True)
            assert not p.exists();image.save(p,format='WEBP',lossless=True,quality=90,method=4)
            with Image.open(p) as decoded:assert np.array_equal(np.asarray(decoded.convert('RGB')),np.asarray(image))
            rows.append(dict(arm=name,array_row=i,stimulus_id=image_rows[i]['stimulus_id'],image_seed=seed,
                path=str(p),sha256=sha(p),bytes=p.stat().st_size,engineering_T_training_image=True))
    write(out/'completion.json',dict(status='ENGINEERING_IMAGE_GRADIENT_AND_FIXED_T_IMAGE_PATH_COMPLETE',
        registration_sha256=sha(out/'registration.json'),generator=generator_receipt,profile=profile,
        peak_GPU_allocated_bytes=peak,inference_generator=inference_receipt,T_image_proofs=image_proofs,generated_images=rows,
        formal_training_complete=False,component_image_gain_established=False,
        dataset_or_sampling_reduced_for_formal_results=False))
    print('BOUNDED_COMPONENT_ENGINEERING_PROFILE_COMPLETE',flush=True)


if __name__=='__main__':main()
