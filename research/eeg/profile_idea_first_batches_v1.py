"""One T-only, fixed-precision batch-cost probe; all scratch weights discarded.

No method/seed search. Larger batches retain the original steps, CFG, weights,
preprocessing and metric algebra. This measures whether the complete matched
idea study can fit; it does not authorize adapter training or change a protocol.
"""
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import io
import json
import time
import zipfile

from profile_standard_generator import load_pipeline, torch, ROOT
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import numpy as np
import torch.nn.functional as F


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def timed(call):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    value = call()
    torch.cuda.synchronize()
    return value, dict(seconds=time.monotonic()-started,
                       peak_cuda_allocated=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    assert plan['version'] == 'idea_first_batch_cost_probe_v1'
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    parent = Path('/proc')/str(__import__('os').getppid())/'cmdline'
    assert parent.exists() and str(ROOT/'run_gpu_job.py').encode() in parent.read_bytes()
    started = time.monotonic()
    output = Path(plan['output_root'])
    assert not output.exists()
    output.mkdir(parents=True)
    protocol = json.loads((ROOT/'configs/split_manifest.json').read_text())
    rows = sorted(protocol['image_rows']['T'])[:64]
    assert rows == plan['training_array_rows'] and len(rows) == 64
    assert not set(rows) & set(protocol['image_rows']['D']+protocol['image_rows']['V'])
    cfg = json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    assert sha(cfg['visual_features_path']) == cfg['visual_features_sha256']
    native = torch.load(cfg['visual_features_path'], weights_only=True, mmap=True)[rows].clone()
    pipe, infos, receipt, prompts = load_pipeline()
    pipe.set_progress_bar_config(disable=True)
    prompt, negative, pool, negative_pool = prompts
    c = native.cuda().to(torch.bfloat16)
    seeds = [int(hashlib.sha256(f'1729|T-only-batch-profile|{row}'.encode()).hexdigest()[:16], 16) % (2**63-1) for row in rows]
    noise = randn_tensor((64, 4, 64, 64),
        generator=[torch.Generator(device='cuda').manual_seed(s) for s in seeds],
        device='cuda', dtype=torch.bfloat16)

    def generate(batch):
        result = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch):
                cond = c[start:start+batch]
                n = len(cond)
                latent = pipe(prompt_embeds=prompt.repeat(n,1,1), negative_prompt_embeds=negative.repeat(n,1,1),
                    pooled_prompt_embeds=pool.repeat(n,1), negative_pooled_prompt_embeds=negative_pool.repeat(n,1),
                    ip_adapter_image_embeds=[torch.cat([torch.zeros_like(cond), cond])[:,None,:]],
                    height=512, width=512, num_inference_steps=20, guidance_scale=5.,
                    latents=noise[start:start+n].clone(), output_type='latent').images
                assert torch.isfinite(latent).all()
                result.append(latent.cpu())
        return torch.cat(result)

    generations = {}
    latents = {}
    for batch in [4,16]:
        latents[batch], generations[str(batch)] = timed(lambda: generate(batch))
        generations[str(batch)].update(queries=64, UNet_sample_evaluations=64*40,
            seconds_per_image=generations[str(batch)]['seconds']/64,
            status='COMPLETED_FIXED_T_GENERATION')
        print(json.dumps(dict(stage='generation_batch_probe', batch=batch, **generations[str(batch)])), flush=True)
    latent_relative_rmse = float((latents[16].float()-latents[4].float()).square().mean().sqrt()/latents[4].float().square().mean().sqrt())
    time_ids = pipe._get_add_time_ids((512,512),(0,0),(512,512),dtype=prompt.dtype,
                                    text_encoder_projection_dim=pool.shape[-1]).cuda()

    @torch.inference_mode()
    def eps(values, z):
        n = len(values)
        return pipe.unet(z, torch.full((n,),500,device='cuda',dtype=torch.long),
            encoder_hidden_states=prompt.repeat(n,1,1),
            added_cond_kwargs=dict(text_embeds=pool.repeat(n,1), time_ids=time_ids.repeat(n,1),
                                   image_embeds=[values[:,None,:]]), return_dict=False)[0].float()
    e16 = eps(c[:16], noise[:16])
    e4 = torch.cat([eps(c[i:i+4], noise[i:i+4]) for i in range(0,16,4)])
    changed = c[:16].clone(); changed[0] = c[16]
    intervention = eps(changed, noise[:16])
    epsilon_relative_rmse = float((e16-e4).square().mean().sqrt()/e4.square().mean().sqrt())
    isolation_delta = float((intervention[1:]-e16[1:]).abs().max())
    changed_delta = float((intervention[0]-e16[0]).abs().max())
    assert isolation_delta == 0 and changed_delta > 0
    assert epsilon_relative_rmse < .02, 'Same tolerance as existing batch-isolation readiness; do not promote incompatible kernels'
    paths = []
    with torch.inference_mode():
        for start in range(0,64,4):
            value = latents[16][start:start+4].cuda().float()
            decoded = pipe.vae.decode(value/pipe.vae.config.scaling_factor, return_dict=False)[0]
            assert torch.isfinite(decoded).all()
            images = pipe.image_processor.postprocess(decoded, output_type='pil')
            for offset, image in enumerate(images):
                path = output/f'T_{rows[start+offset]:05d}.jpg'
                image.save(path, quality=90, subsampling=0); paths.append(path)
    # True T images provide the same latent/noise objective as the real cost
    # probe. These are 64 distinct images, never fake EEG repetitions.
    image_rows = {int(x['array_row']):x for x in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    pixels = []; archives = {}
    for row in rows:
        item = image_rows[row]
        if item['image_zip'] not in archives:
            archives[item['image_zip']] = zipfile.ZipFile(item['image_zip'])
        archive = archives[item['image_zip']]
        raw = archive.read(item['image_member']); assert hashlib.sha256(raw).hexdigest() == item['image_sha256']
        with Image.open(io.BytesIO(raw)) as image:
            value = np.array(image.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),copy=True)
        pixels.append(torch.from_numpy(value).permute(2,0,1).float()/127.5-1)
    for archive in archives.values(): archive.close()
    with torch.inference_mode():
        target = torch.cat([pipe.vae.encode(torch.stack(pixels[i:i+4]).cuda()).latent_dist.mode()*pipe.vae.config.scaling_factor for i in range(0,64,4)])
    target = target.clone(); del pixels
    selected = []
    for name, parameter in pipe.unet.named_parameters():
        if name.startswith('encoder_hid_proj.'):
            parameter.requires_grad_(True); parameter.data = parameter.data.float(); selected.append(parameter)
    assert sum(p.numel() for p in selected) == 8400896
    pipe.unet.train(); pipe.unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(selected, lr=1e-4, weight_decay=.01)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    training = {}
    for batch in [16,64]:
        durations = []; torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(4):
                torch.cuda.synchronize(); tick = time.monotonic()
                epsilon = torch.randn_like(target[:batch]); t = torch.randint(0,1000,(batch,),device='cuda')
                z = scheduler.add_noise(target[:batch], epsilon, t)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    predicted = pipe.unet(z.to(torch.bfloat16), t, encoder_hidden_states=prompt.repeat(batch,1,1),
                        added_cond_kwargs=dict(text_embeds=pool.repeat(batch,1),time_ids=time_ids.repeat(batch,1),
                            image_embeds=[native[:batch].cuda()[:,None,:]]),return_dict=False)[0]
                    loss = F.mse_loss(predicted.float(), epsilon)
                assert torch.isfinite(loss); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True)
                assert float(norm)>0 and all(p.grad is not None for p in selected)
                assert all(p.grad is None for n,p in pipe.unet.named_parameters() if not n.startswith('encoder_hid_proj.'))
                optimizer.step(); torch.cuda.synchronize(); durations.append(time.monotonic()-tick)
            steady = sum(durations[1:])/3
            training[str(batch)] = dict(status='PASSED_COST_AND_GRADIENT_PROBE',steady_seconds_per_step=steady,
                seconds_per_example=steady/batch,peak_cuda_allocated=torch.cuda.max_memory_allocated(),updates=4)
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True); clear()
            training[str(batch)] = dict(status='OOM_NOT_SCIENTIFIC_FAILURE')
        print(json.dumps(dict(stage='training_batch_probe', batch=batch, **training[str(batch)])),flush=True)
    del loss, predicted, epsilon, t, z, norm
    del pipe, optimizer, selected, target, c, noise, prompts, prompt, negative, pool, negative_pool, time_ids
    clear()
    from semantic_evaluator import caption_models,caption_features,detection_model,detection_features
    from image_evaluation import detected_categories
    processor,model,text,info = caption_models()
    captions = {};caption_timing = {}
    for batch in [16,64]:
        try:
            captions[batch],caption_timing[str(batch)] = timed(lambda: caption_features(paths,processor,model,text,batch_size=batch))
        except torch.cuda.OutOfMemoryError:
            clear();caption_timing[str(batch)] = dict(status='OOM_NOT_PROMOTED')
    caption_equal = 64 in captions and captions[16][0] == captions[64][0]
    caption_feature_maxdelta = float(np.max(np.abs(captions[16][1]-captions[64][1]))) if caption_equal else None
    del model,text,processor;clear()
    detector,classes,config = detection_model()
    detections = {};detection_timing = {}
    for batch in [4,16]:
        try:
            detections[batch],detection_timing[str(batch)] = timed(lambda: detection_features(paths,detector,classes,batch_size=batch))
        except torch.cuda.OutOfMemoryError:
            clear();detection_timing[str(batch)] = dict(status='OOM_NOT_PROMOTED')
    category_equal = 16 in detections and all(set(detected_categories(a,float(t))) == set(detected_categories(b,float(t)))
        for a,b in zip(detections[4],detections[16]) for t in np.arange(.05,.95,.05))
    report = dict(status='T_ONLY_FIXED_PRECISION_BATCH_COSTS_MEASURED',seed=1729,array_rows=rows,
        plan_sha256=sha(plan_path),generator_revision=receipt['revision'],generation=generations,training=training,
        caption=caption_timing,detection=detection_timing,caption_strings_equal_64_T_images=caption_equal,
        caption_feature_maxdelta=caption_feature_maxdelta,ObjectF1_categories_equal_all_thresholds_64_T_images=category_equal,
        epsilon_relative_RMSE_batch16_vs4=epsilon_relative_rmse,full_trajectory_latent_relative_RMSE=latent_relative_rmse,
        other15_queries_intervention_max_delta=isolation_delta,changed_query_max_delta=changed_delta,
        sources_unchanged=True,updated_weights_discarded=True,formal_adapter_training=False,
        scientific_mechanism_result=False,all_future_formal_arms_require_same_runtime=True,
        no_claim_of_all_query_numerical_identity=True,seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'audit/idea_first_batch_cost_profile_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
