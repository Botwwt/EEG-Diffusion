"""Fixed four diverse T targets: actual same-seed oracle/point/prior images."""
from profile_standard_generator import *
from generate_reference_images import sample_prior,query_seed,prior_seed,randn_tensor,DiffusionPriorUNet,DDPMScheduler
import csv
import io
import zipfile
from PIL import ImageDraw

DIAG=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S0_prior_T_images_batchsafe_v2')


@torch.inference_mode()
def clipped_sample(model,c,seeds):
    gens=[torch.Generator(device='cuda').manual_seed(prior_seed(seed)) for seed in seeds]
    scheduler=DDPMScheduler(clip_sample=True);scheduler.set_timesteps(50,device='cuda')
    h=randn_tensor((len(c),1024),generator=gens,device='cuda',dtype=torch.float32)
    for t in scheduler.timesteps:
        times=torch.full((len(h),),float(t),device='cuda')
        cond=model(h,times,c);uncond=model(h,times,None)
        h=scheduler.step(uncond+5*(cond-uncond),int(t),h,generator=gens).prev_sample
    return h


def main():
    started=time.monotonic();torch.set_num_threads(2);assert not DIAG.exists();DIAG.mkdir(parents=True)
    cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text());protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    training=Path(cfg['output_root'])/'full';proof=json.loads((training/'sub-01/completion.json').read_text())
    rows=[r['array_row'] for r in json.loads((ROOT/'audit/budget_sampler_profile.json').read_text())['files']]
    assert all(i in protocol['image_rows']['T'] for i in rows)
    manifest={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    seeds=[query_seed('sub-01',manifest[i]['stimulus_id'],'R4',protocol['version']) for i in rows]
    cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
    c=cache['condition'][rows].cuda()
    real=torch.load(cfg['visual_features_path'],weights_only=True,mmap=True)[rows].clone().cuda()
    prior=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval().requires_grad_(False)
    prior.load_state_dict(torch.load(proof['prior']['checkpoint'],weights_only=True),strict=True)
    with torch.inference_mode():
        values=dict(ORACLE_native=real,REF_REG_point=c,PRIOR_unclipped=sample_prior(prior,c,seeds),PRIOR_author_clipped=clipped_sample(prior,c,seeds))
    del prior
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline();pipe.set_progress_bar_config(disable=True)
    files={};statistics={}
    for name,native in values.items():
        folder=DIAG/name;folder.mkdir();generators=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
        with torch.inference_mode():
            noise=randn_tensor((4,4,64,64),generator=generators,device='cuda',dtype=torch.bfloat16)
            prepared=native.to(torch.bfloat16)
            latents=pipe(prompt_embeds=prompt.repeat(4,1,1),negative_prompt_embeds=negative.repeat(4,1,1),
                pooled_prompt_embeds=pool.repeat(4,1),negative_pooled_prompt_embeds=negative_pool.repeat(4,1),
                ip_adapter_image_embeds=[torch.cat([torch.zeros_like(prepared),prepared])[:,None,:]],
                height=512,width=512,num_inference_steps=20,guidance_scale=5.,latents=noise,output_type='latent').images
            decoded=pipe.vae.decode(latents.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
            assert torch.isfinite(latents).all() and torch.isfinite(decoded).all()
            images=pipe.image_processor.postprocess(decoded,output_type='pil')
        paths=[]
        for i,image in zip(rows,images):
            p=folder/f'{i:08d}.jpg';image.save(p,quality=90,subsampling=0);paths.append(str(p))
        files[name]=paths
        statistics[name]=dict(native_norms=native.norm(dim=-1).tolist(),native_cosines_to_real=torch.nn.functional.cosine_similarity(native,real).tolist(),
            initial_image_noise_sha256=[hashlib.sha256(x.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for x in noise],
            latent_RMS=float(latents.float().square().mean().sqrt()),decoded_fraction_outside_unit_range=float((decoded.abs()>1).float().mean()))
    assert len({tuple(v['initial_image_noise_sha256']) for v in statistics.values()})==1
    targets=[]
    with zipfile.ZipFile(manifest[rows[0]]['image_zip']) as archive:
        for i in rows:
            raw=archive.read(manifest[i]['image_member']);assert hashlib.sha256(raw).hexdigest()==manifest[i]['image_sha256']
            p=DIAG/f'target_{i:08d}.jpg';p.write_bytes(raw);targets.append(str(p))
    width=220;cell=230;sheet=Image.new('RGB',(width*5,cell*4+30),'white');draw=ImageDraw.Draw(sheet)
    names=['TARGET',*values]
    for col,name in enumerate(names):draw.text((col*width+5,5),name,fill='black')
    for j,i in enumerate(rows):
        pictures=[targets[j]]+[files[name][j] for name in values]
        for col,p in enumerate(pictures):
            with Image.open(p) as im:sheet.paste(im.convert('RGB').resize((220,200)),(col*width,30+j*cell))
        draw.text((5,30+j*cell+202),f"row {i}: {manifest[i]['stimulus_id']}",fill='black')
    contact=ROOT/'audit/prior_T_diagnostic_contact_batchsafe_v2.jpg';sheet.save(contact,quality=90)
    report=dict(status='ACTUAL_T_IMAGE_DIAGNOSTIC_COMPLETED_NO_FORMAL_METHOD_SELECTION',subject='sub-01',rows=rows,seeds=seeds,
        targets=[dict(stimulus_id=manifest[i]['stimulus_id'],image_zip=manifest[i]['image_zip'],image_member=manifest[i]['image_member'],
            target_image_sha256=manifest[i]['image_sha256'],target_path=targets[j],score_target_path=targets[j]) for j,i in enumerate(rows)],
        generated_paths=files,statistics=statistics,contact_sheet=str(contact),seconds=time.monotonic()-started,
        oracle_target_use='T diagnostic only; never a ranked EEG method',images_per_query_per_case=1,best_of_n=False,
        checkpoints_changed=False,training_updates=0,full_V_evaluated=False)
    (ROOT/'audit/prior_T_image_diagnosis_batchsafe_v2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],seconds=report['seconds'],contact_sheet=str(contact))),flush=True)


if __name__=='__main__':main()
