"""Actual fixed T-image readiness for the registered v prior, no seed selection."""
from profile_standard_generator import *
from generate_reference_images import sample_prior,randn_tensor,DiffusionPriorUNet
from PIL import ImageDraw

DIAG=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S0_prior_velocity_T_images_v1')


def main():
    started=time.monotonic();assert not DIAG.exists();DIAG.mkdir(parents=True)
    previous=json.loads((ROOT/'audit/prior_T_image_diagnosis_batchsafe_v2.json').read_text())
    cfg=json.loads((ROOT/'configs/prior_velocity_repair_resolved.json').read_text())
    trained=Path(cfg['output_root'])/'full';assert (trained/'completion.json').exists()
    proof=json.loads((trained/'sub-01/completion.json').read_text())
    assert proof['prior']['prediction_type']=='v_prediction'
    assert sha(Path(proof['prior']['checkpoint']))==proof['prior']['checkpoint_sha256']
    rows=previous['rows'];seeds=previous['seeds']
    cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
    assert torch.equal(cache['array_rows'],torch.arange(16540));c=cache['condition'][rows].cuda()
    prior=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval().requires_grad_(False)
    prior.load_state_dict(torch.load(proof['prior']['checkpoint'],weights_only=True),strict=True)
    with torch.inference_mode():native=sample_prior(prior,c,seeds,prediction_type='v_prediction')
    del prior
    true=torch.load(cfg['visual_features_path'],weights_only=True,mmap=True)[rows].clone().cuda()
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline();pipe.set_progress_bar_config(disable=True)
    generators=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
    with torch.inference_mode():
        noise=randn_tensor((4,4,64,64),generator=generators,device='cuda',dtype=torch.bfloat16)
        noise_sha=[hashlib.sha256(x.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for x in noise]
        assert noise_sha==previous['statistics']['ORACLE_native']['initial_image_noise_sha256']
        value=native.to(torch.bfloat16)
        latents=pipe(prompt_embeds=prompt.repeat(4,1,1),negative_prompt_embeds=negative.repeat(4,1,1),
            pooled_prompt_embeds=pool.repeat(4,1),negative_pooled_prompt_embeds=negative_pool.repeat(4,1),
            ip_adapter_image_embeds=[torch.cat([torch.zeros_like(value),value])[:,None,:]],height=512,width=512,
            num_inference_steps=20,guidance_scale=5.,latents=noise,output_type='latent').images
        decoded=pipe.vae.decode(latents.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
        assert torch.isfinite(latents).all() and torch.isfinite(decoded).all()
        images=pipe.image_processor.postprocess(decoded,output_type='pil')
    paths=[]
    for row,image in zip(rows,images):
        p=DIAG/f'{row:08d}.jpg';image.save(p,quality=90,subsampling=0);paths.append(str(p))
    cases=previous['generated_paths']|dict(PRIOR_V=paths);names=['TARGET',*cases]
    width=220;cell=230;sheet=Image.new('RGB',(width*len(names),cell*4+30),'white');draw=ImageDraw.Draw(sheet)
    for col,name in enumerate(names):draw.text((col*width+5,5),name,fill='black')
    for i,row in enumerate(rows):
        pictures=[previous['targets'][i]['target_path']]+[cases[name][i] for name in cases]
        for col,path in enumerate(pictures):
            with Image.open(path) as im:sheet.paste(im.convert('RGB').resize((220,200)),(col*width,30+i*cell))
        draw.text((5,30+i*cell+202),f'row {row}: '+previous['targets'][i]['stimulus_id'],fill='black')
    contact=ROOT/'audit/prior_velocity_T_diagnostic_contact.jpg';sheet.save(contact,quality=90)
    report=dict(status='ACTUAL_VELOCITY_T_IMAGES_READY_FOR_REVIEW_NOT_FORMAL_V_RESULTS',rows=rows,subject='sub-01',seeds=seeds,
        generated_paths=paths,targets=previous['targets'],native_norms=native.norm(dim=-1).tolist(),
        native_cosines_to_true=torch.nn.functional.cosine_similarity(native,true).tolist(),
        initial_image_noise_sha256=noise_sha,matched_image_noise_all_cases=True,
        prior_parameters=9675648,prior_epochs=400,prior_updates=5200,
        EEG_encoder_checkpoint_sha256=proof['encoder']['checkpoint_sha256'],prior_checkpoint=proof['prior']['checkpoint'],
        prior_checkpoint_sha256=proof['prior']['checkpoint_sha256'],prediction_type='v_prediction',
        prior_steps=50,prior_CFG=5.,image_steps=20,image_CFG=5.,
        latent_RMS=float(latents.float().square().mean().sqrt()),decoded_fraction_outside_unit_range=float((decoded.abs()>1).float().mean()),
        contact_sheet=str(contact),training_updates_this_diagnostic=0,full_V_evaluated=False,
        hypothesis='Prediction parameterization/associated weighting, not model capacity; engineering repair, no novelty claim',
        seconds=time.monotonic()-started)
    (ROOT/'audit/prior_velocity_T_image_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['targets','generated_paths','seeds','initial_image_noise_sha256']}),flush=True)


if __name__=='__main__':main()
