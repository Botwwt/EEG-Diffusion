"""Fixed8 oracle-norm intervention; same EEG direction/noise, diagnostic only."""
from pathlib import Path
import hashlib
import json
import time
from profile_standard_generator import load_pipeline,torch,ROOT
from generate_reference_images import tensor_sha,randn_tensor
from PIL import Image,ImageDraw,ImageOps


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    started=time.monotonic();cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    interface=json.loads((ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json').read_text())
    for name,expected in interface['source_sha256'].items():assert sha(ROOT/name)==expected
    qa=json.loads((ROOT/'audit/V_fixed8_partial_visual_QA.json').read_text());queries=qa['queries'];assert len(queries)==8
    oracle=json.loads((ROOT/'audit/V_fixed8_oracle_diagnosis.json').read_text())
    oracle_by_id={q['query_id']:q for q in oracle['queries']}
    proof=json.loads((Path(cfg['output_root'])/'full/sub-01/completion.json').read_text())
    assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
    cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
    assert cache['encoder_sha256']==proof['encoder']['checkpoint_sha256']
    assert torch.equal(cache['array_rows'],torch.arange(16540))
    assert sha(cfg['visual_features_path'])==cfg['visual_features_sha256']
    features=torch.load(cfg['visual_features_path'],weights_only=True,mmap=True)
    out=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/diagnostics/V_fixed8_oracle_norm_only_v1')
    assert not out.exists();out.mkdir(parents=True)
    pipe,infos,model_receipt,(prompt,negative,pool,negative_pool)=load_pipeline()
    pipe.set_progress_bar_config(disable=True);counts=dict(calls=0,sample_evaluations=0)
    def hook(module,inputs):counts['calls']+=1;counts['sample_evaluations']+=len(inputs[0])
    handle=pipe.unet.register_forward_pre_hook(hook);results=[]
    for start in range(0,8,4):
        batch=queries[start:start+4];b=len(batch);rows=[q['array_row'] for q in batch]
        predicted=cache['condition'][rows].clone().cuda();real=features[rows].clone().cuda()
        scales=real.norm(dim=-1)/predicted.norm(dim=-1)
        scaled=predicted*scales[:,None]
        assert torch.allclose(torch.nn.functional.cosine_similarity(predicted,scaled),torch.ones(b,device='cuda'),atol=1e-6)
        assert torch.allclose(real.norm(dim=-1),scaled.norm(dim=-1),atol=1e-5)
        arm_data={}
        for arm,c in [('EEG_REPLAY',predicted),('ORACLE_NORM_ONLY',scaled)]:
            folder=out/arm;folder.mkdir(exist_ok=True)
            seeds=[q['image_seed'] for q in batch]
            generators=[torch.Generator(device='cuda').manual_seed(s) for s in seeds]
            with torch.inference_mode():
                noise=randn_tensor((b,4,64,64),generator=generators,device='cuda',dtype=torch.bfloat16)
                assert [tensor_sha(v) for v in noise]==[q['initial_image_noise_sha256'] for q in batch]
                c=c.to(torch.bfloat16)
                z=pipe(prompt_embeds=prompt.repeat(b,1,1),negative_prompt_embeds=negative.repeat(b,1,1),
                    pooled_prompt_embeds=pool.repeat(b,1),negative_pooled_prompt_embeds=negative_pool.repeat(b,1),
                    ip_adapter_image_embeds=[torch.cat([torch.zeros_like(c),c])[:,None,:]],
                    height=512,width=512,num_inference_steps=20,guidance_scale=5.,latents=noise,output_type='latent').images
                assert torch.isfinite(z).all()
                decoded=pipe.vae.decode(z.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
                assert torch.isfinite(decoded).all()
                images=pipe.image_processor.postprocess(decoded,output_type='pil')
            paths=[]
            for query,image in zip(batch,images):
                path=folder/f'{query["array_row"]:08d}.jpg';image.save(path,quality=90,subsampling=0);paths.append(path)
            arm_data[arm]=dict(paths=paths,latents=z.cpu(),condition=c.float().cpu())
        for i,q in enumerate(batch):
            replay=arm_data['EEG_REPLAY']['paths'][i];scaled_path=arm_data['ORACLE_NORM_ONLY']['paths'][i]
            assert sha(q['generated_path'])==q['generated_sha256']
            assert sha(replay)==q['generated_sha256'],'Require exact point-condition replay before interpreting a norm intervention'
            old_oracle=oracle_by_id[q['query_id']];assert sha(old_oracle['oracle_path'])==old_oracle['oracle_sha256']
            results.append(dict(query_id=q['query_id'],stimulus_id=q['stimulus_id'],array_row=q['array_row'],
                target_path=q['QA_extracted_original_target_path'],formal_EEG_path=q['generated_path'],
                EEG_replay_path=str(replay),EEG_replay_sha256=sha(replay),formal_JPEG_replay_exact=True,
                oracle_norm_path=str(scaled_path),oracle_norm_sha256=sha(scaled_path),
                true_native_condition_oracle_path=old_oracle['oracle_path'],image_seed=q['image_seed'],
                initial_image_noise_sha256=q['initial_image_noise_sha256'],
                raw_EEG_norm=float(predicted[i].norm()),true_native_norm=float(real[i].norm()),
                oracle_norm_scale=float(scales[i]),FP32_input_direction_cosine=float(torch.nn.functional.cosine_similarity(predicted[i:i+1],scaled[i:i+1])[0]),
                deployed_BF16_direction_cosine=float(torch.nn.functional.cosine_similarity(
                    arm_data['EEG_REPLAY']['condition'][i:i+1],arm_data['ORACLE_NORM_ONLY']['condition'][i:i+1])[0]),
                final_latent_max_abs_delta=float((arm_data['EEG_REPLAY']['latents'][i]-arm_data['ORACLE_NORM_ONLY']['latents'][i]).abs().max())))
    handle.remove();assert counts==dict(calls=80,sample_evaluations=640)
    sheet=Image.new('RGB',(1500,1080),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'FIXED8 DIAGNOSTIC: TARGET | FORMAL EEG | ORACLE NORM ONLY | TRUE CLIP; no scores, no training',fill='black')
    for i,q in enumerate(results):
        x=8+(i%2)*748;y=36+(i//2)*260
        draw.text((x,y),f'{q["array_row"]} {q["stimulus_id"]}, norm scale={q["oracle_norm_scale"]:.3f}',fill='black')
        for j,path in enumerate([q['target_path'],q['formal_EEG_path'],q['oracle_norm_path'],q['true_native_condition_oracle_path']]):
            with Image.open(path) as im:small=ImageOps.contain(im.convert('RGB'),(178,220))
            tile=Image.new('RGB',(182,224),'#eeeeee');tile.paste(small,((182-small.width)//2,(224-small.height)//2));sheet.paste(tile,(x+j*185,y+24))
    contact=ROOT/'audit/V_fixed8_oracle_norm_contact.jpg';sheet.save(contact,quality=90,subsampling=0)
    report=dict(status='FIXED8_PRIVILEGED_NORM_INTERVENTION_COMPLETED_NO_FORMAL_METRICS',queries=results,
        input_information='Predicted EEG direction plus query true-image native feature norm; privileged diagnostic only',
        formal_image_method=False,all8_point_formal_JPEGs_replayed_exactly=True,images_generated=16,
        UNet_counts=counts,training_updates=0,formal_image_scores_computed=False,final_test_accessed=False,
        resolution=512,steps=20,CFG=5.,IP_scale=1.,contact_sheet=str(contact),
        source_sha256=sha(__file__),seconds=time.monotonic()-started,
        limits=['Eight predetermined V examples; no full-partition performance, blinded accuracy or capacity inference.',
            'True per-query norm is unavailable at formal inference. No deployable norm rule or new checkpoint is selected.',
            'BFloat16 quantization introduces a minute directional difference, quantified per query.',
            'A single norm-matching intervention does not exclude all possible normalization/interface problems.'])
    p=ROOT/'audit/V_fixed8_oracle_norm_diagnosis.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='queries'}),flush=True)


if __name__=='__main__':main()
