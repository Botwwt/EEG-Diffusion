"""Exactly the fixed8 V QA queries, true native CLIP, identical image noise.

Diagnostic only. No target latent initialization, weight training, new seed or
formal-method ranking. This is not the complete16600-query V oracle arm.
"""
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
    assert all(q['partition']=='V' and q['subject']=='sub-01' for q in queries)
    assert sha(Path(cfg['visual_features_path']))==cfg['visual_features_sha256']
    features=torch.load(cfg['visual_features_path'],weights_only=True,mmap=True)
    output=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/diagnostics/V_fixed8_true_native_oracle_v1')
    assert not output.exists();output.mkdir(parents=True)
    pipe,infos,model_receipt,(prompt,negative,pool,negative_pool)=load_pipeline();pipe.set_progress_bar_config(disable=True)
    count=dict(calls=0,sample_evaluations=0)
    def hook(module,inputs):count['calls']+=1;count['sample_evaluations']+=len(inputs[0])
    handle=pipe.unet.register_forward_pre_hook(hook);results=[]
    for start in range(0,8,4):
        batch=queries[start:start+4];b=len(batch);indices=[q['array_row'] for q in batch]
        c=features[indices].clone().cuda();seeds=[q['image_seed'] for q in batch]
        generators=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
        with torch.inference_mode():
            noise=randn_tensor((b,4,64,64),generator=generators,device='cuda',dtype=torch.bfloat16)
            hashes=[tensor_sha(value) for value in noise]
            assert hashes==[q['initial_image_noise_sha256'] for q in batch]
            condition=c.to(torch.bfloat16)
            latents=pipe(prompt_embeds=prompt.repeat(b,1,1),negative_prompt_embeds=negative.repeat(b,1,1),
                pooled_prompt_embeds=pool.repeat(b,1),negative_pooled_prompt_embeds=negative_pool.repeat(b,1),
                ip_adapter_image_embeds=[torch.cat([torch.zeros_like(condition),condition])[:,None,:]],
                height=512,width=512,num_inference_steps=20,guidance_scale=5.,latents=noise,output_type='latent').images
            assert torch.isfinite(latents).all()
            decoded=pipe.vae.decode(latents.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
            assert torch.isfinite(decoded).all();images=pipe.image_processor.postprocess(decoded,output_type='pil')
        for i,(query,image) in enumerate(zip(batch,images)):
            path=output/f'{query["array_row"]:08d}.jpg';image.save(path,quality=90,subsampling=0)
            assert sha(Path(query['generated_path']))==query['generated_sha256']
            assert sha(Path(query['QA_extracted_original_target_path']))==query['target_image_sha256']
            results.append(dict(query_id=query['query_id'],stimulus_id=query['stimulus_id'],concept_id=query['concept_id'],
                array_row=query['array_row'],target_path=query['QA_extracted_original_target_path'],target_sha256=query['target_image_sha256'],
                EEG_generated_path=query['generated_path'],EEG_generated_sha256=query['generated_sha256'],oracle_path=str(path),oracle_sha256=sha(path),
                image_seed=query['image_seed'],initial_image_noise_sha256=hashes[i],real_native_norm=float(c[i].norm()),
                target_image_features_used_only_in_this_diagnostic=True))
    handle.remove();assert count['calls']==40 and count['sample_evaluations']==320
    sheet=Image.new('RGB',(1100,1050),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'FIXED8 PARTIAL V DIAGNOSIS: TARGET | EEG REG | TRUE CLIP ORACLE; same noise; no scores',fill='black')
    for i,row in enumerate(results):
        x=8+(i%2)*548;y=36+(i//2)*250
        draw.text((x,y),f'{row["array_row"]} {row["stimulus_id"]}',fill='black')
        draw.text((x,y+14),'TARGET               EEG REG             TRUE CLIP ORACLE',fill='black')
        for j,path in enumerate([row['target_path'],row['EEG_generated_path'],row['oracle_path']]):
            with Image.open(path) as im:image=ImageOps.contain(im.convert('RGB'),(172,200))
            canvas=Image.new('RGB',(176,204),'#eeeeee');canvas.paste(image,((176-image.width)//2,(204-image.height)//2))
            sheet.paste(canvas,(x+j*180,y+32))
    contact=ROOT/'audit/V_fixed8_oracle_contact.jpg';sheet.save(contact,quality=90,subsampling=0)
    report=dict(status='FIXED8_V_NATIVE_ORACLE_DIAGNOSTIC_COMPLETED_NOT_FULL_EVALUATION',queries=results,
        same_image_noise_as_actual_EEG_images=True,generator_revision=model_receipt['revision'],resolution=512,
        steps=20,CFG=5.,IP_scale=1.,UNet_counts=count,training_updates=0,formal_image_metrics_computed=False,
        full_V_hypothesis_rejected=False,final_test_accessed=False,contact_sheet=str(contact),
        source_sha256=sha(__file__),seconds=time.monotonic()-started,
        limits=['Exactly the predetermined first8 distinct V concepts; not representative full-V accuracy or formal blinded human scoring.',
            'True image features are oracle-only; formal EEG inference does not read them.',
            'Oracle success would support an EEG-condition bottleneck, not identify capacity versus objective/optimization/data limitations.'])
    (ROOT/'audit/V_fixed8_oracle_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['queries','limits']}),flush=True)


if __name__=='__main__':main()
