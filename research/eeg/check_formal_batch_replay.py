"""One original complete V batch: replay and single-query true-CLIP intervention.

Diagnostic case is the fixed8 badger query (native retrieval Top5 but wrong
object). No seed search or partial-partition method score is performed.
"""
from pathlib import Path
import csv
import json
import hashlib
import time
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from profile_standard_generator import load_pipeline,torch,ROOT
from generate_reference_images import tensor_sha,randn_tensor,sha


def main():
    started=time.monotonic();config=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    interface=json.loads((ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json').read_text())
    for name,expected in interface['source_sha256'].items():assert sha(ROOT/name)==expected
    fixed=json.loads((ROOT/'audit/V_fixed8_partial_visual_QA.json').read_text())['queries']
    target=next(q for q in fixed if q['stimulus_id']=='badger_01b.jpg')
    queries=[]
    with (ROOT/'audit/query_manifest_R4.csv').open() as file:
        for row in csv.DictReader(file):
            if row['subject_id']=='sub-01' and row['role']=='V':queries.append(row)
    position=next(i for i,q in enumerate(queries) if q['stimulus_id']==target['stimulus_id'])
    start=(position//4)*4;batch=queries[start:start+4];changed=position-start;assert len(batch)==4
    image_root=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/images/S1_full_native_batchsafe_v2/V/REF_REG_correct_R4')
    metas=[json.loads((image_root/'receipts/sub-01'/f'{int(q["array_row"]):08d}.json').read_text()) for q in batch]
    proof=json.loads((Path(config['output_root'])/'full/sub-01/completion.json').read_text())
    assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
    cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
    assert cache['encoder_sha256']==proof['encoder']['checkpoint_sha256']
    rows=[int(q['array_row']) for q in batch];c=cache['condition'][rows].clone().cuda()
    for i,m in enumerate(metas):
        assert m['query_id']==batch[i]['query_id'] and m['encoder_sha256']==proof['encoder']['checkpoint_sha256']
        assert tensor_sha(c[i])==m['EEG_native_condition_sha256']==m['deployed_native_condition_sha256']
        assert sha(m['generated_path'])==m['generated_sha256']
    assert sha(config['visual_features_path'])==config['visual_features_sha256']
    features=torch.load(config['visual_features_path'],weights_only=True,mmap=True)
    intervention=c.clone();intervention[changed]=features[rows[changed]].cuda()
    assert all(torch.equal(c[i],intervention[i]) for i in range(4) if i!=changed)
    output=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/diagnostics/V_original_batch_single_query_intervention_v1')
    assert not output.exists();output.mkdir(parents=True)
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline();pipe.set_progress_bar_config(disable=True)
    nfe=dict(calls=0,sample_evaluations=0)
    def count(module,inputs):nfe['calls']+=1;nfe['sample_evaluations']+=len(inputs[0])
    handle=pipe.unet.register_forward_pre_hook(count)
    generators=[torch.Generator(device='cuda').manual_seed(m['image_seed']) for m in metas]
    noise=randn_tensor((4,4,64,64),generator=generators,device='cuda',dtype=torch.bfloat16)
    assert [tensor_sha(value) for value in noise]==[m['initial_image_noise_sha256'] for m in metas]
    images={};latent_states={}
    for label,native in [('EEG_REPLAY',c),('ONLY_ONE_TRUE_CONDITION',intervention)]:
        value=native.to(torch.bfloat16)
        with torch.inference_mode():
            latents=pipe(prompt_embeds=prompt.repeat(4,1,1),negative_prompt_embeds=negative.repeat(4,1,1),
                pooled_prompt_embeds=pool.repeat(4,1),negative_pooled_prompt_embeds=negative_pool.repeat(4,1),
                ip_adapter_image_embeds=[torch.cat([torch.zeros_like(value),value])[:,None,:]],height=512,width=512,
                num_inference_steps=20,guidance_scale=5.,latents=noise.clone(),output_type='latent').images
            assert torch.isfinite(latents).all();latent_states[label]=latents.cpu()
            decoded=pipe.vae.decode(latents.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
            assert torch.isfinite(decoded).all();pictures=pipe.image_processor.postprocess(decoded,output_type='pil')
        folder=output/label;folder.mkdir();images[label]=[]
        for m,picture in zip(metas,pictures):
            path=folder/f'{m["array_row"]:08d}.jpg';picture.save(path,quality=90,subsampling=0);images[label].append(path)
    handle.remove();assert nfe['calls']==40 and nfe['sample_evaluations']==320
    rows_out=[]
    for i,m in enumerate(metas):
        with Image.open(m['generated_path']) as im:a=np.asarray(im.convert('RGB'),dtype=np.float32)
        with Image.open(images['EEG_REPLAY'][i]) as im:b=np.asarray(im.convert('RGB'),dtype=np.float32)
        rows_out.append(dict(query_id=m['query_id'],stimulus_id=m['stimulus_id'],array_row=m['array_row'],condition_changed=i==changed,
            actual_formal_path=m['generated_path'],actual_formal_sha256=m['generated_sha256'],replay_path=str(images['EEG_REPLAY'][i]),
            replay_sha256=sha(images['EEG_REPLAY'][i]),intervention_path=str(images['ONLY_ONE_TRUE_CONDITION'][i]),
            intervention_sha256=sha(images['ONLY_ONE_TRUE_CONDITION'][i]),image_seed=m['image_seed'],initial_image_noise_sha256=m['initial_image_noise_sha256'],
            exact_formal_JPEG_replay=sha(images['EEG_REPLAY'][i])==m['generated_sha256'],formal_replay_pixel_RMSE=float(np.mean((a-b)**2)**.5),
            intervention_latent_max_abs_delta=float((latent_states['EEG_REPLAY'][i]-latent_states['ONLY_ONE_TRUE_CONDITION'][i]).abs().max()),
            intervention_JPEG_identical_to_replay=sha(images['EEG_REPLAY'][i])==sha(images['ONLY_ONE_TRUE_CONDITION'][i])))
    unchanged=[r for r in rows_out if not r['condition_changed']]
    assert all(r['intervention_latent_max_abs_delta']==0 and r['intervention_JPEG_identical_to_replay'] for r in unchanged),'A query intervention affected another query'
    sheet=Image.new('RGB',(940,1050),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'ORIGINAL FULL BATCH REPLAY / ONLY BADGER CONDITION REPLACED; no new seed or score',fill='black')
    for i,row in enumerate(rows_out):
        y=35+i*250;draw.text((8,y),row['stimulus_id']+(' [CHANGED]' if row['condition_changed'] else ' [condition unchanged]'),fill='black')
        for j,path in enumerate([row['actual_formal_path'],row['replay_path'],row['intervention_path']]):
            with Image.open(path) as im:image=ImageOps.contain(im.convert('RGB'),(290,200))
            canvas=Image.new('RGB',(300,204),'#eeeeee');canvas.paste(image,((300-image.width)//2,(204-image.height)//2))
            draw.text((8+j*310,y+14),['FORMAL EEG','SAME-BATCH EEG REPLAY','SINGLE-QUERY TRUE CONDITION'][j],fill='black')
            sheet.paste(canvas,(8+j*310,y+30))
    contact=ROOT/'audit/V_original_batch_replay_contact.jpg';sheet.save(contact,quality=90,subsampling=0)
    report=dict(status='FULL_TRAJECTORY_QUERY_ISOLATION_PASSED_FORMAL_REPLAY_RECORDED',subject='sub-01',queries=rows_out,
        diagnostic_case_selection='Previously fixed badger query; nativeTop5 but visibly wrong EEG reconstruction; diagnostic only',
        exact_original_formal_JPEG_replay_all_queries=all(r['exact_formal_JPEG_replay'] for r in rows_out),
        unchanged_query_count=3,all_unchanged_query_latent_deltas_zero=True,changed_query_index=changed,
        same_initial_noise_all_conditions=True,UNet_counts=nfe,training_updates=0,formal_image_scores_computed=False,
        final_test_accessed=False,contact_sheet=str(contact),source_sha256=sha(__file__),seconds=time.monotonic()-started,
        limits=['One original complete batch, not full V performance or proof of the encoder capacity cause.',
            'True image condition is privileged oracle input used only for this diagnostic.'])
    (ROOT/'audit/V_original_batch_replay_and_isolation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['queries','limits']}),flush=True)


if __name__=='__main__':main()
