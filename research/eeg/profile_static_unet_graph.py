"""T-only eager/graph actual-image timing and numerical comparison, one mode.

Fixed512/20/Euler/CFG5/IP1, same original noise/weights, no formal gain claim.
This can neither replace existing formal files nor authorize unmatched methods.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import time
import zipfile

from profile_standard_generator import load_pipeline,torch,ROOT
from generate_reference_images import randn_tensor,tensor_sha
from static_unet_cuda_graph import install_static_graph
from PIL import Image,ImageDraw,ImageOps
import numpy as np


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    start=time.monotonic();cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    frozen=json.loads((ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json').read_text())
    for name,digest in frozen['source_sha256'].items():assert sha(ROOT/name)==digest
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text());t=set(protocol['image_rows']['T'])
    metadata={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    rows=[];concepts=set()
    for i in sorted(t):
        if metadata[i]['concept_id'] not in concepts:rows.append(i);concepts.add(metadata[i]['concept_id'])
        if len(rows)==4:break
    assert len(rows)==4
    query_map={}
    with (ROOT/'audit/query_manifest_R4.csv').open() as file:
        for q in csv.DictReader(file):
            if q['subject_id']=='sub-01' and q['role']=='T' and int(q['array_row']) in rows:
                query_map[int(q['array_row'])]=q
    assert len(query_map)==4
    features=torch.load(cfg['visual_features_path'],weights_only=True,mmap=True)
    assert sha(cfg['visual_features_path'])==cfg['visual_features_sha256']
    out=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/diagnostics/T_static_unet_graph_cost_v1')
    assert not out.exists();out.mkdir(parents=True)
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline();pipe.set_progress_bar_config(disable=True)
    c=features[rows].clone().cuda().to(torch.bfloat16)
    seeds=[int(query_map[i]['generation_seed']) for i in rows]
    with torch.inference_mode():
        noise=randn_tensor((4,4,64,64),generator=[torch.Generator(device='cuda').manual_seed(s) for s in seeds],
            device='cuda',dtype=torch.bfloat16)
    arguments=dict(prompt_embeds=prompt.repeat(4,1,1),negative_prompt_embeds=negative.repeat(4,1,1),
        pooled_prompt_embeds=pool.repeat(4,1),negative_pooled_prompt_embeds=negative_pool.repeat(4,1),
        ip_adapter_image_embeds=[torch.cat([torch.zeros_like(c),c])[:,None,:]],
        height=512,width=512,num_inference_steps=20,guidance_scale=5.,output_type='latent')
    measurements={};outputs={};graph=None
    for mode in ['eager','CUDA_graph']:
        if mode=='CUDA_graph':graph=install_static_graph(pipe.unet)
        samples=[]
        for repetition in [0,1]:
            torch.cuda.synchronize();tick=time.monotonic()
            with torch.inference_mode():z=pipe(latents=noise.clone(),**arguments).images
            torch.cuda.synchronize();elapsed=time.monotonic()-tick
            assert torch.isfinite(z).all();samples.append(z.cpu())
            measurements[f'{mode}_{repetition}']=elapsed
            (out/'progress.json').write_text(json.dumps(dict(stage=mode,repetition=repetition,seconds=elapsed))+'\n')
            print(json.dumps(dict(stage=mode,repetition=repetition,seconds=elapsed)),flush=True)
        assert torch.equal(samples[0],samples[1]),'Repeated identical query streams must replay exactly within a backend'
        outputs[mode]=samples[-1]
    equal=torch.equal(outputs['eager'],outputs['CUDA_graph'])
    delta=outputs['eager'].float()-outputs['CUDA_graph'].float()
    relative=float(delta.square().mean().sqrt()/outputs['eager'].float().square().mean().sqrt())
    paths={};pixels={}
    for mode,value in outputs.items():
        with torch.inference_mode():
            decoded=pipe.vae.decode(value.cuda().float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
            images=pipe.image_processor.postprocess(decoded,output_type='pil')
        folder=out/mode;folder.mkdir();paths[mode]=[];pixels[mode]=[]
        for row,im in zip(rows,images):
            path=folder/f'{row:08d}.jpg';im.save(path,quality=90,subsampling=0)
            paths[mode].append(str(path));pixels[mode].append(np.array(im))
    speedup=measurements['eager_1']/measurements['CUDA_graph_1']
    samples=[]
    for j,i in enumerate(rows):
        m=metadata[i]
        with zipfile.ZipFile(m['image_zip']) as archive:raw=archive.read(m['image_member'])
        assert hashlib.sha256(raw).hexdigest()==m['image_sha256']
        target=out/f'target_{i:08d}.jpg';target.write_bytes(raw)
        samples.append(dict(query_id=query_map[i]['query_id'],array_row=i,stimulus_id=m['stimulus_id'],image_seed=seeds[j],
            original_noise_sha256=tensor_sha(noise[j]),target=str(target),eager=paths['eager'][j],CUDA_graph=paths['CUDA_graph'][j],
            JPEG_bitwise_equal=sha(paths['eager'][j])==sha(paths['CUDA_graph'][j]),
            pre_JPEG_pixel_RMSE=float(np.sqrt(np.mean((pixels['eager'][j].astype(float)-pixels['CUDA_graph'][j])**2)))))
    sheet=Image.new('RGB',(1000,560),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'FIXED T: TARGET | EAGER | CUDA GRAPH; same noise/model/512/20steps/CFG5/IP1; runtime only',fill='black')
    for j,q in enumerate(samples):
        x=8+(j%2)*496;y=36+(j//2)*260;draw.text((x,y),q['stimulus_id'],fill='black')
        for k,name in enumerate(['target','eager','CUDA_graph']):
            with Image.open(q[name]) as im:small=ImageOps.contain(im.convert('RGB'),(156,220))
            tile=Image.new('RGB',(160,224),'#eeeeee');tile.paste(small,((160-small.width)//2,(224-small.height)//2))
            sheet.paste(tile,(x+k*164,y+22))
    contact=ROOT/'audit/T_static_unet_graph_contact.jpg';sheet.save(contact,quality=90,subsampling=0)
    a=dict(status='FIXED_T_RUNTIME_PROFILE_COMPLETED_NOT_FORMAL_METHOD',torch=torch.__version__,
        CUDA_graph_supported=True,conditions='true native image features for T-only runtime/oracle check',
        resolution=512,steps=20,CFG=5.,IP_scale=1.,training_updates=0,queries=4,
        measured_pipeline_seconds=measurements,steady_pipeline_speedup=speedup,
        per_image_UNet_pipeline_seconds_eager=measurements['eager_1']/4,
        per_image_UNet_pipeline_seconds_graph=measurements['CUDA_graph_1']/4,
        full_latents_bitwise_equal=equal,latent_relative_RMSE=relative,graph_counts=graph.counts,
        samples=samples,contact_sheet=str(contact),peak_cuda_allocated=torch.cuda.max_memory_allocated(),
        generator_revision=receipt['revision'],source_sha256=sha(__file__),graph_source_sha256=sha(ROOT/'static_unet_cuda_graph.py'),
        all_query_reuse_of_old_images_proven=False,formal_generation_runtime_changed=False,
        final_test_accessed=False,formal_image_scores_computed=False,seconds=time.monotonic()-start,
        limits=['Four fixed T queries; no EEG reconstruction, capacity or full-set numerical-equivalence claim.',
            'Timing excludes VAE/JPEG saving; first-run setup and steady timing are reported separately.',
            'One fixed graph shape/mode; no automatic fallback or optimization grid.',
            'Warmup/capture model evaluations are extra compute and must be included in any promoted runtime budget.',
            'Any formal use requires full matched baseline/control/candidate accounting and source/protocol provenance.'])
    p=ROOT/'audit/static_unet_graph_cost_profile.json';assert not p.exists();p.write_text(json.dumps(a,indent=2)+'\n')
    print(json.dumps({k:v for k,v in a.items() if k!='samples'}),flush=True)


if __name__=='__main__':main()
