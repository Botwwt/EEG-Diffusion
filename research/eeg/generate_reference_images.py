"""Actual standard-SDXL images from frozen full EEG references, one seed/query.

The image sampler never opens target images or target visual feature tensors.
Target ZIP pointers in receipts exist only for the separate scoring process.
"""
from profile_standard_generator import load_pipeline, torch, ROOT
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time

SOURCE=Path('/root/atm_official_20261016')
sys.path[:0]=[str(SOURCE),str(SOURCE/'Generation'),
    '/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages',
    '/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages']
from models.atms import ATMS
from diffusion_prior import DiffusionPriorUNet
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def write(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def query_seed(subject,stimulus,R,version):
    return int(hashlib.sha256(f'1729|{subject}|{stimulus}|{R}|{version}'.encode()).hexdigest()[:16],16)%(2**63-1)


def prior_seed(image_seed):
    return int(hashlib.sha256(f'{image_seed}|prior'.encode()).hexdigest()[:16],16)%(2**63-1)


def permuted_sources(queries,subject,partition,R):
    groups={}
    for query in queries:groups.setdefault(query['concept_id'],[]).append(query)
    concepts=sorted(groups);assert len(concepts)>1
    assert len({len(v) for v in groups.values()})==1
    shift=1+int(hashlib.sha256(f'1729|EEG-permutation|{subject}|{partition}|{R}'.encode()).hexdigest()[:16],16)%(len(concepts)-1)
    result={}
    for i,c in enumerate(concepts):
        source=groups[concepts[(i+shift)%len(concepts)]]
        for target,other in zip(sorted(groups[c],key=lambda q:int(q['array_row'])),sorted(source,key=lambda q:int(q['array_row']))):
            assert target['concept_id']!=other['concept_id'];result[target['stimulus_id']]=other
    assert len(result)==len(queries);return result


@torch.inference_mode()
def sample_prior(model,conditions,seeds,prediction_type='epsilon'):
    generators=[torch.Generator(device='cuda').manual_seed(prior_seed(seed)) for seed in seeds]
    scheduler=DDPMScheduler(clip_sample=False,prediction_type=prediction_type);scheduler.set_timesteps(50,device='cuda')
    h=randn_tensor((len(conditions),1024),generator=generators,device='cuda',dtype=torch.float32)
    for t in scheduler.timesteps:
        ts=torch.full((len(h),),float(t),device='cuda')
        conditional=model(h,ts,conditions);unconditional=model(h,ts,None)
        eps=unconditional+5.*(conditional-unconditional)
        h=scheduler.step(eps,int(t),h,generator=generators).prev_sample
    assert torch.isfinite(h).all();return h


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['REF_REG','REF_PRIOR','REF_PRIOR_V'],required=True)
    parser.add_argument('--condition',choices=['correct','permuted','missing'],default='correct')
    parser.add_argument('--partition',choices=['V','benchmark'],default='V');parser.add_argument('--R',choices=['R1','R4','R80'],default='R4')
    args=parser.parse_args();started=time.monotonic()
    interface_config=ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json'
    interface=json.loads(interface_config.read_text())
    for name,expected_sha in interface['source_sha256'].items():assert sha(ROOT/name)==expected_sha,(name,'frozen source mismatch')
    assert json.loads((ROOT/'audit/batched_native_condition_check.json').read_text())['status']=='PASSED_ACTUAL_UNET_QUERY_ISOLATION'
    config=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    training=Path(config['output_root'])/'full';assert (training/'completion.json').exists()
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    assert config['seed']==1729
    if args.partition=='benchmark':
        unlock=json.loads((ROOT/'configs/FINAL_EVAL_UNLOCK.json').read_text())
        assert unlock['method_and_checkpoint_selection_frozen'] and unlock['split_sha256']==config['split_sha256']
    else:assert args.R=='R4'
    assert sha(ROOT/'configs/split_manifest.json')==config['split_sha256']
    queries=[]
    with (ROOT/'audit/query_manifest_R4.csv').open() as file:
        for row in csv.DictReader(file):
            if (args.partition=='V' and row['role']=='V') or (args.partition=='benchmark' and row['dataset_split']=='test'):
                row['repeat_protocol']=args.R
                row['query_id']=f'{row["subject_id"]}/{row["stimulus_id"]}/{args.R}'
                actual_seed=query_seed(row['subject_id'],row['stimulus_id'],args.R,protocol['version'])
                if args.R=='R4':assert int(row['generation_seed'])==actual_seed
                row['generation_seed']=actual_seed
                queries.append(row)
    expected=16600 if args.partition=='V' else 2000;assert len(queries)==expected
    arm=f'{args.method}_{args.condition}_{args.R}'
    out=DATA/'images/S1_full_native_batchsafe_v2'/args.partition/arm;out.mkdir(parents=True,exist_ok=True)
    receipt_root=out/'receipts';receipt_root.mkdir(exist_ok=True)
    pipe,infos,model_receipt,(prompt,negative,pool,negative_pool)=load_pipeline()
    pipe.set_progress_bar_config(disable=True)
    nfe=dict(calls=0,sample_evaluations=0)
    def hook(module,inputs):nfe['calls']+=1;nfe['sample_evaluations']+=len(inputs[0])
    handle=pipe.unet.register_forward_pre_hook(hook)
    generated=[];condition_exports=[]
    for subject in protocol['subjects']:
        selected=[q for q in queries if q['subject_id']==subject]
        folder=training/subject;proof=json.loads((folder/'completion.json').read_text())
        prediction_type='epsilon'
        if args.method=='REF_PRIOR_V':
            repaired=json.loads((ROOT/'configs/prior_velocity_repair_resolved.json').read_text())
            repaired_root=Path(repaired['output_root'])/'full';assert (repaired_root/'completion.json').exists()
            repaired_proof=json.loads((repaired_root/subject/'completion.json').read_text())
            assert repaired_proof['encoder']['checkpoint_sha256']==proof['encoder']['checkpoint_sha256']
            proof=proof|dict(prior=repaired_proof['prior']);prediction_type='v_prediction'
        assert sha(proof['encoder']['checkpoint'])==proof['encoder']['checkpoint_sha256']
        assert sha(proof['prior']['checkpoint'])==proof['prior']['checkpoint_sha256']
        if args.partition=='V':
            cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
            assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
            assert cache['encoder_sha256']==proof['encoder']['checkpoint_sha256']
            mapping={int(i):k for k,i in enumerate(cache['array_rows'])}
            conditions=cache['condition'][[mapping[int(q['array_row'])] for q in selected]].cuda()
        else:
            data_rows=list(csv.DictReader((ROOT/'audit/data_manifest.csv').open()))
            row=next(r for r in data_rows if r['subject']==subject and r['dataset_split']=='test' and r['repeat_protocol']==args.R)
            assert sha(row['path'])==row['sha256']
            eeg=torch.load(row['path'],weights_only=True,mmap=True).clone().cuda()
            encoder=ATMS().cuda().eval().requires_grad_(False)
            encoder.load_state_dict(torch.load(proof['encoder']['checkpoint'],weights_only=True),strict=True)
            with torch.inference_mode():
                conditions=torch.cat([encoder(eeg[i:i+128],torch.full((len(eeg[i:i+128]),),int(subject[-2:]),dtype=torch.long,device='cuda')).float() for i in range(0,len(eeg),128)])
            del eeg,encoder
        sources=selected
        if args.condition=='permuted':
            permutation=permuted_sources(selected,subject,args.partition,args.R)
            source_positions={q['stimulus_id']:i for i,q in enumerate(selected)}
            sources=[permutation[q['stimulus_id']] for q in selected]
            conditions=conditions[[source_positions[q['stimulus_id']] for q in sources]]
        elif args.condition=='missing':
            encoder=ATMS().cuda().eval().requires_grad_(False)
            encoder.load_state_dict(torch.load(proof['encoder']['checkpoint'],weights_only=True),strict=True)
            with torch.inference_mode():
                zero=encoder(torch.zeros(1,63,250,device='cuda'),torch.tensor([int(subject[-2:])],device='cuda')).float()
            conditions=zero.repeat(len(selected),1);del encoder
            sources=[None]*len(selected)
        prior=None
        if args.method in ['REF_PRIOR','REF_PRIOR_V']:
            prior=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval().requires_grad_(False)
            prior.load_state_dict(torch.load(proof['prior']['checkpoint'],weights_only=True),strict=True)
        image_dir=out/subject;image_dir.mkdir(exist_ok=True)
        metadata_dir=receipt_root/subject;metadata_dir.mkdir(exist_ok=True)
        for start in range(0,len(selected),4):
            batch=selected[start:start+4];source_batch=sources[start:start+4]
            paths=[image_dir/f'{int(q["array_row"]):08d}.jpg' for q in batch]
            metas=[metadata_dir/f'{int(q["array_row"]):08d}.json' for q in batch]
            if all(p.exists() for p in metas):
                for path,meta,q in zip(paths,metas,batch):
                    row=json.loads(meta.read_text());assert sha(path)==row['generated_sha256'] and row['query_id']==q['query_id']
                    assert row['image_seed']==int(q['generation_seed']);generated.append(row)
                continue
            assert not any(p.exists() for p in paths+metas),'Partial batch requires explicit orphan recovery; do not overwrite scientific history'
            before=dict(nfe);torch.cuda.synchronize();tick=time.monotonic()
            seeds=[int(q['generation_seed']) for q in batch]
            eeg_condition=conditions[start:start+len(batch)]
            with torch.inference_mode():
                native=sample_prior(prior,eeg_condition,seeds,prediction_type=prediction_type) if prior is not None else eeg_condition
                c=native.to(torch.bfloat16);b=len(batch)
                image_generators=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
                noise=randn_tensor((b,4,64,64),generator=image_generators,device='cuda',dtype=torch.bfloat16)
                latents=pipe(prompt_embeds=prompt.repeat(b,1,1),negative_prompt_embeds=negative.repeat(b,1,1),
                    pooled_prompt_embeds=pool.repeat(b,1),negative_pooled_prompt_embeds=negative_pool.repeat(b,1),
                    ip_adapter_image_embeds=[torch.cat([torch.zeros_like(c),c])[:,None,:]],
                    height=512,width=512,num_inference_steps=20,guidance_scale=5.,latents=noise,output_type='latent').images
                assert torch.isfinite(latents).all()
                decoded=pipe.vae.decode(latents.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
                assert torch.isfinite(decoded).all();images=pipe.image_processor.postprocess(decoded,output_type='pil')
            torch.cuda.synchronize();seconds=time.monotonic()-tick
            assert nfe['calls']-before['calls']==20 and nfe['sample_evaluations']-before['sample_evaluations']==40*b
            for i,(q,source,path,meta,image) in enumerate(zip(batch,source_batch,paths,metas,images)):
                image.save(path,format='JPEG',quality=90,subsampling=0)
                row=dict(query_id=q['query_id'],subject=subject,stimulus_id=q['stimulus_id'],concept_id=q['concept_id'],array_row=int(q['array_row']),
                    repeat_protocol=args.R,partition=args.partition,method=args.method,intervention=args.condition,
                    EEG_source_stimulus_id=source['stimulus_id'] if source else None,missing_input_definition='all-zero 63x250 EEG through the actual frozen encoder' if source is None else None,
                    target_image_zip=q['image_zip'],target_image_member=q['image_member'],target_image_sha256=q['image_sha256'],
                    generated_path=str(path),generated_sha256=sha(path),image_seed=seeds[i],prior_seed=prior_seed(seeds[i]),
                    initial_image_noise_sha256=tensor_sha(noise[i]),EEG_native_condition_sha256=tensor_sha(eeg_condition[i]),
                    deployed_native_condition_sha256=tensor_sha(native[i]),native_norm=float(native[i].norm()),
                    encoder_checkpoint=proof['encoder']['checkpoint'],encoder_sha256=proof['encoder']['checkpoint_sha256'],
                    prior_checkpoint=proof['prior']['checkpoint'] if prior else None,prior_sha256=proof['prior']['checkpoint_sha256'] if prior else None,
                    prior_prediction_type=prediction_type if prior else None,
                    generator_revision=model_receipt['revision'],steps=20,CFG=5.,IP_scale=1.,UNet_NFE=40,
                    image_interface_version='native_query_batchsafe_v2',image_interface_config_sha256=sha(interface_config),
                    projected_native_image_tokens_per_query=4,
                    sampling_seconds_per_query_in_batch=seconds/b,target_image_or_visual_feature_read_at_inference=False,
                    compression=dict(format='JPEG',quality=90,subsampling=0),formal_image_metric_evaluated=False)
                write(meta,row);generated.append(row)
            condition_exports.append(dict(query_ids=[q['query_id'] for q in batch],native_condition=native.cpu()))
            if len(generated)%128<4:
                write(out/'progress.json',dict(status='RUNNING',arm=arm,completed=len(generated),expected=expected,seconds=time.monotonic()-started))
                print(json.dumps(dict(stage='image_generation',arm=arm,completed=len(generated),expected=expected,seconds=time.monotonic()-started)),flush=True)
        del conditions,prior;torch.cuda.empty_cache()
    handle.remove();assert len(generated)==expected and len({r['query_id'] for r in generated})==expected
    with (out/'manifest.jsonl').open('w') as file:
        for row in generated:file.write(json.dumps(row)+'\n')
    # Feature exports aid later auditing; image receipts remain the authoritative output.
    if condition_exports:torch.save(condition_exports,out/'generated_native_conditions.pt')
    write(out/'completion.json',dict(status='COMPLETED',arm=arm,queries=expected,seconds=time.monotonic()-started,
        counts=nfe,image_manifest_sha256=sha(out/'manifest.jsonl'),target_images_accessed=False))


if __name__=='__main__':main()
