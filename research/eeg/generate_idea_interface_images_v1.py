"""Actual fixed-seed EEG inference for frozen and all five interface arms.

No target image/visual tensor reads. T-readiness is explicitly diagnostic;
formal V always contains all16600 queries. Historical images are never reused
across nonidentical full encoder or projection checkpoints.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import time
from generate_reference_images import (torch,ROOT,sha,tensor_sha,query_seed,prior_seed,
    permuted_sources,ATMS,randn_tensor)
from profile_standard_generator import load_pipeline

ARMS=['Frozen','Clean','Iso','Moment','Conditional','RealPaired']


def load(path):return json.loads(Path(path).read_text())


def write(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser()
    for name in ['recipe','runtime-config','full-training-config','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--adapter-completion')
    parser.add_argument('--partition',choices=['V','T_readiness'],default='V')
    parser.add_argument('--condition',choices=['correct','permuted','missing'],default='correct')
    args=parser.parse_args();started=time.monotonic()
    parent=Path('/proc')/str(os.getppid())/'cmdline'
    assert parent.exists() and str(ROOT/'run_gpu_job.py').encode() in parent.read_bytes(),'Use registered total-budget wrapper'
    recipe_path=Path(args.recipe);recipe=load(recipe_path);runtime_path=Path(args.runtime_config);runtime=load(runtime_path)
    assert recipe['seed']==1729 and runtime['recipe_sha256']==sha(recipe_path)
    assert runtime['resolution']==512 and runtime['steps']==20 and runtime['CFG']==5. and runtime['IP_scale']==1.
    assert runtime['generation_batch'] in [4,16]
    for name,digest in runtime['source_sha256'].items():assert sha(ROOT/name)==digest,name
    config_path=Path(args.full_training_config);config=load(config_path)
    assert runtime['full_training_config_sha256']==sha(config_path)
    cache_config=load(runtime['native_cache_config_path'])
    assert sha(runtime['native_cache_config_path'])==runtime['native_cache_config_sha256']
    assert cache_config['full_reference_root']==str(Path(config['output_root'])/'full')
    assert config['study']=='idea_first_direct_ATMS_v1' and config['encoder_patience']==50
    assert config['condition_chain']=='REF_REG' and config['seed']==1729
    protocol=load(ROOT/'configs/split_manifest.json');assert sha(ROOT/'configs/split_manifest.json')==config['split_sha256']
    root=Path(config['output_root'])/'full';assert load(root/'completion.json')['status']=='COMPLETED'
    assert len(load(root/'completion.json')['subjects'])==10
    full_sha=sha(config_path)
    if args.arm=='Frozen':assert args.adapter_completion is None
    else:
        assert args.adapter_completion
        adapted=load(args.adapter_completion)
        assert adapted['status']=='COMPLETED_REGISTERED_MATCHED_FULL_QUERY_PASS' and adapted['arm']==args.arm
        assert adapted['recipe_sha256']==sha(recipe_path) and adapted['seen_training_queries']==132300
        assert adapted['each_subject_query_coverage']==[13230]*10 and adapted['updates']==(132300+recipe['batch_size']-1)//recipe['batch_size']
        assert sha(adapted['checkpoint'])==adapted['checkpoint_sha256']
        adapter_input=Path(args.adapter_completion).parent/'input_provenance.json'
        assert sha(adapter_input)==adapted['source_input_provenance_sha256']
        assert load(adapter_input)['cache_config_sha256']==runtime['native_cache_config_sha256']
    queries=[]
    with (ROOT/'audit/query_manifest_R4.csv').open() as file:
        for row in csv.DictReader(file):
            if row['role']==('V' if args.partition=='V' else 'T'):
                assert int(row['generation_seed'])==query_seed(row['subject_id'],row['stimulus_id'],'R4',protocol['version'])
                queries.append(row)
    if args.partition=='T_readiness':
        chosen=[]
        for subject in protocol['subjects']:
            seen=set()
            for row in sorted((q for q in queries if q['subject_id']==subject),key=lambda q:int(q['array_row'])):
                if row['concept_id'] in seen:continue
                chosen.append(row);seen.add(row['concept_id'])
                if len(seen)==4:break
        queries=chosen
    expected=16600 if args.partition=='V' else 40
    assert len(queries)==expected and len({q['query_id'] for q in queries})==expected
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'completion.json').exists()
    lock=dict(arm=args.arm,intervention=args.condition,partition=args.partition,queries=expected,
        recipe_sha256=sha(recipe_path),runtime_sha256=sha(runtime_path),full_training_config_sha256=full_sha,
        adapter_completion_sha256=sha(args.adapter_completion) if args.adapter_completion else None,
        target_images_read=False,query_ids=[q['query_id'] for q in queries])
    if (output/'input_provenance.json').exists():assert load(output/'input_provenance.json')==lock
    else:write(output/'input_provenance.json',lock)
    pipe,infos,model_receipt,prompts=load_pipeline();pipe.set_progress_bar_config(disable=True)
    if args.arm!='Frozen':assert adapted['generator_revision']==model_receipt['revision']
    prompt,negative,pool,negative_pool=prompts
    if args.arm!='Frozen':
        trained=torch.load(adapted['checkpoint'],map_location='cpu',weights_only=True)
        assert trained['arm']==args.arm and trained['recipe_sha256']==sha(recipe_path)
        selected={name:p for name,p in pipe.unet.named_parameters() if name.startswith('encoder_hid_proj.')}
        assert selected.keys()==trained['state_dict'].keys()
        assert sum(p.numel() for p in selected.values())==8400896
        with torch.no_grad():
            for name,parameter in selected.items():
                parameter.copy_(trained['state_dict'][name].to(parameter.device,dtype=parameter.dtype))
                assert torch.equal(parameter.cpu(),trained['state_dict'][name].to(parameter.dtype))
    actual_projection={name:tensor_sha(p.detach().cpu()) for name,p in pipe.unet.named_parameters() if name.startswith('encoder_hid_proj.')}
    counts=dict(calls=0,sample_evaluations=0)
    def count(module,inputs):counts['calls']+=1;counts['sample_evaluations']+=len(inputs[0])
    hook=pipe.unet.register_forward_pre_hook(count);records=[];reused=0;batch_size=runtime['generation_batch']
    for subject in protocol['subjects']:
        selected_queries=[q for q in queries if q['subject_id']==subject]
        proof=load(root/subject/'completion.json');assert proof['prior'] is None
        assert sha(proof['encoder']['checkpoint'])==proof['encoder']['checkpoint_sha256']
        assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
        saved=torch.load(proof['condition_cache'],map_location='cpu',weights_only=True)
        assert saved['encoder_sha256']==proof['encoder']['checkpoint_sha256'] and saved['training_role']=='full'
        assert torch.equal(saved['array_rows'],torch.arange(16540))
        positions=[int(q['array_row']) for q in selected_queries]
        native=saved['condition'][positions].clone().cuda();sources=selected_queries
        if args.condition=='permuted':
            permutation=permuted_sources(selected_queries,subject,'V' if args.partition=='V' else 'T_readiness','R4')
            by_id={q['stimulus_id']:i for i,q in enumerate(selected_queries)}
            sources=[permutation[q['stimulus_id']] for q in selected_queries]
            native=native[[by_id[q['stimulus_id']] for q in sources]]
        elif args.condition=='missing':
            encoder=ATMS().cuda().eval().requires_grad_(False)
            encoder.load_state_dict(torch.load(proof['encoder']['checkpoint'],map_location='cpu',weights_only=True),strict=True)
            with torch.inference_mode():value=encoder(torch.zeros(1,63,250,device='cuda'),torch.tensor([int(subject[-2:])],device='cuda')).float()
            native=value.repeat(len(selected_queries),1);sources=[None]*len(selected_queries);del encoder
        for start in range(0,len(selected_queries),batch_size):
            qbatch=selected_queries[start:start+batch_size];source_batch=sources[start:start+batch_size]
            folder=output/'images'/subject;folder.mkdir(parents=True,exist_ok=True)
            receipt_folder=output/'receipts'/subject;receipt_folder.mkdir(parents=True,exist_ok=True)
            paths=[folder/(Path(q['stimulus_id']).stem+'.jpg') for q in qbatch]
            metas=[receipt_folder/(Path(q['stimulus_id']).stem+'.json') for q in qbatch]
            if all(p.exists() for p in paths+metas):
                for q,path,meta in zip(qbatch,paths,metas):
                    item=load(meta);assert sha(path)==item['generated_sha256'] and item['query_id']==q['query_id']
                    assert item['input_provenance_sha256']==sha(output/'input_provenance.json');records.append(item);reused+=1
                continue
            assert not any(p.exists() for p in paths+metas),'Partial batch: preserve files and explicitly review orphan recovery; no overwrite/reroll'
            c=native[start:start+len(qbatch)];b=len(c);seeds=[int(q['generation_seed']) for q in qbatch]
            before=dict(counts);torch.cuda.synchronize();tick=time.monotonic()
            with torch.inference_mode():
                condition=c.to(torch.bfloat16)
                noise=randn_tensor((b,4,64,64),generator=[torch.Generator(device='cuda').manual_seed(seed) for seed in seeds],device='cuda',dtype=torch.bfloat16)
                latent=pipe(prompt_embeds=prompt.repeat(b,1,1),negative_prompt_embeds=negative.repeat(b,1,1),
                    pooled_prompt_embeds=pool.repeat(b,1),negative_pooled_prompt_embeds=negative_pool.repeat(b,1),
                    ip_adapter_image_embeds=[torch.cat([torch.zeros_like(condition),condition])[:,None,:]],height=512,width=512,
                    num_inference_steps=20,guidance_scale=5.,latents=noise,output_type='latent').images
                assert torch.isfinite(latent).all()
                decoded=pipe.vae.decode(latent.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
                assert torch.isfinite(decoded).all();images=pipe.image_processor.postprocess(decoded,output_type='pil')
            torch.cuda.synchronize();seconds=time.monotonic()-tick
            assert counts['calls']-before['calls']==20 and counts['sample_evaluations']-before['sample_evaluations']==40*b
            for i,(query,source,path,meta,image) in enumerate(zip(qbatch,source_batch,paths,metas,images)):
                temporary=path.with_name(path.name+'.tmp');image.save(temporary,format='JPEG',quality=90,subsampling=0);os.replace(temporary,path)
                item=dict(query_id=query['query_id'],subject=subject,stimulus_id=query['stimulus_id'],concept_id=query['concept_id'],
                    array_row=int(query['array_row']),repeat_protocol='R4',partition=args.partition,method='IDEA1_'+args.arm,intervention=args.condition,
                    EEG_source_stimulus_id=source['stimulus_id'] if source else None,
                    missing_input_definition='all-zero63x250 EEG through the actual frozen aligned encoder' if source is None else None,
                    target_image_zip=query['image_zip'],target_image_member=query['image_member'],target_image_sha256=query['image_sha256'],
                    generated_path=str(path),generated_sha256=sha(path),image_seed=seeds[i],prior_seed=prior_seed(seeds[i]),
                    initial_image_noise_sha256=tensor_sha(noise[i]),EEG_native_condition_sha256=tensor_sha(c[i]),
                    deployed_native_condition_sha256=tensor_sha(c[i]),native_norm=float(c[i].norm()),
                    encoder_checkpoint=proof['encoder']['checkpoint'],encoder_sha256=proof['encoder']['checkpoint_sha256'],
                    prior_checkpoint=None,prior_sha256=None,prior_prediction_type=None,
                    projection_checkpoint=adapted['checkpoint'] if args.arm!='Frozen' else None,
                    projection_checkpoint_sha256=adapted['checkpoint_sha256'] if args.arm!='Frozen' else None,
                    actual_deployed_projection_tensor_sha256=actual_projection,
                    generator_revision=model_receipt['revision'],steps=20,CFG=5.,IP_scale=1.,UNet_NFE=40,
                    image_interface_version='idea_first_query_batchsafe_native_projection_v1',
                    image_interface_config_sha256=sha(runtime_path),projected_native_image_tokens_per_query=4,
                    full_training_config_sha256=full_sha,recipe_sha256=sha(recipe_path),
                    input_provenance_sha256=sha(output/'input_provenance.json'),sampling_seconds_per_query_in_batch=seconds/b,
                    target_image_or_visual_feature_read_at_inference=False,formal_evaluation=args.partition=='V',
                    compression=dict(format='JPEG',quality=90,subsampling=0))
                write(meta,item);records.append(item)
            write(output/'progress.json',dict(status='RUNNING',arm=args.arm,completed=len(records),expected=expected,
                seconds=time.monotonic()-started,counts_this_attempt=counts,reused_query_outputs=reused))
            if len(records)%128 < batch_size:print(json.dumps(dict(stage='idea_actual_images',arm=args.arm,completed=len(records),expected=expected)),flush=True)
        del native,saved;torch.cuda.empty_cache()
    hook.remove();assert len(records)==expected and len({x['query_id'] for x in records})==expected
    with (output/'manifest.jsonl').open('w') as file:
        for row in records:file.write(json.dumps(row)+'\n')
    write(output/'completion.json',dict(status='COMPLETED',arm=args.arm,queries=expected,subjects=10,
        partition=args.partition,formal_evaluation=args.partition=='V',intervention=args.condition,
        image_manifest_sha256=sha(output/'manifest.jsonl'),seconds=time.monotonic()-started,
        counts=counts,reused_query_outputs=reused,total_counts_include_recovery_in_ledger=True,
        target_images_accessed=False,full_training_config_sha256=full_sha,recipe_sha256=sha(recipe_path),
        image_interface_config_sha256=sha(runtime_path),actual_deployed_projection_tensor_sha256=actual_projection))


if __name__=='__main__':main()
