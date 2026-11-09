"""Own-arm fixed T32 or full V native images; targets joined after inference."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import brainhive_paired_study_contract_20261025_v1 as C


def sample_native(pipe,prompts,tokens,image_seed,torch):
    """One native image, with real NFE/IP/VAE and initial-noise observation."""
    initial=[];counts=[0,0,0];decoded_count=[0]
    def unet_hook(module,arguments,result):
        counts[0]+=1
        if not initial:initial.append(hashlib.sha256(arguments[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        assert torch.isfinite(result[0]).all()
    def key_hook(module,arguments):
        assert arguments[0].shape==(1,4,2048) and arguments[0].data_ptr()==tokens.data_ptr();counts[1]+=1
    def value_hook(module,arguments):
        assert arguments[0].shape==(1,4,2048) and arguments[0].data_ptr()==tokens.data_ptr();counts[2]+=1
    def vae_hook(module,arguments,result):assert torch.isfinite(result).all();decoded_count[0]+=1
    handles=[pipe.unet.register_forward_hook(unet_hook),pipe.vae.decoder.register_forward_hook(vae_hook)]
    processors=[p for p in pipe.unet.attn_processors.values() if type(p).__name__=='IPAttnProcessor2_0'];assert len(processors)==70
    for processor in processors:handles += [processor.to_k_ip.register_forward_pre_hook(key_hook),processor.to_v_ip.register_forward_pre_hook(value_hook)]
    try:
        image=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(image_seed),
            prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':tokens},
            num_inference_steps=4,guidance_scale=0.,output_type='pil').images[0].convert('RGB')
    finally:
        for handle in handles:handle.remove()
    assert counts==[4,280,280] and decoded_count==[1] and image.size==(512,512)
    return image,dict(initial_UNet_input_sha256=initial[0],UNet_NFE=4,native_IP_K_calls=280,native_IP_V_calls=280)


def infer_queries(config,split):
    protocol=C.bound(config['input_bindings']['split'])
    contract=C.bound(config['input_bindings']['input_contract'])
    sources={(s['subject'],x['role']):x for s in contract['subjects'] for x in s['sources']}
    from brainhive_functional_training_contract_v2 import query_identity
    queries=[]
    if split=='V':
        inputs=C.bound(config['input_bindings']['V_inputs'])
        from generate_BrainHIVE_functional_alignment_V_v2 import validate_blind_inputs
        validate_blind_inputs(inputs,protocol)
        for subject in inputs['subjects']:
            source=sources[(subject,'full')]
            reg=C.read(source['parent_registration_path'])
            assert set(reg['training_array_rows'])==set(protocol['image_rows']['T'])
            for row in inputs['inputs']:
                identity,noise=query_identity(subject,row,protocol['version'])
                queries.append(dict(subject=subject,stimulus_id=row['stimulus_id'],array_row=row['array_row'],
                    query_id=identity,image_seed=noise,source_role='full'))
        assert len(queries)==16600
    else:
        document=C.bound(config['input_bindings']['postfit_queries'])
        assert len(document['rows'])==32
        for row in document['rows']:
            assert row['array_row'] in protocol['image_rows']['T']
            identity,noise=query_identity(row['subject'],row,protocol['version'])
            assert identity==row['query_id'] and noise==row['image_seed']
            source=sources[(row['subject'],row['source_role'])]
            assert row['condition_cache_sha256']==source['condition_cache_sha256']
            assert any(x['array_row']==row['array_row'] and x['cache_row']==row['cache_row'] for x in source['groups']['component_train'])
            queries.append({k:row[k] for k in ('subject','stimulus_id','array_row','query_id','image_seed','source_role')})
    for key in {(query['subject'],query['source_role']) for query in queries}:
        source=sources[key]
        for key,digest in [('condition_cache_path','condition_cache_sha256'),('checkpoint_path','checkpoint_sha256'),('parent_registration_path','parent_registration_sha256')]:
            assert C.sha(source[key])==source[digest]
    assert len({q['query_id'] for q in queries})==len(queries)
    return queries,sources


def verify_prefix(output,queries,arm,trained,study_core):
    path=output/'inference_manifest.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    assert len(rows)<=len(queries)
    for index,row in enumerate(rows):
        query=queries[index]
        for key in ('query_id','subject','stimulus_id','array_row','image_seed'):assert row[key]==query[key]
        assert row['global_query_index']==index and row['arm']==arm and row['study_core_sha256']==study_core
        assert row['component_checkpoint_sha256']==trained['checkpoint']['sha256']
        expected=output/'pixels'/query['subject']/(Path(query['stimulus_id']).stem+'.webp')
        assert row['generated_path']==str(expected) and C.sha(expected)==row['generated_sha256']
    actual=set((output/'pixels').rglob('*.webp'))
    assert actual=={Path(r['generated_path']) for r in rows},'Orphan images retained; explicit recovery required'
    assert not list(output.rglob('*.pending'))
    return rows


def attach_V_targets(config,output,rows):
    assert len(rows)==16600
    binding=config['input_bindings'];gallery_path=Path(binding['gallery_manifest']['path'])
    assert C.sha(gallery_path)==binding['gallery_manifest']['sha256']
    gallery={r['stimulus_id']:r for r in map(json.loads,gallery_path.read_text().splitlines())}
    target_path=Path(binding['V_targets']['path']);assert C.sha(target_path)==binding['V_targets']['sha256']
    targets={r['stimulus_id']:r for r in map(json.loads,target_path.read_text().splitlines())}
    assert len(gallery)==len(targets)==1660 and set(gallery)==set(targets)=={r['stimulus_id'] for r in rows}
    for stimulus,target in targets.items():
        actual=gallery[stimulus]
        assert target['image_sha256']==actual['image_sha256']==C.sha(actual['image_path'])
        assert target['concept_id']==actual['concept_id']
    joined=[]
    for row in rows:
        target=targets[row['stimulus_id']];assert row['array_row']==int(target['array_row'])
        joined.append(row|dict(concept_id=target['concept_id'],target_image_sha256=target['image_sha256'],
            target_image_path=gallery[row['stimulus_id']]['image_path'],score_target_path=gallery[row['stimulus_id']]['image_path'],
            target_mapping_attached_after_inference=True))
    path=output/'manifest.jsonl';value=''.join(json.dumps(r)+'\n' for r in joined)
    if path.exists():assert path.read_text()==value
    else:
        with path.open('x') as stream:stream.write(value);stream.flush();os.fsync(stream.fileno())
    return C.ref(path)


def main():
    invoked=time.monotonic();parser=argparse.ArgumentParser()
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--arm',choices=C.ARMS,required=True)
    parser.add_argument('--split',choices=['T32','V'],required=True);parser.add_argument('--segment',choices=[0,1],type=int,default=0)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args();config=C.read(args.registration)
    study_core=C.validate(config,all_inputs=True);C.require_admission(config)
    queries,sources=infer_queries(config,args.split)
    folder=C.DATA/'arms'/args.arm/'train';trained=C.read(folder/'completion.json')
    assert trained['status']=='PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE' and trained['study_core_sha256']==study_core
    assert trained['updates']==24807 and trained['primary_exposures']==trained['paired_exposures']==396900
    checkpoint=Path(trained['checkpoint']['path']);assert C.sha(checkpoint)==trained['checkpoint']['sha256']
    if args.preflight_only:print('PAIRED_GENERATION_ALL_IDENTITIES_CPU_PREFLIGHT_PASSED');return
    C.setup_runtime(config)
    import torch
    import numpy as np
    from PIL import Image
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=C.DATA/'arms'/args.arm/('postfit_T32' if args.split=='T32' else 'images');output.mkdir(parents=True,exist_ok=True)
    assert not (output/'completion.json').exists()
    maximum=1200 if args.split=='T32' else 86400
    job_name=config['generation_job_names'][args.arm][args.split][args.segment]
    wrapper,budget=C.verify_wrapper(config,job_name,maximum,output)
    budget.document['maximum_output_bytes']=(64<<20) if args.split=='T32' else config['arm_disk_budget_bytes']
    rows=verify_prefix(output,queries,args.arm,trained,study_core)
    prior_seconds=0.
    if args.segment:
        old=C.read(C.ROOT/'audit'/(config['generation_job_names'][args.arm][args.split][0]+'_job.json'))
        assert old['status']!='RUNNING';prior_seconds=old['charged_seconds']
    state=torch.load(checkpoint,weights_only=True,map_location='cpu')
    assert state['study_core_sha256']==study_core and state['arm']==args.arm and state['seed']==1729
    assert C.sha(folder/'input_provenance.json')==state['provenance_sha256']
    calibration=C.bound(trained['calibration']);assert calibration==state['calibration']
    assert calibration['strengths']==[0.,.25,.5,.75,1.]
    assert calibration['selected_strength']==calibration['strengths'][min(range(5),key=lambda k:(calibration['Dfit_token_errors'][k],k))]
    model=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True);model.radius=.25*calibration['selected_strength']
    loader_dir=output/('loader_'+str(os.getpid()));loader_dir.mkdir();audit=ConvertedWeightAudit(torch,loader_dir);audit.install()
    try:pipe,adapter,projector,prompts,host=load_native(training=False)
    finally:audit.restore()
    loader_proof=audit.complete();budget.check()
    cache_key=None;data=weight=bias=None;stopped=False
    with (output/'inference_manifest.jsonl').open('a',buffering=1) as log:
        for index,query in enumerate(queries):
            if index<len(rows):continue
            elapsed=time.monotonic()-invoked
            if elapsed>maximum-120 or prior_seconds+elapsed>172800-120:stopped=True;break
            if index%128==0:C.resources(config);C.validate(config);budget.check()
            source=sources[(query['subject'],query['source_role'])];key=(query['subject'],query['source_role'])
            if cache_key!=key:
                data=torch.load(source['condition_cache_path'],map_location='cpu',mmap=True,weights_only=True)
                assert data['model_sha256']==source['checkpoint_sha256']
                brain=torch.load(source['checkpoint_path'],map_location='cpu',weights_only=True)['model']
                weight=brain['block.layers.0.norm.weight'][None].cuda();bias=brain['block.layers.0.norm.bias'][None].cuda()
                lookup={int(value):i for i,value in enumerate(data['array_rows'])};cache_key=key
            position=lookup[query['array_row']];condition=data['condition'][position:position+1]
            assert condition.shape==(1,1024) and torch.isfinite(condition).all()
            with torch.inference_mode():
                changed,info=model(condition.cuda(),weight,bias,gate_mode='fixed')
                tokens=projector(adapter(changed.to(torch.float16)))
                image,observation=sample_native(pipe,prompts,tokens,query['image_seed'],torch)
            path=output/'pixels'/query['subject']/(Path(query['stimulus_id']).stem+'.webp');path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.pending');assert not path.exists() and not temporary.exists()
            image.save(temporary,format='WEBP',lossless=True,quality=90,method=4)
            assert temporary.stat().st_size<=1<<20,'Fixed lossless RGB output byte bound exceeded; preserve artifact'
            with Image.open(temporary) as decoded:assert np.array_equal(np.asarray(image),np.asarray(decoded.convert('RGB')))
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
            temporary.rename(path)
            row=query|dict(arm=args.arm,global_query_index=index,study_core_sha256=study_core,
                method='BrainHIVE_paired_effect_v1_'+args.arm,intervention='correct',repeat_protocol='R4',
                EEG_source_stimulus_id=query['stimulus_id'],generated_path=str(path),generated_sha256=C.sha(path),
                component_checkpoint_sha256=trained['checkpoint']['sha256'],parent_checkpoint_sha256=source['checkpoint_sha256'],
                condition_cache_sha256=source['condition_cache_sha256'],condition_cache_row=position,
                condition_sha256=hashlib.sha256(condition.contiguous().numpy().tobytes()).hexdigest(),
                initial_UNet_input_sha256=observation['initial_UNet_input_sha256'],tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
                UNet_NFE=4,native_IP_K_calls=280,native_IP_V_calls=280,normal_inference_target_access=False,
                teacher_or_negative_pair_at_inference=False,split=args.split)
            log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno());rows.append(row)
            if len(rows)%100==0:print(json.dumps(dict(stage=args.split,arm=args.arm,queries=len(rows),expected=len(queries))),flush=True)
    if stopped:
        C.atomic(output/'progress.json',dict(status='PAIRED_GENERATION_FINITE_STOP_EXACT_PREFIX',queries=len(rows)))
        raise RuntimeError('Finite generation stop; prefix retained')
    assert len(rows)==len(queries)
    manifest=attach_V_targets(config,output,rows) if args.split=='V' else C.ref(output/'inference_manifest.jsonl')
    budget.check()
    C.atomic(output/'completion.json',dict(status='PAIRED_GENERATION_COMPLETE',arm=args.arm,split=args.split,
        study_core_sha256=study_core,queries=len(rows),manifest=manifest,inference_manifest=C.ref(output/'inference_manifest.jsonl'),
        checkpoint=trained['checkpoint'],host=host,loader_proof=loader_proof,normal_inference_target_access=False,
        training_seed=1729,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
        wrapper=wrapper,seconds=prior_seconds+time.monotonic()-invoked))
    print('PAIRED_GENERATION_COMPLETE',flush=True)


if __name__=='__main__':main()
