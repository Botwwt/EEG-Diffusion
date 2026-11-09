"""D37 actual native images, or candidate-only DINO encoding, on the second GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import brainhive_temporal_image_screen_20261028_v1 as U


def bootstrap(args):
    start=time.monotonic();spec,prepared,queries,groups=U.prepared(args.spec)
    execution=U.read(args.execution)
    assert execution['status']=='D37_IMAGE_EXECUTORS_LOCKED_BEFORE_RESEARCH_IMAGES' and execution['spec']['sha256']==U.sha(args.spec)
    for item in execution['source_pins']:U.check(item)
    U.stage_gate(args.spec,args.stage)
    import brainhive_paired_study_contract_20261025_v1 as C
    manifest=U.ROOT/'audit/diagnostic_memory_recovery_20261025_v3/isolated_runtime_manifest.json'
    C.setup_runtime(dict(runtime_manifest=dict(path=str(manifest),sha256=U.sha(manifest))))
    import torch
    from verify_HIVE_second_host_native_GPU_20261028_v1 import wrapper,memory
    job=wrapper(args.job_name);torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not torch.cuda.is_initialized();base=memory()['VmRSS']
    def budget():
        assert time.monotonic()-start<850
        assert memory()['VmHWM']-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        assert sum(p.stat().st_size for p in (U.B/'screen_images_v1').rglob('*.json'))<96<<20
    def stats():return dict(seconds=time.monotonic()-start,peak_extra_CPU_bytes=memory()['VmHWM']-base,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
    return spec,prepared,queries,execution,torch,job,budget,stats


def saved_rows(count,queries):
    rows=[];folder=U.B/'screen_images_v1/records'
    for i,query in enumerate(queries[:count]):
        for arm in U.ARMS:
            path=folder/(f'{i:03d}_{arm}.json');row=U.read(path)
            assert all(row[k]==query[k] for k in U.IDENTITY) and row['query_index']==i and row['arm']==arm
            assert row['spec']['sha256']==U.sha(U.ROOT/'configs/temporal_conditional_attention_20261028_v1/image_spec_v1.json')
            assert U.sha(row['generated_path'])==row['generated_sha256']
            assert row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280
            assert row['additional_temporal_K_calls']==row['additional_temporal_V_calls']==280
            rows.append(row)
    return rows


def generate(args):
    spec,prepared,queries,execution,torch,job,budget,stats=bootstrap(args)
    assert 0<=args.query_start<args.query_end<=args.stage and 0<args.query_end-args.query_start<=80
    assert (args.stage==80 and args.query_start==0 and args.query_end==80) or args.stage==400 and args.query_start>=80
    if args.query_start:saved_rows(args.query_start,queries)
    from PIL import Image
    from brainhive_component_generator_v1 import load_native
    from brainhive_temporal_IP_binding_20261028_v2 import TemporalContext,bind_temporal_IP,sample_temporal_observed
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import save_image
    root=U.B/'screen_images_v1';(root/'records').mkdir(parents=True,exist_ok=True);(root/'pixels').mkdir(exist_ok=True)
    attempt=root/'attempts'/args.job_name;assert not attempt.exists();attempt.mkdir(parents=True)
    U.write(attempt/'input_provenance.json',dict(spec=U.ref(args.spec),execution=U.ref(args.execution),prepared=U.ref(U.B/'prepared_image_v1/completion.json'),job=job,range=[args.query_start,args.query_end]))
    condition=np.load(U.check(prepared['files']['ordinary_conditions400']),mmap_mode='r',allow_pickle=False)
    details=np.load(U.check(prepared['files']['temporal_tokens400']),mmap_mode='r',allow_pickle=False)
    donor=np.load(U.check(prepared['files']['same_concept_donor400']),allow_pickle=False)
    scaling=np.load(U.check(prepared['files']['wrong_RMS_scale400']),allow_pickle=False)
    original=[json.loads(x) for x in U.check(spec['baseline_manifest']).read_text().splitlines()]
    old={(row['query_id'],row['arm']):row for row in original}
    context=TemporalContext();observer=ConvertedWeightAudit(torch,attempt/'converted_weights');(attempt/'converted_weights').mkdir();observer.install()
    try:pipe,adapter,projector,prompts,host=load_native(training=False)
    finally:observer.restore()
    loading=observer.complete();assert host==spec['expected_host']
    # Guard the actual host state as well as the loader's 2641 conversion checks.
    frozen={name:U.byte_sha(t.detach().cpu().numpy()) for name,t in pipe.unet.state_dict().items()}
    processors=bind_temporal_IP(pipe.unet,context);generated=0;reused=0
    try:
        with torch.inference_mode():
            c=torch.tensor(condition[:1],device='cuda');tokens=projector(adapter(c.half()))
            reference=spec['baseline_first_ordinary'];assert U.byte_sha(tokens.cpu().numpy())==reference['tokens_sha256']
            context.configure(None,enabled=False);budget()
            image,proof=sample_temporal_observed(pipe,prompts,tokens,queries[0]['image_seed'],torch)
            with Image.open(reference['generated_path']) as im:pixels=np.asarray(im.convert('RGB')).copy()
            assert U.sha(reference['generated_path'])==reference['generated_sha256']
            assert np.array_equal(np.asarray(image),pixels)
            assert proof['initial_UNet_input_sha256']==reference['initial_UNet_input_sha256'] and proof['ancestral_noise_observations']==reference['ancestral_noise_observations']
            parity=save_image(attempt/'ordinary_parity.webp',image)
            U.write(attempt/'ordinary_parity.json',dict(status='D37_DCHECK_ORDINARY_DISABLED_PIXEL_EXACT',**proof,**parity,baseline=reference))
            for i in range(args.query_start,args.query_end):
                query=queries[i];base=old[(query['query_id'],'ordinary')]
                c=torch.tensor(condition[i:i+1],device='cuda');tokens=projector(adapter(c.half()))
                assert U.byte_sha(c.cpu().numpy())==base['condition_sha256'] and U.byte_sha(tokens.cpu().numpy())==base['tokens_sha256']
                for arm in U.ARMS:
                    budget();record=root/'records'/(f'{i:03d}_{arm}.json')
                    if record.exists():
                        row=U.read(record);assert row['query_id']==query['query_id'] and row['arm']==arm and row['spec']['sha256']==U.sha(args.spec)
                        assert U.sha(row['generated_path'])==row['generated_sha256'];reused+=1;continue
                    d=np.asarray(details[i:i+1]).copy();source_index=i
                    if arm=='same_concept_wrong':
                        source_index=int(donor[i]);d=np.asarray(details[source_index:source_index+1]).copy()*scaling[i:i+1,:,None,None]
                    mode='flat_residual' if arm=='flat_residual' else 'conditional'
                    context.configure(torch.tensor(d,device='cuda'),mode=mode)
                    image,proof=sample_temporal_observed(pipe,prompts,tokens,query['image_seed'],torch)
                    assert proof['initial_UNet_input_sha256']==base['initial_UNet_input_sha256'] and proof['initial_noise_latents_sha256']==base['initial_noise_latents_sha256']
                    assert proof['ancestral_noise_observations']==base['ancestral_noise_observations']
                    assert len(context.calls)==70 and set(context.calls.values())=={4}
                    marginal=max(v['native_marginal_max_error'] for v in context.records.values())
                    if mode=='conditional':assert marginal<2e-6
                    pending=attempt/(f'{i:03d}_{arm}.webp');saved=save_image(pending,image);target=root/'pixels'/(f'{i:03d}_{arm}.webp')
                    if target.exists():assert U.sha(target)==saved['generated_sha256'],'Orphan mismatch preserved; do not overwrite'
                    else:os.link(pending,target)
                    saved['generated_path']=str(target)
                    row=dict(**query,arm=arm,query_index=i,spec=U.ref(args.spec),execution=U.ref(args.execution),**proof,**saved,
                        condition_sha256=base['condition_sha256'],condition_FP16_sha256=base['condition_FP16_sha256'],tokens_sha256=base['tokens_sha256'],
                        temporal_direction_sha256=U.byte_sha(d),temporal_EEG_donor_index=source_index,temporal_EEG_donor_query_id=queries[source_index]['query_id'],
                        matched_RMS_control=arm=='same_concept_wrong',active_IP_layers=70,total_active_IP_calls=280,
                        native_marginal_max_error=marginal,maximum_layer_delta_RMS=max(v['delta_RMS'] for v in context.records.values()),
                        normal_inference_target_access=False,original_ordinary_unchanged=True)
                    U.write(record,row);generated+=1
                U.write(attempt/(f'query_{i:03d}_complete.json'),dict(query_index=i,rows_committed=3,new_images_this_job=generated,reused_images_this_job=reused,**stats()))
    finally:pipe.unet.set_attn_processor(dict(processors))
    for name,t in pipe.unet.state_dict().items():assert U.byte_sha(t.detach().cpu().numpy())==frozen[name]
    for item in execution['source_pins']:U.check(item)
    final=saved_rows(args.query_end,queries);budget()
    U.write(attempt/'completion.json',dict(status='D37_FIXED_TEMPORAL_IMAGE_SEGMENT_COMPLETE',spec=U.ref(args.spec),execution=U.ref(args.execution),
        stage=args.stage,query_start=args.query_start,query_end=args.query_end,new_images=generated,reused_images=reused,parity_images=1,
        full_prefix_rows=len(final),all_UNet_weights_unchanged=True,host=host,loader_proof=loading,job=job,**stats()))
    print(json.dumps(dict(status='D37_FIXED_TEMPORAL_IMAGE_SEGMENT_COMPLETE',query_end=args.query_end,new_images=generated,**stats())),flush=True)


def encode(args):
    spec,prepared,queries,execution,torch,job,budget,stats=bootstrap(args)
    rows=saved_rows(args.stage,queries);audit=U.read(U.B/f'screen{args.stage}/image_audit.json')
    assert audit['status']=='D37_ALL_STAGE_IMAGE_PIXELS_AND_INPUTS_CPU_AUDITED' and audit['images']==args.stage*3
    folder=U.B/f'screen{args.stage}/features';assert not folder.exists();folder.mkdir(parents=True)
    import image_evaluation as E
    assert E.DINO_SHA256==spec['DINO_weight']['sha256'] and U.sha(E.DINO_WEIGHT)==E.DINO_SHA256
    model=E.dino_model();budget()
    original=np.load(U.check(spec['baseline_features']),allow_pickle=False);assert original.shape==(400,3,768)
    anchor=E.dino_features([Path(spec['baseline_first_ordinary']['generated_path'])],model,real_images=False).float().numpy()[0]
    parity_error=float(np.abs(anchor-original[0,1]).max());assert parity_error<2e-5
    prefix=np.empty((0,3,768),np.float32);previous=None
    if args.stage==400:
        previous=U.B/'screen80/features/completion.json';prev=U.read(previous)
        assert prev['status']=='D37_ALL_THREE_IMAGE_ARMS_DINO_CACHED' and prev['spec']['sha256']==U.sha(args.spec)
        prefix=np.load(U.check(prev['features']),allow_pickle=False);assert prefix.shape==(80,3,768)
    fresh=rows[len(prefix)*3:];pieces=[]
    for lo in range(0,len(fresh),32):
        budget();pieces.append(E.dino_features([Path(row['generated_path']) for row in fresh[lo:lo+32]],model,real_images=False).float().numpy())
    result=np.concatenate([prefix,np.concatenate(pieces).reshape(-1,3,768)]);assert result.shape==(args.stage,3,768)
    assert result[:len(prefix)].tobytes()==prefix.tobytes() and np.isfinite(result).all()
    assert np.allclose(np.linalg.norm(result.astype(np.float64),axis=-1),1,atol=2e-5)
    U.array(folder/'features.npy',result);U.write(folder/'image_rows.json',rows)
    budget();U.write(folder/'completion.json',dict(status='D37_ALL_THREE_IMAGE_ARMS_DINO_CACHED',stage=args.stage,arms=list(U.ARMS),spec=U.ref(args.spec),
        execution=U.ref(args.execution),features=U.ref(folder/'features.npy'),rows=U.ref(folder/'image_rows.json'),image_audit=U.ref(U.B/f'screen{args.stage}/image_audit.json'),
        previous80=U.ref(previous) if previous else None,reused_rows=len(prefix)*3,new_generated_images_encoded=len(fresh),
        baseline_images_encoded=1,baseline_ordinary_feature_anchor_max_error=parity_error,target_images_encoded=0,
        DINO_weight=U.ref(E.DINO_WEIGHT),image_evaluation=U.ref(E.__file__),job=job,**stats()))
    print(json.dumps(dict(status='D37_ALL_THREE_IMAGE_ARMS_DINO_CACHED',stage=args.stage,anchor_error=parity_error,**stats())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--execution',type=Path,required=True)
    p.add_argument('--mode',choices=['generate','encode'],required=True);p.add_argument('--stage',type=int,choices=[80,400],required=True)
    p.add_argument('--query-start',type=int,default=0);p.add_argument('--query-end',type=int,default=80);p.add_argument('--job-name',required=True);a=p.parse_args()
    try:generate(a) if a.mode=='generate' else encode(a)
    except BaseException as error:
        folder=U.B/'failures';folder.mkdir(exist_ok=True)
        U.write(folder/(a.job_name+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise
