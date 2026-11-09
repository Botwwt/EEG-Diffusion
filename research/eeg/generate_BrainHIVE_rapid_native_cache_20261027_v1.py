"""Fill the shared native T400 cache: reuse80, render320 plus one parity image.

The existing full OOF native condition bank is used directly. Targets are
attached only after every new image is generated. No EEG/F fitting occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import zipfile
import numpy as np
from PIL import Image
import brainhive_rapid_panel_cache_20261027_v1 as K
import generate_BrainHIVE_trialset_fastscreen_20261027_v1 as G
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N
import generate_BrainHIVE_prediction_fastscreen_20261027_v1 as P

ROOT=Path(__file__).resolve().parent
sha,ref,read,rows,atomic=K.sha,K.ref,K.read,K.rows,K.atomic
OUTPUT_CAP=1024<<20


def validate(args):
    panel=K.validate_panel(args.panel)
    native=read(args.native_parent_binding)
    assert native['status']=='COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS' and native['prediction_coordinates']==P.NATIVE_COORDINATES
    assert native['predictions_shape']==[13230,1024] and native['oof_coverage_min']==native['oof_coverage_max']==1
    assert native['prediction_index_sha256']==panel['index']['sha256']
    assert sha(native['predictions_path'])==native['predictions_sha256']
    predictions=np.load(native['predictions_path'],mmap_mode='r',allow_pickle=False)
    assert predictions.shape==(13230,1024) and predictions.dtype==np.float32 and np.isfinite(predictions).all()
    index=sorted(rows(panel['index']['path']),key=lambda row:row['row']);positions={row['row']:i for i,row in enumerate(index)}
    reference=read(args.legacy_completion)
    assert reference['status']=='COMPLETED_GENERIC_NATIVE_PREDICTION_SCREEN' and 'native_parent' in reference['arms']
    assert reference['panel']['sha256']==panel['old_panel']['sha256']
    K.verified(reference['manifest']);prior=read(K.verified(reference['input_provenance']))
    assert prior['predictions']['native_parent']['sha256']==native['predictions_sha256']
    assert prior['native_registration']['sha256']==sha(args.native_registration)
    cached={row['query_id']:row for row in rows(reference['manifest']['path']) if row['arm']=='native_parent'}
    prefix=panel['queries'][:panel['legacy_prefix_queries']]
    assert set(cached)=={query['query_id'] for query in prefix}
    for query in prefix:
        row=cached[query['query_id']]
        assert all(row[key]==query[key] for key in K.IDENTITY_KEYS)
        raw=np.array(predictions[positions[query['row']]:positions[query['row']]+1],copy=True)
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['condition_sha256']
        assert sha(row['generated_path'])==row['generated_sha256'] and row['UNet_NFE']==4
        with Image.open(row['generated_path']) as image:
            assert image.size==(512,512) and hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==row['generated_pixel_sha256']
    containers={}
    for item in panel['target_gallery']:containers.setdefault(item['image_zip'],[]).append(item)
    for path,members in containers.items():
        with zipfile.ZipFile(path) as archive:
            for item in members:assert archive.getinfo(item['image_member']).file_size==item['image_bytes']
    return panel,native,predictions,positions,reference,cached


def attach_targets(panel,generated,output):
    by_id={row['query_id']:row for row in panel['target_gallery']};targets={};joined=[]
    for row in generated:
        item=by_id[row['query_id']]
        if item['query_id'] not in targets:
            if row.get('cache_reused'):
                path=Path(row['target_image_path']);assert sha(path)==item['image_sha256']
            else:
                with zipfile.ZipFile(item['image_zip']) as archive:payload=archive.read(item['image_member'])
                assert hashlib.sha256(payload).hexdigest()==item['image_sha256']
                path=output/'targets'/f'{item["target_index"]:03d}_{item["array_row"]:05d}_{Path(item["stimulus_id"]).name}'
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
            with Image.open(path) as image:image.convert('RGB').load()
            targets[item['query_id']]=dict(target_image_path=str(path),score_target_path=str(path),target_image_sha256=item['image_sha256'],target_index=item['target_index'])
        joined.append(row|targets[item['query_id']]|dict(target_mapping_attached_after_inference=True))
    assert len(targets)==len(panel['queries'])
    path=output/'manifest.jsonl'
    with path.open('x',encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(row,allow_nan=False)+'\n' for row in joined));stream.flush();os.fsync(stream.fileno())
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','native-parent-binding','legacy-completion','native-registration','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu-job-name');parser.add_argument('--max-seconds',type=int,default=1800)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    invoked=time.monotonic();assert 0<args.max_seconds<=1800
    panel,native,predictions,positions,reference,cached=validate(args)
    total=len(panel['queries']);prefix=panel['legacy_prefix_queries'];new_count=total-prefix
    if args.preflight_only:
        print(json.dumps(dict(status='RAPID_NATIVE_CACHE_CPU_PREFLIGHT_PASSED',queries=total,reuse_images=prefix,
            planned_new_query_images=new_count,planned_parity_images=1,actual_new_images_generated=0,starts_GPU=False)));return 0
    assert args.gpu_job_name and not args.output_dir.exists()
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(args.native_registration);C.validate(config);C.setup_runtime(config)
    import torch
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from brainhive_paired_supervision_20261026_v3 import safe_bytes_used
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output_dir.mkdir(parents=True);D.torch=torch
    document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,maximum_output_bytes=OUTPUT_CAP,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
    wrapper=D.verify_wrapper(document);budget=D.Budget(document);stopped=[False]
    signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    generated=[];actual_new=0;parity_passed=False
    try:
        C.resources(config,training_start=True)
        provenance=dict(schema=K.BASELINE_SCHEMA,source=ref(__file__),kit_source=ref(K.__file__),sampler_source=ref(N.__file__),
            legacy_helper_source=ref(G.__file__),prediction_helper_source=ref(P.__file__),
            panel=ref(args.panel),native_parent_binding=ref(args.native_parent_binding),legacy_completion=ref(args.legacy_completion),
            native_registration=ref(args.native_registration),normal_inference_target_access=False,no_EEG_refit=True,no_visual_F_access=True,
            expected_queries=total,reuse_images=prefix,planned_new_query_images=new_count,planned_parity_images=1)
        atomic(args.output_dir/'input_provenance.json',provenance)
        loader=args.output_dir/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader_proof=audit.complete();assert host==reference['host']
        first=panel['queries'][0];raw=np.array(predictions[positions[first['row']]:positions[first['row']]+1],copy=True)
        with torch.inference_mode():
            tokens=projector(adapter(torch.from_numpy(raw).cuda().to(torch.float16)))
            image,observation=N.sample_observed(pipe,prompts,tokens,first['image_seed'],torch)
        old=cached[first['query_id']]
        assert observation['initial_noise_latents_sha256']==old['initial_noise_latents_sha256']
        assert observation['initial_UNet_input_sha256']==old['initial_UNet_input_sha256']
        assert hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest()==old['tokens_sha256']
        parity=N.save_image(args.output_dir/'parity'/'actual_direct_native.webp',image);actual_new+=1
        with Image.open(old['generated_path']) as prior:parity_passed=np.array_equal(np.asarray(image.convert('RGB')),np.asarray(prior.convert('RGB')))
        atomic(args.output_dir/'native_pixel_parity.json',dict(status='ACTUAL_DIRECT_NATIVE_PIXEL_PARITY_PASSED' if parity_passed else 'ACTUAL_DIRECT_NATIVE_PIXEL_PARITY_FAILED',
            query=first,parity=parity,observation=observation,legacy_generated_sha256=old['generated_sha256'],new_query_images_started=False))
        assert parity_passed,'STOP before new320: native cached pixel parity failed'
        inference=args.output_dir/'inference_manifest.jsonl'
        with inference.open('x',encoding='utf-8') as stream:
            for i,query in enumerate(panel['queries']):
                assert not stopped[0] and time.monotonic()-invoked<args.max_seconds-120
                budget.check();assert safe_bytes_used(args.output_dir)<OUTPUT_CAP-(16<<20)
                if i<prefix:
                    row=cached[query['query_id']]|dict(cache_reused=True,cache_source_completion=ref(args.legacy_completion),global_query_index=i)
                else:
                    position=positions[query['row']];raw=np.array(predictions[position:position+1],copy=True)
                    with torch.inference_mode():
                        condition=torch.from_numpy(raw).cuda();tokens=projector(adapter(condition.to(torch.float16)))
                        image,observation=N.sample_observed(pipe,prompts,tokens,query['image_seed'],torch)
                    saved=N.save_image(args.output_dir/'native_parent'/'pixels'/f'{i:03d}_{query["array_row"]:05d}.webp',image);actual_new+=1
                    row=query|observation|saved|dict(arm='native_parent',global_query_index=i,prediction_row=position,
                        condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),condition_coordinates=P.NATIVE_COORDINATES,
                        prediction_file_sha256=native['predictions_sha256'],condition_FP16_sha256=hashlib.sha256(condition.to(torch.float16).cpu().contiguous().numpy().tobytes()).hexdigest(),
                        tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),cache_reused=False,
                        method='BrainHIVE_rapid_cached_native',intervention='correct',repeat_protocol='R4',split='T_HELD_CONCEPT_EXPOSED_DEVELOPMENT_SCREEN',
                        normal_inference_target_access=False,oracle_normalization=False,one_image_per_query=True)
                generated.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                if (i+1)%20==0:print(json.dumps(dict(stage='RAPID_NATIVE_CACHE_FILL',queries=i+1,total=total,new_images_generated=actual_new,seconds=time.monotonic()-invoked)),flush=True)
        assert len(generated)==total and actual_new==new_count+1 and parity_passed
        manifest=attach_targets(panel,generated,args.output_dir)
        for key in ('panel','native_parent_binding','legacy_completion','native_registration','source','kit_source','sampler_source','legacy_helper_source','prediction_helper_source'):
            K.verified(provenance[key])
        assert sha(native['predictions_path'])==native['predictions_sha256'];budget.check()
        atomic(args.output_dir/'completion.json',dict(schema=K.BASELINE_SCHEMA,status=K.BASELINE_STATUS,queries=total,images=total,
            panel=ref(args.panel),native_parent_binding=ref(args.native_parent_binding),legacy_completion=ref(args.legacy_completion),
            manifest=ref(manifest),inference_manifest=ref(inference),input_provenance=ref(args.output_dir/'input_provenance.json'),
            host=host,loader_proof=loader_proof,actual_native_pixel_parity=ref(args.output_dir/'native_pixel_parity.json'),
            reused_existing_images=prefix,new_query_images=new_count,parity_images=1,actual_new_images_generated=actual_new,
            new_UNet_evaluations=4*actual_new,native_NFE_per_image=4,resolution=[512,512],CFG=0,IP_scale=1,
            no_best_of_N=True,normal_inference_target_access=False,target_pixels_loaded_after_generation=True,
            no_EEG_refit=True,no_visual_F_access=True,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
            seconds=time.monotonic()-invoked,development_only=True,independent_confirmation=False))
        return 0
    except BaseException as error:
        atomic(args.output_dir/'failure.json',dict(status='INCOMPLETE_RAPID_NATIVE_CACHE_NO_ADAPTIVE_RETRY',error=repr(error),
            retained_rows=len(generated),actual_new_images_generated=actual_new,parity_passed=parity_passed,seconds=time.monotonic()-invoked))
        raise


if __name__=='__main__':raise SystemExit(main())
