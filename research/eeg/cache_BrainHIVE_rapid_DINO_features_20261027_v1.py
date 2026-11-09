"""Encode immutable native400 and target400 features once for both screen stages.

This entry point exports a cache, not an effect score.  It never reuses old
eighty-gallery scores, retrains EEG models, downloads weights, or generates an
image.  Future candidate scoring must bind this exact four-hundred gallery.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from PIL import Image
import brainhive_rapid_panel_cache_20261027_v1 as K

ROOT=Path(__file__).resolve().parent
sha,ref,read,rows,atomic=K.sha,K.ref,K.read,K.rows,K.atomic


def validate_images(completion_path):
    complete=read(completion_path)
    assert complete['schema']==K.BASELINE_SCHEMA and complete['status']==K.BASELINE_STATUS
    assert complete['queries']==complete['images']==400
    assert complete['reused_existing_images']==80 and complete['new_query_images']==320 and complete['parity_images']==1
    assert complete['actual_new_images_generated']==321 and complete['native_NFE_per_image']==4
    assert complete['normal_inference_target_access'] is False and complete['target_pixels_loaded_after_generation'] is True
    panel=K.validate_panel(K.verified(complete['panel']))
    source_manifest=K.verified(complete['manifest']);K.verified(complete['input_provenance']);K.verified(complete['actual_native_pixel_parity'])
    assert read(complete['actual_native_pixel_parity']['path'])['status']=='ACTUAL_DIRECT_NATIVE_PIXEL_PARITY_PASSED'
    source=rows(source_manifest);by_id={row['query_id']:row for row in source}
    assert len(source)==len(by_id)==400 and set(by_id)=={row['query_id'] for row in panel['queries']}
    ordered=[];paths=set();inodes=set()
    for i,query in enumerate(panel['queries']):
        row=by_id[query['query_id']]
        assert all(row[key]==query[key] for key in K.IDENTITY_KEYS)
        assert row['UNet_NFE']==4 and row['normal_inference_target_access'] is False
        assert row['target_index']==i and row['target_image_sha256']==panel['target_gallery'][i]['image_sha256']
        path=Path(row['generated_path']);info=path.stat()
        assert str(path.resolve()) not in paths and (info.st_dev,info.st_ino) not in inodes
        paths.add(str(path.resolve()));inodes.add((info.st_dev,info.st_ino))
        assert sha(path)==row['generated_sha256'] and sha(row['target_image_path'])==row['target_image_sha256']
        with Image.open(path) as image:
            assert image.size==(512,512) and hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()==row['generated_pixel_sha256']
        ordered.append(row)
    return complete,panel,ordered


def save_npy(path,value):
    with Path(path).open('xb') as stream:
        np.save(stream,np.asarray(value),allow_pickle=False);stream.flush();os.fsync(stream.fileno())


def compatible_views(output,panel_ref,baseline_path,target_path,ordered,provenance):
    """Zero-copy NPY views for the common CPU rapid scorer's fixed contract."""
    encoder=dict(weight_sha256=K.DINO_SHA,feature_recipe='normalized layer11 CLS+mean-patch concatenation, L2-normalized',
        feature_dimensions=768,real_preprocess='RGB/255; CPU bilinear256 antialias align_cornersFalse; ImageNet normalization',
        generated_preprocess='RGB/255; CUDA bilinear256 antialias align_cornersFalse; ImageNet normalization')
    references={}
    for kind,arm,path,label in [('targets','target_gallery',target_path,'target'),('generated','native_parent',baseline_path,'native')]:
        records=[]
        for row in ordered:
            item={key:row[key] for key in K.IDENTITY_KEYS}
            item.update(image_sha256=row['target_image_sha256'] if kind=='targets' else row['generated_sha256'],arm=arm)
            if kind=='generated':
                item.update(target_image_sha256=row['target_image_sha256'],initial_noise_latents_sha256=row['initial_noise_latents_sha256'],
                    initial_UNet_input_sha256=row['initial_UNet_input_sha256'],condition_sha256=row['condition_sha256'],
                    condition_FP16_sha256=row['condition_FP16_sha256'],tokens_sha256=row['tokens_sha256'],generated_pixel_sha256=row['generated_pixel_sha256'])
            records.append(item)
        row_path=output/f'{label}_rows.json';atomic(row_path,records)
        source_path=output/f'{label}_image_source.jsonl'
        with source_path.open('x',encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(row,allow_nan=False)+'\n' for row in records));stream.flush();os.fsync(stream.fileno())
        document=dict(schema='BrainHIVE_rapid_DINO_feature_cache_20261027_v1',status='COMPLETE_FROZEN_DINO_FEATURE_CACHE',
            kind=kind,arm=arm,panel=panel_ref,features=ref(path),rows=ref(row_path),shape=[len(records),768],dtype='float32',
            encoder=encoder,image_source_manifest=ref(source_path),generation=None if kind=='targets' else dict(
                NFE=4,resolution=[512,512],CFG=0,IP_scale=1,one_image_per_query=True,no_best_of_N=True),
            source=ref(__file__),host=provenance['host'],baseline_completion=provenance['baseline_completion'],
            DINO_weight=provenance['DINO_weight'],image_evaluation_source=provenance['image_evaluation_source'],
            locked_metrics_source=provenance['locked_metrics_source'],transform=K.TRANSFORM)
        manifest=output/f'{label}_cache.json';atomic(manifest,document);references[label]=ref(manifest)
    return references


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-completion','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu-job-name');parser.add_argument('--native-registration',type=Path)
    parser.add_argument('--max-seconds',type=int,default=600);parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    assert 0<args.max_seconds<=600 and 1<=args.batch<=32
    complete,panel,ordered=validate_images(args.baseline_completion)
    if args.preflight_only:
        print(json.dumps(dict(status='RAPID400_DINO_CACHE_INPUTS_CPU_VERIFIED',queries=400,generated_images=400,target_images=400,starts_GPU=False)));return 0
    assert args.gpu_job_name and args.native_registration and not args.output.exists()
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(args.native_registration);C.validate(config);C.setup_runtime(config)
    import torch
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    import image_evaluation as E
    assert E.DINO_SHA256==K.DINO_SHA and E.DINO_WEIGHT.is_file() and sha(E.DINO_WEIGHT)==K.DINO_SHA
    args.output.mkdir(parents=True);invoked=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    D.torch=torch
    document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,maximum_output_bytes=32<<20,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
    wrapper=D.verify_wrapper(document);budget=D.Budget(document)
    torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def check():
        assert not stopped[0] and time.monotonic()-invoked<args.max_seconds,'Finite rapid feature-cache limit'
        budget.check()
    try:
        provenance=dict(schema=K.FEATURE_SCHEMA,source=ref(__file__),kit_source=ref(K.__file__),
            image_evaluation_source=ref(ROOT/'image_evaluation.py'),locked_metrics_source=ref(ROOT/'locked_metrics.py'),
            DINO_weight=ref(E.DINO_WEIGHT),transform=K.TRANSFORM,baseline_completion=ref(args.baseline_completion),
            host=complete['host'],panel=complete['panel'],generation_manifest=complete['manifest'],wrapper=wrapper,
            fixed_target_gallery=400,both_stage_gallery_sizes={'80':400,'400':400},generated_images_encoded_once=400,
            targets_encoded_once=400,old80gallery_scores_read=False,no_images_generated=True,no_EEG_refit=True)
        atomic(args.output/'provenance.json',provenance);check();model=E.dino_model();check()
        def extract(paths,real):
            output=[]
            for start in range(0,len(paths),args.batch):
                check();output.append(E.dino_features(paths[start:start+args.batch],model,batch_size=args.batch,real_images=real))
            result=torch.cat(output).float().numpy()
            assert result.shape==(400,768) and np.isfinite(result).all()
            assert np.allclose(np.linalg.norm(result.astype(np.float64),axis=1),1,atol=2e-5)
            return result
        baseline=extract([Path(row['generated_path']) for row in ordered],False)
        targets=extract([Path(row['target_image_path']) for row in ordered],True)
        del model;torch.cuda.empty_cache();check()
        baseline_path=args.output/'baseline_features.npy';target_path=args.output/'target_features.npy'
        save_npy(baseline_path,baseline);save_npy(target_path,targets)
        feature_index=[]
        for i,row in enumerate(ordered):
            feature_index.append({key:row[key] for key in K.IDENTITY_KEYS}|dict(feature_row=i,target_index=i,
                generated_path=row['generated_path'],generated_sha256=row['generated_sha256'],generated_pixel_sha256=row['generated_pixel_sha256'],
                target_image_path=row['target_image_path'],target_image_sha256=row['target_image_sha256'],
                condition_sha256=row['condition_sha256'],condition_FP16_sha256=row['condition_FP16_sha256'],tokens_sha256=row['tokens_sha256'],
                initial_noise_latents_sha256=row['initial_noise_latents_sha256'],initial_UNet_input_sha256=row['initial_UNet_input_sha256']))
        index_path=args.output/'feature_index.json';atomic(index_path,dict(schema='BrainHIVE_rapid_feature_index_20261027_v1',rows=feature_index))
        for i,row in enumerate(ordered):
            assert sha(row['generated_path'])==row['generated_sha256'] and sha(row['target_image_path'])==row['target_image_sha256']
        K.verified(provenance['baseline_completion']);K.verified(provenance['panel']);K.verified(provenance['generation_manifest'])
        for key in ('source','kit_source','image_evaluation_source','locked_metrics_source','DINO_weight'):K.verified(provenance[key])
        views=compatible_views(args.output,complete['panel'],baseline_path,target_path,ordered,provenance)
        atomic(args.output/'completion.json',dict(status=K.FEATURE_STATUS,schema=K.FEATURE_SCHEMA,queries=400,gallery_queries=400,
            baseline_features=ref(baseline_path),target_features=ref(target_path),features_shape=[400,768],feature_index=ref(index_path),
            panel=complete['panel'],baseline_completion=ref(args.baseline_completion),host=complete['host'],DINO_weight=ref(E.DINO_WEIGHT),
            transform=K.TRANSFORM,image_evaluation_source=ref(ROOT/'image_evaluation.py'),locked_metrics_source=ref(ROOT/'locked_metrics.py'),
            provenance=ref(args.output/'provenance.json'),both_stage_gallery_sizes={'80':400,'400':400},
            native_cache=views['native'],target_cache=views['target'],
            no_images_generated=True,no_EEG_refit=True,actual_new_images_generated=0,seconds=time.monotonic()-invoked,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,development_only=True,independent_confirmation=False))
        return 0
    except BaseException as error:
        atomic(args.output/'failure.json',dict(status='INCOMPLETE_RAPID_DINO_CACHE_NO_REUSE',error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-invoked))
        raise


if __name__=='__main__':raise SystemExit(main())
