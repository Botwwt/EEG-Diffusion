"""Candidate-only DINO cache: encode80, then only320 on stage expansion.

Target400 and native400 caches are never re-encoded. A stage400 export retains
the exact eighty feature rows and their metadata from its stage80 cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from PIL import Image
import brainhive_rapid_candidate_pack_20261027_v1 as P
import brainhive_rapid_panel_cache_20261027_v1 as C
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v1 as S

ROOT=Path(__file__).resolve().parent
sha,ref,read,rows,atomic=C.sha,C.ref,C.read,C.rows,C.atomic


def feature_record(row):
    item={key:row[key] for key in C.IDENTITY_KEYS}
    return item|dict(arm=row['arm'],image_sha256=row['generated_sha256'],target_image_sha256=row['target_image_sha256'],
        generated_pixel_sha256=row['generated_pixel_sha256'],initial_noise_latents_sha256=row['initial_noise_latents_sha256'],
        initial_UNet_input_sha256=row['initial_UNet_input_sha256'],condition_sha256=row['condition_sha256'],
        condition_FP16_sha256=row['condition_FP16_sha256'],tokens_sha256=row['tokens_sha256'],pack_sha256=row['pack_sha256'])


def merge_features(prefix,old_records,new_features,new_records):
    assert prefix.dtype==new_features.dtype==np.float32 and prefix.ndim==new_features.ndim==2
    assert prefix.shape[1]==new_features.shape[1]==768 and len(prefix)==len(old_records) and len(new_features)==len(new_records)
    result=np.concatenate([prefix,new_features],axis=0)
    records=list(old_records)+list(new_records)
    assert result[:len(prefix)].tobytes()==prefix.tobytes() and records[:len(old_records)]==old_records
    return result,records


def export_cache(output,features,records,complete,generation_ref,native_cache_ref,native_document,extractor_sources,previous_ref=None,counts=None):
    output=Path(output);feature_path=output/'candidate_features.npy'
    with feature_path.open('xb') as stream:np.save(stream,features,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    row_path=output/'candidate_rows.json';atomic(row_path,records)
    source_path=output/'candidate_image_source.jsonl'
    with source_path.open('x',encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(row,allow_nan=False)+'\n' for row in records));stream.flush();os.fsync(stream.fileno())
    document=dict(schema=S.CACHE_SCHEMA,status='COMPLETE_FROZEN_DINO_FEATURE_CACHE',kind='generated',arm=complete['arm'],
        panel=complete['panel'],features=ref(feature_path),rows=ref(row_path),shape=list(features.shape),dtype='float32',
        encoder=native_document['encoder'],image_source_manifest=ref(source_path),generation=S.GENERATION,
        generation_completion=generation_ref,pack=complete['pack'],host_family='BrainHIVE',host=complete['host'],host_cache_key=complete['host_cache_key'],
        native_cache=native_cache_ref,extractor_sources=extractor_sources,stage=complete['stage'],stage1_cache=previous_ref,
        counts=counts or {},no_target_or_native_reencoding=True,development_only=True)
    path=output/'candidate_cache.json';atomic(path,document)
    return path


def validate(args):
    complete=read(args.generation_completion)
    assert complete['schema']==P.GEN_SCHEMA and complete['status']==P.GEN_STATUS and complete['stage'] in ('80','400')
    stage=complete['stage'];count=int(stage)
    assert complete['queries']==count and complete['native_NFE_per_image']==4
    assert complete['host_family']=='BrainHIVE' and complete['host_cache_key']==P.host_key(complete['host'])
    assert complete['normal_inference_target_access'] is False and complete['targets_attached_after_generation'] is True
    assert read(C.verified(complete['zero_module_parity']))['status']=='ACTUAL_ZERO_MODULE_NATIVE_PIXEL_PARITY_PASSED'
    for binding in complete['runtime_sources'].values():C.verified(binding)
    assert complete['runtime_sources']['generator']['sha256']==sha(ROOT/'generate_BrainHIVE_rapid_candidate_20261027_v1.py')
    panel=C.validate_panel(C.verified(complete['panel']))
    native=S.load_cache(args.native_cache,complete['panel']['path'],panel,'generated',require_full=True)
    native_document=native['document']
    assert native_document['arm']=='native_parent' and native_document['host']==complete['host']
    assert native_document['DINO_weight']['sha256']==C.DINO_SHA and native_document['transform']==C.TRANSFORM
    assert native_document['image_evaluation_source']['sha256']==sha(ROOT/'image_evaluation.py')
    assert native_document['locked_metrics_source']['sha256']==sha(ROOT/'locked_metrics.py')
    assert native_document['baseline_completion']['sha256']==complete['baseline_completion']['sha256']
    source_rows=rows(C.verified(complete['manifest']));assert len(source_rows)==count
    assert [row['query_id'] for row in source_rows]==panel['stages'][stage]['query_ids']
    for query,row in zip(panel['queries'][:count],source_rows):
        assert all(row[key]==query[key] for key in C.IDENTITY_KEYS) and row['arm']==complete['arm']
        assert row['pack_sha256']==complete['pack']['sha256'] and row['UNet_NFE']==4
        assert sha(row['generated_path'])==row['generated_sha256']
    prefix=np.empty((0,768),np.float32);old_records=[];previous_ref=None
    if stage=='400':
        assert args.stage1_cache and complete['reused_candidate_images']==80 and complete['new_query_images']==320
        old=S.load_cache(args.stage1_cache,complete['panel']['path'],panel,'generated')
        assert old['features'].shape==(80,768) and old['document']['stage']=='80'
        assert old['document']['generation_completion']['sha256']==complete['stage1_completion']['sha256']
        assert old['document']['pack']['sha256']==complete['pack']['sha256'] and old['document']['host_cache_key']==complete['host_cache_key']
        assert old['document']['native_cache']['sha256']==sha(args.native_cache)
        assert old['document']['extractor_sources']==extractor_sources()
        assert old['document']['encoder']==native_document['encoder']
        prefix,old_records=S.ordered(old,panel['queries'][:80])
        assert old_records==[feature_record(row) for row in source_rows[:80]],'Stage2 changed an already encoded prefix image/proof'
        previous_ref=ref(args.stage1_cache)
    else:
        assert args.stage1_cache is None and complete['reused_candidate_images']==0 and complete['new_query_images']==80
    assert complete['actual_new_images_generated']==complete['new_query_images']+1 and complete['parity_images']==1
    C.verified(complete['pack'])
    return complete,panel,native_document,source_rows,prefix,old_records,previous_ref


def extractor_sources():
    return dict(extractor=ref(__file__),pack=ref(P.__file__),panel=ref(C.__file__),
        image_evaluation=ref(ROOT/'image_evaluation.py'),locked_metrics=ref(ROOT/'locked_metrics.py'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('generation-completion','native-cache','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage1-cache',type=Path);parser.add_argument('--native-registration',type=Path)
    parser.add_argument('--gpu-job-name');parser.add_argument('--max-seconds',type=int,default=600);parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args();assert 0<args.max_seconds<=600 and 1<=args.batch<=32
    complete,panel,native,ordered,prefix,old_records,previous_ref=validate(args)
    new_rows=ordered[len(prefix):]
    if args.preflight_only:
        print(json.dumps(dict(status='CPU_CANDIDATE_ONLY_DINO_CACHE_INPUTS_VERIFIED',stage=complete['stage'],
            new_images_to_encode=len(new_rows),reused_feature_rows=len(prefix),native_images_to_encode=0,target_images_to_encode=0,GPU_started=False)));return 0
    assert args.gpu_job_name and args.native_registration and not args.output.exists()
    import brainhive_paired_study_contract_20261025_v1 as F
    config=F.read(args.native_registration);F.validate(config);F.setup_runtime(config)
    import torch
    import image_evaluation as E
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    assert E.DINO_SHA256==C.DINO_SHA and sha(E.DINO_WEIGHT)==C.DINO_SHA
    args.output.mkdir(parents=True);invoked=time.monotonic();D.torch=torch;stopped=[False]
    signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,maximum_output_bytes=32<<20,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
    wrapper=D.verify_wrapper(document);budget=D.Budget(document)
    torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=extractor_sources();controls=dict(generation=ref(args.generation_completion),native_cache=ref(args.native_cache),
        pack=complete['pack'],panel=complete['panel'],generation_manifest=complete['manifest'],zero_module_parity=complete['zero_module_parity'])
    if previous_ref:controls['stage1_cache']=previous_ref
    def check():
        assert not stopped[0] and time.monotonic()-invoked<args.max_seconds,'Finite candidate-only feature budget'
        budget.check()
    try:
        atomic(args.output/'input_provenance.json',dict(generation=controls['generation'],native_cache=controls['native_cache'],
            stage1_cache=previous_ref,extractor_sources=sources,DINO_weight=ref(E.DINO_WEIGHT),transform=C.TRANSFORM,wrapper=wrapper,
            new_images_encoded=len(new_rows),reused_feature_rows=len(prefix),native_images_encoded=0,target_images_encoded=0))
        check();model=E.dino_model();pieces=[]
        for start in range(0,len(new_rows),args.batch):
            check();batch=new_rows[start:start+args.batch]
            pieces.append(E.dino_features([Path(row['generated_path']) for row in batch],model,batch_size=args.batch,real_images=False))
        new_features=torch.cat(pieces).float().numpy();del model;torch.cuda.empty_cache();check()
        assert new_features.shape==(len(new_rows),768) and np.isfinite(new_features).all()
        assert np.allclose(np.linalg.norm(new_features.astype(np.float64),axis=1),1,atol=2e-5)
        features,records=merge_features(prefix,old_records,new_features,[feature_record(row) for row in new_rows])
        for source in list(sources.values())+list(controls.values()):C.verified(source)
        for row in ordered:assert sha(row['generated_path'])==row['generated_sha256']
        counts=dict(new_candidate_images_encoded=len(new_rows),reused_candidate_feature_rows=len(prefix),native_images_encoded=0,target_images_encoded=0,
            actual_new_images_generated=0)
        cache_path=export_cache(args.output,features,records,complete,controls['generation'],controls['native_cache'],native,sources,previous_ref,counts)
        S.load_cache(cache_path,complete['panel']['path'],panel,'generated',require_full=complete['stage']=='400')
        atomic(args.output/'completion.json',dict(status='COMPLETE_BRAINHIVE_RAPID_CANDIDATE_DINO_CACHE',stage=complete['stage'],queries=len(features),
            candidate_cache=ref(cache_path),generation_completion=controls['generation'],native_cache=controls['native_cache'],
            stage1_cache=previous_ref,pack=complete['pack'],host_family='BrainHIVE',host_cache_key=complete['host_cache_key'],
            counts=counts,seconds=time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes))
        return 0
    except BaseException as error:
        atomic(args.output/'failure.json',dict(status='INCOMPLETE_RAPID_CANDIDATE_FEATURE_CACHE_NO_REUSE',error=repr(error),seconds=time.monotonic()-invoked))
        raise


if __name__=='__main__':raise SystemExit(main())
