"""All-image provenance and fixed paired semantic/detail/low-level readout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,csv,gzip,hashlib,io,json,sys,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--stage',type=int,required=True);ap.add_argument('--mode',choices=['audit','score'],required=True);args=ap.parse_args()
q=json.loads(Path(args.policy).read_bytes());out=Path(q['output_root']);sys.path.insert(0,str(out))
import atm_error_geometry_contract_20261104_v1 as K
assert K.sha(__file__)==q['sources']['audit_score']['sha256'] and args.stage in (80,400)
parent,train,e,C=K.setup(q);K.resources(q,parent,train,e,C)
prepared=K.read(out/'prepared.json');queries=K.read(K.checked(prepared['queries']));conditions=np.load(K.checked(prepared['conditions']),mmap_mode='r')
base=Path(q['baseline80_root']);baseline=[json.loads(line) for line in (base/'images.jsonl').read_text().splitlines()]
metadata=K.read(K.checked(prepared['target_metadata_ref']))['Dcheck'];assert len(metadata)==400
identity=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')
for i,row in enumerate(metadata):assert all(row[k]==queries[i][k] for k in identity)
records_by_arm={}
for arm in K.ARMS:
    p=out/'images'/arm/'images.jsonl'
    new=[json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    rows=K.baseline_records(q)[arm][:args.stage] if arm in K.BASELINES else new
    assert len(rows)==args.stage and [r['query_index'] for r in rows]==list(range(args.stage))
    records_by_arm[arm]=rows
ordered=[records_by_arm[arm][i] for i in range(args.stage) for arm in K.ARMS]
if args.mode=='audit':
    assert not (out/f'images{args.stage}_audit.json').exists();inodes=set();pixel_groups={};constant=0
    for i in range(args.stage):
        reference_noise=records_by_arm['parent'][i]['noise']
        for j,arm in enumerate(K.ARMS):
            row=records_by_arm[arm][i];assert row['arm']==arm and all(row[k]==queries[i][k] for k in identity)
            path=K.checked(row['image']);st=path.stat();key=(st.st_dev,st.st_ino);assert key not in inodes;inodes.add(key)
            pixels=np.asarray(Image.open(path).convert('RGB')).copy();assert pixels.shape==(512,512,3)
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['pixels_sha256']
            value=np.ascontiguousarray(conditions[i,j][None]);assert hashlib.sha256(value.tobytes()).hexdigest()==row['condition_FP32_sha256']
            assert hashlib.sha256(value.astype(np.float16).tobytes()).hexdigest()==row['condition_FP16_sha256']
            assert row['noise']==reference_noise and len(reference_noise['initial'])==1 and len(reference_noise['ancestral'])==4
            if arm not in K.BASELINES:
                assert row['prepared']==K.ref(out/'prepared.json') and row['target_image_sha256']==metadata[i]['image_sha256']
            pixel_groups.setdefault(row['pixels_sha256'],[]).append([arm,i])
            constant+=int(np.max(pixels.astype(np.float64).std(axis=(0,1)))<2.55)
    if args.stage==400:
        old=K.read(K.checked(K.read(out/'images80_audit.json')['ordered_records']))
        assert ordered[:80*len(K.ARMS)]==old
    K.write(out/f'ordered_images{args.stage}.json',ordered)
    K.write(out/f'images{args.stage}_audit.json',{'status':'PASS_ALL_STAGE_IMAGES_IDENTITY_CONDITIONS_NOISE_PIXELS',
        'stage':args.stage,'images':len(K.ARMS)*args.stage,'ordered_records':K.ref(out/f'ordered_images{args.stage}.json'),
        'all_actual_conditions_FP32_FP16_verified':True,'all_same_query_initial_and4_ancestral_noises_equal':True,
        'no_cross_arm_hardlink_or_path_reuse':True,'exact_pixel_duplicate_groups':[v for v in pixel_groups.values() if len(v)>1],
        'near_spatial_constant_images_descriptive':constant,'full_denominator_retained':True})
else:
    assert not (out/f'readout{args.stage}.json').exists()
    audit=K.read(out/f'images{args.stage}_audit.json');assert audit['status']=='PASS_ALL_STAGE_IMAGES_IDENTITY_CONDITIONS_NOISE_PIXELS'
    cache=K.read(out/f'features{args.stage}_completion.json');features=np.load(K.checked(cache['features']),allow_pickle=False)
    assert features.shape==(args.stage,len(K.ARMS),768)
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    import score_low_level as L
    target=np.load(K.checked(prepared['target_feature_ref']),allow_pickle=False)
    assert target.shape==(400,768)
    selfcheck=S.metric_arrays(target,target,queries,queries);assert np.all(selfcheck['DINO2AFC']==1) and np.all(selfcheck['gallery_top1']==1)
    arrays={arm:S.metric_arrays(features[:,j],target,queries[:args.stage],queries) for j,arm in enumerate(K.ARMS)}
    for arm in K.ARMS:
        for key in ('PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error'):arrays[arm][key]=np.empty(args.stage)
    zips={}
    try:
        for i in range(args.stage):
            meta=metadata[i]
            if meta['image_zip'] not in zips:zips[meta['image_zip']]=zipfile.ZipFile(meta['image_zip'])
            raw=zips[meta['image_zip']].read(meta['image_member']);assert hashlib.sha256(raw).hexdigest()==meta['image_sha256']
            trgb,ty=L.pixels(io.BytesIO(raw));tm=L.moments(ty)
            for arm in K.ARMS:
                row=records_by_arm[arm][i];rgb,y=L.pixels(K.checked(row['image']))
                arrays[arm]['PixCorr_RGB'][i]=L.pixcorr(rgb,trgb)
                arrays[arm]['SSIM_Y'][i]=L.ssim_y(y,ty,tm)
                arrays[arm]['mean_RGB_absolute_error'][i]=float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())
    finally:
        for archive in zips.values():archive.close()
    comparisons=K.paired(arrays,queries[:args.stage]);interpretation=K.interpret(comparisons,args.stage)
    metrics=tuple(arrays['parent'])
    path=out/f'all{args.stage}_paired_metrics.npz';temp=path.with_name(path.name+'.pending')
    with temp.open('xb') as stream:np.savez_compressed(stream,**{arm+'__'+key:values for arm,row in arrays.items() for key,values in row.items()})
    temp.replace(path)
    csvpath=out/f'all{args.stage}_per_query.csv.gz'
    with gzip.open(csvpath,'wt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['query_id','arm','query_index',*metrics]);writer.writeheader()
        for i,query in enumerate(queries[:args.stage]):
            for arm in K.ARMS:writer.writerow({'query_id':query['query_id'],'arm':arm,'query_index':i,**{k:float(arrays[arm][k][i]) for k in metrics}})
    K.write(out/f'readout{args.stage}.json',{'status':'COMPLETE_FIXED_NINE_ARM_NATIVE_ATM_ERROR_GEOMETRY_PAIRED_READOUT','stage':args.stage,
        'queries':args.stage,'gallery':400,'arms':list(K.ARMS),'metrics':list(metrics),'arrays':K.ref(path),'per_query':K.ref(csvpath),
        'means':{arm:{key:float(v.mean()) for key,v in row.items()} for arm,row in arrays.items()},
        'fixed_comparisons':comparisons,'interpretation':interpretation,
        'metric_higher_is_better':{key:key!='mean_RGB_absolute_error' for key in metrics},
        'source_metrics':K.ref(S.__file__),'source_low_level':K.ref(L.__file__),
        'target_self_pair_DINO2AFC':1.,'bootstrap_seed':1729,'bootstrap_draws':1000,
        'new_fits':len(K.MODELS),'training_epochs_per_model':150,'new_images_created_by_this_CPU_scorer':0,'all_results_development_single_training_seed':True})
    K.resources(q,parent,train,e,C)
print(json.dumps({'status':'COMPLETE_'+args.mode.upper(),'stage':args.stage}),flush=True)
