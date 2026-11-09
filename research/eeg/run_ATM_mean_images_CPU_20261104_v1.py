"""Native EEG condition replay and all fixed image metrics, no tuning."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,io,json,resource,sys,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import atm_mean_image_contract_20261104_v1 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('prepare','audit','score'),required=True);ap.add_argument('--stage',type=int,default=400);a=ap.parse_args()
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(a.policy);K=c['K'];q=c['q'];out=c['out'];C.resources(c)
def guard():
    assert time.monotonic()-started<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
assert K.ref(__file__)==q['sources']['CPU']
if a.mode=='prepare':
    C.prepare(c,a.policy);C.resources(c);print(json.dumps(dict(status='COMPLETE_INPUT_PREPARATION')));raise SystemExit(0)
prepared=K.read(out/'prepared.json');assert prepared['policy']==K.ref(a.policy)
conditions=np.load(K.checked(prepared['conditions']),allow_pickle=False)
if True:
    assert a.stage==400;queries=c['queries'];records=C.records(c,a.stage)
    for arm in C.ARMS:assert [r['query_index'] for r in records[arm]]==list(range(a.stage))
    ordered=[records[arm][i] for i in range(a.stage) for arm in C.ARMS]
    if a.mode=='audit':
        assert not (out/f'image_audit{a.stage}.json').exists();paths=set();inodes=set();dups={};constant=0
        for i in range(a.stage):
            guard()
            for j,arm in enumerate(C.ARMS):
                r=records[arm][i];assert r['arm']==arm and all(r[k]==queries[i][k] for k in C.IDENTITY)
                path=K.checked(r['image']);s=path.stat();key=(s.st_dev,s.st_ino);assert str(path) not in paths and key not in inodes;paths.add(str(path));inodes.add(key)
                pixels=np.asarray(Image.open(path).convert('RGB')).copy();assert pixels.shape==(512,512,3) and hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixels_sha256']
                value=np.ascontiguousarray(conditions[i,j][None]);assert hashlib.sha256(value.tobytes()).hexdigest()==r['condition_FP32_sha256'] and hashlib.sha256(value.astype(np.float16).tobytes()).hexdigest()==r['condition_FP16_sha256']
                assert r['noise']==records['direct'][i]['noise'] and len(r['noise']['initial'])==1 and len(r['noise']['ancestral'])==4
                if arm in C.MODELS:assert r['normal_inference_only_EEG'] and not r['query_image_or_concept_read_at_inference'] and r['prepared']==K.ref(out/'prepared.json')
                dups.setdefault(r['pixels_sha256'],[]).append([arm,i]);constant+=int(np.max(pixels.astype(float).std(axis=(0,1)))<2.55)
        K.write(out/f'ordered_images{a.stage}.json',ordered);K.write(out/f'image_audit{a.stage}.json',dict(status='PASS_ALL_NATIVE_EEG_IMAGE_PROVENANCE',stage=a.stage,images=a.stage*6,ordered_records=K.ref(out/f'ordered_images{a.stage}.json'),all_conditions_noise_identity_pixels_verified=True,near_constant_count=constant,exact_duplicate_pixel_groups=[x for x in dups.values() if len(x)>1]))
    else:
        assert not (out/f'readout{a.stage}.json').exists();assert K.read(out/f'image_audit{a.stage}.json')['status']=='PASS_ALL_NATIVE_EEG_IMAGE_PROVENANCE'
        import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
        import score_low_level as L
        assert K.ref(S.__file__)==q['source_metrics'] and K.ref(L.__file__)==q['source_low_level']
        features=np.load(K.checked(K.read(out/f'encode_{a.stage}_completion.json')['features']),allow_pickle=False);assert features.shape==(a.stage,6,768)
        target=np.load(K.checked(prepared['target_feature_ref']),allow_pickle=False);selfcheck=S.metric_arrays(target,target,queries,queries)
        assert np.all(selfcheck['DINO2AFC']==1) and np.all(selfcheck['gallery_top1']==1)
        previous=np.load(K.checked(q['baseline_metrics']),allow_pickle=False)
        arrays={arm:{key:previous[arm+'__'+key].copy() for key in C.METRICS} for arm in C.BASES}
        arrays.update({arm:S.metric_arrays(features[:,C.ARMS.index(arm)],target,queries,queries) for arm in C.MODELS})
        metadata=K.read(K.checked(prepared['target_metadata_ref']))['Dcheck'];zips={}
        for arm in C.MODELS:
            for key in C.METRICS[5:]:arrays[arm][key]=np.empty(a.stage)
        try:
            for i in range(a.stage):
                guard()
                m=metadata[i];assert all(m[k]==queries[i][k] for k in C.IDENTITY)
                if m['image_zip'] not in zips:zips[m['image_zip']]=zipfile.ZipFile(m['image_zip'])
                raw=zips[m['image_zip']].read(m['image_member']);assert hashlib.sha256(raw).hexdigest()==m['image_sha256'];trgb,ty=L.pixels(io.BytesIO(raw));tm=L.moments(ty)
                for arm in C.MODELS:
                    rgb,y=L.pixels(K.checked(records[arm][i]['image']));arrays[arm]['PixCorr_RGB'][i]=L.pixcorr(rgb,trgb);arrays[arm]['SSIM_Y'][i]=L.ssim_y(y,ty,tm);arrays[arm]['mean_RGB_absolute_error'][i]=float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())
        finally:
            for z in zips.values():z.close()
        pairs=C.paired(arrays,queries[:a.stage]);path=out/f'metrics{a.stage}.npz';temp=path.with_name(path.name+'.pending')
        with temp.open('xb') as f:np.savez_compressed(f,**{arm+'__'+key:value for arm,row in arrays.items() for key,value in row.items()})
        temp.replace(path)
        K.write(out/f'readout{a.stage}.json',dict(status='COMPLETE_SIX_ARM_T_MEAN_SHIFT_FULL400_PAIRED_READOUT',stage=a.stage,queries=a.stage,gallery=400,arms=list(C.ARMS),means={arm:{key:float(value.mean()) for key,value in row.items()} for arm,row in arrays.items()},
            fixed_comparisons=pairs,interpretation=C.interpret(pairs,a.stage),arrays=K.ref(path),bootstrap=dict(seed=1729,unit='concept',replicates=1000),metrics=list(C.METRICS),source_metrics=q['source_metrics'],source_low_level=q['source_low_level'],
            new_training_updates=0,new_adapter_parameters=0,known_mean_shift_interventions_NOT_original=True,all_results_single_seed_development=True,second_independent_method_or_SOTA=False))
guard();C.resources(c);print(json.dumps(dict(status='COMPLETE_'+a.mode,stage=a.stage,seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)),flush=True)
