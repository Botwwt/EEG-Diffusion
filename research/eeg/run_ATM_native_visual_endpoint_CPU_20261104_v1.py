"""Lock all400 conditions, audit all320 images, and report fixed80 diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import atm_native_visual_endpoint_contract_20261104_v1 as V

ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('prepare','audit','score'),required=True);args=ap.parse_args()
c=V.setup(args.policy);K=c['K'];q=c['q'];out=c['out'];V.resources(c)
assert K.ref(__file__)==q['sources']['CPU']
oldprep=K.read(K.checked(q['base_prepared']));queries=K.read(K.checked(oldprep['queries']))
metadata=K.read(K.checked(oldprep['target_metadata_ref']))['Dcheck']
assert len(queries)==len(metadata)==400
for query,meta in zip(queries,metadata):assert all(query[k]==meta[k] for k in V.IDENTITY)

if args.mode=='prepare':
    assert not (out/'prepared.json').exists() and not (out/'images').exists()
    base=np.load(K.checked(oldprep['conditions']),allow_pickle=False);teacher=np.load(K.checked(q['teacher400']),allow_pickle=False)
    assert base.shape==(400,9,1024) and teacher.shape==(400,1024) and teacher.dtype==np.float32 and np.isfinite(teacher).all()
    mapping=V.wrong_map(queries);conditions=np.stack([base[:,2],base[:,8],teacher,teacher[mapping]],axis=1)
    assert conditions.dtype==np.float32 and np.isfinite(conditions).all()
    assert np.array_equal(conditions[:,:2],base[:,[2,8]]) and np.array_equal(conditions[:,2],teacher) and np.array_equal(conditions[:,3],teacher[mapping])
    assert len({r['concept_id'] for r in queries[:80]})==8 and set(mapping[:80])==set(range(80))
    K.save_array(out/'conditions400.npy',conditions);K.save_array(out/'same_concept_wrong_indices.npy',mapping);K.write(out/'queries400.json',queries)
    K.write(out/'prepared.json',dict(status='COMPLETE_ALL400_PRIVILEGED_ENDPOINT_CONDITIONS_PRE_IMAGE',policy=K.ref(args.policy),conditions=K.ref(out/'conditions400.npy'),queries=K.ref(out/'queries400.json'),
        same_concept_wrong_indices=K.ref(out/'same_concept_wrong_indices.npy'),target_metadata_ref=oldprep['target_metadata_ref'],target_feature_ref=oldprep['target_feature_ref'],
        teacher400=q['teacher400'],baseline_features=q['base_features'],fixed_first80_queries=True,normal_EEG_method=False,no_training_or_strength_selection=True))
else:
    prepared=K.read(out/'prepared.json');conditions=np.load(K.checked(prepared['conditions']),allow_pickle=False)
    mapping=np.load(K.checked(prepared['same_concept_wrong_indices']),allow_pickle=False);records=V.records(c)
    for arm in V.ARMS:assert len(records[arm])==80 and [r['query_index'] for r in records[arm]]==list(range(80))
    ordered=[records[arm][i] for i in range(80) for arm in V.ARMS]
    if args.mode=='audit':
        assert not (out/'image_audit.json').exists();inodes=set();paths=set();duplicates={};constant=0
        for i in range(80):
            for j,arm in enumerate(V.ARMS):
                r=records[arm][i];assert r['arm']==arm and all(r[k]==queries[i][k] for k in V.IDENTITY)
                path=K.checked(r['image']);st=path.stat();key=(st.st_dev,st.st_ino)
                assert str(path) not in paths and key not in inodes;paths.add(str(path));inodes.add(key)
                pixels=np.asarray(Image.open(path).convert('RGB')).copy();assert pixels.shape==(512,512,3)
                assert hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixels_sha256']
                value=np.ascontiguousarray(conditions[i,j][None])
                assert hashlib.sha256(value.tobytes()).hexdigest()==r['condition_FP32_sha256']
                assert hashlib.sha256(value.astype(np.float16).tobytes()).hexdigest()==r['condition_FP16_sha256']
                assert r['noise']==records['direct'][i]['noise'] and len(r['noise']['initial'])==1 and len(r['noise']['ancestral'])==4
                if arm in V.NEW_ARMS:
                    ti=i if arm=='oracle_CLIPH' else int(mapping[i]);assert r['visual_condition_query_index']==ti
                    assert r['visual_condition_stimulus_id']==queries[ti]['stimulus_id'] and r['privileged_true_image_diagnostic'] and not r['normal_EEG_method']
                    assert r['prepared']==K.ref(out/'prepared.json') and Path(q['project_root']) in path.parents
                duplicates.setdefault(r['pixels_sha256'],[]).append([arm,i]);constant+=int(np.max(pixels.astype(float).std(axis=(0,1)))<2.55)
        K.write(out/'ordered_images80.json',ordered)
        K.write(out/'image_audit.json',dict(status='PASS_ALL320_ENDPOINT_IMAGE_PROVENANCE',images=320,new_images=160,reused_images=160,ordered_records=K.ref(out/'ordered_images80.json'),
            all_conditions_FP32_FP16_equal=True,all_query_noises_equal=True,no_duplicate_image_paths_or_inodes=True,near_constant_count=constant,exact_pixel_duplicate_groups=[v for v in duplicates.values() if len(v)>1]))
    else:
        assert not (out/'readout80.json').exists()
        done=K.read(out/'encode_completion.json');features=np.load(K.checked(done['features']),allow_pickle=False);assert features.shape==(80,4,768)
        import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
        import score_low_level as L
        assert K.ref(S.__file__)==q['source_metrics'] and K.ref(L.__file__)==q['source_low_level']
        target=np.load(K.checked(prepared['target_feature_ref']),allow_pickle=False);assert target.shape==(400,768)
        selfcheck=S.metric_arrays(target,target,queries,queries);assert np.all(selfcheck['DINO2AFC']==1) and np.all(selfcheck['gallery_top1']==1)
        arrays={arm:S.metric_arrays(features[:,j],target,queries[:80],queries) for j,arm in enumerate(V.ARMS)}
        for arm in V.ARMS:
            for key in ('PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error'):arrays[arm][key]=np.empty(80)
        zips={}
        try:
            for i in range(80):
                meta=metadata[i]
                if meta['image_zip'] not in zips:zips[meta['image_zip']]=zipfile.ZipFile(meta['image_zip'])
                raw=zips[meta['image_zip']].read(meta['image_member']);assert hashlib.sha256(raw).hexdigest()==meta['image_sha256']
                trgb,ty=L.pixels(io.BytesIO(raw));tm=L.moments(ty)
                for arm in V.ARMS:
                    rgb,y=L.pixels(K.checked(records[arm][i]['image']));arrays[arm]['PixCorr_RGB'][i]=L.pixcorr(rgb,trgb)
                    arrays[arm]['SSIM_Y'][i]=L.ssim_y(y,ty,tm);arrays[arm]['mean_RGB_absolute_error'][i]=float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())
        finally:
            for z in zips.values():z.close()
        temp=out/'metrics80.npz.pending';path=out/'metrics80.npz'
        with temp.open('xb') as f:np.savez_compressed(f,**{a+'__'+k:v for a,row in arrays.items() for k,v in row.items()})
        temp.replace(path)
        K.write(out/'readout80.json',dict(status='COMPLETE_PRIVILEGED_NATIVE_ATM_VISUAL_ENDPOINT_DIAGNOSTIC',queries=80,gallery=400,arms=list(V.ARMS),
            means={a:{k:float(v.mean()) for k,v in row.items()} for a,row in arrays.items()},fixed_comparisons=V.paired(arrays,queries[:80]),arrays=K.ref(path),
            source_metrics=q['source_metrics'],source_low_level=q['source_low_level'],bootstrap_replicates=1000,bootstrap_seed=1729,
            privileged_visual_arms=list(V.NEW_ARMS),normal_EEG_method_gain=False,new_training_updates=0,automatic_fit_or_expansion=False,second_method_or_SOTA=False,
            teacher_advantage_not_required_for_engineering_pass=True,all_eight_metrics_retained=True))
V.resources(c);print(json.dumps(dict(status='COMPLETE_ENDPOINT_'+args.mode.upper())),flush=True)
