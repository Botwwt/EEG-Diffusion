"""Full image provenance audit and all-metric paired R80 readout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('audit','score'),required=True)
a=ap.parse_args();sys.path.insert(0,str(Path(a.policy).parent))
import enigma_R80_count_image_contract_20261104_v1 as C
started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
c=C.setup(a.policy);K=c['K'];q=c['q'];out=c['out'];C.resources(c)
assert K.ref(__file__)==q['sources']['CPU']
def guard():
    assert time.monotonic()-started<850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes']
try:
    if a.mode=='audit':
        assert not (out/'image_audit.json').exists()
        profile=K.read(out/'profile_completion.json');assert profile['status']=='PASS_NATIVE_PIXEL_NOISE_AND_ALL800_IP_TOKEN_PROFILE'
        tokens=np.load(K.checked(profile['tokens']),allow_pickle=False);assert tokens.shape==(200,4,4,2048)
        records={arm:[json.loads(s) for s in (out/'images'/arm/'images.jsonl').read_text().splitlines()] for arm in C.ARMS}
        assert all([r['query_index'] for r in rows]==list(range(200)) for rows in records.values())
        paths=set();inodes=set();duplicates={};near_constant=[];ordered=[]
        for i in range(200):
            guard();target=c['rows'][i]['target'];K.checked(target)
            for n,arm in enumerate(C.ARMS):
                row=records[arm][i];assert all(row[k]==c['rows'][i][k] for k in C.IDENTITY)
                assert row['target']==target and row['arm']==arm and row['condition_bank']==q['measurement_conditions']
                path=K.checked(row['image']);st=path.stat();inode=(st.st_dev,st.st_ino)
                assert str(path) not in paths and inode not in inodes
                assert path.is_relative_to(out/'images'/arm) and str(path)!=target['path'];paths.add(str(path));inodes.add(inode)
                rgb=np.asarray(Image.open(path).convert('RGB')).copy();assert rgb.shape==(512,512,3)
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['pixels_sha256']
                v=np.ascontiguousarray(c['values'][i,n][None]);assert hashlib.sha256(v.tobytes()).hexdigest()==row['condition_FP32_sha256']
                assert hashlib.sha256(v.astype(np.float16).tobytes()).hexdigest()==row['condition_FP16_sha256']
                assert hashlib.sha256(np.ascontiguousarray(tokens[i,n][None]).tobytes()).hexdigest()==row['tokens_FP16_sha256']
                assert row['noise']==records[C.ARMS[0]][i]['noise'] and len(row['noise']['initial'])==1 and len(row['noise']['ancestral'])==4
                assert not row['query_image_or_label_network_input'] and row['normal_inference_uses_only_EEG_and_known_R']
                assert row['UNet_NFE']==4 and row['EEG_deployment_forward_calls']==1
                duplicates.setdefault(row['pixels_sha256'],[]).append([arm,i])
                if np.max(rgb.astype(np.float64).std((0,1)))<2.55:near_constant.append([arm,i])
                ordered.append(row)
        K.write(out/'ordered_images.json',ordered)
        K.write(out/'image_audit.json',dict(status='PASS_ALL800_R80_IMAGE_IDENTITY_CONDITIONS_TOKENS_NOISE_PIXELS',
            images=800,queries=200,arms=list(C.ARMS),ordered_records=K.ref(out/'ordered_images.json'),
            all800_inodes_and_paths_distinct=True,near_constant_images=near_constant,
            exact_duplicate_pixel_groups=[v for v in duplicates.values() if len(v)>1],
            all800_FP32_FP16_tokens_and_noise_verified=True,source_event_to_stimulus_not_independently_replayed=True))
    else:
        assert not (out/'readout.json').exists()
        audit=K.read(out/'image_audit.json');assert audit['status']=='PASS_ALL800_R80_IMAGE_IDENTITY_CONDITIONS_TOKENS_NOISE_PIXELS'
        encoded=K.read(out/'encode_completion.json');assert encoded['status']=='COMPLETE_ALL800_AND200_TARGET_FROZEN_DINO'
        features=np.load(K.checked(encoded['features']),allow_pickle=False);target=np.load(K.checked(encoded['target']),allow_pickle=False)
        assert features.shape==(200,4,768) and target.shape==(200,768)
        self_values,_=C.metrics(target,target)
        assert np.all(self_values['DINO2AFC']==1) and np.all(self_values['gallery_top1']==1)
        import score_low_level as L
        assert K.ref(L.__file__)==q['runtime_sources']['low_level']
        ordered=K.read(K.checked(audit['ordered_records']));records={arm:[r for r in ordered if r['arm']==arm] for arm in C.ARMS}
        arrays={};matrices=[]
        for n,arm in enumerate(C.ARMS):
            arrays[arm],matrix=C.metrics(features[:,n],target);matrices.append(matrix)
            for key in C.METRICS[4:7]:arrays[arm][key]=np.empty(200,np.float64)
            paper=K.read(out/('paper_'+arm+'_completion.json'))
            assert paper['status']=='COMPLETE_ALL200_ORIGINAL_AUTHOR_EIGHT_METRICS' and paper['arm']==arm
            values=np.load(K.checked(paper['values']),allow_pickle=False);assert values.shape==(200,8)
            for k,key in enumerate(C.PAPER):arrays[arm][key]=values[:,k]
        for i in range(200):
            guard();trgb,ty=L.pixels(K.checked(c['rows'][i]['target']));tm=L.moments(ty)
            for arm in C.ARMS:
                rgb,y=L.pixels(K.checked(records[arm][i]['image']))
                arrays[arm]['PixCorr_RGB'][i]=L.pixcorr(rgb,trgb)
                arrays[arm]['SSIM_Y'][i]=L.ssim_y(y,ty,tm)
                arrays[arm]['mean_RGB_absolute_error'][i]=float(abs(rgb.mean((0,1))-trgb.mean((0,1))).mean())
        pairs,diffs=C.paired(arrays)
        K.save_array(out/'cosine_matrices_FP32.npy',np.stack(matrices,axis=1))
        for name,values in [('metrics.npz',{arm+'__'+key:values for arm,row in arrays.items() for key,values in row.items()}),('paired_differences.npz',diffs)]:
            path=out/name;tmp=path.with_name(path.name+'.pending')
            with tmp.open('xb') as f:np.savez_compressed(f,**values)
            tmp.replace(path)
        K.write(out/'readout.json',dict(status='COMPLETE_FOUR_ARM_R80_200_ALL15_METRICS',
            queries=200,subjects=1,physical_R=80,gallery=200,arms=list(C.ARMS),metrics=list(C.METRICS),
            means={arm:{key:float(v.mean()) if np.isfinite(v).all() else None for key,v in row.items()} for arm,row in arrays.items()},
            defined_counts={arm:{key:int(np.isfinite(v).sum()) for key,v in row.items()} for arm,row in arrays.items()},
            fixed_comparisons=pairs,arrays=K.ref(out/'metrics.npz'),paired_differences=K.ref(out/'paired_differences.npz'),
            bootstrap=dict(seed=1729,unit='concept_one_stimulus_per_concept',replicates=1000),
            all800_images_before_any_effect_read=True,no_within_concept10_metrics_defined=True,
            distance_metrics_lower_better=['mean_RGB_absolute_error','eval_effnet','eval_swav'],
            single_subject_T1323_reference_NOT_full1654_or10subject_benchmark=True,
            no_new_training_or_strength_seed_scan=True,all_metric_tradeoffs_retained=True,
            no_semantic_noninferiority_from_crossing_CI=True,originality_or_SOTA_not_established=True))
    guard();C.resources(c)
    print(json.dumps(dict(status='COMPLETE_'+a.mode,seconds=time.monotonic()-started,
        extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0)),flush=True)
except BaseException as exc:
    K.write(out/(a.mode+'_failure.json'),dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',error=repr(exc),traceback=traceback.format_exc()));raise
