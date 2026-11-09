"""Read effects only after all3 fixed fits plus all3 whole-stage image/features."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,csv,datetime,json,time,traceback
from pathlib import Path
import AR_anchored_images_contract_CPU_20261029_v1 as C

METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc')

def load_arm_features(a,arm,stage,panel):
    import numpy as np
    root=Path(a['output_root'])/arm;path=root/'artifacts'/f'stage{stage}.json';audit=json.loads(path.read_bytes())
    assert audit['status']=='COMPLETE_FIXED_AR_ALL_QUERY_ARTIFACT_AUDIT' and audit['stage']==stage and audit['arm']==arm
    assert audit['expected_queries']==audit['valid_queries']==stage and audit['invalid_queries']==0
    assert audit['all_query_ids']==[q['query_id'] for q in panel['queries'][:stage]]
    feature_parts=[];references=[];image_inodes=[]
    for lo in range(0,stage,20):
        out=root/'feature_chunks'/f'{lo:05d}_{lo+20:05d}';done_path=out/'completion.json';done=json.loads(done_path.read_bytes())
        assert done['status']=='COMPLETE_FIXED_AR_FROZEN_DINO_FEATURE_CHUNK' and done['arm']==arm and done['range']==[lo,lo+20]
        assert done['stage']==(80 if lo<80 else 400) and done['queries']==20 and done['feature_dimension']==768
        assert done['DINO_weight']['sha256']==C.DINO_SHA and done['image_evaluation_source']['sha256']==a['image_evaluation_sha256']
        assert done['target_features_or_scores_read'] is False and done['effect_metrics_computed']==0
        rows=json.loads(C.checked(done['rows']).read_bytes());x=np.load(C.checked(done['features']),allow_pickle=False)
        assert x.shape==(20,768) and x.dtype==np.float32 and np.isfinite(x).all() and np.all(np.linalg.norm(x,axis=1)>0)
        for index,row in enumerate(rows,lo):
            assert row['query_index']==index and row['query_id']==panel['queries'][index]['query_id']
            record,record_ref=C.validate_record(a,arm,index,panel['queries'][index]);assert row['record']['sha256']==record_ref['sha256']==audit['records'][index]['record']['sha256']
            C.checked(row['image']);assert row['image']['sha256']==record['image']['sha256'] and row['raw_pixel_SHA256']==record['raw_pixel_SHA256']
            image,_=C.image_paths(a,arm,index);stat=image.stat();image_inodes.append((stat.st_dev,stat.st_ino))
        assert len(rows)==20;feature_parts.append(x);references.append(dict(completion=C.ref(done_path),features=done['features'],rows=done['rows']))
    return np.concatenate(feature_parts),dict(artifact_audit=C.ref(path),chunks=references),image_inodes

def independent_arrays(features,targets,queries,gallery):
    """Separate direct rank implementation; same frozen feature inputs."""
    import numpy as np
    normalized=features/np.linalg.norm(features,axis=1,keepdims=True);gallery_unit=targets/np.linalg.norm(targets,axis=1,keepdims=True)
    similarity=normalized@gallery_unit.T;positions={q['stimulus_id']:i for i,q in enumerate(gallery)};result={name:[] for name in METRICS}
    for index,q in enumerate(queries):
        target=positions[q['stimulus_id']];scores=similarity[index];positive=scores[target]
        others=np.delete(scores,target);within=[i for i,row in enumerate(gallery) if row['concept_id']==q['concept_id']];assert len(within)==10
        competitors=[i for i in within if i!=target]
        row=dict(target_cosine=float(positive),DINO2AFC=float((np.count_nonzero(others<positive)+.5*np.count_nonzero(others==positive))/399),gallery_top1=float(np.argmax(scores)==target),within_concept10_top1=float(within[int(np.argmax(scores[within]))]==target),within_concept10_two_afc=float((sum(scores[i]<positive for i in competitors)+.5*sum(scores[i]==positive for i in competitors))/9))
        for name in METRICS:result[name].append(row[name])
    return {name:np.array(values,np.float64) for name,values in result.items()}

def independent_contrast(candidate,baseline,queries):
    import numpy as np
    concepts=sorted({q['concept_id'] for q in queries});indices=[[i for i,q in enumerate(queries) if q['concept_id']==concept] for concept in concepts]
    difference={name:candidate[name]-baseline[name] for name in METRICS}
    by_concept=np.array([[difference[name][group].mean() for name in METRICS] for group in indices]);draws=np.random.default_rng(1729).integers(0,len(concepts),(1000,len(concepts)))
    boot=np.array([by_concept[chosen].mean(0) for chosen in draws]);interval=np.percentile(boot,[2.5,97.5],axis=0)
    return {name:dict(difference=float(difference[name].mean()),descriptive_concept_bootstrap95=interval[:,i].tolist()) for i,name in enumerate(METRICS)}

def run(admission,stage):
    a,_=C.load_admission(admission);C.validate_sources(a,Path(__file__));policy,panel=C.validate_panel(a);assert stage in (80,400);C.validate_expansion(a,stage)
    # Required before opening target features or deriving a single effect.
    fits={arm:C.validate_fit(a,arm)[1] for arm in C.ARMS}
    import numpy as np
    features={};inputs={};all_inodes=[]
    for arm in C.ARMS:features[arm],inputs[arm],inodes=load_arm_features(a,arm,stage,panel);all_inodes.extend(inodes)
    assert len(set(all_inodes))==len(all_inodes),'Cross-arm hardlink/image path reuse is not an independent generated artifact'
    root=Path(a['output_root']);screens=root/'screens';screens.mkdir(exist_ok=True);out=screens/f'stage{stage}';assert not out.exists();out.mkdir();began=time.monotonic()
    try:
        cache=json.loads(C.checked(a['target_cache']).read_bytes());assert cache['panel']['sha256']==C.PANEL_SHA and cache['status']=='EXISTING_LOCKED_TARGET_FEATURES_REUSED_WITH_COMPLETE_AR_EEG_IDENTITY_JOIN'
        assert a['target_features']['sha256']==cache['features']['sha256'] and a['target_features']['bytes']==cache['features']['bytes']
        assert cache['encoder']['weight_sha256']==C.DINO_SHA and cache['encoder']['image_evaluation']['sha256']==a['image_evaluation_sha256']
        targets=np.load(C.checked(a['target_features']),allow_pickle=False);assert targets.shape==(400,768) and np.isfinite(targets).all()
        import AR_anchored_image_screen_math_20261029_v1 as math
        assert C.sha(Path(math.__file__))==C.MATH_SHA
        result,arrays=math.score_features(features,targets,panel['queries'][:stage],panel['target_gallery'],stage)
        independent={arm:independent_arrays(features[arm],targets,panel['queries'][:stage],panel['target_gallery']) for arm in C.ARMS}
        errors={arm:{name:float(np.max(np.abs(independent[arm][name]-arrays[arm][name]))) for name in METRICS} for arm in C.ARMS}
        assert all(value<=2e-6 for row in errors.values() for value in row.values())
        pairs={'paired_minus_native':('paired_anchored','native_global'),'paired_minus_donor':('paired_anchored','donor_anchored'),'donor_minus_native':('donor_anchored','native_global')}
        secondary={name:independent_contrast(independent[first],independent[second],panel['queries'][:stage]) for name,(first,second) in pairs.items()}
        max_ci_error=max(abs(float(expected)-float(actual)) for contrast in pairs for name in METRICS for expected,actual in zip(secondary[contrast][name]['descriptive_concept_bootstrap95'],result['contrasts'][contrast][name]['descriptive_concept_bootstrap95']))
        assert max_ci_error<=2e-6 and math.stage_decision(secondary,stage)==result['decision']
        assert time.monotonic()-began<a['resources']['maximum_CPU_score_seconds']
        table=out/'per_query.csv'
        with table.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['arm','query_index','query_id','stimulus_id','concept_id',*METRICS]);writer.writeheader()
            for arm in C.ARMS:
                for index,q in enumerate(panel['queries'][:stage]):writer.writerow(dict(arm=arm,query_index=index,query_id=q['query_id'],stimulus_id=q['stimulus_id'],concept_id=q['concept_id'],**{name:float(arrays[arm][name][index]) for name in METRICS}))
        np.savez_compressed(out/'all_metrics.npz',**{arm+'__'+name:array for arm,row in arrays.items() for name,array in row.items()})
        C.write_once(out/'independent_CPU_rescore.json',dict(status='PASS_SEPARATE_RANK_AND_WHOLE_CONCEPT_BOOTSTRAP_IMPLEMENTATION',feature_encoder_independently_rerun=False,raw_fixed_features_reused=True,metric_cells=3*stage*len(METRICS),arrays_max_abs_errors=errors,max_interval_abs_error=max_ci_error,all_three_fixed_contrasts_recomputed=True,decision_identical=True))
        result.update(status='COMPLETE_FIXED_THREE_ARM_AR_IMAGE_SCREEN',stage=stage,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),panel=a['panel'],policy=a['policy'],fit_closures=fits,inputs=inputs,
            image_screen_source_sha256=C.MATH_SHA,locked_metric_source_sha256=C.METRIC_SHA,target_cache=a['target_cache'],target_features=a['target_features'],per_query_table=C.ref(table),all_arrays=C.ref(out/'all_metrics.npz'),independent_CPU_rescore=C.ref(out/'independent_CPU_rescore.json'),
            full_query_denominator_per_arm=stage,all3_fixed_training_and_all3_artifact_feature_closures_required_before_effect_reads=True,stage80_images_and_features_reused=(stage==400),training_seed_variability_assessed=False,already_exposed_development_panel=True,independent_confirmation=False,SOTA_established=False,seconds=time.monotonic()-began)
        C.write_once(out/'summary.json',result)
    except BaseException as error:
        C.write_once(out/'failure.json',dict(status='AR_FIXED_SCREEN_FAILURE_PRESERVED_NO_PARTIAL_EFFECT_PUBLICATION',error=repr(error),traceback=traceback.format_exc(),stage=stage,full_query_denominator_per_arm=stage,seconds=time.monotonic()-began));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);p.add_argument('--stage',type=int,required=True);q=p.parse_args();run(q.admission,q.stage)
