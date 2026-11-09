"""Independent D37 fixed-gallery ranking and whole-concept bootstrap arithmetic.

Uses the same NumPy backend, not a second image encoder or UNet implementation.
"""
import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np
import brainhive_temporal_image_screen_20261028_v1 as U


def metrics(features,target,count,queries):
    x=features/np.linalg.norm(features,axis=1,keepdims=True);y=target/np.linalg.norm(target,axis=1,keepdims=True);similarity=x@y.T
    out={k:[] for k in ['target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc']}
    for i in range(count):
        correct=similarity[i,i];same=np.array([j for j,q in enumerate(queries) if q['concept_id']==queries[i]['concept_id']]);assert len(same)==10
        all_others=np.delete(similarity[i],i);local_others=similarity[i,same[same!=i]]
        out['target_cosine'].append(float(correct));out['DINO2AFC'].append(float((np.count_nonzero(all_others<correct)+.5*np.count_nonzero(all_others==correct))/399))
        out['gallery_top1'].append(float(int(np.argmax(similarity[i]))==i));out['within_concept10_top1'].append(float(int(same[np.argmax(similarity[i,same])])==i))
        out['within_concept10_two_afc'].append(float((np.count_nonzero(local_others<correct)+.5*np.count_nonzero(local_others==correct))/9))
    return {k:np.array(v,np.float64) for k,v in out.items()}


def paired(a,b,queries):
    keys=['target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc']
    concepts=sorted({q['concept_id'] for q in queries});groups=[[i for i,q in enumerate(queries) if q['concept_id']==c] for c in concepts]
    matrix=np.array([[(a[k][ids]-b[k][ids]).mean() for k in keys] for ids in groups]);rng=np.random.default_rng(1729)
    choices=rng.integers(0,len(groups),size=(1000,len(groups)));draws=matrix[choices].mean(1);limits=np.quantile(draws,[.025,.975],axis=0)
    return {k:dict(difference=float((a[k]-b[k]).mean()),descriptive_concept_bootstrap95=limits[:,j].tolist(),positive_concepts=int((matrix[:,j]>0).sum()),concepts=len(groups)) for j,k in enumerate(keys)}


def run(spec_path,stage):
    start=time.monotonic();spec,_,queries,_=U.prepared(spec_path);folder=U.B/f'screen{stage}/score';done=U.read(folder/'completion.json')
    assert done['status']==f'D37_ALL_THREE_FIXED{stage}_IMAGE_ARMS_SCORED' and done['spec']['sha256']==U.sha(spec_path)
    features_done=U.read(U.check(done['feature_completion']));candidate=np.load(U.check(features_done['features']),allow_pickle=False)
    baseline=np.load(U.check(spec['baseline_features']),allow_pickle=False);target=np.load(U.check(spec['target_features']),allow_pickle=False)
    reference=U.read(U.check(done['metric_arrays']));result={}
    for i,arm in enumerate(U.BASE_ARMS):result[arm]=metrics(baseline[:stage,i],target,stage,queries)
    for i,arm in enumerate(U.ARMS):result[arm]=metrics(candidate[:,i],target,stage,queries)
    maximum=0.
    for arm in result:
        for key,value in result[arm].items():maximum=max(maximum,float(np.abs(value-reference[arm][key]).max()));assert np.allclose(value,reference[arm][key],atol=1e-12,rtol=0)
    against={arm:paired(result[arm],result['ordinary'],queries[:stage]) for arm in U.ARMS}
    controls={arm:paired(result['conditional'],result[arm],queries[:stage]) for arm in ['flat_residual','same_concept_wrong']}
    for observed,expected in [(against,done['paired_against_old_ordinary']),(controls,done['primary_minus_matched_controls'])]:
        for arm in observed:
            for key in observed[arm]:
                assert observed[arm][key]['concepts']==expected[arm][key]['concepts'] and observed[arm][key]['positive_concepts']==expected[arm][key]['positive_concepts']
                assert abs(observed[arm][key]['difference']-expected[arm][key]['difference'])<1e-12
                assert np.allclose(observed[arm][key]['descriptive_concept_bootstrap95'],expected[arm][key]['descriptive_concept_bootstrap95'],atol=1e-12,rtol=0)
    p=against['conditional'];mean=p['DINO2AFC']['difference'];lo,hi=p['DINO2AFC']['descriptive_concept_bootstrap95']
    positive=mean>=.01 and p['target_cosine']['difference']>0 and p['within_concept10_top1']['difference']>=0
    action='STOP_INSTANCE' if mean<=-.02 and hi<0 else 'EXPAND400' if stage==80 else 'STOP_INSTANCE' if hi<.01 else 'PROMISING_FOR_FULL_EVALUATION' if positive and lo>0 else 'STOP_INSTANCE'
    assert action==done['primary_decision']['action']
    value=dict(status='D37_ALL_FIXED_QUERY_RANKS_CONTROL_DIFFERENCES_AND_BOOTSTRAPS_INDEPENDENT_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage=stage,source=U.ref(__file__),score_completion=U.ref(folder/'completion.json'),all_scalar_metrics=stage*6*5,
        maximum_metric_absolute_error=maximum,all5_fixed_contrasts_1000_bootstraps_verified=True,primary_decision=action,
        same_Numpy_backend_not_independent_image_feature_encoder=True,no_new_GPU_or_model_fit=True,seconds=time.monotonic()-start)
    U.write(folder/'independent_CPU_audit.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--stage',type=int,choices=[80,400],required=True);a=p.parse_args();run(a.spec,a.stage)
