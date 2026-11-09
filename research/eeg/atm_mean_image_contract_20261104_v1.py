"""Full400 image falsification of T-only native encoder mean shifts, no fit."""
import json,sys
from pathlib import Path
import numpy as np
ARMS=('direct','nce_continued','conditional','restore_old_mean','restore_visual_mean','opposite_old_shift')
BASES=ARMS[:3];MODELS=ARMS[3:]
IDENTITY=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
CONTRASTS=(('restore_old_mean','nce_continued'),('restore_visual_mean','nce_continued'),('restore_old_mean','restore_visual_mean'),('restore_old_mean','direct'),('restore_visual_mean','direct'),('restore_old_mean','opposite_old_shift'),('opposite_old_shift','nce_continued'),('conditional','nce_continued'),('nce_continued','direct'))
_c=None
def setup(path):
    global _c
    q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_FULL400_T_MEAN_SHIFT_IMAGE_FALSIFICATION' and q['host']=='old' and q['seed']==1729
    assert q['arms']==list(ARMS) and q['new_training_updates']==0 and q['new_images']==q['reused_images']==1200 and q['maximum_output_bytes']==768<<20
    if _c is None:
        sys.path.insert(0,str(Path(q['reference_image_policy']['path']).parent));import atm_BN_image_contract_20261104_v1 as B
        base=B.setup(q['reference_image_policy']['path']);K=base['K'];K.checked(q['reference_image_policy']);K.checked(q['mean_policy'])
        mp=K.read(q['mean_policy']['path']);done=K.read(K.checked(q['mean_completion']));assert done['status']=='COMPLETE_FULL_T_NATIVE_EEG_MEAN_SHIFT_PREPARATION_NO_IMAGES' and done['policy']==q['mean_policy'] and done['N']==13230 and done['concepts']==1323
        owner=K.read(Path(q['mean_policy']['path']).parent/'actual_launch.json');assert not K.process(owner['PID'],owner['start_ticks'])['live'] and not (Path(mp['output_root'])/'failure.json').exists()
        image_owner=K.read(Path(q['reference_image_policy']['path']).parent/'actual_launch.json');assert not K.process(image_owner['PID'],image_owner['start_ticks'])['live']
        K.checked(done['conditions']);K.checked(done['statistics'])
        for pin in [*mp['sources'].values(),*q['sources'].values()]:K.checked(pin)
        for key in ('baseline_features','baseline_metrics','source_metrics','source_low_level'):K.checked(q[key])
        assert all(v['all400_old_GPU_conditions_CPU_max_error']<5e-5 for v in done['all_T_encoder_certificates'].values())
        _c=dict(base,q=q,K=K,B=B,base=base,out=Path(q['output_root']),mean_done=done,mean_policy=mp)
    assert _c['q']==q;return _c

def resources(c,starting=False):
    r=c['B'].resources(c['base'],starting=starting);allocated=c['up']['cv']['ctx']['base']['ctx']['C'].allocated
    mean_used=allocated(c['mean_policy']['project_root']);mean_remaining=c['mean_policy']['maximum_output_bytes']-mean_used;assert mean_remaining>=0
    own=allocated(c['q']['project_root']) if Path(c['q']['project_root']).exists() else 0;assert 0<=own<=c['q']['maximum_output_bytes']
    previous=r['required_including_this_project']+mean_remaining;required=previous+c['q']['maximum_output_bytes']-own;assert r['free_bytes']>=required
    return dict(required_remaining_bytes=previous,required_including_this_project=required,free_bytes=r['free_bytes'],own_used_bytes=own,own_remaining_bytes=c['q']['maximum_output_bytes']-own,mean128MiB_remaining=mean_remaining,inherited=r,all_prior_forecasts_retained=True)

def prepare(c,path):
    K=c['K'];out=c['out'];assert not (out/'prepared.json').exists();values=np.load(K.checked(c['mean_done']['conditions']),allow_pickle=False)
    original=np.load(K.checked(c['base']['BN_GPU']['conditions']),allow_pickle=False)
    assert values.shape==(400,6,1024) and values.dtype==np.float32 and np.isfinite(values).all() and np.array_equal(values[:,:3],original[:,:3])
    result=dict(status='COMPLETE_FIXED_T_ONLY_MEAN_SHIFT_IMAGE_INPUTS',policy=K.ref(path),conditions=c['mean_done']['conditions'],queries=K.ref(c['up']['out']/'queries400.json'),mean_completion=c['q']['mean_completion'],mean_policy=c['q']['mean_policy'],
        target_metadata_ref=c['oldprep']['target_metadata_ref'],target_feature_ref=c['oldprep']['target_feature_ref'],baseline_features=c['q']['baseline_features'],baseline_metrics=c['q']['baseline_metrics'],normal_inference_only_EEG=True,
        query_images_or_concepts_not_inference_inputs=True,all_means_fitted_T_only=True,new_ordinary_used=False,EEG_forward_cost={a:1 for a in MODELS},original_ordinary_unchanged=True)
    K.write(out/'prepared.json',result)

def records(c,n):
    assert n==400;result=dict(c['baseline'])
    for arm in MODELS:
        p=c['out']/'images'/arm/'images.jsonl';result[arm]=[json.loads(s) for s in p.read_text().splitlines()];assert len(result[arm])==400
    return result

def paired(arrays,queries):
    concepts=sorted({r['concept_id'] for r in queries});groups=[[i for i,r in enumerate(queries) if r['concept_id']==g] for g in concepts];assert len(groups)==40 and all(len(v)==10 for v in groups)
    draw=np.random.default_rng(1729).integers(0,40,(1000,40));result={}
    for a,b in CONTRASTS:
        result[a+'-'+b]={}
        for key in METRICS:
            d=arrays[a][key]-arrays[b][key];m=np.array([d[g].mean() for g in groups]);result[a+'-'+b][key]=dict(difference=float(d.mean()),descriptive_concept_bootstrap95=np.quantile(m[draw].mean(1),[.025,.975]).tolist(),positive_concepts=int((m>0).sum()),concepts=40)
    return result

def interpret(pairs,n):
    assert n==400
    return dict(action='COMPLETE_FIXED_T_MEAN_SHIFT_IMAGE_FALSIFICATION',all8_metrics_and9_contrasts_retained=True,no_candidate_selection_or_automatic_fit=True,no_equivalence_from_crossing_CI=True,classical_mean_shift_NOT_new_method_or_EEG_information=True,second_method_or_SOTA=False)
