"""Fixed full400 six-arm image interventions after genuine BN-state closure."""
import json,sys
from pathlib import Path
import numpy as np
ARMS=('direct','nce_continued','conditional','BN_only','nce_direction_only','nce_radius_only')
BASES=ARMS[:3]
MODELS=ARMS[3:]
IDENTITY=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
CONTRASTS=(('BN_only','direct'),('BN_only','nce_continued'),('nce_direction_only','direct'),('nce_direction_only','nce_continued'),('nce_radius_only','direct'),('nce_continued','nce_radius_only'),('nce_continued','direct'),('conditional','nce_continued'))
_c=None
def setup(path):
    global _c
    q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_FULL400_BN_RADIUS_IMAGE_INTERVENTIONS' and q['host']=='old' and q['seed']==1729
    assert q['arms']==list(ARMS) and q['maximum_output_bytes']==768<<20 and q['new_training_updates']==0 and q['new_images']==1200 and q['fixed_queries']==400
    if _c is None:
        sys.path.insert(0,str(Path(q['BN_policy']['path']).parent));import atm_BN_radius_contract_20261104_v1 as B
        bc=B.setup(q['BN_policy']['path']);K=bc['K'];K.checked(q['BN_policy']);done=K.read(K.checked(q['BN_completion']))
        assert done['status']=='COMPLETE_BN_RADIUS_NATIVE_FUNCTION_PREPARATION_NO_IMAGES' and done['policy']==q['BN_policy'] and done['original_failure_retained'] and done['completed_BN_prefix_SHA_unchanged']
        for directory in (Path(q['BN_policy']['path']).parent,Path(done['recovery_policy']['path']).parent):
            launch=K.read(directory/'actual_launch.json');assert not K.process(launch['PID'],launch['start_ticks'])['live']
        rq=K.read(K.checked(done['recovery_policy']))
        for pin in rq['sources'].values():K.checked(pin)
        job=K.read(K.checked(done['job']));assert job['status']=='COMPLETED' and job['exit_code']==0 and not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
        assert K.read(K.checked(q['BN_CPU_audit']))['status']=='PASS_FULL_T_BN_STATE_AND_ALL400_NATIVE_CPU_CONDITIONS_RADIUS_PROJECTOR_AUDIT'
        gpu=K.read(K.checked(done['GPU_completion']));K.checked(gpu['conditions']);up=bc['up'];oldprep=K.read(up['out']/'prepared.json')
        oldaudit=K.read(up['out']/'image_audit400.json');oldrecords=K.read(K.checked(oldaudit['ordered_records']));baseline={arm:[r for r in oldrecords if r['arm']==arm] for arm in BASES}
        assert all(len(v)==400 and [r['query_index'] for r in v]==list(range(400)) for v in baseline.values())
        for pin in q['sources'].values():K.checked(pin)
        for key in ('baseline_features','baseline_metrics','source_metrics','source_low_level'):K.checked(q[key])
        _c=dict(q=q,K=K,B=B,bc=bc,up=up,out=Path(q['output_root']),repo=bc['repo'],parent=bc['parent'],cv=up['cv'],queries=bc['queries'],baseline=baseline,BN_GPU=gpu,oldprep=oldprep)
    assert _c['q']==q;return _c
def resources(c,starting=False):
    parent=c['B'].resources(c['bc'],starting=starting);used=c['up']['cv']['ctx']['base']['ctx']['C'].allocated(c['q']['project_root']) if Path(c['q']['project_root']).exists() else 0
    assert used<=c['q']['maximum_output_bytes'];required=parent['required_including_this_project']+c['q']['maximum_output_bytes']-used;assert parent['free_bytes']>=required
    return dict(required_remaining_bytes=parent['required_including_this_project'],required_including_this_project=required,free_bytes=parent['free_bytes'],own_used_bytes=used,own_remaining_bytes=c['q']['maximum_output_bytes']-used,inherited=parent,all_prior_forecasts_retained=True)
def prepare(c,path):
    K=c['K'];out=c['out'];assert not (out/'prepared.json').exists();values=np.load(K.checked(c['BN_GPU']['conditions']),allow_pickle=False);assert values.shape==(400,6,1024) and values.dtype==np.float32 and np.isfinite(values).all()
    old=np.load(K.checked(c['oldprep']['conditions']),allow_pickle=False);assert np.array_equal(values[:,:3],old[:,[0,2,3]])
    result=dict(status='COMPLETE_FROZEN_SIX_ARM_IMAGE_INPUTS',policy=K.ref(path),conditions=c['BN_GPU']['conditions'],queries=c['up']['K'].ref(c['bc']['out']/'queries400.json'),target_metadata_ref=c['oldprep']['target_metadata_ref'],target_feature_ref=c['oldprep']['target_feature_ref'],
        BN_completion=c['q']['BN_completion'],BN_CPU_audit=c['q']['BN_CPU_audit'],baseline_features=c['q']['baseline_features'],baseline_metrics=c['q']['baseline_metrics'],
        normal_inference_only_EEG=True,query_images_or_concepts_not_inference_inputs=True,new_ordinary_used=False,EEG_forward_cost=dict(BN_only=1,nce_direction_only=2,nce_radius_only=2),original_ordinary_unchanged=True)
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
    return dict(action='COMPLETE_FIXED_FULL400_CAUSAL_INPUT_INTERVENTIONS',all8_metrics_and8_contrasts_retained=True,no_candidate_selection_or_automatic_fit=True,no_equivalence_from_crossing_CI=True,BN_and_radius_classical_controls_NOT_novel_method=True,second_method_or_SOTA=False)
