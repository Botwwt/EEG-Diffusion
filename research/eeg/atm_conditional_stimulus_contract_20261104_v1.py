"""Three matched EEG encoder continuations; no adapter or prior changes."""
import json,sys
from pathlib import Path
import numpy as np
ARMS=('direct','T_mean','nce_continued','conditional','conditional_wrong')
MODELS=ARMS[2:]
CONTRASTS=(('conditional','direct'),('conditional','nce_continued'),('conditional','conditional_wrong'),('conditional','T_mean'),('nce_continued','direct'),('conditional_wrong','direct'))
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
IDENTITY=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')
_c=None
def setup(path):
    global _c
    q=json.loads(Path(path).read_bytes())
    assert q['status']=='LOCKED_NATIVE_ATM_EEG_CONDITIONAL_STIMULUS_TARGET_DIAGNOSTIC' and q['host']=='old' and q['seed']==1729
    assert q['epochs']==10 and q['batch']==160 and q['maximum_output_bytes']==1280<<20 and q['arms']==list(ARMS)
    assert q['new_models']==3 and q['updates_per_model']==830 and q['new_updates']==2490 and q['new_exposures']==396900
    if _c is None:
        sys.path.insert(0,str(Path(q['endpoint_policy']['path']).parent));import atm_native_visual_endpoint_contract_20261104_v1 as V
        cv=V.setup(q['endpoint_policy']['path']);K=cv['K'];K.checked(q['endpoint_policy'])
        assert K.read(K.checked(q['endpoint_completion']))['status']=='COMPLETE_PRIVILEGED_NATIVE_ATM_ENDPOINT_NO_METHOD_GAIN_CLAIM'
        launch=K.read(Path(q['endpoint_policy']['path']).parent/'actual_launch.json');assert not K.process(launch['PID'],launch['start_ticks'])['live']
        for pin in q['sources'].values():K.checked(pin)
        for pin in q['model_source_pins']:K.checked(pin)
        for key in ('EEG','encoder','visual_target','legacy_inputs'):K.checked(q[key])
        inputs=K.read(q['legacy_inputs']['path']);assert inputs['subjects']['sub-01']['input']==q['EEG'] and inputs['subjects']['sub-01']['models']['full']['checkpoint']==q['encoder']
        rows=np.array(inputs['T_array_rows'],np.int64);assert len(rows)==13230 and len(set(rows//10))==1323
        assert np.array_equal(rows.reshape(-1,10)//10,np.repeat((rows[::10]//10)[:,None],10,axis=1))
        oldprep=K.read(K.checked(cv['q']['base_prepared']));queries=K.read(K.checked(oldprep['queries']));assert len(queries)==400 and not set(rows)&{r['array_row'] for r in queries}
        oldrecords=K.read(K.checked(cv['q']['base_ordered_records']))
        baseline={arm:[r for r in oldrecords if r['arm']==arm] for arm in ARMS[:2]};assert all(len(v)==400 for v in baseline.values())
        _c=dict(q=q,K=K,V=V,cv=cv,out=Path(q['output_root']),repo=cv['repo'],parent=cv['parent'],T=rows,queries=queries,oldprep=oldprep,baseline=baseline)
    assert _c['q']==q;return _c
def resources(c,starting=False):
    q=c['q'];r=c['V'].resources(c['cv'],starting=starting)
    used=c['cv']['ctx']['base']['ctx']['C'].allocated(q['project_root']) if Path(q['project_root']).exists() else 0
    assert used<=q['maximum_output_bytes'];required=r['required_including_this_project']+q['maximum_output_bytes']-used
    assert r['free_bytes']>=required
    return dict(required_remaining_bytes=r['required_including_this_project'],required_including_this_project=required,free_bytes=r['free_bytes'],own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,inherited=r,all_prior_forecasts_retained=True)
def records(c,n):
    K=c['K'];result={a:c['baseline'][a][:n] for a in ARMS[:2]}
    for arm in MODELS:
        p=c['out']/'images'/arm/'images.jsonl';result[arm]=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        assert len(result[arm])==n
    return result
def paired(arrays,queries):
    concepts=sorted({r['concept_id'] for r in queries});groups=[[i for i,r in enumerate(queries) if r['concept_id']==g] for g in concepts]
    assert len(groups) in (8,40) and all(len(g)==10 for g in groups);draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));result={}
    for a,b in CONTRASTS:
        result[a+'-'+b]={}
        for key in METRICS:
            d=arrays[a][key]-arrays[b][key];m=np.array([d[g].mean() for g in groups])
            result[a+'-'+b][key]=dict(difference=float(d.mean()),descriptive_concept_bootstrap95=np.quantile(m[draw].mean(1),[.025,.975]).tolist(),positive_concepts=int((m>0).sum()),concepts=len(groups))
    return result
def interpret(pairs,n):
    harm={a:(pairs[a+'-direct']['DINO2AFC']['difference']<=-.01 and pairs[a+'-direct']['DINO2AFC']['descriptive_concept_bootstrap95'][1]<0 and pairs[a+'-direct']['within_concept10_two_afc']['difference']<0 and pairs[a+'-direct']['within_concept10_two_afc']['descriptive_concept_bootstrap95'][1]<0) for a in ('conditional','nce_continued','conditional_wrong')}
    return dict(action=('STOP_ALL3_JOINT_CLEAR_HARM' if all(harm.values()) else 'EXPAND_ALL3_TO400') if n==80 else 'COMPLETE_FIXED_TRAINING_TARGET_DIAGNOSTIC',joint_harm=harm,
        all8_metrics_retained=True,known_conditional_contrastive_baseline_NOT_novel_method=True,no_automatic_new_fits=True,second_method_or_SOTA=False)
