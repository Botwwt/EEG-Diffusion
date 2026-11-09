"""All-four-arm image decision after exact fits/artifacts/features are closed."""
import argparse,datetime,hashlib,json,time
from pathlib import Path
import numpy as np
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as locked
from AR_anchored_images_score_CPU_20261029_v1 import independent_arrays,independent_contrast

ARMS=('coupled','visual','marginal','donor')
PAIRS={'coupled_minus_visual':('coupled','visual'),'coupled_minus_marginal':('coupled','marginal'),'coupled_minus_donor':('coupled','donor'),'visual_minus_marginal':('visual','marginal')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decision(contrasts,stage):
    assert stage in (80,400)
    main=contrasts['coupled_minus_visual']['DINO2AFC']
    if stage==80:
        return 'STOP_INSTANCE' if main['difference']<=-.02 and main['descriptive_concept_bootstrap95'][1]<0 else 'EXPAND400'
    passed=[]
    for name in ('coupled_minus_visual','coupled_minus_marginal'):
        row=contrasts[name];passed.append(row['DINO2AFC']['difference']>=.01 and row['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and row['within_concept10_top1']['difference']>=0)
    return 'PROMISING_FOR_MULTISUBJECT_COMPLETE_EVALUATION' if all(passed) else 'STOP_INSTANCE'
def run(received,out):
    began=time.monotonic();manifest=json.loads((received/'receiving_completion.json').read_bytes());assert manifest['status']=='BOTH_NATIVE_MARGINAL_FIXED_FINAL_AND_ALL4_IMAGE_FEATURE_AUDITS_EXACTLY_RECEIVED'
    stage=manifest['stage'];assert stage in (80,400)
    for item in manifest['files'].values():
        p=Path(item['local']);assert p.stat().st_size==item['remote']['bytes'] and sha(p)==item['remote']['sha256']
    # Full fit/model/Adam/coverage and artifact audits precede target features.
    fits={};audits={}
    for host in ('old','new'):
        fits[host]=json.loads((received/host/'fit_completion.json').read_bytes());cpu=json.loads((received/host/'fit_CPU_audit.json').read_bytes());audit=json.loads((received/host/'artifact_audit.json').read_bytes())
        assert fits[host]['status']=='FULL_T_NATIVE_MARGINAL_FIXED_FINAL_FIT_COMPLETE' and fits[host]['epoch']==10 and fits[host]['EEG_updates']==2070
        assert cpu['status']=='PASS_FULL_T_FIXED_FINAL_EEG_POSTERIOR_OPTIMIZER_COVERAGE_AND_RNG_CPU' and cpu['final_model']==fits[host]['final_model']
        assert audit['status']=='PASS_INDEPENDENT_NATIVE_MARGINAL_ALL_QUERY_ARTIFACT_SEED_EEG_ROW_AND_FEATURE_AUDIT' and audit['stage']==stage and audit['valid_per_arm']==stage and audit['invalid_queries']==0
        audits[host]=audit
    assert fits['old']['initial_EEG_model_digest']==fits['new']['initial_EEG_model_digest']
    for key in ('panel','DINO_weight','image_evaluation'):assert audits['old'][key]['sha256']==audits['new'][key]['sha256']
    panel=json.loads((received/'old/panel400.json').read_bytes());gallery=panel['target_gallery'];queries=panel['queries'][:stage]
    assert len(gallery)==400 and panel['queries']==gallery and len({q['query_id'] for q in gallery})==400
    assert len({q['concept_id'] for q in queries})==stage//10
    cache=json.loads((received/'old/target_cache.json').read_bytes());assert cache['panel']['sha256']==sha(received/'old/panel400.json')
    assert cache['features']['sha256']==sha(received/'old/targets400.npy')
    assert cache['encoder']['weight_sha256']=='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'
    assert sha(Path(locked.__file__))=='bc9610b97501cdf078789f913b0520b023c0079801e24c37093cbac406189009'
    targets=np.load(received/'old/targets400.npy',allow_pickle=False);assert targets.shape==(400,768) and np.isfinite(targets).all()
    arrays={};separate={};features={}
    for arm in ARMS:
        host='old' if arm in ('coupled','donor') else 'new';features[arm]=np.load(received/host/(arm+'_features.npy'),allow_pickle=False)
        assert features[arm].shape==(stage,768) and np.isfinite(features[arm]).all()
        arrays[arm]=locked.metric_arrays(features[arm],targets,queries,gallery)
        separate[arm]=independent_arrays(features[arm],targets,queries,gallery)
        for metric in locked.METRICS:assert np.max(np.abs(arrays[arm][metric]-separate[arm][metric]))<=2e-6
    contrasts={name:locked.compare(arrays[a],arrays[b],queries) for name,(a,b) in PAIRS.items()}
    alternative={name:independent_contrast(separate[a],separate[b],queries) for name,(a,b) in PAIRS.items()}
    for name in PAIRS:
        for metric in locked.METRICS:
            assert abs(contrasts[name][metric]['difference']-alternative[name][metric]['difference'])<=2e-6
            assert np.max(np.abs(np.array(contrasts[name][metric]['descriptive_concept_bootstrap95'])-alternative[name][metric]['descriptive_concept_bootstrap95']))<=2e-6
    assert decision(contrasts,stage)==decision(alternative,stage)
    out.mkdir();np.savez_compressed(out/'per_query_all_metrics.npz',**{arm+'__'+metric:values for arm,row in arrays.items() for metric,values in row.items()})
    np.savez_compressed(out/'all4_frozen_features.npz',targets=targets,**features)
    # Explicit complete training exposure, weighted by the actual last batch.
    training={}
    for host in ('old','new'):
        events=json.loads((received/host/'fit_events.json').read_bytes());assert len(events)==2070
        summary=[]
        for epoch in range(1,11):
            items=[e for e in events if e['epoch']==epoch];weights=np.array([e['queries'] for e in items]);assert len(items)==207 and weights.sum()==13230
            summary.append(dict(epoch=epoch,**{key:float(np.average([e[key] for e in items],weights=weights)) for key in ('NLL_nats_per_query','KL_nats_per_query','ELBO_nats_per_token','EEG_KL_nats_per_token','preclip_EEG_gradient','backbone_gradient')}))
        training[host]=summary
    result=dict(status='COMPLETE_ALL4_FIXED_FINAL_NATIVE_MARGINAL_IMAGE_SCREEN_AND_INDEPENDENT_CPU_RESCORE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,decision=decision(contrasts,stage),
        means={arm:{metric:float(value.mean()) for metric,value in row.items()} for arm,row in arrays.items()},contrasts=contrasts,training_epochs_query_weighted=training,
        independent_rank_and_bootstrap_recomputed=True,metric_cells=4*stage*5,all_query_denominators_retained=True,receiving_completion_sha256=sha(received/'receiving_completion.json'),panel_sha256=sha(received/'old/panel400.json'),
        fits={host:fits[host]['final_model'] for host in fits},training_seed_variability_assessed=False,independent_confirmation=False,SOTA_established=False,
        primary80='coupled-minus-visual DINO mean<=-0.02 and concept-bootstrap upper<0 stops; all other results expand400',positive400='coupled versus visual and marginal each delta>=0.01/lower>0 and within-concept10Top1 nonnegative; donor evidence separate',seconds=time.monotonic()-began)
    with (out/'summary.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],stage=stage,decision=result['decision'],means=result['means'],contrasts={k:v['DINO2AFC'] for k,v in contrasts.items()},output=str(out))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--received',required=True,type=Path);p.add_argument('--output',required=True,type=Path);v=p.parse_args();run(v.received,v.output)
