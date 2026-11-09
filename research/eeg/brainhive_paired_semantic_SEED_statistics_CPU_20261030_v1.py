"""Full400 single-subject development arithmetic; never official R80 scores."""
import numpy as np
METRICS=('EffNet','CapSim','ObjectF1','SEED','PixCorr_RGB','SSIM_Y')

def statistics(values,queries,arms,*,draw_count=1000,seed=1729):
    assert len(queries)==400 and len({q['query_id'] for q in queries})==400
    concepts=sorted({q['concept_id'] for q in queries});assert len(concepts)==40
    groups=np.array([[i for i,q in enumerate(queries) if q['concept_id']==c] for c in concepts]);assert groups.shape==(40,10)
    assert all(a.shape==(400,6) and np.isfinite(a).all() for a in values.values())
    for a in values.values():assert np.max(np.abs(a[:,3]-a[:,:3].mean(1)))<2e-12
    draws=np.random.default_rng(seed).integers(0,40,(draw_count,40));contrasts={};maximum=0.
    for left in arms:
        contrasts[left]={}
        for right in values:
            if left==right:continue
            delta=values[left]-values[right];clusters=delta[groups].mean(1);sample=clusters[draws].mean(1)
            counts=np.array([np.bincount(d,minlength=40) for d in draws]);weighted=counts@clusters/40
            error=float(np.max(np.abs(sample-weighted)));maximum=max(maximum,error);assert error<2e-12
            ci=np.quantile(sample,[.025,.975],axis=0)
            contrasts[left][right]={key:dict(difference=float(delta[:,j].mean()),descriptive_concept_bootstrap95=ci[:,j].tolist(),positive_concepts=int((clusters[:,j]>0).sum()),concepts=40) for j,key in enumerate(METRICS)}
    return dict(means={a:{k:float(v[:,j].mean()) for j,k in enumerate(METRICS)} for a,v in values.items()},contrasts=contrasts,weighted_bootstrap_maximum_error=maximum,draws=draws)

def next_action(upstream_decision,contrasts):
    assert upstream_decision['action']=='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION'
    primary=contrasts['joint_correct']['ordinary'];seed=primary['SEED'];f1=primary['ObjectF1']
    support=seed['difference']>0 and seed['descriptive_concept_bootstrap95'][0]>0 and f1['difference']>=0
    return dict(action='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION' if support else 'HOLD_FOR_ANALYSIS_NO_AUTOMATIC_MULTISUBJECT_FIT',
        development_SEED_and_ObjectF1_support=support,upstream_DINO_and_all4_mechanism_gate_retained=True,
        CLIP_training_reward_NOT_independent_confirmation=True,low_level_metrics_reported_without_posthoc_gate=True,
        official_R80=False,independent_H=False,SOTA=False)
