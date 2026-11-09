"""Fixed input interventions, separate from incorrect-training-target controls."""
BASES=('ordinary','joint_correct')
GENERATED_ARMS=('ordinary_within','ordinary_across','joint_within','joint_across')
ALL_ARMS=('ordinary','joint_correct')+GENERATED_ARMS
CONTRASTS={
    'joint_minus_ordinary':{'joint_correct':1.,'ordinary':-1.},
    'joint_correct_minus_within':{'joint_correct':1.,'joint_within':-1.},
    'joint_correct_minus_across':{'joint_correct':1.,'joint_across':-1.},
    'ordinary_correct_minus_within':{'ordinary':1.,'ordinary_within':-1.},
    'ordinary_correct_minus_across':{'ordinary':1.,'ordinary_across':-1.},
    'joint_minus_ordinary_within':{'joint_within':1.,'ordinary_within':-1.},
    'joint_minus_ordinary_across':{'joint_across':1.,'ordinary_across':-1.},
    'incremental_within_EEG_dependence':{'joint_correct':1.,'joint_within':-1.,'ordinary':-1.,'ordinary_within':1.},
    'incremental_across_EEG_dependence':{'joint_correct':1.,'joint_across':-1.,'ordinary':-1.,'ordinary_across':1.},
}
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc')
def mapping(queries):
    import numpy as np
    assert len(queries)==400 and len({q['query_id'] for q in queries})==400 and len({q['stimulus_id'] for q in queries})==400
    groups={}
    for i,q in enumerate(queries):groups.setdefault(q['concept_id'],[]).append(i)
    assert len(groups)==40 and all(len(g)==10 for g in groups.values())
    groups=[sorted(groups[c],key=lambda i:queries[i]['stimulus_id']) for c in sorted(groups)];within=np.empty(400,np.int64);across=within.copy()
    for concept,g in enumerate(groups):
        for position,i in enumerate(g):within[i]=g[(position+1)%10];across[i]=groups[(concept+1)%40][position]
    for a in (within,across):assert sorted(a.tolist())==list(range(400)) and (a!=np.arange(400)).all()
    assert all(queries[i]['concept_id']==queries[j]['concept_id'] for i,j in enumerate(within))
    assert all(queries[i]['concept_id']!=queries[j]['concept_id'] for i,j in enumerate(across))
    return dict(within=within.tolist(),across=across.tolist())
def source(arm,index,maps):
    assert arm in GENERATED_ARMS
    return ('ordinary' if arm.startswith('ordinary') else 'joint_correct'),maps[arm.rsplit('_',1)[1]][index]
def sampling_spec(arm,index,queries,maps):
    base,donor=source(arm,index,maps)
    return base,donor,queries[index]['image_seed']
def comparisons(metrics,queries):
    import numpy as np
    assert set(metrics)==set(ALL_ARMS) and all(set(v)==set(METRICS) for v in metrics.values())
    concepts=sorted({q['concept_id'] for q in queries});assert len(queries)==400 and len(concepts)==40
    groups=[[i for i,q in enumerate(queries) if q['concept_id']==c] for c in concepts];assert all(len(g)==10 for g in groups)
    draws=np.random.default_rng(1729).integers(0,40,(1000,40));result={};arrays={};max_error=0.
    for label,coefficients in CONTRASTS.items():
        delta=np.stack([sum(coefficient*np.asarray(metrics[arm][k],dtype=np.float64) for arm,coefficient in coefficients.items()) for k in METRICS],axis=1);assert delta.shape==(400,5) and np.isfinite(delta).all()
        arrays[label]=delta;clusters=np.stack([delta[g].mean(0) for g in groups]);boot=clusters[draws].mean(1);counts=np.array([np.bincount(d,minlength=40) for d in draws]);weighted=counts@clusters/40
        error=float(np.max(np.abs(boot-weighted)));assert error<2e-12;max_error=max(max_error,error);ci=np.quantile(boot,[.025,.975],axis=0)
        result[label]={k:dict(difference=float(delta[:,j].mean()),descriptive_concept_bootstrap95=ci[:,j].tolist()) for j,k in enumerate(METRICS)}
    def supported(label,metric):
        v=result[label][metric];return v['difference']>1e-12 and v['descriptive_concept_bootstrap95'][0]>1e-12
    interpretation=dict(uses_paired_EEG_for_coarse_reconstruction=supported('joint_correct_minus_across','DINO2AFC'),
        uses_paired_EEG_for_withinconcept_identity=supported('joint_correct_minus_within','within_concept10_two_afc'),
        incremental_withinconcept_EEG_dependence_over_ordinary=supported('incremental_within_EEG_dependence','within_concept10_two_afc'),
        numerical_zero_tolerance=1e-12,false_means_not_established_not_no_EEG_information=True,not_a_physiological_causal_claim=True,does_not_change_upstream_efficacy_or_secondary_decision=True,one_noise_seed_per_query=True,independent_H=False,SOTA=False)
    return result,arrays,interpretation,max_error
