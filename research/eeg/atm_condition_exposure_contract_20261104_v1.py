"""Frozen exposure/encoder-role diagnostic; never an EEG inference candidate."""
import json,sys
from pathlib import Path
import numpy as np

QUALITY=('raw_target_MSE','target_cosine','within10_top1','within10_two_afc','target_minus_wrong_mean','condition_norm')
VIEWS=('encoder_seen_concept','encoder_unseen_concept')

def setup(path):
    s=json.loads(Path(path).read_bytes())
    assert s['status']=='REGISTERED_FROZEN_ENCODER_EXPOSURE_ROLE_AND_NATIVE_PRIOR_ALL_T'
    assert s['host']=='old' and s['seed']==1729 and s['queries']==13230
    assert s['maximum_output_bytes']==256<<20 and s['new_images']==s['new_training_updates']==0
    sys.path.insert(0,str(Path(s['true_forward_policy']['path']).parent))
    import atm_true_forward_noise_contract_20261103_v1 as F
    c=F.setup(s['true_forward_policy']['path'])
    K=c['K'];K.checked(s['true_forward_policy']);K.checked(s['true_forward_completion'])
    assert K.read(s['true_forward_completion']['path'])['status']=='COMPLETE_FROZEN_ALL_T_TRUE_FORWARD_NATIVE_EEG_SCORE_DIAGNOSTIC'
    for pin in s['sources'].values():K.checked(pin)
    for h in s['closed_controllers']:assert not K.process(h['PID'],h['start_ticks'])['live']
    assert s['views']==list(VIEWS) and s['quality_metrics']==list(QUALITY)
    return dict(s=s,F=F,base=c,K=K)

def resources(c,starting=False):
    s,F,base,K=(c[x] for x in ('s','F','base','K'))
    inherited=F.resources(base,starting=starting)
    used=base['ctx']['C'].allocated(s['code_root'])+base['ctx']['C'].allocated(s['output_root'])
    assert used<=s['maximum_output_bytes']
    required=inherited['required_remaining_bytes']+s['maximum_output_bytes']-used
    assert inherited['free_bytes']>=required
    return dict(inherited=inherited,own_used_bytes=used,required_remaining_bytes=required,free_bytes=inherited['free_bytes'])

def summaries(quality,new_score,baseline,T):
    assert quality.shape==(13230,2,6) and new_score.shape==(13230,50,8) and baseline.shape==(13230,50,4,8)
    concepts=sorted(set(r//10 for r in T));groups=[[i for i,r in enumerate(T) if r//10==g] for g in concepts]
    assert len(groups)==1323 and all(len(v)==10 for v in groups)
    qc=np.stack([np.asarray(quality[v],np.float64).mean(0) for v in groups])
    sc=np.stack([np.asarray(new_score[v],np.float64).mean(0) for v in groups])
    bc=np.stack([np.asarray(baseline[v],np.float64).mean(0) for v in groups])
    draws=np.random.default_rng(1729).integers(0,1323,(1000,1323))
    def report(values,names):
        return {name:dict(difference=float(values[:,j].mean()),descriptive_concept_bootstrap95=np.quantile(values[draws,j].mean(1),[.025,.975]).tolist(),positive_concepts=int((values[:,j]>0).sum()),concepts=1323) for j,name in enumerate(names)}
    comparisons={'quality_seen_minus_unseen':report(qc[:,0]-qc[:,1],QUALITY)}
    from atm_true_forward_noise_contract_20261103_v1 import METRICS
    for idx,name in ((1,'prior_seen_encoder_minus_native_unseen_encoder'),(0,'prior_seen_encoder_minus_missing')):
        comparisons[name]=report((sc-bc[:,:,idx]).mean(1),METRICS)
    return qc,sc,bc,comparisons
