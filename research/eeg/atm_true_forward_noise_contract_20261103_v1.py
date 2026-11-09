"""Frozen true-forward EEG evidence diagnostic, with no calibration or fitting."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

METHODS=('missing','correct','sameconcept_wrong','native_CFG5')
METRICS=('epsilon_MSE','raw_clean_MSE','clipped_clean_MSE','clipped_target_cosine',
         'clipped_within10_top1','clipped_within10_two_afc','raw_clean_norm','clipped_clean_norm')
TIMES=list(range(980,-1,-20))

def read(p):return json.loads(Path(p).read_bytes())
def setup(path):
    s=read(path)
    assert s['status']=='REGISTERED_FROZEN_TRUE_FORWARD_NATIVE_PRIOR_EEG_EVIDENCE_ALL_T'
    assert s['host']=='old' and s['seed']==1729 and s['queries']==13230 and s['concepts']==1323
    assert s['timesteps']==TIMES and s['methods']==list(METHODS) and s['metrics']==list(METRICS)
    assert s['maximum_output_bytes']==256<<20 and s['new_training_updates']==s['new_images']==0
    sys.path.insert(0,str(Path(s['base_policy']['path']).parent))
    import brainhive_relative_fixed_output_contract_20261103_v1 as K
    ctx=K.setup(str(K.checked(s['base_policy'])))
    for pin in s['sources'].values():K.checked(pin)
    for v in s['closed_controllers']:assert not K.process(v['PID'],v['start_ticks'])['live']
    parent=read(K.checked(s['parent_policy']));done=read(K.checked(s['parent_completion']))
    assert done['policy']==s['parent_policy'] and done['OOF_T_queries']==13230
    prior_done=read(K.checked(s['unclipped_completion']))
    assert prior_done['status']=='COMPLETE_ONE_NATIVE_PRIOR_T_ONLY_CONFIG_DIAGNOSTIC'
    legacy=read(K.checked(parent['legacy_inputs']));sub=legacy['subjects']['sub-01']
    split=read(K.checked(legacy['split']));T=sorted(split['image_rows']['T'])
    banks=[v for v in done['banks'] if v['role']!='full']
    assert len(banks)==2 and sorted(banks[0]['rows']+banks[1]['rows'])==T
    for b in banks:
        for key in ('prior_final','permissions'):K.checked(b[key])
        p=read(b['permissions']['path']);K.checked(sub['models'][b['role']]['checkpoint'])
        assert p['export_rows']==b['rows']==sorted(sub['models'][b['role']]['export_array_rows'])
        assert p['export_is_T_OOF'] and not {i//10 for i in p['training_rows']}&{i//10 for i in p['export_rows']}
    K.checked(sub['input']);K.checked(parent['original_visual_target']);K.checked(parent['native_prior_source'])
    return dict(s=s,K=K,ctx=ctx,parent=parent,parent_done=done,sub=sub,T=T,banks=banks)

def resources(c,starting=False):
    s,K,ctx=c['s'],c['K'],c['ctx'];r=K.resources(ctx,starting=starting)
    extra=[];required=r['required_remaining_bytes']
    for g in s['additional_future_groups']:
        used=sum(ctx['C'].allocated(p) for p in g['roots']);assert used<=g['cap']
        required+=g['cap']-used;extra.append(dict(name=g['name'],used=used,remaining=g['cap']-used))
    used=ctx['C'].allocated(s['code_root'])+ctx['C'].allocated(s['output_root'])
    assert used<=s['maximum_output_bytes'];required+=s['maximum_output_bytes']-used
    assert r['free_bytes']>=required
    return dict(inherited=r,other_groups=extra,own_used_bytes=used,required_remaining_bytes=required,free_bytes=r['free_bytes'])

def seed(row,t):
    return int(hashlib.sha256(f'1729|true_forward_EEG_noise|{row}|{t}'.encode()).hexdigest()[:15],16)

def summarize(raw,T):
    """Uniform native-time average, then equal concept bootstrap; no time selection."""
    assert raw.shape==(13230,50,4,8)
    concepts=list(dict.fromkeys(i//10 for i in T))
    groups=[[j for j,row in enumerate(T) if row//10==concept] for concept in concepts]
    assert len(groups)==1323 and all(len(g)==10 for g in groups)
    cm=np.stack([np.asarray(raw[g],np.float64).mean(0) for g in groups])
    qm=np.asarray(raw,np.float64).mean(1)
    means=cm.mean((0,1));time_means=cm.mean(0)
    draws=np.random.default_rng(1729).integers(0,1323,(1000,1323));comparisons={}
    for left,right in ((1,0),(1,2),(3,1)):
        name=METHODS[left]+'-'+METHODS[right]
        d=cm[:,:,left]-cm[:,:,right];value=d.mean(1);ci=np.quantile(value[draws].mean(1),[.025,.975],axis=0)
        comparisons[name]={key:dict(difference=float(value[:,k].mean()),descriptive_concept_bootstrap95=ci[:,k].tolist(),
            positive_concepts=int((value[:,k]>0).sum()),concepts=1323) for k,key in enumerate(METRICS)}
    return cm,qm,means,time_means,comparisons
