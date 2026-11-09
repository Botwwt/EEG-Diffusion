"""Privileged visual endpoint diagnosis on the native ATM/h94 host, no fit."""
import json, sys
from pathlib import Path
import numpy as np

ARMS=('direct','T_mean','oracle_CLIPH','same_concept_CLIPH')
NEW_ARMS=ARMS[2:]
CONTRASTS=(('oracle_CLIPH','direct'),('oracle_CLIPH','same_concept_CLIPH'),
           ('same_concept_CLIPH','direct'),('oracle_CLIPH','T_mean'))
IDENTITY=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')
_context=None

def wrong_map(queries):
    groups={}
    for i,r in enumerate(queries):groups.setdefault(r['concept_id'],[]).append(i)
    assert len(queries)==400 and len(groups)==40 and all(len(g)==10 for g in groups.values())
    result=np.empty(400,np.int64)
    for g in groups.values():
        for i,j in zip(g,g[1:]+g[:1]):
            assert queries[i]['stimulus_id']!=queries[j]['stimulus_id']
            result[i]=j
    assert sorted(result.tolist())==list(range(400)) and np.all(result!=np.arange(400))
    return result

def setup(policy):
    global _context
    q=json.loads(Path(policy).read_bytes()) if not isinstance(policy,dict) else policy
    assert q['status']=='LOCKED_PRIVILEGED_NATIVE_ATM_CLIPH_ENDPOINT_DIAGNOSTIC'
    assert q['host']=='old' and q['seed']==1729 and q['maximum_output_bytes']==128<<20
    assert q['arms']==list(ARMS) and q['queries']==80 and q['gallery']==400
    assert q['new_training_updates']==0 and q['new_research_images']==160 and not q['automatic_fit_or_expansion']
    if _context is None:
        sys.path.insert(0,str(Path(q['base_policy']['path']).parent))
        import atm_error_geometry_contract_20261104_v1 as K
        bq=K.read(K.checked(q['base_policy']));parent,ctx,e,C=K.setup(bq)
        done=K.read(K.checked(q['base_completion']))
        assert done['status']=='COMPLETE_NATIVE_ATM_EEG_ERROR_GEOMETRY_PRIORS_PAIRED_IMAGES' and done['stage400_completed']
        for h in q['closed_controllers']:assert not K.process(h['PID'],h.get('start_ticks'))['live']
        for pin in q['sources'].values():K.checked(pin)
        for key in ('teacher400','base_prepared','base_ordered_records','base_features','source_metrics','source_low_level'):K.checked(q[key])
        _context=dict(q=q,K=K,bq=bq,parent=parent,ctx=ctx,e=e,C=C,
                      repo=Path(ctx['base']['ctx']['relative']['repository_root']),out=Path(q['output_root']))
    assert _context['q']==q
    return _context

def resources(c,starting=False):
    K=c['K'];q=c['q'];r=K.resources(c['bq'],c['parent'],c['ctx'],c['e'],c['C'],starting=starting)
    used=c['ctx']['base']['ctx']['C'].allocated(q['project_root']) if Path(q['project_root']).exists() else 0
    assert used<=q['maximum_output_bytes']
    required=r['required_including_this_project']+q['maximum_output_bytes']-used
    assert r['free_bytes']>=required
    return dict(inherited=r,required_remaining_bytes=r['required_including_this_project'],
                required_including_this_project=required,free_bytes=r['free_bytes'],own_used_bytes=used,
                own_remaining_bytes=q['maximum_output_bytes']-used,all_existing_forecasts_retained=True)

def records(c):
    K=c['K'];q=c['q'];old=K.read(K.checked(q['base_ordered_records']));result={}
    for arm in ARMS[:2]:
        result[arm]=[r for r in old if r['arm']==arm][:80]
        assert len(result[arm])==80
    for arm in NEW_ARMS:
        path=c['out']/'images'/arm/'images.jsonl'
        result[arm]=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    return result

def paired(arrays,queries):
    groups=[[i for i,r in enumerate(queries) if r['concept_id']==x] for x in sorted({r['concept_id'] for r in queries})]
    assert len(groups)==8 and all(len(g)==10 for g in groups)
    draw=np.random.default_rng(1729).integers(0,8,(1000,8));result={}
    for a,b in CONTRASTS:
        result[a+'-'+b]={}
        for key in arrays[a]:
            delta=arrays[a][key]-arrays[b][key];means=np.array([delta[g].mean() for g in groups])
            result[a+'-'+b][key]=dict(difference=float(delta.mean()),descriptive_concept_bootstrap95=np.quantile(means[draw].mean(1),[.025,.975]).tolist(),positive_concepts=int((means>0).sum()),concepts=8)
    return result
