"""Prior ON/OFF x fixed ordinary ON/OFF, with every old resource promise kept."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ARMS=('parent','ordinary','direct','direct_ordinary')
BASELINES=('parent','ordinary')
NEW_ARMS=('direct','direct_ordinary')
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
CONTRASTS=(('direct','parent'),('direct_ordinary','ordinary'),('direct_ordinary','direct'),('ordinary','parent'))
_context=None
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(pin):
    actual=ref(pin['path']);assert all(actual[k]==v for k,v in pin.items()),dict(expected=pin,actual=actual)
    return Path(pin['path'])
def write(path,value):
    p=Path(path);temp=p.with_name(p.name+'.pending');assert not temp.exists()
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def save_array(path,value):
    p=Path(path);temp=p.with_name(p.name+'.pending');assert not temp.exists()
    with temp.open('xb') as f:np.save(f,value,allow_pickle=False)
    temp.replace(p)
def process(pid,ticks=None):
    try:s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return dict(PID=pid,state='MISSING',live=False)
    return dict(PID=pid,start_ticks=int(s[19]),state=s[0],live=(ticks is None or int(s[19])==ticks) and s[0] not in ('Z','X'))
def setup(q):
    global _context
    assert q['status']=='LOCKED_NATIVE_PRIOR_BYPASS_FACTORIAL_PAIRED_IMAGE_DIAGNOSTIC'
    assert q['host']=='old' and q['seed']==1729 and q['maximum_output_bytes']==512<<20
    assert q['arms']==list(ARMS) and q['new_training_updates']==0 and q['strength']==1 and q['radius']==.25
    for pin in q['sources'].values():checked(pin)
    if _context is None:
        sys.path.insert(0,str(checked(q['exposure_policy']).parent))
        import atm_condition_exposure_contract_20261104_v1 as X
        c=X.setup(q['exposure_policy']['path'])
        assert read(checked(q['exposure_completion']))['status']=='COMPLETE_FROZEN_CONDITION_EXPOSURE_ROLE_ALL_T'
        for h in q['closed_controllers']:assert not process(h['PID'],h['start_ticks'])['live']
        old=read(checked(q['postfit_policy']));done=read(checked(q['postfit_completion']))
        assert done['status']=='COMPLETE_NATIVE_ATM_FOUR_ARM_PAIRED_IMAGE_LIFECYCLE' and done['stage400_completed']
        prepared=read(checked(done['prepared']));assert prepared['policy']==q['postfit_policy']
        checked(q['ordinary_state']);assert q['ordinary_state']==prepared['heads']['ordinary']['state']
        for pin in old['sources'].values():checked(pin)
        sys.path.insert(0,str(Path(old['output_root'])))
        # Reuse exactly one already-audited native host runtime initialization.
        parent=dict(c['base']['parent']);parent['maximum_output_bytes']=q['maximum_output_bytes']
        _context=(parent,dict(X=X,c=c,old=old,old_done=done,old_prepared=prepared),None,None)
    return _context
def resources(q,parent,train,e,C,starting=False):
    inherited=train['X'].resources(train['c'],starting=starting)
    used=train['c']['base']['ctx']['C'].allocated(q['project_root']);assert used<=q['maximum_output_bytes']
    remaining=q['maximum_output_bytes']-used;required=inherited['required_remaining_bytes']+remaining
    assert inherited['free_bytes']>=required
    # Compatibility with the existing generator's fast guard, which itself adds
    # the current project's remaining allowance to required_remaining_bytes.
    return dict(required_remaining_bytes=inherited['required_remaining_bytes'],
        required_including_this_project=required,free_bytes=inherited['free_bytes'],own_used_bytes=used,
        own_remaining_bytes=remaining,inherited=inherited,legacy50GiB_and8GiB_buffer_unchanged=True)
def baseline_records(q):
    _,ctx,_,_=setup(q);oldout=Path(ctx['old']['output_root'])
    audit=read(oldout/'images400_audit.json');rows=read(checked(audit['ordered_records']))
    assert len(rows)==1600
    return {arm:[v for v in rows if v['arm']==arm] for arm in BASELINES}
def paired(arrays,queries):
    groups=[[i for i,v in enumerate(queries) if v['concept_id']==c] for c in sorted({v['concept_id'] for v in queries})]
    assert len(groups) in (8,40) and all(len(g)==10 for g in groups)
    draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)))
    def stats(delta):
        values={}
        for key,d in delta.items():
            cm=np.array([d[g].mean() for g in groups]);ci=np.quantile(cm[draw].mean(1),[.025,.975])
            values[key]=dict(difference=float(d.mean()),descriptive_concept_bootstrap95=ci.tolist(),positive_concepts=int((cm>0).sum()),concepts=len(groups))
        return values
    answer={a+'-'+b:stats({key:np.asarray(arrays[a][key])-arrays[b][key] for key in arrays[a]}) for a,b in CONTRASTS}
    answer['interaction_ordinary_without_prior_minus_with_prior']=stats({key:(arrays['direct_ordinary'][key]-arrays['direct'][key])-(arrays['ordinary'][key]-arrays['parent'][key]) for key in arrays['parent']})
    return answer
def interpret(comparisons,stage):
    assert stage in (80,400);harm={}
    for arm,baseline in (('direct','parent'),('direct_ordinary','ordinary')):
        pair=comparisons[arm+'-'+baseline];d=pair['DINO2AFC'];w=pair['within_concept10_two_afc']
        harm[arm]=d['difference']<=-.01 and d['descriptive_concept_bootstrap95'][1]<0 and w['difference']<0 and w['descriptive_concept_bootstrap95'][1]<0
    joint=all(harm.values())
    return dict(action=('STOP_BOTH_BYPASS_CONFIGS_JOINT_CLEAR_HARM' if joint else 'EXPAND_BOTH_BYPASS_CONFIGS_TO400') if stage==80 else 'COMPLETE_DEVELOPMENT_FACTORIAL',
        both_bypass_configurations_joint_clear_harm=joint,individual_joint_harm=harm,all8_metrics_retained=True,
        single_metric_failure_does_not_reject_method_family=True,development_only=True,nominal_optional_stopping_error_control=False,
        bypass_is_structural_baseline_not_second_innovation=True,ordinary_frozen_under_changed_input_distribution=True,
        no_claim_of_equivalence_or_SOTA=True,second_independent_method_established=False,SOTA_established=False)
