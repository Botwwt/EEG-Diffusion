"""Fixed single-subject error-geometry prior study; inherited assets remain sealed."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ARMS=('parent','ordinary','direct','direct_ordinary','native_v','iso_error','paired_error','wrong_error','T_mean')
BASELINES=ARMS[:4];NEW_ARMS=ARMS[4:];MODELS=NEW_ARMS[:4]
METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
CONTRASTS=(('paired_error','direct'),('paired_error','native_v'),('paired_error','iso_error'),('paired_error','wrong_error'),('paired_error','T_mean'),('paired_error','ordinary'),('native_v','direct'),('iso_error','direct'),('wrong_error','direct'),('T_mean','direct'))
_context=None
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def checked(pin):
    actual=ref(pin['path']);assert all(actual[k]==v for k,v in pin.items()),dict(expected=pin,actual=actual);return Path(pin['path'])
def write(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.pending');assert not tmp.exists();tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save_array(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.pending');assert not tmp.exists()
    with tmp.open('xb') as f:np.save(f,value,allow_pickle=False)
    tmp.replace(p)
def process(pid,ticks=None):
    try:s=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return dict(PID=pid,state='MISSING',live=False)
    return dict(PID=pid,start_ticks=int(s[19]),state=s[0],live=(ticks is None or int(s[19])==ticks) and s[0] not in ('Z','X'))
def setup(q):
    global _context
    assert q['status']=='LOCKED_NATIVE_ATM_PAIRED_EEG_ERROR_GEOMETRY_PRIOR_STUDY'
    assert q['host']=='old' and q['seed']==1729 and q['maximum_output_bytes']==2304<<20
    assert q['epochs']==150 and q['batch']==1024 and q['arms']==list(ARMS) and q['models']==list(MODELS)
    for p in q['sources'].values():checked(p)
    if _context is None:
        sys.path.insert(0,str(checked(q['factorial_policy']).parent))
        import atm_prior_factorial_contract_20261104_v1 as F
        oldq=F.read(q['factorial_policy']['path']);oldparent,ctx,e,C=F.setup(oldq)
        done=F.read(checked(q['factorial_completion']));assert done['stage400_completed'] and done['status']=='COMPLETE_NATIVE_PRIOR_ON_OFF_ORDINARY_ON_OFF_PAIRED_IMAGES'
        for h in q['closed_controllers']:assert not process(h['PID'],h.get('start_ticks'))['live']
        parent=dict(oldparent);parent['maximum_output_bytes']=q['maximum_output_bytes']
        _context=(parent,dict(F=F,oldq=oldq,oldparent=oldparent,oldctx=ctx,old_done=done,c=ctx['c'],base=ctx['c']['base'],e=e,C=C),e,C)
    return _context
def resources(q,parent,ctx,e,C,starting=False):
    r=ctx['F'].resources(ctx['oldq'],ctx['oldparent'],ctx['oldctx'],e,C,starting=starting)
    used=ctx['base']['ctx']['C'].allocated(q['project_root']);assert used<=q['maximum_output_bytes']
    inherited=r['required_including_this_project'];required=inherited+q['maximum_output_bytes']-used;assert r['free_bytes']>=required
    return dict(required_remaining_bytes=inherited,required_including_this_project=required,free_bytes=r['free_bytes'],own_used_bytes=used,
        own_remaining_bytes=q['maximum_output_bytes']-used,inherited=r,legacy50GiB_and8GiB_buffer_unchanged=True)
def baseline_records(q):
    _,c,_,_=setup(q);out=Path(c['oldq']['output_root']);rows=read(checked(read(out/'images400_audit.json')['ordered_records']));assert len(rows)==1600
    return {arm:[v for v in rows if v['arm']==arm] for arm in BASELINES}
def paired(arrays,queries):
    groups=[[i for i,v in enumerate(queries) if v['concept_id']==c] for c in sorted({v['concept_id'] for v in queries})]
    assert len(groups) in (8,40) and all(len(g)==10 for g in groups);draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)))
    result={}
    for a,b in CONTRASTS:
        row={}
        for key in METRICS:
            d=arrays[a][key]-arrays[b][key];cm=np.array([d[g].mean() for g in groups]);ci=np.quantile(cm[draw].mean(1),[.025,.975])
            row[key]=dict(difference=float(d.mean()),descriptive_concept_bootstrap95=ci.tolist(),positive_concepts=int((cm>0).sum()),concepts=len(groups))
        result[a+'-'+b]=row
    return result
def interpret(comparisons,stage):
    harm={}
    for arm in MODELS:
        x=comparisons[arm+'-direct'];d=x['DINO2AFC'];w=x['within_concept10_two_afc']
        harm[arm]=d['difference']<=-.01 and d['descriptive_concept_bootstrap95'][1]<0 and w['difference']<0 and w['descriptive_concept_bootstrap95'][1]<0
    joint=all(harm.values())
    return dict(action=('STOP_ALL_NEW_ARMS_JOINT_CLEAR_HARM' if joint else 'EXPAND_ALL_NEW_ARMS_TO400') if stage==80 else 'COMPLETE_FIXED_ERROR_GEOMETRY_DEVELOPMENT_COMPARISON',
        all_four_priors_joint_clear_harm=joint,individual_joint_harm=harm,all8_metrics_retained=True,
        no_single_metric_automatic_family_rejection=True,development_only=True,nominal_optional_stopping_error_control=False,
        v_and_gaussian_preconditioning_are_existing_methods=True,new_independent_second_method_established=False,SOTA_established=False)
