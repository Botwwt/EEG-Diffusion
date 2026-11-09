"""One continuous constrained EEG fit and its same-loss unconstrained control."""
import json,os,sys
from pathlib import Path
import numpy as np
ARMS=('frozen_positive_NCE','C_trunk_N_head','margin_constrained','response_only')
CONTRASTS=tuple((ARMS[j],ARMS[i]) for i in range(4) for j in range(i+1,4))
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
R4_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
R80_METRICS=('target_cosine','DINO2AFC','gallery_top1','gallery_top5','PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')+PAPER

def setup(path):
    q=json.loads(Path(path).read_bytes())
    assert q['status']=='LOCKED_CONTINUOUS_MARGIN_FULL_T10_R4_R80' and q['host']=='old' and q['seed']==1729
    assert q['arms']==list(ARMS) and q['epochs']==10 and q['attempts_per_arm']==830
    assert q['maximum_output_bytes']==576<<20 and q['wrapper_reserve_bytes']==2<<20
    sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
    import atm_response_repair_contract_20261105_v196 as B
    base=B.setup(q['prior_policy']['path']);K=base['K'];K.checked(q['prior_policy'])
    for pin in list(q['sources'].values())+list(q['runtime_sources'].values()):K.checked(pin)
    profile=K.read(K.checked(q['profile_completion']))
    assert profile['status']=='PASS_CONTINUOUS_ADAM_RESTART_ROLLBACK_AND_NATIVE_RAW_REPLAY'
    assert profile['first']['accepted'] and profile['second']['accepted']
    assert profile['second_step_resumed_model_Adam_RNG_event_and_allT_byteexact'] and profile['all_original_weights_buffers_fullT_restored']
    pp=K.read(K.checked(q['profile_policy']))
    for key in ['core','checkpoint','schedule']:assert pp['sources'][key]['sha256']==q['sources'][key]['sha256']
    for name,pin in q['models'].items():assert pin==base['q']['models'][name];K.checked(pin)
    out=Path(q['output_root']);fit=Path(q['project_root'])/'fit';values=None
    if (fit/'conditions_completion.json').exists():
        done=K.read(fit/'conditions_completion.json');assert done['status']=='COMPLETE_CONTINUOUS_MARGIN_EEG_ONLY1200_CONDITIONS'
        val=np.load(K.checked(done['conditions']));assert val.shape==(600,2,1024) and np.isfinite(val).all()
        values=np.concatenate([base['values'][:,:2],val],axis=1)
    c=dict(base);c.update(q=q,K=K,out=out,fit=fit,values=values,previous_study=base)
    return c

def resources(c,starting=False):
    q,K=c['q'],c['K'];ledger=q['resource_ledger'];previous=c['previous_study']['q']
    groups=ledger['new_closed_unused_settlements'];assert len(groups)==3
    expected=[([previous['project_root']],768<<20),([previous['resource_ledger']['frozen_parent_root']],384<<20),
      ([str(Path(previous['project_root']).parent/n) for n in ['VAR_full_T_native_bank_code_20261102_v1','VAR_full_T_native_bank_20261102_v1']],1400<<20)]
    credit=0
    for s,(roots,cap) in zip(groups,expected):
        assert s['roots']==roots and s['cap']==cap
        actual=sum(K.allocated(p) for p in roots);assert actual==s['actual_allocated_bytes']
        assert s['unused_future_bytes']==cap-actual and s['unused_future_bytes']>=0
        credit+=s['unused_future_bytes']
        for pin in s['terminal_proofs']:K.checked(pin)
    assert ledger['parent_quote']['own_used_bytes']==groups[0]['actual_allocated_bytes']
    # Preserve the already-credited six VAR roots and their actual allocations.
    old=previous['resource_ledger']['one_new_settlement']
    assert sum(K.allocated(p) for p in old['roots'])==old['actual_allocated_bytes']
    assert not any(p in old['roots'] for s in groups for p in s['roots'])
    used=K.allocated(q['project_root']);assert used<=q['maximum_output_bytes']
    required=ledger['parent_quote']['required_including_this_project']-credit+q['maximum_output_bytes']+q['wrapper_reserve_bytes']-used
    assert required>=58<<30
    fs=os.statvfs(q['project_root']);free=fs.f_bavail*fs.f_frsize;assert free>=required,(free,required)
    mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
    head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cu=Path('/sys/fs/cgroup/memory.current')
    if mx.exists() and cu.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cu.read_text()))
    assert head>=(24<<30 if starting else 2<<30)
    return dict(required_including_this_project=required,free_bytes=free,own_used_bytes=used,
                own_remaining_bytes=q['maximum_output_bytes']-used,CPU_headroom_bytes=head,
                newly_closed_unused_bytes_once=credit,all_other_promises_50GiB_floor_8GiB_buffer_held=True,no_deletion=True)

def records(c,domain):
    n=400 if domain=='R4' else 200;result={}
    for arm in ARMS:
        result[arm]=c['reference_records'][domain][arm] if arm in ARMS[:2] else [json.loads(v) for v in (c['out']/'images'/domain/arm/'images.jsonl').read_text().splitlines()]
        assert [r['query_index'] for r in result[arm]]==list(range(n))
    return result

def paired(arrays,queries,domain):
    groups=[[i for i,r in enumerate(queries) if r['concept_id']==g] for g in sorted({r['concept_id'] for r in queries})]
    assert len(groups)==(40 if domain=='R4' else 200)
    draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));result={};diffs={}
    for a,b in CONTRASTS:
        name=a+'-'+b;result[name]={}
        for key in R4_METRICS if domain=='R4' else R80_METRICS:
            delta=arrays[a][key]-arrays[b][key];valid=np.isfinite(delta);gm=np.array([delta[g].mean() for g in groups])
            result[name][key]=dict(difference=float(delta.mean()) if valid.all() else None,descriptive_concept_bootstrap95=np.quantile(gm[draw].mean(1),[.025,.975]).tolist() if valid.all() else None,defined_queries=int(valid.sum()),total_queries=len(queries),concepts=len(groups))
            diffs[name+'__'+key]=delta
    return result,diffs
