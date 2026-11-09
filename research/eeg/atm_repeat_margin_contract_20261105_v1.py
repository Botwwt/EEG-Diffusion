import sys,json
from pathlib import Path
import atm_repeat_margin_core_20261105_v1 as F

def setup(path):
    q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_FROZEN_FULL_T_REPEAT_MARGIN_DIAGNOSIS'
    assert q['host']=='old' and q['states']==list(F.STATES) and q['seed']==1729 and q['new_updates']==0 and q['new_images']==0
    assert q['diagnostic_cap_bytes']==64<<20
    sys.path.insert(0,str(Path(q['training_policy']['path']).parent))
    import atm_repeat_count_training_contract_20261105_v1 as T
    c=T.setup(q['training_policy']['path']);K=c['K'];K.checked(q['training_policy'])
    assert q['project_root']==c['q']['project_root'] and q['maximum_output_bytes']==c['q']['maximum_output_bytes']==1200<<20
    for pin in q['sources'].values():K.checked(pin)
    for pin in q['checkpoints'].values():K.checked(pin)
    for pin in q['closed_completions']:K.checked(pin)
    for pin in q['closed_launches']:
        a=K.read(K.checked(pin));assert not K.process(a['PID'],a['start_ticks'])['live']
    return dict(q=q,K=K,out=Path(q['output_root']),physical=c['physical'],T=c['T'],prior=c,training_contract=T)

def resources(c,starting=False):
    r=c['training_contract'].resources(c['prior'],starting);K,q=c['K'],c['q']
    used=sum(K.allocated(q[x]) for x in ('code_root','output_root') if Path(q[x]).exists())
    assert used<=q['diagnostic_cap_bytes']
    assert r['own_remaining_bytes']>=q['diagnostic_cap_bytes']-used
    return dict(**r,diagnostic_used_bytes=used,diagnostic_cap_bytes=q['diagnostic_cap_bytes'],all_old_forecasts_still_charged=True)
