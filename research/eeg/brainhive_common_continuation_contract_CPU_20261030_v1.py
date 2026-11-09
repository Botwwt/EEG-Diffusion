"""Ordinary-initialized same-head single-seed scientific lifecycle contract."""
import hashlib,json,os
from pathlib import Path
ARMS=('joint_same','relative_only','AP_only','joint_mismatched_teacher')
STATUS='REGISTERED_SINGLE1729_ORDINARY_COMMON_RELATIVE_AP_FULL_LIFECYCLE_v1'
FRACTIONS=(0.,.25,.5,.75,1.)
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):
    a=ref(x['path']);assert a['sha256']==x['sha256']
    if 'bytes' in x:assert a['bytes']==x['bytes']
    return Path(x['path'])
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending')
    with tmp.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def spec(p):
    s=read(p);assert s['status']==STATUS and s['host']=='old' and s['training_seed']==1729 and s['arms']==list(ARMS)
    assert s['R80_used_for_direction_selection'] and not s['multiple_seeds_allowed']
    for pin in s['source_pins']+s['input_pins']:checked(pin)
    return s
def image_decision(contrasts,stage):
    main=contrasts['joint_same'];v=main['ordinary']['DINO2AFC'];mean=v['difference'];lo,hi=v['descriptive_concept_bootstrap95']
    if stage==80:
        stop=mean<=-.02 and hi<0
        return dict(action='STOP_INSTANCE' if stop else 'EXPAND400',reason='CLEAR_PRIMARY_HARM' if stop else 'EXPAND_ALL_FOUR_UNLESS_CLEAR_PRIMARY_HARM')
    assert stage==400
    strong=('ordinary','relative_only','AP_only','joint_mismatched_teacher')
    support={a:bool(main[a]['DINO2AFC']['difference']>=.01 and main[a]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and main[a]['target_cosine']['difference']>0 and main[a]['within_concept10_top1']['difference']>=0) for a in strong}
    return dict(action='PROMISING_FOR_FULL_EVALUATION' if all(support.values()) else 'STOP_INSTANCE',strong_baseline_support=support,not_SOTA=True)
