"""Named finite single-seed relative-evidence mechanism controls."""
import hashlib,json,os
from pathlib import Path
ARMS=('conditional_same','conditional_different','conditional_same_mismatched_teacher','continued_F_cosine')
STATUS='REGISTERED_SINGLE1729_RELATIVE_EVIDENCE_CONTINUATION_AND_FIXED_IMAGES_v1'
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):
    actual=ref(x['path']);assert actual['sha256']==x['sha256']
    if 'bytes' in x:assert actual['bytes']==x['bytes']
    return Path(x['path'])
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending')
    with tmp.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def spec(path):
    s=read(path);assert s['status']==STATUS and s['host']=='old' and s['training_seed']==1729
    assert s['arms']==list(ARMS) and not s['multiple_seeds_allowed'] and s['R80_used_for_direction_selection']
    for item in s['source_pins']+s['input_pins']:checked(item)
    return s
def image_decision(contrasts,stage):
    # Controls are always reported together; only the predeclared primary can
    # trigger expansion. Never promote a control by selecting the best score.
    main=contrasts['conditional_same'];o=main['ordinary']['DINO2AFC']
    mean=o['difference'];lo,hi=o['descriptive_concept_bootstrap95']
    if stage==80:
        return dict(action='STOP_INSTANCE' if mean<=-.02 and hi<0 else 'EXPAND400',reason='CLEAR_PRIMARY_HARM' if mean<=-.02 and hi<0 else 'EXPAND_ALL_FOUR_ARMS_UNLESS_CLEAR_PRIMARY_HARM')
    assert stage==400
    strong=('ordinary','starting_F','continued_F_cosine')
    support={b:bool(main[b]['DINO2AFC']['difference']>=.01 and main[b]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and main[b]['target_cosine']['difference']>0 and main[b]['within_concept10_top1']['difference']>=0) for b in strong}
    return dict(action='PROMISING_FOR_FULL_EVALUATION' if all(support.values()) else 'STOP_INSTANCE',strong_baseline_support=support,not_SOTA=True)
