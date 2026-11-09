"""D37 fixed image screen contracts, no model loading or score-based fitting."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
P=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
W=Path('/root/brainhive_reproduction_20261020_v1')
B=P/'rapid_iteration_20261027_v1/candidates/D37_temporal_conditional_attention'
ARMS=('conditional','flat_residual','same_concept_wrong')
BASE_ARMS=('parent','ordinary','risk')
IDENTITY=('query_id','subject_id','concept_id','stimulus_id','array_row','partition','image_seed')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def ref(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def read(path):return json.loads(Path(path).read_bytes())


def check(item):
    p=Path(item['path']);assert p.is_file() and sha(p)==item['sha256'],str(p)
    if 'bytes' in item:assert p.stat().st_size==item['bytes']
    return p


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def array(path,value):
    with Path(path).open('xb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())


def queries(panel):
    rows=panel['Dcheck'];assert len(rows)==400 and len({q['query_id'] for q in rows})==400
    assert all(q['subject_id']=='sub-01' and q['partition']=='Dcheck' for q in rows)
    groups={}
    for i,q in enumerate(rows):groups.setdefault(q['concept_id'],[]).append(i)
    assert len(groups)==40 and all(len(v)==10 for v in groups.values())
    assert len({q['concept_id'] for q in rows[:80]})==8
    assert all(len([q for q in rows[:80] if q['concept_id']==c])==10 for c in {q['concept_id'] for q in rows[:80]})
    return rows,groups


def control_map(groups):
    donor=np.empty(400,np.int64)
    for ids in groups.values():
        for i,j in zip(ids,ids[1:]+ids[:1]):donor[i]=j
    assert (donor!=np.arange(400)).all() and sorted(donor.tolist())==list(range(400))
    return donor


def stage_gate(spec,stage):
    if stage==80:return
    assert stage==400
    prior=read(B/'screen80/score/completion.json')
    assert prior['status']=='D37_ALL_THREE_FIXED80_IMAGE_ARMS_SCORED' and prior['spec']['sha256']==sha(spec)
    assert prior['primary_decision']['action']=='EXPAND400'
    audit=read(B/'screen80/score/independent_CPU_audit.json')
    assert audit['status']=='D37_ALL_FIXED_QUERY_RANKS_CONTROL_DIFFERENCES_AND_BOOTSTRAPS_INDEPENDENT_CPU_VERIFIED'
    assert audit['score_completion']['sha256']==sha(B/'screen80/score/completion.json') and audit['primary_decision']=='EXPAND400'


def byte_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepared(spec_path):
    spec=read(spec_path);assert spec['status']=='D37_FIXED_THREE_ARM_IMAGE_SCREEN_REGISTERED'
    for item in spec['source_pins']:check(item)
    path=B/'prepared_image_v1/completion.json';done=read(path)
    assert done['status']=='D37_ALL400_TEMPORAL_DIRECTIONS_AND_MATCHED_CONTROLS_LOCKED_BEFORE_IMAGES'
    assert done['spec']['sha256']==sha(spec_path)
    panel=read(check(spec['panel']));rows,groups=queries(panel)
    assert done['query_ids']==[q['query_id'] for q in rows] and done['arms']==list(ARMS)
    for item in done['files'].values():check(item)
    return spec,done,rows,groups
