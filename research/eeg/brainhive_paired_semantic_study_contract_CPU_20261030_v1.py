"""Immutable checkpoints, full pair coverage, and fixed image decisions."""
import hashlib,json,math,os,tempfile
from pathlib import Path

ARMS=('joint_correct','EEG_only_correct','P_only_correct','joint_swapped_targets','joint_absolute_only')
IMAGE_ARMS=('parent','ordinary')+ARMS

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(x):
    actual=ref(x['path']);assert set(x)>={'path','sha256'}
    assert all(actual[k]==v for k,v in x.items());return Path(x['path'])
def raw_json(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
def atomic(p,v,*,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert replace or not p.exists()
    fd,name=tempfile.mkstemp(prefix=p.name+'.pending.',dir=p.parent)
    with os.fdopen(fd,'wb') as f:f.write(raw_json(v));f.flush();os.fsync(f.fileno())
    if replace:os.replace(name,p)
    else:assert not p.exists();Path(name).rename(p)
def sync_dir(p):
    fd=os.open(p,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def validate_rows(rows,order,start,end,pairs):
    assert len(rows)==end-start and end<=len(order)
    for update,row in enumerate(rows,start+1):
        i=int(order[update-1]);pair=pairs[i]
        assert row['update']==update and row['pair_index']==i and row['array_rows']==pair['array_rows']
        assert row['T_indices']==pair['T_indices'] and row['primary_EEG_exposures']==2*update
        assert len(row['proof']['steps'])==4 and len(row['proof']['ancestral'])==4 and row['proof']['UNet_NFE']==4
        assert [x['timestep'] for x in row['proof']['steps']]==[999.,749.,499.,249.]
        assert all(math.isfinite(row[k]) for k in ('loss','absolute','pairing','effect'))
        assert all(math.isfinite(v['gradient_norm_before_clip']) and v['gradient_norm_before_clip']>0 for v in row['gradients'].values())

def publish_checkpoint(root,state,rows,features,previous,protocol,pairs,torch,np):
    root=Path(root);checkpoints=root/'checkpoints';checkpoints.mkdir(exist_ok=True)
    n=state['updates'];start=previous['end_update'] if previous else 0
    validate_rows(rows,state['order'],start,n,pairs)
    assert features.shape==(n-start,2,768) and np.isfinite(features).all()
    assert state['registration_sha256']==ref(protocol)['sha256']
    log=b''.join((json.dumps(r,allow_nan=False)+'\n').encode() for r in rows)
    assert hashlib.sha256(log).hexdigest()==state['log_sha256']
    final=checkpoints/f'step-{n:06d}';assert not final.exists()
    pending=Path(tempfile.mkdtemp(prefix=f'step-{n:06d}.pending.',dir=checkpoints))
    with (pending/'state.pt').open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    assert (pending/'state.pt').stat().st_size<=192<<20
    with (pending/'training_log.jsonl').open('xb') as f:f.write(log);f.flush();os.fsync(f.fileno())
    with (pending/'generated_features.npy').open('xb') as f:np.save(f,features,allow_pickle=False);f.flush();os.fsync(f.fileno())
    commit=dict(schema='PAIRED_EEG_P_IMMUTABLE_CHECKPOINT_V1',arm=state['arm'],start_update=start,end_update=n,
        registration=ref(protocol),previous_commit=previous['commit'] if previous else None,
        state_sha256=sha(pending/'state.pt'),log_sha256=sha(pending/'training_log.jsonl'),features_sha256=sha(pending/'generated_features.npy'),
        primary_EEG_exposures=2*n,forward_image_UNet_evaluations=8*n)
    atomic(pending/'commit.json',commit);sync_dir(pending);pending.rename(final);sync_dir(checkpoints)
    return dict(**commit,commit=ref(final/'commit.json'),directory=str(final))

def scan_checkpoints(root,protocol,arm,order,pairs,*,all_state_hashes=False):
    root=Path(root);folder=root/'checkpoints';found=[];end=0;previous=None
    if not folder.exists():return found
    for p in sorted(folder.iterdir()):
        if '.pending.' in p.name:
            assert p.is_dir() and p.name.startswith('step-');continue
        assert p.is_dir() and not p.is_symlink() and p.name.startswith('step-')
        assert set(x.name for x in p.iterdir())=={'state.pt','training_log.jsonl','generated_features.npy','commit.json'}
        c=read(p/'commit.json');assert c['schema']=='PAIRED_EEG_P_IMMUTABLE_CHECKPOINT_V1' and c['arm']==arm
        assert c['registration']==ref(protocol) and c['previous_commit']==previous and c['start_update']==end
        assert p.name==f"step-{c['end_update']:06d}" and c['end_update']>end
        assert sha(p/'training_log.jsonl')==c['log_sha256'] and sha(p/'generated_features.npy')==c['features_sha256']
        rows=[json.loads(line) for line in (p/'training_log.jsonl').read_text().splitlines()]
        validate_rows(rows,order,end,c['end_update'],pairs)
        if all_state_hashes:assert sha(p/'state.pt')==c['state_sha256']
        previous=ref(p/'commit.json');end=c['end_update'];found.append(dict(**c,commit=previous,directory=str(p)))
    if found:assert sha(Path(found[-1]['directory'])/'state.pt')==found[-1]['state_sha256']
    return found

def image_decision(contrasts,stage):
    primary=contrasts['joint_correct'];base=primary['ordinary'];d=base['DINO2AFC']
    if stage==80:
        stop=d['difference']<=-.02 and d['descriptive_concept_bootstrap95'][1]<0
        return dict(action='STOP_INSTANCE' if stop else 'EXPAND_ALL_TO400',reason='CLEAR_PRIMARY_HARM' if stop else 'NO_CLEAR_PRIMARY_HARM')
    assert stage==400
    efficacy=d['difference']>=.01 and d['descriptive_concept_bootstrap95'][0]>0 and base['within_concept10_top1']['difference']>=0
    mechanism={a:primary[a]['DINO2AFC']['difference']>0 and primary[a]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 for a in ARMS if a!='joint_correct'}
    return dict(action='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION' if efficacy and all(mechanism.values()) else 'STOP_INSTANCE',
        efficacy_support=efficacy,mechanism_support=mechanism,not_SOTA=True)
