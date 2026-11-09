"""Two-slot atomic commits. Only this new study's rolling slots are replaced."""
import hashlib,json,os
from pathlib import Path

def ref(p):
    p=Path(p)
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha)

def checked(pin):
    assert ref(pin['path'])==pin;return Path(pin['path'])

def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def commit(directory,state,policy_pin,slot_cap=8<<20,hook=None):
    import torch
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);pointer=d/'checkpoint.json'
    previous=json.loads(pointer.read_bytes()) if pointer.exists() else None
    if previous:
        checked(previous['checkpoint']);assert previous['policy']==policy_pin
        assert state['attempts']>previous['attempts']
    slot=0 if previous is None else 1-previous['slot'];target=d/f'state_slot{slot}.pt';pending=d/'state.pending'
    assert not pending.exists()
    payload=dict(state=state,policy=policy_pin,format='CN_BASE_PLUS_TRAINABLE_AND_ADAM_V1')
    with pending.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    assert pending.stat().st_size<=slot_cap
    if hook:hook('pending_durable')
    pending.replace(target);_sync_dir(d)
    if hook:hook('slot_replaced')
    receipt=dict(slot=slot,attempts=state['attempts'],accepted=state['accepted'],exposures=state['exposures'],policy=policy_pin,checkpoint=ref(target))
    temp=d/'checkpoint.pending.json';assert not temp.exists()
    with temp.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(pointer);_sync_dir(d)
    if hook:hook('pointer_committed')
    finish_event(d,state)
    return receipt

def finish_event(d,state):
    event=state['last_event']
    if event is None:assert state['attempts']==0;return
    assert event['attempt']==state['attempts'];directory=Path(d)/'events';directory.mkdir(exist_ok=True)
    p=directory/f'{state["attempts"]:04d}.json'
    if p.exists():assert json.loads(p.read_bytes())==event
    else:
        tmp=directory/(p.name+'.pending');assert not tmp.exists()
        with tmp.open('x',encoding='utf-8') as f:json.dump(event,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
        tmp.replace(p);_sync_dir(directory)

def load(directory,policy_pin):
    import torch
    d=Path(directory);receipt=json.loads((d/'checkpoint.json').read_bytes());assert receipt['policy']==policy_pin
    payload=torch.load(checked(receipt['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
    assert payload['format']=='CN_BASE_PLUS_TRAINABLE_AND_ADAM_V1' and payload['policy']==policy_pin
    state=payload['state'];assert (state['attempts'],state['accepted'],state['exposures'])==(receipt['attempts'],receipt['accepted'],receipt['exposures'])
    finish_event(d,state)
    return state,receipt
