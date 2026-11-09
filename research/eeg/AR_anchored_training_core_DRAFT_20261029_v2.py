"""NOT-ADMITTED training core for three matched anchored EEG AR branches.

No CLI, GPU dispatch, model downloads or files are opened on import. The caller
must separately bind the actual VQ/VAR profile, immutable banks and resources.
Logical batches use weighted microbatch accumulation. This is a NEW stochastic
recipe, not bitwise-equivalent to stochastic full-batch16 training. Job splits
occur only after a whole logical optimizer step; partial gradients/RNG cannot
be checkpointed as a completed step. Epoch order and aggregates are state.
"""
from contextlib import nullcontext
import hashlib,json,math,os,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from eeg_alignment_training_core_20261028_v1 import cpu_tree

ARMS=('native_global','paired_anchored','donor_anchored')

def donor_derangement(concept_ids,epoch,*,seed=1729):
    """Training-only manifest-order cyclic derangement, no training RNG draws."""
    assert seed==1729 and epoch>=1
    groups={}
    for index,concept in enumerate(concept_ids):groups.setdefault(str(concept),[]).append(index)
    donors=np.empty(len(concept_ids),dtype=np.int64)
    for concept,positions in groups.items():
        assert len(positions)>=2
        digest=hashlib.sha256(f'{seed}|AR-local-donor|{epoch}|{concept}'.encode('utf8')).digest()
        shift=1+int.from_bytes(digest[:8],'little')%(len(positions)-1)
        donors[positions]=np.roll(np.asarray(positions,dtype=np.int64),-shift)
    assert np.array_equal(np.sort(donors),np.arange(len(donors))) and (donors!=np.arange(len(donors))).all()
    assert all(concept_ids[i]==concept_ids[j] for i,j in enumerate(donors))
    return donors

def transfer_local_deviations(own_c,source_c,source_memory):
    """Retain own c; replace only H_source-c_source. Never chooses a donor."""
    assert own_c.shape==source_c.shape and source_memory.shape[0]==len(own_c)
    assert source_memory.shape[2]==own_c.shape[1]
    return own_c[:,None]+(source_memory-source_c[:,None])

def optimizer_for(model):
    assert {id(p) for p in model.parameters()}=={id(p) for p in model.decoder.parameters()}
    assert all(p.requires_grad for p in model.parameters())
    return torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)

def new_cursor():
    return dict(epoch=1,completed_epochs=0,position=0,order=torch.empty(0,dtype=torch.int64),updates=0,
        query_exposures=0,epoch_loss_sum=0.,epoch_rows=0,epoch_records=[],logical_update_in_progress=False)

def validate_cursor(cursor,count,batch,total_epochs):
    assert cursor['logical_update_in_progress'] is False, 'Partial-microbatch update cannot be resumed/checkpointed'
    steps=math.ceil(count/batch);position=cursor['position'];epoch=cursor['epoch']
    assert 1<=epoch<=total_epochs and 0<=position<=count
    assert position==count or position%batch==0
    assert cursor['updates']==(epoch-1)*steps+math.ceil(position/batch)
    assert cursor['query_exposures']==(epoch-1)*count+position
    assert cursor['completed_epochs']==epoch-(position<count)
    assert len(cursor['epoch_records'])==cursor['completed_epochs']
    if cursor['order'].numel():
        assert cursor['order'].dtype==torch.int64 and torch.equal(torch.sort(cursor['order']).values,torch.arange(count))
    else:assert epoch==1 and position==0 and cursor['updates']==0
    assert cursor['epoch_rows']==position

def run_updates(model,optimizer,shuffle_rng,cursor,*,arm,global_bank,memory_bank,target_bank,concept_ids,
                total_epochs=10,batch=16,micro_batch=2,maximum_updates,device,guard=lambda:None,observer=None):
    assert arm in ARMS and maximum_updates>0 and isinstance(micro_batch,int) and 1<=micro_batch<=batch
    count=len(global_bank);d=model.decoder;vq=d.vae_proxy[0]
    assert len(memory_bank)==len(target_bank)==len(concept_ids)==count
    assert global_bank.shape==(count,1024) and memory_bank.shape[0]==count and memory_bank.shape[2]==1024
    assert target_bank.shape==(count,d.L) and d.cond_drop_rate==.1
    validate_cursor(cursor,count,batch,total_epochs)
    vq.eval().requires_grad_(False);model.train();events=[];cached_donors={}
    for _ in range(maximum_updates):
        if cursor['completed_epochs']==total_epochs:break
        guard()
        if not cursor['order'].numel() or cursor['position']==count:
            if cursor['position']==count:
                cursor['epoch']+=1;cursor['position']=0;cursor['epoch_rows']=0;cursor['epoch_loss_sum']=0.
            cursor['order']=torch.randperm(count,generator=shuffle_rng)
        epoch=cursor['epoch'];lo=cursor['position'];ids=cursor['order'][lo:lo+batch].numpy()
        donors=None
        if arm=='donor_anchored':
            if epoch not in cached_donors:cached_donors[epoch]=donor_derangement(concept_ids,epoch)
            donors=cached_donors[epoch][ids]
        optimizer.zero_grad(set_to_none=True)
        cursor['logical_update_in_progress']=True;logical_loss=0.;micro_rows=[]
        for micro_lo in range(0,len(ids),micro_batch):
            guard();micro_ids=ids[micro_lo:micro_lo+micro_batch];micro_rows.append(len(micro_ids))
            own_c=torch.tensor(np.asarray(global_bank[micro_ids]),device=device)
            own_target=torch.tensor(np.asarray(target_bank[micro_ids]),dtype=torch.long,device=device)
            assert own_target.min()>=0 and own_target.max()<d.V
            local_memory=None
            if arm=='paired_anchored':local_memory=torch.tensor(np.asarray(memory_bank[micro_ids]),device=device)
            elif arm=='donor_anchored':
                micro_donors=donors[micro_lo:micro_lo+micro_batch]
                local_memory=transfer_local_deviations(own_c,
                    torch.tensor(np.asarray(global_bank[micro_donors]),device=device),
                    torch.tensor(np.asarray(memory_bank[micro_donors]),device=device))
            with torch.no_grad():
                shifted=vq.quantize.idxBl_to_var_input(list(torch.split(own_target,[side**2 for side in d.patch_nums],dim=1)))
            assert shifted.shape==(len(micro_ids),d.L-d.first_l,d.Cvae)
            context=torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
            with context:
                logits=model(shifted,own_c,eeg_memory=local_memory)
                micro_loss=F.cross_entropy(logits.float().reshape(-1,d.V),own_target.reshape(-1))
                weighted_loss=micro_loss*(len(micro_ids)/len(ids))
            assert torch.isfinite(weighted_loss);weighted_loss.backward()
            logical_loss+=float(micro_loss.detach())*(len(micro_ids)/len(ids))
            # No state publication is permitted while these partial gradients
            # and partially consumed stochastic streams are in progress.
            del logits,weighted_loss,micro_loss,shifted,own_target,own_c,local_memory
            guard()
        assert sum(micro_rows)==len(ids)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();cursor['updates']+=1;cursor['position']+=len(ids)
        cursor['query_exposures']+=len(ids);cursor['epoch_rows']+=len(ids)
        cursor['epoch_loss_sum']+=logical_loss*len(ids);cursor['logical_update_in_progress']=False
        events.append(dict(epoch=epoch,update=cursor['updates'],array_positions=ids.tolist(),
            donor_positions=None if donors is None else donors.tolist(),loss=logical_loss,gradient_norm=float(norm),micro_batch_rows=micro_rows))
        if cursor['position']==count:
            assert cursor['epoch_rows']==count
            cursor['completed_epochs']+=1
            cursor['epoch_records'].append(dict(epoch=epoch,updates=cursor['updates'],query_exposures=cursor['query_exposures'],
                queries=count,tokens_per_query=d.L,mean_token_CE=cursor['epoch_loss_sum']/count,
                order_sha256=hashlib.sha256(cursor['order'].numpy().astype('<i8').tobytes()).hexdigest(),
                donor_map_sha256=None if arm!='donor_anchored' else hashlib.sha256(cached_donors[epoch].astype('<i8').tobytes()).hexdigest()))
        assert all(p.grad is None for p in vq.parameters())
        if observer:observer(events[-1],cursor)
        guard()
    validate_cursor(cursor,count,batch,total_epochs)
    return events

def snapshot(model,optimizer,shuffle_rng,cursor,*,arm,spec_sha256,device,micro_batch=2):
    assert cursor['logical_update_in_progress'] is False, 'Never checkpoint halfway through accumulation'
    return dict(schema='AR_anchored_complete_T_optimizer_step_microbatch_v2',arm=arm,spec_sha256=spec_sha256,micro_batch=micro_batch,
        model=cpu_tree(model.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),cursor=cpu_tree(cursor),
        shuffle_rng=shuffle_rng.get_state().clone(),torch_rng=torch.get_rng_state().clone(),
        cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()] if str(device).startswith('cuda') else [])

def publish_step_checkpoint(root,state,*,fault=None):
    """Immutable same-filesystem state+cursor commit; no retention/deletion policy."""
    root=Path(root);assert root.is_absolute() and root.resolve(strict=True)==root and not root.is_symlink()
    update=state['cursor']['updates'];assert update>0
    final=root/f'update_{update:08d}';assert not final.exists()
    pending=Path(tempfile.mkdtemp(prefix=final.name+'.pending.',dir=root))
    def digest(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    with (pending/'state.pt').open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    cursor=state['cursor'];metadata=dict(schema=state['schema'],arm=state['arm'],spec_sha256=state['spec_sha256'],
        updates=update,epoch=cursor['epoch'],position=cursor['position'],completed_epochs=cursor['completed_epochs'],
        query_exposures=cursor['query_exposures'],state_sha256=digest(pending/'state.pt'))
    with (pending/'commit.json').open('x') as f:json.dump(metadata,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if fault:fault('payload_written_before_directory_commit')
    pending.rename(final)
    return dict(path=str(final/'commit.json'),sha256=digest(final/'commit.json'),**metadata)

def read_step_checkpoint(commit_ref):
    path=Path(commit_ref['path']);assert not path.is_symlink()
    def digest(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    assert digest(path)==commit_ref['sha256'];commit=json.loads(path.read_bytes())
    payload=path.parent/'state.pt';assert digest(payload)==commit['state_sha256']
    state=torch.load(payload,map_location='cpu',weights_only=True)
    assert state['arm']==commit['arm'] and state['spec_sha256']==commit['spec_sha256']
    for field in ('updates','epoch','position','completed_epochs','query_exposures'):assert state['cursor'][field]==commit[field]
    return state

def _sync_directory(path):
    if os.name=='posix':
        descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)

def _json_write_once(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    _sync_directory(path.parent)

def _json_replace(path,value,tag):
    temporary=path.with_name(path.name+f'.pending.{os.getpid()}.{tag}')
    _json_write_once(temporary,value);os.replace(temporary,path);_sync_directory(path.parent)

def write_update_event(root,event,*,arm,spec_sha256):
    """Optional producer sink; full per-update rows are retained independently."""
    root=Path(root);assert root.is_absolute() and root.resolve(strict=True)==root and not root.is_symlink()
    assert arm in ARMS and event['update']>0
    value=dict(schema='AR_anchored_immutable_update_event_v1',arm=arm,spec_sha256=spec_sha256,**event)
    _json_write_once(root/f"update_{event['update']:08d}.json",value)

def _digest_file(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def create_rotating_store(root,*,arm,spec_sha256,authorize_owned_previous_retirement=False):
    """Caller must explicitly bind this transient retention in resource admission."""
    root=Path(root);assert root.is_absolute() and root.resolve(strict=True)==root and not root.is_symlink()
    assert arm in ARMS and authorize_owned_previous_retirement is True and not any(root.iterdir())
    marker=dict(schema='AR_private_bounded_resume_store_v1',arm=arm,spec_sha256=spec_sha256,
        allow_owned_previous_retirement=True,maximum_resume_payloads=2,
        retention='Current and previous at rest; while preparing keep verified current, retire owned previous, then write replacement; immutable metadata retained')
    _json_write_once(root/'store.json',marker);(root/'events').mkdir()
    _json_write_once(root/'manifest.json',dict(status='EMPTY',arm=arm,spec_sha256=spec_sha256,current=None,previous=None))

def _owned_store(root):
    root=Path(root);assert root.is_absolute() and root.resolve(strict=True)==root and not root.is_symlink()
    marker=json.loads((root/'store.json').read_bytes());assert marker['schema']=='AR_private_bounded_resume_store_v1'
    assert marker['allow_owned_previous_retirement'] is True and marker['maximum_resume_payloads']==2
    manifest=json.loads((root/'manifest.json').read_bytes())
    assert manifest['arm']==marker['arm'] and manifest['spec_sha256']==marker['spec_sha256']
    return root,marker,manifest

def _verify_owned_slot(root,descriptor):
    assert descriptor['slot'] in ('slot_A','slot_B')
    slot=root/descriptor['slot'];assert slot.resolve(strict=True).parent==root and not slot.is_symlink()
    assert {p.name for p in slot.iterdir()}=={'state.pt','commit.json'}
    for p in slot.iterdir():assert p.is_file() and not p.is_symlink() and p.resolve(strict=True).parent==slot
    commit_path=slot/'commit.json';assert _digest_file(commit_path)==descriptor['commit_sha256']
    commit=json.loads(commit_path.read_bytes());assert _digest_file(slot/'state.pt')==descriptor['state_sha256']==commit['state_sha256']
    assert (slot/'state.pt').stat().st_size==descriptor['state_bytes']
    assert commit['updates']==descriptor['updates'] and commit['arm']==descriptor['arm'] and commit['spec_sha256']==descriptor['spec_sha256']
    return slot,commit

def publish_rotating_checkpoint(root,state,*,fault=None):
    """Two-payload peak. Never retires current, legacy data, or unknown files."""
    root,marker,manifest=_owned_store(root)
    assert manifest['status'] in ('EMPTY','READY'), 'Interrupted preparation needs inspection; no automatic cleanup'
    assert state['arm']==marker['arm'] and state['spec_sha256']==marker['spec_sha256']
    update=state['cursor']['updates'];assert update>0
    current=manifest['current'];previous=manifest['previous']
    if current:
        _verify_owned_slot(root,current);assert update>current['updates']
    if previous:_verify_owned_slot(root,previous)
    target_name='slot_A' if current is None or current['slot']=='slot_B' else 'slot_B'
    if previous:assert previous['slot']==target_name
    reservation=dict(status='PREPARING_CHECKPOINT',arm=marker['arm'],spec_sha256=marker['spec_sha256'],
        current=current,previous=None,retiring_previous=previous,next_slot=target_name,next_update=update)
    _json_write_once(root/'events'/f'reservation_{update:08d}.json',reservation)
    _json_replace(root/'manifest.json',reservation,f'reserve_{update}')
    if fault:fault('reservation_committed_before_retirement')
    target=root/target_name
    if previous:
        old_slot,_=_verify_owned_slot(root,previous)
        assert old_slot==target and current['slot']!=target_name
        # Explicit resolved, fixed slot and exact owned filenames; no recursive
        # deletion or computed cross-shell file operation is used.
        (old_slot/'state.pt').unlink();(old_slot/'commit.json').unlink();old_slot.rmdir();_sync_directory(root)
        _json_write_once(root/'events'/f'retired_{update:08d}.json',dict(retired=previous,retained_current=current,reason='Bounded registered transient-resume retention'))
    else:assert not target.exists()
    target.mkdir()
    with (target/'state.pt').open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    cursor=state['cursor'];commit=dict(schema=state['schema'],arm=state['arm'],spec_sha256=state['spec_sha256'],
        updates=update,epoch=cursor['epoch'],position=cursor['position'],completed_epochs=cursor['completed_epochs'],
        query_exposures=cursor['query_exposures'],state_sha256=_digest_file(target/'state.pt'))
    _json_write_once(target/'commit.json',commit)
    descriptor=dict(slot=target_name,updates=update,arm=state['arm'],spec_sha256=state['spec_sha256'],
        state_sha256=commit['state_sha256'],state_bytes=(target/'state.pt').stat().st_size,
        commit_sha256=_digest_file(target/'commit.json'))
    if fault:fault('new_payload_complete_before_manifest_commit')
    ready=dict(status='READY',arm=state['arm'],spec_sha256=state['spec_sha256'],current=descriptor,previous=current)
    _json_write_once(root/'events'/f'committed_{update:08d}.json',ready)
    _json_replace(root/'manifest.json',ready,f'commit_{update}')
    assert len(list(root.glob('slot_*/state.pt')))<=2
    return ready

def read_rotating_current(root):
    root,marker,manifest=_owned_store(root)
    assert manifest['current'] is not None
    slot,commit=_verify_owned_slot(root,manifest['current'])
    state=torch.load(slot/'state.pt',map_location='cpu',weights_only=True)
    assert state['arm']==marker['arm'] and state['spec_sha256']==marker['spec_sha256']
    for field in ('updates','epoch','position','completed_epochs','query_exposures'):assert state['cursor'][field]==commit[field]
    return state,manifest

def restore(state,model,optimizer,shuffle_rng,*,arm,spec_sha256,count,batch,total_epochs,device,micro_batch=2):
    assert state['schema']=='AR_anchored_complete_T_optimizer_step_microbatch_v2' and state['arm']==arm and state['spec_sha256']==spec_sha256 and state['micro_batch']==micro_batch
    validate_cursor(state['cursor'],count,batch,total_epochs)
    expected=model.state_dict();assert set(state['model'])==set(expected)
    for key,value in state['model'].items():
        assert value.shape==expected[key].shape and value.dtype==expected[key].dtype
        if key.endswith('attn_bias_for_masking'):assert torch.equal(value,expected[key].cpu())
        elif value.is_floating_point():assert torch.isfinite(value).all()
    groups=state['optimizer']['param_groups'];assert len(groups)==1
    group=groups[0];assert group['lr']==1e-4 and group['weight_decay']==.01 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
    assert len(group['params'])==len(list(model.parameters())) and len(set(group['params']))==len(group['params'])
    assert state['cursor']['updates']>0 and set(group['params'])==set(state['optimizer']['state'])
    for identifier,parameter in zip(group['params'],model.parameters()):
        item=state['optimizer']['state'][identifier]
        assert float(item['step'])==state['cursor']['updates']
        for key in ('exp_avg','exp_avg_sq'):assert item[key].shape==parameter.shape and item[key].dtype==parameter.dtype and torch.isfinite(item[key]).all()
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    shuffle_rng.set_state(state['shuffle_rng']);torch.set_rng_state(state['torch_rng'])
    if str(device).startswith('cuda'):
        assert len(state['cuda_rng'])==torch.cuda.device_count();torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:assert not state['cuda_rng']
    return cpu_tree(state['cursor'])
