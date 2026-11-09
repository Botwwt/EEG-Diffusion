"""Independent decoding of a committed prefix/final ordinary checkpoint."""
import hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');folder=R/'runs/HIVE_matching_ordinary_sub03_v259';out=R/('runs/matching_ordinary_state_audit_'+os.environ['SLURM_JOB_ID']);out.mkdir();start=time.monotonic()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();p=out/n;t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import torch
    from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
    torch.set_num_threads(1)
    p=folder/'resume.pt';receipt=json.loads((folder/'checkpoint_receipt.json').read_text());digest=sha(p);assert digest==receipt['checkpoint_sha256']
    state=torch.load(p,map_location='cpu',weights_only=False);provenance=json.loads((folder/'input_provenance.json').read_text());policy=json.loads((R/'code/matching_ordinary_policy_v259.json').read_text())
    assert provenance['subject']=='sub-03' and provenance['old_ordinary_untouched'] and provenance['train_queries']==13230
    assert state['input_provenance_sha256']==sha(folder/'input_provenance.json') and provenance['policy_sha256']==sha(R/'code/matching_ordinary_policy_v259.json')
    for name,d in policy['source_sha256'].items():assert sha(R/'code'/name)==d
    torch.manual_seed(1729);initial=BoundedConditionAdapter(predict_risk=False);model=BoundedConditionAdapter(predict_risk=False);model.load_state_dict(state['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==525568 and all(torch.isfinite(v).all() for v in state['model'].values())
    assert torch.equal(initial.direction.weight,torch.zeros_like(initial.direction.weight)) and model.direction.weight.norm()>0
    changes={n:float((v-initial.state_dict()[n]).abs().max()) for n,v in state['model'].items()};assert max(changes.values())>0
    def seed(n):return int(hashlib.sha256(('1729|BrainHIVE_bounded_v1|'+n).encode()).hexdigest()[:16],16)%(2**63-1)
    orders=[torch.randperm(13230,generator=torch.Generator().manual_seed(seed('pass'+str(i)))) for i in range(3)]
    assert [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders]==provenance['orders_sha256']
    updates=state['updates'];seen=state['seen'];assert updates==receipt['updates'] and seen==receipt['seen']==state['cursor'] and len(state['history'])==updates
    coverage=torch.zeros(3,13230,dtype=torch.bool);count=0;loss_sum=0.;last_seconds=0.
    for k,e in enumerate(state['history'],1):
        ep=(k-1)//827;step=(k-1)%827;offset=step*16;idx=orders[ep][offset:offset+16]
        assert e['update']==k and e['pass_number']==ep+1 and e['offset']==offset and e['n']==len(idx) and not coverage[ep,idx].any()
        coverage[ep,idx]=True;count+=len(idx);assert e['seen']==count and e['seconds']>=last_seconds;last_seconds=e['seconds']
        assert abs(e['loss']-e['epsilon_loss']-.1*e['token_loss'])<3e-7
        assert e['gradient_norm']>0 and e['direction_weight_gradient_norm']>0 and e['peak_GPU_allocated_bytes']<20<<30
        loss_sum+=e['loss']*len(idx)
    assert count==seen and torch.equal(state['coverage'],coverage) and abs(loss_sum-state['loss_sum'])<1e-8
    opt=state['optimizer'];assert len(opt['state'])==4 and len(opt['param_groups'])==1
    group=opt['param_groups'][0];assert group['lr']==1e-4 and group['weight_decay']==.01 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
    for v in opt['state'].values():
        assert float(v['step'])==updates and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
    for s in [state['CPU_RNG'],state['noise_rng']]+state['CUDA_RNG']:assert s.dtype==torch.uint8 and s.numel()>=16
    assert len(state['CUDA_RNG'])==1 and sha(p)==digest
    if updates==4:assert seen==64 and sha(folder/'first4_committed.pt')==digest
    result=dict(status='PASS_COMMITTED_MATCHING_ORDINARY_MODEL_ADAM_RNG_AND_EXACT_COVERAGE',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),seconds=time.monotonic()-start,
        checkpoint_sha256=digest,updates=updates,exposures=seen,expected_updates=2481,expected_exposures=39690,parameters=525568,coverage_exact=True,all_committed_losses_reconciled=True,
        parameter_max_changes=changes,RNG_state_structure_pass=True,CUDA_training_replay_performed=False,full_training_complete=updates==2481,new_images=0,new_optimizer_updates=0,
        immutable_old_ordinary_changed=False,trained_prefix_is_not_efficacy=True)
    write('completion.json',result);print(json.dumps(result),flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
