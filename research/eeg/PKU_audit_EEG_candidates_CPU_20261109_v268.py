"""Independent case checkpoints, exact coverage, frozen coordinates and folding."""
import argparse,hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');ap.add_argument('--case');args=ap.parse_args()
R=Path('/dfs/share/maleiLab/wwt/eeg');ROOT=R/'runs/EEG_candidates_v267';audit=ROOT/'audits';audit.mkdir(exist_ok=True);start=time.monotonic()
def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import torch
    from PKU_eeg_candidate_core_20261109_v266 import CASES,seed,sha,posterior_indices,negative_cycle,make_model,encode
    torch.set_num_threads(4)
    common=json.loads((ROOT/'shared_provenance.json').read_text());policy=json.loads((R/'code/EEG_two_core_candidates_v266.json').read_text())
    for n,d in common['source_pins'].items():assert sha(R/'code'/n)==d
    assert sha(R/'code/EEG_two_core_candidates_v266.json')==common['policy_sha256']
    assert sha(R/'runs/HIVE_native_sub03_5096542/full/brain_final.pt')==common['native_parent_sha256']
    stats=torch.load(ROOT/'spatial_statistics.pt',map_location='cpu',weights_only=True);temp=torch.load(ROOT/'energy_temperature.pt',map_location='cpu',weights_only=True)
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());neg,probe=negative_cycle(split['image_rows']['T']);post,other=posterior_indices()
    assert temp['T_positions']==probe and len(probe)==256 and stats['post']==post and stats['other']==other
    expected_tau=max(float(temp['paired_gaps'].double().square().mean().sqrt()),1e-4);assert abs(temp['tau']-expected_tau)<1e-7 and common['tau']==temp['tau']
    B=stats['B'].double();CPP=stats['CPP'];CNP=stats['CNP'];normal=float((CNP-B@CPP-stats['ridge']*B).abs().max()/(1+CNP.abs().max()));assert normal<1e-6
    order=torch.randperm(13230,generator=torch.Generator().manual_seed(seed('order')));assert hashlib.sha256(order.numpy().tobytes()).hexdigest()==common['order_sha256']
    assert hashlib.sha256(neg.numpy().tobytes()).hexdigest()==common['negative_cycle_sha256']
    parent=torch.load(R/'runs/HIVE_native_sub03_5096542/full/brain_final.pt',map_location='cpu',weights_only=True)['model']
    raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);xx=raw[split['image_rows']['T'][:2]].clone();del raw
    results=[]
    for case in CASES if args.all else [args.case]:
        assert case in CASES;folder=ROOT/case;receipt=json.loads((folder/'checkpoint_receipt.json').read_text());digest=sha(folder/'state.pt');assert digest==receipt['checkpoint_sha256']
        state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False);assert state['case']==case and state['provenance_sha256']==sha(ROOT/'shared_provenance.json')
        model=make_model(case,stats,device='cpu');model.load_state_dict(state['model'],strict=True);assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
        if case.startswith('A'):
            for k in ['block.layers.0.norm.weight','block.layers.0.norm.bias']:assert torch.equal(state['model'][k],parent[k])
            assert any(not torch.equal(state['model'][k],parent[k]) for k in parent if not k.startswith('block.layers.0.norm.'))
        else:
            assert all(torch.equal(state['model']['base.'+k],v) for k,v in parent.items()) and model.innovation.weight.norm()>0
            with torch.no_grad():
                model.double();W,b=model.folded_projection();x=xx.double();actual=model.base.proj_in(x[:,post,:].reshape(2,-1))+model.innovation(model.residual(x).reshape(2,-1));fold=torch.nn.functional.linear(x.reshape(2,-1),W,b)
                assert torch.allclose(actual,fold,atol=1e-9,rtol=1e-9);fold_error=float((actual-fold).abs().max());model.float()
        cursor=state['cursor'];updates=state['updates'];assert updates==receipt['updates']==len(state['history']) and cursor==receipt['exposures'] and updates==(cursor+15)//16
        coverage=torch.zeros(13230,dtype=torch.bool);seen=0;last_seconds=0.
        for j,e in enumerate(state['history'],1):
            offset=(j-1)*16;idx=order[offset:offset+16];assert e['update']==j and e['offset']==offset and e['n']==len(idx) and not coverage[idx].any()
            seen+=len(idx);coverage[idx]=True;assert e['exposures']==seen and e['seconds']>=last_seconds;last_seconds=e['seconds']
            assert abs(e['loss']-e['positive_energy']-.1*e['token_error']-e['contrastive'])<3e-7
            assert e['grad_norm']>0 and e['peak_GPU_allocated_bytes']<20<<30
            if case=='A_positive' or case.startswith('B'):assert e['contrastive']==0
        assert seen==cursor and torch.equal(coverage,state['coverage'])
        params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==(6450176 if case.startswith('A') else 11776000)
        opt=state['optimizer'];group=opt['param_groups'][0];assert len(opt['state'])==len(params) and group['lr']==(1e-5 if case.startswith('A') else 1e-4) and group['weight_decay']==.01
        for v in opt['state'].values():assert float(v['step'])==updates and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
        for v in [state['CPU_RNG'],state['noise_RNG']]+state['CUDA_RNG']:assert v.dtype==torch.uint8 and v.numel()>=16
        assert len(state['CUDA_RNG'])==1 and sha(folder/'state.pt')==digest
        if args.all:assert updates==2 and cursor==32 and sha(folder/'prefix2.pt')==digest
        if cursor==13230:
            assert updates==827 and coverage.all();export=torch.load(folder/'full_conditions.pt',map_location='cpu',weights_only=True)
            assert export['checkpoint_sha256']==digest and export['array_rows'].tolist()==split['image_rows']['T']+split['image_rows']['D'] and torch.isfinite(export['condition']).all()
        result=dict(status='PASS_CANDIDATE_MODEL_ADAM_RNG_COVERAGE_AND_FROZEN_COORDINATES',case=case,updates=updates,exposures=cursor,full_T_complete=cursor==13230,checkpoint_sha256=digest,
            trainable_parameters=sum(p.numel() for p in params),folded_projection_CPU_FP64_max_abs=fold_error if case.startswith('B') else None,CUDA_trajectory_replay=False,new_updates=0,new_images=0)
        write(audit/(case+'_'+os.environ['SLURM_JOB_ID']+'.json'),result);results.append(result);del model,state,opt
    complete=dict(status='PASS_ALL6_TWO_UPDATE_PREFIXES_INDEPENDENT_CPU' if args.all else 'PASS_ONE_CASE_INDEPENDENT_CPU',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),seconds=time.monotonic()-start,cases=results,spatial_normal_equation_scaled_error=normal,old_ordinary_untouched=True,new_images=0)
    if args.all:write(audit/'profile_completion.json',complete)
    print(json.dumps(complete),flush=True)
except BaseException as e:
    write(audit/('failure_'+os.environ['SLURM_JOB_ID']+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
