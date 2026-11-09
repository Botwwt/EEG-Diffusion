"""Independent anchored EEG model/Adam/RNG/coverage/cap checks."""
import argparse,hashlib,json,os,socket,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--all',action='store_true');ap.add_argument('--case');args=ap.parse_args()
import torch
from PKU_anchored_encoder_core_20261109_v286 import CASES,make_model,encode,posterior_indices,seed,sha
R=Path('/dfs/share/maleiLab/wwt/eeg');root=R/'runs/EEG_anchor_v286';folder=root/'audits';folder.mkdir(exist_ok=True);torch.set_num_threads(4)
parent=torch.load(R/'runs/HIVE_native_sub03_5096542/full/brain_final.pt',map_location='cpu',weights_only=True)['model'];split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());post,_=posterior_indices()
raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);x=raw[split['image_rows']['T'][:32]].clone();del raw
order=torch.randperm(13230,generator=torch.Generator().manual_seed(seed('order')));results=[]
for case in CASES if args.all else [args.case]:
    p=root/case;receipt=json.loads((p/'checkpoint_receipt.json').read_text());digest=sha(p/'state.pt');assert digest==receipt['checkpoint_sha256'];s=torch.load(p/'state.pt',map_location='cpu',weights_only=False)
    assert s['provenance_sha256']==sha(root/'shared_provenance.json') and s['case']==case
    m=make_model(case,device='cpu');m.load_state_dict(s['model'],strict=True)
    assert all(torch.equal(s['model']['anchor.'+k],v) for k,v in parent.items())
    for k in ['block.layers.0.norm.weight','block.layers.0.norm.bias']:assert torch.equal(s['model']['candidate.'+k],parent[k])
    assert all(torch.isfinite(v).all() for v in s['model'].values())
    assert any(not torch.equal(s['model']['candidate.'+k],v) for k,v in parent.items())
    assert s['updates']==receipt['updates']==len(s['history']) and s['cursor']==receipt['exposures'] and s['updates']==(s['cursor']+15)//16
    coverage=torch.zeros(13230,dtype=torch.bool);seen=0
    for j,row in enumerate(s['history'],1):
        ix=order[(j-1)*16:j*16];assert row['offset']==(j-1)*16 and row['update']==j and row['n']==len(ix) and not coverage[ix].any();coverage[ix]=True;seen+=len(ix);assert row['exposures']==seen
        assert abs(row['loss']-row['positive_energy']-.1*row['token_error']-row['contrastive'])<3e-7 and row['grad_norm']>0 and row['peak_GPU_allocated_bytes']<20<<30
        if case=='C_positive':assert row['contrastive']==0
    assert seen==s['cursor'] and torch.equal(coverage,s['coverage'])
    pars=[p for p in m.parameters() if p.requires_grad];opt=s['optimizer'];assert sum(p.numel() for p in pars)==6450176 and len(opt['state'])==len(pars)
    assert opt['param_groups'][0]['lr']==1e-5 and opt['param_groups'][0]['weight_decay']==.01
    for v in opt['state'].values():assert float(v['step'])==s['updates'] and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
    for v in [s['CPU_RNG'],s['noise_RNG']]+s['CUDA_RNG']:assert v.dtype==torch.uint8 and v.numel()>=16
    assert s['CPU_RNG'].numel()==torch.get_rng_state().numel();cpu_gen=torch.Generator().set_state(s['CPU_RNG']);assert torch.equal(cpu_gen.get_state(),s['CPU_RNG'])
    assert len(s['CUDA_RNG'])==1
    with torch.no_grad():old=m.anchor(x[:,post,:]);new=encode(m,case,x,post)
    g=parent['block.layers.0.norm.weight'];b=parent['block.layers.0.norm.bias'];u=(old-b)/g;v=(new-b)/g;cos=torch.nn.functional.cosine_similarity(u,v,dim=-1);assert float(cos.min())>=1/(1.0625**.5)-2e-6 and torch.allclose(u.norm(dim=1),v.norm(dim=1),atol=2e-5,rtol=2e-6)
    if args.all:assert s['updates']==2 and s['cursor']==32 and sha(p/'prefix2.pt')==digest
    if s['cursor']==13230:
        assert s['updates']==827 and coverage.all();export=torch.load(p/'full_conditions.pt',map_location='cpu',weights_only=True);assert export['subject']=='sub-03' and export['checkpoint_sha256']==digest and export['array_rows'].tolist()==split['image_rows']['T']+split['image_rows']['D'] and torch.isfinite(export['condition']).all()
    r=dict(status='PASS_ANCHORED_MODEL_ADAM_RNG_ALL_COVERAGE_FROZEN_ANCHOR_AND_CAP',case=case,updates=s['updates'],exposures=s['cursor'],full_T_complete=s['cursor']==13230,checkpoint_sha256=digest,min_real_EEG_standardized_cosine=float(cos.min()),source_sha256=sha(__file__),CUDA_trajectory_replayed=False,new_images=0)
    (folder/(case+'_'+os.environ['SLURM_JOB_ID']+'.json')).write_text(json.dumps(r,indent=2)+'\n');results.append(r)
if args.all:(folder/'profile_completion.json').write_text(json.dumps(dict(status='PASS_ALL3_ANCHORED_TWO_UPDATE_PREFIXES_CPU',cases=results),indent=2)+'\n')
print(json.dumps(results),flush=True)
