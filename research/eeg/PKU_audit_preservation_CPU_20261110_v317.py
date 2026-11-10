"""CPU checkpoint/optimizer/order proof; not an independent UNet trajectory replay."""
import argparse,json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args()
import torch
from PKU_conditional_preservation_core_20261110_v317 import R,ROOT,CASES,flags,sha,write
from PKU_eeg_candidate_core_20261109_v266 import seed
torch.set_num_threads(4);assert a.case in CASES;p=ROOT/a.case;start=time.monotonic();receipt=json.loads((p/'checkpoint_receipt.json').read_text());digest=sha(p/'state.pt');assert digest==receipt['checkpoint_sha256']
s=torch.load(p/'state.pt',map_location='cpu',weights_only=False);common=json.loads((p/'provenance.json').read_text());assert s['case']==a.case and s['provenance_sha256']==sha(p/'provenance.json')
assert s['identity']['native_target_zero_change_exact'];assert len(s['model'])==(1 if a.case.startswith('P_') else 70)
count=0
for name,weights in s['model'].items():
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in weights.values());count+=sum(v.numel() for v in weights.values())
assert count==8400896 and common['trainable_parameters']==8388608
assert s['updates']==receipt['updates']==len(s['history']) and s['cursor']==receipt['exposures'] and s['updates']==(s['cursor']+15)//16
order=torch.randperm(13230,generator=torch.Generator().manual_seed(seed('order')));coverage=torch.zeros(13230,dtype=torch.bool);flags_=flags(a.case)
for j,row in enumerate(s['history'],1):
    ix=order[(j-1)*16:j*16];assert row['update']==j and row['offset']==(j-1)*16 and row['n']==len(ix) and row['exposures']==min(j*16,13230);assert not coverage[ix].any();coverage[ix]=True
    expected=row['positive_energy']+.1*row['token_error']+row['visual_replay']+row['contrastive']+row['EEG_preservation'];assert abs(row['loss']-expected)<1e-6
    assert row['grad_norm']>0 and row['peak_GPU_allocated_bytes']<20<<30 and row['visual_replay']>=0 and row['contrastive']>=0
assert torch.equal(coverage,s['coverage']) and int(coverage.sum())==s['cursor']
group=s['optimizer']['param_groups'][0];assert group['lr']==1e-5 and group['weight_decay']==.01 and len(s['optimizer']['state'])==len(group['params'])
for v in s['optimizer']['state'].values():assert float(v['step'])==s['updates'] and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
for v in [s['CPU_RNG'],s['noise_RNG']]+s['CUDA_RNG']:assert v.dtype==torch.uint8 and v.numel()>=16
g=torch.Generator().set_state(s['CPU_RNG']);assert torch.equal(g.get_state(),s['CPU_RNG']) and len(s['CUDA_RNG'])==1
assert sha(p/'state.pt')==digest
if s['cursor']==13230:
    assert s['updates']==827 and coverage.all();done=json.loads((p/'completion.json').read_text());assert done['checkpoint_sha256']==digest and sha(p/'target_final.pt')==done['target_sha256']
anchor=torch.load(p/'anchor.pt',weights_only=True,map_location='cpu');assert sha(p/'anchor.pt')==common['anchor_sha256']
from safetensors.torch import load_file
initial=load_file(str(R/'assets/native_generator_v243/prior_H14/proj/diffusion_pytorch_model.safetensors'))
current=s['model']['P']
for key in ['proj.bias','norm.weight','norm.bias']:assert torch.equal(current[key],initial[key].float())
constraint=0.
if flags_['anchor']!='free':
    v=(anchor['mu'] if flags_['anchor']=='mean' else anchor['random_direction']).double()
    residual=(current['proj.weight'].double()-initial['proj.weight'].float().double())@v
    constraint=float(residual.abs().max());assert constraint<2e-5
out=ROOT/'audits';out.mkdir(exist_ok=True);r=dict(status='PASS_EXISTING_DOWNSTREAM_MODEL_ADAM_RNG_COVERAGE_FIXED_INPUTS',case=a.case,updates=s['updates'],exposures=s['cursor'],full_T_complete=s['cursor']==13230,checkpoint_sha256=digest,model_parameters=count,trainable_parameters=8388608,anchor_constraint_CPU64_max_abs=constraint,job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,CUDA_trajectory_replayed=False,new_updates=0,new_images=0)
write(out/(a.case+'_'+os.environ['SLURM_JOB_ID']+'.json'),r);print(json.dumps(r),flush=True)
