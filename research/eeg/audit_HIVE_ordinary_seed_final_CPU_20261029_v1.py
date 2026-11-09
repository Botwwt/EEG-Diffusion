"""Final fixedseed metadata/model/Adam/order/coverage audit, no target effects."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
from brainhive_seed_repro_R80_contract_20261029_v1 import validate_replicate_state
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])==x;return Path(x['path'])
def run(path,output):
 study=read(path);fit=read(checked(study['training_protocol']));root=Path(fit['output_root']);closed=read(root/'training_completion.json');assert closed['status']=='BOTH_PREDECLARED_UNINTERRUPTED_ORDINARY_REPLICATES_FULL_T_COMPLETE' and closed['protocol']==study['training_protocol'] and closed['seeds']==[2027,3407]
 sys.path.insert(0,fit['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
 C.setup_runtime(dict(runtime_manifest=fit['runtime_manifest']));import torch
 torch.set_num_threads(2);reports=[]
 for number in (2027,3407):
  done=read(root/f'seed{number}/part00/completion.json');assert done['seed']==number and done['status']=='COMPLETE_FULL10_TRUE_OOF_T_THREE_PASSES_ORDINARY_REPLICATE' and done['updates']==24807 and done['seen']==396900 and done['all_frozen_weights_unchanged']
  assert done['protocol']==study['training_protocol'];state=torch.load(checked(done['checkpoint']),weights_only=False,map_location='cpu');final=torch.load(checked(done['component']),weights_only=True,map_location='cpu');calibration=read(checked(done['calibration']))
  assert state['updates']==24807 and state['seen']==396900 and state['seed']==number and state['coverage'].shape==(3,132300) and state['coverage'].all();assert state['protocol_sha256']==study['training_protocol']['sha256']
  assert sum(x.numel() for x in final['model'].values())==525568 and all(torch.equal(x,final['model'][k]) for k,x in state['model'].items()) and all(torch.isfinite(x).all() for x in final['model'].values());assert final['calibration']==calibration
  expected=[]
  for i in range(3):
   seed=int(hashlib.sha256((str(number)+'|BrainHIVE_bounded_v1|pass'+str(i)).encode()).hexdigest()[:16],16)%(2**63-1);order=torch.randperm(132300,generator=torch.Generator().manual_seed(seed));expected.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
  assert state['orders_sha256']==done['orders_sha256']==expected and len(state['optimizer']['state'])==4
  assert all(int(x['step'])==24807 and torch.isfinite(x['exp_avg']).all() and torch.isfinite(x['exp_avg_sq']).all() and (x['exp_avg_sq']>=0).all() for x in state['optimizer']['state'].values())
  group=state['optimizer']['param_groups'][0];assert group['lr']==1e-4 and group['weight_decay']==.01 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
  assert all(isinstance(state[k],torch.Tensor) and state[k].dtype==torch.uint8 for k in ('noise_rng','torch_rng'));assert len(state['cuda_rng'])==1 and state['cuda_rng'][0].dtype==torch.uint8
  spec=dict(training_seed=number,training_protocol=study['training_protocol'],ordinary_strength=calibration['selected_strength']);validate_replicate_state(final,spec)
  reports.append(dict(seed=number,component=done['component'],final_completion=ref(root/f'seed{number}/part00/completion.json'),checkpoint=done['checkpoint'],calibration=done['calibration'],selected_strength=calibration['selected_strength'],updates=24807,seen=396900,all3coverage_and_orders=True,model_matches_final_optimizer_checkpoint=True,Adam4states_finite_steps24807=True,all_RNG_present=True))
 assert not torch.cuda.is_initialized();assert ref(fit['ordinary']['path'])==fit['ordinary'];result=dict(status='PASS_BOTH_FIXED_FINAL_SEEDS_ALL10_OOF_THREE_PASSES_MODEL_ADAM_COVERAGE_AND_DFIT_CALIBRATION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_protocol=study['training_protocol'],reports=reports,original_ordinary_unchanged=True,CUDA_initialized=False,image_effects_read=False,temporary_profile_weights_used=False)
 with Path(output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=result['status'],seeds=[r['seed'] for r in reports],new_images=0)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.study,a.output)
