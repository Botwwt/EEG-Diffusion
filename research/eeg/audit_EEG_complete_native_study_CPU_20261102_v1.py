"""Independent profile target gradients; full fit state/order/Adam audit, no CUDA."""
import argparse,datetime,hashlib,importlib.util,os,time
from pathlib import Path
import numpy as np
import EEG_complete_native_study_contract_CPU_20261102_v1 as C

def softmax(x):
 x=np.asarray(x,np.float64);v=np.exp(x-x.max(-1,keepdims=True));return v/v.sum(-1,keepdims=True)
def logsumexp(x,axis=-1):
 x=np.asarray(x,np.float64);m=x.max(axis,keepdims=True);return (m+np.log(np.exp(x-m).sum(axis,keepdims=True))).squeeze(axis)
def independent(arm,x,z,e,H):
 B,S,K=x.shape;p=softmax(x);gx=np.zeros_like(x,np.float64);gz=np.zeros_like(z,np.float64);count=2 if arm=='global_coarse' else 10
 logpi=x.astype(np.float64)-logsumexp(x,axis=-1)[:,:,None]
 if arm=='chain_full':
  result=[H.inference(x[i].astype(np.float64),H.sigmoid(z[i]),e[i]) for i in range(B)]
  loss=-np.mean([r['log_evidence'] for r in result]);gx=np.stack([r['negative_log_evidence_gradient_logits'] for r in result])/B;gz=np.stack([r['negative_log_evidence_gradient_stay_logits'] for r in result])/B
 elif arm=='independent_full':
  joint=logpi+e;post=softmax(joint);gx=(p-post)/B;loss=-logsumexp(joint).sum(1).mean()
 elif arm=='visual_global_full':
  teacher=softmax(e.sum(1));gx[:,0]=(p[:,0]-teacher)/B;loss=-(teacher*logpi[:,0]).sum(1).mean()
 else:
  joint=logpi[:,0]+e[:,:count].sum(1);post=softmax(joint);gx[:,0]=(p[:,0]-post)/B;loss=-logsumexp(joint).mean()
 return loss,gx,gz

def profile_audit(protocol):
 q=C.setup(protocol);C.resources(q);C.roots();import torch
 import EEG_complete_native_runtime_20261102_v1 as R
 c,bq=C.bank_contract();d,n,prev,_=c.dependency(bq)
 pin=next(v for v in C.read(d.previous()[0].NATIVE['path'])['source_pins'] if Path(v['path']).name=='EEG_multiscale_native_class_DRAFT_CPU_20261101_v1.py')
 sp=importlib.util.spec_from_file_location('independent_complete_HMM_profiles',C.checked(pin));H=importlib.util.module_from_spec(sp);sp.loader.exec_module(H)
 errors={};bindings={};elements=0
 for arm in C.ARMS:
  root=C.OUT/'tasks'/(arm+'_profile');done=C.read(root/'completion.json');assert done['status']=='PASS_REAL64_ALL_TARGET_PROFILE_AND_EXACT_STEP2_TO3_RESUME' and done['protocol']==C.ref(protocol)
  assert done['arm']==arm and done['retained_trajectory_updates']==3 and done['actual_temporary_optimizer_calls']==4 and done['model_Adam_RNG_resume_bit_exact']
  maximum=[0.,0.,0.]
  for pin in done['traces']:
   with np.load(C.checked(pin),allow_pickle=False) as f:
    assert np.array_equal(f['rows'],np.linspace(0,13229,64,dtype=np.int64)) and f['emissions'].shape==f['logits'].shape==(64,10,1000)
    loss,gx,gz=independent(arm,f['logits'],f['stay'],f['emissions'],H)
    err=[abs(loss-float(f['loss'])),float(np.abs(gx-f['gradient_logits']).max()),float(np.abs(gz-f['gradient_stay']).max())]
    assert err[0]<1e-7 and max(err[1:])<2e-7;maximum=[max(a,b) for a,b in zip(maximum,err)];elements+=gx.size+gz.size
  before=torch.load(C.checked(done['before2']),map_location='cpu',weights_only=False);after=torch.load(C.checked(done['after3']),map_location='cpu',weights_only=False)
  for value,steps in [(before,2),(after,3)]:
   assert value['arm']==arm and value['protocol']==C.ref(protocol) and value['purpose']=='ENGINEERING_ONLY_NEVER_GENERATION' and value['updates']==steps
   assert value['optimizer']['state'] and all(int(v['step'])==steps and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in value['optimizer']['state'].values())
   assert value['CPU_RNG'].dtype==torch.uint8 and len(value['CUDA_RNG'])==1 and value['CUDA_RNG'][0].dtype==torch.uint8
  assert R.digest(after['model'])==done['final_digest'] and R.digest(before['model'])!=done['final_digest']
  errors[arm]=dict(max_loss_error=maximum[0],max_logit_gradient_error=maximum[1],max_stay_gradient_error=maximum[2],all3_real64_traces_checked=True,optimizer_snapshots_steps=[2,3],actual_GPU_same_hardware_replay_proof_checked=True,CPU_full_backbone_backward_not_replayed=True)
  bindings[arm]=C.ref(root/'completion.json');del before,after
 assert not torch.cuda.is_initialized()
 C.write(C.OUT/'all6_profiles_CPU_audit.json',dict(status='PASS_ALL6_REAL_PROFILE_GRADIENTS_AND_SERIALIZED_RESUME_STATES',protocol=C.ref(protocol),profiles=bindings,independent_gradient_elements=elements,errors=errors,new_formal_models=0,new_research_images=0,CUDA_initialized=False))

def fit_audit(protocol,arm,epoch):
 q=C.setup(protocol,formal=True);assert arm in C.ARMS and epoch in (5,10);C.roots();import torch
 import EEG_complete_native_runtime_20261102_v1 as R
 root=C.OUT/'fits'/arm;commit=C.read(root/'commit.json');assert commit['protocol']==C.ref(protocol) and commit['epoch']==epoch and commit['updates']==epoch*207
 state=torch.load(C.checked(commit['checkpoint']),map_location='cpu',weights_only=False);assert state['protocol']==C.ref(protocol) and state['arm']==arm and state['purpose']=='FORMAL_FIXED_FULL_T'
 assert state['epoch']==epoch and state['updates']==epoch*207 and state['query_exposures']==epoch*13230 and len(state['coverage'])==epoch and len(state['events'])==epoch*207
 p=R.model(q,arm,'cpu');assert R.digest(p.state_dict())==state['initial_model_digest'];initial=R.cpu_tree(p.state_dict());p.load_state_dict(state['model'],strict=True)
 assert R.digest(p.state_dict())!=state['initial_model_digest']
 count=0
 for k in range(epoch):
  order=torch.randperm(13230,generator=torch.Generator().manual_seed(1729+k)).numpy();coverage=state['coverage'][k]
  assert coverage['epoch']==k+1 and coverage['queries']==13230 and coverage['order_sha256']==hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest()
  for j,lo in enumerate(range(0,13230,64)):
   rows=order[lo:lo+64];event=state['events'][k*207+j]
   assert event['epoch']==k+1 and event['batch']==j+1 and event['queries']==len(rows) and event['rows_sha256']==hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()
   assert all(np.isfinite(event[key]) for key in ('optimized_loss','observed_NLL_nats_per_query','preclip_parameter_gradient','backbone_gradient'));count+=len(rows)
 assert count==epoch*13230
 names=[name for name,v in p.named_parameters() if v.requires_grad];assert names==state['parameter_names']
 groups=state['optimizer']['param_groups'];assert len(groups)==1 and groups[0]['lr']==1e-4 and groups[0]['weight_decay']==.01 and groups[0]['betas']==(.9,.999) and groups[0]['eps']==1e-8
 by_id=dict(zip(groups[0]['params'],[v for v in p.parameters() if v.requires_grad]));assert state['optimizer']['state'] and set(state['optimizer']['state'])<=set(by_id)
 for key,value in state['optimizer']['state'].items():
  assert int(value['step'])==epoch*207 and value['exp_avg'].shape==value['exp_avg_sq'].shape==by_id[key].shape
  assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all() and (value['exp_avg_sq']>=0).all()
 assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point()) and state['CPU_RNG'].dtype==torch.uint8 and len(state['CUDA_RNG'])==1
 if arm!='image_prior_full':assert any(e['backbone_gradient']>0 for e in state['events'][1:])
 final=None
 if epoch==10:
  done=C.read(root/'completion.json');assert done['status']=='COMPLETE_FIXED10_FULL_T_EEG_ARM' and done['epoch']==10 and done['updates']==2070
  model=torch.load(C.checked(done['final_model']),map_location='cpu',weights_only=True);assert model['purpose']=='FORMAL_FIXED_FULL_T_FINAL_ONLY' and R.tree_equal(model['model'],state['model'])
  assert (root/'fit5_CPU_audit.json').exists();final=done['final_model']
 assert not torch.cuda.is_initialized()
 C.write(root/f'fit{epoch}_CPU_audit.json',dict(status='PASS_ALL_FIXED_EEG_EPOCH_STATE_ADAM_COVERAGE',protocol=C.ref(protocol),arm=arm,epoch=epoch,updates=epoch*207,exposures=count,checkpoint=commit['checkpoint'],final_model=final,full_batch_order_replayed=True,all_Adam_counters_moments_and_RNG_checked=True,all_training_backbone_gradients_not_independently_recomputed=True,CUDA_initialized=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['profiles','fit5','fit10'],required=True);p.add_argument('--arm',choices=C.ARMS);v=p.parse_args()
 profile_audit(v.protocol) if v.mode=='profiles' else fit_audit(v.protocol,v.arm,int(v.mode[3:]))
