"""Full fixed-epoch identity, independent CPU64 inference and response targets."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,hashlib,importlib.util,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_continuous_margin_contract_20261106_v204 as C
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=['fit_audit'],required=True);a=ap.parse_args()
c=C.setup(a.policy);q,K,tc=c['q'],c['K'],c['train'];out=c['fit'];started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['fit_CPU'];C.resources(c)
sys.path[:0]=tc['physical']['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from safetensors import safe_open
import atm_response_contrast_repair_core_20261105_v195 as H
torch.set_num_threads(2)
def guard():
 assert time.monotonic()-started<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<5<<30
 assert not torch.cuda.is_initialized()
try:
 src={k:torch.load(K.checked(v),map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()}
 initial={n:src['NN'][n] if n.startswith('proj_eeg.') else v for n,v in src['CC'].items()}
 mean=torch.load(K.checked(tc['physical']['EEG']),map_location='cpu',weights_only=True,mmap=True)
 test=torch.load(K.checked(tc['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True)
 visual=torch.load(K.checked(tc['physical']['visual_target']),map_location='cpu',weights_only=True,mmap=True)
 saved=np.load(K.checked(K.read(out/'conditions_completion.json')['conditions']));ids=[r['array_row'] for r in tc['queries']];proofs=[];count=0
 import atm_margin_checkpoint_20261106_v202 as CK
 from atm_margin_repair_schedule_20261106_v202 import epoch_batches
 import torch.nn.functional as TF
 vr=TF.normalize(visual[tc['T']].double(),dim=1)
 Qdone=K.read(K.checked(q['Qbank_completion']));bank=np.load(K.checked(Qdone['bank']))
 rng=np.random.default_rng(1729);schedule=np.concatenate([rng.permutation(280) for _ in range(3)])[:830]
 for j,arm in enumerate(C.ARMS[2:]):
  guard();dest=out/arm;done=K.read(dest/'fit_10_completion.json')
  assert done['status']=='COMPLETE_FIXED10_CONTINUOUS_MARGIN_ARM' and done['attempts']==830 and done['exposures']==132300
  state,receipt=CK.load(dest,q['state_compatibility_policy']);assert receipt['checkpoint']==done['checkpoint'] and state['attempts']==830
  assert len(state['names'])==31 and set(state['names'])==set(state['trainable'])
  assert sum(v.numel() for v in state['trainable'].values())==545930
  assert all(not n.startswith('proj_eeg.') and n!='logit_scale' for n in state['names'])
  final={**initial,**state['trainable']};net=ATMS().double();net.load_state_dict(final,strict=True);net.eval();H.prepare_student(net)
  for n,v in net.named_buffers():assert torch.equal(v,initial[n].double() if initial[n].is_floating_point() else initial[n])
  for n,v in final.items():
   if n not in state['names']:assert torch.equal(v,initial[n])
  assert all(int(v['step'])==state['accepted'] for v in state['optimizer']['state'].values())
  assert len(state['optimizer']['state'])==31 and state['accepted']>=2
  if arm=='response_only':assert state['accepted']==830
  cumulative_exposures=accepted=0
  for e in range(10):
   order=np.random.default_rng(1739+e).permutation(1323);bs=list(epoch_batches(order.tolist()))
   coverage=np.zeros(13230,np.int16);replay=np.zeros(13230,np.int16)
   for bi,b in enumerate(bs):
    ix=e*83+bi+1;event=K.read(dest/'events'/f'{ix:04d}.json')
    assert event['attempt']==ix and event['epoch']==e+1 and event['batch']==bi+1 and event['unit']==int(schedule[ix-1])
    assert event['queries']==len(b['response_positions']) and event['constraint_queries']==len(b['constraint_positions'])
    coverage[b['response_positions']]+=1;replay[b['constraint_positions']]+=1
    cumulative_exposures+=event['queries'];accepted+=event['accepted']
    assert event['exposures']==cumulative_exposures and event['accepted_total']==accepted
    if event['accepted']:
     assert event['alpha']>0 and event['actual_parameter_step_norm']>0
     if arm=='margin_constrained':assert event['after']['minimum_change']>=-1e-6 and event['after']['decreased']==0 and event['after']['response_reduction']>1e-7
    else:assert event['alpha'] is None and event['actual_parameter_step_norm']==0
   assert np.all(coverage==1) and np.all(replay==2)
   er=K.read(dest/f'epoch{e+1:02d}.json');assert er['attempts']==83*(e+1) and er['accepted_total']==accepted and er['exposures']==13230*(e+1)
   K.checked(er['full_T_metrics'])
  assert cumulative_exposures==132300 and accepted==state['accepted']
  mx=sq=ss=0.
  with torch.inference_mode():
   for lo in range(0,600,50):
    guard();x=mean[ids[lo:lo+50]] if lo<400 else test[lo-400:lo-350]
    pred=net(x.double(),torch.ones(len(x),dtype=torch.long)).numpy();delta=pred-saved[lo:lo+50,j]
    np.testing.assert_allclose(pred,saved[lo:lo+50,j],atol=3e-5,rtol=3e-5);mx=max(mx,float(abs(delta).max()));sq+=float((delta**2).sum());ss+=float((pred**2).sum());count+=len(x)
   # Recompute every final T margin in independent CPU64 arithmetic, in batches.
   stored=np.load(dest/'epoch10_allT.npz');assert np.array_equal(stored['T_rows'],tc['T']);tmx=0.
   for lo in range(0,13230,80):
    guard();hi=min(lo+80,13230);positions=torch.arange(lo,hi);ii=torch.arange(hi-lo)
    pred=TF.normalize(net(mean[tc['T'][lo:hi]].double(),torch.ones(hi-lo,dtype=torch.long)),dim=1)
    scores=pred@vr.T;own=scores[ii,positions].clone();scores[ii,positions]=-torch.inf
    same=positions[:,None]//10==torch.arange(13230)[None,:]//10
    measures=torch.stack([own,own-scores.masked_fill(~same,-torch.inf).max(1).values,own-scores.max(1).values],1).numpy()
    np.testing.assert_allclose(measures,stored['metrics'][lo:hi],atol=3e-5,rtol=3e-5)
    tmx=max(tmx,float(abs(measures-stored['metrics'][lo:hi]).max()))
  # First objective uses actual NN response, not the old projected-output target.
  unit=int(schedule[0]);ti,li=divmod(unit,70);qb=bank[f'l{li:02d}'][:,ti];heads,dim=qb.shape[1],qb.shape[-1]
  Q=torch.tensor(qb.transpose(1,0,2,3).reshape(1,heads,8,dim).copy()).double();weights={};ip=Qdone['layouts'][li]['ip_index']
  with safe_open(K.checked(c['cv']['bq']['h94']),framework='pt',device='cpu') as f:
   for key,name in [('P_weight','image_proj.proj.weight'),('P_bias','image_proj.proj.bias'),('P_gamma','image_proj.norm.weight'),('P_beta','image_proj.norm.bias'),('K',f'ip_adapter.{ip}.to_k_ip.weight'),('V',f'ip_adapter.{ip}.to_v_ip.weight')]:weights[key]=f.get_tensor(name).half().double()
  pos=np.asarray(list(epoch_batches(np.random.default_rng(1739).permutation(1323).tolist()))[0]['response_positions']);rows=tc['T'][pos]
  with torch.inference_mode():
   reference=ATMS().double().eval();reference.load_state_dict(src['NN']);old=H.response(reference(mean[rows].double(),torch.ones(len(rows),dtype=torch.long)),Q,weights)
   reference.load_state_dict(initial);current=H.response(reference(mean[rows].double(),torch.ones(len(rows),dtype=torch.long)),Q,weights)
   teachers=H.response(visual[rows].double(),Q,weights);scale=(teachers-teachers.mean(0)).square().mean();objective=float((current-old).square().mean()/scale)
  event=K.read(dest/'events/0001.json');np.testing.assert_allclose(objective,event['response_loss_before'],atol=1e-5,rtol=1e-4)
  proofs.append(dict(arm=arm,checkpoint=done['checkpoint'],attempts=830,accepted=accepted,exposures=132300,CPU64_queries=600,max_abs_error=mx,relative_L2=(sq/ss)**.5,CPU64_all_T_queries=13230,all_T_metric_max_abs_error=tmx,first_NN_response_objective_CPU64=objective,all10orders_and_scope_coverage=True,all_frozen_parameters_buffers_exact=True,optimizer_steps_equal_actual_accepted=True))
 # Complete encoder-space outcomes, fixed before any image score is read.
 bp=K.read(K.checked(q['encoder_measurement_plan']));spec=importlib.util.spec_from_file_location('fixed_measure',K.checked(bp['sources']['core']));F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
 targets={'R4':np.load(K.checked(bp['R4_teacher'])),'R80':torch.load(K.checked(bp['R80_teacher']),map_location='cpu',weights_only=True,mmap=True).numpy()}
 weights={k:v.astype(np.float64) for k,v in np.load(K.checked(bp['projector'])).items()};arrays={};domains={};cis=0
 for domain,offset,n in [('R4',0,400),('R80',400,200)]:
  guard();rows=c['queries'][domain];groups=[[i for i,z in enumerate(rows) if z['concept_id']==g] for g in sorted({z['concept_id'] for z in rows})]
  metrics={arm:F.measure(c['values'][offset:offset+n,j],targets[domain],groups,weights,True) for j,arm in enumerate(C.ARMS)}
  arrays.update({domain+'__'+arm+'__'+k:v for arm,mm in metrics.items() for k,v in mm.items()});draw=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));contrasts={}
  for aa,bb in C.CONTRASTS:
   contrasts[aa+'-'+bb]={}
   for key in metrics[aa]:
    d=metrics[aa][key]-metrics[bb][key];gm=np.array([d[g].mean() for g in groups]);contrasts[aa+'-'+bb][key]=dict(difference=float(d.mean()),CI95=np.quantile(gm[draw].mean(1),[.025,.975]).tolist());cis+=1
  domains[domain]=dict(queries=n,means={arm:{k:float(v.mean()) for k,v in mm.items()} for arm,mm in metrics.items()},contrasts=contrasts)
 assert cis==156 and count==1200;np.savez_compressed(out/'encoder_metrics.npz',**arrays);K.write(out/'encoder_readout.json',dict(status='COMPLETE_ALL600_F_P_FOUR_ARMS_156_CIS',domains=domains,exposed_single_subject_development=True))
 guard();C.resources(c);K.write(out/'fit_audit_completion.json',dict(status='PASS_ALL1200_CPU64_CONTINUOUS_MARGIN_FULL10_EPOCHS',models=proofs,conditions=K.ref(out/'conditions.npy'),encoder_readout=K.ref(out/'encoder_readout.json'),full_T_epochs_per_arm=10,attempts_per_arm=830,exposures_per_arm=132300,CPU64_queries=count,CUDA_initialized=False,seconds=time.monotonic()-started))
except BaseException as exc:K.write(out/'fit_audit_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
