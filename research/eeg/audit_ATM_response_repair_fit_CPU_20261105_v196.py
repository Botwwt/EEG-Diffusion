"""Full fixed-epoch identity, independent CPU64 inference and response targets."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,hashlib,importlib.util,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_response_repair_contract_20261105_v196 as C
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
 src={k:torch.load(K.checked(v),map_location='cpu',weights_only=True,mmap=True)['model'] for k,v in q['models'].items()};initial={n:src['NN'][n] if n.startswith('proj_eeg.') else v for n,v in src['CC'].items()}
 mean=torch.load(K.checked(tc['physical']['EEG']),map_location='cpu',weights_only=True,mmap=True);test=torch.load(K.checked(tc['R80']['EEG_R80']),map_location='cpu',weights_only=True,mmap=True)
 visual=torch.load(K.checked(tc['physical']['visual_target']),map_location='cpu',weights_only=True,mmap=True)
 saved=np.load(K.checked(K.read(out/'conditions_completion.json')['conditions']));ids=[r['array_row'] for r in tc['queries']];proofs=[];count=0
 h94=c['cv']['bq']['h94'];K.checked(h94)
 for j,arm in enumerate(C.ARMS[2:]):
  guard();dest=out/arm;done=K.read(dest/'fit_10_completion.json');assert done['updates']==830 and done['exposures']==132300
  final=torch.load(K.checked(done['checkpoint']),map_location='cpu',weights_only=True,mmap=True);net=ATMS().double();net.load_state_dict(final['model'],strict=True);net.eval();H.prepare_student(net)
  for n,v in net.named_buffers():assert torch.equal(v,initial[n].double() if initial[n].is_floating_point() else initial[n])
  changed=[];frozen=0
  for n,p in net.named_parameters():
   if not p.requires_grad:assert torch.equal(p,initial[n].double());frozen+=p.numel()
   elif not torch.equal(final['model'][n],initial[n]):changed.append(n)
  assert any(n.startswith('encoder.') for n in changed) and any(n.startswith('enc_eeg.') for n in changed)
  assert all(int(v['step'])==830 for v in final['optimizer']['state'].values())
  epochs=[json.loads(s) for s in (dest/'epochs.jsonl').read_text().splitlines()];assert len(epochs)==10
  for e,v in enumerate(epochs):
   order=np.random.default_rng(1739+e).permutation(1323);assert v['epoch']==e+1 and v['updates']==83*(e+1) and v['exposures']==13230*(e+1)
   assert v['coverage_min']==v['coverage_max']==1 and v['concept_order_sha256']==hashlib.sha256(order.tobytes()).hexdigest()
  mx=sq=ss=0.
  with torch.inference_mode():
   for lo in range(0,600,50):
    guard();x=mean[ids[lo:lo+50]] if lo<400 else test[lo-400:lo-350];pred=net(x.double(),torch.ones(len(x),dtype=torch.long)).numpy();delta=pred-saved[lo:lo+50,j]
    np.testing.assert_allclose(pred,saved[lo:lo+50,j],atol=3e-5,rtol=3e-5);mx=max(mx,float(abs(delta).max()));sq+=float((delta**2).sum());ss+=float((pred**2).sum());count+=len(x)
  fixture=np.load(dest/'first_batch_fixture.npz');unit=int(fixture['unit']);ti,li=divmod(unit,70);weights={}
  with safe_open(h94['path'],framework='pt',device='cpu') as f:
   for key,name in [('P_weight','image_proj.proj.weight'),('P_bias','image_proj.proj.bias'),('P_gamma','image_proj.norm.weight'),('P_beta','image_proj.norm.bias'),('K',f'ip_adapter.{1+2*li}.to_k_ip.weight'),('V',f'ip_adapter.{1+2*li}.to_v_ip.weight')]:weights[key]=f.get_tensor(name).half().double()
  Q=torch.tensor(fixture['Q']).double();rows=fixture['T_rows'];assert np.array_equal(rows,tc['T'][(np.random.default_rng(1739).permutation(1323)[:16,None]*10+np.arange(10)).reshape(-1)])
  with torch.inference_mode():
   teacher=H.response(visual[rows].double(),Q,weights).numpy();old=H.response(torch.tensor(fixture['NN_condition']).double(),Q,weights).numpy();updated=H.response(torch.tensor(fixture['CN_condition']).double(),Q,weights).numpy()
  np.testing.assert_allclose(old,fixture['old_response'],atol=1e-5,rtol=1e-5);np.testing.assert_allclose(updated,fixture['updated_response'],atol=1e-5,rtol=1e-5)
  D=teacher-teacher.mean(0);delta=updated-old;u,s,vt=np.linalg.svd(D,full_matrices=False);keep=s>s.max()*1e-6;p=(delta@vt[keep].T)@vt[keep];target=fixture['target'];actual=target-old
  energy_error=float(abs((actual**2).sum(1)-(p**2).sum(1)).max()/max(float((delta**2).sum(1).max()),1e-20))
  distance_error=float(abs(((target-updated)**2).sum(1)-((p-delta)**2).sum(1)).max()/max(float((delta**2).sum(1).max()),1e-20))
  assert energy_error<1e-4 and distance_error<1e-4
  projection_error=float(np.linalg.norm(actual-p)/max(np.linalg.norm(p),1e-20))
  if arm=='response_repair':assert projection_error<1e-4
  proofs.append(dict(arm=arm,checkpoint=done['checkpoint'],CPU64_queries=600,max_abs_error=mx,relative_L2=(sq/ss)**.5,frozen_projection_logit_parameters=frozen,changed_names=changed,first_actual_Q_unit_CPU64_SVD_projection_error=projection_error,matched_energy_error=energy_error,matched_initial_distance_error=distance_error,all10orders_coverage=True,allBNexact=True))
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
 guard();C.resources(c);K.write(out/'fit_audit_completion.json',dict(status='PASS_ALL1200_CPU64_RESPONSE_REPAIR_FULL10_EPOCHS',models=proofs,conditions=K.ref(out/'conditions.npy'),encoder_readout=K.ref(out/'encoder_readout.json'),full_T_epochs_per_arm=10,updates_per_arm=830,exposures_per_arm=132300,CPU64_queries=count,CUDA_initialized=False,seconds=time.monotonic()-started))
except BaseException as exc:K.write(out/'fit_audit_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
