"""One data-fixed radius, exact scalar KKT solve; no fitted predictor or grid."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']=os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
q=json.loads(Path(sys.argv[1]).read_bytes());out=Path(q['output_root']);root=Path(q['parent_root']);start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
sys.path.insert(0,str(root/'code'));import atm_fixed_projection_contract_20261105_v190 as C
c=C.setup(root/'code/policy.json');K=c['K'];C.resources(c,True);assert K.ref(__file__)==q['sources']['worker']
sys.path.insert(0,str(Path(q['original_core']['path']).parent));K.checked(q['original_core'])
from atm_response_realizability_20261105_v194 import projector,response
try:
 prior=K.read(K.checked(q['prior_completion']));equations=np.load(K.checked(prior['normal_equations']));cases=np.load(K.checked(q['inputs']['cases']));bank=np.load(K.checked(q['inputs']['conditions']))
 import torch
 from safetensors import safe_open
 torch.set_num_threads(2);K.checked(prior['h94']);weights={}
 with safe_open(prior['h94']['path'],framework='pt',device='cpu') as f:
  for n in ['image_proj.proj.weight','image_proj.proj.bias','image_proj.norm.weight','image_proj.norm.bias',*[f'ip_adapter.{idx}.{kind}.weight' for idx in [1,139] for kind in ['to_k_ip','to_v_ip']]]:weights[n]=f.get_tensor(n).half().double().numpy()
 w,b,gamma,beta=[weights['image_proj.'+n] for n in ['proj.weight','proj.bias','norm.weight','norm.bias']];rows=[];arrays={}
 for oldrow in prior['rows']:
  assert time.monotonic()-start<170 and time.time()<q['absolute_deadline_epoch']
  qi=oldrow['query'];base,CN,G,B,target=[equations[f'q{qi}__'+n] for n in ['base','CN','G','B','target']];R=float(np.linalg.norm(CN-base));eig,U=np.linalg.eigh(G);BU=B@U;calls=0
  assert eig[0]>0 and np.linalg.norm(equations[f'q{qi}__raw_step'])<R<np.linalg.norm(equations[f'q{qi}__metric_step'])
  def at(mu):
   global calls
   calls+=1;invroot=1/np.sqrt(eig+mu);H=BU*invroot[None,:];uh,sv,vh=np.linalg.svd(H,full_matrices=False);keep=sv>1e-10*sv[0]
   v=U@(invroot*(vh[keep].T@((uh[:,keep].T@target)/sv[keep])))
   lam=uh[:,keep]@((uh[:,keep].T@target)/(sv[keep]**2));return v,lam,int(keep.sum())
  lo=0.;hi=float(eig[-1]);v,lam,rank=at(hi);bracket_calls=1
  while np.linalg.norm(v)>R:
   hi*=10;v,lam,rank=at(hi);bracket_calls+=1;assert bracket_calls<32
  iterations=0
  for iterations in range(80):
   mu=(lo+hi)/2;v,lam,rank=at(mu);n=float(np.linalg.norm(v))
   if n>R:lo=mu
   else:hi=mu
   if abs(n-R)<=1e-8*R:break
  assert abs(n-R)<=1e-8*R
  kres=float(np.linalg.norm(G@v+mu*v-B.T@lam)/max(np.linalg.norm(G@v+mu*v),np.linalg.norm(B.T@lam),1e-12));cres=float(np.linalg.norm(B@v-target)/np.linalg.norm(target));assert kres<1e-8 and cres<1e-8
  proposed=base+v;pn,pp=projector(base,w,b,gamma,beta),projector(proposed,w,b,gamma,beta);pt=[projector(t.astype(np.float16).astype(float),w,b,gamma,beta) for t in bank['teacher'][(qi//10)*10:(qi//10+1)*10]];units=[]
  for ti in [0,3]:
   for li in [0,69]:
    label=f'q{qi}_t{ti}_l{li}';Q=cases[label+'__Q'][0].astype(float);idx=1 if li==0 else 139;wk,wv=[weights[f'ip_adapter.{idx}.{kind}.weight'] for kind in ['to_k_ip','to_v_ip']]
    original=response(pn,Q,wk,wv)[0];actual=response(pp,Q,wk,wv)[0];teachers=np.stack([response(t,Q,wk,wv)[0] for t in pt]);D=teachers-teachers.mean(0);scale=float((D*D).mean())
    distances=((actual[None]-teachers)**2).reshape(10,-1).mean(1)/scale;change=actual-original;achieved=D.reshape(10,-1)@change.ravel()/(change.size*scale)
    units.append(dict(label=label,distances=distances.tolist(),response_departure=float((change*change).mean()/scale),achieved=achieved.tolist()))
  dist=np.mean([u['distances'] for u in units],axis=0);correct=qi%10;ach=np.concatenate([u['achieved'] for u in units]);summary=dict(normalized_MSE=float(dist[correct]),top1=int(dist.argmin()==correct),AFC=float(((dist>dist[correct]).sum()+.5*((dist==dist[correct]).sum()-1))/9),
   mean_wrong_correct_margin=float((dist.sum()-10*dist[correct])/9),mean_response_departure=float(np.mean([u['response_departure'] for u in units])),relative_actual_margin_target_error=float(np.linalg.norm(ach-target)/np.linalg.norm(target)),condition_step_norm=float(np.linalg.norm(v)))
  rows.append(dict(query=qi,radius=R,mu=mu,root_calls=calls,bracket_calls=bracket_calls,bisection_iterations=iterations+1,constraint_rank=rank,relative_constraint_residual=cres,relative_KKT_residual=kres,linear_response_departure=float(v@G@v),summary=summary,actual_units=units))
  arrays[f'q{qi}__step']=v;arrays[f'q{qi}__lambda']=lam
 np.savez_compressed(out/'bounded_solutions.npz',**arrays);C.resources(c);assert K.allocated(out)<q['maximum_output_bytes'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30
 result=dict(status='COMPLETE_CN_RADIUS_TRUST_REGION_DIAGNOSTIC',rows=rows,prior_completion=q['prior_completion'],solutions=K.ref(out/'bounded_solutions.npz'),seconds=time.monotonic()-start,new_fit=0,new_GPU=False,new_images=0,
  limits='One posthoc geometry-driven radius exactly existing CN condition displacement; no performance-selected strength. Exact constrained LINEAR surrogate solution, nonlinear result separately reported; two seen engineering queries/privileged teachers, not deployable efficacy.')
 K.write(out/'completion.json',result);print(json.dumps(result))
except BaseException as exc:K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
