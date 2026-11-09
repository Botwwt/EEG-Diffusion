"""Two frozen engineering queries; shared raw condition across four real Q units."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']=os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
q=json.loads(Path(sys.argv[1]).read_bytes());root=Path(q['parent_root']);out=Path(q['output_root']);start=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
sys.path.insert(0,str(root/'code'));import atm_fixed_projection_contract_20261105_v190 as C
c=C.setup(root/'code/policy.json');K=c['K'];before=C.resources(c,True)
assert K.ref(__file__)==q['sources']['worker'];K.checked(q['sources']['core'])
from atm_response_realizability_20261105_v194 import projector,response,solve,oracle
def guard():
 assert time.monotonic()-start<850 and time.time()<q['absolute_deadline_epoch']
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<10<<30 and K.allocated(out)<q['maximum_output_bytes'];C.resources(c)
try:
 for pin in q['inputs'].values():K.checked(pin)
 bank=np.load(q['inputs']['conditions']['path']);cases=np.load(q['inputs']['cases']['path']);tokens=np.load(q['inputs']['tokens']['path'])
 import torch
 from safetensors import safe_open
 torch.set_num_threads(2)
 h94=c['cv']['bq']['h94'];K.checked(h94);weights={}
 with safe_open(h94['path'],framework='pt',device='cpu') as f:
  for name in ['image_proj.proj.weight','image_proj.proj.bias','image_proj.norm.weight','image_proj.norm.bias',*[f'ip_adapter.{index}.{kind}.weight' for index in [1,139] for kind in ['to_k_ip','to_v_ip']]]:
   weights[name]=f.get_tensor(name).half().double().numpy()
 w,b,gamma,beta=[weights['image_proj.'+n] for n in ['proj.weight','proj.bias','norm.weight','norm.bias']]
 rng=np.random.default_rng(1729);rows=[];arrays={};errors=[]
 # In the native ordering verified by load_host(), layer zero -> index1, layer69 -> index139.
 for qi in [0,79]:
  guard();base=bank['conditions'][qi,0].astype(np.float16).astype(float);cn=bank['conditions'][qi,2].astype(np.float16).astype(float)
  teachers=bank['teacher'][(qi//10)*10:(qi//10+1)*10].astype(np.float16).astype(float)
  p,jp=projector(base,w,b,gamma,beta,True);pc=projector(cn,w,b,gamma,beta);pt=np.stack([projector(v,w,b,gamma,beta) for v in teachers])
  per=[];G=np.zeros((1024,1024));blocks=[];targets=[];case_errors=[]
  for ti in [0,3]:
   for li in [0,69]:
    guard();label=f'q{qi}_t{ti}_l{li}';Q=cases[label+'__Q'][0].astype(float);idx=1 if li==0 else 139
    wk,wv=[weights[f'ip_adapter.{idx}.{kind}.weight'] for kind in ['to_k_ip','to_v_ip']]
    o,k,v,J=response(p,Q,wk,wv,jp);oc=response(pc,Q,wk,wv)[0];ot=np.stack([response(t,Q,wk,wv)[0] for t in pt])
    # Native saved arrays include FP16 arithmetic; this is its continuous FP64 surrogate.
    err={}
    for key,x in [('K',k),('V',v)]:
     ref=cases[label+'__'+key][0].astype(float);err[key]=float(abs(x-ref).max()/max(1.,float(abs(ref).max())));assert err[key]<.005,(label,key,err[key])
    ref=cases[label+'__native'][0].astype(float);err['response']=float(abs(o-ref).max()/max(1.,float(abs(ref).max())));assert err['response']<.005,(label,err)
    scale=float(((ot-ot.mean(0))**2).mean());assert scale>1e-12;D=(ot-ot.mean(0)).reshape(10,-1);dim=o.size
    B=D@J/(dim*scale);target=D@(oc-o).ravel()/(dim*scale);G+=J.T@J/(4*dim*scale);blocks.append(B);targets.append(target)
    oo,op=oracle(o,oc,ot)
    # Three predeclared directions, two central differences each, all eight actual units.
    directions=[cn-base,rng.normal(size=1024),rng.normal(size=1024)]
    for di,direction in enumerate(directions):
     direction/=np.linalg.norm(direction);exact=J@direction
     for step in [1e-4,1e-5]:
      plus=response(projector(base+step*direction,w,b,gamma,beta),Q,wk,wv)[0];minus=response(projector(base-step*direction,w,b,gamma,beta),Q,wk,wv)[0]
      fd=(plus-minus).ravel()/(2*step);e=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-12));assert e<1e-6,(label,e);errors.append(dict(label=label,direction=di,step=step,relative_error=e))
    # Independent autograd VJP through native linear/LN/softmax operations, one fixed random cotangent per actual unit.
    tc=torch.tensor(base,requires_grad=True,dtype=torch.float64);pp=torch.nn.functional.layer_norm(torch.nn.functional.linear(tc,torch.from_numpy(w.reshape(8192,1024)),torch.from_numpy(b.reshape(8192))).reshape(4,2048),(2048,),torch.from_numpy(gamma),torch.from_numpy(beta),1e-5)
    heads,hd=Q.shape[0],Q.shape[-1];tk=torch.nn.functional.linear(pp,torch.from_numpy(wk)).reshape(4,heads,hd).permute(1,0,2);tv=torch.nn.functional.linear(pp,torch.from_numpy(wv)).reshape(4,heads,hd).permute(1,0,2)
    to=torch.softmax(torch.from_numpy(Q)@tk.transpose(-1,-2)/hd**.5,-1)@tv;cot=rng.normal(size=o.shape);grad=torch.autograd.grad((to*torch.from_numpy(cot)).sum(),tc)[0].numpy();gref=J.T@cot.ravel()
    err['autograd_VJP']=float(np.linalg.norm(grad-gref)/max(np.linalg.norm(gref),1e-12));assert err['autograd_VJP']<1e-10;case_errors.append(dict(label=label,errors=err))
    per.append(dict(label=label,Q=Q,wk=wk,wv=wv,NN=o,CN=oc,visual=ot,oracle=oo,oracle_proof=op,scale=scale,B=B,target=target))
    del J,tc,pp,tk,tv,to,grad
  B=np.concatenate(blocks);target=np.concatenate(targets)
  raw,rawlam,rawproof=solve(B,target);metric,metlam,metproof=solve(B,target,G)
  candidates={'NN':base,'CN':cn,'minimum_condition_norm':base+raw,'minimum_linear_response':base+metric}
  evaluation={name:[] for name in [*candidates,'independent_response_oracle']}
  for unit in per:
   for name in evaluation:
    actual=unit['oracle'] if name=='independent_response_oracle' else response(projector(candidates[name],w,b,gamma,beta),unit['Q'],unit['wk'],unit['wv'])[0]
    distances=((actual[None]-unit['visual'])**2).reshape(10,-1).mean(1)/unit['scale'];delta=actual-unit['NN'];teachercenter=unit['visual']-unit['visual'].mean(0)
    achieved=teachercenter.reshape(10,-1)@delta.ravel()/(actual.size*unit['scale'])
    evaluation[name].append(dict(label=unit['label'],distances=distances.tolist(),response_departure=float((delta*delta).mean()/unit['scale']),achieved=achieved.tolist(),target=unit['target'].tolist()))
  summary={}
  for name,units in evaluation.items():
   dist=np.mean([v['distances'] for v in units],axis=0);ach=np.concatenate([v['achieved'] for v in units]);correct=qi%10
   summary[name]=dict(normalized_MSE=float(dist[correct]),top1=int(dist.argmin()==correct),AFC=float(((dist>dist[correct]).sum()+.5*((dist==dist[correct]).sum()-1))/9),
     mean_wrong_correct_margin=float((dist.sum()-10*dist[correct])/9),mean_response_departure=float(np.mean([v['response_departure'] for v in units])),
     relative_actual_margin_target_error=float(np.linalg.norm(ach-target)/max(np.linalg.norm(target),1e-12)),condition_step_norm=None if name=='independent_response_oracle' else float(np.linalg.norm(candidates[name]-base)))
  rawproof['linear_response_departure']=float(raw@G@raw);metproof['linear_response_departure']=float(metric@G@metric)
  rows.append(dict(query=qi,array_row=int(bank['T_rows'][qi]),correct_teacher_index=qi%10,summary=summary,minimum_condition_proof=rawproof,minimum_response_proof=metproof,unit_audits=case_errors,
   oracle_units=[v['oracle_proof'] for v in per],actual_units=evaluation))
  for name,value in dict(G=G,B=B,target=target,raw_step=raw,raw_lambda=rawlam,metric_step=metric,metric_lambda=metlam,base=base,CN=cn).items():arrays[f'q{qi}__{name}']=value
  K.write(out/'progress.json',dict(completed_queries=len(rows),target_queries=2,updates=0,new_GPU=False));guard()
 np.savez_compressed(out/'normal_equations.npz',**arrays);guard()
 done=dict(status='COMPLETE_TWO_QUERY_SHARED_CONDITION_FEASIBILITY_NOT_EFFICACY',rows=rows,finite_difference_checks=errors,inputs=q['inputs'],h94=h94,normal_equations=K.ref(out/'normal_equations.npz'),
  original_conditions_unchanged=K.ref(q['inputs']['conditions']['path'])==q['inputs']['conditions'],new_fits=0,new_GPU=False,new_images=0,seconds=time.monotonic()-start,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,
  limits='Two fixed engineering queries, two layers and two times per query. Continuous FP64 function of original FP16 weight bytes, not derivative of FP16 cast. Privileged teacher contrasts and true concept only in diagnostic. Linear optimum is not nonlinear optimum, EEG method, heldout benefit or novelty.')
 K.write(out/'completion.json',done);print(json.dumps(dict(status=done['status'],seconds=done['seconds'],summaries=[r['summary'] for r in rows])))
except BaseException as exc:K.write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
