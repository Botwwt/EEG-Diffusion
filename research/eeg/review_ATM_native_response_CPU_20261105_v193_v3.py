"""Independent FP64 actual-Q cases and complete response-table interpretation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,sys,time,traceback
from pathlib import Path
import numpy as np
q=json.loads(Path(sys.argv[1]).read_bytes());root=Path(q['parent_root']);out=Path(q['output_root']);started=time.monotonic()
sys.path.insert(0,str(root/'code'));import atm_fixed_projection_contract_20261105_v190 as C
c=C.setup(root/'code/policy.json');K=c['K'];C.resources(c)
assert K.ref(__file__)==q['sources']['review']
sys.path.insert(0,str(Path(q['original_GPU_sources']['core']['path']).parent))
from atm_native_response_geometry_20261105_v193_v2 import numpy_attention,numpy_measures,summarize
try:
 job=K.read(c['repo']/'audit'/(q['job_name']+'_job.json'));assert job['exit_code']==1 and not K.process(job['pid'])['live'] and not K.process(job['wrapper_pid'])['live']
 for pin in q['original_GPU_sources'].values():K.checked(pin)
 failure=K.read(K.checked(q['artifacts']['failure']))
 assert "conditions=d['conditions']" in failure['traceback'] and "line 80" in failure['traceback'] and failure['error'].startswith('IndexError(')
 progress=K.read(K.checked(q['artifacts']['progress']));assert progress['complete_queries']==80 and progress['actual_attention_units']==22400
 done=q['artifacts'];trajectory=K.read(K.checked(done['trajectories']));assert len(trajectory)==80
 for i,v in enumerate(trajectory):
  assert v['query']==i and v['array_row']==q['T80_array_rows'][i] and v['seed']==1729+v['array_row'] and v['correct_teacher_index']==i%10
  assert v['teacher_rows']==q['T80_array_rows'][(i//10)*10:(i//10+1)*10]
  assert len(v['noise']['initial'])==1 and len(v['noise']['ancestral'])==4
 assert K.read(K.checked(done['layouts']))['all44800_KV_values_bitexact_after_reshape']
 K.write(out/'array_recovery_receipt.json',dict(status='COMPLETE_ARRAYS_RECOVERED_FROM_FAILED_FINAL_RECEIPT',GPU_exit_code=1,original_failure=q['artifacts']['failure'],all22400_unit_assertions_passed_before_save=True,post_loop_host_weight_assertion_passed_inferred_from_exact_failure_line=True,maximum_GPU_CPU_memory_bytes=None,unsaved_native_error_maximum=None,no_new_GPU=True,no_repeated_forward=True,artifacts=done))
 z=np.load(K.checked(done['responses']));cases=np.load(K.checked(done['cases']));maxerr={};rows=[]
 # Algebra fixtures independently exercise spatial projection and risk identity.
 rng=np.random.default_rng(1729)
 for constant in [False,True]:
  response=rng.normal(size=(14,2,16,5))
  if constant:response[:]=response[:,:,:1]
  d,s,en,al=numpy_measures(response,3)
  assert np.max(abs(d[:,:,0]-d[:,:,1]-d[:,:,2]))<1e-12
  assert np.max(abs(d[1:,3]-d[:1,3]-en+2*al))<1e-12
  if constant:assert np.max(abs(d[:,:,2]))<1e-28
 for label in sorted({k.split('__')[0] for k in cases.files}):
  x={n:cases[label+'__'+n].astype(np.float64) for n in ['Q','K','V','native','distance','scale','energy','align']}
  resp=numpy_attention(x['Q'],x['K'],x['V']);query=int(label.split('_')[0][1:]);actual=numpy_measures(resp,query%10)
  errors={}
  for key,value in zip(['distance','scale','energy','align'],actual):
   err=float(abs(value-x[key]).max());bound=2e-5*max(1.,float(abs(value).max()));assert err<bound,(label,key,err,bound)
   errors[key]=err;maxerr[key]=max(maxerr.get(key,0.),err)
  err=float(abs(resp[:1]-x['native']).max()/max(1.,float(abs(resp[:1]).max())));assert err<.005;errors['native_SDPA_scaled']=err
  flip=int((actual[0][:,:,0].argmin(1)!=x['distance'][:,:,0].argmin(1)).sum())
  rows.append(dict(label=label,errors=errors,FP32_FP64_full_distance_rank_flips=flip))
 assert len(rows)==8
 d,s,e,al=[z[k].astype(np.float64) for k in ['distance','scale','energy','alignment']];assert np.all(z['coverage']==1)
 decomp=float(abs(d[:,:,:,:,:,0]-d[:,:,:,:,:,1]-d[:,:,:,:,:,2]).max());assert decomp<2e-5*max(1.,float(d.max()))
 correct=np.arange(80)%10;selected=np.stack([d[i,:,:,1:,correct[i]]-d[i,:,:,:1,correct[i]] for i in range(80)])
 risk=float(abs(selected-e+2*al).max());assert risk<2e-5*max(1.,float(d.max()))
 result,per,draw,scores=summarize(d,s,e,al)
 np.savez_compressed(out/'per_query.npz',**{arm+'__'+key:val for arm,metrics in per.items() for key,val in metrics.items()},bootstrap=draw,scores=scores)
 # Independent order of averaging, directly over 280 units per query.
 slow=np.array([sum(d[i,t,l]/np.maximum(s[i,t,l],1e-12)[None,None,:] for t in range(4) for l in range(70))/280 for i in range(80)])
 aggregate_error=float(abs(slow-scores).max());assert aggregate_error<1e-10*max(1.,float(abs(scores).max()))
 paths=[K.checked(done[k]) for k in ['conditions','responses','cases','tokens','trajectories']]+[out/'per_query.npz']
 result.update(status='PASS_ALL22400_RESPONSE_UNITS_EIGHT_CPU64_CASES_T80_READOUT',case_audits=rows,max_case_errors=maxerr,
  full_distance_decomposition_error=decomp,full_risk_identity_error=risk,independent_aggregation_error=aggregate_error,per_query=K.ref(out/'per_query.npz'),
  registered_contrasts=6,registered_metrics=9,descriptive_intervals=54,all_artifact_pins=[K.ref(p) for p in paths],CPU64_scope='All320 EEG forwards repeated; 8 actual QKV units independently recomputed, NOT all22400 neural units',
  no_generated_image_effect_scoring=True,new_updates=0,seconds=time.monotonic()-started)
 assert K.allocated(out)<q['maximum_output_bytes'];C.resources(c);K.write(out/'review_completion.json',result)
 print(json.dumps(dict(status=result['status'],means=result['means'],max_case_errors=maxerr,seconds=result['seconds'])))
except BaseException as exc:
 K.write(out/'review_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
