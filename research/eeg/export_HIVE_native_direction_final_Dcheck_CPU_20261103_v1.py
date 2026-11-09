"""Export fixed final encoders and an unchanged native encoder through identical CPU math."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import brainhive_native_direction_image_contract_20261103_v1 as I
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();started=time.monotonic();s,train,e,old,base,G,H,C=I.setup(a.protocol);dep=I.dependency(s,train,C)
 if dep['status']!='READY_ALL30_FIXED_FINALS_AND_ACTUAL_OWNER_EXIT':print(json.dumps(dep),flush=True);return
 before=I.resources(s,train,e,C,True,True);out=Path(s['output_root'])/'conditions';assert not out.exists();out.mkdir(parents=True)
 try:
  import torch
  from brainhive_native_models_v1 import classes
  from brainhive_bounded_component_v1 import BoundedConditionAdapter
  torch.set_num_threads(2);assert not torch.cuda.is_initialized();asset=I.assets(s,base,C);panel=asset['panel'];ids=[v['array_row'] for v in panel]
  mp=next(v for v in e['full_models'] if v['subject']=='sub-01');rp=next(v for v in e['R4_payloads'] if v['subject']=='sub-01');bind=C.read(C.checked(e['actual_bindings']))
  raw=torch.load(C.checked(rp['payload']),weights_only=True,map_location='cpu',mmap=True);x=raw[ids][:,bind['selected_indices'],:].reshape(400,4250).contiguous();del raw;assert torch.isfinite(x).all()
  initial=torch.load(C.checked(mp['model']),weights_only=True,map_location='cpu')['model'];gamma=initial['block.layers.0.norm.weight'];beta=initial['block.layers.0.norm.bias'];scope=classes()
  head=BoundedConditionAdapter().eval().requires_grad_(False);head.load_state_dict(torch.load(C.checked(base['ordinary']),weights_only=True,map_location='cpu')['model'],strict=True)
  native_cache=torch.load(C.checked(mp['condition_cache']),weights_only=True,map_location='cpu',mmap=True);assert native_cache['model_sha256']==mp['model']['sha256'];positions={int(row):i for i,row in enumerate(native_cache['array_rows'].tolist())};cached=native_cache['condition'][[positions[row] for row in ids]].clone();del native_cache
  inputs=np.empty((400,4,1024),np.float32);conditions=np.empty_like(inputs);sources=[];cache_drift=None
  for j,arm in enumerate(I.ARMS):
   pin=mp['model'] if arm=='reencoded_ordinary' else dep['models']['sub-01/'+arm];saved=torch.load(C.checked(pin),weights_only=True,map_location='cpu');model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True)
   for key in ('block.layers.0.norm.weight','block.layers.0.norm.bias'):assert torch.equal(saved['model'][key],initial[key])
   if j:assert saved['epoch']==3 and saved['updates']==2481 and saved['subject']=='sub-01' and saved['arm']==arm and saved['protocol']==s['training_protocol']
   with torch.inference_mode():
    for lo in range(0,400,128):
     c=model(x[lo:lo+128]);inputs[lo:lo+len(c),j]=c.numpy();conditions[lo:lo+len(c),j]=head(c,gamma.expand(len(c),-1),beta.expand(len(c),-1))[0].numpy()
   if j==0:
    difference=inputs[:,0].astype(np.float64)-cached.numpy();cache_drift=dict(max_abs=float(abs(difference).max()),relative_L2=float(np.linalg.norm(difference)/np.linalg.norm(cached.numpy().astype(np.float64))),exact_FP16_elements=bool(np.array_equal(inputs[:,0].astype(np.float16),cached.numpy().astype(np.float16))),
     scope='Unchanged native model CPU FP32 versus saved original CUDA condition cache. Fresh baseline images remain a separate arm even when numerical differences are small.')
    assert np.allclose(inputs[:,0],cached.numpy(),atol=1e-4,rtol=1e-4),cache_drift
   sources.append(dict(arm=arm,model=pin,raw_R4=rp['payload'],ordinary=base['ordinary'],subject='sub-01',normal_inference_current_query_only=True));del model,saved
  assert np.isfinite(inputs).all() and np.isfinite(conditions).all();np.save(out/'native_conditions_before_ordinary.npy',inputs,allow_pickle=False);np.save(out/'conditions_after_old_ordinary.npy',conditions,allow_pickle=False)
  done=dict(status='COMPLETE_ALL400_FOUR_ARM_FIXED_FINAL_CPU_EEG_CONDITIONS',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),protocol=C.ref(a.protocol),source=C.ref(__file__),training_completion=dep['source_completion'],arms=list(I.ARMS),queries=panel,
   inputs=C.ref(out/'native_conditions_before_ordinary.npy'),conditions=C.ref(out/'conditions_after_old_ordinary.npy'),models=sources,raw_R4=rp['payload'],ordinary=base['ordinary'],original_cache=mp['condition_cache'],cache_drift=cache_drift,
   condition_shape=[400,4,1024],condition_coordinates='Original native BrainMLP output affine -> exact unchanged oldordinary strength1 radius0.25 -> native A input.',
   encoder_math='CPU FP32 eval with frozen original LN affine; identical for fresh baseline and three trained models. Original cached CUDA baseline also reported separately.',
   target_F_or_images_read_for_conditions=False,Dcheck_scores_used_for_choices=False,new_training_updates=0,new_images=0,NEW_contacted=False,seconds=time.monotonic()-started,resources_before=before,resources_after=I.resources(s,train,e,C))
  assert not torch.cuda.is_initialized() and time.monotonic()-started<s['CPU_stage_seconds'];G.write(out/'completion.json',done);print(json.dumps(done),flush=True)
 except BaseException as error:G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,no_automatic_retry=True));raise
if __name__=='__main__':main()
