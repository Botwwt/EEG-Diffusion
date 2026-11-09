"""Independent native linear/GELU/LN and ordinary algebra; exhaustive identity/pixel/noise audit."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,io,json,time,traceback,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import brainhive_native_direction_image_contract_20261103_v1 as I
def recompute_native_and_ordinary(eeg,state,ordinary):
 import torch
 import torch.nn.functional as F
 gamma=state['block.layers.0.norm.weight'];beta=state['block.layers.0.norm.bias']
 with torch.inference_mode():
  x=F.linear(eeg,state['proj_in.weight'],state['proj_in.bias']);x=x+F.linear(F.gelu(F.linear(x,state['block.layers.0.ff.w1.weight'])),state['block.layers.0.ff.w2.weight']);c=F.layer_norm(x,(1024,),gamma,beta,1e-6)
  u=(c-beta)/gamma;hidden=F.gelu(F.linear(u,ordinary['trunk.0.weight'],ordinary['trunk.0.bias']));d=F.linear(hidden,ordinary['direction.weight'],ordinary['direction.bias']);d=d-d.mean(-1,keepdim=True);norm=u.norm(dim=-1,keepdim=True);r2=norm.square();d=d-(d*u).sum(-1,keepdim=True)/r2*u
  delta=.25*r2.sqrt()*d/(r2+d.square().sum(-1,keepdim=True)).sqrt();scale=norm/(u+delta).norm(dim=-1,keepdim=True);corrected=c+gamma*((scale-1)*u+scale*delta)
 return c,corrected
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--stage',type=int,choices=(80,400),required=True);a=p.parse_args();started=time.monotonic();s,train,e,old,base,G,H,C=I.setup(a.protocol)
 dep=I.dependency(s,train,C)
 if dep['status']!='READY_ALL30_FIXED_FINALS_AND_ACTUAL_OWNER_EXIT':print(json.dumps(dep));return
 root=Path(s['output_root']);out=root/f'audit{a.stage}';assert not out.exists();out.mkdir();before=I.resources(s,train,e,C)
 try:
  import torch
  import torch.nn.functional as F
  torch.set_num_threads(2);assert not torch.cuda.is_initialized();asset=I.assets(s,base,C);cond=C.read(root/'conditions/completion.json');bank=np.load(C.checked(cond['conditions']),allow_pickle=False);encoder_values=np.load(C.checked(cond['inputs']),allow_pickle=False)
  assert cond['arms']==list(I.ARMS) and cond['protocol']==C.ref(a.protocol) and not cond['Dcheck_scores_used_for_choices'] and not cond['target_F_or_images_read_for_conditions']
  ids=[q['array_row'] for q in asset['panel']];bind=C.read(C.checked(e['actual_bindings']));raw=torch.load(C.checked(cond['raw_R4']),weights_only=True,map_location='cpu',mmap=True);eeg=raw[ids][:,bind['selected_indices']].reshape(400,4250).contiguous();del raw
  ordinary=torch.load(C.checked(base['ordinary']),weights_only=True,map_location='cpu')['model'];mp=next(v for v in e['full_models'] if v['subject']=='sub-01');parent=torch.load(C.checked(mp['model']),weights_only=True,map_location='cpu')['model'];gamma=parent['block.layers.0.norm.weight'];beta=parent['block.layers.0.norm.bias'];max_encoder=0.;max_condition=0.
  for j,arm in enumerate(I.ARMS):
   pin=cond['models'][j]['model'];assert cond['models'][j]['arm']==arm and pin==(mp['model'] if j==0 else dep['models']['sub-01/'+arm]);state=torch.load(C.checked(pin),weights_only=True,map_location='cpu')['model']
   assert set(state)=={'proj_in.weight','proj_in.bias','block.layers.0.ff.w1.weight','block.layers.0.ff.w2.weight','block.layers.0.norm.weight','block.layers.0.norm.bias'}
   for key in ('block.layers.0.norm.weight','block.layers.0.norm.bias'):assert torch.equal(state[key],parent[key])
   for lo in range(0,400,128):
    c,corrected=recompute_native_and_ordinary(eeg[lo:lo+128],state,ordinary)
    expected=torch.from_numpy(encoder_values[lo:lo+len(c),j]);expected2=torch.from_numpy(bank[lo:lo+len(c),j]);max_encoder=max(max_encoder,float((c-expected).abs().max()));max_condition=max(max_condition,float((corrected-expected2).abs().max()))
    assert torch.allclose(c,expected,atol=1e-6,rtol=1e-6) and torch.allclose(corrected,expected2,atol=1e-6,rtol=1e-6)
  records=[];segments=[]
  for lo in range(0,a.stage,80):
   path=root/'generation'/f'{lo:03d}_{lo+80:03d}'/'completion.json';done=C.read(path);assert done['status']=='COMPLETE_FIXED80_FOUR_ARM_NATIVE_FINAL_IMAGE_SEGMENT' and done['protocol']==C.ref(a.protocol) and done['range']==[lo,lo+80]
   part=[json.loads(v) for v in C.checked(done['evidence']['manifest']).read_text().splitlines()];assert len(part)==320;records+=part;segments.append(C.ref(path));features=np.load(C.checked(done['evidence']['features']),allow_pickle=False);assert features.shape==(80,4,768) and np.isfinite(features).all()
  assert len(records)==a.stage*4;paths=set();inodes=set();oldinodes=set();pixel_groups={};spatial_constant=0;archives={};targetshas=[]
  try:
   for i,q in enumerate(asset['panel'][:a.stage]):
    meta=asset['metadata'][i]
    if meta['image_zip'] not in archives:archives[meta['image_zip']]=zipfile.ZipFile(meta['image_zip'])
    data=archives[meta['image_zip']].read(meta['image_member']);assert len(data)==int(meta['image_bytes']) and hashlib.sha256(data).hexdigest()==meta['image_sha256'];targetshas.append(meta['image_sha256'])
    for row in asset['old_manifest'][3*i:3*i+3]:
     pth=Path(row['generated_path']);st=pth.stat();oldinodes.add((st.st_dev,st.st_ino));assert C.sha(pth)==row['generated_sha256']
     with Image.open(pth) as im:im.load();pix=np.asarray(im.convert('RGB'));assert im.size==(512,512) and hashlib.sha256(pix.tobytes()).hexdigest()==row['generated_pixel_sha256']
    for j,arm in enumerate(I.ARMS):
     row=records[4*i+j]
     for key,value in q.items():assert row[key]==value
     assert row['arm']==arm and row['query_index']==i and row['EEG_source_query_id']==q['query_id'] and row['EEG_source_array_row']==q['array_row']
     assert row['UNet_NFE']==4 and row['native_IP_K_calls']==280 and row['native_IP_V_calls']==280 and row['extra_UNet_evaluations']==0 and row['normal_inference_only_current_query_EEG'] and not row['normal_inference_target_access']
     assert row['encoder_model']==cond['models'][j]['model'] and row['ordinary']==base['ordinary'] and row['condition_source']==cond['conditions'] and row['evaluation_target_SHA256']==meta['image_sha256']
     for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert row[key]==asset['old_manifest'][3*i][key]
     assert row['condition_sha256']==hashlib.sha256(bank[i,j:j+1].tobytes()).hexdigest() and row['condition_FP16_sha256']==hashlib.sha256(bank[i,j:j+1].astype(np.float16).tobytes()).hexdigest()
     pth=Path(row['generated_path']);assert pth.resolve().is_relative_to(root.resolve()) and str(pth) not in paths;paths.add(str(pth));st=pth.stat();inode=(st.st_dev,st.st_ino);assert inode not in inodes;inodes.add(inode);assert C.sha(pth)==row['generated_sha256']
     with Image.open(pth) as im:im.load();pix=np.asarray(im.convert('RGB'));assert im.size==(512,512) and hashlib.sha256(pix.tobytes()).hexdigest()==row['generated_pixel_sha256']
     spatial_constant+=int(np.max(np.std(pix.astype(np.float64),axis=(0,1)))<1e-12);pixel_groups.setdefault(row['generated_pixel_sha256'],[]).append((i,arm))
    if (i+1)%80==0:assert time.monotonic()-started<s['CPU_stage_seconds']-30;I.resources(s,train,e,C)
  finally:
   for archive in archives.values():archive.close()
  assert not inodes.intersection(oldinodes) and not torch.cuda.is_initialized();complete=dict(status='PASS_ALL_FIXED_NATIVE_EEG_IMAGE_IDENTITIES_CONDITIONS_NOISE_PIXELS_BEFORE_SCORE',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),protocol=C.ref(a.protocol),source=C.ref(__file__),queries=a.stage,
   conditions=C.ref(root/'conditions/completion.json'),segments=segments,all400x4_native_encoder_and_ordinary_algebra_checked=True,maximum_encoder_abs_error=max_encoder,maximum_condition_abs_error=max_condition,
   old_images_SHA_and_pixels_checked=a.stage*3,new_images_SHA_and_pixels_checked=a.stage*4,targets_SHA_checked=a.stage,all_target_SHA256=targetshas,
   exact_duplicate_new_pixel_groups=[v for v in pixel_groups.values() if len(v)>1],spatially_constant_new_images=spatial_constant,no_baseline_file_or_inode_reuse=True,
   no_exclusions_from_duplicate_or_constant_flags=True,full_registered_denominators_retained=True,metrics_not_read_for_input_audit=True,new_GPU_jobs=0,NEW_contacted=False,seconds=time.monotonic()-started,
   resources_before=before,resources_after=I.resources(s,train,e,C))
  assert time.monotonic()-started<s['CPU_stage_seconds'];G.write(out/'inputs.json',complete);print(json.dumps(complete),flush=True)
 except BaseException as error:G.write(out/'inputs_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,full_registered_denominator=a.stage*4,no_automatic_retry=True));raise
if __name__=='__main__':main()
