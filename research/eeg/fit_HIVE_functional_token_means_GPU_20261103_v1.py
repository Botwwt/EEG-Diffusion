"""Fit only training-token means; all pretrained and ordinary weights frozen."""
import argparse,datetime,gc,hashlib,json,time,traceback
from pathlib import Path
import brainhive_functional_mean_contract_20261103_v1 as G

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--job-name',required=True);p.add_argument('--max-seconds',type=int,default=300);a=p.parse_args()
 start=time.monotonic();s=G.setup(a.protocol);owner=G.wrapper(s,a.job_name);before=G.resources(s,start=True);base=G.rss()
 import numpy as np,torch
 import brainhive_native_models_v1 as N
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 from brainhive_component_data_v1 import load_training
 from brainhive_functional_mean_calibration_CPU_20261103_v1 import calibration_fields
 out=Path(s['output_root'])/'statistics';assert not out.exists();out.mkdir(parents=True)
 stage='INPUTS'
 try:
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  data=load_training('cpu');assert data['condition'].shape==(132300,1024) and data['weight'].shape==(20,1024) and data['target'].shape==(13230,1024)
  contract=G.read(s['input_pins']['component_contract']['path']);split=G.read(s['input_pins']['split']['path']);T=set(split['image_rows']['T']);provenance=[]
  for sub in contract['subjects']:
   for source in sub['sources']:
    if source['role'] not in ('oof0','oof1'):continue
    reg=G.read(G.checked(dict(path=source['parent_registration_path'],sha256=source['parent_registration_sha256'])))
    held={r['array_row'] for r in source['groups']['component_train']};trained=set(reg['training_array_rows'])
    assert held.isdisjoint(trained) and held|trained==T and set(reg['excluded_oof_rows'])==held
    provenance.append(dict(subject=sub['subject'],role=source['role'],held=len(held),trained=len(trained),cache=dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256']),model=dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256'])))
  assert len(provenance)==20 and sum(v['held'] for v in provenance)==132300
  saved=torch.load(G.checked(s['ordinary']),weights_only=True,map_location='cpu');head=BoundedConditionAdapter().eval().requires_grad_(False);head.load_state_dict(saved['model'],strict=True)
  G.resources(s,start=True);assert not torch.cuda.is_initialized();torch.cuda.set_per_process_memory_fraction(min(1.,(20<<30)/torch.cuda.get_device_properties(0).total_memory));torch.cuda.reset_peak_memory_stats()
  models=N.prior_models('cuda',include_attention=False);A=models['fusion_encoder'].adapter.half();P=models['proj'].half();head=head.cuda();del models;gc.collect()
  inputs={k:v.cuda() if torch.is_tensor(v) else v for k,v in data.items()};del data;gc.collect()
  target=inputs['target'];real_tokens=[]
  stage='VISUAL_TOKENS'
  with torch.inference_mode():
   for lo in range(0,13230,256):
    G.budget(s,start,a.max_seconds,base,torch);real_tokens.append(P(A(target[lo:lo+256].half())))
  real=torch.cat(real_tokens);del real_tokens;assert real.shape==(13230,4,2048) and real.dtype==torch.float16
  sums={'visual':np.zeros((4,2048),np.float64),'native':np.zeros((4,2048),np.float64),'ordinary':np.zeros((4,2048),np.float64)}
  squares={key:0. for key in sums};loss={'native':0.,'ordinary':0.};batch_sums={key:[] for key in sums};counts={key:[] for key in sums};sample={}
  for lo in range(0,13230,256):
   v=real[lo:lo+256].cpu().numpy().astype(np.float64);total=v.sum(0);sums['visual']+=total;squares['visual']+=float(np.square(v).sum());batch_sums['visual'].append(total);counts['visual'].append(len(v))
  stage='EEG_AND_ORDINARY_TOKENS'
  with torch.inference_mode():
   for lo in range(0,132300,256):
    G.budget(s,start,a.max_seconds,base,torch,refresh=(lo%(256*32)==0))
    c=inputs['condition'][lo:lo+256];ids=inputs['affine_ids'][lo:lo+256];w=inputs['weight'][ids];b=inputs['bias'][ids]
    corrected=head(c,w,b)[0];native=P(A(c.half()));ordinary=P(A(corrected.half()))
    truth=real[torch.arange(lo,lo+len(c),device='cuda')%13230].cpu().numpy().astype(np.float64)
    for key,tokens in (('native',native),('ordinary',ordinary)):
     v=tokens.cpu().numpy().astype(np.float64);total=v.sum(0);sums[key]+=total;squares[key]+=float(np.square(v).sum());loss[key]+=float(np.square(v-truth).sum());batch_sums[key].append(total);counts[key].append(len(v))
    if lo==0:sample=dict(c=c[:16].cpu().numpy(),weight=w[:16].cpu().numpy(),bias=b[:16].cpu().numpy(),corrected=corrected[:16].cpu().numpy(),native_tokens=native[:16].cpu().numpy(),ordinary_tokens=ordinary[:16].cpu().numpy(),visual_tokens=real[:16].cpu().numpy())
  assert sum(counts['visual'])==13230 and sum(counts['native'])==sum(counts['ordinary'])==132300
  means={key:sums[key]/sum(counts[key]) for key in sums};fields={key:calibration_fields(means['visual'],means[key]) for key in ('native','ordinary')}
  np.savez(out/'token_means_and_fields.npz',**{key+'_mean':v for key,v in means.items()},**{key+'_'+name:v for key,field in fields.items() for name,v in field.items() if name!='proof'})
  np.savez_compressed(out/'all_batch_sufficient_statistics.npz',**{key+'_sums':np.stack(batch_sums[key]) for key in sums},**{key+'_counts':np.asarray(counts[key],np.int64) for key in sums})
  np.savez(out/'fixed16_engineering_samples.npz',**sample)
  fit={}
  for key in ('native','ordinary'):
   before_loss=loss[key]/(132300*4*2048);d=fields[key]['shared_delta'];full=fields[key]['full_delta'];alpha=fields[key]['proof']['full_match_factor']
   fit[key]=dict(before_T_token_MSE=before_loss,shared_after_LS_MSE=before_loss-float(d@d)/2048,
    matched_full_after_LS_MSE=before_loss-(2*alpha-alpha*alpha)*float(np.square(full).sum())/(4*2048),
    pointwise_token_variance_trace=squares[key]/132300-float(np.square(means[key]).sum()),field=fields[key]['proof'])
  G.budget(s,start,a.max_seconds,base,torch,refresh=True)
  assert all(not p.requires_grad for m in (A,P,head) for p in m.parameters()) and G.ref(s['ordinary']['path'])['sha256']==s['ordinary']['sha256']
  complete=dict(status='COMPLETE_ALL10_REAL_OOF_T_TOKEN_MEAN_STATISTICS_NO_SGD',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=G.ref(a.protocol),source=G.ref(__file__),wrapper=owner,T_queries=132300,T_images=13230,OOF_roles=20,visual_token_images_encoded=13230,EEG_token_queries_encoded=132300,ordinary_token_queries_encoded=132300,
   provenance=provenance,fields=G.ref(out/'token_means_and_fields.npz'),batch_statistics=G.ref(out/'all_batch_sufficient_statistics.npz'),samples=G.ref(out/'fixed16_engineering_samples.npz'),counts={k:sum(v) for k,v in counts.items()},fit=fit,precision='Native A/P FP16; ordinary head FP32; actual output sums accumulated CPUFP64; batch256 is not claimed bit-identical to singleton inference.',
   all_weights_frozen=True,SGD_updates=0,new_EEG_backbone_updates=0,new_images=0,new_UNet_calls=0,normal_inference_target_access=False,ordinary_unchanged=True,NEW_contacted=False,
   seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-base,resources_before=before,resources_after=G.resources(s))
  G.write(out/'completion.json',complete);print(json.dumps(complete),flush=True)
 except BaseException as error:G.write(out/'failure.json',dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start,no_automatic_retry=True));raise
if __name__=='__main__':main()
