"""Independent whole saved-term arithmetic and declared precision decision."""
import argparse,datetime,importlib.util,time
from pathlib import Path
import numpy as np
import VAR_declared_FP16_contract_CPU_20261102_v1 as C
def posterior(x):
 x=np.asarray(x,np.float64);x=x-x.max(-1,keepdims=True);y=np.exp(x);return y/y.sum(-1,keepdims=True)
def run(protocol):
 q=C.setup(protocol);n,_=C.dependency(q);c,oldq=C.previous();p=C.OUT/'profile';done=C.read(p/'completion.json')
 assert done['status']=='COMPLETE_DECLARED_FP16_FULL_T8_K1000_680_REFERENCE_AND_GENERATION_PROFILE'
 assert done['protocol']==C.ref(protocol) and done['class_batches']==128 and done['all_weights_byte_unchanged']
 for pin in done['files']:C.checked(pin)
 a=np.load(p/'true_token_logprobs.npy',mmap_mode='r');b=np.load(oldq['reference']['token_logprobs']['path'],mmap_mode='r');ll=np.load(p/'scale_log_likelihoods.npy');old=np.load(oldq['reference']['scale_likelihoods']['path'])
 assert a.shape==b.shape==(8,1000,680) and a.dtype==b.dtype==np.float32
 assert ll.shape==old.shape==(8,1000,10) and ll.dtype==np.float64
 assert np.array_equal(np.load(p/'true_codes.npy'),np.load(oldq['reference']['true_codes']['path']))
 assert np.array_equal(np.load(p/'coverage.npy'),np.ones((8,1000),np.uint8))
 mx=0.;means=np.zeros((8,1000));terms=0;bounds=np.cumsum([0]+[x*x for x in q['scales']])
 for k,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
  summed=np.zeros((8,1000))
  for token in range(lo,hi):
   v=a[:,:,token];assert np.isfinite(v).all() and (v<=0).all();summed+=v;err=np.abs(v.astype(np.float64)-b[:,:,token]);mx=max(mx,float(err.max()));means+=err/680;terms+=v.size
  assert np.array_equal(summed,ll[:,:,k])
 assert terms==5440000 and mx==done['maximum_target_logprob_difference']
 sp=importlib.util.spec_from_file_location('complete_frozen_HMM_reference',n.CODE/'EEG_multiscale_native_class_DRAFT_CPU_20261101_v1.py');H=importlib.util.module_from_spec(sp);sp.loader.exec_module(H)
 tv=[];arg=[]
 for i in range(8):
  row=[]
  for rho in (0.,.5,1.):
   x=H.inference(np.zeros((10,1000)),np.full(9,rho),ll[i].T)['class_posterior'];y=H.inference(np.zeros((10,1000)),np.full(9,rho),old[i].T)['class_posterior'];row.append((np.abs(x-y).sum(1)*.5).tolist())
  tv.append(row);arg.append(bool(ll[i].sum(1).argmax()==old[i].sum(1).argmax()))
 direct=posterior(ll.sum(2));direct_old=posterior(old.sum(2));directtv=np.abs(direct-direct_old).sum(1)*.5
 prefix=np.load(p/'all_T8_true_prefix_class0_logprobs.npy');pe=float(np.abs(prefix-a[:,0,:]).max());assert pe==done['maximum_full_vs_prefix_logprob_difference']
 with np.load(p/'top0_codes.npz') as f:
  for k,side in enumerate(q['scales']):
   v,w=f['first_'+str(k)],f['second_'+str(k)];assert v.shape==w.shape==(1,side*side) and np.array_equal(v,w) and ((v>=0)&(v<4096)).all()
 from PIL import Image
 for name in ('first_top0','second_top0'):
  v=np.load(p/(name+'_pixels.npy'));assert v.shape==(3,256,256) and v.dtype==np.float32 and np.isfinite(v).all()
  with Image.open(p/(name+'.png')) as im:assert np.array_equal(np.asarray(im.convert('RGB')),(v.transpose(1,2,0).clip(0,1)*255).round().astype(np.uint8))
 assert np.array_equal(np.load(p/'first_top0_pixels.npy'),np.load(p/'second_top0_pixels.npy'))
 g=q['fidelity_guard'];max_tv=float(np.max(tv));decision=bool(mx<=g['max_token_logprob_error'] and means.max()<=g['max_class_mean_abs_token_error'] and max_tv<=g['max_HMM_posterior_TV'] and all(arg) and pe<=g['max_full_vs_prefix_logprob_error'])
 C.write(C.OUT/'independent_CPU_audit.json',dict(status='COMPLETE_ALL_DECLARED_FP16_REFERENCE_COMPARISONS_NOT_EEG_EFFICACY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=C.ref(protocol),all_new_token_terms=terms,all80000_scale_sums_exact=True,max_token_logprob_error=mx,max_class_mean_abs_token_error=float(means.max()),all8_global_argmax_same=all(arg),per_query_argmax_same=arg,all_query_rho_stage_posterior_TV=tv,maximum_HMM_TV=max_tv,independent_direct_global_TV=directtv.tolist(),full_vs_prefix_logprob_error=pe,top0_repeat_codes=680,top0_repeat_pixels=196608,precision_fidelity_guard=g,predeclared_precision_fidelity_support=decision,reference8_over_new_speed=oldq['reference_loop_seconds']/done['loop_seconds'],full13230_same_loop_projection_hours=done['loop_seconds']/8*13230/3600,raw_complete_evidence_retained_on_OLD=True,new_EEG_fit_or_images=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
