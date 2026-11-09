"""Independent full actual-prefix NLL/support/identity audit; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
M=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1');R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
ROOT=M/'AR_Dfit200_actual_prefix_20261029_v1/result_v1';OUT=ROOT.parent/'independent_CPU_audit_v1';SIDES=[1,2,3,4,5,6,8,10,13,16];WIDTHS=np.square(SIDES)
def ref(p):
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def check(x):p=Path(x['path']);assert ref(p)==x;return p
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def nll64(logits,targets):
 a=logits.astype(np.float64);maximum=a.max(-1);return maximum+np.log(np.exp(a-maximum[...,None]).sum(-1))-np.take_along_axis(a,targets[...,None],-1)[...,0]
def run():
 started=time.monotonic();assert not OUT.exists();OUT.mkdir();done=json.loads((ROOT/'completion.json').read_bytes())
 assert done['status']=='COMPLETE_FIXED_NATIVE_DFIT200_ACTUAL_PREFIX_8CONDITIONS_AND_ALL_PRESERVED_GENERATIONS' and done['queries']==200 and done['concepts']==20 and done['images']==200 and done['engineering_same_seed_replays']==2 and done['total_generate_calls']==202 and done['new_optimizer_updates']==0
 assert done['all_VAR_VQ_weights_unchanged'] is True and done['original_Dfit_5condition_result_unchanged'] is True and done['main_training_or_Dcheck_image_gate_changed'] is False
 specpath=check(done['spec']);assert ref(specpath)['sha256']=='1f38c74b030f0692449168a0b0cc5dd1c1a507408ee0c471990cc323933ddc52';s=json.loads(specpath.read_bytes())
 for pin in s['source_pins']:check(pin)
 rows=json.loads(check(s['queries']).read_bytes())['rows'];assert len(rows)==200 and len(set(r['stimulus_id'] for r in rows))==200
 concepts=sorted(set(r['concept_id'] for r in rows));groups=[[i for i,r in enumerate(rows) if r['concept_id']==c] for c in concepts];assert len(groups)==20 and all(len(x)==10 for x in groups)
 c=np.load(check(s['c']),allow_pickle=False);target=np.load(check(s['VQ_targets']),allow_pickle=False);sampled=np.load(check(done['sampled_codes']),allow_pickle=False)
 assert c.shape==(200,1024) and c.dtype==np.float32 and target.shape==sampled.shape==(200,680) and target.dtype==sampled.dtype==np.int16 and sampled.min()>=0 and sampled.max()<4096
 old=json.loads(check(s['closed_teacher_diagnostic']).read_bytes());old_token=np.load(check(old['all_token_NLL']),allow_pickle=False)
 token=np.load(check(done['all_token_NLL']),allow_pickle=False);scale=np.load(check(done['all_scale_NLL']),allow_pickle=False);support=np.load(check(done['target_top600_support']),allow_pickle=False)
 assert token.shape==(200,8,680) and token.dtype==np.float32 and np.isfinite(token).all() and token.min()>=-1e-6
 assert scale.shape==(200,8,10) and scale.dtype==np.float64 and np.isfinite(scale).all() and support.shape==(200,2,680) and support.dtype==np.bool_
 calculated=np.stack([x.mean(-1,dtype=np.float64) for x in np.split(token,np.cumsum(WIDTHS)[:-1],axis=-1)],-1);assert np.array_equal(calculated,scale)
 assert np.max(abs(token[:,5]-old_token[:,0]))<=2e-5 and np.max(abs(token[:,6]-old_token[:,2]))<=2e-5
 assert np.max(abs(token[:,1,0]-token[:,5,0]))<=2e-5 and np.max(abs(token[:,3,0]-token[:,6,0]))<=2e-5
 assert len(done['records'])==200 and len(list((ROOT/'images').glob('*.png')))==200 and len(list((ROOT/'engineering_replays').glob('*.png')))==2
 pixel_hashes=[];constant_queries=[]
 for i,pin in enumerate(done['records']):
  record=json.loads(check(pin).read_bytes());row=rows[i];seed=int.from_bytes(hashlib.sha256(f"1729|AR-Dfit-actual-prefix|sub-01|{row['stimulus_id']}|R4".encode()).digest()[:8],'little')%(2**63-1)
  assert record['query_index']==i and record['stimulus_id']==row['stimulus_id'] and record['array_row']==row['array_row'] and record['seed']==seed
  assert record['own_EEG_c_sha256']==hashlib.sha256(c[i].tobytes()).hexdigest() and record['sampled_codes_sha256']==hashlib.sha256(sampled[i].tobytes()).hexdigest()
  assert record['CFG']==1.5 and record['top_k']==600 and record['top_p']==0 and record['best_of_N']==1
  with Image.open(check(record['image'])) as im:array=np.array(im.convert('RGB'))
  assert array.shape==(256,256,3) and array.dtype==np.uint8 and hashlib.sha256(array.tobytes()).hexdigest()==record['pixel_sha256'];pixel_hashes.append(record['pixel_sha256'])
  if np.max(array.reshape(-1,3).std(0))<1e-8:constant_queries.append(i)
  if i<2:
   with Image.open(ROOT/'engineering_replays'/f'query_{i:03d}_same_seed.png') as im:assert np.array_equal(np.array(im.convert('RGB')),array)
 cached_NLL_error=0.;support_tie_differences=0;alpha=np.repeat(np.array([1.5*k/9 for k in range(10)],np.float32),WIDTHS);plus=np.repeat(np.array([1+1.5*k/9 for k in range(10)],np.float32),WIDTHS)
 for i in range(2):
  logits=np.load(ROOT/'first2_logits'/f'query_{i:03d}_conditional_and_zero.npy',mmap_mode='r',allow_pickle=False);assert logits.shape==(2,680,4096) and logits.dtype==np.float32 and np.isfinite(logits).all()
  for lo in range(0,680,32):
   hi=min(lo+32,680);raw=np.array(logits[:,lo:hi]);truth=np.broadcast_to(target[i,lo:hi].astype(np.int64),(2,hi-lo));actual=nll64(raw,truth)
   cached_NLL_error=max(cached_NLL_error,float(np.max(abs(actual-token[i,[1,3],lo:hi]))));assert np.max(abs(actual-token[i,[1,3],lo:hi]))<2e-3
   guided=plus[lo:hi,None]*raw[0]-alpha[lo:hi,None]*raw[1];guided_loss=nll64(guided,target[i,lo:hi].astype(np.int64))
   assert np.max(abs(guided_loss-token[i,4,lo:hi]))<3e-3
   cutoff=np.partition(guided,4096-600,axis=-1)[:,4096-600];target_logits=np.take_along_axis(guided,target[i,lo:hi,None],-1)[:,0];exact=target_logits>=cutoff;different=exact!=support[i,0,lo:hi]
   assert np.all(abs(target_logits[different]-cutoff[different])<3e-3);support_tie_differences+=int(different.sum())
 draws=np.random.default_rng(1729).integers(0,20,(1000,20));np.save(OUT/'bootstrap_concept_draws.npy',draws.astype(np.int16),allow_pickle=False);contrasts={
  'own_minus_donor_EEG_at_actual_prefix':scale[:,2]-scale[:,1],
  'own_minus_zero_EEG_at_actual_prefix':scale[:,3]-scale[:,1],
  'own_minus_zero_EEG_at_sameconcept_donor_GT_prefix':scale[:,0]-np.stack([x.mean(-1,dtype=np.float64) for x in np.split(old_token[:,3],np.cumsum(WIDTHS)[:-1],axis=-1)],-1),
  'actual_prefix_NLL_penalty_at_own_EEG':scale[:,1]-scale[:,5],
  'CFG_NLL_penalty_vs_own_at_actual_prefix':scale[:,4]-scale[:,1],
  'CFG_NLL_penalty_vs_own_at_GT_prefix':scale[:,7]-scale[:,5]}
 analysis={}
 for name,x in contrasts.items():
  by_concept=np.array([x[g].mean(0) for g in groups]);boot=by_concept[draws].mean(1);weighted=x@WIDTHS/680;weighted_boot=(by_concept@WIDTHS/680)[draws].mean(1)
  analysis[name]=dict(mean_nats_per_token_by_scale=x.mean(0).tolist(),descriptive95_by_scale=np.percentile(boot,[2.5,97.5],axis=0).T.tolist(),all680_token_weighted_mean=float(weighted.mean()),all680_token_weighted_descriptive95=np.percentile(weighted_boot,[2.5,97.5]).tolist())
 support_by_scale=np.stack([x.mean(-1) for x in np.split(support,np.cumsum(WIDTHS)[:-1],axis=-1)],-1)
 summary=dict(status='ALL200_ACTUAL_PREFIX_DIAGNOSTIC_CONTRASTS_DESCRIPTIVE_DEVELOPER_EXTENSION',conditions=done['conditions'],mean_NLL_by_condition_and_scale=scale.mean(0).tolist(),contrasts=analysis,
  target_in_top600_fraction_by_actual_then_GT_prefix_and_scale=support_by_scale.mean(0).tolist(),fraction_own_beats_zero_at_actual_prefix_by_scale=(scale[:,1]<scale[:,3]).mean(0).tolist(),
  inference='Exploratory extension after teacher-prefix findings. This is one frozen native baseline, not new-method image benefit; CFG NLL here is untruncated. A target excluded from top600 has zero actual truncated-sampler probability, not a finite surrogate score. No nominal multiple-look/contrast error control or causal/MI claim.')
 write(OUT/'diagnostic_summary.json',summary)
 plot=None
 try:
  import matplotlib
  matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
  for ax,name in zip(axes.flat,list(contrasts)[:2]+['CFG_NLL_penalty_vs_own_at_actual_prefix','actual_prefix_NLL_penalty_at_own_EEG']):
   x=analysis[name];ci=np.array(x['descriptive95_by_scale']);ax.plot(range(10),x['mean_nats_per_token_by_scale'],'o-');ax.fill_between(range(10),ci[:,0],ci[:,1],alpha=.2);ax.axhline(0,color='black',linewidth=.6);ax.set_xticks(range(10),SIDES);ax.set_xlabel('VQ scale side');ax.set_ylabel('Difference (nats/token)');ax.set_title(name.replace('_',' '),fontsize=9)
  fig.suptitle('Actual sampled prefix; fixed native final; all Dfit200\nDescriptive developer diagnostic; no new-method image comparison',fontsize=11);p=OUT/'actual_prefix_all_scales.png';fig.savefig(p,dpi=160);plt.close(fig);plot=ref(p)
 except ImportError:plot=dict(status='PLOT_LIBRARY_UNAVAILABLE_SCIENTIFIC_ARRAY_AUDIT_UNCHANGED')
 receipt=R/'audit/AR_native_Dfit200_actual_prefix_20261029_v1_job.json';job=json.loads(receipt.read_bytes());assert job['status']=='COMPLETED' and job['exit_code']==0 and job['GPU_UUID']=='GPU-62a1aee8-4740-dc59-93f4-d50644db650d'
 resume=M/'AR_native_actual_prefix_reserved_slot_20261029_v1/waiter_resumed.json';assert json.loads(resume.read_bytes())['resume_signal_sent'] is True
 result=dict(status='PASS_ALL200_ACTUAL_IMAGES_CODES_8CONDITIONS_10SCALES_AND_FIRST2_RAW_LOGIT_CPU_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_completion=ref(ROOT/'completion.json'),summary=ref(OUT/'diagnostic_summary.json'),plot=plot,wrapper_receipt=ref(receipt),original_image_waiter_resumed=ref(resume),
  queries=200,conditions=8,token_NLL_values=1088000,all16000_scale_values_recomputed_exact=True,all200_image_and_EEG_code_seed_identity_SHA_verified=True,
  images=200,same_seed_engineering_replays=2,unique_query_pixel_SHAs=len(set(pixel_hashes)),constant_query_indices=constant_queries,first2_cached_NLL_max_error=cached_NLL_error,first2_top600_boundary_differences=support_tie_differences,
  all1000_concept_bootstrap_draws_saved=True,bootstrap_draws=ref(OUT/'bootstrap_concept_draws.npy'),bootstrap_seed=1729,contrasts=analysis,new_GPU_jobs=0,seconds=time.monotonic()-started)
 write(OUT/'completion.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');q=p.parse_args()
 if q.wait:
  deadline=time.monotonic()+3600
  while not (ROOT/'completion.json').exists():
   assert not (ROOT/'failure.json').exists() and not (M/'AR_native_actual_prefix_reserved_slot_20261029_v1/failure.json').exists(),'Preserve original diagnostic failure; no CPU closure fabricated'
   assert time.monotonic()<deadline;time.sleep(10)
  deadline=time.monotonic()+30
  while True:
   rp=R/'audit/AR_native_Dfit200_actual_prefix_20261029_v1_job.json';resume=M/'AR_native_actual_prefix_reserved_slot_20261029_v1/waiter_resumed.json'
   if rp.is_file() and json.loads(rp.read_bytes()).get('status')!='RUNNING' and resume.is_file():break
   assert time.monotonic()<deadline;time.sleep(.2)
 try:run()
 except BaseException as error:
  if OUT.exists():write(OUT/'failure.json',dict(status='ACTUAL_PREFIX_INDEPENDENT_CPU_AUDIT_FAILURE_PRESERVED',error=repr(error),traceback=traceback.format_exc()))
  raise
