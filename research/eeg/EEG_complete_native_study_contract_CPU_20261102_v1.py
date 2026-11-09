"""New full EEG experiment. Legacy admissions bind inputs, never authorize old fits."""
import hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
B=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
CODE=B/'VAR_full_EEG_study_code_20261102_v1';OUT=B/'VAR_full_EEG_study_20261102_v1'
BANK=dict(path=(B/'VAR_full_T_native_bank_code_20261102_v1/protocol.json').as_posix(),bytes=3379,sha256='26994a32e29844a8423a8222fed1022116d11fb2f3e3b5ff6b94028287196c2b')
GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
ARMS=('global_full','global_coarse','chain_full','independent_full','visual_global_full','image_prior_full')
CANDIDATES=('global_full','chain_full')
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p);assert p.is_file() and not p.is_symlink();a=p.stat()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 b=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
 return dict(path=str(p),bytes=b.st_size,sha256=h)
def checked(pin):assert ref(pin['path'])==pin;return Path(pin['path'])
def write(p,x,update=False):
 p=Path(p);assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.pending');assert not t.exists()
 with t.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 if not update:assert not p.exists()
 os.replace(t,p)
def process(pid,ticks):
 try:
  v=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return dict(PID=pid,start_ticks=int(v[19]),state=v[0],exact_live=int(v[19])==ticks and v[0] not in ('Z','X'))
 except FileNotFoundError:return dict(PID=pid,present=False,exact_live=False)
def bank_contract():
 q=read(checked(BANK));pin=next(v for v in q['source_pins'] if Path(v['path']).name=='VAR_full_T_native_bank_contract_CPU_20261102_v1.py')
 sp=importlib.util.spec_from_file_location('same_full_bank_for_EEG',checked(pin));c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c);c.setup(BANK['path']);return c,q
def bank_state():
 c,q=bank_contract();p=c.OUT;owner=process(525462,119859959);last=dict(exact_live=False)
 if (p/'progress.json').exists():
  v=read(p/'progress.json');last=process(v['active_PID'],v['active_start_ticks'])
 if (p/'failure.json').exists():return dict(status='FAILED_UPSTREAM_BANK_NO_RESTART',owner=owner,last_worker=last)
 if owner['exact_live'] or last['exact_live']:return dict(status='WAIT_SAME_BANK_LIVE',owner=owner,last_worker=last)
 if not (p/'completion.json').exists():return dict(status='BANK_EXITED_WITHOUT_WHOLE_SUCCESS_NO_RESTART',owner=owner,last_worker=last)
 done=read(p/'completion.json');assert done['status']=='COMPLETE_FULL_T13230_NATIVE_K1000_ALL680_BANK_AND_WHOLE_CPU_AUDIT' and done['protocol']==BANK
 a=read(checked(done['CPU_audit']));assert a['status']=='PASS_ALL13230_ROWS_132300000_RETAINED_SCALE_CELLS_AND_COMPLETE_CLASS_COVERAGE' and len(a['parts'])==207
 return dict(status='READY_FULL_BANK_WHOLE_SUCCESS_EXACT_EXIT',owner=owner,last_worker=last,completion=ref(p/'completion.json'),CPU_audit=done['CPU_audit'])
def validate(q):
 assert q['status']=='REGISTERED_CONDITIONAL_COMPLETE_NATIVE_EEG_FULL_LIFECYCLE'
 assert q['host']=='old' and q['GPU_UUID']==GPU and q['code_root']==CODE.as_posix() and q['output_root']==OUT.as_posix() and q['bank_protocol']==BANK
 assert tuple(q['arms'])==ARMS and tuple(q['candidates'])==CANDIDATES
 assert q['recipe']==dict(seed=1729,epochs=10,batch=64,updates=2070,queries=13230,concepts=1323,query_exposures=132300,lr=1e-4,weight_decay=.01,betas=[.9,.999],eps=1e-8,gradient_clip=1.,EEG_forward_mode='eval_with_grad',final_epoch_only=True,loss_normalization='nats_per_query_no_scale_dependent_multiplier')
 assert q['maximum_output_bytes']==5<<30 and q['maximum_waiting_metadata_bytes']==16<<20
 assert q['GPU_hard_seconds']==900 and q['GPU_soft_seconds']==780 and q['maximum_GPU_bytes']==20<<30 and q['maximum_extra_CPU_bytes']==10<<30
 assert q['minimum_start_CPU_headroom_bytes']==24<<30 and q['minimum_runtime_CPU_headroom_bytes']==10<<30
 assert q['maximum_queue_seconds']==72*3600 and q['wait_for_bank_seconds']==48*3600 and q['no_automatic_scientific_retry']
 assert q['stage80']=='first_two_queries_of_each_of_the40_fixed_concepts' and q['gallery_size']==400
 assert q['CPU_input_seconds']==1200 and q['CPU_profile_seconds']==900 and q['CPU_fit_audit_seconds']==900 and q['CPU_prediction_seconds']==900 and q['CPU_artifact_seconds']==1800 and q['CPU_score_seconds']==1800
 assert q['CPU_prediction_tolerance']==dict(atol=5e-4,rtol=1e-4)
 assert q['bank_started_UTC']=='2026-11-02T00:35:10.177577+00:00'
 assert q['inference']==dict(resolution=256,CFG=0,top_k=0,scales=10,one_image=True,best_of_N=1,EEG_FP32=True,decoder_precision='DECLARED_DECODER_ONLY_AMP_FP16',VQ_FP32=True,latent_chain_private_RNG=True,ten_fresh_prefix_calls_for_every_arm=True,latent_probability_precision='FP64')
 assert q['new_seed_search'] is False and q['original_ordinary_immutable'] and q['NEW_reserved']
 return q
def setup(protocol,need_bank=True,formal=False):
 assert Path(protocol)==CODE/'protocol.json';q=validate(read(protocol))
 for pin in q['source_pins']+q['inherited_sources']:checked(pin)
 r=read(CODE/'actual_conditional_source_review.json');assert r['status']=='PASS_CONDITIONAL_FULL_EEG_CODE_SCOPE_REVIEW_NOT_FORMAL_ADMISSION' and r['protocol']==ref(protocol) and r['source_pins']==q['source_pins']
 if need_bank:assert bank_state()['status']=='READY_FULL_BANK_WHOLE_SUCCESS_EXACT_EXIT'
 if formal:
  a=read(OUT/'actual_formal_admission.json');assert a['status']=='ADMITTED_ONLY_AFTER_ACTUAL_FULL_INPUT_ALL6_PROFILE_AND_CPU_AUDIT' and a['protocol']==ref(protocol)
  expected={'inputs':'PASS_COLD_COMPLETE_T13230_DCHECK400_REAL_EEG_INPUT_JOIN_TRANSFORM','profiles_CPU':'PASS_ALL6_REAL_PROFILE_GRADIENTS_AND_SERIALIZED_RESUME_STATES','generation_profile':'PASS_ALL6_REAL_GENERATION_PROFILE_IMAGES_AND_LATENT_LAW'}
  for key,status in expected.items():
   item=read(checked(a[key]));assert item['status']==status and item['protocol']==ref(protocol)
 return q
def source_inputs(q,binary=False):
 base=read(checked(q['base_EEG_spec']));a=read(checked(q['legacy_input_admission']));image=read(checked(q['legacy_image_spec']))
 assert base['training_admission']==q['legacy_input_admission']['path'] and a['host_role']=='old'
 for p in base['source_pins']+a['source_pins']:checked(p)
 stats=read(checked(base['full_EEG']['normalization']));assert stats['role']=='full' and stats['training_positions']==list(range(13230)) and stats['fit_only_allowed_role_T']
 assert len(stats['channel_names'])==63 and len(set(k.upper() for k in stats['channel_names']))==63
 if binary:
  for pin in (base['T_mean63'],base['full_EEG']['model'],a['data']['tokens'],a['data']['EEG_identities'],a['data']['token_identities'],image['Dcheck_mean63'],image['physical_queries'],image['panel']):checked(pin)
 return base,a,image,stats
def roots():
 sys.path.insert(0,str(CODE));c,q=bank_contract();d,n,prev,_=c.dependency(q);s=read(d.previous()[0].NATIVE['path'])
 for p in reversed(s['native_module_roots']):sys.path.insert(0,p)
 sys.path.insert(0,s['assets']['vendor']);sys.path.insert(0,str(n.CODE));sys.path.insert(0,str(d.CODE));sys.path.insert(0,str(CODE));return s

def resources(q,starting=False,waiting=False):
 c,bq=bank_contract();used=allocated(CODE)+allocated(OUT)
 if waiting:
  assert used<=q['maximum_waiting_metadata_bytes'];r=c.resources(bq,starting=False)
  required=r['required_remaining_bytes']+q['maximum_waiting_metadata_bytes']-used;free=shutil.disk_usage(B).free;assert free>=required
  return dict(waiting_only=True,used_bytes=used,required_remaining_bytes=required,free_bytes=free,no_large_input_or_GPU_loaded=True)
 assert bank_state()['status']=='READY_FULL_BANK_WHOLE_SUCCESS_EXACT_EXIT'
 d,n,prev,_=c.dependency(bq);m=n.mean_contract();mean=m.validate(read(n.MEAN_PROTOCOL['path']));control=read(n.CONTROL_PROTOCOL['path'])
 meanused=allocated(m.OUT)+sum(allocated(Path(p)) for p in control['additional_study_roots']);assert meanused<=4<<30
 parent=read(checked(m.PARENT));released_parent=max(0,(8<<30)-allocated(parent['output_root']))
 # Both source studies are canonically complete and all exact handles absent.
 assert m.dependency(mean)['status']=='READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS'
 native=[B/f'VAR_multiscale_native_profile{k}_20261101_v{v}' for v in (1,2,3) for k in ('_code','')]+[B/'VAR_full_native_throughput_top0_code_20261101_v1',B/'VAR_full_native_throughput_top0_20261101_v1']
 nativeused=sum(allocated(p) for p in native);fp16used=allocated(d.CODE)+allocated(d.OUT);bankused=allocated(c.CODE)+allocated(c.OUT)
 assert nativeused<=128<<20 and fp16used<=32<<20 and bankused<=1400<<20 and used<=q['maximum_output_bytes']
 required=(50<<30)+(8<<30)+(14<<30)+(64<<20)+(32<<20)+(128<<20)-nativeused+(32<<20)-fp16used+(1400<<20)-bankused+q['maximum_output_bytes']-used
 free=shutil.disk_usage(B).free;head=m.headroom();floor=q['minimum_start_CPU_headroom_bytes' if starting else 'minimum_runtime_CPU_headroom_bytes'];assert free>=required and head>=floor
 return dict(used_bytes=used,free_bytes=free,required_remaining_bytes=required,effective_CPU_headroom_bytes=head,only_closed_parent_unused_budget_released=released_parent,only_closed_mean_unused_budget_released=(4<<30)-meanused,legacy50GiB_and_buffer8GiB_unchanged=True,physical_reservation=False)
def allocated(root):
 total=0;seen=set();p=Path(root)
 if not p.exists():return 0
 for f in p.rglob('*'):
  assert not f.is_symlink()
  try:s=f.stat()
  except FileNotFoundError:continue
  if f.is_file() and (s.st_dev,s.st_ino) not in seen:seen.add((s.st_dev,s.st_ino));total+=s.st_blocks*512
 return total
