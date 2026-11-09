"""Complete native EEG training: fixed paired T coverage, transactions and resource ledger."""
import copy,hashlib,importlib.util,json,math,os,shutil,sys
from pathlib import Path
import numpy as np
ARMS=('positive_only','directional','sameconcept_wrong_EEG')
STATUS='REGISTERED_FULL_T10_NATIVE_EEG_DIRECTIONAL_THREE_ARM_THREE_PASS_STUDY'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(pin):
 actual=ref(pin['path']);assert all(actual[k]==v for k,v in pin.items()),pin;return Path(pin['path'])
def allocated(root):
 root=Path(root);total=0
 if root.exists():
  for p in root.rglob('*'):
   assert not p.is_symlink()
   if p.is_file():total+=p.stat().st_blocks*512
 return total
def directory_fingerprint(root):
 root=Path(root);h=hashlib.sha256();count=0;logical=0;used=0
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():
   s=p.stat();h.update(json.dumps([p.relative_to(root).as_posix(),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns],separators=(',',':')).encode());count+=1;logical+=s.st_size;used+=s.st_blocks*512
 return dict(path=str(root),files=count,logical_bytes=logical,allocated_bytes=used,stat_SHA256=h.hexdigest())
def prior_commitments(engineering):
 """Reproduce the unchanged old ledger in new code; never patch its guards."""
 old=read(checked(engineering['previous_gradient_protocol']));base=read(checked(old['base_protocol']));rp=base['resource_protocol'];r=read(checked(rp))
 pin=next(v for v in r['source_pins'] if Path(v['path']).name=='EEG_complete_native_study_contract_CPU_20261102_v1.py')
 assert pin['sha256']=='75d69443b5cfad1ef80785b28a0c001d29239621ba57bde138308e5c223df8c1'
 module=importlib.util.spec_from_file_location('directional_preserved_prior_resources',checked(pin));C=importlib.util.module_from_spec(module);module.loader.exec_module(C);C.setup(rp['path'])
 c,bq=C.bank_contract();assert C.bank_state()['status']=='READY_FULL_BANK_WHOLE_SUCCESS_EXACT_EXIT'
 d,n,prev,_=c.dependency(bq);m=n.mean_contract();mean=m.validate(read(n.MEAN_PROTOCOL['path']));assert m.dependency(mean)['status']=='READY_ACTUAL_CLOSED_SOURCE_AND_TRAIN_MEANS'
 used=sum(C.allocated(p) for p in (C.CODE,C.OUT,C.OLD_CODE,C.OLD_OUT,C.OLD2_CODE,C.OLD2_OUT));diag=C.allocated(C.DIAG_CODE)+C.allocated(C.DIAG_OUT)
 native=[C.B/f'VAR_multiscale_native_profile{k}_20261101_v{v}' for v in (1,2,3) for k in ('_code','')]+[C.B/'VAR_full_native_throughput_top0_code_20261101_v1',C.B/'VAR_full_native_throughput_top0_20261101_v1']
 nu=sum(C.allocated(p) for p in native);fu=C.allocated(d.CODE)+C.allocated(d.OUT);bu=C.allocated(c.CODE)+C.allocated(c.OUT)
 assert used<=r['maximum_output_bytes'] and diag<=8<<20 and nu<=128<<20 and fu<=32<<20 and bu<=1400<<20
 required=(50<<30)+(8<<30)+(14<<30)+(64<<20)+(32<<20)+(128<<20)-nu+(32<<20)-fu+(1400<<20)-bu+r['maximum_output_bytes']-used+(8<<20)-diag
 external_required=required
 meanused=allocated(base['output_root']);assert meanused<=1<<30;required+=(1<<30)-meanused
 groups=old['upstream_output_groups']+[dict(name='closed_overlap_gradient',cap=512<<20,roots=[old['output_root'],old['code_root']]+old['preserved_roots']),dict(name='native_direction_engineering',cap=384<<20,roots=[engineering['output_root'],engineering['code_root']])]
 usage={}
 for group in groups:
  size=sum(allocated(p) for p in group['roots']);assert size<=group['cap'];usage[group['name']]=dict(cap=group['cap'],used=size);required+=group['cap']-size
 return dict(required_before_F_release=required,external_required_before_other_groups=external_required,remaining_upstream_static_F_cap=12<<30,
  legacy_floor_bytes=50<<30,disk_buffer_bytes=8<<30,all_other_upstream_static_bytes=(2<<30)+(64<<20)+(32<<20),
  groups=usage,mean_used_bytes=meanused,headroom_bytes=m.headroom(),free_bytes=shutil.disk_usage(C.B).free)
def validate_release(pin,check_fingerprint=True):
 ledger=read(checked(pin));assert ledger['status']=='VERIFIED_CLOSED_F12GIB_UNUSED_FUTURE_RESERVATION_RELEASE_ONLY'
 assert ledger['released_future_bytes']==12<<30 and not ledger['files_removed'] and ledger['legacy50GiB_buffer8GiB_unchanged']
 done=read(checked(ledger['F_completion']));study=read(checked(done['study']))
 assert done['status']=='COMPLETE_FULL1654_TWO_F_FITS_4000_R80_IMAGES_ALL16_FIXED_COMPARISONS' and study['maximum_output_bytes']==12<<30
 for owner in ledger['closed_owner_PIDs']:assert not Path('/proc',str(owner)).exists()
 if check_fingerprint:
  for snapshot in ledger['F_roots_retained_stat_fingerprints']:assert directory_fingerprint(snapshot['path'])==snapshot
 return ledger
def setup(protocol):
 s=read(protocol);assert s['status']==STATUS and s['host']=='old' and s['seed']==1729 and tuple(s['arms'])==ARMS
 assert s['subjects']==[f'sub-{i:02d}' for i in range(1,11)] and s['epochs']==3 and s['pair_batch']==8
 assert s['T_images']==13230 and s['T_concepts']==1323 and s['updates_per_subject']==2481 and s['primary_exposures_per_subject']==39690
 assert s['primary_exposures_per_arm']==396900 and s['total_updates_three_arms']==74430
 assert s['maximum_output_bytes']==10<<30 and s['maximum_GPU_bytes']==20<<30 and s['maximum_extra_CPU_bytes']==10<<30
 assert s['GPU_hard_seconds']==900 and s['GPU_soft_seconds']==780 and s['maximum_queue_seconds']==7*24*3600 and s['maximum_segments_per_model']==64
 assert s['optimizer']==dict(lr=1e-5,weight_decay=.01,betas=[.9,.999],eps=1e-8,clip=1.)
 assert s['loss']==dict(token=.1,directional=.1,positive='native_epsilon_MSE',directional_state='normalized_harmonic_overlap',same_base_for_all=True)
 assert s['encoder_forward']=='FP32_eval_with_grad_outputLN_affine_frozen' and s['fixed_final_epoch']==3 and s['ordinary_immutable'] and not s['NEW_contacted']
 for pin in s['source_pins']:checked(pin)
 e=read(checked(s['engineering_protocol']));sys.path.insert(0,str(Path(s['engineering_protocol']['path']).parent))
 import brainhive_native_direction_contract_20261103_v1 as E
 _,old,base,G,H=E.setup(s['engineering_protocol']['path'])
 a=read(checked(s['engineering_CPU_audit']));assert a['status']=='PASS_ALL80_NATIVE_DIRECTION_LOSSES_FULL6450176_GRADIENT_SNAPSHOT_AND_FIRST_ADAM_CPU'
 validate_release(s['closed_F_budget_review'])
 return s,e,old,base,G,H
def resources(s,e,starting=False):
 original=prior_commitments(e);release=validate_release(s['closed_F_budget_review']);used=allocated(s['code_root'])+allocated(s['output_root']);assert used<=s['maximum_output_bytes']
 required=original['required_before_F_release']-release['released_future_bytes']+s['maximum_output_bytes']-used
 assert required>=(50<<30)+(8<<30) and original['free_bytes']>=required and original['headroom_bytes']>=(24 if starting else 10)<<30
 return dict(inherited=original,only_closed_F_future_bytes_released=release['released_future_bytes'],own_used_bytes=used,own_remaining_bytes=s['maximum_output_bytes']-used,
  required_remaining_bytes=required,free_bytes=original['free_bytes'],headroom_bytes=original['headroom_bytes'],no_file_deletion=True,physical_reservation=False)
def stable_seed(subject,epoch,namespace):return int(hashlib.sha256(f'1729|native_EEG_direction|{subject}|{epoch}|{namespace}'.encode()).hexdigest()[:15],16)
def pair_plan(concepts,subject,epoch):
 assert epoch in (0,1,2);pairs=[]
 for key,rows in sorted(concepts.items()):
  assert len(rows)==10 and len(set(rows))==10
  order=np.random.default_rng(stable_seed(subject,epoch,'pairs|'+key)).permutation(sorted(rows))
  for i in range(0,10,2):pairs.append([order[i],order[i+1],order[(i+2)%10],order[(i+3)%10]])
 result=np.array(pairs,np.int64);assert result.shape==(6615,4)
 shuffle=np.random.default_rng(stable_seed(subject,epoch,'order')).permutation(6615);result=result[shuffle]
 assert len(np.unique(result[:,:2]))==13230 and all(len(set(row))==4 for row in result)
 return result
def rng_probe(noise_rng,axial_rng,torch):
 clone=torch.Generator(device='cpu');clone.set_state(noise_rng.get_state());axial=np.random.default_rng();axial.bit_generator.state=copy.deepcopy(axial_rng.bit_generator.state)
 t=torch.randint(0,1000,(2,),generator=clone);eps=torch.randn((2,4,64,64),generator=clone)
 return dict(t=t.tolist(),epsilon_SHA256=hashlib.sha256(eps.numpy().tobytes()).hexdigest(),axial_normal=axial.normal(size=16).tolist(),axial_uniform=axial.random(16).tolist())
def harmonic_states(zi,zj,eps,t,alpha,axial_rng):
 """Exact axial rejection; every member kept. All inputs are CPU arrays."""
 out=[];attempts=[]
 for left,right,noise,tt in zip(zi,zj,eps,t):
  a=float(np.sqrt(alpha[int(tt)]));sigma=float(np.sqrt(1-alpha[int(tt)]));delta=np.asarray(left,np.float64)-right;length=np.linalg.norm(delta);assert length>0
  direction=delta/length;d=a/sigma*length;ax=float(np.sum(np.asarray(noise,np.float64)*direction));accepted=None
  # New study's fixed block RNG order; exact distribution, not the old scalar RNG stream.
  for used in range(0,1000000,1024):
   size=min(1024,1000000-used);candidates=axial_rng.normal(size=size);uniforms=axial_rng.random(size);log_accept=np.log(2.)-np.logaddexp(.5*d*candidates,-.5*d*candidates)
   good=np.flatnonzero(np.log(np.maximum(uniforms,np.finfo(float).tiny))<=log_accept)
   if len(good):accepted=float(candidates[good[0]]);attempts.append(used+int(good[0])+1);break
  if accepted is None:raise RuntimeError('Finite harmonic sampler rejection bound, no row drop or proposal change')
  out.append(.5*a*(np.asarray(left,np.float64)+right)+sigma*(np.asarray(noise,np.float64)+(accepted-ax)*direction))
 return np.array(out,np.float32),attempts
def atomic_json(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.pending');assert not temp.exists()
 with temp.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 os.replace(temp,p)
def save_resume(root,state,torch):
 root=Path(root);counter=state['snapshot_id'];slot=root/('resume_'+('A' if counter%2==0 else 'B')+'.pt');temp=slot.with_name(slot.name+'.pending');assert not temp.exists()
 torch.save(state,temp)
 with temp.open('rb') as stream:os.fsync(stream.fileno())
 os.replace(temp,slot);record=dict(snapshot_id=counter,checkpoint=ref(slot),updates=state['updates'],primary_exposures=state['primary_exposures'],epoch=state['epoch'],pair_offset=state['pair_offset'])
 atomic_json(root/'resume_index.json',record);return record
