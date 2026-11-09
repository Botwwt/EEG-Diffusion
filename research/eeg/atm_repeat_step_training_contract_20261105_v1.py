"""Same native task, one paired-repeat update rule, two discriminating controls."""
import json,os,sys
from pathlib import Path
import atm_repeat_step_training_core_20261105_v1 as F

def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_PAIRED_REPEAT_STEP_T10_WITH_FULL_IMAGES'
 assert q['host']=='old' and q['seed']==1729 and q['arms']==list(F.ARMS) and q['epochs']==10 and q['updates_per_arm']==830
 assert q['maximum_output_bytes']==1200<<20 and q['new_models']==3 and q['new_updates']==2490
 sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
 import atm_repeat_count_training_contract_20261105_v1 as T
 c=T.setup(q['prior_policy']['path']);K=c['K'];K.checked(q['prior_policy'])
 for pin in q['sources'].values():K.checked(pin)
 profile=K.read(K.checked(q['profile_completion']));assert profile['status']=='COMPLETE_REAL_TRAIN_REPEAT_STEP_PROFILE'
 j=K.read(K.checked(profile['receipt']));assert j['status']=='COMPLETED' and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
 for pin in q['closed_launches']:
  a=K.read(K.checked(pin));assert not K.process(a['PID'],a['start_ticks'])['live']
 assert q['checkpoint']==c['q']['checkpoint'];K.checked(q['checkpoint'])
 for pin in q['reference_checkpoints'].values():K.checked(pin)
 K.checked(q['reference_conditions']);ledger=K.read(K.checked(q['reserve_ledger']))
 assert ledger['status']=='THREE_CLOSED_UNUSED_FORECASTS_EXPLICIT_SETTLEMENT_NO_DELETION'
 assert ledger['legacy_floor_bytes']==50<<30 and ledger['buffer_bytes']==8<<30 and len(ledger['settlements'])==3
 return dict(q=q,K=K,out=Path(q['output_root']),prior=c,physical=c['physical'],T=c['T'],ledger=ledger,queries=c['queries'],R80=c['R80'],image_reference=c['image_reference'])

def resources(c,starting=False):
 q,K,l=c['q'],c['K'],c['ledger'];credit=0
 assert K.allocated(c['prior']['q']['project_root'])==l['prior_project_used'],'Prior project changed after its frozen resource quote'
 K.checked(l['charge_snapshot'])
 for s in l['settlements']:
  pq=K.read(K.checked(s['policy']));done=K.read(K.checked(s['completion']))
  assert done['status']==s['completion_status'] and pq['maximum_output_bytes']==s['maximum_output_bytes']
  assert K.allocated(s['project'])==s['actual_allocated_bytes']
  for pin in s['all_launches']:
   a=K.read(K.checked(pin));assert not K.process(a['PID'],a.get('start_ticks'))['live']
  for pin in s['jobs']:
   j=K.read(K.checked(pin));assert j['status']=='COMPLETED' and j['exit_code']==0
   assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
  assert s['unused_future_allowance']==s['maximum_output_bytes']-s['actual_allocated_bytes'];credit+=s['unused_future_allowance']
 assert credit==l['settled_unused_future_bytes']
 base=l['prior_quote']['required_including_this_project']-credit;assert base>=58<<30
 used=K.allocated(q['project_root']) if Path(q['project_root']).exists() else 0;assert used<=q['maximum_output_bytes']
 st=os.statvfs(Path(q['project_root']).parent);free=st.f_bavail*st.f_frsize;required=base+q['maximum_output_bytes']-used
 assert free>=required,dict(free=free,required=required)
 mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
 head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cur=Path('/sys/fs/cgroup/memory.current')
 if mx.exists() and cur.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cur.read_text()))
 assert head>=(24<<30 if starting else 2<<30)
 return dict(free_bytes=free,required_including_this_project=required,own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,
  settled_unused_future_bytes=credit,all_other_forecasts_held_at_snapshot=True,actual_files_deleted=0,CPU_headroom_bytes=head)
