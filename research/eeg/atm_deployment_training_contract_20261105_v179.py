"""Fixed two-arm deployment-supervision study; inherited assets and full image reserve."""
import json,os,sys
from pathlib import Path
ARMS=('stochastic_P','deployment_P')
ALL_ARMS=('frozen_positive_NCE','R4_continued')+ARMS
def setup(path):
 q=json.loads(Path(path).read_bytes())
 assert q['status']=='LOCKED_DEPLOYMENT_SUPERVISION_T10_FULL_R4_R80' and q['host']=='old'
 assert q['seed']==1729 and q['arms']==list(ARMS) and q['epochs']==10 and q['lambda_P']==1.
 assert q['maximum_output_bytes']==768<<20 and q['new_images']==1200
 sys.path.insert(0,str(Path(q['parent_policy']['path']).parent))
 import atm_parent_R80_contract_20261105_v174 as R
 parent=R.setup(q['parent_policy']['path']);K=parent['K'];K.checked(q['parent_policy'])
 for p in q['sources'].values():K.checked(p)
 for p in [q['profile_policy'],q['profile_completion'],q['profile_audit'],q['checkpoint'],q['projector']]:K.checked(p)
 assert K.read(q['profile_completion']['path'])['status']=='PASS_REAL_TWO_BATCH_THREE_ROUTE_PROFILE'
 assert K.read(q['profile_audit']['path'])['status']=='PASS_SIX_REAL_ROUTE_LOSSES_AND_NATIVE_STEP_CONTROL'
 ledger=K.read(K.checked(q['reserve_ledger']));assert ledger['status']=='THREE_ADDITIONAL_CLOSED_FORECASTS_SETTLED_ONCE_FOR_DEPLOYMENT_STUDY'
 return dict(q=q,K=K,parent=parent,old=parent['n'],physical=q['physical'],ledger=ledger,out=Path(q['output_root']))
def resources(c,starting=False):
 q,K,l=c['q'],c['K'],c['ledger'];credit=0
 K.checked(l['source_probe'])
 for s in l['settlements']:
  assert K.allocated(s['project'])==s['actual_allocated_bytes']
  pq=K.read(K.checked(s['policy']));done=K.read(K.checked(s['completion']))
  assert pq['maximum_output_bytes']==s['maximum_output_bytes'] and done['status']==s['completion_status']
  for pin in s['all_launches']:
   a=K.read(K.checked(pin));assert not K.process(a['PID'],a.get('start_ticks'))['live']
  for pin in s['jobs']:
   j=K.read(K.checked(pin));assert j['status']=='COMPLETED' and j['exit_code']==0
   assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
  assert s['unused_future_allowance']==s['maximum_output_bytes']-s['actual_allocated_bytes'];credit+=s['unused_future_allowance']
 assert credit==795447296==l['settled_unused_future_bytes']
 used=K.allocated(q['project_root']) if Path(q['project_root']).exists() else 0
 assert used<=q['maximum_output_bytes']
 base=l['prior_quote']['required_including_this_project']-credit+l['profile_wrapper_reserve_bytes']
 required=base+q['maximum_output_bytes']-used;assert base>=58<<30
 st=os.statvfs(Path(q['project_root']).parent);free=st.f_bavail*st.f_frsize;assert free>=required,(free,required)
 mi={s.split(':',1)[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if ':' in s and len(s.split())>=2 and s.split()[1].isdigit()}
 head=mi['MemAvailable'];mx=Path('/sys/fs/cgroup/memory.max');cu=Path('/sys/fs/cgroup/memory.current')
 if mx.exists() and cu.exists() and mx.read_text().strip()!='max':head=min(head,int(mx.read_text())-int(cu.read_text()))
 assert head>=(24<<30 if starting else 2<<30)
 return dict(free_bytes=free,required_including_this_project=required,own_used_bytes=used,own_remaining_bytes=q['maximum_output_bytes']-used,
 CPU_headroom_bytes=head,settled_closed_unused_bytes=credit,all_actual_files_retained=True,all_other_forecasts_and_50GiB_floor_8GiB_buffer_held=True)
