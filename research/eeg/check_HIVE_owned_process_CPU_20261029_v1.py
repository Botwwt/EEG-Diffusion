import json,subprocess
from brainhive_owned_process_CPU_20261029_v1 import identity,stop_owned
checks=[];processes=[]
try:
 p=subprocess.Popen(['/bin/sleep','15'],start_new_session=True);processes.append(p);pin=identity(p.pid);assert pin and pin['pid']==pin['pgid'];stop_owned(p);assert p.poll() is not None;checks.append('direct_own_group_TERMed_and_reaped')
 leaf=subprocess.Popen(['/bin/sleep','15'],start_new_session=True);owner=subprocess.Popen(['/bin/sleep','.3'],start_new_session=True);processes += [leaf,owner];pin=identity(leaf.pid);stale=dict(pin);stale['start_ticks']+=1;stop_owned(owner,stale);assert leaf.poll() is None and owner.poll() is not None;checks.append('mismatched_start_identity_does_not_kill_live_other_group')
 pin=identity(leaf.pid);stop_owned(leaf,pin);assert leaf.poll() is not None;checks.append('exact_pinned_child_TERMed_and_reaped')
finally:
 for p in processes:
  if p.poll() is None:stop_owned(p)
print(json.dumps(dict(status='PASS_OWN_PROCESS_GROUP_IDENTITY_AND_CLEANUP_CPU',checks=checks,new_GPU_calls=0,only_spawned_sleep_processes_touched=True)))
