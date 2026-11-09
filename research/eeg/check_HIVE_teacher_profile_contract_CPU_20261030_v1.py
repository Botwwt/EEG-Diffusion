import json
from brainhive_teacher_profile_contract_CPU_20261030_v1 import dependency_state

dependencies=[dict(PID=11,start_ticks=101,failure='fitfail',completion='fitdone',success_status='fit_OK',completion_link_field='protocol',protocol=dict(path='fit',sha256='a')),dict(PID=12,start_ticks=102,failure='evalfail',completion='evaldone',success_status='eval_OK',completion_link_field='study',protocol=dict(path='eval',sha256='b'))]
done={d['completion']:dict(status=d['success_status'],**{d['completion_link_field']:d['protocol']}) for d in dependencies}
checks=[]
assert not dependency_state(dependencies,lambda pid,ticks:True,{}.get)['ready'];checks.append('two_live_exact_handles_wait_without_GPU')
assert not dependency_state(dependencies,lambda pid,ticks:pid==12,done.get)['ready'];checks.append('evaluation_keeps_dependency_closed_after_training_success')
assert dependency_state(dependencies,lambda pid,ticks:False,done.get)['ready'];checks.append('both_exact_protocol_linked_terminal_successes_admit_only_engineering')
def rejects(data,live=lambda pid,ticks:False):
    try:dependency_state(dependencies,live,data.get)
    except RuntimeError:return
    raise AssertionError('Unproven upstream study must be rejected')
rejects({});checks.append('missing_exact_handles_without_completion_do_not_restart_or_admit')
rejects(dict(done,fitfail=dict(error='failed')),lambda pid,ticks:True);checks.append('failure_is_not_overridden_by_an_old_or_live_handle')
wrong={**done,'evaldone':dict(status='eval_OK',study=dict(path='other',sha256='b'))};rejects(wrong);checks.append('same_status_different_protocol_rejected')
wrong={**done,'evaldone':dict(status='NOT_COMPLETE',study=dependencies[1]['protocol'])};rejects(wrong);checks.append('partial_closure_rejected')
print(json.dumps(dict(status='PASS_DEPENDENT_TEACHER_PROFILE_CPU_CONTRACT',checks=checks,new_GPU_calls=0)))
