import json
import numpy as np
from brainhive_joint_student_followon_contract_CPU_20261030_v1 import dependencies_ready,validate_teacher_arrays
checks=[]
dependency=dict(PID=17,start_ticks=123,failure='failure',completion='completion',success_status='DONE',completion_protocol_field='profile.protocol',protocol=dict(path='old',sha256='x'))
done={'completion':dict(status='DONE',profile=dict(protocol=dependency['protocol']))}
assert not dependencies_ready([dependency],lambda *unused:True,done.get)['ready'];checks.append('live_exact_upstream_waits_even_if_completion_file_exists')
assert dependencies_ready([dependency],lambda *unused:False,done.get)['ready'];checks.append('terminal_exact_nested_protocol_success_admits_only_followon_engineering')
for data in ({},{'completion':dict(status='DONE',profile=dict(protocol=dict(path='other',sha256='x')))},{**done,'failure':dict(error='bad')}):
    try:dependencies_ready([dependency],lambda *unused:False,data.get)
    except RuntimeError:pass
    else:raise AssertionError('Unproven dependency admitted')
checks.append('missing_handle_wrong_nested_protocol_or_failure_rejected')
rng=np.random.default_rng(1729);arms=['correct_EEG','visual_only','sameconcept_wrong_EEG'];data={}
for j in range(8):
    for arm in arms:
        mean=rng.normal(size=1023);mean/=np.linalg.norm(mean);kappa=np.array(100000.);key=f'{j}_{arm}'
        data.update({key+'_mean':mean,key+'_concentration':kappa,key+'_moment':(mean*kappa/(kappa+1022))[None]})
assert validate_teacher_arrays(data,arms)['teachers']==24;checks.append('all24_teachers_and_analytic_PowerSpherical_moments_required')
for mutation in ('missing','wrongmoment','wrongshape'):
    bad=dict(data)
    if mutation=='missing':bad.pop('0_correct_EEG_mean')
    elif mutation=='wrongmoment':bad['0_correct_EEG_moment']=np.zeros((1,1023))
    else:bad['0_correct_EEG_mean']=np.ones(1024)
    try:validate_teacher_arrays(bad,arms)
    except ValueError:pass
    else:raise AssertionError('Invalid teacher artifact admitted')
checks.append('partial_wrong_moment_or_wrong_coordinate_artifacts_rejected')
print(json.dumps(dict(status='PASS_JOINT_STUDENT_FOLLOWON_CPU_CONTRACT',checks=checks,new_GPU_calls=0)))
