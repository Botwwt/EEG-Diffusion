import hashlib,json
from pathlib import Path
import numpy as np
import AR_Dfit_context_diagnostic_math_CPU_20261029_v1 as M
L=Path(__file__).resolve().parent;groups=[f'concept{i:02d}' for i in range(20) for _ in range(10)];a=np.ones((200,5,10))
for i,value in enumerate((.8,1.2,1.5,1.,1.1)):a[:,i]=value
result=M.summarize(a,groups);expected=dict(own_minus_donor_EEG_at_GT_prefix=.4,own_minus_zero_EEG_at_GT_prefix=.7,own_minus_donor_EEG_at_donor_prefix=.1,GT_minus_donor_prefix_at_own_EEG=.2);checks={}
for name,value in expected.items():
    row=result['contrasts'][name];assert np.allclose(row['mean_log_likelihood_difference_nats_per_token'],value,atol=1e-12,rtol=0)
    assert np.allclose(row['descriptive_concept_bootstrap95_per_scale'],value,atol=1e-12,rtol=0);checks[name+'_sign_and_nats_units']=True
for name,values,concepts in [('missing_query',a[:-1],groups[:-1]),('wrong_condition_count',a[:,:4],groups),('NaN_not_omitted',a.copy(),groups),('bad_concept_groups',a,['same']*200)]:
    if name=='NaN_not_omitted':values[0,0,0]=np.nan
    try:M.summarize(values,concepts)
    except AssertionError:checks[name+'_rejected']=True
    else:raise AssertionError(name+' accepted invalid source')
assert result['current_training_or_image_policy_changed'] is False and result['multiple_contrasts_or_optional_looks_have_no_nominal_error_control'] is True
out=L/'audit/AR_Dfit_context_diagnostic_20261029_v1';out.mkdir(exist_ok=True);p=out/'math_CPU_actual_v1.json';assert not p.exists()
v=dict(status='PASS_DFIT_TEACHER_CONTEXT_CONTRAST_SIGNS_UNITS_AND_FULL_QUERY_CONTRACT_CPU',checks=checks,synthetic_input_only=True,real_models_or_EEG_or_images_used=False,GPU_jobs=0,diagnostic_results_available=False,source_sha256=hashlib.sha256((L/'AR_Dfit_context_diagnostic_math_CPU_20261029_v1.py').read_bytes()).hexdigest())
p.write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v))
