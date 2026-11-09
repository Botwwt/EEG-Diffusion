import json,numpy as np
from brainhive_seed_joint_statistics_20261029_v1 import joint_statistics
checks=[]
for value in (0.,.025,-.025):
 x=joint_statistics(np.full((3,10,200),value));assert abs(x['mean_difference']-value)<1e-14 and max(abs(np.array(x['descriptive_seed_subject_concept_bootstrap95'])-value))<1e-14
checks.append('zero_positive_negative_constant_effects_keep_full6000_and_exact_intervals')
d=np.broadcast_to(np.array([-.04,0.,.04])[:,None,None],(3,10,200));x=joint_statistics(d);assert np.allclose(x['per_seed_mean_differences'],[-.04,0,.04]) and x['descriptive_seed_subject_concept_bootstrap95'][0]<0<x['descriptive_seed_subject_concept_bootstrap95'][1];checks.append('between_run_variation_not_erased_by_duplicating_queries')
rng=np.random.default_rng(9);d=rng.normal(0,.1,(3,10,200));a=joint_statistics(d);b=joint_statistics(-d);assert np.allclose(a['descriptive_seed_subject_concept_bootstrap95'],-np.array(b['descriptive_seed_subject_concept_bootstrap95'])[::-1]);checks.append('paired_difference_sign_and_interval_reverse_together')
d[0,0,0]=np.nan
try:joint_statistics(d)
except AssertionError:checks.append('undefined_query_rejected_no_silent_denominator_shrink')
else:raise AssertionError('Dropped undefined query')
print(json.dumps(dict(status='PASS_CROSSED_SEED_SUBJECT_CONCEPT_DESCRIPTIVE_STATISTICS',checks=checks,actual_newseed_effect_measured=False)))
