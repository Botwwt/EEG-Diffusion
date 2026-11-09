import json
import numpy as np
from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import paired,fixed_draws,COMPARISONS
checks=[];si,ci=fixed_draws();x=np.arange(2000,dtype=np.float64).reshape(10,200)/2000
zero=paired(x,x,si,ci);assert zero['paired_left_minus_right']==0 and zero['descriptive_subject_concept_bootstrap95']==[0,0];checks.append('identical_outputs_remain_exact_zero')
positive=paired(x+.1,x,si,ci);assert abs(positive['paired_left_minus_right']-.1)<1e-14 and positive['descriptive_subject_concept_bootstrap95'][0]>0;checks.append('known_positive_constant_paired_gain')
negative=paired(x-.2,x,si,ci);assert negative['descriptive_subject_concept_bootstrap95'][1]<0;checks.append('negative_control_is_reported_negative_not_selected_away')
rng=np.random.default_rng(1);left=rng.normal(size=(10,200));right=rng.normal(size=(10,200));record=paired(left,right,si,ci);checks.append('crossed_subject_then_concept_resampling_matches_independent_einsum')
for a,b in ((x[:9],x[:9]),(np.full((10,200),np.nan),x)):
    try:paired(a,b,si,ci)
    except ValueError:pass
    else:raise AssertionError('Incomplete or nonfinite data admitted')
checks.append('partial_or_undefined_metrics_rejected_without_row_deletion')
assert COMPARISONS==( ('ordinary','F_only'),('ordinary','token_only'),('F_only','parent'),('token_only','parent'),('ordinary','parent') );checks.append('all_five_comparisons_predeclared_no_winner_selection')
assert all(np.array_equal(a,b) for stream1,stream2 in zip((si,ci),fixed_draws()) for a,b in zip(stream1,stream2));checks.append('fixed1729_bootstrap_stream_replay')
print(json.dumps(dict(status='PASS_SINGLE_SEED_R80_LOSS_CONTROL_STATISTICS_CPU',checks=checks,new_GPU_calls=0)))
