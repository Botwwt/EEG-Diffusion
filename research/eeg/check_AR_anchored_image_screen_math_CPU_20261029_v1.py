import copy,json
from AR_anchored_image_screen_math_20261029_v1 import stage_decision,validate_math_source
validate_math_source()
def contrast(gain,low=None,high=None,cos=.001,top=0.):
    return dict(DINO2AFC=dict(difference=gain,descriptive_concept_bootstrap95=[gain if low is None else low,gain if high is None else high]),target_cosine=dict(difference=cos),within_concept10_top1=dict(difference=top))
def cases(primary,control=None):return dict(paired_minus_native=primary,paired_minus_donor=control or contrast(.02),donor_minus_native=contrast(0))
checks={}
checks['80_equal_harm_threshold_stops']=stage_decision(cases(contrast(-.02,high=-.001)),80)['action']=='STOP_INSTANCE'
checks['80_uncertain_harm_expands']=stage_decision(cases(contrast(-.021,high=0)),80)['action']=='EXPAND400'
checks['80_small_gain_still_expands']=stage_decision(cases(contrast(.001)),80)['action']=='EXPAND400'
checks['400_no_rounding_up']=stage_decision(cases(contrast(.009875)),400)['action']=='STOP_INSTANCE'
checks['400_both_comparators_needed']=stage_decision(cases(contrast(.02),contrast(.009)),400)['action']=='STOP_INSTANCE'
checks['400_top1_harm_stops']=stage_decision(cases(contrast(.02,top=-.001)),400)['action']=='STOP_INSTANCE'
checks['400_interval_crossing_stops']=stage_decision(cases(contrast(.02,low=0)),400)['action']=='STOP_INSTANCE'
checks['400_exact_practical_threshold_passes']=stage_decision(cases(contrast(.01)),400)['action']=='PROMISING_FOR_COMPLETE_MULTISUBJECT_EVALUATION'
assert all(checks.values()),checks
print(json.dumps(dict(status='PASS_FIXED_AR_80_400_GATE_BOUNDARIES_AND_NO_ROUNDING',checks=checks,GPU_jobs=0,new_images=0)))
