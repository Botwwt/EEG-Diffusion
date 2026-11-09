"""Scientific denominator, correlated metrics, and forward-gate tests."""
import json,numpy as np
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import ARMS,IMAGE_ARMS
from brainhive_paired_semantic_SEED_statistics_CPU_20261030_v1 import statistics,next_action
def main():
    checks=[];q=[dict(query_id=str(i),concept_id=str(i//10)) for i in range(400)]
    rng=np.random.default_rng(1729);base=rng.uniform(.1,.7,(400,6));base[:,3]=base[:,:3].mean(1);values={a:base.copy() for a in IMAGE_ARMS};values['joint_correct'][:,:3]+=.02;values['joint_correct'][:,3]=values['joint_correct'][:,:3].mean(1)
    result=statistics(values,q,ARMS);assert len(result['contrasts'])==5 and all(len(v)==6 for v in result['contrasts'].values());checks.append('all30_fixed_comparisons_use_full400')
    ci=result['contrasts']['joint_correct']['ordinary']['SEED'];assert abs(ci['difference']-.02)<1e-12 and max(abs(np.array(ci['descriptive_concept_bootstrap95'])-.02))<1e-12;checks.append('constant_paired_gain_exact')
    assert result['weighted_bootstrap_maximum_error']<2e-12;checks.append('independent_concept_weighted_draws_agree')
    zero=result['contrasts']['P_only_correct']['EEG_only_correct']['ObjectF1'];assert zero['difference']==0 and zero['descriptive_concept_bootstrap95']==[0.,0.];checks.append('zero_control_not_selected_as_gain')
    gate=dict(action='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION');assert next_action(gate,result['contrasts'])['action']==gate['action'];checks.append('positive_secondary_gate_can_support_next_evaluation')
    def rejects(fn):
        try:fn()
        except (AssertionError,ValueError,KeyError):return
        raise AssertionError('Expected rejection')
    rejects(lambda:statistics(values,q[:-1],ARMS));checks.append('missing_query_rejected')
    broken={k:v.copy() for k,v in values.items()};broken['joint_correct'][0,0]=np.nan;rejects(lambda:statistics(broken,q,ARMS));checks.append('undefined_metric_never_drops_denominator')
    broken={k:v.copy() for k,v in values.items()};broken['joint_correct'][0,3]+=.01;rejects(lambda:statistics(broken,q,ARMS));checks.append('SEED_really_means_three_proxies')
    rejects(lambda:next_action(dict(action='STOP_INSTANCE'),result['contrasts']));checks.append('no_secondary_rescue_of_failed_primary')
    r=result['contrasts'];r['joint_correct']['ordinary']['ObjectF1']['difference']=-.001;assert next_action(gate,r)['action']=='HOLD_FOR_ANALYSIS_NO_AUTOMATIC_MULTISUBJECT_FIT';checks.append('negative_ObjectF1_not_hidden')
    r['joint_correct']['ordinary']['ObjectF1']['difference']=.01;r['joint_correct']['ordinary']['SEED']['descriptive_concept_bootstrap95'][0]=-.001;assert next_action(gate,r)['action'].startswith('HOLD');checks.append('SEED_interval_crossing_not_confirmed')
    from audit_HIVE_paired_semantic_SEED_CPU_20261030_v1 import manual_F1
    from image_evaluation import object_f1_pair
    def row(labels,scores=None,boxes=None):return dict(labels=labels,scores=scores or [.95]*len(labels),bboxes=boxes or [[0,0,10,10] for v in labels])
    fixtures=[(row([]),row([]),1.),(row([1]),row([1]),1.),(row([1]),row([2]),0.),(row([]),row([1]),0.),(row([1,1]),row([1]),1.),(row([1],boxes=[[0,0,7,7]]),row([]),1.)]
    for left,right,expected in fixtures:assert abs(manual_F1(left,right,np)-expected)<2e-12 and abs(manual_F1(left,right,np)-object_f1_pair(left,right))<2e-12
    checks.append('independent_six_F1_empty_threshold_area_and_duplicate_fixtures')
    import torch;assert not torch.cuda.is_initialized();checks.append('all_checks_CPU_only')
    print(json.dumps(dict(status='PASS_ACTUAL_SMALL_SEED_STATISTICS_CONTRACT_CHECKS',checks=checks,real_new_images_or_effects=False,GPU_calls=0)))
if __name__=='__main__':main()
