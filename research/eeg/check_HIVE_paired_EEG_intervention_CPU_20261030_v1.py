"""Counterfactual EEG source, finite denominator, and interaction checks."""
import copy,json,numpy as np
from brainhive_paired_EEG_intervention_CPU_20261030_v1 import *
def main():
    checks=[];q=[dict(query_id=str(i),stimulus_id=f'{i:03d}',concept_id=str(i//10),image_seed=1729+i) for i in range(400)];maps=mapping(q)
    assert len(set(maps['within']))==len(set(maps['across']))==400;checks.append('both_maps_are_full_permutations')
    assert all(q[i]['concept_id']==q[j]['concept_id'] and i!=j for i,j in enumerate(maps['within']));checks.append('within_is_other_stimulus_sameconcept')
    assert all(q[i]['concept_id']!=q[j]['concept_id'] for i,j in enumerate(maps['across']));checks.append('across_is_other_concept')
    for arm in GENERATED_ARMS:
        for i in range(400):base,j,seed=sampling_spec(arm,i,q,maps);assert base in BASES and j!=i and seed==q[i]['image_seed'] and seed!=q[j]['image_seed']
    checks.append('all1600_sources_resolve_without_changing_queryseed')
    def rejects(fn):
        try:fn()
        except (AssertionError,ValueError,KeyError):return
        raise AssertionError('Expected rejection')
    bad=copy.deepcopy(q);bad[1]['stimulus_id']=bad[0]['stimulus_id'];rejects(lambda:mapping(bad));checks.append('duplicate_stimulus_rejected')
    rejects(lambda:mapping(q[:-1]));checks.append('missing_query_rejected')
    values={a:{k:np.zeros(400) for k in METRICS} for a in ALL_ARMS}
    for k in METRICS:values['joint_correct'][k][:]=.8;values['ordinary'][k][:]=.6;values['joint_within'][k][:]=values['joint_across'][k][:]=.1;values['ordinary_within'][k][:]=values['ordinary_across'][k][:]=.2
    result,raw,interpret,error=comparisons(values,q);assert all(interpret[k] for k in ('uses_paired_EEG_for_coarse_reconstruction','uses_paired_EEG_for_withinconcept_identity','incremental_withinconcept_EEG_dependence_over_ordinary'));checks.append('positive_paired_and_interaction_effects_detected')
    assert len(result)==len(raw)==9 and error<2e-12;checks.append('all9_contrasts_and_weighted_bootstrap')
    for k in METRICS:assert abs(raw['incremental_within_EEG_dependence'][:,METRICS.index(k)].mean()-.3)<1e-12
    checks.append('difference_of_differences_is_explicit')
    for k in METRICS:values['joint_within'][k][:]=.5;values['joint_across'][k][:]=.5;values['ordinary_within'][k][:]=.3;values['ordinary_across'][k][:]=.3
    result,raw,interpret,error=comparisons(values,q);assert not interpret['incremental_withinconcept_EEG_dependence_over_ordinary'];checks.append('equal_EEG_dependence_not_falsely_incremental')
    values['joint_correct']['DINO2AFC'][0]=np.nan;rejects(lambda:comparisons(values,q));checks.append('undefined_query_retains_full_denominator_and_rejects')
    print(json.dumps(dict(status='PASS_FROZEN_EEG_INPUT_INTERVENTION_CPU_CHECKS',checks=checks,new_models_or_images=0,GPU_calls=0)))
if __name__=='__main__':main()
