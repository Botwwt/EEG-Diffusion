"""Exercise eight-path ranks, fixed hypothesis gates and interaction equations."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
import audit_HIVE_noordinary_terminal_local_20261101_v1 as A

def main():
    checks=[];q=[dict(stimulus_id=str(i),concept_id=str(i//10)) for i in range(400)];v=np.eye(400,768)
    m=A.metrics(v[:80],v,q[:80],q);assert np.array_equal(m,np.ones((80,5)));checks.append('all80_self_identification_in400_gallery')
    ties=A.metrics(np.ones((80,768)),v,q[:80],q)
    assert np.array_equal(ties[:,1],np.full(80,.5)) and np.array_equal(ties[:,4],np.full(80,.5))
    assert ties[:,2].sum()==1 and ties[:,3].sum()==8;checks.append('half_credit_ties_and_withinconcept_indices')
    def comparisons(d=.02,lo=.01,hi=.03,top=.01):
        return {a:{b:{'DINO2AFC':dict(difference=d,descriptive_concept_bootstrap95=[lo,hi]),'within_concept10_top1':dict(difference=top)} for b in A.METHODS if b!=a} for a in A.CANDIDATES}
    assert len(A.METHODS)==8 and len(A.CANDIDATES)==6 and sum(map(len,comparisons().values()))==42
    checks.append('exactly_three_fits_eight_paths_and42_comparisons')
    c=comparisons(-.03,-.04,-.01);assert A.decision(c,80)['action']=='STOP_INSTANCE';checks.append('both_modes_harm_stops80')
    c['joint_correct_plus_ordinary']['ordinary']['DINO2AFC']['difference']=0
    assert A.decision(c,80)['action']=='EXPAND_ALL_TO400';checks.append('one_ambiguous_mode_expands_all')
    c=comparisons();assert all(x['supported'] for x in A.decision(c,400)['modes'].values());checks.append('both_declared400_support_paths')
    c['joint_correct']['joint_swapped_targets']['DINO2AFC']['difference']=0
    d=A.decision(c,400);assert not d['modes']['standalone']['supported'] and d['modes']['composition']['supported'];checks.append('composition_never_rescues_failed_standalone_claim')
    c['joint_correct_plus_ordinary']['joint_swapped_targets_plus_ordinary']['DINO2AFC']['difference']=0
    assert A.decision(c,400)['action']=='STOP_INSTANCE';checks.append('neither_control_failure_is_ignored')
    assert A.decision(comparisons(top=-.01),400)['action']=='STOP_INSTANCE';checks.append('withinconcept_harm_blocks_both400_modes')
    rng=np.random.default_rng(1729);delta=rng.normal(size=(80,5));result=A.contrast(delta,q[:80]);groups=[np.arange(i,i+10) for i in range(0,80,10)]
    draws=np.random.default_rng(1729).integers(0,8,(1000,8));samples=np.array([np.concatenate([delta[groups[j]] for j in row]).mean(0) for row in draws]);ci=np.quantile(samples,[.025,.975],axis=0)
    assert max(np.max(np.abs(np.array(result[k]['descriptive_concept_bootstrap95'])-ci[:,j])) for j,k in enumerate(A.KEYS))<2e-12
    checks.append('count_weight_bootstrap_matches_direct_resampled_queries')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'actual/images').mkdir(parents=True)
        levels=dict(parent=0.,ordinary=.2,joint_correct=.3,joint_correct_plus_ordinary=.6,joint_swapped_targets=.1,joint_swapped_targets_plus_ordinary=.25,joint_absolute_only=.05,joint_absolute_only_plus_ordinary=.25)
        fields={a+'__'+k:np.full(80,t) for a,t in levels.items() for k in A.KEYS}
        np.savez_compressed(root/'actual/images/all_query_values80.npz',**fields)
        known=dict(joint_correct=.1,joint_swapped_targets=-.05,joint_absolute_only=0.)
        expected=dict(stage=80,does_not_rescue_image_gate=True,interaction={a:{k:dict(difference_in_differences=t,descriptive_concept_bootstrap95=[t,t]) for k in A.KEYS} for a,t in known.items()})
        path=root/'actual/composition_interaction.json';path.write_bytes(json.dumps(expected).encode())
        assert A.verify_composition(root,80,q[:80])['maximum_error']<2e-12
        checks.append('all3_analytic_positive_negative_zero_interactions_recomputed')
        expected['interaction']['joint_correct']['DINO2AFC']['difference_in_differences']=-.1
        path.write_bytes(json.dumps(expected).encode())
        try:A.verify_composition(root,80,q[:80])
        except AssertionError:pass
        else:raise AssertionError('Wrong interaction sign was accepted')
        checks.append('altered_reported_interaction_sign_rejected')
    out=Path(__file__).resolve().parent/'audit/HIVE_noordinary_terminal_reader_preparation_20261101_v1/actual_local_CPU_checks.json'
    value=dict(status='PASS_NEW_NOO_TERMINAL_LOCAL_MATH_CHECKS',checks=checks,actual_new_terminal_archive_NOT_available_yet=True,new_GPU_jobs=0)
    if out.exists():out=out.with_name('actual_local_CPU_checks_v2.json')
    assert not out.exists();out.write_bytes((json.dumps(value,indent=2)+'\n').encode());print(json.dumps(value))
if __name__=='__main__':main()
