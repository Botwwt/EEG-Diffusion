"""D11 privileged visual-teacher diagnostic; never an EEG inference method."""
import numpy as np
import brainhive_gain_calibration_20261027_v1 as U

SCHEMA = 'BrainHIVE_Dcheck_visual_teacher_reference_20261027_v1'
ARMS = ('oracle_F', 'same_concept_wrong_F', 'other_concept_F')


def source_rows(queries):
    assert len(queries) == 400
    concepts = [queries[i]['concept_id'] for i in range(0, 400, 10)]
    assert len(set(concepts)) == 40
    for i, concept in enumerate(concepts): assert all(q['concept_id'] == concept for q in queries[i*10:i*10+10])
    correct = np.arange(400)
    same = correct//10*10+(correct+1)%10
    permutation = U.derangement(40)
    other = (permutation[:, None]*10+np.arange(10)[None]).reshape(-1)
    assert all(queries[i]['concept_id'] == queries[j]['concept_id'] and i != j for i, j in enumerate(same))
    assert all(queries[i]['concept_id'] != queries[j]['concept_id'] for i, j in enumerate(other))
    return np.stack([correct, same, other], axis=1)


def score(generated, targets, baseline, queries, gallery, stage, d10_features):
    assert stage == 80 and generated.shape == (80, 3, 768) and baseline.shape == (400, 3, 768)
    assert d10_features.shape == (80, 2, 768) and targets.shape == (400, 768)
    references = {arm: U.S.metric_arrays(baseline[:80, j], targets, queries, gallery) for j, arm in enumerate(U.ARMS)}
    for j, arm in enumerate(('D10_support', 'D10_shuffled_EEG_support')):
        references[arm] = U.S.metric_arrays(d10_features[:, j], targets, queries, gallery)
    teachers = {arm: U.S.metric_arrays(generated[:, j], targets, queries, gallery) for j, arm in enumerate(ARMS)}
    contrasts = {arm: U.S.compare(teachers['oracle_F'], value, queries) for arm, value in (references | {a:v for a,v in teachers.items() if a != 'oracle_F'}).items()}
    def supports(arm, metrics):
        return all(contrasts[arm][metric]['difference'] > 0 and contrasts[arm][metric]['descriptive_concept_bootstrap95'][0] > 0 for metric in metrics)
    semantic = supports('other_concept_F', ('target_cosine', 'DINO2AFC'))
    detail = supports('same_concept_wrong_F', ('target_cosine', 'within_concept10_top1'))
    parent = supports('parent', ('target_cosine', 'DINO2AFC'))
    return dict(status='COMPLETE_D11_PRIVILEGED_TEACHER_REFERENCE_READOUT', queries=80, concepts=8, gallery_targets=400,
        means={arm: {k: float(x.mean()) for k,x in value.items()} for arm,value in (references | teachers).items()},
        correct_teacher_minus_each_reference=contrasts,
        diagnostic_interpretation=dict(semantic_direction_supported=semantic, same_concept_stimulus_direction_supported=detail,
            oracle_exceeds_native_parent_on_registered_metrics=parent,
            rule='all named metric differences and descriptive95 lower bounds strictly positive; otherwise not resolved'),
        decision=dict(action='DIAGNOSTIC_COMPLETE_NO_FIT_OR_CANDIDATE_PROMOTION', scientific_method_gain=False),
        diagnostic_only=True, oracle_uses_query_visual_supervision=True, normal_inference_supported=False,
        independent_confirmation=False, mathematical_upper_bound_established=False, SOTA_established=False)


def self_test():
    q = [dict(concept_id=str(i//10)) for i in range(400)]
    rows = source_rows(q); assert rows.shape == (400, 3)
    assert np.array_equal(np.sort(rows[:, 1]), np.arange(400)) and np.array_equal(np.sort(rows[:, 2]), np.arange(400))
    assert np.array_equal(rows[:, 0], np.arange(400))
    return dict(status='D11_SOURCE_MAPPING_CPU_CHECKS_PASSED', checks=3)
