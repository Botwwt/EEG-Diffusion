"""D10 frozen T-only visual support projection and registered readout."""
import numpy as np
import brainhive_gain_calibration_20261027_v1 as U

SCHEMA = 'BrainHIVE_T_visual_support_projection_20261027_v1'
ARMS = ('support', 'shuffled_EEG_support')


def nearest_support(conditions, bank):
    c = np.asarray(conditions, np.float64); f = np.asarray(bank, np.float64)
    assert c.ndim == f.ndim == 2 and c.shape[1] == f.shape[1]
    assert np.isfinite(c).all() and np.isfinite(f).all()
    cn = np.linalg.norm(c, axis=1); fn = np.linalg.norm(f, axis=1)
    assert cn.min() > 1e-10 and fn.min() > 1e-10 and len(f) >= 2
    f = f / fn[:, None]; indices = []; cosine = []; margins = []
    for lo in range(0, len(c), 64):
        score = (c[lo:lo+64] / cn[lo:lo+64, None]) @ f.T
        chosen = score.argmax(1)  # sorted T array rows, lowest row wins ties
        best = score[np.arange(len(chosen)), chosen]
        second = np.partition(score, -2, axis=1)[:, -2]
        indices.append(chosen); cosine.append(best); margins.append(best-second)
    return np.concatenate(indices), np.concatenate(cosine), np.concatenate(margins)


def control_permutation(queries):
    assert len(queries) == 400
    concepts = [queries[i]['concept_id'] for i in range(0, 400, 10)]
    assert len(set(concepts)) == 40
    for i, concept in enumerate(concepts):
        assert all(q['concept_id'] == concept for q in queries[10*i:10*i+10])
    group_order = U.derangement(40)
    order = (group_order[:, None]*10 + np.arange(10)[None]).reshape(-1)
    assert sorted(order.tolist()) == list(range(400))
    assert all(queries[i]['concept_id'] != queries[j]['concept_id'] for i, j in enumerate(order))
    return order


def score(generated, targets, baseline, queries, gallery, stage):
    assert stage in (80, 400) and generated.shape == (stage, 2, 768)
    assert baseline.shape == (400, 3, 768) and targets.shape == (400, 768)
    assert len(queries) == stage and len(gallery) == 400
    original = {arm: U.S.metric_arrays(baseline[:stage, i], targets, queries, gallery) for i, arm in enumerate(U.ARMS)}
    candidate = {arm: U.S.metric_arrays(generated[:, i], targets, queries, gallery) for i, arm in enumerate(ARMS)}
    contrasts = {arm: U.S.compare(candidate['support'], value, queries) for arm, value in original.items()}
    intervention = U.S.compare(candidate['support'], candidate['shuffled_EEG_support'], queries)
    decision = U.S.decision(contrasts['risk'], str(stage))
    if stage == 400 and decision['action'] == 'PROMISING_FOR_FULL_EVALUATION':
        if not all(U.S.decision(value, '400')['action'] == 'PROMISING_FOR_FULL_EVALUATION' for value in contrasts.values()):
            decision = decision | dict(action='STOP_INSTANCE', reason='NOT_PROMISING_AGAINST_ALL3_FROZEN_ARMS')
        elif intervention['DINO2AFC']['difference'] <= 0 or intervention['DINO2AFC']['descriptive_concept_bootstrap95'][0] <= 0:
            decision = decision | dict(action='STOP_INSTANCE', reason='NO_REGISTERED_PAIRED_EEG_INCREMENT_OVER_SHUFFLED_CONTROL')
    self_test = U.S.metric_arrays(targets, targets, gallery, gallery)
    assert np.all(self_test['DINO2AFC'] == 1) and np.all(self_test['gallery_top1'] == 1)
    return dict(status='COMPLETE_D10_FIXED_SUPPORT_PAIRED_READOUT', stage=stage, queries=stage,
        gallery_targets=400, concepts=stage//10,
        means={arm: {key: float(value.mean()) for key, value in values.items()} for arm, values in (original | candidate).items()},
        support_minus_each_baseline=contrasts, support_minus_shuffled_EEG=intervention,
        decision=decision, risk_chosen_as_primary_in_D09_Dfit_before_D09_check_images=True,
        exclusions=0, normal_inference_target_access=False, independent_confirmation=False, SOTA_established=False)


def self_test():
    bank = np.array([[2., 0.], [0., 7.], [-3., 0.]])
    ids, _, margins = nearest_support(np.array([[1., .1], [-1., .1]]), bank)
    assert ids.tolist() == [0, 2] and (margins > 0).all()
    tied, _, _ = nearest_support(np.array([[1., 0.]]), np.array([[2., 0.], [8., 0.], [0., 1.]]))
    assert tied.tolist() == [0]
    assert np.array_equal(bank[ids], np.array([[2., 0.], [-3., 0.]]))
    try: nearest_support(np.zeros((1, 2)), bank)
    except AssertionError: pass
    else: raise AssertionError('Zero norm must not silently select a prototype')
    queries = [dict(concept_id=str(i//10)) for i in range(400)]
    permutation = control_permutation(queries)
    assert np.array_equal(np.sort(permutation), np.arange(400))
    return dict(status='D10_CPU_MATH_CHECKS_PASSED', checks=5)
