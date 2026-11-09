"""Exhaustible synthetic math checks; no EEG/VAR/image data or GPU invocation."""
import argparse
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import EEG_native_finite_telescoping_DRAFT_CPU_20261101_v1 as M


def exhaustive(h, ell, hierarchy):
    truth = M.dense_reference(h, ell)
    records = []
    mean_g = np.zeros(len(h))
    mean_loss = cost = mse = biased_mse = 0.
    mean_biased = np.zeros(len(h))
    permutations = list(itertools.permutations(hierarchy.tail))
    for order in permutations:
        for j, p in enumerate(hierarchy.level_probabilities, start=1):
            called = []
            def callback(ids):
                called.append(tuple(ids))
                return ell[list(ids)]
            value = M.query_hierarchy(h, hierarchy, order, j, callback)
            assert called == [value['queried_classes']]
            assert len(set(called[0])) == len(called[0])
            weight = p/len(permutations)
            g = value['negative_log_evidence_logit_gradient']
            bg = value['fine']['negative_log_evidence_logit_gradient']
            mean_g += weight*g
            mean_biased += weight*bg
            mean_loss += weight*value['negative_log_evidence']
            cost += weight*value['queried_class_count']
            mse += weight*float(np.square(g-truth['negative_log_evidence_logit_gradient']).sum())
            biased_mse += weight*float(np.square(bg-truth['negative_log_evidence_logit_gradient']).sum())
            records.append(value)
    ge = float(np.max(np.abs(mean_g-truth['negative_log_evidence_logit_gradient'])))
    le = abs(mean_loss-truth['negative_log_evidence'])
    assert ge < 2e-13 and le < 2e-12
    assert abs(cost-hierarchy.expected_queried_classes) < 1e-12
    return dict(permutations=len(permutations), levels=len(hierarchy.level_probabilities),
                exact_gradient=truth['negative_log_evidence_logit_gradient'].tolist(),
                exact_expected_corrected_gradient=mean_g.tolist(),
                max_gradient_error=ge, loss_error=le,
                expected_queried_classes=cost, full_classes=len(h),
                corrected_gradient_MSE=mse,
                uncorrected_fine_gradient_MSE=biased_mse,
                uncorrected_fine_mean_gradient_bias_L2=float(np.linalg.norm(
                    mean_biased-truth['negative_log_evidence_logit_gradient'])),
                corrected_draw_min_loss=min(r['negative_log_evidence'] for r in records),
                corrected_draw_max_gradient_norm=max(float(np.linalg.norm(
                    r['negative_log_evidence_logit_gradient'])) for r in records)), records


def check():
    t0 = time.monotonic()
    cases = []
    for h, ell, hierarchy in [
        (np.array([.3,-.9,1.1]), np.array([-3.,-1.,-5.]), M.Hierarchy(3,(),(1,2,3),(.7,.3))),
        (np.array([.3,-.9,1.1,.6]), np.array([-3.,-1.,-5.,-2.]), M.Hierarchy(4,(1,),(1,2,3),(.8,.2))),
        (np.array([.8,-.1,.4,-.7]), np.array([-5.,-5.,-5.,-5.]), M.Hierarchy(4,(),(1,2,4),(.9,.1))),
        (np.array([-.3,.9,-1.1,.6,.2]), np.array([-3.,-1.,-5.,-2.,-7.]), M.Hierarchy(5,(0,),(1,2,4),(.6,.4)))
    ]:
        result, _ = exhaustive(h, ell, hierarchy)
        cases.append(result)
    assert np.linalg.norm(cases[2]['exact_gradient']) < 1e-14
    assert cases[2]['uncorrected_fine_mean_gradient_bias_L2'] > 1e-3

    h = np.array([.3,-.9,1.1,.6])
    ell = np.array([-3.,-1.,-5.,-2.])
    hierarchy = M.Hierarchy(4,(1,),(1,2,3),(.8,.2))
    analytic = M.dense_reference(h, ell)['negative_log_evidence_logit_gradient']
    fd = np.zeros(4)
    for k in range(4):
        step = np.zeros(4); step[k] = 1e-5
        # Independently computed normalized finite mixture, no producer lse.
        def objective(v):
            p = np.exp(v)/np.exp(v).sum()
            return -math.log(sum(float(p[c])*math.exp(float(ell[c])) for c in range(4)))
        fd[k] = (objective(h+step)-objective(h-step))/(2e-5)
    assert np.max(np.abs(analytic-fd)) < 2e-10

    order = (0,2,3)
    def at(e):
        return M.query_hierarchy(h, hierarchy, order, 1, lambda ids:e[list(ids)])
    a, b = at(ell), at(ell+1000.)
    shift_error = float(np.max(np.abs(a['negative_log_evidence_logit_gradient']-
                                     b['negative_log_evidence_logit_gradient'])))
    assert shift_error < 5e-13
    assert abs((b['negative_log_evidence']-a['negative_log_evidence'])+1000.) < 2e-12

    # Complete endpoint equality for every permutation, not only its mean.
    endpoint_error = 0.
    for order in itertools.permutations(hierarchy.tail):
        value = M.query_hierarchy(h, hierarchy, order, 2, lambda ids:ell[list(ids)])
        endpoint_error = max(endpoint_error, float(np.max(np.abs(
            value['fine']['negative_log_evidence_logit_gradient']-analytic))))
    assert endpoint_error < 1e-13

    refusals = []
    for label, invalid in [
        ('missing_full_endpoint', M.Hierarchy(4,(),(1,2,3),(.5,.5))),
        ('zero_probability_full_correction', M.Hierarchy(4,(),(1,2,4),(1.,0.))),
        ('negative_probability', M.Hierarchy(4,(),(1,2,4),(1.1,-.1))),
        ('duplicate_head', M.Hierarchy(4,(1,1),(1,2),(1.,))),
        ('non_increasing_levels', M.Hierarchy(4,(),(1,2,2,4),(.3,.3,.4)))
    ]:
        try:
            invalid.validate()
        except AssertionError:
            refusals.append(label)
        else:
            raise AssertionError('Invalid hierarchy was accepted: '+label)

    # Generic exact counterexample: an unbiased random gradient can point in
    # the opposite direction in expectation AFTER nonlinear L2 clipping.
    draws = np.array([[10.,-10.],[-1.,1.]])
    probabilities = np.array([.1,.9])
    true_mean = probabilities@draws
    clipped = draws/np.maximum(1.,np.linalg.norm(draws,axis=1))[:,None]
    clipped_mean = probabilities@clipped
    assert float(true_mean@clipped_mean) < 0.
    clipping = dict(raw_mean=true_mean.tolist(), clipped_mean=clipped_mean.tolist(),
                    dot_with_raw_mean=float(true_mean@clipped_mean),
                    scope='GENERIC_COUNTEREXAMPLE_NOT_MEASURED_EEG_GRADIENTS')

    # Reusing the same random bank is not conditionally unbiased once theta
    # depends on that bank. Fresh noise at theta_1 still has expectation zero.
    eta = .2
    stale = []
    for epsilon in (-1.,1.):
        theta_1 = -eta*epsilon
        stale.append(dict(theta_1=theta_1, stale_conditional_gradient=epsilon,
                          exact_target_gradient=0., fresh_conditional_gradient_mean=0.))
        assert epsilon == -theta_1/eta

    # A finite endpoint does NOT justify dropping rare full-level draws.
    _, records = exhaustive(h, ell, hierarchy)
    truncated = [r for r in records if r['level']==1]
    truncated_mean = np.mean([r['negative_log_evidence_logit_gradient'] for r in truncated],axis=0)
    truncation_error = float(np.linalg.norm(truncated_mean-analytic))
    assert truncation_error > 1e-3

    # Sparse access test: unqueried likelihood values can be unspecified.
    queried = []
    def sparse(ids):
        queried.extend(ids)
        assert tuple(ids)==(1,0,2)
        return ell[list(ids)]
    value = M.query_hierarchy(h,hierarchy,(0,2,3),1,sparse)
    assert 3 not in queried and value['queried_class_count']==3

    # This checks the sampler's deterministic API, not RNG unbiasedness.
    sample_a = M.sample(h,hierarchy,lambda ids:ell[list(ids)],np.random.default_rng(1729))
    sample_b = M.sample(h,hierarchy,lambda ids:ell[list(ids)],np.random.default_rng(1729))
    assert sample_a['queried_classes']==sample_b['queried_classes']
    assert np.array_equal(sample_a['negative_log_evidence_logit_gradient'],
                          sample_b['negative_log_evidence_logit_gradient'])

    paths = [Path(M.__file__),Path(__file__)]
    return dict(status='PASS_SYNTHETIC_FINITE_TELESCOPING_PROOF_AND_COUNTEREXAMPLES_NOT_EEG_EFFICACY',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                seconds=time.monotonic()-t0,
                classical_prior='https://proceedings.mlr.press/v130/shi21d.html',
                full_permutation_cases=cases,
                dense_finite_difference_max_error=float(np.max(np.abs(analytic-fd))),
                shared_likelihood_offset_gradient_max_error=shift_error,
                full_endpoint_max_error=endpoint_error, rejected_hierarchies=refusals,
                clipping_counterexample=clipping,
                stale_bank_conditional_counterexample=stale,
                dropped_full_level_gradient_bias_L2=truncation_error,
                callback_queries_selected_union_once=True,
                full_endpoint_probability_must_remain_positive=True,
                native_GPU_executions=0, EEG_data_or_models=0, new_research_images=0,
                new_training_admission=False, proven_speedup=False, novelty_claim=False,
                mathematical_scope='FINITE_GLOBAL_CLASS_FIXED_LOGITS_UNCLIPPED_ONLY',
                sources=[dict(path=str(p.resolve()),bytes=len(p.read_bytes()),
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    target = Path(args.output)
    assert not target.exists(), 'Preserve the previous check output.'
    result = check()
    raw = (json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
