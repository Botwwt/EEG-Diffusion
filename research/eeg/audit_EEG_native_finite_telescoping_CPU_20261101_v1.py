"""Independent direct finite sums, with exact conditional clipping diagnostics."""
import argparse
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import EEG_native_finite_telescoping_DRAFT_CPU_20261101_v1 as M


def run():
    cases = []
    for h, ell, scheme in [
        ([.3,-.9,1.1],[-3.,-1.,-5.],M.Hierarchy(3,(),(1,2,3),(.7,.3))),
        ([.3,-.9,1.1,.6],[-3.,-1.,-5.,-2.],M.Hierarchy(4,(1,),(1,2,3),(.8,.2))),
        ([.8,-.1,.4,-.7],[-5.,-5.,-5.,-5.],M.Hierarchy(4,(),(1,2,4),(.9,.1))),
        ([-.3,.9,-1.1,.6,.2],[-3.,-1.,-5.,-2.,-7.],M.Hierarchy(5,(0,),(1,2,4),(.6,.4)))
    ]:
        K = len(h)
        # Direct math.fsum reference; no producer's logsumexp/dense_reference.
        denominator = math.fsum(math.exp(v) for v in h)
        prior = [math.exp(v)/denominator for v in h]
        joint = [prior[c]*math.exp(ell[c]) for c in range(K)]
        Z = math.fsum(joint)
        exact_g = np.array([prior[c]-joint[c]/Z for c in range(K)])
        rows = []
        orders = list(itertools.permutations(scheme.tail))
        comparisons = []
        for order in orders:
            targets = []
            for size in scheme.tail_sizes:
                head = set(scheme.head)
                selected_tail = set(order[:size])
                weights = [joint[c]*(1. if c in head else len(scheme.tail)/size)
                           if c in head or c in selected_tail else 0. for c in range(K)]
                total = math.fsum(weights)
                targets.append((-math.log(total),np.array([prior[c]-weights[c]/total for c in range(K)])))
            for j,p in enumerate(scheme.level_probabilities,start=1):
                loss = targets[0][0]+(targets[j][0]-targets[j-1][0])/p
                gradient = targets[0][1]+(targets[j][1]-targets[j-1][1])/p
                actual = M.query_hierarchy(np.array(h),scheme,order,j,
                                          lambda ids:np.array([ell[c] for c in ids]))
                comparisons.append(abs(loss-actual['negative_log_evidence']))
                comparisons.append(float(np.max(np.abs(gradient-actual['negative_log_evidence_logit_gradient']))))
                clipped = gradient/max(1.,float(np.linalg.norm(gradient)))
                rows.append(dict(weight=p/len(orders),loss=loss,gradient=gradient,clipped=clipped))
        mean = np.array([math.fsum(r['weight']*r['gradient'][c] for r in rows) for c in range(K)])
        clipped_mean = np.array([math.fsum(r['weight']*r['clipped'][c] for r in rows) for c in range(K)])
        expected_loss = math.fsum(r['weight']*r['loss'] for r in rows)
        assert np.max(np.abs(mean-exact_g)) < 5e-14
        assert abs(expected_loss+math.log(Z)) < 2e-14
        assert max(comparisons) < 1e-13
        variance = math.fsum(r['weight']*float(np.square(r['gradient']-exact_g).sum()) for r in rows)
        cases.append(dict(classes=K,head=list(scheme.head),sizes=list(scheme.tail_sizes),
                          probabilities=list(scheme.level_probabilities),
                          exhaustive_realizations=len(rows),
                          exact_gradient=exact_g.tolist(),expected_gradient=mean.tolist(),
                          expected_unit_L2_clipped_gradient=clipped_mean.tolist(),
                          clipping_mean_bias_L2=float(np.linalg.norm(clipped_mean-exact_g)),
                          unclipped_gradient_MSE=variance,
                          all_realization_producer_max_error=max(comparisons)))
    assert np.linalg.norm(cases[2]['exact_gradient']) < 1e-14
    assert cases[2]['clipping_mean_bias_L2'] > .001
    sources = [Path(__file__),Path(M.__file__)]
    return dict(status='PASS_INDEPENDENT_DIRECT_FINITE_SUMS_AND_CORRECTED_ESTIMATOR_CLIPPING_COUNTEREXAMPLE',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,
                primary_decision='DO_NOT_ADMIT_NAIVE_CORRECTION_WITH_EXISTING_GRADIENT_CLIPPING',
                actual_EEG_gradient_bias_measured=False,
                actual_VAR_variance_or_sparse_runtime_measured=False,
                classical_identity_not_novel=True, new_GPU_or_training_or_images=0,
                sources=[dict(path=str(p.resolve()),bytes=len(p.read_bytes()),
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    destination=Path(args.output)
    assert not destination.exists()
    result=run()
    raw=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
