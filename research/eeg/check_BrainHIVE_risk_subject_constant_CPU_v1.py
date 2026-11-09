"""Exact small mathematical checks; no Torch import or real tensor reads."""
import argparse
import json
import math
from pathlib import Path
import sys

import audit_BrainHIVE_risk_subject_constant_CPU_v1 as audit


def near(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def check():
    # Subject-mean confounding: pooled MSE is .26, subject constants reduce it to .01.
    fit = {'a': [.2, .4], 'b': [1.2, 1.4]}
    check_values = {'a': [.2, .4], 'b': [1.2, 1.4]}
    confounded = audit.subject_constant_statistics(fit, check_values, .005)
    near(confounded['aggregate']['reconstructed_Dfit_pooled_constant_risk'], .8)
    near(confounded['aggregate']['reconstructed_Dcheck_MSE_using_Dfit_pooled_constant'], .26)
    near(confounded['aggregate']['Dcheck_MSE_using_Dfit_subject_constant'], .01)
    near(confounded['aggregate']['head_relative_MSE_reduction_vs_subject_constant'], .5)
    # No between-subject mean difference: the two constant predictors coincide.
    unconfounded = audit.subject_constant_statistics({'a': [.2, .4], 'b': [.2, .4]},
        {'a': [.1, .5], 'b': [.1, .5]}, .03)
    near(unconfounded['aggregate']['reconstructed_Dcheck_MSE_using_Dfit_pooled_constant'], .04)
    near(unconfounded['aggregate']['Dcheck_MSE_using_Dfit_subject_constant'], .04)
    near(unconfounded['aggregate']['subject_constant_relative_MSE_reduction_vs_pooled'], 0)
    # Dcheck mean shifts to .6; using it illicitly would report .01 rather than .10.
    shifted = audit.subject_constant_statistics({'a': [.2, .4]}, {'a': [.5, .7]}, .02)
    near(shifted['per_subject'][0]['Dfit_subject_constant_risk'], .3)
    near(shifted['per_subject'][0]['Dcheck_observed_mean_risk'], .6)
    near(shifted['aggregate']['Dcheck_MSE_using_Dfit_subject_constant'], .10)
    assert not math.isclose(shifted['aggregate']['Dcheck_MSE_using_Dfit_subject_constant'], .01)
    # Aggregation is weighted by query counts, not an unweighted mean of subjects.
    weighted = audit.subject_constant_statistics({'a': [0.], 'b': [1.]}, {'a': [0.], 'b': [0., 0., 0.]}, .1)
    near(weighted['aggregate']['Dcheck_MSE_using_Dfit_subject_constant'], .75)
    for bad in (float('nan'), -0.1, 2.1):
        try:
            audit.subject_constant_statistics({'a': [bad]}, {'a': [.5]}, .1)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid risk accepted')
    assert 'torch' not in sys.modules
    return dict(schema='BrainHIVE_risk_subject_constant_CPU_contract_v1', status='PASS_SYNTHETIC_MATH_ONLY',
        checks=['subject_mean_confounding', 'no_subject_mean_confounding',
            'Dfit_mean_used_despite_Dcheck_mean_shift', 'query_weighted_aggregation', 'finite_0_to_2_risk_guard'],
        expected_MSE=dict(confounded_pooled=.26, confounded_subject=.01,
            unconfounded_pooled_and_subject=.04, shifted_Dfit_constant=.10),
        source_sha256={Path(__file__).name: audit.sha(__file__), Path(audit.__file__).name: audit.sha(audit.__file__)},
        actual_geometry_tensors_read=False, head_replayed=False, GPU_used=False, torch_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    output = check()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(json.dumps(output, sort_keys=True, indent=2) + '\n')
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
