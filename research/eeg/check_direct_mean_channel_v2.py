"""Synthetic CPU checks for data-role isolation and absence of a CI bypass."""
import json
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import torch
from direct_mean_error_channel_v2 import designs, fit_mean, predict, calibration, sample


def main():
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(1729)
    x = torch.randn(200, 4, generator=g)
    a = torch.tensor([[1., .2, 0., 0.], [0., 2., .1, 0.], [0., 0., 1., .4], [.1, 0., 0., .5]])
    y = x@a+torch.tensor([1., -2., .5, .25])
    folds = torch.arange(200)%2
    ds = designs(x, folds, .01)
    full = fit_mean(ds['full'], y)
    assert float((predict(full, x)-y).square().mean()) < .001
    changed = y.clone()
    changed[folds == 0] += 1000
    first = fit_mean(ds['holdout0'], y)
    second = fit_mean(ds['holdout0'], changed)
    assert torch.equal(first['weight'], second['weight']) and torch.equal(first['bias'], second['bias'])
    fit = torch.arange(200)<100
    residual = torch.randn(200, 4, generator=g)
    actual = y+torch.randn(200, 4, generator=g)
    s1 = calibration(y, actual, fit, residual)
    actual[~fit] += 1000
    s2 = calibration(y, actual, fit, residual)
    for key in ['residual_mean', 'bias']:
        assert torch.equal(s1[key], s2[key])
    assert s1['scale'] == s2['scale']
    constant = dict(weight=torch.zeros(4, 4), bias=torch.ones(4))
    neighbors = torch.arange(32).repeat(200, 1)
    seeds = list(range(1729, 1929))
    z1, indices1 = sample(predict(constant, x), residual, neighbors, seeds, s1)
    z2, indices2 = sample(predict(constant, x+10000), residual, neighbors, seeds, s1)
    assert torch.equal(z1, z2) and indices1 == indices2
    zero = dict(residual_mean=residual.mean(0), bias=torch.zeros(4), scale=0.)
    z3, _ = sample(predict(full, x), residual, neighbors, seeds, zero)
    assert torch.equal(z3, predict(full, x))
    try:
        calibration(y, actual, fit, torch.zeros_like(residual))
    except AssertionError:
        pass
    else:
        raise AssertionError('Degenerate residual distribution must be rejected')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASSED_SYNTHETIC_CPU_SOFTWARE_CHECKS',
        checks=['known_affine_signal_recovery', 'heldout_target_isolation',
                'Dcheck_not_used_by_calibration', 'no_original_CI_identity_bypass',
                'zero_noise_returns_conditional_mean', 'degenerate_residual_rejected'],
        scientific_fit_performed=False, images_generated=0, GPU_used=False)))


if __name__ == '__main__':
    main()
