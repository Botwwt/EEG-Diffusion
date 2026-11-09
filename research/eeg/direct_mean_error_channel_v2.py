"""CPU-only visual-to-EEG mean and cross-fitted residual simulator.

This is an exploratory replacement of the v1 residual transplant, not a trained
image generator. All means are fitted on T; only calibration uses Dfit. A sample
never adds the original CI back to the predicted conditional mean.
"""
import hashlib
import torch


def designs(ci, folds, relative_ridge):
    assert ci.device.type == 'cpu' and ci.dtype == torch.float32
    assert folds.shape == (len(ci),) and set(folds.tolist()) == {0, 1}
    result = {}
    for role in ['full', 'holdout0', 'holdout1']:
        mask = torch.ones(len(ci), dtype=torch.bool) if role == 'full' else folds != int(role[-1])
        x = ci[mask].double()
        mean = x.mean(0)
        x = x-mean
        covariance = x.T@x/len(x)
        ridge = relative_ridge*float(torch.trace(covariance))/ci.shape[1]
        assert ridge > 0
        factor = torch.linalg.cholesky(covariance+ridge*torch.eye(ci.shape[1], dtype=torch.float64))
        result[role] = dict(mask=mask, centered_x=x, mean=mean, factor=factor, ridge=ridge)
    return result


def fit_mean(design, ce):
    y = ce[design['mask']].double()
    mean_y = y.mean(0)
    weight = torch.cholesky_solve(design['centered_x'].T@(y-mean_y)/len(y), design['factor'])
    return dict(weight=weight.float(), bias=(mean_y-design['mean']@weight).float(),
                relative_ridge_rule='lambda=.01*trace(T_covariance)/dimensions',
                actual_ridge=design['ridge'], fitted_queries=len(y))


def predict(model, ci):
    return ci@model['weight']+model['bias']


def calibration(mean_predictions, actual_ce, fit_mask, residual):
    assert mean_predictions.shape == actual_ce.shape and fit_mask.dtype == torch.bool
    rmean = residual.double().mean(0)
    centered = residual.double()-rmean
    source_trace = centered.square().sum(-1).mean()
    errors = actual_ce[fit_mask].double()-mean_predictions[fit_mask].double()
    bias = errors.mean(0)
    target_trace = (errors-bias).square().sum(-1).mean()
    assert source_trace > 1e-12 and target_trace > 1e-12, 'Degenerate residual distribution'
    return dict(residual_mean=rmean.float(), bias=bias.float(),
                scale=float(torch.sqrt(target_trace/source_trace)),
                source_trace=float(source_trace), Dfit_target_trace=float(target_trace),
                Dfit_queries=int(fit_mask.sum()), roles=['T_crossfit_residual', 'Dfit'])


def sample(mean_predictions, residual, neighbors, seeds, state):
    assert neighbors.shape[0] == len(mean_predictions) == len(seeds)
    chosen = []
    for i, seed in enumerate(seeds):
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        chosen.append(int(neighbors[i, torch.randint(neighbors.shape[1], (), generator=generator)]))
    centered = residual[chosen]-state['residual_mean']
    result = mean_predictions+state['bias']+state['scale']*centered
    assert torch.isfinite(result).all()
    return result, chosen


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
