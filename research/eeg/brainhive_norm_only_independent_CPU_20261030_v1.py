"""Independent bracketed scalar solve of the affine-norm quadratic.

No Torch/model imports. The full raw Euclidean output is an observed input;
this intentionally does not replace the original failed full-T MLP guard.
"""
import numpy as np
from scipy.optimize import brentq


def independent_projection(condition, raw, weight, bias):
    c, z, w, b = [np.asarray(x, np.float64) for x in (condition, raw, weight, bias)]
    assert c.shape == z.shape == w.shape == b.shape and c.shape[1] == 1024
    assert np.all(abs(w) > 1e-6) and np.isfinite(z).all()
    r2 = np.einsum('ij,ij->i', (c-b)/w, (c-b)/w)
    a = np.einsum('ij,ij->i', z/w, z/w)
    cross = np.einsum('ij,ij->i', z/w, b/w)
    d = np.einsum('ij,ij->i', b/w, b/w) - r2
    assert np.all(a > 0) and np.all(d < 0)
    factors = []
    for ai, bi, di in zip(a, cross, d):
        def f(s): return ai*s*s-2*bi*s+di
        upper = 1.
        while f(upper) <= 0:
            upper *= 2
            assert upper < 2**30
        factors.append(brentq(f, 0., upper, xtol=5e-324, rtol=9e-16))
    factors = np.asarray(factors)[:, None]
    projected = z * factors
    norm_error = abs(np.einsum('ij,ij->i',(projected-b)/w,(projected-b)/w)-r2)/r2
    cosine = np.einsum('ij,ij->i', z, projected)/(np.linalg.norm(z,axis=1)*np.linalg.norm(projected,axis=1))
    assert norm_error.max() < 1e-12 and abs(cosine-1).max() < 1e-12
    return projected, factors, dict(queries=len(c),maximum_ideal_relative_squared_norm_error=float(norm_error.max()),maximum_ideal_direction_cosine_error=float(abs(cosine-1).max()))


def check_actual(actual, condition, raw, weight, bias):
    expected, factors, report = independent_projection(condition,raw,weight,bias)
    actual=np.asarray(actual,np.float32);assert actual.shape==expected.shape
    scaled=float((abs(actual-expected)/(1+abs(expected))).max())
    c,z,w,b=[np.asarray(x,np.float64) for x in (condition,raw,weight,bias)]
    relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    r2=np.sum(((c-b)/w)**2,axis=1)
    actual_r2=np.sum(((actual-b)/w)**2,axis=1)
    direction=np.sum(z*actual,axis=1)/(np.linalg.norm(z,axis=1)*np.linalg.norm(actual,axis=1))
    assert scaled<2e-6 and relative<1e-5
    assert np.max(abs(actual_r2-r2)/r2)<2e-6 and np.max(abs(direction-1))<2e-7
    report.update(scaled_error=scaled,relative_L2=relative,maximum_actual_norm2_relative_error=float(np.max(abs(actual_r2-r2)/r2)),maximum_actual_direction_cosine_error=float(np.max(abs(direction-1))),positive_factor_quantiles=np.quantile(factors,[0,.5,1]).tolist(),whole_T_MLP_guard_replaced=False)
    return report
