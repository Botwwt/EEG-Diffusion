"""D23: change the native first projection under a fixed repeat-noise metric.

This is a training parameterization of the existing W, not an inference module.
The metric is a regularized within-repeat covariance proxy, not known physiology.
Stiefel geometry and normalized projected-gradient steps are established methods.
"""
import torch

ARMS=('repeat_noise_constrained','permuted_noise_constrained','repeat_noise_unconstrained')

def shrink_mean4_noise(sum_outer,stimuli):
    assert sum_outer.ndim==2 and sum_outer.shape[0]==sum_outer.shape[1] and stimuli>0
    n=sum_outer.double()/(12*stimuli) # four-repeat unbiased trial covariance /4
    scale=n.trace()/len(n);assert scale>0
    return .9*n+.1*scale*torch.eye(len(n),device=n.device,dtype=n.dtype)

def geometry(weight,noise):
    w=weight.double();n=noise.double();l=torch.linalg.cholesky(n)
    s=torch.linalg.cholesky(w@n@w.T)
    q=torch.linalg.solve_triangular(s,w@l,upper=False)
    assert (q@q.T-torch.eye(len(w),device=w.device,dtype=w.dtype)).abs().max()<1e-8
    return dict(noise=n,noise_cholesky=l,hidden_cholesky=s,Q0=q.float(),W0=weight.float().clone())

def whiten(eeg,g):
    return torch.linalg.solve_triangular(g['noise_cholesky'],eeg.double().T,upper=False).T.float()

def residual_preactivation(raw,white,q,g,bias):
    # An exact zero update follows the original FP32 W/bias arithmetic.
    return torch.nn.functional.linear(raw,g['W0'],bias)+(white@(q-g['Q0']).T)@g['hidden_cholesky'].float().T

def folded_weight(q,g):
    change=g['hidden_cholesky']@(q-g['Q0']).double()
    delta=torch.linalg.solve_triangular(g['noise_cholesky'].T,change.T,upper=True).T
    return (g['W0'].double()+delta).float()

def tangent_gradient(q,gradient):
    cross=gradient@q.T
    return gradient-.5*(cross+cross.T)@q

def retract_rows(q):
    value=q.double();l=torch.linalg.cholesky(value@value.T)
    return torch.linalg.solve_triangular(l,value,upper=False).float()

def fixed_step(q,gradient,constrained,fraction=.001):
    assert fraction==.001
    g=tangent_gradient(q,gradient) if constrained else gradient
    norm=g.norm();assert torch.isfinite(norm)
    if norm==0:return q.clone(),dict(gradient_norm=0.,linearized_risk_change=0.,step_norm=0.)
    radius=fraction*(q.shape[0]**.5)
    delta=-radius*g/norm
    # Exact minimizer of the linearized loss over the tangent Euclidean ball;
    # retraction enforces the finite-step row-orthogonality constraint.
    value=retract_rows(q+delta) if constrained else q+delta
    return value,dict(gradient_norm=float(norm),linearized_risk_change=float((gradient*delta).sum()),step_norm=float((value-q).norm()))

def metric_error(weight,g):
    w=weight.double();expected=g['hidden_cholesky']@g['hidden_cholesky'].T
    return float((w@g['noise']@w.T-expected).norm()/expected.norm())
