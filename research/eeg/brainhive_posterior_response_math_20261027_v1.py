"""Research primitives, not an admitted EEG model or a diffusion posterior sampler.

The Gaussian conditional is exact only for the declared linear Gaussian model.
The attention average is exact for its supplied finite distribution, at fixed Q.
Neither statement makes a nonlinear multi-layer UNet a posterior sampler.
"""
import numpy as np


def conditional_innovation(observed_eeg, mean_x, mean_eeg, covariance_x,
                           cross_covariance, covariance_eeg):
    """Schur solution for a joint Gaussian already conditional on (z_t,c,t).

    This API deliberately does not accept a second independent c-derived prior.
    Inputs are conditional moments, not unconditional EEG noise estimates.
    A degenerate conditional EEG covariance needs an explicitly reviewed support
    model; this prototype rejects it rather than silently adding a ridge.
    """
    y,m,ym,s,c,r=[np.asarray(x,np.float64) for x in
                  (observed_eeg,mean_x,mean_eeg,covariance_x,cross_covariance,covariance_eeg)]
    assert y.shape==ym.shape and y.ndim==2 and m.ndim==2 and len(y)==len(m)
    nx,ny=m.shape[1],y.shape[1]
    assert s.shape==(nx,nx) and c.shape==(nx,ny) and r.shape==(ny,ny)
    assert all(np.isfinite(x).all() for x in (y,m,ym,s,c,r))
    assert np.allclose(s,s.T) and np.allclose(r,r.T)
    joint=np.block([[s,c],[c.T,r]])
    np.linalg.cholesky(joint)  # Reject inconsistent cross-covariance models.
    gain=np.linalg.solve(r,c.T).T
    residual=y-ym;post=s-gain@c.T;post=.5*(post+post.T)
    np.linalg.cholesky(post)
    return dict(mean=m+residual@gain.T,covariance=post,innovation=residual,gain=gain)


def gaussian_conditional(observation, mean, covariance, operator, offset, noise):
    """f~N(mean,covariance), e|f~N(operator@f+offset,noise).

    Rows of observation are independent queries. No second EEG-derived prior is
    multiplied into this model. Cholesky rejects non-positive-definite models.
    """
    e, m, s, h, b, r = [np.asarray(x, dtype=np.float64) for x in
                        (observation, mean, covariance, operator, offset, noise)]
    n = len(m); k = len(b)
    assert e.ndim == 2 and e.shape[1] == k
    assert s.shape == (n, n) and h.shape == (k, n) and r.shape == (k, k)
    for x in (e, m, s, h, b, r):
        assert np.isfinite(x).all()
    assert np.allclose(s, s.T) and np.allclose(r, r.T)
    np.linalg.cholesky(s); np.linalg.cholesky(r)
    observed_covariance = h @ s @ h.T + r
    gain = np.linalg.solve(observed_covariance, h @ s).T
    posterior_mean = m + (e - b - h @ m) @ gain.T
    # Joseph form is algebraically equal to S-KHS and preserves PSD numerically.
    identity = np.eye(n) - gain @ h
    posterior_covariance = identity @ s @ identity.T + gain @ r @ gain.T
    posterior_covariance = .5 * (posterior_covariance + posterior_covariance.T)
    np.linalg.cholesky(posterior_covariance)
    return posterior_mean, posterior_covariance


def whitened_spectrum(covariance, operator, noise):
    """e-b-Hm whitened by Cholesky(noise); f-m=L*z, z~N(0,I).

    Returns all latent directions, including the unobserved null space.
    The same singular values determine mean gain and remaining variance.
    """
    s, h, r = [np.asarray(x, np.float64) for x in (covariance, operator, noise)]
    l = np.linalg.cholesky(s); lr = np.linalg.cholesky(r)
    u, singular, vh = np.linalg.svd(np.linalg.solve(lr, h @ l), full_matrices=True)
    values = np.zeros(s.shape[0]); values[:len(singular)] = singular
    return dict(prior_factor=l, noise_factor=lr, left=u, right=vh.T,
                singular_values=values, mean_gain=values / (1 + values**2),
                remaining_variance=1 / (1 + values**2))


def marginal_attention(query, keys, values, weights):
    """One fixed-query Bayes response for an explicit finite condition law.

    query [B,H,N,D]; keys/values [B,M,H,4,D]; weights [B,M].
    Each hypothesis has its OWN four-token softmax normalizer. Averaging occurs
    AFTER native attention, never by flattening M*4 into one token dimension.
    There is no added parameter, query-dependent posterior reweighting, dropout,
    or sampling inside this function. Shared atoms/weights must be supplied to
    all layers by the caller. Extra attention work scales with M, not UNet NFE.
    """
    import torch
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    assert query.ndim == 4 and keys.ndim == values.ndim == 5
    assert keys.shape == values.shape and keys.shape[3] == 4
    b, count, heads, _, dim = keys.shape
    assert query.shape[0] == b and query.shape[1] == heads and query.shape[3] == dim
    assert weights.shape == (b, count) and count > 0
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    assert torch.allclose(weights.sum(1), torch.ones(b, device=weights.device,
                                                 dtype=weights.dtype), atol=1e-6, rtol=0)
    # A delta law goes through the actual native SDPA without a new arithmetic
    # path. FP16 pixel equivalence still needs a separate full-UNet replay.
    if count == 1:
        return sdpa(query, keys[:, 0], values[:, 0], dropout_p=0., is_causal=False)
    result = None
    for i in range(count):
        part = sdpa(query, keys[:, i], values[:, i], dropout_p=0., is_causal=False)
        term = part.float() * weights[:, i, None, None, None].float()
        result = term if result is None else result + term
    return result.to(query.dtype)


def gaussian_log_score(target, mean, covariance):
    """Proper log score under this Gaussian law; not an EEG image metric."""
    target, mean, s = [np.asarray(x, np.float64) for x in (target, mean, covariance)]
    assert target.shape == mean.shape and target.ndim == 2
    chol = np.linalg.cholesky(s)
    residual = np.linalg.solve(chol, (target - mean).T).T
    return .5 * (np.square(residual).sum(1) + 2*np.log(np.diag(chol)).sum()
                 + s.shape[0]*np.log(2*np.pi))


def state_conditioned_innovation(eeg, noisy_state, mean, covariance, operator,
                                offset, noise, alpha, sigma):
    """Exact ONLY for x~N(m,S), e=H*x+b+eta, z=alpha*x+sigma*eps.

    eta~N(0,R), eps~N(0,I) independent of each other and x. Here x is the
    SAME variable being noised, not a CLIP F substituted for a VAE latent.
    Native Turbo does not supply these assumptions or this covariance.
    """
    assert np.isfinite(alpha) and alpha > 0 and np.isfinite(sigma) and sigma > 0
    m=np.asarray(mean,np.float64);s=np.asarray(covariance,np.float64)
    h=np.asarray(operator,np.float64);r=np.asarray(noise,np.float64)
    z=np.asarray(noisy_state,np.float64);e=np.asarray(eeg,np.float64)
    b=np.asarray(offset,np.float64);n=len(m)
    image_mean,image_cov=gaussian_conditional(z,m,s,alpha*np.eye(n),
                                             np.zeros(n),sigma**2*np.eye(n))
    observed_cov=h@image_cov@h.T+r
    gain=np.linalg.solve(observed_cov,h@image_cov).T
    innovation=e-b-image_mean@h.T
    corrected=image_mean+innovation@gain.T
    i_minus=np.eye(n)-gain@h
    corrected_cov=i_minus@image_cov@i_minus.T+gain@r@gain.T
    return dict(prior_denoised_mean=image_mean,prior_conditional_covariance=image_cov,
                innovation=innovation,gain=gain,corrected_mean=corrected,
                corrected_covariance=corrected_cov,
                conditional_epsilon=(z-alpha*corrected)/sigma)
