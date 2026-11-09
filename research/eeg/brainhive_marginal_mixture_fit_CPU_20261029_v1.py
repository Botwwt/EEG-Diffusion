"""Fixed-capacity classical vMF EM and exact EEG exponential tilting.

No neural network, decoder or new innovation is claimed by this density code.
"""
import numpy as np
from scipy import optimize, special
from brainhive_spherical_predictive_density_CPU_20261029_v1 import (
    fit_vmf_concentration, vmf_log_normalizer, vmf_log_normalizer_array,
    vmf_mean_cosine_array, vmf_cosine_quantiles,
)


def fit_tied_vmf_mixture(values, components=32, iterations=30):
    values=np.asarray(values,dtype=np.float64)
    count,dimension=values.shape
    assert count>components and components>=1
    assert np.max(abs(np.linalg.norm(values,axis=1)-1))<1e-10
    # Deterministic farthest-first initialization, with array-order tie breaking.
    centers=[0]
    largest_similarity=values@values[0]
    for unused in range(1,components):
        index=int(np.argmin(largest_similarity))
        assert index not in centers
        centers.append(index)
        largest_similarity=np.maximum(largest_similarity,values@values[index])
    means=values[centers].copy()
    weights=np.full(components,1/components)
    concentration=fit_vmf_concentration(dimension,float(np.linalg.norm(values.mean(axis=0))))
    history=[]
    for epoch in range(iterations):
        logits=concentration*(values@means.T)+np.log(weights)
        log_norm=special.logsumexp(logits,axis=1)
        likelihood=float(vmf_log_normalizer(dimension,concentration)+log_norm.mean())
        if history:
            assert likelihood>=history[-1]['mean_logscore']-1e-7,('EM_likelihood_decreased',epoch,likelihood,history[-1])
        responsibilities=np.exp(logits-log_norm[:,None])
        masses=responsibilities.sum(axis=0)
        assert np.min(masses)>1e-8,('empty_EM_component',float(masses.min()))
        sums=responsibilities.T@values
        lengths=np.linalg.norm(sums,axis=1)
        assert np.all(lengths>1e-12)
        history.append(dict(iteration=epoch,mean_logscore=likelihood,concentration=concentration,minimum_component_effective_count=float(masses.min())))
        weights=masses/count
        means=sums/lengths[:,None]
        concentration=fit_vmf_concentration(dimension,float(lengths.sum()/count))
    final=float(vmf_log_normalizer(dimension,concentration)+special.logsumexp(concentration*(values@means.T)+np.log(weights),axis=1).mean())
    assert final>=history[-1]['mean_logscore']-1e-7
    history.append(dict(iteration=iterations,mean_logscore=final,concentration=concentration))
    return dict(weights=weights,means=means,concentrations=np.full(components,concentration),history=history)


def conditional_parameters(weights,concentrations,EEG_component_cosines,scale,dimension):
    cosines=np.asarray(EEG_component_cosines,dtype=np.float64)
    concentrations=np.asarray(concentrations,dtype=np.float64)
    lengths=np.sqrt(np.maximum(0,concentrations**2+scale**2+2*concentrations*scale*cosines))
    log_weights=np.log(weights)+vmf_log_normalizer_array(dimension,concentrations)-vmf_log_normalizer_array(dimension,lengths)
    log_evidence=special.logsumexp(log_weights,axis=1)
    return lengths,np.exp(log_weights-log_evidence[:,None]),log_evidence


def fit_conditional_tilt(weights,concentrations,EEG_component_cosines,EEG_target_cosines,dimension,max_scale=4096.):
    cosines=np.asarray(EEG_component_cosines,dtype=np.float64)
    target=np.asarray(EEG_target_cosines,dtype=np.float64)
    evaluations=[]
    def derivative(scale):
        lengths,posterior_weights,log_z=conditional_parameters(weights,concentrations,cosines,scale,dimension)
        axial=np.divide(scale+concentrations*cosines,lengths,out=np.zeros_like(lengths),where=lengths>0)
        value=float(np.mean(target-np.sum(posterior_weights*vmf_mean_cosine_array(dimension,lengths)*axial,axis=1)))
        evaluations.append(dict(scale=float(scale),derivative=value))
        return value
    low=derivative(0.)
    if low<=0:
        return 0.,dict(boundary='zero',derivative_at_solution=low,evaluations=evaluations)
    high=derivative(max_scale)
    if high>=0:
        return max_scale,dict(boundary='fixed_upper4096',derivative_at_solution=high,evaluations=evaluations)
    scale=float(optimize.brentq(derivative,0.,max_scale,xtol=1e-7))
    residual=derivative(scale)
    assert abs(residual)<1e-7
    return scale,dict(boundary=None,derivative_at_solution=residual,evaluations=evaluations)


def component_coverage(weights,concentrations,EEG_component_cosines,target_component_cosines,EEG_target_cosines,scale,dimension):
    """Expected component-PIT coverage, not a 95% region in observed v-space.

    Under the assumed joint model, drawing a component from its responsibility
    given an observed v gives a uniform component-PIT. We integrate that label
    rather than adding a Monte Carlo label draw.
    """
    lengths,conditional_weights,log_z=conditional_parameters(weights,concentrations,EEG_component_cosines,scale,dimension)
    scalar=np.asarray(EEG_target_cosines)
    axial=(concentrations*target_component_cosines+scale*scalar[:,None])/lengths
    likelihood_logits=np.log(weights)+vmf_log_normalizer_array(dimension,concentrations)+concentrations*target_component_cosines
    responsibilities=np.exp(likelihood_logits-special.logsumexp(likelihood_logits,axis=1)[:,None])
    # The tilt is the same for every component for fixed E,v, so these are also
    # the conditional component responsibilities. No query label is supplied.
    low,high=float(lengths.min()),float(lengths.max())
    if high-low<1e-10:
        knots=np.array([low,low+1e-9])
    else:
        knots=np.linspace(low,high,33)
    quantiles=np.array([vmf_cosine_quantiles(dimension,k) for k in knots])
    maximum_error=0.
    for k in np.linspace(low,high,9)[1:-1]+(high-low)/512:
        exact=vmf_cosine_quantiles(dimension,k)
        approximated=np.array([np.interp(k,knots,quantiles[:,j]) for j in range(3)])
        maximum_error=max(maximum_error,float(np.max(abs(exact-approximated))))
    assert maximum_error<1e-5
    lower=np.interp(lengths,knots,quantiles[:,0])
    upper=np.interp(lengths,knots,quantiles[:,2])
    inside=(axial>=lower)&(axial<=upper)
    coverage=np.sum(responsibilities*inside,axis=1)
    return coverage,dict(knots=knots.tolist(),quantiles=quantiles.tolist(),maximum_offknot_quantile_error=maximum_error),dict(lengths=lengths,responsibilities=responsibilities,axial=axial,posterior_weights=conditional_weights,log_evidence=log_z)
