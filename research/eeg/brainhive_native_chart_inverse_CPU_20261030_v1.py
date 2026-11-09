"""Classical quadratic inverse between two affine LayerNorm direction charts.

This proves coordinate attainability only. It neither fits an EEG network nor
claims that unseen EEG can predict any chosen visual direction.
"""
import numpy as np


def chart_parameters(EEG_weight,EEG_bias,F_weight,F_bias,radius):
    gamma=np.asarray(EEG_weight,dtype=np.float64)
    beta=np.asarray(EEG_bias,dtype=np.float64)
    Fgamma=np.asarray(F_weight,dtype=np.float64)
    Fbeta=np.asarray(F_bias,dtype=np.float64)
    assert gamma.ndim==1 and gamma.shape==beta.shape==Fgamma.shape==Fbeta.shape
    if np.min(abs(gamma))<=1e-6 or np.min(abs(Fgamma))<=1e-6:
        raise ValueError('An affine scale is singular')
    dimension=len(gamma)
    if not 0<radius<np.sqrt(dimension):
        raise ValueError('A finite LayerNorm preimage needs radius below sqrt(d)')
    diagonal=Fgamma/gamma
    total=float(diagonal.sum())
    if abs(total)<=1e-10*np.linalg.norm(diagonal):
        raise ValueError('The centered affine projection is singular')
    shift=(Fbeta-beta)/gamma
    origin=shift-diagonal*(shift.sum()/total)
    gap=float(radius**2-origin@origin)
    if gap<=0:
        raise ValueError('Global positive-ray coverage is unproven: origin is outside or on the EEG sphere')
    return dict(diagonal=diagonal,total=total,origin=origin,radius=float(radius),positive_radius_gap=gap)


def inverse_direction(directions,EEG_weight,EEG_bias,F_weight,F_bias,radius):
    """Given zero-mean F-unit w, find an EEG-affine output with that direction."""
    w=np.asarray(directions,dtype=np.float64)
    assert w.ndim==2 and np.max(abs(w.mean(axis=-1)))<1e-10
    assert np.max(abs(np.linalg.norm(w,axis=-1)-1))<1e-10
    p=chart_parameters(EEG_weight,EEG_bias,F_weight,F_bias,radius)
    d=p['diagonal'];A=p['origin']
    raw=w*d
    B=raw-d*(raw.sum(axis=-1)/p['total'])[:,None]
    squared=np.einsum('ij,ij->i',B,B)
    if np.min(squared)<=1e-20:
        raise ValueError('The direction projection is singular')
    cross=B@A
    discriminant=np.sqrt(cross**2+squared*p['positive_radius_gap'])
    distance=np.empty(len(w));positive=cross>=0
    distance[positive]=p['positive_radius_gap']/(discriminant[positive]+cross[positive])
    distance[~positive]=(discriminant[~positive]-cross[~positive])/squared[~positive]
    pre_affine=A+distance[:,None]*B
    condition=np.asarray(EEG_bias)+np.asarray(EEG_weight)*pre_affine
    return condition,pre_affine,dict(ray_distance=distance,quadratic_squared=squared,quadratic_cross=cross,positive_radius_gap=p['positive_radius_gap'],origin_norm=float(np.linalg.norm(A)))


def layernorm_preimage(pre_affine,epsilon):
    """A finite centered input whose epsilon-LayerNorm output is pre_affine."""
    u=np.asarray(pre_affine,dtype=np.float64)
    assert epsilon>0 and np.max(abs(u.mean(axis=-1)))<1e-10
    dimension=u.shape[-1]
    variance=np.mean(u*u,axis=-1)
    if np.max(variance)>=1:
        raise ValueError('LayerNorm unit-variance boundary has no finite preimage')
    multiplier=np.sqrt(epsilon/(1-variance))
    return u*multiplier[:,None]
