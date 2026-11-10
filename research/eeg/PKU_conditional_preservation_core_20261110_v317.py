"""Existing P weight affine-subspace projection, with EEG domain preservation."""
import torch
from PKU_downstream_core_20261109_v303 import R,sha,write,save,states,load_states,fixed_inputs
ROOT=R/'runs/EEG_preservation_v317'
CASES=['P_abs_free','P_rel_free','P_abs_mean','P_rel_mean','P_rel_random']

def flags(case):
    assert case in CASES
    return dict(relative='_rel_' in case,anchor='mean' if case.endswith('_mean') else 'random' if case.endswith('_random') else 'free')

def constrain(weight,initial,anchor):
    if anchor is None:return 0.
    with torch.no_grad():
        a=anchor.float();den=a.square().sum();assert den>1e-8
        delta=weight-initial
        correction=(delta@a)/den
        weight.sub_(correction[:,None]*a[None,:])
        residual=(weight-initial)@a
        return float(residual.abs().max())

def check_fixture():
    g=torch.Generator().manual_seed(1729)
    base=torch.randn((7,11),generator=g,dtype=torch.float64)
    proposal=base+torch.randn((7,11),generator=g,dtype=torch.float64)*.2
    anchor=torch.randn(11,generator=g,dtype=torch.float64)
    projected=proposal-(((proposal-base)@anchor)/anchor.square().sum())[:,None]*anchor[None,:]
    assert float(((projected-base)@anchor).abs().max())<1e-12
    again=projected-(((projected-base)@anchor)/anchor.square().sum())[:,None]*anchor[None,:]
    assert float((again-projected).abs().max())<1e-12
    # Orthogonal correction gives the nearest feasible point in Frobenius norm.
    tangent=torch.randn((7,11),generator=g,dtype=torch.float64)
    tangent-=((tangent@anchor)/anchor.square().sum())[:,None]*anchor[None,:]
    assert float((projected-proposal).square().sum())<=float((projected+tangent-proposal).square().sum())+1e-12
    dp=torch.randn((3,4),generator=g,dtype=torch.float64);dn=torch.randn((3,4),generator=g,dtype=torch.float64)
    absolute=.5*(dp.square().mean()+dn.square().mean())
    mean=((dp+dn)*.5).square().mean();difference=(dp-dn).square().mean()
    assert abs(float(absolute-mean-.25*difference))<1e-12
    return dict(FP64_constraint_and_idempotence=True,Frobenius_projection_fixture=True,absolute_common_plus_difference_identity=True)
