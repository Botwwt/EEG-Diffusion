"""Classical two-objective normalized steepest direction, not new theory.

The bounded head and native radius remain unchanged. Calibration below would
interpolate PARAMETERS with the original ordinary state, not change its radius.
No fit/image lifecycle is admitted by importing this engineering module.
"""
import torch

ARMS=('joint_same','relative_only','AP_only','joint_mismatched_teacher')
FRACTIONS=(0.,.25,.5,.75,1.)

def direction(relative,AP,mode='joint'):
    assert relative.shape==AP.shape and relative.ndim==1
    assert torch.isfinite(relative).all() and torch.isfinite(AP).all()
    assert mode in ('joint','relative','AP')
    nr=relative.norm();na=AP.norm();tiny=1e-20
    ur=relative/nr if nr>tiny else torch.zeros_like(relative)
    ua=AP/na if na>tiny else torch.zeros_like(AP)
    if mode=='relative':value=-ur
    elif mode=='AP':value=-ua
    else:
        total=ur+ua;nn=total.norm();value=-total/nn if nn>tiny else torch.zeros_like(total)
    return value,dict(relative_norm=nr,AP_norm=na,relative_derivative=relative@value,AP_derivative=AP@value)

def interpolate_state(reference,candidate,fraction):
    assert fraction in FRACTIONS and set(reference)==set(candidate)
    assert all(reference[k].shape==candidate[k].shape for k in reference)
    if fraction==0:return {k:v.clone() for k,v in reference.items()}
    if fraction==1:return {k:v.clone() for k,v in candidate.items()}
    return {k:v+(candidate[k]-v)*fraction for k,v in reference.items()}
