"""Shared-token mean calibration; elementary attention algebra, not a new theorem.

The learned EEG encoder, ordinary, A/P and native attention weights stay fixed.
A common four-token shift changes only a constant value-output term at fixed
module inputs. This says nothing about preserving the full UNet trajectory.
"""
import numpy as np

ARMS=('native_shared_mean','ordinary_shared_mean','ordinary_full_mean_matched','ordinary_wrong_field')


def calibration_fields(visual_token_mean,EEG_token_mean):
    visual=np.asarray(visual_token_mean,np.float64)
    eeg=np.asarray(EEG_token_mean,np.float64)
    assert visual.shape==eeg.shape==(4,2048) and np.isfinite(visual).all() and np.isfinite(eeg).all()
    full=visual-eeg
    common=full.mean(axis=0)
    shared=np.broadcast_to(common,full.shape)
    full_norm=float(np.linalg.norm(full));shared_norm=float(np.linalg.norm(shared))
    factor=shared_norm/full_norm if full_norm else 0.
    assert -1e-12<=factor<=1+1e-12
    matched=full*factor
    return dict(shared_delta=common,full_delta=full,matched_full_delta=matched,
                proof=dict(full_token_norm=full_norm,shared_token_norm=shared_norm,full_match_factor=factor,
                           matched_token_norm=float(np.linalg.norm(matched)),
                           mean_centered_orthogonality=float(np.sum(shared*(full-shared)))))


def output_bias(shared_delta,value_weight,output_weight,ip_scale=1.,rescale=1.):
    d=np.asarray(shared_delta,np.float64);wv=np.asarray(value_weight,np.float64);wo=np.asarray(output_weight,np.float64)
    assert d.shape==(2048,) and wv.shape[1]==2048 and wo.shape[1]==wv.shape[0]
    assert np.isfinite(d).all() and np.isfinite(wv).all() and np.isfinite(wo).all() and rescale>0
    return (wo@(wv@d))*(float(ip_scale)/float(rescale))


def wrong_field(bias):
    bias=np.asarray(bias,np.float64)
    assert bias.ndim==1 and len(bias)>1
    return np.roll(bias,len(bias)//2)
