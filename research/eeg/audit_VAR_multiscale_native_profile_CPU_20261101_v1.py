"""Independent NumPy full cell sums/condition contrasts and engineering traces."""
import argparse
import json
from pathlib import Path
import sys
import time
import VAR_multiscale_profile_contract_CPU_20261101_v1 as C


def logsumexp(x,axis):
    import numpy as np
    maximum=np.max(x,axis=axis,keepdims=True)
    return np.squeeze(maximum+np.log(np.exp(x-maximum).sum(axis=axis,keepdims=True)),axis=axis)


def summarize(scale_ll):
    import numpy as np
    assert scale_ll.shape==(8,1000,10) and scale_ll.dtype==np.float64 and np.isfinite(scale_ll).all()
    ll=scale_ll.transpose(0,2,1);sizes=np.array([1,4,9,16,25,36,64,100,169,256])
    logp=ll-logsumexp(ll,2)[:,:,None];p=np.exp(logp)
    entropy=-(p*logp).sum(2)
    cumulative=ll.cumsum(1);cum_logp=cumulative-logsumexp(cumulative,2)[:,:,None];cum_p=np.exp(cum_logp)
    coarse=cum_p[:,1];full=cum_p[:,-1]
    return dict(per_T_per_scale_class_log_likelihood_range=np.ptp(ll,axis=2).tolist(),
        per_T_per_scale_centered_class_std=np.std(ll,axis=2).tolist(),
        per_T_per_scale_centered_class_std_per_token=(np.std(ll,axis=2)/sizes[None]).tolist(),
        uniform_prior_per_scale_entropy=entropy.tolist(),
        uniform_prior_per_scale_KL_to_uniform=(np.log(1000)-entropy).tolist(),
        coarse5_vs_full680_global_class_posterior_total_variation=(.5*np.abs(coarse-full).sum(1)).tolist(),
        coarse5_vs_full680_global_class_argmax_changed=(coarse.argmax(1)!=full.argmax(1)).tolist(),
        cumulative_uniform_prior_entropy=(-(cum_p*cum_logp).sum(2)).tolist(),
        interpretations=['Class sensitivity is not EEG predictability or image improvement.',
            'Eight fixed training images do not establish population significance or same-concept detail.',
            'Large exact-likelihood posterior concentration can amplify model mismatch; it is not calibrated EEG certainty.'],
        choose_classes_scales_temperature_or_checkpoint_from_this_probe=False)


def run(protocol):
    import numpy as np
    started=time.monotonic();s=C.setup(protocol);out=C.OUT/'profile';r=C.read(out/'completion.json')
    assert r['status']=='COMPLETE_PRETRAINED_T8_ALL1000_FULL680_NATIVE_VAR_ENGINEERING_ONLY'
    assert r['protocol']==C.ref(protocol) and r['all_VAR_VQ_weights_unchanged'] is True
    assert r['fixed_class_all680_tokens_exact'] is True and r['fixed_class_decoded_pixels_exact'] is True
    for pin in r['files']:C.checked(pin)
    ll=np.load(out/'scale_log_likelihoods.npy',allow_pickle=False)
    lp=np.load(out/'true_token_logprobs.npy',mmap_mode='r',allow_pickle=False)
    assert lp.shape==(8,1000,680) and lp.dtype==np.float32 and np.isfinite(lp).all() and (lp<=0).all()
    boundaries=np.cumsum([0]+[p*p for p in s['scales']]);error=0.
    # Different iteration order from the producer: all ten scales, all8000 pairs.
    for k,(lo,hi) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        expected=np.asarray(lp[:,:,lo:hi],dtype=np.float64).sum(2)
        error=max(error,float(np.max(np.abs(expected-ll[:,:,k]))));assert np.array_equal(expected,ll[:,:,k])
    assert np.array_equal(np.load(out/'coverage.npy',allow_pickle=False),np.ones((8,1000),np.uint8))
    codes=np.load(out/'true_codes.npy',allow_pickle=False);assert codes.shape==(8,680) and codes.min()>=0 and codes.max()<4096
    native=np.load(out/'fixed_native_pixels.npy',allow_pickle=False);fresh=np.load(out/'fixed_fresh_pixels.npy',allow_pickle=False)
    assert np.array_equal(native,fresh) and native.shape==(3,256,256)
    with np.load(out/'fixed_class_native_and_fresh_trace.npz',allow_pickle=False) as f:
        assert len(f.files)==40
        for k,n in enumerate(np.diff(boundaries)):
            assert f[f'cached_codes_{k}'].shape==f[f'fresh_codes_{k}'].shape==(1,n)
            assert np.array_equal(f[f'cached_codes_{k}'],f[f'fresh_codes_{k}'])
            assert f[f'cached_logits_{k}'].shape==f[f'fresh_logits_{k}'].shape==(2,n,4096)
            assert np.allclose(f[f'cached_logits_{k}'],f[f'fresh_logits_{k}'],**s['numerical_guard'])
    with np.load(out/'switched_class_fresh_trace.npz',allow_pickle=False) as f:
        assert np.array_equal(f['classes'],[[0,11,287,999,0,11,287,999,0,11]])
        for k,n in enumerate(np.diff(boundaries)):
            assert f[f'codes_{k}'].shape==(1,n) and f[f'logits_{k}'].shape==(2,n,4096)
            assert np.isfinite(f[f'logits_{k}']).all()
    from EEG_multiscale_native_class_DRAFT_CPU_20261101_v1 import inference
    posteriors={};evidence=[];endpoint_error=0.
    for i in range(8):
        ell=ll[i].T;row=[]
        for rho in (0.,.5,1.):
            value=inference(np.zeros((10,1000)),np.full(9,rho),ell)
            assert np.allclose(value['class_posterior'].sum(1),1.,atol=1e-11,rtol=1e-11)
            posteriors[f'T{i}_rho{rho}_posterior']=value['class_posterior']
            posteriors[f'T{i}_rho{rho}_prior_logits_gradient']=value['negative_log_evidence_gradient_logits']
            row.append(value['log_evidence'])
            if rho==0:expected=float((logsumexp(ell,1)-np.log(1000)).sum())
            elif rho==1:expected=float(logsumexp(ell.sum(0),0)-np.log(1000))
            else:continue
            endpoint_error=max(endpoint_error,abs(value['log_evidence']-expected))
            assert abs(value['log_evidence']-expected)<2e-8
        evidence.append(row)
    np.savez(C.OUT/'T8_uniform_prior_HMM_probability_audit.npz',**posteriors)
    result=dict(status='PASS_ALL80000_NATIVE_SCALE_SUMS_AND_FIXED_ENGINEERING_ARTIFACTS_CPU',
        protocol=C.ref(protocol),profile_completion=C.ref(out/'completion.json'),
        all80000_scale_log_likelihood_cells_exact=True,scale_sum_maxerror=error,
        teacher_token_terms_audited=8*1000*680,fixed_native_replay_codes=680,exact_FP32_pixel_elements=3*256*256,
        class_contrast_diagnostics=summarize(ll),
        actual_likelihood_uniform_prior_HMM_log_evidence_rho_0_half_1=evidence,
        actual_likelihood_HMM_endpoint_maxerror=endpoint_error,
        uniform_image_only_posterior_is_not_EEG_conditioned_or_image_gain=True,
        actual_T8_loop_seconds=r['loop_seconds'],actual_complete_profile_seconds=r['seconds'],
        full13230_naive_same_batch_compute_seconds_linear_projection=r['loop_seconds']*13230/8,
        projection_is_not_actual_full_bank_runtime_or_admission=True,
        EEG_predictability_or_new_image_gain=False,novelty_confirmed=False,new_optimizer_updates=0,
        new_GPU_jobs=0,new_research_images=0,seconds=time.monotonic()-started)
    C.write(C.OUT/'actual_independent_CPU_audit.json',result);C.resources(s)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
