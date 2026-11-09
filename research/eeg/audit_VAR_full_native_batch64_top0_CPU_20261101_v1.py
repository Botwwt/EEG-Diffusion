"""All5.44M new FP32 terms versus audited old table; no EEG interpretation."""
import argparse
import datetime
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import VAR_full_native_throughput_contract_CPU_20261101_v1 as C


def posterior(logweights):
    v=np.asarray(logweights,np.float64);v=v-v.max(-1,keepdims=True)
    w=np.exp(v);return w/w.sum(-1,keepdims=True)


def fidelity_decision(maximum,all_argmax_same,guard):
    assert np.isfinite(maximum) and maximum>=0 and guard>0
    return bool(maximum<=guard and all_argmax_same)


def run(protocol):
    import resource
    started=time.monotonic();base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    q=C.setup(protocol);C.resources(q);n,_=C.dependency(q);p=C.OUT/'profile'
    done=C.read(p/'completion.json');assert done['status']=='COMPLETE_SAME_NATIVE_FP32_BATCH64_T8_K1000_FULL680_AND_TOP0_PARITY'
    assert done['protocol']==C.ref(protocol) and done['class_batches']==128 and done['all_weights_byte_unchanged']
    for pin in done['files']:C.checked(pin)
    values=np.load(p/'true_token_logprobs.npy',mmap_mode='r',allow_pickle=False)
    old=np.load(q['reference']['token_logprobs']['path'],mmap_mode='r',allow_pickle=False)
    ll=np.load(p/'scale_log_likelihoods.npy',allow_pickle=False)
    old_ll=np.load(q['reference']['scale_likelihoods']['path'],allow_pickle=False)
    assert values.shape==old.shape==(8,1000,680) and values.dtype==old.dtype==np.float32
    assert ll.shape==old_ll.shape==(8,1000,10) and ll.dtype==np.float64
    assert np.array_equal(np.load(p/'coverage.npy',allow_pickle=False),np.ones((8,1000),np.uint8))
    assert np.array_equal(np.load(p/'true_codes.npy',allow_pickle=False),np.load(q['reference']['true_codes']['path'],allow_pickle=False))
    maximum=0.;sum_error=0.;terms=0;bounds=np.cumsum([0]+[x*x for x in q['scales']])
    for k,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
        summed=np.zeros((8,1000))
        for token in range(lo,hi):
            a,b=values[:,:,token],old[:,:,token]
            assert np.isfinite(a).all() and (a<=0).all() and np.allclose(a,b,**q['numerical_guard'])
            maximum=max(maximum,float(np.abs(a-b).max()));summed+=a;terms+=8000
        sum_error=max(sum_error,float(np.abs(summed-ll[:,:,k]).max()));assert np.array_equal(summed,ll[:,:,k])
    assert maximum==done['maximum_target_logprob_difference'] and terms==5440000
    spec=importlib.util.spec_from_file_location('frozen_HMM_execution_fidelity',n.CODE/'EEG_multiscale_native_class_DRAFT_CPU_20261101_v1.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    diffs=[];argmax=[];maximum_post=0.
    for i in range(8):
        row=[]
        for rho in (0.,.5,1.):
            a=H.inference(np.zeros((10,1000)),np.full(9,rho),ll[i].T)
            b=H.inference(np.zeros((10,1000)),np.full(9,rho),old_ll[i].T)
            error=float(np.abs(a['class_posterior']-b['class_posterior']).max())
            row.append(error);maximum_post=max(maximum_post,error)
        diffs.append(row)
        argmax.append(bool(np.argmax(ll[i].sum(1))==np.argmax(old_ll[i].sum(1))))
    # Independent direct global posterior, not only the validated HMM routine.
    direct=posterior(ll.sum(2));direct_old=posterior(old_ll.sum(2))
    direct_error=float(np.abs(direct-direct_old).max());maximum_post=max(maximum_post,direct_error)
    assert fidelity_decision(maximum_post,all(argmax),q['posterior_max_abs_guard'])
    with np.load(p/'top0_codes.npz',allow_pickle=False) as f:
        assert len(f.files)==20
        count=0
        for k,side in enumerate(q['scales']):
            x,y=f[f'cached_{k}'],f[f'fresh_{k}']
            assert x.shape==y.shape==(1,side*side) and np.array_equal(x,y) and ((x>=0)&(x<4096)).all();count+=x.size
        assert count==680
    decoded={}
    for name in ('cached_top0','fresh_top0'):
        pixels=np.load(p/(name+'_pixels.npy'),allow_pickle=False)
        assert pixels.shape==(3,256,256) and pixels.dtype==np.float32 and np.isfinite(pixels).all()
        expected=(pixels.transpose(1,2,0).clip(0,1)*255).round().astype(np.uint8)
        with Image.open(p/(name+'.png')) as image:assert np.array_equal(np.asarray(image.convert('RGB')),expected)
        decoded[name]=pixels
    assert np.array_equal(decoded['cached_top0'],decoded['fresh_top0'])
    assert done['top0_all680_codes_exact'] and done['top0_decoded_FP32_pixels_exact']
    assert time.monotonic()-started<q['CPU_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base<q['maximum_extra_CPU_bytes']
    import torch
    assert not torch.cuda.is_initialized()
    C.write(C.OUT/'independent_CPU_audit.json',dict(status='PASS_ALL5440000_NEW_NATIVE_TERMS_AND_TOP0_FIXED_ENGINEERING_ARTIFACTS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=C.ref(protocol),new_token_terms=terms,
        all80000_scale_sums_exact=True,sum_maxerror=sum_error,token_logprob_maxerror=maximum,
        uniform_HMM_posterior_maxerror_rho0_half_1=diffs,maximum_uniform_posterior_difference=maximum_post,
        all8_global_class_argmax_same=all(argmax),independent_direct_global_posterior_difference=direct_error,
        top0_matched_sampled_codes=680,top0_matched_FP32_pixel_elements=196608,
        GPU_multinomial_not_replayed_on_CPU=True,actual_frozen_native_forward_relies_on_pinned_GPU_receipt=True,
        loop_speed_ratio_reference8_over_batch64=q['reference_loop_seconds']/done['loop_seconds'],
        all13230_same_loop_extrapolation_hours=done['loop_seconds']/8*13230/3600,
        no_EEG_training_or_picture_gain=True,new_GPU_jobs=0,new_optimizer_updates=0,new_research_images=0,
        seconds=time.monotonic()-started,extra_CPU_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-base,CUDA_initialized=False))
    C.resources(q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
