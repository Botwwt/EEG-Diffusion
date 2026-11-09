"""Untruncated likelihood versus threshold top-k sampling: no new EEG fit.

The sealed native profile remains unchanged. After its full independent audit,
this reads only saved engineering logits/codes. It cannot recover true-T8
target ranks from gathered log probabilities or establish EEG/image efficacy.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np

NATIVE_PROTOCOL_SHA='00279f7b4e8a0dee658a2c0c7352fd53692b639d1d9f21615d8b6166340fc891'
NATIVE_AUDIT_STATUS='PASS_LOCAL_FULL_NATIVE_ENGINEERING_NUMERICAL_AND_PROVENANCE_AUDIT_NOT_EEG_EFFICACY'


def digest(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def categorical(logits,top_k):
    """Native threshold rule: ties at the kth largest value remain supported."""
    x=np.asarray(logits,dtype=np.float64)
    assert x.ndim>=1 and x.shape[-1]>=2 and np.isfinite(x).all()
    assert isinstance(top_k,int) and top_k>=0
    shift=x-x.max(axis=-1,keepdims=True)
    weights=np.exp(shift)
    full=weights/weights.sum(axis=-1,keepdims=True)
    k=min(top_k,x.shape[-1])
    threshold=None if k==0 else np.take(np.partition(x,x.shape[-1]-k,axis=-1),-k,axis=-1)[...,None]
    support=np.ones_like(x,dtype=bool) if k==0 else x>=threshold
    mass=np.sum(full*support,axis=-1)
    assert ((mass>0)&(mass<=1.+1e-14)).all()
    truncated=np.where(support,full/mass[...,None],0.)
    return dict(full=full,truncated=truncated,retained_mass=mass,support=support)


def mixture_posterior(probabilities,prior,target):
    p=np.asarray(probabilities,dtype=np.float64);pi=np.asarray(prior,dtype=np.float64)
    assert p.ndim==2 and pi.shape==(p.shape[0],) and (p>=0).all() and (pi>0).all()
    assert np.allclose(p.sum(1),1.,atol=1e-14,rtol=0) and abs(pi.sum()-1.)<1e-14
    weights=pi*p[:,target];z=float(weights.sum())
    if z==0.:return dict(evidence=0.,posterior=None,negative_log_evidence_gradient=None,
                        status='ZERO_TARGET_SUPPORT_NO_FINITE_LOG_EVIDENCE_OR_POSTERIOR')
    q=weights/z
    return dict(evidence=z,posterior=q,negative_log_evidence_gradient=pi-q,status='DEFINED')


def saved_path_statistics(logit_parts,code_parts,top_k):
    rows=[];total_full=total_truncated=0.;count=0
    for k,(x,y) in enumerate(zip(logit_parts,code_parts)):
        logits=np.asarray(x);codes=np.asarray(y)
        assert logits.ndim==2 and codes.shape==(len(logits),)
        assert np.issubdtype(codes.dtype,np.integer) and ((codes>=0)&(codes<logits.shape[-1])).all()
        p=categorical(logits,top_k)
        indices=np.arange(len(codes))
        full=p['full'][indices,codes];truncated=p['truncated'][indices,codes]
        assert (truncated>0).all(),'Saved engineering sample is outside registered top-k support.'
        log_full=np.log(full);log_truncated=np.log(truncated)
        delta=-np.log(p['retained_mass'])
        assert np.allclose(log_truncated-log_full,delta,atol=5e-14,rtol=0)
        total_full+=float(log_full.sum());total_truncated+=float(log_truncated.sum());count+=len(codes)
        rows.append(dict(scale=k,tokens=len(codes),retained_mass_min=float(p['retained_mass'].min()),
            retained_mass_mean=float(p['retained_mass'].mean()),retained_mass_max=float(p['retained_mass'].max()),
            actual_threshold_support_count_min=int(p['support'].sum(-1).min()),
            actual_threshold_support_count_max=int(p['support'].sum(-1).max()),
            saved_path_untruncated_log_probability=float(log_full.sum()),
            saved_path_top_k_log_probability=float(log_truncated.sum()),
            saved_path_log_density_ratio=float(delta.sum())))
    return dict(tokens=count,scales=rows,untruncated_log_probability=total_full,
        top_k_log_probability=total_truncated,saved_path_log_density_ratio=total_truncated-total_full,
        interpretation='SAME_SAVED_ENGINEERING_PATH_UNDER_TWO_KERNELS_NOT_TWO_GENERATION_RUNS_OR_EEG_GAIN')


def run(root,output):
    root=Path(root);output=Path(output)
    if not (root/'local_independent_audit.json').is_file():
        return dict(status='WAIT_REAL_NATIVE_FULL_INDEPENDENT_NUMERICAL_AUDIT',result_payloads_read=0,output_created=False)
    started=time.monotonic()
    audit=json.loads((root/'local_independent_audit.json').read_bytes())
    assert audit['status']==NATIVE_AUDIT_STATUS and audit['protocol']['sha256']==NATIVE_PROTOCOL_SHA
    assert audit['numerical']['all_true_token_terms']==5440000 and audit['numerical']['all_scale_sum_cells']==80000
    assert audit['numerical']['HMM_posterior_cells_audited']==audit['numerical']['HMM_prior_gradient_cells_audited']==240000
    assert not output.exists()
    receiving=json.loads((root/'local_receiving_completion.json').read_bytes())
    manifest_raw=(root/'terminal_manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    assert digest(manifest_raw)==receiving['manifest'] and receiving['synthetic_fixture'] is False
    assert manifest['synthetic_fixture'] is False and manifest['protocol']==audit['protocol']
    references={item['member']:item['ref'] for item in manifest['files']}
    results={};source_refs=[]
    for name,variants in [('fixed_class_native_and_fresh_trace.npz',('cached','fresh')),
                          ('switched_class_fresh_trace.npz',('switched',))]:
        member='result/profile/'+name;path=root/member;raw=path.read_bytes()
        pin=references[member]
        assert digest(raw)=={k:pin[k] for k in ('bytes','sha256')}
        source_refs.append(dict(path=str(path.resolve()),**digest(raw)))
        with np.load(path,allow_pickle=False) as data:
            for variant in variants:
                prefix='' if variant=='switched' else variant+'_'
                logits=[data[f'{prefix}logits_{k}'][0] for k in range(10)]
                codes=[data[f'{prefix}codes_{k}'][0] for k in range(10)]
                results[variant]=saved_path_statistics(logits,codes,600)
                assert results[variant]['tokens']==680
        assert time.monotonic()-started<120
    result=dict(status='COMPLETE_REAL_SAVED_ENGINEERING_PATH_LIKELIHOOD_SAMPLER_LAW_DIAGNOSTIC_NOT_EEG_EFFECT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started,
        source_protocol=audit['protocol'],full_independent_audit=digest((root/'local_independent_audit.json').read_bytes()),
        paths=results,source_trace_files=source_refs,
        token_conditionals_examined=2040,unique_generated_paths=2,
        fixed_cached_fresh_paths_are_replays_not_independent_samples=True,
        arithmetic='FP64 probabilities recomputed from saved FP32 full-vocabulary logits',
        actual_training_T8_target_top600_support_not_recoverable_from_saved_gathered_logprobs=True,
        actual_truncated_T8_HMM_posterior_not_measured=True,
        no_additional_pretrained_forward_or_GPU_or_images=True,
        future_exact_objective_generation_must_use_same_kernel=True,
        new_EEG_training_admitted=False,novelty_or_SOTA=False,
        diagnostic_source=digest(Path(__file__).read_bytes()))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.open('xb').write((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.output),allow_nan=False))
