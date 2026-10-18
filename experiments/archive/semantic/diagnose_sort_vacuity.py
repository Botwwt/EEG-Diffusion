"""Posthoc failure audit only: no parameter fitting, sampler or endpoint changes."""
import argparse
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
from sort_endpoint import load,write,digest,geometry,rank_metrics
from sort_math import unit,conformal_scores,conformal_quantile


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic();cfg=load(a.config);base=Path(cfg['out']);prep=base/'prepare'
    assert not torch.cuda.is_available() and load(base/'evaluate/results.json')['status']=='completed'
    packet=np.load(prep/'EEG_ONLY.npz');bank=unit(np.load(prep/'fit_bank_unit.npy'));contract=load(prep/'data_contract.json')
    primaryq=load(base/'calibrate/calibration.json')['q']['repeat']['0.1'];rows={};details={}
    for group in ['calibration','development']:
        Y=unit(np.load(prep/(group+'_targets_EVALUATION_OR_CALIBRATION_ONLY.npz'))['unit'])
        a0,D,mu,valid,sigma,pairs=geometry(cfg,packet,group,'paired',bank,contract['fit_scale_floor'])
        margin=np.einsum('bld,bd->bl',D,Y);scores=conformal_scores(D,mu,sigma,valid,Y)
        critical=np.where(valid,cfg['rho']*mu/sigma,0.).max(1)
        rank,win,cos=rank_metrics(a0,Y);correct=rank==1
        rows[group]=dict(queries=len(Y),a0_Recall1=float(correct.mean()),
            sign_agreement_edge_fraction=float((margin[valid]>0).mean()),rho_margin_satisfied_edge_fraction=float((margin[valid]>=cfg['rho']*mu[valid]).mean()),
            uncalibrated_set_target_exclusion=float((scores>1e-8).mean()),
            uncalibrated_target_exclusion_given_a0_correct=float((scores[correct]>1e-8).mean()) if correct.any() else None,
            critical_q_quantiles=np.quantile(critical,[0,.5,.9,.95,1]).tolist(),main_calibration_q=primaryq,
            active_queries_at_primary_q=int((critical>primaryq).sum()),
            score_max_local_q90=float(conformal_quantile(np.maximum((cfg['rho']*mu[:,:32]-margin[:,:32])/sigma[:,:32],0).max(1),.1)),
            score_max_global_q90=float(conformal_quantile(np.maximum((cfg['rho']*mu[:,32:]-margin[:,32:])/sigma[:,32:],0).max(1),.1)),
            selected_edges=64,source_images_from_fit_bank_only=True)
        details[group]=dict(scores=scores.tolist(),critical_q=critical.tolist(),a0_rank=rank.tolist(),
                           margin_mean=margin.mean(1).tolist(),margin_min=margin.min(1).tolist())
    # Exact counterexample with disjoint target/gallery and fit-anchor vectors.
    a0=np.array([1.,0.]);Y=np.array([.6,.8]);g=np.array([-.8,.6]);bp=np.array([.6,-.8]);bn=np.array([-.6,.8]);d=unit(bp-bn)
    synthetic=dict(label='synthetic-pass',a0_correctly_retrieves_target=bool(a0@Y>a0@g),
                   eeg_fit_anchor_margin=float(d@a0),true_image_fit_anchor_margin=float(d@Y),
                   implication_from_correct_retrieval_to_anchor_order_is_false=True)
    assert synthetic['a0_correctly_retrieves_target'] and d@a0>0 and d@Y<0
    write(out/'per_query.json',details)
    report=dict(status='completed',source_sha256=digest(__file__),posthoc=True,official_test_read=False,new_optimizer_updates=0,new_samples=0,new_images=0,
        results=rows,synthetic_query_order_counterexample=synthetic,
        scale_invariance='For positive global c, sigma_new=c*sigma and q_new=q/c, hence every bound rho*mu-q*sigma is unchanged. Uniform scale inflation cannot fix vacuity.',
        limitations='This diagnoses fixed rules and frozen a0; not all ordinal models. Local/global diagnostic quantiles are not deployed or selected on dev.',
        seconds=time.monotonic()-start)
    write(out/'results.json',report);print(json.dumps(report))


if __name__=='__main__':main()
