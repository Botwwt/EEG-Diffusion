"""Independent scalar audit of the closed matched-EEG intervention diagnostic."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

L=Path(__file__).resolve().parent
P=L/'audit/EEG_native_marginal_images_20261029_v2/received_stage400_v1'
OUT=L/'audit/VAR_closed_coupled_EEG_swap_20261101_v1'


def read(path):return json.loads(Path(path).read_bytes())


def digest(path):
    raw=Path(path).read_bytes();return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def normalized_scalars(array):
    return [[float(value)/math.sqrt(math.fsum(float(v)*float(v) for v in row)) for value in row] for row in array]


def main():
    started=time.monotonic();result=read(OUT/'analysis.json');stored=np.load(OUT/'all400_quantities_and_bootstrap.npz',allow_pickle=False)
    for pin in result['source_refs']:
        assert digest(pin['path'])=={k:pin[k] for k in ('bytes','sha256')}
    panel=read(P/'old/panel400.json')['queries'];correct=read(P/'old/coupled_records.json');donor=read(P/'old/donor_records.json')
    received=read(P/'receiving_completion.json')
    for name,pin in received['files'].items():assert digest(P/name)=={k:pin['remote'][k] for k in ('bytes','sha256')}
    mapping=[row['EEG_row'] for row in donor]
    assert sorted(mapping)==list(range(400)) and all(index!=other for index,other in enumerate(mapping))
    for index,other in enumerate(mapping):
        assert panel[index]['concept_id']==panel[other]['concept_id']
        assert correct[index]['latent_seed']==donor[index]['latent_seed'] and correct[index]['token_seed']==donor[index]['token_seed']
        assert correct[index]['fit_model_sha256']==donor[index]['fit_model_sha256']
        assert correct[other]['EEG_parameters_sha256']==donor[index]['EEG_parameters_sha256']
        position=index
        for _ in range(10):position=mapping[position]
        assert position==index
    target=normalized_scalars(np.load(P/'old/targets400.npy',allow_pickle=False))
    coupled=normalized_scalars(np.load(P/'old/coupled_features.npy',allow_pickle=False))
    wrong=normalized_scalars(np.load(P/'old/donor_features.npy',allow_pickle=False))
    scalar=np.array([math.fsum((a-b)*(t-u) for a,b,t,u in zip(coupled[i],wrong[i],target[i],target[mapping[i]]))/2 for i in range(400)])
    name='crossed_EEG_target_feature_contrast';column=list(stored['quantity_names']).index(name)
    maxerror=float(np.max(np.abs(scalar-stored['query_values'][:,column])));assert maxerror<1e-12
    concepts=sorted({row['concept_id'] for row in panel});groups=[[i for i,row in enumerate(panel) if row['concept_id']==c] for c in concepts]
    means=np.array([math.fsum(float(scalar[i]) for i in group)/10 for group in groups]);assert means.shape==(40,)
    draws=stored['bootstrap_draws'];assert np.array_equal(draws,np.random.default_rng(1729).integers(0,40,size=(1000,40)))
    bootstrap=np.array([math.fsum(float(means[i]) for i in row)/40 for row in draws])
    assert np.max(np.abs(bootstrap-stored['bootstrap_values'][:,column]))<1e-12
    ci=np.quantile(bootstrap,[.025,.975]);assert np.max(np.abs(ci-result['summary'][name]['descriptive_concept_bootstrap95']))<1e-12
    # Target differences sum to zero within each complete donor cycle.
    target=np.array(target);coupled=np.array(coupled);wrong=np.array(wrong);dt=target-target[mapping];dg=coupled-wrong
    centered=np.empty_like(dg);cycle_residual=[]
    for group in groups:
        cycle_residual.append(float(np.max(np.abs(dt[group].sum(0)))))
        centered[group]=dg[group]-dg[group].mean(0)
    centered_values=np.einsum('nd,nd->n',centered,dt)/2
    centered_concept_means=np.array([centered_values[group].mean() for group in groups])
    assert np.max(np.abs(centered_concept_means-means))<1e-12
    # Report every omission; no deletion changes the primary 400-row result.
    omitted_draws=np.random.default_rng(1729).integers(0,39,size=(1000,39));jackknife=[]
    for index,concept in enumerate(concepts):
        remaining=np.delete(means,index);samples=remaining[omitted_draws].mean(1)
        jackknife.append(dict(omitted_concept=concept,diagnostic_mean=float(remaining.mean()),
            diagnostic_descriptive_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),primary_queries_not_removed=True))
    alignment=np.einsum('nd,nd->n',dg,dt)
    delta_norm=np.linalg.norm(dg,axis=1);target_norm=np.linalg.norm(dt,axis=1)
    assert np.all(target_norm>0);cosine=np.divide(alignment,delta_norm*target_norm,out=np.zeros(400),where=delta_norm>0)
    audit=dict(status='PASS_INDEPENDENT_ALL400_SCALAR_SOURCE_RNG_DONOR_CYCLE_AND1000_BOOTSTRAP_AUDIT',
        source_analysis=digest(OUT/'analysis.json'),source_program=digest(__file__),all400_rows_audited=True,
        shared_latent_and_token_seed_records_verified=400,independent_scalar_maxerror=maxerror,
        independent_bootstrap_maxerror=float(np.max(np.abs(bootstrap-stored['bootstrap_values'][:,column]))),
        target_cycle_coordinate_sum_maxerror=max(cycle_residual),concept_constant_feature_change_cancels=True,
        centered_and_uncentered_concept_mean_maxerror=float(np.max(np.abs(centered_concept_means-means))),
        all40_leave_one_concept_diagnostics=jackknife,
        leave_one_mean_range=[min(row['diagnostic_mean'] for row in jackknife),max(row['diagnostic_mean'] for row in jackknife)],
        minimum_leave_one_bootstrap_lower=min(row['diagnostic_descriptive_bootstrap95'][0] for row in jackknife),
        positive_original_concepts=int((means>0).sum()),zero_image_feature_change_rows=int((delta_norm==0).sum()),
        mean_feature_change_target_change_cosine=float(cosine.mean()),
        all_primary_queries_retained=400,post_hoc_exploratory_not_independent_confirmation=True,
        no_original_gate_or_registered_experiment_changed=True,new_GPU_jobs=0,new_optimizer_updates=0,new_images=0,
        actual_GPU_RNG_or_image_encoder_reforward=False,new_training_admission=False,seconds=time.monotonic()-started)
    path=OUT/'independent_scalar_and_concept_influence_audit.json';assert not path.exists()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<1<<20
    with path.open('x',encoding='utf-8') as stream:json.dump(audit,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='all40_leave_one_concept_diagnostics'}))


if __name__=='__main__':main()
