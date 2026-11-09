"""Independent full replay of all 48 existing T8 Monte Carlo diagnostics.

Uses direct probabilities, inverse-CDF uniforms and per-bank bincounts instead
of the producer's log-normalized scatter accumulation. No proposal, budget,
EEG parameter or training configuration is selected from these results.
"""
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def digest(path):
    raw=Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def cosine(a,b):
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))


def main():
    started=time.monotonic();root=Path('audit/VAR_multiscale_native_profile_complete_20261101_v3')
    diagnostic=Path('audit/VAR_full_target_sampling_gradient_actual_20261101_v2/actual_real_native_gradient_sampling_diagnostic.json')
    actual=json.loads(diagnostic.read_bytes())
    assert actual['status']=='COMPLETE_REAL_NATIVE_FULL_GLOBAL_CLASS_SAMPLING_GRADIENT_DIAGNOSTIC_NOT_EEG_FIT'
    assert actual['sample_budgets']==[16,64,256] and actual['independent_banks_per_configuration']==1024 and actual['diagnostic_seed']==1729
    proof=json.loads((root/'local_independent_audit.json').read_bytes())
    assert proof['numerical']['all_true_token_terms']==5440000 and proof['numerical']['HMM_prior_gradient_cells_audited']==240000
    file=root/'result/profile/scale_log_likelihoods.npy'
    assert {k:digest(file)[k] for k in ('bytes','sha256')}=={k:actual['input']['source'][k] for k in ('bytes','sha256')}
    likelihood=np.load(file,allow_pickle=False);assert likelihood.shape==(8,1000,10)
    all_errors=[];rows=[];all_banks=all_draws=0
    for i,row in enumerate(actual['rows']):
        total=np.zeros(1000)
        for k in range(10):total+=likelihood[i,:,k]
        centered=total-total.max();f=np.exp(centered)
        true=f/f.sum();prior=np.full(1000,.001);true_g=true-prior
        coarse=likelihood[i,:,0]+likelihood[i,:,1]
        coarse_exp=np.exp(coarse-coarse.max());coarse_p=coarse_exp/coarse_exp.sum()
        all_errors.append(abs(float(np.linalg.norm(true_g))-row['exact_gradient_norm']))
        all_errors.append(abs(cosine(coarse_p-prior,true_g)-row['coarse5_vs_full680_gradient_cosine']))
        result=dict(T=i,configurations=[])
        for proposal_index,(name,q) in enumerate((('uniform',prior.copy()),('half_uniform_half_coarse5',.5*prior+.5*coarse_p))):
            entry=row['proposals'][name]
            one=entry['one_sample_exact']
            all_errors.append(abs(float(np.linalg.norm(q-true))-one['mean_gradient_L2_bias']))
            # Exact one-sample expectation: likelihood remains unbiased, log-gradient does not.
            all_errors.append(abs(1.-one['expected_likelihood_ratio']))
            cdf=np.cumsum(q);cdf/=cdf[-1]
            for source in entry['sampled']:
                m=source['samples'];n=1024
                rng=np.random.default_rng(np.random.SeedSequence([1729,i,proposal_index,m]))
                uniforms=rng.random((n,m))
                ids=np.searchsorted(cdf,uniforms,side='right')
                weights=f[ids]/(1000*q[ids])
                totals=weights.sum(1)
                posterior=np.stack([np.bincount(ids[j],weights=weights[j],minlength=1000)/totals[j] for j in range(n)])
                assert np.allclose(posterior.sum(1),1.,atol=1e-14,rtol=0)
                ess=totals**2/np.square(weights).sum(1)
                mean=posterior.mean(0);g=mean-prior
                ratios=totals/m/(f.sum()/1000)
                logs=np.log(totals/m)
                projections=(posterior-prior)@true_g
                measures=dict(expected_likelihood_ratio_MC_estimate=float(ratios.mean()),
                    likelihood_ratio_MC_standard_error=float(ratios.std(ddof=1)/np.sqrt(n)),
                    exact_log_evidence_minus_mean_log_estimate=float(np.log(f.sum()/1000)-logs.mean()),
                    mean_log_estimate_MC_standard_error=float(logs.std(ddof=1)/np.sqrt(n)),
                    mean_gradient_L2_distance_to_exact=float(np.linalg.norm(mean-true)),
                    mean_gradient_MC_noise_RMS=float(np.sqrt(posterior.var(0,ddof=1).sum()/n)),
                    mean_gradient_cosine_to_exact=cosine(g,true_g),
                    mean_gradient_dot_exact=float(projections.mean()),
                    gradient_dot_exact_MC_standard_error=float(projections.std(ddof=1)/np.sqrt(n)),
                    exact_gradient_squared_norm=float(true_g@true_g),
                    importance_effective_sample_size_mean=float(ess.mean()),
                    importance_effective_sample_size_min=float(ess.min()))
                errors={key:abs(value-source[key]) for key,value in measures.items()}
                assert max(errors.values())<5e-9,(i,name,m,errors)
                all_errors.extend(errors.values());all_banks+=n;all_draws+=n*m
                result['configurations'].append(dict(proposal=name,M=m,banks=n,
                    maximum_statistic_error=max(errors.values()),
                    mean_gradient_L2_distance=measures['mean_gradient_L2_distance_to_exact'],
                    mean_gradient_MC_noise_RMS=measures['mean_gradient_MC_noise_RMS'],
                    bias_distance_divided_by_MC_noise_RMS=measures['mean_gradient_L2_distance_to_exact']/measures['mean_gradient_MC_noise_RMS'],
                    draw_indices_SHA256=hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()))
                assert time.monotonic()-started<120
        rows.append(result)
    assert all_banks==49152 and all_draws==5505024
    result=dict(status='PASS_ALL48_REAL_T8_SAMPLING_CONFIGURATIONS_INDEPENDENT_DIRECT_PROBABILITY_REPLAY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started,
        configurations=48,banks=all_banks,individual_class_draws=all_draws,
        maximum_statistic_error=max(all_errors),rows=rows,
        producer_diagnostic=digest(diagnostic),native_likelihood=digest(file),source=digest(__file__),
        scope='UNIFORM_PRIOR_GLOBAL_CLASS_LOGIT_GRADIENT_MC_ONLY_NOT_TRAINED_EEG',
        no_configuration_selected=True,new_GPU_or_training_or_images=0)
    destination=diagnostic.parent/'independent_full_MC_replay.json';assert not destination.exists()
    destination.open('xb').write((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({key:result[key] for key in ('status','seconds','configurations','banks','individual_class_draws','maximum_statistic_error')}))


if __name__=='__main__':main()
