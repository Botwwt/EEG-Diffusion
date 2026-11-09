"""Post-terminal numerical diagnostic, not a new EEG model or training recipe.

Uses the already measured, complete native likelihood table. At uniform logits
and rho=1 it compares exact full-680-token gradients with finite importance
banks. Proposals are fixed with respect to EEG parameters. No sparse native
execution, GPU speedup, EEG predictability, or image gain is established here.
The seven sealed native-engineering sources are not imported or modified.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np


PROTOCOL_SHA = '87d502f75f81ddc8f1b0855a1ab05520f0cf8d309f09a9bbb8f5097b72c16799'
AUDIT_STATUS = 'PASS_LOCAL_FULL_NATIVE_ENGINEERING_NUMERICAL_AND_PROVENANCE_AUDIT_NOT_EEG_EFFICACY'
SAMPLES = (16, 64, 256)
REPLICATES = 1024
SEED = 1729
MAX_SECONDS = 120


def digest(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def lse(value, axis=-1):
    value = np.asarray(value, np.float64)
    maximum = np.max(value, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.exp(value - maximum).sum(axis=axis, keepdims=True)), axis=axis)


def probabilities(logits):
    logits = np.asarray(logits, np.float64)
    assert logits.ndim == 1 and len(logits) >= 2 and np.isfinite(logits).all()
    return np.exp(logits - lse(logits))


def setup(logits, log_likelihood, proposal):
    theta = np.asarray(logits, np.float64)
    ell = np.asarray(log_likelihood, np.float64)
    q = np.asarray(proposal, np.float64)
    assert theta.ndim == 1 and theta.shape == ell.shape == q.shape
    assert len(theta) >= 2 and np.isfinite(theta).all() and np.isfinite(ell).all()
    assert np.isfinite(q).all() and (q > 0).all() and abs(q.sum() - 1.) < 1e-13
    # A class-common shift cancels from posterior, gradient and all error ratios.
    offset = float(ell.max())
    centered = ell - offset
    logpi = theta - lse(theta)
    logz = float(lse(logpi + centered))
    posterior = np.exp(logpi + centered - logz)
    pi = np.exp(logpi)
    return dict(logpi=logpi, pi=pi, centered=centered, offset=offset,
                centered_log_evidence=logz, log_evidence=logz + offset,
                posterior=posterior, positive_log_likelihood_gradient=posterior-pi, q=q)


def fixed_bank(state, indices):
    """Each row is an independently drawn importance bank, sampled with replacement."""
    ids = np.asarray(indices)
    K = len(state['pi'])
    assert ids.ndim == 2 and ids.shape[1] >= 1 and np.issubdtype(ids.dtype, np.integer)
    assert ((ids >= 0) & (ids < K)).all()
    logw = state['logpi'][ids] + state['centered'][ids] - np.log(state['q'][ids])
    norm = lse(logw, 1)
    weights = np.exp(logw - norm[:, None])
    posterior = np.zeros((len(ids), K), np.float64)
    np.add.at(posterior, (np.arange(len(ids))[:, None], ids), weights)
    assert np.allclose(posterior.sum(1), 1., atol=1e-13, rtol=0)
    return dict(posterior=posterior,
                centered_log_estimate=norm-np.log(ids.shape[1]),
                importance_effective_sample_size=1./(weights*weights).sum(1))


def cosine(a, b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return None if denominator <= 1e-14 else float(np.dot(a, b)/denominator)


def exact_one_sample(state):
    """Exact expectation over all K banks; no Monte Carlo error for M=1.

    For one draw the expected log-estimator gradient is q-pi, regardless of
    the full target likelihood. This is the log-estimator objective, not an
    unbiased estimator of the original complete-likelihood gradient.
    """
    q = state['q']
    values = state['logpi'] + state['centered'] - np.log(q)
    estimate_gradient = q-state['pi']
    exact_gradient = state['positive_log_likelihood_gradient']
    return dict(samples=1, expectation='EXACT_ALL_CLASSES_NO_MONTE_CARLO',
                expected_likelihood_ratio=float(np.dot(q, np.exp(values-state['centered_log_evidence']))),
                exact_log_evidence_minus_expected_log_estimate=float(state['centered_log_evidence']-np.dot(q, values)),
                mean_gradient_L2_bias=float(np.linalg.norm(estimate_gradient-exact_gradient)),
                mean_gradient_cosine_to_exact=cosine(estimate_gradient, exact_gradient),
                expected_gradient_is_q_minus_pi=True,
                full_likelihood_still_evaluated_for_this_diagnostic=True)


def sampled_statistics(state, samples, replicates, seed_sequence, *, deadline):
    rng = np.random.default_rng(seed_sequence)
    K = len(state['pi'])
    total = np.zeros(K)
    square = np.zeros(K)
    logvalues = []
    ratios = []
    projections = []
    ess = []
    exact_gradient = state['positive_log_likelihood_gradient']
    for begin in range(0, replicates, 64):
        assert time.monotonic() < deadline, 'Finite CPU diagnostic deadline exceeded; no fit or native retry.'
        n = min(64, replicates-begin)
        ids = rng.choice(K, size=(n, samples), replace=True, p=state['q'])
        bank = fixed_bank(state, ids)
        p = bank['posterior']
        total += p.sum(0)
        square += (p*p).sum(0)
        value = bank['centered_log_estimate']
        logvalues.extend(value.tolist())
        ratios.extend(np.exp(value-state['centered_log_evidence']).tolist())
        projections.extend(((p-state['pi']) @ exact_gradient).tolist())
        ess.extend(bank['importance_effective_sample_size'].tolist())
    mean_p = total/replicates
    variance = np.maximum(0., (square-total*total/replicates)/(replicates-1))
    gradient = mean_p-state['pi']
    logvalues = np.asarray(logvalues)
    ratios = np.asarray(ratios)
    projections = np.asarray(projections)
    ess = np.asarray(ess)
    return dict(samples=samples, independent_banks=replicates,
                expected_likelihood_ratio_MC_estimate=float(ratios.mean()),
                likelihood_ratio_MC_standard_error=float(ratios.std(ddof=1)/np.sqrt(replicates)),
                exact_log_evidence_minus_mean_log_estimate=float(state['centered_log_evidence']-logvalues.mean()),
                mean_log_estimate_MC_standard_error=float(logvalues.std(ddof=1)/np.sqrt(replicates)),
                mean_gradient_L2_distance_to_exact=float(np.linalg.norm(gradient-exact_gradient)),
                mean_gradient_MC_noise_RMS=float(np.sqrt(variance.sum()/replicates)),
                mean_gradient_cosine_to_exact=cosine(gradient, exact_gradient),
                mean_gradient_dot_exact=float(projections.mean()),
                gradient_dot_exact_MC_standard_error=float(projections.std(ddof=1)/np.sqrt(replicates)),
                exact_gradient_squared_norm=float(np.dot(exact_gradient, exact_gradient)),
                importance_effective_sample_size_mean=float(ess.mean()),
                importance_effective_sample_size_min=float(ess.min()),
                per_bank_log_estimate_can_exceed_exact=True,
                uncertainty_scope='IID_MONTE_CARLO_ERROR_ONLY_NOT_EEG_OR_TRAINING_SEED_UNCERTAINTY')


def load_terminal(root):
    root = Path(root)
    if not root.exists():
        return dict(status='WAIT_NATIVE_TERMINAL_ARCHIVE_NOT_RECEIVED', result_payloads_read=0)
    audit_file = root/'local_independent_audit.json'
    if not audit_file.exists():
        return dict(status='WAIT_COMPLETE_INDEPENDENT_NATIVE_NUMERICAL_AUDIT', result_payloads_read=0)
    audit = json.loads(audit_file.read_bytes())
    assert audit['status'] == AUDIT_STATUS and audit['protocol']['sha256'] == PROTOCOL_SHA
    assert audit['numerical']['all_true_token_terms'] == 5440000
    assert audit['numerical']['all_scale_sum_cells'] == 80000
    assert audit['second_method_or_SOTA_or_CCF_A'] is False
    manifest = json.loads((root/'terminal_manifest.json').read_bytes())
    assert manifest['synthetic_fixture'] is False and manifest['protocol']['sha256'] == PROTOCOL_SHA
    receiving = json.loads((root/'local_receiving_completion.json').read_bytes())
    assert receiving['synthetic_fixture'] is False
    assert digest((root/'terminal_manifest.json').read_bytes()) == receiving['manifest']
    member = 'result/profile/scale_log_likelihoods.npy'
    matches = [r for r in manifest['files'] if r['member'] == member]
    assert len(matches) == 1
    source = root/member
    source_ref = digest(source.read_bytes())
    assert source_ref == {k: matches[0]['ref'][k] for k in ('bytes','sha256')}
    ll = np.load(source, allow_pickle=False)
    assert ll.shape == (8,1000,10) and ll.dtype == np.float64 and np.isfinite(ll).all() and (ll<=0).all()
    return dict(status='READY_REAL_NATIVE_TERMINAL_INPUT', likelihoods=ll,
                source=dict(path=str(source), **source_ref),
                independent_audit=dict(path=str(audit_file), **digest(audit_file.read_bytes())))


def run(root, destination):
    started = time.monotonic()
    loaded = load_terminal(root)
    if loaded['status'] != 'READY_REAL_NATIVE_TERMINAL_INPUT':
        print(json.dumps(loaded)); return loaded
    destination = Path(destination)
    assert not destination.exists(), 'Preserve prior diagnostic evidence; no overwrite.'
    results = []
    for index, ll in enumerate(loaded.pop('likelihoods')):
        full = ll.sum(1)
        coarse = ll[:,:2].sum(1)  # Complete first two native scales: 1+4 tokens.
        coarse_p = probabilities(coarse)
        row = dict(T_profile_row=index, priors='UNIFORM_1000_CLASS_ONLY_NOT_TRAINED_EEG', rho=1.,
                   full_native_tokens=680, coarse_native_tokens=5, proposals={})
        for proposal_index, (name, q) in enumerate((('uniform', np.full(1000,.001)),
                    ('half_uniform_half_coarse5', .5/1000+.5*coarse_p))):
            state = setup(np.zeros(1000), full, q)
            row['exact_gradient_norm'] = float(np.linalg.norm(state['positive_log_likelihood_gradient']))
            row['coarse5_vs_full680_gradient_cosine'] = cosine(coarse_p-state['pi'],state['positive_log_likelihood_gradient'])
            row['proposals'][name] = dict(minimum_proposal_probability=float(q.min()),
                one_sample_exact=exact_one_sample(state), sampled=[sampled_statistics(state,m,REPLICATES,
                    np.random.SeedSequence([SEED,index,proposal_index,m]),deadline=started+MAX_SECONDS) for m in SAMPLES])
        results.append(row)
    result = dict(status='COMPLETE_REAL_NATIVE_FULL_GLOBAL_CLASS_SAMPLING_GRADIENT_DIAGNOSTIC_NOT_EEG_FIT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), input=loaded, rows=results,
        source=digest(Path(__file__).read_bytes()), seconds=time.monotonic()-started,
        diagnostic_seed=SEED, sample_budgets=list(SAMPLES), independent_banks_per_configuration=REPLICATES,
        no_configuration_selected_as_training_recipe=True, real_EEG_logits_tested=False,
        copy_refresh_half_case_tested=False, all_native_emissions_already_computed=True,
        sparse_native_speedup_implemented_or_measured=False, uncertainty_is_MC_only=True,
        new_GPU_jobs=0,new_optimizer_updates=0,new_research_images=0,
        second_method_or_SOTA_or_CCF_A=False)
    encoded=json.dumps(result,indent=2,allow_nan=False).encode('utf-8')+b'\n'
    assert len(encoded) <= 1<<20 and time.monotonic()-started < MAX_SECONDS
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'actual_real_native_gradient_sampling_diagnostic.json').write_bytes(encoded)
    print(json.dumps(dict(status=result['status'],destination=str(destination),seconds=result['seconds'],
                          actual_native_rows=8,real_EEG_logits_tested=False)))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.destination)
