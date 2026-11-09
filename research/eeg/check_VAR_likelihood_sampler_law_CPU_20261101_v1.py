"""Exact finite support counterexamples, no native or EEG data."""
import json
import math
from pathlib import Path
import numpy as np
import diagnose_VAR_likelihood_sampler_law_CPU_20261101_v1 as M


def check():
    # Both classes retain target0, yet their posterior preference reverses.
    probs=np.array([[.4,.5,.06,.04],[.39,.38,.12,.11]])
    state=M.categorical(np.log(probs),2)
    full=M.mixture_posterior(state['full'],[.5,.5],0)
    top=M.mixture_posterior(state['truncated'],[.5,.5],0)
    assert full['posterior'][0]>.5 and top['posterior'][0]<.5
    assert np.allclose(state['full'],probs,atol=1e-15,rtol=0)
    # Direct scalar normalization of each retained support, independent of the implementation.
    direct=np.array([[.4/.9,.5/.9,0.,0.],[.39/.77,.38/.77,0.,0.]])
    assert np.max(np.abs(direct-state['truncated']))<1e-15
    assert np.allclose(M.categorical(np.log(probs),0)['truncated'],probs,atol=1e-15,rtol=0)
    zero=M.mixture_posterior(M.categorical(np.log(probs),1)['truncated'],[.5,.5],3)
    assert zero['evidence']==0. and zero['posterior'] is None
    # The actual threshold sampler includes ties, so support need not equal k.
    tied=M.categorical(np.log([[.4,.2,.2,.2]]),2)
    assert tied['support'].sum()==4 and np.array_equal(tied['full'],tied['truncated'])
    trace=M.saved_path_statistics([np.log(probs[:1])],[np.array([0])],2)
    assert abs(trace['saved_path_log_density_ratio']+math.log(.9))<1e-14
    # Complete finite sequence probabilities normalize under EACH kernel.
    # They are different normalized laws, not an invalid arithmetic normalization.
    total_full=total_top=0.
    for first in range(4):
        for second in range(4):
            total_full+=float(probs[0,first]*probs[1,second])
            total_top+=float(direct[0,first]*direct[1,second])
    assert abs(total_full-1.)<1e-14 and abs(total_top-1.)<1e-14
    output=Path('audit/VAR_likelihood_sampler_law_20261101_v1/actual_synthetic_support_checks.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS_SYNTHETIC_NORMALIZATION_POSTERIOR_REVERSAL_ZERO_SUPPORT_AND_TIE_RULE',
        all_tests_use_synthetic_probabilities=True,
        full_posterior=full['posterior'].tolist(),top2_posterior=top['posterior'].tolist(),
        full_gradient=full['negative_log_evidence_gradient'].tolist(),top2_gradient=top['negative_log_evidence_gradient'].tolist(),
        top2_retained_masses=state['retained_mass'].tolist(),
        no_support_posterior=zero,tie_threshold_actual_support=4,
        saved_path_identity=trace,new_GPU_or_training_or_research_images=0,
        source=M.digest(Path(M.__file__).read_bytes()))
    output.open('xb').write((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':check()
