"""Exact paired-function and two-path accounting; no fitted predictor."""
import numpy as np

DOMAINS=('DDPM_T','ordinary_path','biased_path')
CASES=('ordinary_correct','ordinary_donor','biased_correct','biased_donor')
EPS=1e-12

def vector_stats(value,reference):
    x=np.asarray(value,np.float64).reshape(-1);r=np.asarray(reference,np.float64).reshape(-1)
    assert x.shape==r.shape and np.isfinite(x).all() and np.isfinite(r).all()
    xx=float(x@x/len(x));rr=float(r@r/len(r));xr=float(x@r/len(x));change=float(np.mean((x-r)**2))
    valid=rr>EPS
    return dict(value_energy=xx,reference_energy=rr,dot=xr,change_energy=change,
                valid_reference=valid,aligned_gain=xr/rr if valid else None,
                relative_change_energy=change/rr if valid else None,
                cosine=xr/np.sqrt(xx*rr) if valid and xx>EPS else None)

def symmetric_paths(ordinary_at_ordinary,biased_at_ordinary,ordinary_at_biased,biased_at_biased):
    a,b,c,d=[np.asarray(v,np.float64) for v in
             (ordinary_at_ordinary,biased_at_ordinary,ordinary_at_biased,biased_at_biased)]
    assert a.shape==b.shape==c.shape==d.shape and all(np.isfinite(v).all() for v in (a,b,c,d))
    direct=.5*((b-a)+(d-c));state=.5*((c-a)+(d-b));total=d-a
    error=float(np.max(np.abs(direct+state-total)))
    energy_error=abs(float(np.mean(total**2))-float(np.mean(direct**2)+np.mean(state**2)+2*np.mean(direct*state)))
    assert error<1e-10 and energy_error<1e-10
    return dict(direct=direct,state=state,total=total,identity_error=error,energy_identity_error=energy_error,
                direct_energy=float(np.mean(direct**2)),state_energy=float(np.mean(state**2)),
                cross_energy=float(2*np.mean(direct*state)),total_energy=float(np.mean(total**2)))

def numpy_checks():
    rng=np.random.default_rng(1729)
    a,b,c,d=rng.normal(size=(4,32));proof=symmetric_paths(a,b,c,d)
    assert vector_stats(np.zeros(4),np.zeros(4))['aligned_gain'] is None
    assert vector_stats(a,a)['relative_change_energy']==0
    # A constant local offset changes a downstream nonlinear paired effect.
    x1,x2,offset=2.,1.,.5
    linear_before=x1-x2;linear_after=(x1+offset)-(x2+offset)
    quadratic_before=x1*x1-x2*x2;quadratic_after=(x1+offset)**2-(x2+offset)**2
    assert linear_before==linear_after and quadratic_after!=quadratic_before
    assert quadratic_after-quadratic_before==2*offset*(x1-x2)
    assert np.allclose(symmetric_paths(a,b,a,b)['state'],0)
    assert np.allclose(symmetric_paths(a,a,c,c)['direct'],0)
    return dict(status='PASS_EXACT_PATH_ACCOUNTING_LOCAL_NONLINEAR_COUNTEREXAMPLE_AND_ZERO_GUARDS',
                vector_identity_error=proof['identity_error'],energy_identity_error=proof['energy_identity_error'],
                scalar_nonlinear_paired_effect_change=quadratic_after-quadratic_before,
                no_theoretical_novelty_or_EEG_effect_claim=True)
