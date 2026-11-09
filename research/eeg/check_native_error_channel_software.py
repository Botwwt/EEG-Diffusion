"""Synthetic numerical checks only, not evidence from actual EEG errors."""
from pathlib import Path
import json
import hashlib
import torch

from native_error_channel import (NativeErrorChannel,calibrate_from_dfit,
                                 apply_calibration,stream_seed)


def main():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(1729)
    n,d=960,16;concepts=[f'concept-{i//6:03}' for i in range(n)]
    centers=torch.randn(n//6,d,generator=g)
    ci=centers.repeat_interleave(6,0)+.01*torch.randn(n,d,generator=g)
    # Residual depends on image content. NN is allowed to fail on real data.
    ce=.4*ci+.02*torch.randn(n,d,generator=g)
    fit_ci=torch.randn(400,d,generator=g)
    fit_oof=torch.stack([.4*fit_ci+.02*torch.randn(400,d,generator=g) for _ in range(2)])
    fit_full=.4*fit_ci+.02*torch.randn(400,d,generator=g)
    model=NativeErrorChannel.fit(ci,ce,concepts,fit_ci,fit_oof,fit_full,[.5,.5],8,8,.1)
    q=ci[:40];qc=concepts[:40];seeds=[stream_seed(1729,'synthetic',i) for i in range(len(q))]
    nearest=model.nearest(q,qc)
    for i,row in enumerate(nearest):
        cs=[concepts[int(j)] for j in row]
        assert qc[i] not in cs and len(set(cs))==8
    for arm in ['Clean','Iso','Moment','Conditional']:
        a,_=model.sample(arm,q,qc,seeds,nearest)
        b,_=model.sample(arm,q,qc,seeds,nearest)
        assert torch.equal(a,b)
    # Per-query streams must be invariant to batching/reordering.
    a,_=model.sample('Moment',q,qc,seeds)
    order=torch.arange(len(q)-1,-1,-1)
    b,_=model.sample('Moment',q[order],[qc[i] for i in order],[seeds[i] for i in order])
    assert torch.equal(a[order],b)
    # Calibration must match Dfit mean and trace in double precision.
    calibration=calibrate_from_dfit(fit_ci,fit_oof,fit_full,[.5,.5])
    delta=apply_calibration((fit_oof-fit_ci).reshape(-1,d),calibration).double()
    target=(fit_full-fit_ci).double();mu=target.mean(0)
    assert torch.max(abs(delta.mean(0)-mu))<1e-6
    trace=(delta-delta.mean(0)).square().sum(-1).mean()
    target_trace=(target-mu).square().sum(-1).mean()
    assert abs(float(trace/target_trace)-1)<1e-6
    # Gaussian factors must preserve the registered total residual energy.
    state=model.state
    implied=float(state['bias'].double().square().sum()+state['covariance_cholesky'].double().square().sum())
    assert abs(implied/state['energy']-1)<1e-6
    # No target/features/seed are silently changed by the clean arm.
    clean,_=model.sample('Clean',q,qc,seeds);assert torch.equal(clean,q)
    destination=Path(__file__).parent/'audit/native_error_channel_software_checks.json'
    destination.write_text(json.dumps(dict(status='SYNTHETIC_SOFTWARE_CHECKS_PASSED',
        tests=['distinct_other_concepts','fixed_query_reproducibility','batch_order_invariance',
               'Dfit_bias_trace_calibration','Moment_energy_preservation','Clean_identity'],
        dimensions=d,synthetic_T_rows=n,actual_EEG_data_used=False,
        fitted_scientific_error_channel=False,GPU_initialized=torch.cuda.is_initialized(),
        source_sha256=hashlib.sha256(Path(__file__).with_name('native_error_channel.py').read_bytes()).hexdigest(),
        limits=['No actual OOF teacher, Dcheck distribution validation, adapter or image gain tested.']),indent=2)+'\n')
    print(destination.read_text())


if __name__=='__main__':main()
