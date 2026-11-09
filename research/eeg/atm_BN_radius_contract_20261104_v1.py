"""T-only BN state control and exact old/new radius-direction interventions."""
import json,sys
from pathlib import Path
import numpy as np
ARMS=('direct','nce_continued','conditional','BN_only','nce_direction_only','nce_radius_only')
BN_NAMES=('enc_eeg.0.tsconv.2','enc_eeg.0.tsconv.5')
_cache=None
def decompose(old,new):
    a=np.asarray(old,np.float64);b=np.asarray(new,np.float64);assert a.shape==b.shape and a.ndim==2 and np.isfinite(a).all() and np.isfinite(b).all()
    ra=np.linalg.norm(a,axis=1,keepdims=True);rb=np.linalg.norm(b,axis=1,keepdims=True);assert (ra>1e-8).all() and (rb>1e-8).all()
    return (b*(ra/rb)).astype(np.float32),(a*(rb/ra)).astype(np.float32)
def setup(path):
    global _cache
    q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_T_ONLY_BN_RADIUS_DIAGNOSTIC' and q['host']=='old' and q['seed']==1729
    assert q['arms']==list(ARMS) and q['BN_statistic_updates']==830 and q['optimizer_updates']==0 and q['new_images']==0 and q['maximum_output_bytes']==128<<20
    if _cache is None:
        sys.path.insert(0,str(Path(q['study_policy']['path']).parent));import atm_conditional_stimulus_contract_20261104_v1 as C
        parent=C.setup(q['study_policy']['path']);K=parent['K'];K.checked(q['study_policy']);g=K.read(K.checked(q['gradient_policy']))
        for upstream in (parent['q'],g):
            launch=K.read(Path(upstream['code_root'])/'actual_launch.json');assert not K.process(launch['PID'],launch['start_ticks'])['live']
        assert K.read(K.checked(q['study_completion']))['status']=='COMPLETE_NATIVE_EEG_CONDITIONAL_STIMULUS_TARGET_DIAGNOSTIC'
        assert K.read(K.checked(q['gradient_completion']))['status']=='COMPLETE_BOUNDED_ALL_T_GRADIENT_DIAGNOSIS'
        for pin in q['sources'].values():K.checked(pin)
        _cache=dict(q=q,up=parent,C=C,K=K,g=g,out=Path(q['output_root']),repo=parent['repo'],parent=parent['parent'],T=parent['T'],queries=parent['queries'])
    assert _cache['q']==q;return _cache
def resources(c,starting=False):
    parent=c['C'].resources(c['up'],starting=starting);allocator=c['up']['cv']['ctx']['base']['ctx']['C'].allocated
    g=c['g'];gused=allocator(g['project_root']);assert gused<=g['maximum_output_bytes'];used=allocator(c['q']['project_root']) if Path(c['q']['project_root']).exists() else 0
    assert used<=c['q']['maximum_output_bytes'];required=parent['required_including_this_project']+g['maximum_output_bytes']-gused+c['q']['maximum_output_bytes']-used
    assert parent['free_bytes']>=required
    return dict(free_bytes=parent['free_bytes'],required_including_this_project=required,own_used_bytes=used,own_remaining_bytes=c['q']['maximum_output_bytes']-used,
        inherited_main_and_prior=parent,gradient_remaining_bytes=g['maximum_output_bytes']-gused,all_unreleased_forecasts_retained=True)
