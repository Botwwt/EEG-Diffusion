"""New 8MiB diagnostic above every remaining forecast; no further settlement."""
import sys
from pathlib import Path
def setup(path):
    import json
    q=json.loads(Path(path).read_bytes())
    assert q['status']=='LOCKED_NATIVE_GLOBAL_NCE_AND_ACTUAL_ADAM_REPEAT_DIAGNOSIS'
    assert q['host']=='old' and q['seed']==1729 and q['maximum_output_bytes']==8<<20
    assert q['new_models']==q['new_updates']==q['new_images']==0
    sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
    import atm_full_repeat_gradient_contract_20261105_v1 as P
    prior=P.setup(q['prior_policy']['path']);K=prior['K'];K.checked(q['prior_policy'])
    done=K.read(K.checked(q['prior_completion']));assert done['status']=='COMPLETE_ATMS_WHOLE_ENCODER_REPEAT_GRADIENT_AND_FIXED_NEIGHBORHOOD'
    launch=K.read(Path(q['prior_policy']['path']).parent/'actual_launch.json');assert not K.process(launch['PID'],launch['start_ticks'])['live']
    receipt=K.read(K.checked(done['receipt']));assert receipt['status']=='COMPLETED'
    assert not K.process(receipt['pid'])['live'] and not K.process(receipt['wrapper_pid'])['live']
    for pin in q['sources'].values():K.checked(pin)
    objective=K.checked(q['native_objective']);sys.path.insert(0,str(objective.parent))
    assert q['checkpoint']==prior['q']['checkpoint'];K.checked(q['checkpoint'])
    return dict(q=q,K=K,P=P,prior=prior,T=prior['T'],out=Path(q['output_root']),physical=prior['physical'])
def resources(c,starting=False):
    q,K=c['q'],c['K'];old=c['P'].resources(c['prior'],starting=starting)
    used=K.allocated(q['project_root']);assert used<=q['maximum_output_bytes']
    required=old['required_including_this_project']+q['maximum_output_bytes']-used
    assert old['free_bytes']>=required,(old['free_bytes'],required)
    return dict(free_bytes=old['free_bytes'],required_including_this_project=required,own_used_bytes=used,
        no_new_settlement_or_deletion=True,all_prior_forecasts_retained=True)
