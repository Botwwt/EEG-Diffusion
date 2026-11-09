"""Settle only one completed unused forecast; preserve all actual files and caps."""
import sys
from pathlib import Path


def setup(path):
    import json
    q=json.loads(Path(path).read_bytes())
    assert q['status']=='LOCKED_FULL_ATMS_REPEAT_GRADIENT_AND_FIXED_NEIGHBORHOOD'
    assert q['host']=='old' and q['seed']==1729 and q['maximum_output_bytes']==16<<20
    assert q['new_models']==q['new_updates']==q['new_images']==0
    sys.path.insert(0,str(Path(q['prior_policy']['path']).parent))
    import atm_repeat_gradient_contract_20261105_v1 as P
    prior=P.setup(q['prior_policy']['path']);K=prior['K'];K.checked(q['prior_policy'])
    done=K.read(K.checked(q['prior_completion']));assert done['status']=='COMPLETE_ATMS_POSITIVE_ENCODER_REPEAT_GRADIENT_DIAGNOSTIC'
    launch=K.read(Path(q['prior_policy']['path']).parent/'actual_launch.json')
    assert not K.process(launch['PID'],launch['start_ticks'])['live']
    receipt=K.read(K.checked(done['receipt']))
    assert receipt['status']=='COMPLETED' and not K.process(receipt['pid'])['live'] and not K.process(receipt['wrapper_pid'])['live']
    for pin in q['sources'].values():K.checked(pin)
    assert q['checkpoint']==prior['q']['checkpoints']['nce_continued']
    K.checked(q['checkpoint'])
    assert q['settlement']['project']==prior['q']['project_root']
    assert q['settlement']['original_cap']==prior['q']['maximum_output_bytes']==64<<20
    return dict(q=q,K=K,P=P,prior=prior,T=prior['T'],out=Path(q['output_root']),physical=prior['prior']['q'])


def resources(c,starting=False):
    q,K=c['q'],c['K']
    # The just-closed ATM forecast is settled explicitly below. Evaluate its
    # inherited ENIGMA ledger directly so its unused future is not double charged.
    prior=K.resources(c['prior']['prior'],starting=starting)
    row=q['settlement'];actual=K.allocated(row['project'])
    assert actual==row['allocated'] and row['unused']==row['original_cap']-actual
    # Older forecasts are unchanged except this explicitly closed unused remainder.
    used=K.allocated(q['project_root']);assert used<=q['maximum_output_bytes']
    required=prior['required_including_this_project']+q['maximum_output_bytes']-used
    assert prior['free_bytes']>=required,(prior['free_bytes'],required)
    return dict(free_bytes=prior['free_bytes'],required_including_this_project=required,own_used_bytes=used,
        settled_closed_unused_bytes=row['unused'],actual_files_deleted=0,other_forecasts_unchanged=True)
