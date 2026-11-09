"""A focused ATMS diagnostic; inherits the completed ENIGMA study's full reserves."""
import importlib.util
import sys
from pathlib import Path


def setup(path):
    import json
    q = json.loads(Path(path).read_bytes())
    assert q['status'] == 'LOCKED_ATMS_POSITIVE_ENCODER_REPEAT_GRADIENT_DIAGNOSTIC'
    assert q['host'] == 'old' and q['seed'] == 1729 and q['maximum_output_bytes'] == 64 << 20
    assert q['arms'] == ['old_direct', 'nce_continued'] and q['new_updates'] == q['new_images'] == 0
    sys.path.insert(0, str(Path(q['prior_policy']['path']).parent))
    import enigma_repeat_task_risk_contract_20261105_v1 as K
    K.checked(q['prior_policy'])
    prior = K.setup(q['prior_policy']['path'])
    done = K.read(K.checked(q['prior_completion']))
    assert done['status'] == 'COMPLETE_T_ONLY_ALL13230_ALL15_ENCODER_OOF_TASK_RISK_AND_CPU_AUDIT'
    launch = K.read(Path(q['prior_policy']['path']).parent/'actual_launch.json')
    assert not K.process(launch['PID'], launch['start_ticks'])['live']
    for pin in q['sources'].values(): K.checked(pin)
    oldq = K.read(K.checked(q['ATM_policy']))
    assert K.read(K.checked(q['ATM_completion']))['status'].startswith('COMPLETE')
    for pin in oldq['model_source_pins']: K.checked(pin)
    for pin in q['checkpoints'].values(): K.checked(pin)
    for key in ('EEG', 'visual_target', 'legacy_inputs'):
        assert oldq[key]['sha256'] == prior['q'][key]['sha256']
    return dict(q=q, K=K, prior=prior, T=prior['T'], out=Path(q['output_root']))


def resources(c, starting=False):
    q, K = c['q'], c['K']
    prior = K.resources(c['prior'], starting=starting)
    used = K.allocated(q['project_root'])
    assert used <= q['maximum_output_bytes']
    required = prior['required_including_this_project'] + q['maximum_output_bytes'] - used
    assert prior['free_bytes'] >= required, (prior['free_bytes'], required)
    return dict(free_bytes=prior['free_bytes'], required_including_this_project=required,
                own_used_bytes=used, maximum_output_bytes=q['maximum_output_bytes'],
                all_prior_forecasts_retained=True, no_deletion_or_reserve_release=True)
