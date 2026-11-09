"""Meaningful tie/identity/statistical/decision checks before any new image."""
import argparse, importlib.util, sys
from pathlib import Path

def checks(repository):
    import numpy as np
    sys.path.insert(0, str(repository))
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as E
    spec = importlib.util.spec_from_file_location('functional_score_tests', Path(__file__).with_name('score_HIVE_functional_mean_CPU_20261103_v1.py'))
    score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)
    panel = [dict(stimulus_id=str(i), concept_id=str(i//10)) for i in range(20)]
    eye = np.eye(20, dtype=np.float32)
    exact = E.metric_arrays(eye, eye, panel, panel)
    assert all(np.all(v == 1) for v in exact.values())
    # Every gallery item tied: half-credit for AFC; deterministic first index for Top1.
    tied = np.ones((20, 2), dtype=np.float32)
    tie = E.metric_arrays(tied, tied, panel, panel)
    assert np.all(tie['DINO2AFC'] == .5) and np.all(tie['within_concept10_two_afc'] == .5)
    assert tie['gallery_top1'].sum() == 1 and tie['within_concept10_top1'].sum() == 2
    reversed_values = E.metric_arrays(eye[::-1], eye, panel, panel)
    assert reversed_values['gallery_top1'].sum() == 0
    assert np.all(reversed_values['DINO2AFC'] == 9/19)
    paired = E.compare(exact, exact, panel)
    assert all(v['difference'] == 0 and v['descriptive_concept_bootstrap95'] == [0, 0] for v in paired.values())
    def candidate(global_mean, global_high, within_mean, within_high):
        return dict(DINO2AFC=dict(difference=global_mean, descriptive_concept_bootstrap95=[global_mean-.01, global_high]),
                    within_concept10_two_afc=dict(difference=within_mean, descriptive_concept_bootstrap95=[within_mean-.01, within_high]))
    assert score.decision(candidate(-.02, -.01, -.01, .01), 80)['action'] == 'EXPAND400'
    assert score.decision(candidate(-.02, -.01, -.02, -.01), 80)['action'] == 'STOP_EXPANSION_JOINT_HARM'
    assert score.decision(candidate(.02, .03, .0, .01), 400)['action'] == 'PROMISING_FOR_FULL_METRICS_AND_SUBJECTS'
    assert score.decision(candidate(.02, .03, -.01, .0), 400)['action'] == 'NO_CONFIRMED_INCREMENT_ON_THIS_PANEL'
    return dict(status='PASS_TIES_IDENTITIES_ZERO_PAIRED_STATS_AND_JOINT_DECISION_CHECKS', checks=9,
                any_one_negative_metric_does_not_stop_expansion=True, real_image_effects_measured=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repository', required=True)
    p.add_argument('--image-protocol')
    a = p.parse_args()
    result = checks(Path(a.repository))
    if a.image_protocol:
        import brainhive_functional_mean_contract_20261103_v1 as G
        s = G.read(a.image_protocol)
        assert s['status'] == 'REGISTERED_FUNCTIONAL_MEAN_IMAGES_WITH_PAIRING_CONTROLS'
        assert s['maximum_controller_seconds'] == 7200 and s['image_arms'] == [
            'native_shared_mean', 'ordinary_shared_mean', 'ordinary_full_mean_matched',
            'ordinary_wrong_field', 'ordinary_shared_donor', 'ordinary_donor']
        for pin in s['source_pins']: G.checked(pin)
        base = G.setup(G.checked(s['base_protocol']))
        done = G.read(G.checked(s['statistics_completion']))
        audit = G.read(G.checked(s['statistics_CPU_audit']))
        assert done['protocol'] == s['base_protocol'] and done['fields'] == audit['source_files']['fields']
        assert s['policy'] == base['future_image_policy']
        assert not any((Path(base['output_root'])/name).exists() for name in ('profile', 'generation', 'score80', 'score400', 'image_queue'))
        result['resources'] = G.resources(base, start=True)
        result['image_protocol'] = G.ref(a.image_protocol)
    import json
    print(json.dumps(result))

if __name__ == '__main__':
    main()
