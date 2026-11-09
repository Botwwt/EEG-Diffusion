"""Fixed gallery, all registered contrasts, and a joint early-harm rule."""
import argparse, os, sys, time
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import brainhive_functional_mean_contract_20261103_v1 as G

COMPARISONS = {
    'main_vs_ordinary': ('ordinary_shared_mean', 'ordinary'),
    'native_shared_vs_parent': ('native_shared_mean', 'parent'),
    'main_vs_matched_full_mean': ('ordinary_shared_mean', 'ordinary_full_mean_matched'),
    'main_vs_wrong_field': ('ordinary_shared_mean', 'ordinary_wrong_field'),
    'main_vs_risk': ('ordinary_shared_mean', 'risk'),
    'shared_correct_vs_same_concept_donor': ('ordinary_shared_mean', 'ordinary_shared_donor'),
    'ordinary_correct_vs_same_concept_donor': ('ordinary', 'ordinary_donor'),
    'matched_full_vs_ordinary': ('ordinary_full_mean_matched', 'ordinary'),
    'wrong_field_vs_ordinary': ('ordinary_wrong_field', 'ordinary'),
}

def decision(paired, n):
    afc = paired['DINO2AFC']
    within = paired['within_concept10_two_afc']
    joint_harm = (afc['difference'] <= -.01 and
                  afc['descriptive_concept_bootstrap95'][1] < 0 and
                  within['descriptive_concept_bootstrap95'][1] < 0)
    promising = (afc['difference'] >= .01 and
                 afc['descriptive_concept_bootstrap95'][0] > 0 and
                 within['difference'] >= 0)
    action = ('STOP_EXPANSION_JOINT_HARM' if joint_harm else 'EXPAND400') if n == 80 else (
        'PROMISING_FOR_FULL_METRICS_AND_SUBJECTS' if promising else 'NO_CONFIRMED_INCREMENT_ON_THIS_PANEL')
    return dict(action=action, joint_harm=joint_harm, promising_primary_signal=promising,
                descriptive_only=True, optional_stopping_has_no_nominal_error_control=True,
                does_not_reject_entire_family=True, SOTA_established=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-protocol', required=True)
    p.add_argument('--queries', type=int, choices=(80, 400), required=True)
    args = p.parse_args()
    started = time.monotonic()
    spec = G.read(args.image_protocol)
    assert spec['status'] == 'REGISTERED_FUNCTIONAL_MEAN_IMAGES_WITH_PAIRING_CONTROLS'
    for pin in spec['source_pins']:
        G.checked(pin)
    s = G.setup(G.checked(spec['base_protocol']))
    G.resources(s)
    import numpy as np
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as E
    from brainhive_functional_mean_integration_20261103_v1 import ARMS
    assert list(ARMS) == spec['image_arms']
    root = Path(s['output_root'])
    n = args.queries
    destination = root / ('score' + str(n))
    assert not destination.exists()
    if n == 400:
        first = G.read(root / 'score80/completion.json')
        assert first['decision']['action'] == 'EXPAND400'
        assert first['image_protocol'] == G.ref(args.image_protocol)
    d09 = G.read(G.checked(s['future_image_refs']['D09_completion']))
    inputs = G.read(G.checked(d09['inputs']))
    panel = G.read(G.checked(inputs['panel']))['Dcheck']
    gallery = np.load(G.checked(d09['check_target_features']), allow_pickle=False)
    old = np.load(G.checked(d09['check_generated_features']), allow_pickle=False)
    assert len(panel) == 400 and gallery.shape == (400, 768) and old.shape == (400, 3, 768)
    new, bindings = [], []
    for lo in range(0, n, 80):
        part = root / 'generation' / f'{lo:03d}_{lo+80:03d}' / 'completion.json'
        done = G.read(part)
        assert done['status'] == 'COMPLETE_FIXED80_QUERY_SIX_ARM_IMAGE_SEGMENT'
        assert done['range'] == [lo, lo + 80] and done['research_images'] == 480
        assert done['image_protocol'] == G.ref(args.image_protocol) and done['arms'] == list(ARMS)
        feature = np.load(G.checked(done['evidence']['generated_features']), allow_pickle=False)
        assert feature.shape == (80, len(ARMS), 768) and np.isfinite(feature).all()
        new.append(feature)
        bindings.append(G.ref(part))
    new = np.concatenate(new)
    queries = panel[:n]
    vectors = {arm: old[:n, j] for j, arm in enumerate(('parent', 'ordinary', 'risk'))}
    vectors.update({arm: new[:, j] for j, arm in enumerate(ARMS)})
    measures = {arm: E.metric_arrays(value, gallery, queries, panel) for arm, value in vectors.items()}
    self_test = E.metric_arrays(gallery, gallery, panel, panel)
    assert np.allclose(self_test['target_cosine'], 1, atol=2e-5)
    assert np.all(self_test['DINO2AFC'] == 1) and np.all(self_test['gallery_top1'] == 1)
    differences = {name: {metric: measures[a][metric] - measures[b][metric] for metric in E.METRICS}
                   for name, (a, b) in COMPARISONS.items()}
    differences['paired_EEG_effect_difference_in_differences'] = {
        metric: differences['shared_correct_vs_same_concept_donor'][metric] -
                differences['ordinary_correct_vs_same_concept_donor'][metric] for metric in E.METRICS}
    zero = {metric: np.zeros(n) for metric in E.METRICS}
    paired = {name: E.compare(value, zero, queries) for name, value in differences.items()}
    destination.mkdir()
    arrays = {'feature__' + arm: value for arm, value in vectors.items()}
    arrays['gallery_features'] = gallery
    arrays.update({arm + '__' + metric: values for arm, row in measures.items() for metric, values in row.items()})
    arrays.update({'difference__' + name + '__' + metric: values
                   for name, row in differences.items() for metric, values in row.items()})
    np.savez_compressed(destination / 'all_features_metrics_differences.npz', **arrays)
    result = dict(status='COMPLETE_FUNCTIONAL_MEAN_ALL_ARMS_FIXED_GALLERY_CPU_READOUT',
                  UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(), image_protocol=G.ref(args.image_protocol),
                  queries=n, concepts=len({q['concept_id'] for q in queries}), gallery_targets=400,
                  methods=list(vectors), means={arm: {k: float(v.mean()) for k, v in row.items()} for arm, row in measures.items()},
                  paired=paired, comparison_definitions=COMPARISONS,
                  decision=decision(paired['main_vs_ordinary'], n),
                  raw=G.ref(destination / 'all_features_metrics_differences.npz'), segments=bindings,
                  same_fixed_seed_gallery_and_cost=True, ordinary_unchanged=True, NEW_contacted=False,
                  development_panel_already_exposed=True, independent_confirmation=False,
                  new_GPU_jobs=0, new_images=0, seconds=time.monotonic()-started)
    G.resources(s)
    G.write(destination / 'completion.json', result)
    print(G.json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
