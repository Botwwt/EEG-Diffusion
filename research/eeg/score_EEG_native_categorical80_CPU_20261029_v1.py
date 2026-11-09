"""Score all five completed categorical arms after exact source/closure checks."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import csv
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as locked
from AR_anchored_images_score_CPU_20261029_v1 import independent_arrays, independent_contrast

ROOT = Path(__file__).resolve().parent
RECEIVED = ROOT / 'audit/EEG_native_categorical_20261029_v1/received_existing_stage80_v1'
OUT = ROOT / 'audit/EEG_native_categorical_20261029_v1/stage80_results_v1'
PANEL_SHA = '068fc441f1332f1d2b08e79a93a93cacdd498efc35b82010080e7f845e76ae1e'
DINO_SHA = '08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'
METRIC_SHA = 'bc9610b97501cdf078789f913b0520b023c0079801e24c37093cbac406189009'
ARMS = {'old': ['coupled', 'same_donor', 'different_donor'], 'new': ['visual', 'uniform']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def seed(stimulus, kind):
    raw = f'1729|EEG-native-marginal-v2|{kind}|sub-01|{stimulus}|R4'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little') % (2**63 - 1)


def run():
    began = time.monotonic()
    receipt = read(RECEIVED / 'receiving_completion.json')
    assert receipt['status'] == 'ALL5_NATIVE_CATEGORICAL_STAGE80_FITS_ARTIFACTS_FEATURES_EXACT_RECEIVED'
    assert receipt['stage'] == 80 and receipt['NEW_read_only_no_GPU_work']
    # Complete all file and scientific closures before opening target features.
    for item in receipt['files'].values():
        path = Path(item['local']); expected = item['remote']
        assert path.is_file() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
    assert sha(locked.__file__) == METRIC_SHA

    def local(host, filename, expected=None):
        path = RECEIVED / host / filename
        actual = receipt['files'][host + '/' + filename]['remote']
        if expected is not None:
            assert all(actual[k] == expected[k] for k in ('bytes', 'sha256'))
        return path

    panel = read(local('old', 'panel400.json'))
    assert sha(local('old', 'panel400.json')) == PANEL_SHA
    queries = panel['queries']; gallery = panel['target_gallery']
    assert len(queries) == len(gallery) == 400
    assert len({q['query_id'] for q in queries}) == 400
    groups = {}
    for i, q in enumerate(queries):
        assert all(q[k] == gallery[i][k] for k in ('query_id', 'stimulus_id', 'array_row', 'concept_id'))
        groups.setdefault(q['concept_id'], []).append(i)
    keys = sorted(groups)
    assert len(groups) == 40 and {len(v) for v in groups.values()} == {10}
    same = {i: groups[k][(j+1) % 10] for k in keys for j, i in enumerate(groups[k])}
    different = {i: groups[keys[(n+1) % 40]][j] for n, k in enumerate(keys) for j, i in enumerate(groups[k])}
    features = {}; records = {}; fits = {}; provenance = {}; seen_images = set(); table_sha = set()
    for host, arms in ARMS.items():
        fit = read(local(host, 'fit_completion.json')); cpu = read(local(host, 'fit_CPU_audit.json'))
        spec = read(local(host, 'image_spec.json')); image_done = read(local(host, 'image_queue_completion.json'))
        artifact = read(RECEIVED / host / 'read_only_artifact_audit.json')
        assert fit['status'] == 'COMPLETE_FIXED10_NATIVE_CATEGORICAL_FULL_T_FIT'
        assert fit['epochs'] == 10 and fit['updates'] == 2070
        assert cpu['status'] == 'PASS_NATIVE_CATEGORICAL_FULL10_MODEL_ADAM_COVERAGE_AND_RNG_CPU'
        assert cpu['model'] == fit['final_model'] and cpu['query_exposures'] == 132300
        assert image_done['status'] == 'COMPLETE_NATIVE_CATEGORICAL_ALL_OWN_STAGE_IMAGES_FEATURES'
        assert artifact['status'] == 'PASS_READ_ONLY_ALL_EXISTING80_IMAGE_PIXELS_IDENTITIES_SEEDS_EEG_ROWS_AND_FEATURES'
        assert artifact['queries_per_arm'] == 80 and artifact['image_count'] == len(arms)*80
        assert artifact['all4_cached400_logits_bit_identical'] and artifact['arms'] == arms
        assert spec['arms'] == arms and spec['panel']['sha256'] == PANEL_SHA
        assert spec['inference']['guidance'] == spec['inference']['top_k'] == spec['inference']['top_p'] == 0
        assert spec['inference']['best_of_N'] == 1
        local(host, 'fit_completion.json', spec['fit_completion'])
        local(host, 'fit_CPU_audit.json', spec['fit_CPU_audit'])
        logits = np.load(local(host, 'EEG_logits400.npy'), allow_pickle=False)
        assert logits.shape == (400, 1000) and logits.dtype == np.float32 and np.isfinite(logits).all()
        fits[host] = fit
        for arm in arms:
            parts = []; all_rows = []
            for lo in range(0, 80, 20):
                done = read(local(host, f'chunk_{lo:04d}_completion.json'))
                assert done['status'] == 'COMPLETE_NATIVE_CATEGORICAL_FIXED_FINAL_IMAGE_FEATURE_CHUNK'
                assert done['range'] == [lo, lo+20] and done['arms'] == arms
                assert done['DINO_weight']['sha256'] == DINO_SHA
                assert done['image_evaluation']['sha256'] == spec['image_evaluation']['sha256']
                assert done['native_VAR_VQ_unchanged'] and done['normal_input_EEG_only']
                assert done['effect_scores_read'] is False and done['target_images_read'] == 0
                local(host, 'EEG_logits400.npy', done['EEG_logits'])
                x = np.load(local(host, f'{arm}_features_{lo:04d}.npy', done['features'][arm]), allow_pickle=False)
                rows = read(local(host, f'{arm}_records_{lo:04d}.json', done['records'][arm]))
                assert x.shape == (20, 768) and x.dtype == np.float32 and np.isfinite(x).all()
                assert np.max(np.abs(np.linalg.norm(x, axis=1)-1)) < 2e-5 and len(rows) == 20
                for i, row in enumerate(rows, lo):
                    q = queries[i]
                    assert row['arm'] == arm and row['index'] == i
                    assert all(row[k] == q[k] for k in ('query_id', 'stimulus_id', 'array_row'))
                    donor = same[i] if arm == 'same_donor' else different[i] if arm == 'different_donor' else i
                    assert row['EEG_row'] == (None if arm == 'uniform' else donor)
                    source = np.zeros((1, 1000), np.float32) if arm == 'uniform' else logits[donor:donor+1]
                    assert row['EEG_logit_row_sha256'] == hashlib.sha256(source.tobytes()).hexdigest()
                    assert row['class_draw_seed'] == seed(q['stimulus_id'], 'latent')
                    assert row['token_seed'] == seed(q['stimulus_id'], 'token')
                    assert 0 <= row['native_class_id'] < 1000 and row['model_SHA256'] == fit['final_model']['sha256']
                    assert row['generation'] == dict(one_class_draw=True, one_image=True, best_of_N=1, guidance=0., top_k=0, top_p=0., resolution=[256,256], scales=[1,2,3,4,5,6,8,10,13,16])
                    assert row['normal_inference_reads_query_image_label_or_gallery'] is False
                    assert row['labels_used_only_to_construct_experimental_EEG_permutations'] == (arm in ('same_donor','different_donor'))
                    key = (host, row['image']['path']); assert key not in seen_images; seen_images.add(key)
                    assert len(row['pixel_SHA256']) == len(row['image']['sha256']) == 64
                    table_sha.add(row['native_class_table_SHA256'])
                parts.append(x); all_rows.extend(rows)
            features[arm] = np.concatenate(parts); records[arm] = all_rows
        provenance[host] = dict(fit=ref(local(host,'fit_completion.json')), CPU_audit=ref(local(host,'fit_CPU_audit.json')), image_spec=ref(local(host,'image_spec.json')), artifact_audit=ref(RECEIVED/host/'read_only_artifact_audit.json'))
    assert fits['old']['initial_EEG_model_digest'] == fits['new']['initial_EEG_model_digest']
    assert len(table_sha) == 1 and len(seen_images) == 400

    cache = read(local('old', 'target_cache.json'))
    assert cache['panel']['sha256'] == PANEL_SHA and cache['encoder']['weight_sha256'] == DINO_SHA
    targets = np.load(local('old', 'targets400.npy', cache['features']), allow_pickle=False)
    assert targets.shape == (400,768)
    gaussian_root = ROOT/'audit/EEG_native_marginal_images_20261029_v2/received_stage80_v1'
    gaussian_receipt = read(gaussian_root/'receiving_completion.json')
    gaussian_path = gaussian_root/'old/coupled_features.npy'
    matching = [v for v in gaussian_receipt['files'].values() if Path(v['local']).resolve() == gaussian_path.resolve()]
    assert len(matching) == 1 and sha(gaussian_path) == matching[0]['remote']['sha256']
    assert sha(gaussian_root/'old/panel400.json') == PANEL_SHA
    gaussian_rows = read(gaussian_root/'old/coupled_records.json')
    assert [x['query_id'] for x in gaussian_rows] == [q['query_id'] for q in queries[:80]]
    features['frozen_Gaussian'] = np.load(gaussian_path,allow_pickle=False)
    assert features['frozen_Gaussian'].shape == (80,768)
    self_metrics = locked.metric_arrays(targets,targets,queries,gallery)
    assert np.all(self_metrics['DINO2AFC'] == 1) and np.all(self_metrics['gallery_top1'] == 1)
    arrays = {a: locked.metric_arrays(x,targets,queries[:80],gallery) for a,x in features.items()}
    independently = {a: independent_arrays(x,targets,queries[:80],gallery) for a,x in features.items()}
    max_array_error = max(float(np.max(np.abs(row[m]-independently[a][m]))) for a,row in arrays.items() for m in locked.METRICS)
    assert max_array_error <= 2e-6
    pairs = {f'coupled_minus_{b}': ('coupled',b) for b in ('visual','uniform','same_donor','different_donor','frozen_Gaussian')}
    pairs['visual_minus_uniform'] = ('visual','uniform')
    contrasts = {key: locked.compare(arrays[a],arrays[b],queries[:80]) for key,(a,b) in pairs.items()}
    independent = {key: independent_contrast(independently[a],independently[b],queries[:80]) for key,(a,b) in pairs.items()}
    max_interval_error = max(abs(x-y) for key,row in contrasts.items() for m,entry in row.items() for x,y in zip(entry['descriptive_concept_bootstrap95'],independent[key][m]['descriptive_concept_bootstrap95']))
    assert max_interval_error <= 2e-6
    main = contrasts['coupled_minus_visual']['DINO2AFC']
    decision = 'STOP_INSTANCE' if main['difference'] <= -.02 and main['descriptive_concept_bootstrap95'][1] < 0 else 'EXPAND400'
    OUT.mkdir(exist_ok=False)
    np.savez_compressed(OUT/'all6_features.npz',**features)
    np.savez_compressed(OUT/'all_metrics.npz',**{a+'__'+m:x for a,row in arrays.items() for m,x in row.items()})
    with (OUT/'per_query.csv').open('x',encoding='utf8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['arm','query_index','query_id','stimulus_id','concept_id',*locked.METRICS]);writer.writeheader()
        for a,row in arrays.items():
            for i,q in enumerate(queries[:80]): writer.writerow(dict(arm=a,query_index=i,query_id=q['query_id'],stimulus_id=q['stimulus_id'],concept_id=q['concept_id'],**{m:float(x[i]) for m,x in row.items()}))
    write(OUT/'independent_CPU_rescore.json',dict(status='PASS_SEPARATE_RANK_AND_WHOLE_CONCEPT_BOOTSTRAP',metric_cells=6*80*5,max_array_error=max_array_error,max_interval_error=max_interval_error,feature_encoder_independently_rerun=False))
    summary = dict(status='COMPLETE_ALL5_CATEGORICAL_FIXED80_IMAGE_SCREEN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=80,gallery_targets=400,concepts=8,decision=decision,
        means={a:{m:float(x.mean()) for m,x in row.items()} for a,row in arrays.items()},contrasts=contrasts,
        provenance=provenance,receiving_completion=ref(RECEIVED/'receiving_completion.json'),source=ref(__file__),locked_metric_source_sha256=METRIC_SHA,
        all5_fit_and_artifact_closures_checked_before_effect_read=True,NEW_existing_results_read_only=True,future_GPU_host='old',
        panel=ref(local('old','panel400.json')),target_features=ref(local('old','targets400.npy')),frozen_Gaussian_features=ref(gaussian_path),
        inference_comparability='All5 categorical arms native256/10scales/oneclass/oneimage/top0. FrozenGaussian top600: comparison is a combined-recipe comparison, not isolated support.',
        stopping_rule='Only coupled-minus-visual DINO mean<=-0.02 AND concept-bootstrap upper<0 stops80; otherwise400.',
        positive400_rule=spec['evaluation']['positive400'],independent_confirmation=False,SOTA_established=False,seconds=time.monotonic()-began)
    write(OUT/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],decision=decision,means=summary['means'],DINO_contrasts={k:v['DINO2AFC'] for k,v in contrasts.items()})))


if __name__ == '__main__':
    run()
