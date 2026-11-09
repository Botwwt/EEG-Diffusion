"""CPU-only paired screening on one frozen 400-target gallery.

80 and 400 refer to query stages, never to different target galleries.
Caches contain frozen features, not models; no torch/CUDA import or image fit.
Intervals and optional stage expansion are exploratory, not error-controlled
confirmation. All registered rows remain in every denominator.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

SCHEMA = 'BrainHIVE_rapid_cached_DINO_CPU_20261027_v1'
CACHE_SCHEMA = 'BrainHIVE_rapid_DINO_feature_cache_20261027_v1'
PANEL_SCHEMA = 'BrainHIVE_rapid_panel400_20261027_v1'
METRICS = ('target_cosine', 'DINO2AFC', 'gallery_top1',
           'within_concept10_top1', 'within_concept10_two_afc')
IDENTITY = ('query_id', 'subject_id', 'concept_id', 'stimulus_id',
            'array_row', 'fold', 'image_seed')
GENERATION = dict(NFE=4, resolution=[512, 512], CFG=0, IP_scale=1,
                  one_image_per_query=True, no_best_of_N=True)
PRACTICAL_GAIN = .01
LARGE_HARM = -.02
DRAW_COUNT = 1000
SEED = 1729
DINO_SHA256 = '08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked_ref(value):
    assert set(value) >= {'path', 'sha256'}
    path = Path(value['path'])
    assert path.is_file() and sha(path) == value['sha256'], f'Cache SHA mismatch: {path}'
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def load_panel(path):
    panel = read(path)
    assert panel['schema'] == PANEL_SCHEMA
    assert panel['status'] == 'FIXED_SINGLE_SUBJECT_T400_WITH_REUSED_T80_PREFIX'
    queries = panel['queries']; gallery = panel['target_gallery']
    assert panel['expected_queries'] == len(queries) == len(gallery) == 400
    assert len({q['query_id'] for q in queries}) == 400
    assert len({q['stimulus_id'] for q in queries}) == 400
    assert len({q['array_row'] for q in queries}) == 400
    assert len({q['subject_id'] for q in queries}) == 1
    grouped = {}
    for q, target in zip(queries, gallery):
        for key in IDENTITY:
            assert q[key] == target[key], f'Panel/gallery identity mismatch: {key}'
        grouped.setdefault(q['concept_id'], []).append(q)
    assert [q['target_index'] for q in gallery] == list(range(400))
    assert all(len(group) == 10 and len({q['fold'] for q in group}) == 1 for group in grouped.values())
    assert len(grouped) * 10 == 400
    for stage, count in (('80', 80), ('400', 400)):
        item = panel['stages'][stage]
        assert item['query_count'] == count and item['target_gallery_query_count'] == 400
        assert item['query_ids'] == [q['query_id'] for q in queries[:count]]
        counts = {}
        for q in queries[:count]: counts[q['concept_id']] = counts.get(q['concept_id'], 0) + 1
        assert all(n == 10 for n in counts.values()), 'Stage must retain entire concepts'
    old = read(checked_ref(panel['old_panel']))
    assert old['queries'] == queries[:80], 'The exposed original 80-query panel changed'
    assert panel['results_or_predictions_read'] is False
    return panel


def load_cache(path, panel_path, panel, expected_kind, require_full=False):
    document = read(path)
    assert document['schema'] == CACHE_SCHEMA and document['status'] == 'COMPLETE_FROZEN_DINO_FEATURE_CACHE'
    assert document['kind'] == expected_kind
    assert document['panel']['sha256'] == sha(panel_path)
    checked_ref(document['panel'])
    encoder = document['encoder']
    assert set(encoder) >= {'weight_sha256', 'feature_recipe', 'feature_dimensions',
                            'real_preprocess', 'generated_preprocess'}
    assert encoder['weight_sha256'] == DINO_SHA256 and 0 < encoder['feature_dimensions'] <= 16384
    assert all(encoder[k] for k in ('feature_recipe', 'real_preprocess', 'generated_preprocess'))
    feature_path = checked_ref(document['features']); row_path = checked_ref(document['rows'])
    source_path = checked_ref(document['image_source_manifest'])
    source_rows = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    source = {}
    for row in source_rows:
        if expected_kind == 'generated' and row.get('arm') != document['arm']: continue
        key = row['query_id']; assert key not in source, 'Duplicate source manifest query'
        source[key] = row
    records = read(row_path)
    assert isinstance(records, list)
    features = np.load(feature_path, allow_pickle=False)
    assert features.dtype == np.float32 and features.ndim == 2
    assert document['dtype'] == 'float32' and document['shape'] == list(features.shape)
    assert features.shape == (len(records), encoder['feature_dimensions'])
    assert np.isfinite(features).all() and np.all(np.linalg.norm(features, axis=1) > 0)
    keyed = {r['query_id']: i for i, r in enumerate(records)}
    assert len(keyed) == len(records), 'Duplicate cache query'
    all_ids = {q['query_id'] for q in panel['queries']}
    stage_ids = set(panel['stages']['80']['query_ids'])
    assert set(keyed) == all_ids or (not require_full and set(keyed) == stage_ids), 'Missing/extra cache rows'
    assert set(source) == set(keyed), 'Feature cache/source manifest coverage differs'
    expected = {q['query_id']: q for q in panel['queries']}
    targets = {q['query_id']: q for q in panel['target_gallery']}
    for item in records:
        key = item['query_id']; query = expected[key]; original = source[key]
        for field in IDENTITY:
            assert item[field] == query[field] == original[field], f'Cache identity mismatch: {field}'
        image_sha = original.get('image_sha256', original.get('generated_sha256'))
        assert item['image_sha256'] == image_sha and len(image_sha) == 64
        if expected_kind == 'targets':
            assert item['image_sha256'] == targets[key]['image_sha256']
        else:
            assert document['generation'] == GENERATION
            assert item['target_image_sha256'] == original['target_image_sha256'] == targets[key]['image_sha256']
            assert item['initial_noise_latents_sha256'] == original['initial_noise_latents_sha256']
            assert len(item['initial_noise_latents_sha256']) == 64
    return dict(document=document, features=features, rows=records, positions=keyed,
                bindings=[ref(path), ref(feature_path), ref(row_path), ref(source_path)],
                input_ref=ref(path))


def ordered(cache, queries):
    positions = [cache['positions'][q['query_id']] for q in queries]
    return cache['features'][positions], [cache['rows'][i] for i in positions]


def metric_arrays(query_features, gallery_features, queries, gallery):
    """Fixed-gallery tie rule: half credit for 2AFC, first gallery row for Top1."""
    assert len(query_features) == len(queries) and len(gallery_features) == len(gallery) > 1
    q = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    g = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
    scores = q @ g.T
    positions = {row['stimulus_id']: i for i, row in enumerate(gallery)}
    assert len(positions) == len(gallery)
    positive = np.array([positions[row['stimulus_id']] for row in queries])
    correct = scores[np.arange(len(queries)), positive]
    afc = ((scores < correct[:, None]).sum(1) +
           .5 * ((scores == correct[:, None]).sum(1) - 1)) / (len(gallery) - 1)
    result = dict(target_cosine=correct.astype(np.float64), DINO2AFC=afc,
                  gallery_top1=(scores.argmax(1) == positive).astype(np.float64))
    within1 = np.empty(len(queries)); within2 = np.empty(len(queries))
    for i, query in enumerate(queries):
        subset = [j for j, target in enumerate(gallery) if target['concept_id'] == query['concept_id']]
        assert len(subset) == 10 and positive[i] in subset
        local = scores[i, subset]
        within1[i] = subset[int(local.argmax())] == positive[i]
        within2[i] = ((local < correct[i]).sum() + .5 * ((local == correct[i]).sum() - 1)) / (len(subset) - 1)
    result.update(within_concept10_top1=within1, within_concept10_two_afc=within2)
    assert set(result) == set(METRICS) and all(np.isfinite(v).all() for v in result.values())
    return result


def compare(candidate, baseline, queries):
    concepts = sorted({q['concept_id'] for q in queries})
    groups = [[i for i, q in enumerate(queries) if q['concept_id'] == c] for c in concepts]
    differences = {key: candidate[key] - baseline[key] for key in METRICS}
    per_concept = np.array([[differences[key][group].mean() for key in METRICS] for group in groups])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(groups), size=(DRAW_COUNT, len(groups)))
    samples = per_concept[draws].mean(1)
    intervals = np.quantile(samples, [.025, .975], axis=0)
    return {key: dict(difference=float(differences[key].mean()),
                     descriptive_concept_bootstrap95=intervals[:, i].tolist(),
                     positive_concepts=int((per_concept[:, i] > 0).sum()), concepts=len(concepts))
            for i, key in enumerate(METRICS)}


def decision(paired, stage):
    afc = paired['DINO2AFC']; mean = afc['difference']
    low, high = afc['descriptive_concept_bootstrap95']
    positive = (mean >= PRACTICAL_GAIN and paired['target_cosine']['difference'] > 0 and
                paired['within_concept10_top1']['difference'] >= 0)
    if mean <= LARGE_HARM and high < 0:
        action, reason = 'STOP_INSTANCE', 'LARGE_HARM'
    elif high < PRACTICAL_GAIN:
        action, reason = 'STOP_INSTANCE', 'DESCRIPTIVE_PRACTICAL_FUTILITY'
    elif stage == '80':
        action, reason = 'EXPAND400', ('PROMISING_PRACTICAL_SIGNAL' if positive else 'UNCERTAIN_OR_METRIC_CONFLICT')
    elif positive and low > 0:
        action, reason = 'PROMISING_FOR_FULL_EVALUATION', 'PRACTICAL_SIGNAL_ON_EXPANDED_DEVELOPMENT_PANEL'
    else:
        action, reason = 'STOP_INSTANCE', 'EXPANDED_SCREEN_INCONCLUSIVE_OR_NOT_PROMISING'
    return dict(action=action, reason=reason, practical_DINO2AFC_gain=PRACTICAL_GAIN,
                large_harm_mean_threshold=LARGE_HARM, descriptive_only=True,
                optional_stopping_has_no_nominal_error_control=True, SOTA_established=False)


def fingerprint(cache, queries):
    values, records = ordered(cache, queries)
    return dict(features_sha256=hashlib.sha256(values.tobytes(order='C')).hexdigest(),
                rows_sha256=digest(records))


def score(panel_path, target_path, native_path, candidate_paths, stage, output, prior_path=None):
    started = time.monotonic(); output = Path(output)
    assert stage in ('80', '400') and not output.exists(), 'Preserve every scoring attempt'
    panel = load_panel(panel_path); all_queries = panel['queries']
    queries = all_queries[:int(stage)]
    targets = load_cache(target_path, panel_path, panel, 'targets', require_full=True)
    native = load_cache(native_path, panel_path, panel, 'generated', require_full=True)
    assert targets['document']['arm'] == 'target_gallery' and native['document']['arm'] == 'native_parent'
    gallery_features, _ = ordered(targets, all_queries)
    baseline_features, baseline_rows = ordered(native, queries)
    baseline = metric_arrays(baseline_features, gallery_features, queries, panel['target_gallery'])
    self_check = metric_arrays(gallery_features, gallery_features, all_queries, panel['target_gallery'])
    assert np.allclose(self_check['target_cosine'], 1, atol=2e-5)
    assert np.all(self_check['DINO2AFC'] == 1) and np.all(self_check['gallery_top1'] == 1), 'Duplicate/colliding target features'
    controls = [ref(panel_path), *targets['bindings'], *native['bindings']]
    candidates = {}; results = {}; table = []
    prior = read(prior_path) if prior_path else None
    if stage == '400':
        assert prior and prior['schema'] == SCHEMA and prior['stage'] == '80'
        assert prior['status'] == 'COMPLETE_CPU_FIXED400_GALLERY_PAIRED_SCREEN'
        assert prior['queries'] == 80 and prior['gallery_targets'] == 400
        assert prior['source']['sha256'] == sha(__file__), 'Stage2 scoring/decision implementation changed'
        for name, path in (('panel', panel_path), ('target_cache', target_path), ('native_cache', native_path)):
            assert prior[name]['sha256'] == sha(path), 'Stage2 changed a frozen panel/gallery/baseline cache'
        checked_ref(prior['per_query']); controls.append(ref(prior_path))
    else:
        assert prior_path is None
    for path in candidate_paths:
        cache = load_cache(path, panel_path, panel, 'generated', require_full=stage == '400')
        arm = cache['document']['arm']; assert arm not in candidates and arm not in ('native_parent', 'target_gallery')
        assert cache['document']['encoder'] == native['document']['encoder'] == targets['document']['encoder']
        values, records = ordered(cache, queries)
        for candidate_row, baseline_row in zip(records, baseline_rows):
            assert candidate_row['initial_noise_latents_sha256'] == baseline_row['initial_noise_latents_sha256'], 'Unpaired query noise'
        prefix = fingerprint(cache, all_queries[:80])
        if stage == '400':
            assert arm in prior['candidates'] and prior['candidates'][arm]['decision']['action'] == 'EXPAND400'
            assert prefix == prior['candidates'][arm]['stage1_fingerprint'], 'Stage2 replaced an already scored first80 image/feature'
        metrics = metric_arrays(values, gallery_features, queries, panel['target_gallery'])
        paired = compare(metrics, baseline, queries)
        candidates[arm] = dict(cache=ref(path), means={k: float(v.mean()) for k, v in metrics.items()},
                               paired_candidate_minus_native=paired, decision=decision(paired, stage), stage1_fingerprint=prefix)
        results[arm] = metrics; controls.extend(cache['bindings'])
    assert candidates
    if stage == '400':
        assert set(candidates) == {arm for arm, value in prior['candidates'].items() if value['decision']['action'] == 'EXPAND400'}
    for i, query in enumerate(queries):
        for arm, metrics in [('native_parent', baseline), *results.items()]:
            table.append({**{k: query[k] for k in IDENTITY}, 'arm': arm,
                          **{key: float(values[i]) for key, values in metrics.items()},
                          **{key + '_minus_native': float(metrics[key][i] - baseline[key][i]) for key in METRICS}})
    for item in controls: checked_ref(item)
    output.mkdir(parents=True)
    per_query = output/'per_query.csv'
    with per_query.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    summary = dict(schema=SCHEMA, status='COMPLETE_CPU_FIXED400_GALLERY_PAIRED_SCREEN', stage=stage,
        queries=len(queries), concepts=len({q['concept_id'] for q in queries}), gallery_targets=len(all_queries),
        rows=len(table), full_registered_stage_denominator=True, hidden_exclusions=0,
        panel=ref(panel_path), target_cache=ref(target_path), native_cache=ref(native_path),
        stage1_summary=ref(prior_path) if prior_path else None, native_means={k: float(v.mean()) for k, v in baseline.items()},
        candidates=candidates, per_query=ref(per_query), input_bindings=controls,
        source=ref(__file__), target_selfcheck={k: float(v.mean()) for k, v in self_check.items()},
        bootstrap=dict(seed=SEED, draws=DRAW_COUNT, unit='paired concepts, all ten queries retained',
                       gallery='one unchanged 400-target gallery; never resampled', descriptive_only=True),
        tie_rules=dict(two_afc='half credit', top1='first row in frozen target gallery'),
        baseline_features_reused=True, new_GPU_calls=0, CPU_only=True,
        prior_80_gallery_means_combined=False, stage2_recomputes_all400=True,
        panel_already_exposed_development=True, independent_confirmation=False,
        optional_stopping_has_no_nominal_error_control=True, SOTA_established=False, seconds=time.monotonic()-started)
    atomic(output/'summary.json', summary)
    return summary


def fixture(root):
    """400 orthogonal targets: all pairwise answers are known analytically."""
    root = Path(root); rows = []
    for i in range(400):
        rows.append(dict(query_id=f'q{i:03}', subject_id='sub-01', concept_id=f'c{i//10:02}',
                         stimulus_id=f's{i:03}', array_row=i, fold=(i//10)%2, image_seed=1729+i))
    old = root/'old_panel.json'; atomic(old, dict(queries=rows[:80]))
    gallery = [q | dict(target_index=i, image_sha256=digest(['target', i])) for i, q in enumerate(rows)]
    panel = root/'panel.json'; atomic(panel, dict(schema=PANEL_SCHEMA, status='FIXED_SINGLE_SUBJECT_T400_WITH_REUSED_T80_PREFIX',
        expected_queries=400, queries=rows, target_gallery=gallery, old_panel=ref(old), results_or_predictions_read=False,
        stages={str(n): dict(query_count=n, query_ids=[q['query_id'] for q in rows[:n]], target_gallery_query_count=400) for n in (80,400)}))
    encoder = dict(weight_sha256=DINO_SHA256, feature_recipe='fixture orthogonal identity', feature_dimensions=400,
                   real_preprocess='fixture', generated_preprocess='fixture')
    def cache(arm, count, values, target=False):
        records = []
        for i, q in enumerate(rows[:count]):
            item = q | dict(image_sha256=gallery[i]['image_sha256'] if target else digest([arm,i]))
            if not target: item.update(arm=arm, target_image_sha256=gallery[i]['image_sha256'], initial_noise_latents_sha256=digest(['noise',i]))
            records.append(item)
        npy = root/f'{arm}{count}.npy'; np.save(npy, values.astype(np.float32), allow_pickle=False)
        identities = root/f'{arm}{count}_rows.json'; atomic(identities, records)
        source = root/f'{arm}{count}_source.jsonl'; source.write_text(''.join(json.dumps(q)+'\n' for q in records), encoding='utf-8')
        manifest = root/f'{arm}{count}_cache.json'; atomic(manifest, dict(schema=CACHE_SCHEMA,status='COMPLETE_FROZEN_DINO_FEATURE_CACHE',
            kind='targets' if target else 'generated',arm=arm,panel=ref(panel),encoder=encoder,features=ref(npy),rows=ref(identities),
            shape=list(values.shape),dtype='float32',image_source_manifest=ref(source),generation=None if target else GENERATION))
        return manifest
    eye = np.eye(400, dtype=np.float32)
    wrong = [i//10*10+(i+1)%10 for i in range(400)]
    target = cache('target_gallery',400,eye,True)
    native = cache('native_parent',400,eye[wrong])
    first = cache('candidate',80,eye[:80]); full = cache('candidate',400,eye)
    return panel,target,native,first,full,rows


def self_test():
    count = 0
    def rejects(call):
        try: call()
        except (AssertionError, KeyError, ValueError): return
        raise AssertionError('Invalid fixture was accepted')
    with tempfile.TemporaryDirectory(prefix='brainhive_cpu_rapid_') as tmp:
        root=Path(tmp); panel,target,native,first,full,queries=fixture(root)
        one=score(panel,target,native,[first],'80',root/'stage1');count+=1
        assert one['concepts'] == len({q['concept_id'] for q in queries[:80]})
        assert one['gallery_targets']==400 and one['candidates']['candidate']['means']['DINO2AFC']==1
        assert np.isclose(one['native_means']['DINO2AFC'],199/399) and one['native_means']['within_concept10_top1']==0;count+=1
        assert one['candidates']['candidate']['decision']['action']=='EXPAND400';count+=1
        two=score(panel,target,native,[full],'400',root/'stage2',root/'stage1'/'summary.json')
        assert two['queries']==400 and two['concepts']==40 and two['rows']==800
        assert two['candidates']['candidate']['decision']['action']=='PROMISING_FOR_FULL_EVALUATION';count+=1
        assert np.allclose(two['candidates']['candidate']['paired_candidate_minus_native']['DINO2AFC']['descriptive_concept_bootstrap95'],[1-199/399]*2,atol=1e-15,rtol=0);count+=1
        bad=read(first);bad['features']['sha256']='0'*64;atomic(root/'badsha.json',bad)
        rejects(lambda:load_cache(root/'badsha.json',panel,read(panel),'generated'));count+=1
        bad=read(first);bad['schema']='other';atomic(root/'bad_schema.json',bad)
        rejects(lambda:load_cache(root/'bad_schema.json',panel,read(panel),'generated'));count+=1
        bad=read(first); rec=read(bad['rows']['path']);rec[0]['stimulus_id']='wrong'
        atomic(root/'bad_rows.json',rec);bad['rows']=ref(root/'bad_rows.json');atomic(root/'bad_align.json',bad)
        rejects(lambda:load_cache(root/'bad_align.json',panel,read(panel),'generated'));count+=1
        bad=read(first); rec=read(bad['rows']['path']);rec=rec[:-1]
        atomic(root/'missing_rows.json',rec);bad['rows']=ref(root/'missing_rows.json');atomic(root/'missing.json',bad)
        rejects(lambda:load_cache(root/'missing.json',panel,read(panel),'generated'));count+=1
        rejects(lambda:score(panel,target,native,[full],'400',root/'no_stage1'));count+=1
        rejects(lambda:score(panel,target,native,[first],'400',root/'incomplete_stage2',root/'stage1'/'summary.json'));count+=1
        def synthetic(mean,low,high,cos=.001,top=0):
            return {'DINO2AFC':{'difference':mean,'descriptive_concept_bootstrap95':[low,high]},
                    'target_cosine':{'difference':cos},'within_concept10_top1':{'difference':top}}
        assert decision(synthetic(.001,-.002,.009),'80')['reason']=='DESCRIPTIVE_PRACTICAL_FUTILITY'
        assert decision(synthetic(.001,-.002,.01),'80')['action']=='EXPAND400'
        assert decision(synthetic(-.03,-.06,-.001),'80')['reason']=='LARGE_HARM'
        assert decision(synthetic(.02,.015,.025,cos=-.01),'80')['action']=='EXPAND400';count+=1
        bad=read(first);bad['image_source_manifest']['sha256']='0'*64;atomic(root/'bad_source.json',bad)
        rejects(lambda:load_cache(root/'bad_source.json',panel,read(panel),'generated'));count+=1
        bad=read(first);bad['encoder']['weight_sha256']='0'*64;atomic(root/'bad_encoder.json',bad)
        rejects(lambda:load_cache(root/'bad_encoder.json',panel,read(panel),'generated'));count+=1
        bad=read(first);rec=read(bad['rows']['path']);rec[0]['initial_noise_latents_sha256']='0'*64
        atomic(root/'wrong_noise_rows.json',rec);bad['rows']=ref(root/'wrong_noise_rows.json')
        noise_source=root/'wrong_noise_source.jsonl';noise_source.write_text(''.join(json.dumps(q)+'\n' for q in rec),encoding='utf-8')
        bad['image_source_manifest']=ref(noise_source);atomic(root/'wrong_noise.json',bad)
        rejects(lambda:score(panel,target,native,[root/'wrong_noise.json'],'80',root/'noise_mismatch'));count+=1
        bad=read(first);values=np.load(bad['features']['path'],allow_pickle=False);values[0,0]=np.nan
        np.save(root/'nan.npy',values,allow_pickle=False);bad['features']=ref(root/'nan.npy');atomic(root/'nan_cache.json',bad)
        rejects(lambda:load_cache(root/'nan_cache.json',panel,read(panel),'generated'));count+=1
        changed=read(full);value=np.load(changed['features']['path'],allow_pickle=False);value[0]=value[1]
        np.save(root/'changed.npy',value,allow_pickle=False);changed['features']=ref(root/'changed.npy');atomic(root/'changed_cache.json',changed)
        rejects(lambda:score(panel,target,native,[root/'changed_cache.json'],'400',root/'replaced_prefix',root/'stage1'/'summary.json'));count+=1
        features=np.ones((400,400),dtype=np.float32)
        constant=metric_arrays(features,np.eye(400,dtype=np.float32),queries,read(panel)['target_gallery'])
        assert np.all(constant['DINO2AFC']==.5) and constant['within_concept10_top1'].mean()==.1;count+=1
    return dict(status='CPU_RAPID_FIXED400_CACHE_MATH_CONTRACT_AND_STAGE_CHECKS_PASSED',tests=count,
                query_rows_tested=400,concepts_derived_from_rows=True,CUDA_imported=False,new_GPU_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('panel','target-cache','native-cache','output','stage1-summary'):
        parser.add_argument('--'+key,type=Path)
    parser.add_argument('--candidate-caches',nargs='+',type=Path)
    parser.add_argument('--stage',choices=('80','400'))
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test: print(json.dumps(self_test(),allow_nan=False));return 0
    assert all((args.panel,args.target_cache,args.native_cache,args.output,args.candidate_caches,args.stage))
    result=score(args.panel,args.target_cache,args.native_cache,args.candidate_caches,args.stage,args.output,args.stage1_summary)
    print(json.dumps(dict(status=result['status'],stage=args.stage,seconds=result['seconds'],
                          decisions={arm:value['decision'] for arm,value in result['candidates'].items()}),allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
