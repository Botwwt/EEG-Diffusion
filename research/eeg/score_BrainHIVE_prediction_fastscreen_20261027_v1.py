"""Generic locked-DINO score for all arms of the exposed development T80 panel.

Every registered arm contributes all 80 queries; no arm, concept or image is
dropped.  All comparisons against the native parent and component-vs-posterior
are descriptive eight-concept bootstrap summaries, not full-V/SOTA evidence.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from PIL import Image

import score_BrainHIVE_trialset_fastscreen_20261027_v1 as S
from generate_BrainHIVE_prediction_fastscreen_20261027_v1 import validate_arms, SCHEMA as GEN_SCHEMA

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_generic_prediction_DINO_T80_20261027_v1'
MAX_SECONDS = 600
OUTPUT_CAP = 64 << 20
sha, read, atomic_json, checked_ref = S.sha, S.read, S.atomic_json, S.checked_ref


def validate_inputs(completion_path, expected_arms=None, check_pixels=True):
    complete = read(completion_path)
    assert complete['status'] == 'COMPLETED_GENERIC_NATIVE_PREDICTION_SCREEN' and complete['schema'] == GEN_SCHEMA
    arms = validate_arms(complete['arms'])
    if expected_arms is not None:
        assert list(arms) == list(expected_arms), 'Explicit arm order changed'
    expected_images = 80 * len(arms)
    assert complete['queries'] == 80 and complete['images'] == expected_images
    assert complete['per_arm_images'] == {arm: 80 for arm in arms}
    assert complete['native_NFE_per_image'] == 4 and complete['resolution'] == [512, 512]
    assert complete['no_best_of_N'] is True and complete['normal_inference_target_access'] is False
    assert complete['target_pixels_loaded_after_generation'] is True
    assert complete['panel_already_exposed_development'] is True and complete['independent_confirmation'] is False
    manifest = checked_ref(complete['manifest'])
    checked_ref(complete['inference_manifest'])
    panel_path = checked_ref(complete['panel'])
    checked_ref(complete['input_provenance'])
    panel = read(panel_path)
    assert panel['expected_queries'] == 80 and len(panel['queries']) == 80
    panel_queries = {query['query_id']: query for query in panel['queries']}
    assert len(panel_queries) == 80
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows) == expected_images
    grouped, paths, inodes = {}, set(), set()
    for row in rows:
        assert row['arm'] in arms and row['query_id'] in panel_queries
        assert row['target_mapping_attached_after_inference'] is True
        assert row['normal_inference_target_access'] is False and row['oracle_normalization'] is False
        assert row['one_image_per_query'] is True and row['repeat_protocol'] == 'R4' and row['UNet_NFE'] == 4
        assert row['panel_already_exposed_development'] is True
        query = panel_queries[row['query_id']]
        for key in ('row', 'source_row', 'array_row', 'subject_id', 'concept_id', 'stimulus_id', 'fold', 'image_seed'):
            assert row[key] == query[key], f'Panel identity mismatch: {key}'
        group = grouped.setdefault(row['query_id'], {})
        assert row['arm'] not in group
        group[row['arm']] = row
        path = Path(row['generated_path']).resolve()
        info = path.stat()
        assert str(path) not in paths and (info.st_dev, info.st_ino) not in inodes, 'Generated path/hardlink reused'
        paths.add(str(path)); inodes.add((info.st_dev, info.st_ino))
        assert sha(path) == row['generated_sha256']
        assert Path(row['target_image_path']).resolve() == Path(row['score_target_path']).resolve()
        if check_pixels:
            with Image.open(path) as image:
                assert image.size == (512, 512)
                assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() == row['generated_pixel_sha256']
    assert set(grouped) == set(panel_queries)
    queries = sorted(panel_queries.values(), key=lambda query: (int(query['fold']), str(query['concept_id']), int(query['array_row'])))
    assert len({query['subject_id'] for query in queries}) == 1
    assert len({query['stimulus_id'] for query in queries}) == 80 and len({query['array_row'] for query in queries}) == 80
    targets, ordered, concepts = [], [], {}
    for query in queries:
        group = grouped[query['query_id']]
        assert set(group) == set(arms)
        first = group[arms[0]]
        for arm in arms[1:]:
            for key in ('target_image_path', 'target_image_sha256', 'initial_noise_latents_sha256', 'initial_UNet_input_sha256'):
                assert group[arm][key] == first[key], f'Paired target/noise mismatch: {key}'
        target = Path(first['target_image_path']).resolve()
        assert sha(target) == first['target_image_sha256']
        if check_pixels:
            with Image.open(target) as image:
                assert min(image.size) > 0
                image.convert('RGB').load()
        targets.append(target)
        concepts.setdefault(query['concept_id'], []).append(query)
        ordered.extend(group[arm] for arm in arms)
    assert len({str(path) for path in targets}) == 80
    assert len(concepts) == 8 and all(len(group) == 10 for group in concepts.values())
    assert all(len({query['fold'] for query in group}) == 1 for group in concepts.values())
    assert {fold: sum(group[0]['fold'] == fold for group in concepts.values()) for fold in (0, 1)} == {0: 4, 1: 4}
    return complete, manifest, panel_path, queries, ordered, targets, arms


def feature_statistics(generated_features, target_features, queries, arms):
    assert generated_features.shape == (80 * len(arms), 768) and target_features.shape == (80, 768)
    assert torch.isfinite(generated_features).all() and torch.isfinite(target_features).all()
    assert (generated_features.norm(dim=1) > 0).all() and (target_features.norm(dim=1) > 0).all()
    metrics = {arm: S.metric_arrays(generated_features[index::len(arms)], target_features, queries) for index, arm in enumerate(arms)}
    original = S.metric_arrays(target_features, target_features, queries)
    assert np.allclose(original['target_cosine'], 1, atol=2e-5)
    assert np.all(original['global80_two_afc'] >= 1 - 1e-6), 'Original self retrieval/feature uniqueness failed'
    generator = np.random.default_rng(1729)
    permutation = generator.permutation(80)
    while np.any(permutation == np.arange(80)):
        permutation = generator.permutation(80)
    permuted = S.metric_arrays(target_features[torch.tensor(permutation)], target_features, queries)
    controls = dict(original_selfcheck={name: float(values.mean()) for name, values in original.items()},
                    deterministic_derangement=permutation.tolist(),
                    deranged_original_target_labels={name: float(values.mean()) for name, values in permuted.items()},
                    permutation_is_diagnostic_not_a_significance_test=True)
    return metrics, controls


def comparisons(metrics, queries, arms, reference='native_parent'):
    assert reference in arms
    pairs = [(arm, reference) for arm in arms if arm != reference]
    if 'component_mean' in arms and 'posterior_mean' in arms:
        pair = ('component_mean', 'posterior_mean')
        if pair not in pairs:
            pairs.append(pair)
    output = {}
    for candidate, baseline in pairs:
        values = S.paired_bootstrap({'mean': metrics[baseline], 'each_mean': metrics[candidate]}, queries)
        renamed = {}
        for name, value in values.items():
            renamed[name] = {('difference' if key == 'each_mean_minus_mean' else key): item for key, item in value.items()}
        output[candidate + '_minus_' + baseline] = dict(candidate=candidate, baseline=baseline, metrics=renamed)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-registration', type=Path)
    parser.add_argument('--gpu-job-name')
    parser.add_argument('--arms', nargs='+')
    parser.add_argument('--reference-arm', default='native_parent')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    assert 0 < args.max_seconds <= MAX_SECONDS and 0 < args.batch <= 32
    invoked = time.monotonic()
    complete, manifest, panel_path, queries, ordered, targets, arms = validate_inputs(args.completion, args.arms)
    assert args.reference_arm in arms
    expected_images = 80 * len(arms)
    completion_sha, manifest_sha = sha(args.completion), sha(manifest)
    if args.preflight_only:
        print(json.dumps(dict(status='GENERIC_COMPLETE_T80_SCORING_INPUTS_CPU_VERIFIED', images=expected_images,
                              queries=80, arms=list(arms), GPU_started=False)), flush=True)
        return 0
    assert args.native_registration and args.gpu_job_name
    assert not args.output.exists(), 'Preserve every attempt'
    args.output.mkdir(parents=True)
    try:
        import brainhive_paired_study_contract_20261025_v1 as C
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        config = C.read(args.native_registration); C.validate(config); C.setup_runtime(config)
        D.torch = torch
        document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds,
                        maximum_output_bytes=OUTPUT_CAP, maximum_extra_CPU_bytes=10 << 30,
                        maximum_GPU_bytes=20 << 30, minimum_CPU_headroom_bytes=10 << 30,
                        resource_authorization=config['resource_authorization'])
        wrapper = D.verify_wrapper(document); budget = D.Budget(document)
        stopped = [False]
        signal.signal(signal.SIGTERM, lambda signum, frame: stopped.__setitem__(0, True))
        def check():
            assert not stopped[0] and time.monotonic() - invoked < args.max_seconds
            budget.check()
            assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file()) <= OUTPUT_CAP
        torch.set_num_threads(2); torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        import image_evaluation as E
        assert E.DINO_SHA256 == S.DINO_SHA256 and E.DINO_WEIGHT.is_file()
        provenance = dict(schema=SCHEMA, source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
            score_helper_source=dict(path=str(Path(S.__file__).resolve()), sha256=sha(S.__file__)),
            generation_completion=dict(path=str(args.completion.resolve()), sha256=completion_sha),
            generation_manifest=dict(path=str(manifest.resolve()), sha256=manifest_sha),
            panel=dict(path=str(panel_path.resolve()), sha256=sha(panel_path)), arms=list(arms),
            image_evaluation_source_sha256=sha(ROOT/'image_evaluation.py'),
            locked_metrics_source_sha256=sha(ROOT/'locked_metrics.py'),
            DINO_weight=dict(path=str(E.DINO_WEIGHT), sha256=E.DINO_SHA256),
            dino_features='layer11 normalized CLS + mean patch, L2-normalized; 256 bilinear antialias; historical real CPU/generated CUDA resize',
            generated_images_encoded_once=expected_images, targets_encoded_once=80,
            wrapper=wrapper, fixed_gallery=80, same_concept_gallery=10,
            panel_already_exposed_development=True, independent_confirmation=False)
        atomic_json(args.output/'provenance.json', provenance)
        check(); model = E.dino_model(); check()
        def extract(paths, real):
            pieces = []
            for start in range(0, len(paths), args.batch):
                check(); pieces.append(E.dino_features(paths[start:start+args.batch], model, batch_size=args.batch, real_images=real))
            return torch.cat(pieces).float()
        generated_features = extract([Path(row['generated_path']) for row in ordered], False)
        target_features = extract(targets, True)
        del model; torch.cuda.empty_cache(); check()
        metrics, controls = feature_statistics(generated_features, target_features, queries, arms)
        paired = comparisons(metrics, queries, arms, args.reference_arm)
        feature_path = args.output/'DINO_features.pt'
        torch.save(dict(generated=generated_features, targets=target_features, arms=list(arms),
                        generated_keys=[(row['query_id'],row['arm']) for row in ordered],
                        target_stimulus_ids=[query['stimulus_id'] for query in queries], DINO_weight_sha256=E.DINO_SHA256), feature_path)
        per_query = args.output/'per_query.csv'; names = list(metrics[arms[0]])
        with per_query.open('x', newline='', encoding='utf-8') as stream:
            fields = ['query_id','subject_id','array_row','concept_id','stimulus_id','fold','arm','generated_path','generated_sha256','target_image_path','target_image_sha256',*names]
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for index,query in enumerate(queries):
                for arm_index,arm in enumerate(arms):
                    row=ordered[index*len(arms)+arm_index]
                    writer.writerow({**{key:row[key] for key in fields if key not in names},
                                     **{name:float(metrics[arm][name][index]) for name in names}})
        per_concept = {concept:{arm:{name:float(values[[i for i,q in enumerate(queries) if q['concept_id']==concept]].mean())
                                    for name,values in metrics[arm].items()} for arm in arms}
                       for concept in sorted({query['concept_id'] for query in queries})}
        assert sha(args.completion)==completion_sha and sha(manifest)==manifest_sha
        for row in ordered: assert sha(row['generated_path'])==row['generated_sha256']
        for index,target in enumerate(targets): assert sha(target)==ordered[index*len(arms)]['target_image_sha256']
        check()
        summary=dict(status='COMPLETED_GENERIC_NATIVE_PREDICTION_DINO_SCREEN',schema=SCHEMA,
            subjects=1,concepts=8,queries=80,arms=list(arms),generated_images=expected_images,distinct_target_images=80,
            rows_per_arm=80,full_registered_panel_denominator=True,
            means={arm:{name:float(value.mean()) for name,value in metrics[arm].items()} for arm in arms},
            paired_comparisons=paired,per_concept=per_concept,controls=controls,
            bootstrap=dict(seed=1729,draws=1000,resampling_unit='eight concepts of ten paired queries',
                           gallery='original fixed80 gallery; not rebuilt by bootstrap',descriptive_only=True),
            original_generation_NFE=4,original_generation_resolution=[512,512],no_best_of_N=True,
            full_V_evaluation=False,SOTA_established=False,panel_already_exposed_development=True,
            independent_confirmation=False,claim_limit='exposed single-subject eight-concept T-only development screen; no fullV/SOTA/independent confirmation/mechanism conclusion',
            generated_features_shape=list(generated_features.shape),target_features_shape=list(target_features.shape),
            features=dict(path=str(feature_path.resolve()),sha256=sha(feature_path)),per_query=dict(path=str(per_query.resolve()),sha256=sha(per_query)),
            provenance_sha256=sha(args.output/'provenance.json'),seconds=time.monotonic()-invoked,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        atomic_json(args.output/'summary.json',summary)
        atomic_json(args.output/'completion.json',dict(status=summary['status'],schema=SCHEMA,
            summary=dict(path=str((args.output/'summary.json').resolve()),sha256=sha(args.output/'summary.json')),
            generation_completion_sha256=completion_sha,generation_manifest_sha256=manifest_sha,
            arms=list(arms),expected_images=expected_images,observed_images=expected_images,
            expected_targets=80,observed_targets=80,per_query_sha256=sha(per_query),features_sha256=sha(feature_path),
            seconds=time.monotonic()-invoked,panel_already_exposed_development=True))
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)
        return 0
    except BaseException as error:
        atomic_json(args.output/'failure.json',dict(status='FAILED_GENERIC_SCREEN_FIXED_DENOMINATOR_NO_RETRY',
            error=repr(error),traceback=traceback.format_exc(),expected_images=expected_images,expected_targets=80,
            arms=list(arms),seconds=time.monotonic()-invoked,no_effect_claim=True))
        raise


if __name__=='__main__':
    raise SystemExit(main())
