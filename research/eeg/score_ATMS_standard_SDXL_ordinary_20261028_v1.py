"""Cache both new ATMS image arms, then apply the fixed paired80->400 rule.

The old D09 target400 DINO features may be reused. Every ATMS native baseline
image and feature is new; BrainHIVE generated images are never a comparator.
CPU scoring reuses the unchanged rapid-v2 metric/tie/bootstrap/decision code.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import generate_ATMS_standard_SDXL_ordinary_20261028_v1 as I
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S

SCHEMA = I.SCHEMA
FEATURE_SCHEMA = 'ATMS_standard_SDXL_ordinary_DINO_feature_cache_20261028_v1'
TRANSFORM = 'image_evaluation.dino_features: RGB/255; 256x256 bilinear antialias align_corners=False; real CPU resize/generated CUDA resize; ImageNet mean/std; normalized layer11 CLS+mean-patch concatenation; L2-normalized768'
sha, ref, read, checked, atomic = I.sha, I.ref, I.read, I.checked, I.atomic


def feature_rows(records):
    keys = (*I.IDENTITY, 'query_index', 'arm', 'generated_sha256', 'generated_pixel_sha256',
        'target_image_sha256', 'initial_noise_latents_sha256', 'initial_UNet_input_sha256',
        'condition_sha256', 'condition_BF16_sha256', 'tokens_sha256', 'prepared')
    return [{key: row[key] for key in keys} for row in records]


def load_images(output, stage, spec, spec_path):
    lock, _, identities, queries = I.prepared(output, spec, spec_path)
    lock_ref = ref(Path(output) / 'prepared/completion.json')
    audit_path = Path(output) / f'image_audit{stage}.json'; audit = read(audit_path)
    assert audit['schema'] == SCHEMA and audit['stage'] == stage
    assert audit['status'] == 'ALL_FIXED_STAGE_ATMS_IMAGES_IDENTITY_CONDITION_NOISE_PIXELS_CPU_AUDITED'
    assert audit['spec'] == ref(spec_path) and audit['prepared'] == lock_ref and audit['images'] == 2 * stage
    closure_path = checked(audit['image_closure']); closure = read(closure_path)
    assert closure['status'] == 'COMPLETE_FIXED_ATMS_PAIRED_IMAGE_STAGE' and closure['stage'] == stage
    assert closure['prepared'] == lock_ref and closure['generation'] == I.GENERATION
    assert closure['HIVE_images_used_as_ATMS_baseline'] is False
    records = I.stage_rows(output, stage, identities, lock_ref)
    assert records == [json.loads(line) for line in checked(closure['manifest']).read_text().splitlines()]
    if stage == 400:
        I.stage400_gate(output, spec_path, lock_ref)
    return lock_ref, queries, closure, ref(closure_path), ref(audit_path), records


def load_feature_cache(path, output, stage, spec, spec_path):
    doc = read(path)
    assert doc['schema'] == FEATURE_SCHEMA and doc['status'] == 'COMPLETE_NEW_ATMS_NATIVE_AND_ORDINARY_DINO_CACHE'
    assert doc['stage'] == stage and doc['shape'] == [stage, 2, 768] and doc['dtype'] == 'float32'
    assert doc['arms'] == list(I.ARMS) and doc['generation'] == I.GENERATION
    assert doc['spec'] == ref(spec_path) and doc['prepared'] == ref(Path(output) / 'prepared/completion.json')
    assert doc['encoder']['weight_sha256'] == S.DINO_SHA256 and doc['encoder']['transform'] == TRANSFORM
    assert doc['extractor_source']['sha256'] == sha(__file__)
    assert doc['image_evaluation_source']['sha256'] == sha(I.ROOT / 'image_evaluation.py')
    assert doc['semantic_evaluator_source']['sha256'] == sha(I.ROOT / 'semantic_evaluator.py')
    values = np.load(checked(doc['features']), allow_pickle=False)
    assert values.shape == (stage, 2, 768) and values.dtype == np.float32 and np.isfinite(values).all()
    assert np.allclose(np.linalg.norm(values.astype(np.float64), axis=-1), 1, atol=2e-5)
    _, _, closure, closure_ref, audit_ref, records = load_images(output, stage, spec, spec_path)
    assert doc['image_closure'] == closure_ref and doc['image_audit'] == audit_ref
    rows = read(checked(doc['rows'])); assert rows == feature_rows(records)
    checked(doc['image_source_manifest']); assert doc['image_source_manifest'] == closure['manifest']
    if stage == 400:
        old_path = checked(doc['stage80_feature_cache']); old = read(old_path)
        prefix = np.load(checked(old['features']), allow_pickle=False)
        assert prefix.shape == (80, 2, 768) and values[:80].tobytes() == prefix.tobytes()
        assert rows[:160] == read(checked(old['rows']))
    else:
        assert doc['stage80_feature_cache'] is None
    return doc, values, rows


def encode(args, spec):
    output = args.output / f'features{args.stage}'; assert not output.exists(); output.mkdir(parents=True)
    lock_ref, queries, closure, closure_ref, audit_ref, records = load_images(args.output, args.stage, spec, args.spec)
    prefix = np.empty((0, 2, 768), np.float32); old_ref = None
    if args.stage == 400:
        old_path = args.output / 'features80/completion.json'
        _, prefix, old_rows = load_feature_cache(old_path, args.output, 80, spec, args.spec)
        assert old_rows == feature_rows(records[:160]); old_ref = ref(old_path)
    budget, wrapper = I.bootstrap(spec, args.max_seconds)
    torch = I.torch
    import image_evaluation as E
    assert E.DINO_SHA256 == S.DINO_SHA256 and sha(E.DINO_WEIGHT) == S.DINO_SHA256
    try:
        model = E.dino_model(); pieces = []
        new = records[len(prefix) * 2:]
        assert len(new) == (160 if args.stage == 80 else 640)
        for start in range(0, len(new), 32):
            I.check_budget(spec, args.output, budget)
            paths = [Path(row['generated_path']) for row in new[start:start + 32]]
            pieces.append(E.dino_features(paths, model, batch_size=32, real_images=False).float().numpy())
        values = np.concatenate([prefix, np.concatenate(pieces).reshape(len(new) // 2, 2, 768)])
        assert values.shape == (args.stage, 2, 768) and np.isfinite(values).all()
        assert values[:len(prefix)].tobytes() == prefix.tobytes()
        assert np.allclose(np.linalg.norm(values.astype(np.float64), axis=-1), 1, atol=2e-5)
        del model; torch.cuda.empty_cache()
        I.save_array(output / 'features.npy', values); atomic(output / 'rows.json', feature_rows(records))
        # Revalidate every committed image after extraction, without reading any target features.
        assert load_images(args.output, args.stage, spec, args.spec)[-1] == records
        I.check_budget(spec, args.output, budget, work=False)
        doc = dict(schema=FEATURE_SCHEMA, status='COMPLETE_NEW_ATMS_NATIVE_AND_ORDINARY_DINO_CACHE',
            stage=args.stage, spec=ref(args.spec), prepared=lock_ref, image_closure=closure_ref,
            image_audit=audit_ref, image_source_manifest=closure['manifest'], shape=list(values.shape),
            dtype='float32', arms=list(I.ARMS), generation=I.GENERATION, features=ref(output / 'features.npy'),
            rows=ref(output / 'rows.json'), stage80_feature_cache=old_ref,
            encoder=dict(weight_sha256=S.DINO_SHA256, feature_dimensions=768, transform=TRANSFORM),
            extractor_source=ref(__file__), image_evaluation_source=ref(I.ROOT / 'image_evaluation.py'),
            semantic_evaluator_source=ref(I.ROOT / 'semantic_evaluator.py'), DINO_weight=ref(E.DINO_WEIGHT),
            new_generated_images_encoded=len(new), preserved_prefix_feature_rows=len(prefix) * 2,
            targets_encoded=0, HIVE_generated_images_encoded=0, new_images=0, wrapper=wrapper, **budget.result())
        atomic(output / 'completion.json', doc)
        load_feature_cache(output / 'completion.json', args.output, args.stage, spec, args.spec)
        print(json.dumps(dict(status=doc['status'], completion=ref(output / 'completion.json'))), flush=True)
    except BaseException as exc:
        import traceback
        atomic(output / 'failure.json', dict(status='ATMS_PAIRED_DINO_ATTEMPT_FAILED_PRESERVED',
            error=repr(exc), traceback=traceback.format_exc(), no_automatic_retry=True, **budget.result()))
        raise


def targets(spec, queries):
    # Reuse only the real-target array from D09; no generated HIVE pixels or
    # generated HIVE feature arrays are loaded or used as an ATMS baseline.
    previous = read(checked(spec['D09_target_provenance']))
    assert previous['status'] == 'COMPLETE_D09_STAGE400'
    target_ref = previous['check_target_features']; supplied = spec['Dcheck_target_features']
    assert target_ref['sha256'] == supplied['sha256'] and Path(target_ref['path']).resolve() == Path(supplied['path']).resolve()
    previous_inputs = read(checked(previous['inputs']))
    assert previous_inputs['panel']['sha256'] == spec['Dcheck_panel']['sha256']
    assert read(checked(previous_inputs['panel']))['Dcheck'] == queries
    value = np.load(checked(supplied), allow_pickle=False)
    assert value.shape == (400, 768) and value.dtype == np.float32 and np.isfinite(value).all()
    assert np.allclose(np.linalg.norm(value.astype(np.float64), axis=-1), 1, atol=2e-5)
    test = S.metric_arrays(value, value, queries, queries)
    assert np.allclose(test['target_cosine'], 1, atol=2e-5)
    assert np.all(test['DINO2AFC'] == 1) and np.all(test['gallery_top1'] == 1)
    return value


def score_values(values, target, queries, gallery, stage):
    assert values.shape == (stage, 2, 768) and target.shape == (400, 768) and len(gallery) == 400
    arrays = {arm: S.metric_arrays(values[:, i], target, queries, gallery) for i, arm in enumerate(I.ARMS)}
    paired = S.compare(arrays[I.ARMS[1]], arrays[I.ARMS[0]], queries)
    return arrays, paired, S.decision(paired, str(stage))


def score(args, spec):
    output = args.output / f'score{args.stage}'; assert not output.exists(); output.mkdir(parents=True)
    source = args.output / f'features{args.stage}/completion.json'
    doc, values, records = load_feature_cache(source, args.output, args.stage, spec, args.spec)
    _, _, _, gallery = I.panel_data(spec)
    target = targets(spec, gallery)
    prior_ref = None
    if args.stage == 400:
        prior_ref = I.stage400_gate(args.output, args.spec, doc['prepared'])
        prior = read(prior_ref['path'])
        assert prior['source']['sha256'] == sha(__file__) and prior['metric_source']['sha256'] == sha(S.__file__)
        checked(prior['feature_completion'])
        old = read(prior['feature_completion']['path'])
        prefix = np.load(checked(old['features']), allow_pickle=False)
        assert prefix.tobytes() == values[:80].tobytes() and records[:160] == read(checked(old['rows']))
        assert prior['target_features'] == spec['Dcheck_target_features']
    arrays, paired, action = score_values(values, target, gallery[:args.stage], gallery, args.stage)
    table = output / 'per_query.csv'
    fields = [*I.IDENTITY, 'arm', *S.METRICS, *[key + '_minus_native' for key in S.METRICS]]
    with table.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for i, query in enumerate(gallery[:args.stage]):
            for arm, metrics in arrays.items():
                writer.writerow({key: query[key] for key in I.IDENTITY} | dict(arm=arm) |
                    {key: float(metrics[key][i]) for key in S.METRICS} |
                    {key + '_minus_native': float(metrics[key][i] - arrays[I.ARMS[0]][key][i]) for key in S.METRICS})
    done = dict(schema=SCHEMA, status='COMPLETE_ATMS_ORDINARY_PAIRED_DINO_SCREEN', stage=args.stage,
        queries=args.stage, gallery_targets=400, arms=list(I.ARMS), generation=I.GENERATION,
        spec=ref(args.spec), prepared=doc['prepared'], image_closure=doc['image_closure'],
        image_audit=doc['image_audit'], feature_completion=ref(source), target_features=spec['Dcheck_target_features'],
        target_feature_provenance=spec['D09_target_provenance'], previous_score=prior_ref,
        source=ref(__file__), metric_source=ref(S.__file__), per_query=ref(table),
        means={arm: {key: float(value.mean()) for key, value in metrics.items()} for arm, metrics in arrays.items()},
        ordinary_minus_native=paired, decision=action, target_self_pairing_DINO2AFC=1.,
        hidden_exclusions=0, full_registered_denominator=True, same400_gallery_for_both_stages=True,
        HIVE_generated_images_used_as_ATMS_baseline=False, native_ATMS_baseline_images_are_new=True,
        no_intermediate_240_scoring=True, no_model_strength_or_seed_selection_using_Dcheck=True,
        inference_cost_unchanged=True, official_author_ATM_full_pipeline_reproduced=False,
        independent_confirmation=False, SOTA_established=False, CUDA_used=False)
    atomic(output / 'completion.json', done)
    print(json.dumps(dict(status=done['status'], decision=action, means=done['means'],
        completion=ref(output / 'completion.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('features', 'score'), required=True)
    parser.add_argument('--stage', type=int, choices=(80, 400), required=True)
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args(); assert 120 <= args.max_seconds <= 1800
    spec = I.load_policy(args.spec, allow_target_features=args.mode == 'score')
    assert Path(spec['image_output_root']).resolve() == args.output.resolve()
    if args.mode == 'features':
        return encode(args, spec)
    return score(args, spec)


if __name__ == '__main__':
    raise SystemExit(main())
