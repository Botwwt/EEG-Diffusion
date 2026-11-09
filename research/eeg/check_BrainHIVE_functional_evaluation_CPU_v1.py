"""Tiny synthetic child-output/identity fixtures; never real image scoring."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import run_BrainHIVE_functional_evaluation_CPU_v1 as entry


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader(); out.writerows(rows)


def rejects(call, part=None):
    try:
        call()
    except (ValueError, KeyError, OSError) as error:
        assert part is None or part in str(error), str(error)
        return
    raise AssertionError('Invalid input admitted')


def fixture(root):
    # Reuse real generation identity fixture + actual tiny pixel artifact audit,
    # not a fabricated formal audit. Only scorer children are injected below.
    import check_BrainHIVE_functional_full_V_artifacts_CPU_v1 as source
    recipe_path, options, folder = source.fixture(root, ['normal', 'constant'], arm='O')
    recipe = json.loads(recipe_path.read_text())
    for filename in ('inference_manifest.jsonl', 'manifest.jsonl'):
        path = folder / filename
        rows = [json.loads(line) | dict(method='BrainHIVE_' + folder.name) for line in path.read_text().splitlines()]
        source.jsonl(path, rows)
    inf_path, proof_path, done_path = (folder / n for n in ('inference_completion.json', 'pairing_audit.json', 'completion.json'))
    inf, proof, done = [json.loads(p.read_text()) for p in (inf_path, proof_path, done_path)]
    inf['inference_manifest_sha256'] = entry.sha(folder / 'inference_manifest.jsonl')
    proof['manifest_sha256'] = done['manifest_sha256'] = entry.sha(folder / 'manifest.jsonl')
    write(inf_path, inf); write(proof_path, proof)
    done.update(inference_completion_sha256=entry.sha(inf_path), pairing_audit_sha256=entry.sha(proof_path))
    write(done_path, done)
    work, repo = options['work'], options['repo']
    write(work / 'audit/main_storage_continuation_20261021_v1.json', dict(
        status='ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED', pid=12345,
        completed=['parent', 'ordinary', 'risk'], failures=[]))
    for name in ('seed_hf_acquisition_completion.json', 'seed_direct_assets_receipt.json'):
        write(repo / 'audit' / name, dict(synthetic_receipt_only=True))
    features = work / 'evaluation' / folder.name / 'features'
    gallery_features = work / 'evaluation' / 'synthetic_shared_gallery' / 'features'
    manifests = [folder / 'manifest.jsonl', Path(recipe['generation_gallery_manifest'])]
    for feature_root, manifest, is_gallery in zip((features, gallery_features), manifests, (False, True)):
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        keys = [r['stimulus_id' if is_gallery else 'query_id'] for r in rows]
        write(feature_root / 'config.json', dict(manifest=str(manifest), manifest_sha256=entry.sha(manifest), keys=keys,
            real_gallery=is_gallery, DINO_weight_sha256='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d',
            SEED_source_commit='5327a134b90b04fb1b37066d98b309c9cc533a02',
            dino_preprocessing='Historical routes CPU resize real/CUDA resize generated; 256 bilinear antialias; layer11 CLS+mean patch concatenated and normalized',
            ObjectF1_grid='public code np.arange(.05,.95,.05); area50; preserve empty detections convention',
            semantic_model_receipts=[entry.sha(repo / 'audit' / name) for name in ('seed_hf_acquisition_completion.json', 'seed_direct_assets_receipt.json')]))
        write(feature_root / 'completion.json', dict(status='COMPLETED', queries=2, manifest_sha256=entry.sha(manifest),
            all_queries_used=True, query_subsampling=False))
        for stage in entry.STAGES:
            path = feature_root / stage / '00000000.pt'; path.parent.mkdir(parents=True)
            path.write_bytes(b'Synthetic bytes only; never loaded as tensors. ' + stage.encode())
    admission_path = Path(recipe['admission_receipt'])
    admission = json.loads(admission_path.read_text()); plan_path = Path(admission['four_arm_lifecycle_plan']['path'])
    plan = json.loads(plan_path.read_text()); own = plan['arms'][0]
    pins = {name: entry.sha(entry.ROOT / name) for name in entry.DEPENDENCIES}
    own.update(lifecycle_source_sha256=pins, metadata_budget_bytes=256 << 20,
        feature_root=str(features), score_root=str(folder / 'scores'), low_level_root=str(folder / 'low_level'),
        artifact_audit_root=str(folder / 'artifacts'))
    plan.update(schema='BrainHIVE_functional_alignment_v2_four_arm_storage_plan', all_four_train_generate_score_lifecycles_implemented=True,
        evaluation=dict(schema=entry.SCHEMA + '_sealed_plan', maximum_seconds=entry.MAX_SECONDS,
            maximum_output_bytes=entry.MAX_BYTES, minimum_CPU_headroom_bytes=entry.MIN_MEMORY,
            lifecycle_source_sha256=pins, gallery_features_root=str(gallery_features)),
        comparison=dict(gallery_features_root=str(gallery_features), gallery_features_reuse_only=True))
    write(plan_path, plan)
    admission['four_arm_lifecycle_plan']['sha256'] = entry.sha(plan_path); write(admission_path, admission)
    recipe['admission_sha256'] = entry.sha(admission_path); write(recipe_path, recipe)
    options['process_live'] = lambda pid: False
    context = entry.artifacts.prepare(recipe_path, time.monotonic() + 30, _fixture=options)
    result = entry.artifacts.audit_context(context, Path(own['artifact_audit_root']), max_seconds=30)
    assert result['complete_technical_coverage'] and result['status'] == 'CPU_TINY_FIXTURE_ONLY'
    return recipe_path, options


def fake_runner(mode='normal'):
    calls = []
    def run(argv, logfile, deadline, context):
        assert argv == entry.commands(context)[len(calls)]
        assert argv[0] == sys.executable and Path(argv[1]).name in ('score_full_manifest.py', 'score_low_level.py')
        calls.append(copy.deepcopy(argv))
        logfile.write_text('Synthetic injected child; no scorer/model/tensor/target pixels executed.\n')
        if len(calls) == 1:
            values = dict(DINO2AFC=.6, DINO_cosine=.2, DINO_Top1=0, DINO_Top5=1, EffNet=.3, CapSim=.2, ObjectF1=.1, SEED=.2)
            rows = [{k: row[k] for k in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'EEG_source_stimulus_id',
                    'target_image_path', 'generated_path', 'score_target_path', 'image_seed')} | values | dict(
                    width=512, height=512, pixel_std=.2, near_black_fraction=0, near_white_fraction=0, blank_proxy=False)
                    for row in context['rows']]
            if mode == 'wrong_target': rows[0]['score_target_path'] = 'another_target.jpg'
            if mode == 'partial': rows = rows[:1]
            csv_write(context['score'] / 'per_query.csv', rows)
            write(context['score'] / 'summary.json', dict(status='COMPLETED', queries=2, subjects=1,
                gallery_images=2, method=context['rows'][0]['method'], intervention='correct', repeat_protocol='R4',
                metric_version='fixed_real_gallery_explicit_stimulus_v1',
                metric_source_sha256=context['source_pins']['locked_metrics.py'],
                summary={k: dict(mean=v) for k,v in values.items()}, per_subject={'sub-01': values}))
            if mode == 'input_drift':
                Path(context['feature_shards'][0]['path']).write_bytes(b'changed after admission')
        else:
            rows = [{k: row[k] for k in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'generated_path')} |
                    dict(status='OK', scoring_target_path=row['score_target_path'], PixCorr_RGB=.2, SSIM_Y=.3,
                         mean_RGB_absolute_error=.1, generated_pixel_std=.2) for row in context['rows']]
            csv_write(context['low'] / 'per_query.csv', rows)
            write(context['low'] / 'summary.json', dict(status='COMPLETED', expected_images=2, actual_present=2,
                verified_and_decoded=2, missing_count=0, corrupt_or_mismatched_count=0,
                manifest_sha256=context['pins'][str(context['manifest'])],
                gallery_manifest_sha256=context['pins'][str(context['gallery_manifest'])],
                source_sha256=context['source_pins']['score_low_level.py'], GPU_used=False))
            write(context['low'] / 'failures.json', [])
            write(context['low'] / 'duplicate_hash_groups.json', {})
        return dict(argv=argv + (['--wrong'] if mode == 'wrong_argv' else []),
            exit_code=1 if mode == 'failure' else 0, seconds=.001, stdout_log=entry.ref(logfile))
    return run, calls


def check():
    assert 'torch' not in sys.modules, 'Wrapper import must not import Torch or model definitions'
    checks = []
    actual_draft = entry.ROOT / 'configs/BrainHIVE_functional_alignment_v2_preregistration_DRAFT.json'
    with patch.object(entry, 'main_terminal', side_effect=AssertionError('DRAFT cannot inspect main or scores')), \
         patch.object(entry.subprocess, 'Popen', side_effect=AssertionError('DRAFT cannot launch a child')):
        rejects(lambda: entry.prepare(actual_draft, time.monotonic() + 10), 'DRAFT')
    checks.append('actual_current_DRAFT_refused_before_main_artifacts_scores_or_child')
    with tempfile.TemporaryDirectory(prefix='hive_evaluation_CPU_') as temporary:
        base = Path(temporary)
        draft = base / 'DRAFT.json'; write(draft, dict(schema='BrainHIVE_functional_alignment_v2_DRAFT'))
        opened = []
        original = entry.read_json
        def traced(path, pins, expected=None):
            opened.append(str(path)); return original(path, pins, expected)
        with patch.object(entry, 'read_json', traced), patch.object(entry.subprocess, 'Popen', side_effect=AssertionError('No child permitted')):
            rejects(lambda: entry.prepare(draft, time.monotonic() + 10), 'DRAFT')
        assert opened == [str(draft)] and list(base.iterdir()) == [draft]
        checks.append('DRAFT_reads_only_recipe_no_subprocess_scores_output_or_Torch')
        for mode in ('normal', 'failure', 'wrong_argv', 'wrong_target', 'input_drift', 'partial'):
            recipe, options = fixture(base / mode)
            context = entry.prepare(recipe, time.monotonic() + 30, _fixture=options)
            runner, calls = fake_runner(mode)
            if mode == 'normal':
                result = entry.evaluate(context, time.monotonic() + 30, _runner=runner,
                    _memory=dict(effective_available_bytes=entry.MIN_MEMORY))
                assert result['status'] == 'CPU_TINY_FIXTURE_ONLY' and result['queries'] == 2
                assert result['actual_full_V16600_evaluation_completed'] is False
                assert len(calls) == 2 and len(result['feature_shards']) == len(result['gallery_feature_shards']) == 4
                assert context['closure'].is_file()
                rejects(lambda: entry.evaluate(context, time.monotonic() + 30, _runner=runner,
                    _memory=dict(effective_available_bytes=entry.MIN_MEMORY)), 'Existing output')
            else:
                rejects(lambda: entry.evaluate(context, time.monotonic() + 30, _runner=runner,
                    _memory=dict(effective_available_bytes=entry.MIN_MEMORY)))
                assert not context['closure'].exists() and (context['log'] / 'failure.json').is_file()
            checks.append('tiny_injected_' + mode)
        recipe, options = fixture(base / 'main_live')
        options['process_live'] = lambda pid: True
        reads = []
        def trace(path, pins, expected=None):
            reads.append(str(path)); return original(path, pins, expected)
        with patch.object(entry, 'read_json', trace):
            rejects(lambda: entry.prepare(recipe, time.monotonic() + 30, _fixture=options), 'actually inactive')
        assert reads == [str(recipe), str(options['work'] / 'audit/main_storage_continuation_20261021_v1.json')]
        checks.append('live_main_refused_before_features_artifact_or_scores')
    assert 'torch' not in sys.modules
    return dict(schema='BrainHIVE_functional_evaluation_CPU_contract_v1', status='PASS_TINY_INJECTED_CHILD_SCHEMA_AND_PROVENANCE_ONLY',
        source_sha256={Path(entry.__file__).name: entry.sha(entry.__file__), Path(__file__).name: entry.sha(__file__)},
        actual_DRAFT_sha256=entry.sha(actual_draft),
        checks=checks, actual_scorer_subprocess_calls=0, synthetic_child_invocations_only=True,
        actual_V_score_rows_read=0, actual_16600_evaluation_completed=False, GPU_used=False, model_loaded=False,
        real_feature_tensors_loaded=False, runtime_or_storage_admission_issued=False,
        real_14400s_full_scoring_runtime_verified=False, metric_definition_changed=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path); args=parser.parse_args()
    result = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
