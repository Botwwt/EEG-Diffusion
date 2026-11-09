"""Own-arm frozen-feature CPU scoring and SHA-bound evaluation closure.

No fitting, feature extraction, model construction, image sampling or metric
selection. Existing scorer CLIs run once in bounded CPU children. A closure is
written only after complete identity/denominator/provenance checks. Synthetic
injection tests are never formal16600 evaluation receipts.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import audit_BrainHIVE_functional_full_V_artifacts_CPU_v1 as artifacts

ROOT, WORK = artifacts.ROOT, artifacts.WORK
SCHEMA = 'BrainHIVE_functional_evaluation_CPU_v1'
STATUS = 'FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED'
MAX_SECONDS, MAX_BYTES, MIN_MEMORY = 14400, 64 << 20, 2 << 30
STAGES = ('DINO', 'EffNet', 'CapSim', 'ObjectF1')
METRICS = ('DINO2AFC', 'DINO_cosine', 'DINO_Top1', 'DINO_Top5', 'EffNet', 'CapSim', 'ObjectF1', 'SEED')
LOW = ('PixCorr_RGB', 'SSIM_Y', 'mean_RGB_absolute_error')
DEPENDENCIES = (Path(__file__).name, 'score_full_manifest.py', 'score_low_level.py',
    'extract_image_features.py', 'image_evaluation.py', 'locked_metrics.py',
    'semantic_evaluator.py', Path(artifacts.__file__).name, Path(artifacts.spatial.__file__).name)
require, sha, read_json = artifacts.require, artifacts.spatial.sha_file, artifacts.read_json


def tick(deadline):
    require(time.monotonic() < deadline, 'Fixed CPU evaluation time limit reached; no retry')


def child(path, root):
    return path.resolve() != root.resolve() and path.resolve().is_relative_to(root.resolve())


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def size(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def pin(path, pins):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/symlink immutable file: ' + str(path))
    actual = sha(path)
    require(str(path) not in pins or pins[str(path)] == actual, 'Input changed while preparing: ' + str(path))
    pins[str(path)] = actual
    return actual


def live(pid):
    require(os.name == 'posix' and Path('/proc').is_dir(), 'Actual Linux /proc required')
    require(type(pid) is int and pid > 0, 'Missing main process identity')
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[0] not in ('Z', 'X')
    except (FileNotFoundError, ProcessLookupError):
        return False


def main_terminal(work, pins, process_live=live):
    main = read_json(work / 'audit/main_storage_continuation_20261021_v1.json', pins)
    require(main.get('status') == 'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED'
            and not process_live(main.get('pid')), 'Original main must be successfully terminal and actually inactive')
    require(set(main['completed']) == {'risk', 'ordinary', 'parent'} and not main['failures'], 'Original main lifecycle not closed')


def feature_inputs(directory, manifest, keys, gallery, pins, repo, deadline):
    directory = Path(directory)
    config_path, done_path = directory / 'config.json', directory / 'completion.json'
    config, done = read_json(config_path, pins), read_json(done_path, pins)
    require(done['status'] == 'COMPLETED' and done['queries'] == len(keys)
            and done['all_queries_used'] is True and done['query_subsampling'] is False, 'Full frozen features incomplete')
    require(config['manifest'] == str(manifest) and config['manifest_sha256'] == done['manifest_sha256'] == pins[str(manifest)]
            and config['keys'] == keys and config['real_gallery'] is gallery, 'Feature config/input key identity differs')
    require(config['DINO_weight_sha256'] == '08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'
            and config['SEED_source_commit'] == '5327a134b90b04fb1b37066d98b309c9cc533a02', 'Frozen evaluator model identity differs')
    require(config['dino_preprocessing'] == 'Historical routes CPU resize real/CUDA resize generated; 256 bilinear antialias; layer11 CLS+mean patch concatenated and normalized'
            and config['ObjectF1_grid'] == 'public code np.arange(.05,.95,.05); area50; preserve empty detections convention', 'Frozen feature preprocessing declaration differs')
    receipts = [repo / 'audit/seed_hf_acquisition_completion.json', repo / 'audit/seed_direct_assets_receipt.json']
    require(config['semantic_model_receipts'] == [pin(p, pins) for p in receipts], 'Feature semantic asset receipts changed')
    expected = [directory / stage / (f'{start:08d}.pt') for stage in STAGES for start in range(0, len(keys), 256)]
    actual = [p for stage in STAGES for p in sorted((directory / stage).glob('*.pt'))]
    require(actual == expected, 'Missing/extra/out-of-order frozen feature shard')
    for path in expected:
        tick(deadline)
        pin(path, pins)
    return [ref(p) for p in expected]


def prepare(recipe_path, deadline, *, _fixture=None):
    """Production count is fixed16600. Fixture injection has no CLI exposure."""
    recipe_path = Path(recipe_path)
    pins = {}
    recipe = read_json(recipe_path, pins)
    require(recipe.get('schema') == 'BrainHIVE_functional_alignment_v2_recipe', 'DRAFT/unknown recipe cannot score or create output')
    require(recipe.get('arm') in ('O', 'P', 'PN', 'S'), 'Unknown arm')
    work = WORK if _fixture is None else _fixture['work']
    repo = ROOT if _fixture is None else _fixture['repo']
    main_terminal(work, pins, live if _fixture is None else _fixture['process_live'])
    context = artifacts.prepare(recipe_path, deadline, _fixture=_fixture)
    context['pins'].update(pins)
    pins = context['pins']
    admission = read_json(recipe['admission_receipt'], pins, recipe['admission_sha256'])
    plan_ref = admission['four_arm_lifecycle_plan']
    plan = read_json(plan_ref['path'], pins, plan_ref['sha256'])
    require(plan['schema'] == 'BrainHIVE_functional_alignment_v2_four_arm_storage_plan'
            and plan['all_four_train_generate_score_lifecycles_implemented'] is True, 'Lifecycle not sealed for scoring')
    registration = plan['evaluation']
    require(registration['schema'] == SCHEMA + '_sealed_plan'
            and registration['maximum_seconds'] == MAX_SECONDS
            and registration['maximum_output_bytes'] == MAX_BYTES
            and registration['minimum_CPU_headroom_bytes'] == MIN_MEMORY, 'Evaluation resource registration differs')
    source_pins = registration['lifecycle_source_sha256']
    # Runtime sources are this real repository, even in tiny data fixtures.
    for name in DEPENDENCIES:
        require(source_pins.get(name) == sha(ROOT / name), 'Sealed evaluation source changed: ' + name)
        pin(ROOT / name, pins)
    entry, metadata = context['entry'], Path(context['entry']['metadata_root'])
    for name in DEPENDENCIES:
        require(entry['lifecycle_source_sha256'].get(name) == source_pins[name], 'Own-arm lifecycle source differs')
    score, low, audit = (Path(entry[k]) for k in ('score_root', 'low_level_root', 'artifact_audit_root'))
    log = metadata / 'evaluation_CPU_v1_logs'
    closure = metadata / 'evaluation_closure.json'
    require(all(child(p, metadata) for p in (score, low, audit, log)), 'Outputs must be charged within own metadata_root')
    paths = [score, low, audit, log]
    require(len({str(p.resolve()) for p in paths}) == 4
            and all(not a.resolve().is_relative_to(b.resolve()) for a in paths for b in paths if a != b), 'Evaluation stage directories alias/nest')
    require(all(not p.exists() for p in (score, low, log, closure)), 'Preserve previous attempt; no overwrite or automatic retry')
    summary = read_json(audit / 'summary.json', pins)
    require(summary['status'] == ('COMPLETE_FULL_V_ARTIFACT_AUDIT' if _fixture is None else 'CPU_TINY_FIXTURE_ONLY')
            and summary['complete_technical_coverage'] is True
            and summary['actual_full_V16600_verified'] is (_fixture is None)
            and summary['arm'] == context['arm'] and summary['candidate'] == recipe['candidate']
            and summary['recipe_core_sha256'] == context['core'], 'Own complete artifact audit missing')
    require(all(summary[k] == context['count'] for k in ('expected_queries', 'full_denominator', 'identity_verified',
        'canonical_path_verified', 'decoded_512_RGB', 'hash_verified_decoded_without_other_failures', 'per_query_rows_written'))
        and summary['per_query_rows_unwritten'] == summary['spatial_flags_unknown'] == 0
        and not summary['failures'] and not summary['source_errors'], 'Artifact audit retains failures/unknown')
    require(summary['source_sha256'] == source_pins[Path(artifacts.__file__).name]
            and summary['measurement_source_sha256'] == source_pins[Path(artifacts.spatial.__file__).name]
            and summary['immutable_metadata_sha256'][str(context['manifest'])] == pins[str(context['manifest'])], 'Artifact source/input differs')
    require(pin(audit / 'per_query.csv.gz', pins) == summary['table_sha256'], 'Artifact table SHA differs')
    pin(audit / 'pixel_duplicate_groups.csv.gz', pins)
    rows = artifacts.read_jsonl(context['manifest'], pins, pins[str(context['manifest'])])
    inference = artifacts.read_jsonl(context['inference'], pins, pins[str(context['inference'])])
    require(len(rows) == len(inference) == context['count'], 'Full generated manifest count differs')
    with gzip.open(audit / 'per_query.csv.gz', 'rt', newline='', encoding='utf-8') as stream:
        checked = list(csv.DictReader(stream))
    require(len(checked) == context['count'], 'Incomplete artifact table')
    for i, (row, inf, checked_row) in enumerate(zip(rows, inference, checked)):
        tick(deadline)
        issues, path_ok, identity, path = artifacts.identity_issues(context, inf, row, i)
        require(not issues and path_ok and checked_row['query_id'] == checked_row['expected_query_id'] == identity,
                'Generated/EEG/seed/target identity differs')
        require(checked_row['generated_path'] == checked_row['expected_generated_path'] == str(path)
                and checked_row['manifest_generated_sha256'] == checked_row['actual_file_sha256'] == row['generated_sha256']
                and checked_row['source_manifest_sha256'] == pins[str(context['manifest'])] and checked_row['status'] == 'OK'
                and all(checked_row[k] == 'True' for k in ('hash_verified', 'decoded_512_RGB', 'identity_verified', 'canonical_path_verified')),
                'Artifact per-query verification incomplete')
    require(len({r['query_id'] for r in rows}) == len({r['generated_path'] for r in rows}) == context['count'], 'Duplicate generated identity/path')
    gallery_manifest = Path(recipe['generation_gallery_manifest'])
    gallery_rows = artifacts.read_jsonl(gallery_manifest, pins, pins[str(gallery_manifest)])
    feature_root = Path(entry['feature_root'])
    require(feature_root == work / 'evaluation' / context['run_name'] / 'features', 'Wrong own-arm frozen features')
    gallery_root = Path(registration['gallery_features_root'])
    require(str(gallery_root) == plan['comparison']['gallery_features_root']
            and plan['comparison']['gallery_features_reuse_only'] is True, 'Gallery reuse identity differs')
    shards = feature_inputs(feature_root, context['manifest'], [r['query_id'] for r in rows], False, pins, repo, deadline)
    gallery_shards = feature_inputs(gallery_root, gallery_manifest, [r['stimulus_id'] for r in gallery_rows], True, pins, repo, deadline)
    used = size(metadata)
    require(entry['metadata_budget_bytes'] - used >= MAX_BYTES, 'Sealed remaining metadata budget cannot fund complete evaluation cap')
    require(shutil.disk_usage(metadata).free >= MAX_BYTES + (64 << 20), 'CPU scoring cap plus64MiB persistent-space buffer unavailable')
    context.update(rows=rows, gallery_manifest=gallery_manifest, feature_root=feature_root, gallery_feature_root=gallery_root,
        feature_shards=shards, gallery_feature_shards=gallery_shards, score=score, low=low, log=log, closure=closure,
        artifact=audit, metadata=metadata, metadata_used=used, source_pins={n: source_pins[n] for n in DEPENDENCIES},
        plan_ref=plan_ref, evaluation_registration=registration)
    return context


def commands(context):
    common = ['--manifest', str(context['manifest'])]
    gallery = ['--gallery-manifest', str(context['gallery_manifest'])]
    return [[sys.executable, str(ROOT / 'score_full_manifest.py'), *common, '--features', str(context['feature_root']),
             *gallery, '--gallery-features', str(context['gallery_feature_root']), '--output', str(context['score'])],
            [sys.executable, str(ROOT / 'score_low_level.py'), *common, *gallery, '--output', str(context['low'])]]


def child_run(argv, logfile, deadline, context):
    """Fixed real source, shell=False, CUDA hidden; stdout retained, never parsed."""
    remaining = max(1, math.ceil(deadline - time.monotonic()))
    require(os.name == 'posix', 'Real bounded scorer children require Linux resource limits')
    import resource
    def bounds():
        resource.setrlimit(resource.RLIMIT_CPU, (remaining, remaining))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    for name in artifacts.spatial.THREAD_VARIABLES:
        env[name] = '1'
    started = time.monotonic()
    with Path(logfile).open('xb') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   shell=False, start_new_session=True, preexec_fn=bounds)
        try:
            while process.poll() is None:
                tick(deadline)
                require(size(context['metadata']) - context['metadata_used'] <= MAX_BYTES - (64 << 10), 'Evaluation output cap reached')
                time.sleep(.25)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return dict(argv=argv, exit_code=process.returncode, seconds=time.monotonic() - started, stdout_log=ref(logfile))


def validate_outputs(context):
    """Retain complete query/target/seed ordering; no metric is a selection gate."""
    pins = {}
    score = read_json(context['score'] / 'summary.json', pins)
    low = read_json(context['low'] / 'summary.json', pins)
    n = context['count']
    require(score['status'] == low['status'] == 'COMPLETED' and score['queries'] == n
            and score['subjects'] == len(context['inputs']['subjects']) and score['gallery_images'] == len(context['gallery']), 'Incomplete scorer denominator')
    require(score['method'] == context['rows'][0]['method'] and score['intervention'] == 'correct'
            and score['repeat_protocol'] == 'R4' and score['metric_version'] == 'fixed_real_gallery_explicit_stimulus_v1'
            and score['metric_source_sha256'] == context['source_pins']['locked_metrics.py'], 'Scorer interface/source differs')
    require(low['expected_images'] == low['actual_present'] == low['verified_and_decoded'] == n
            and low['missing_count'] == low['corrupt_or_mismatched_count'] == 0
            and low['manifest_sha256'] == context['pins'][str(context['manifest'])]
            and low['gallery_manifest_sha256'] == context['pins'][str(context['gallery_manifest'])]
            and low['source_sha256'] == context['source_pins']['score_low_level.py'] and low['GPU_used'] is False, 'Low-level full mechanical closure failed')
    failures = read_json(context['low'] / 'failures.json', pins)
    require(failures == [], 'Low-level failures remain')
    read_json(context['low'] / 'duplicate_hash_groups.json', pins)
    tables = []
    for path in (context['score'] / 'per_query.csv', context['low'] / 'per_query.csv'):
        pin(path, pins)
        with path.open(newline='', encoding='utf-8') as stream:
            tables.append(list(csv.DictReader(stream)))
    require(all(len(table) == len({r['query_id'] for r in table}) == n for table in tables), 'Partial/duplicate scorer CSV')
    sums = collections.defaultdict(float)
    for row, scored, mechanical in zip(context['rows'], *tables):
        for key in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'EEG_source_stimulus_id', 'target_image_path',
                    'generated_path', 'score_target_path', 'image_seed'):
            require(scored[key] == str(row[key]), 'Score target/query/seed/path differs: ' + key)
        for key in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'generated_path'):
            require(mechanical[key] == str(row[key]), 'Low-level identity differs: ' + key)
        require(mechanical['status'] == 'OK' and mechanical['scoring_target_path'] == row['score_target_path'], 'Low-level target/failure differs')
        values = {k: float(scored[k]) for k in METRICS}
        require(all(math.isfinite(v) for v in values.values()) and all(math.isfinite(float(mechanical[k])) for k in LOW), 'Nonfinite score retained; no closure')
        require(all(0 <= values[k] <= 1 for k in ('DINO2AFC', 'ObjectF1'))
                and all(values[k] in (0., 1.) for k in ('DINO_Top1', 'DINO_Top5'))
                and abs(values['SEED'] - sum(values[k] for k in ('EffNet', 'CapSim', 'ObjectF1')) / 3) < 1e-10, 'Invalid frozen metric algebra/range')
        require(int(scored['width']) == int(scored['height']) == 512, 'Unexpected scored image dimensions')
        for key, value in values.items():
            sums[(row['subject'], key)] += value
    for subject in context['inputs']['subjects']:
        count = sum(r['subject'] == subject for r in context['rows'])
        for key in METRICS:
            require(abs(score['per_subject'][subject][key] - sums[(subject, key)] / count) <= 2e-6, 'Per-subject summary/CSV differs')
    for key in METRICS:
        require(abs(score['summary'][key]['mean'] - sum(sums[(s, key)] for s in context['inputs']['subjects']) / n) <= 2e-6,
                'Balanced full-query summary/CSV differs')
    return pins


def evaluate(context, deadline, *, _runner=child_run, _memory=None):
    for name in artifacts.spatial.THREAD_VARIABLES:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    memory = artifacts.spatial.memory_preflight() if _memory is None else _memory
    require(memory['effective_available_bytes'] >= MIN_MEMORY, 'CPU evaluation requires fresh2GiB memory headroom')
    require(all(not p.exists() for p in (context['score'], context['low'], context['log'], context['closure'])), 'Existing output attempt; no retry')
    context['log'].mkdir(parents=True, exist_ok=False)
    started, invocations = time.monotonic(), []
    try:
        for i, argv in enumerate(commands(context)):
            tick(deadline)
            for name, expected in context['source_pins'].items():
                require(sha(ROOT / name) == expected, 'Source changed before child launch: ' + name)
            call = _runner(argv, context['log'] / ('scorer_stdout.log' if i == 0 else 'low_level_stdout.log'), deadline, context)
            invocations.append(call)
            require(call['argv'] == argv and type(call['exit_code']) is int and call['exit_code'] == 0,
                    'Child argv differs or scorer failed; no automatic retry')
            require(call['stdout_log'] == ref(context['log'] / ('scorer_stdout.log' if i == 0 else 'low_level_stdout.log')), 'Child log identity differs')
            require(size(context['metadata']) - context['metadata_used'] <= MAX_BYTES - (64 << 10), 'Evaluation exceeded output cap')
        output_pins = validate_outputs(context)
        for path, expected in context['pins'].items():
            tick(deadline)
            require(sha(path) == expected, 'Immutable source/manifest/config/shard changed: ' + path)
        feature_again = sorted(str(p) for stage in STAGES for p in (context['feature_root'] / stage).glob('*.pt'))
        gallery_again = sorted(str(p) for stage in STAGES for p in (context['gallery_feature_root'] / stage).glob('*.pt'))
        require(feature_again == sorted(r['path'] for r in context['feature_shards'])
                and gallery_again == sorted(r['path'] for r in context['gallery_feature_shards']), 'Frozen shard membership changed')
        report = dict(schema=SCHEMA + '_closure', status=STATUS if not context['fixture'] else 'CPU_TINY_FIXTURE_ONLY',
            arm=context['arm'], candidate=context['recipe']['candidate'], recipe_core_sha256=context['core'], queries=context['count'],
            actual_full_V16600_evaluation_completed=not context['fixture'], actual_invocations=invocations,
            source_sha256=context['source_pins'], immutable_input_sha256=context['pins'], output_sha256=output_pins,
            feature_shards=context['feature_shards'], gallery_feature_shards=context['gallery_feature_shards'],
            full_identity_target_seed_noise_chain_verified=True, complete_low_level_mechanical_coverage=True,
            generation_noise_SHA_joined_from_inference_manifest_not_native_score_column=True,
            model_fitting=False, model_construction=False, feature_extraction=False, GPU_used=False, image_sampling=False,
            chosen_parameters_or_seed=False, automatic_retry=False, perceptual_quality_noninferiority_established=False,
            image_gain_claimed=False, human_quality_ratings_performed=False,
            budget=dict(maximum_seconds=MAX_SECONDS, maximum_output_bytes=MAX_BYTES, seconds=time.monotonic() - started,
                CPU_memory_preflight=memory, actual_full16600_runtime_bound_unmeasured=True,
                all_outputs_charged_within_own_metadata_root=True, output_bytes_before_closure=size(context['metadata']) - context['metadata_used']))
        refs = dict(manifest=context['manifest'], gallery_manifest=context['gallery_manifest'],
            features_config=context['feature_root'] / 'config.json', features_completion=context['feature_root'] / 'completion.json',
            gallery_features_config=context['gallery_feature_root'] / 'config.json', gallery_features_completion=context['gallery_feature_root'] / 'completion.json',
            score_summary=context['score'] / 'summary.json', score_per_query=context['score'] / 'per_query.csv',
            low_summary=context['low'] / 'summary.json', low_per_query=context['low'] / 'per_query.csv',
            artifact_summary=context['artifact'] / 'summary.json', artifact_per_query=context['artifact'] / 'per_query.csv.gz')
        report.update({key: ref(path) for key, path in refs.items()})
        encoded = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
        require(size(context['metadata']) - context['metadata_used'] + len(encoded) <= MAX_BYTES, 'Closure would exceed evaluation output cap')
        require(size(context['metadata']) + len(encoded) <= context['entry']['metadata_budget_bytes'], 'Own metadata allowance exhausted')
        with context['closure'].open('xb') as stream:
            stream.write(encoded)
        return report
    except Exception as error:
        failure = dict(schema=SCHEMA, status='CPU_EVALUATION_FAILED_NO_CLOSURE_NO_AUTOMATIC_RETRY',
            arm=context['arm'], expected_queries=context['count'], error=type(error).__name__ + ': ' + str(error)[:350],
            actual_invocations=invocations, full_denominator_preserved=True, actual_full_V16600_evaluation_completed=False)
        with (context['log'] / 'failure.json').open('x', encoding='utf-8') as stream:
            json.dump(failure, stream, indent=2)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    deadline = time.monotonic() + MAX_SECONDS
    try:
        context = prepare(args.recipe, deadline)
        if args.preflight_only:
            print(json.dumps(dict(status='OWN_ARM_CPU_EVALUATION_PREFLIGHT_COMPLETE_NO_SCORES_READ', arm=context['arm'], queries=context['count'])))
            return
        report = evaluate(context, deadline)
        print(json.dumps(dict(status=report['status'], arm=report['arm'], queries=report['queries'], closure=str(context['closure']))))
    except (KeyError, ValueError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(status='REFUSED_OR_FAILED_NO_EVALUATION_CLOSURE', error=str(error)[:350])))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
