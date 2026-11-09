"""Continue the unchanged three-arm experiment after its recorded storage stop.

This controller owns new state, lock, logs and GPU receipts.  It never rewrites
the failed controller, the frozen recipe, an existing checkpoint, or any saved
image.  The original image producer performs its own exact-manifest resume.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from brainhive_lifecycle_storage_v1 import verify_remaining


ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
PYTHON = '/root/miniconda3/bin/python'
STATE = WORK / 'audit/main_storage_continuation_20261021_v1.json'
LOCK = ROOT / 'audit/main_storage_continuation_20261021_v1.lock'
OLD_STATE = WORK / 'audit/bounded_full_lifecycle_queue_v1.json'
RECIPE = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
EXPECTED_RECIPE_SHA = '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
SUFFIX = '_storage_resume1'
CONTEXT = {}
EXTRA_SOURCES = [
    'queue_BrainHIVE_main_storage_continuation_20261021_v1.py',
    'run_gpu_job_v2.py', 'audit_BrainHIVE_image_arm_CPU_v1.py',
    'extract_image_features.py', 'image_evaluation.py', 'locked_metrics.py',
    'score_full_manifest.py', 'score_low_level.py',
]


class StorageStop(RuntimeError):
    """Stop the whole continuation, rather than cascading resource failures."""


class SharedDependencyFailure(RuntimeError):
    """A failed shared evaluator cannot be retried under an existing receipt."""


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def update(**values):
    CONTEXT.update(values)
    CONTEXT.update(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid())
    STATE.parent.mkdir(parents=True, exist_ok=True)
    pending = STATE.with_name(STATE.name + '.pending')
    pending.write_text(json.dumps(CONTEXT, indent=2, allow_nan=False) + '\n')
    pending.replace(STATE)


def verify_sources(config):
    assert sha(RECIPE) == EXPECTED_RECIPE_SHA, 'The original recipe must remain unchanged'
    for name, expected in config['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(ROOT / 'queue_BrainHIVE_bounded_full_lifecycle_v1.py') == config['queue_source_sha256']
    assert sha(ROOT / 'compare_BrainHIVE_bounded_V_CPU_v1.py') == config['comparison_source_sha256']
    for name, expected in CONTEXT['additional_sources_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(OLD_STATE) == CONTEXT['preserved_old_state_sha256'], 'Preserve the historical failed state'


def storage(config):
    verify_sources(config)
    try:
        result = verify_remaining(config)
    except AssertionError as error:
        raise StorageStop(str(error)) from error
    update(last_remaining_lifecycle=result)
    return result


def checkpoint(arm):
    folder = WORK / 'components/bounded_v1' / arm
    done = read(folder / 'completion.json')
    assert done['status'] == 'COMPLETE_ALL10_REAL_OOF_THREE_PASSES_AND_FIXED_FINAL_COMPONENT'
    assert done['arm'] == arm and done['updates'] == 24807 and done['total_seen_queries'] == 396900
    assert done['parameters'] == (525825 if arm == 'risk' else 525568)
    expected = folder / 'component_final.pt'
    assert Path(done['checkpoint_path']) == expected and sha(expected) == done['checkpoint_sha256']
    assert sha(folder / 'input_provenance.json') == done['input_provenance_sha256']
    provenance = read(folder / 'input_provenance.json')
    assert provenance['recipe_sha256'] == EXPECTED_RECIPE_SHA and provenance['arm'] == arm
    assert sha(folder / 'calibration.json') == done['calibration_sha256']
    return done


def gpu(config, name, entry, arguments, seconds):
    storage(config)  # Check every GPU job, including evaluator jobs.
    receipt = ROOT / 'audit' / (name + '_job.json')
    log_path = WORK / 'logs' / (name + '.log')
    assert not receipt.exists() and not log_path.exists(), 'Use a new attempt; preserve prior receipts/logs'
    argv = [PYTHON, str(ROOT / 'run_gpu_job_v2.py'), '--name', name,
            '--max-seconds', str(seconds), '--', PYTHON, '-u', str(ROOT / entry)] + arguments
    update(status='RUNNING_ORIGINAL_MAIN_STORAGE_CONTINUATION', job=name, argv=argv)
    with log_path.open('x') as log:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        update(wrapper_pid=child.pid)
        code = child.wait()
    update(last_job=name, last_job_exit_code=code, wrapper_pid=None)
    if code != 0:
        with log_path.open('rb') as stream:
            stream.seek(max(0, log_path.stat().st_size - 32768))
            tail = stream.read().decode('utf-8', errors='replace')
        if any(marker in tail for marker in [
                'Remaining main lifecycle storage forecast is not funded',
                'Preserve saved images and stop before private disk buffer is consumed']):
            raise StorageStop('Producer storage guard stopped ' + name + '; inspect its preserved log')
    return code


def cpu(config, entry, arguments, seconds=3600):
    verify_sources(config)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    update(cpu_entry=entry)
    subprocess.run([PYTHON, str(ROOT / entry)] + arguments, cwd=ROOT, timeout=seconds, check=True, env=env)
    update(cpu_entry=None)


def train_ordinary(config):
    folder = WORK / 'components/bounded_v1/ordinary'
    if (folder / 'completion.json').exists():
        checkpoint('ordinary')
        return True
    progress = folder / 'progress.json'
    resume = folder.exists()
    if resume:
        assert (folder / 'resume.pt').exists() and progress.exists(), 'Unfinished original output needs explicit diagnosis'
        assert read(progress)['status'] == 'RESOURCE_STOPPED_RESUMABLE', 'Never overwrite an unexplained partial fit'
        # A pre-existing interrupted fit gets the one original exact-resume bound.
        name = 'BrainHIVE_bounded_full_T_ordinary_v1' + SUFFIX + '_exact_resume1'
        code = gpu(config, name, 'train_BrainHIVE_bounded_component_v1.py',
                   ['--arm', 'ordinary', '--recipe', str(RECIPE), '--resume'], 7200)
    else:
        name = 'BrainHIVE_bounded_full_T_ordinary_v1' + SUFFIX
        code = gpu(config, name, 'train_BrainHIVE_bounded_component_v1.py',
                   ['--arm', 'ordinary', '--recipe', str(RECIPE)], 43200)
        if code != 0 and progress.exists() and read(progress).get('status') == 'RESOURCE_STOPPED_RESUMABLE':
            used = read(ROOT / 'audit' / (name + '_job.json'))['charged_seconds']
            remaining = min(7200, 50400 - used)
            assert remaining > 120
            code = gpu(config, name + '_exact_resume1', 'train_BrainHIVE_bounded_component_v1.py',
                       ['--arm', 'ordinary', '--recipe', str(RECIPE), '--resume'], remaining)
    if code == 0:
        checkpoint('ordinary')
    return code == 0


def images_done(folder):
    path = folder / 'completion.json'
    if not path.exists():
        return False
    done = read(path)
    assert done['status'] == 'ALL_ACTUAL_LOSSLESS_IMAGES_SAVED_INFERENCE_ONLY' and done['queries'] == 16600
    assert (done['role'], done['split'], done['intervention']) == ('full', 'V', 'correct')
    assert sha(folder / 'inference_manifest.jsonl') == done['inference_manifest_sha256']
    return True


def pairing(config, run, folder):
    receipt = folder / 'pairing_audit.json'
    if receipt.exists():
        proof = read(receipt)
        assert proof['status'] == 'ALL_SAVED_IMAGES_TARGETS_AND_STIMULUS_ROWS_JOIN_VERIFIED' and proof['queries'] == 16600
        assert sha(folder / 'manifest.jsonl') == proof['manifest_sha256']
        assert sha(proof['gallery_manifest']) == proof['gallery_manifest_sha256']
    else:
        assert not (folder / 'manifest.jsonl').exists(), 'Preserve an interrupted identity join for diagnosis'
        cpu(config, 'audit_BrainHIVE_image_arm_CPU_v1.py', ['--run-name', run])


def feature_done(folder, manifest, count):
    path = folder / 'completion.json'
    if not path.exists():
        return False
    done = read(path)
    assert done['status'] == 'COMPLETED' and done['queries'] == count
    assert done['manifest_sha256'] == sha(manifest)
    return True


def evaluate(config, arm):
    run = config['run_names'][arm]
    folder = WORK / 'images' / run
    if not images_done(folder):
        code = gpu(config, 'BrainHIVE_' + run + '_images_v1' + SUFFIX, 'generate_BrainHIVE_bounded_V_v1.py',
                   ['--arm', arm, '--recipe', str(RECIPE)], 14400)
        if code != 0:
            # A producer can encounter a changed space forecast between jobs.
            # Propagate the resource stop before trying another arm.
            storage(config)
            return 'Image producer failed; all existing images/manifests preserved'
        assert images_done(folder)
    pairing(config, run, folder)
    manifest = folder / 'manifest.jsonl'
    gallery = WORK / 'gallery/V/manifest.jsonl'
    gallery_features = WORK / 'evaluation/bounded_V_gallery_features_v1'
    if not feature_done(gallery_features, gallery, 1660):
        code = gpu(config, 'BrainHIVE_bounded_V_gallery_features_v1' + SUFFIX, 'extract_image_features.py',
                   ['--manifest', str(gallery), '--output', str(gallery_features), '--real-gallery'], 14400)
        if code != 0:
            storage(config)
            raise SharedDependencyFailure('Shared gallery evaluator failed; preserve its shards and stop dependent evaluation')
        assert feature_done(gallery_features, gallery, 1660)
    features = WORK / 'evaluation' / run / 'features'
    if not feature_done(features, manifest, 16600):
        code = gpu(config, 'BrainHIVE_' + run + '_features_v1' + SUFFIX, 'extract_image_features.py',
                   ['--manifest', str(manifest), '--output', str(features)], 14400)
        if code != 0:
            storage(config)
            return 'Full image-feature evaluator failed; complete image originals and partial shards preserved'
        assert feature_done(features, manifest, 16600)
    output = WORK / 'evaluation' / run
    scores = output / 'scores'
    if (scores / 'summary.json').exists():
        summary = read(scores / 'summary.json')
        assert summary['status'] == 'COMPLETED' and summary['queries'] == 16600
    else:
        assert not (scores / 'per_query.csv').exists(), 'Preserve a partial scoring file; never silently overwrite it'
        cpu(config, 'score_full_manifest.py', ['--manifest', str(manifest), '--features', str(features),
            '--gallery-manifest', str(gallery), '--gallery-features', str(gallery_features), '--output', str(scores)])
    # Validate saved-query identities even when a completed score file is reused.
    rows = {r['query_id']: r for r in map(json.loads, manifest.read_text().splitlines())}
    with (scores / 'per_query.csv').open(newline='') as stream:
        query_scores = list(csv.DictReader(stream))
    assert len(query_scores) == len({r['query_id'] for r in query_scores}) == len(rows) == 16600
    for row in query_scores:
        expected = rows[row['query_id']]
        for key in ['generated_path', 'stimulus_id', 'subject', 'score_target_path']:
            assert row[key] == expected[key]
        assert int(row['image_seed']) == expected['image_seed']
    low = output / 'low_level'
    if not (low / 'summary.json').exists():
        assert not (low / 'per_query.csv').exists(), 'Preserve a partial low-level audit'
        cpu(config, 'score_low_level.py', ['--manifest', str(manifest), '--gallery-manifest', str(gallery), '--output', str(low)])
    summary = read(low / 'summary.json')
    assert summary['verified_and_decoded'] == 16600 and summary['manifest_sha256'] == sha(manifest)
    assert summary['gallery_manifest_sha256'] == sha(gallery)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    args = parser.parse_args()
    assert Path(args.recipe).resolve() == RECIPE.resolve(), 'Only the original frozen recipe is authorized here'
    assert not STATE.exists(), 'This continuation has its own single attempt and preserved terminal state'
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not STATE.exists()
        assert sha(RECIPE) == EXPECTED_RECIPE_SHA
        config = read(RECIPE)
        assert config['fixed_arm_order'] == ['risk', 'ordinary', 'parent']
        assert config['training_job_seconds'] == 43200 and config['generation_job_seconds'] == config['evaluator_job_seconds'] == 14400
        old = read(OLD_STATE)
        assert old['status'] in ['FINITE_MAIN_COMPARISON_PARTIAL_WITH_EXPLICIT_FAILURES', 'FAILED_PRESERVED_NEEDS_EXPLICIT_DIAGNOSIS']
        risk = checkpoint('risk')  # Never refit the already complete idea arm.
        update(status='CHECKING_REMAINING_STORAGE_FOR_ORIGINAL_MAIN_CONTINUATION',
            recipe_path=str(RECIPE), recipe_sha256=EXPECTED_RECIPE_SHA, old_state_path=str(OLD_STATE),
            preserved_old_state_sha256=sha(OLD_STATE), preserved_old_status=old['status'],
            additional_sources_sha256={name: sha(ROOT / name) for name in EXTRA_SOURCES},
            risk_checkpoint_path=risk['checkpoint_path'], risk_checkpoint_sha256=risk['checkpoint_sha256'],
            original_sources_sha256=config['source_sha256'], original_queue_sha256=config['queue_source_sha256'],
            original_comparison_sha256=config['comparison_source_sha256'], completed=[], failures={},
            fixed_arm_order=config['fixed_arm_order'], training_seed=1729, new_candidate_configurations=0,
            no_recipe_or_model_revision=True, no_hyperparameter_search=True, scientific_goal_complete=False)
        try:
            storage(config)
            for arm in config['fixed_arm_order']:
                update(current_arm=arm)
                try:
                    if arm == 'ordinary' and not train_ordinary(config):
                        raise RuntimeError('Original ordinary training failed; checkpoint/logs preserved')
                    if arm == 'risk':
                        assert checkpoint('risk')['checkpoint_sha256'] == CONTEXT['risk_checkpoint_sha256']
                    failure = evaluate(config, arm)
                    if failure:
                        raise RuntimeError(failure)
                    CONTEXT['completed'].append(arm)
                except StorageStop:
                    raise
                except SharedDependencyFailure:
                    raise
                except Exception as error:
                    CONTEXT['failures'][arm] = type(error).__name__ + ': ' + str(error)[:1600]
                update(status='ORIGINAL_MAIN_CONTINUATION_ARM_SAVED_OR_FAILURE_PRESERVED')
            if set(CONTEXT['completed']) == {'risk', 'ordinary', 'parent'}:
                assert not (WORK / 'comparison/bounded_v1').exists(), 'Preserve any existing comparison'
                cpu(config, 'compare_BrainHIVE_bounded_V_CPU_v1.py', ['--recipe', str(RECIPE)], 7200)
                update(status='ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',
                       mechanism_confirmed=False, independent_SOTA_established=False)
            else:
                update(status='FINITE_MAIN_STORAGE_CONTINUATION_PARTIAL_WITH_EXPLICIT_FAILURES')
        except StorageStop as error:
            update(status='STORAGE_FORECAST_STOPPED_PRESERVED_NO_CASCADING_JOBS', error=str(error)[:2400])
            raise
        except Exception as error:
            update(status='FAILED_MAIN_STORAGE_CONTINUATION_PRESERVED_NEEDS_DIAGNOSIS',
                   error_type=type(error).__name__, error=str(error)[:2400])
            raise
        finally:
            verify_sources(config)
            assert checkpoint('risk')['checkpoint_sha256'] == CONTEXT['risk_checkpoint_sha256']
            update(original_recipe_sources_and_completed_risk_checkpoint_unchanged=True,
                   recipe_sha256_at_exit=sha(RECIPE))


if __name__ == '__main__':
    main()
