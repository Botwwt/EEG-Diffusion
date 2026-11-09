"""One diagnosed O restart after page-cache pressure, preserving the failed run.

The original 29 bound sources, recipes, admissions, profiles, queue and logs
are read-only. This controller reuses the original Runner and stage APIs. O
uses its second predeclared job name WITHOUT --resume after proving no saved
checkpoint exists. Its failed four-file directory is SHA-verified and renamed
to an archive. P/PN/S retain their original full training and exact-resume
policy. A failure here never causes another O restart or an automatic retry.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

import brainhive_functional_training_contract_v2 as C
import prepare_BrainHIVE_functional_lifecycle_CPU_20261022_v1 as P
import queue_BrainHIVE_functional_lifecycle_20261022_v1 as Q

ROOT, WORK, BASE, CONFIG = P.ROOT, P.WORK, P.BASE, P.CONFIG
STATE = BASE / 'formal_recovery_queue.json'
ARCHIVE = BASE / 'preserved_failed_O_before_first_checkpoint_20261022_v1'
CACHE_HELPER = 'release_BrainHIVE_verified_runtime_pagecache_CPU_20261022_v1.py'
DURABLE = P.DATA / 'runtime_durable_20261022_v2'
MIN_RESTART_HEADROOM = 24 << 30
ORIGINAL_HEADROOM = 8 << 30
SMALL_FAILED_CAP = 4 << 20


def load_ref(item):
    value = P.load_ref(item)
    if 'bytes' in item:
        assert Path(item['path']).stat().st_size == item['bytes']
    return value


def check_ref(item):
    assert C.sha(item['path']) == item['sha256'], item['path']
    if 'bytes' in item:
        assert Path(item['path']).stat().st_size == item['bytes']


def second_slot_restart_command(recipe):
    name, seconds, entry, args = Q.command(recipe, 'train', index=1)
    assert recipe['arm'] == 'O' and name == recipe['formal_job_names'][1]
    assert args[-1] == '--resume' and args.count('--resume') == 1
    args = args[:-1]
    assert '--resume' not in args
    return name, seconds, entry, args


def validate_authorization(value, recipe):
    assert value['schema'] == 'BrainHIVE_functional_O_headroom_recovery_authorization_20261022_v1'
    assert value['status'] == 'EXPLICIT_SAME_SEED_SECOND_SLOT_O_RESTART_REVIEWED'
    assert value['training_seed'] == 1729 and value['arm'] == 'O'
    assert value['restart_type'] == 'from_scratch_no_checkpoint'
    assert value['maximum_additional_O_fit_attempts'] == 1
    assert value['restart_job_name'] == recipe['formal_job_names'][1]
    assert value['minimum_fresh_effective_headroom_bytes'] == MIN_RESTART_HEADROOM
    assert value['original_failed_attempt_retained'] is True and value['recipe_core_unchanged'] is True
    assert value['controller_source_sha256'] == C.sha(__file__)
    assert value['cache_helper_source_sha256'] == C.sha(ROOT / CACHE_HELPER)


def verify_failure(recipe, evidence, recipe_path):
    """Only the observed first O assertion, with no checkpoint, is restartable."""
    assert set(evidence) == {'formal_queue', 'fit_job', 'fit_log', 'output_files'}
    expected_job = ROOT / 'audit' / (recipe['formal_job_names'][0] + '_job.json')
    expected_log = BASE / (recipe['formal_job_names'][0] + '.log')
    assert Path(evidence['formal_queue']['path']) == BASE / 'formal_queue.json'
    assert Path(evidence['fit_job']['path']) == expected_job
    assert Path(evidence['fit_log']['path']) == expected_log
    queue, job = load_ref(evidence['formal_queue']), load_ref(evidence['fit_job'])
    check_ref(evidence['fit_log'])
    assert queue['status'] == 'FORMAL_LIFECYCLE_STOPPED_WITH_PRESERVED_EVIDENCE_NO_AUTOMATIC_RETRY'
    assert not C.process_live(queue.get('pid')) and not queue.get('unresolved_live_child')
    assert queue['completed'] == [] and len(queue['attempts']) == 1
    attempt = queue['attempts'][0]
    assert queue['current_stage'] == attempt['name'] == recipe['formal_job_names'][0]
    assert attempt['exit_code'] != 0 and Path(attempt['log']) == expected_log
    assert attempt['log_sha256'] == evidence['fit_log']['sha256']
    assert job['status'] == 'FAILED_OR_JOB_TIMEOUT_RESUMABLE' and job['exit_code'] == 1
    assert job['name'] == recipe['formal_job_names'][0]
    assert not C.process_live(job.get('pid')) and not C.process_live(job.get('wrapper_pid'))
    assert 'termination_reason' not in job, 'A killed/time-limited job is a different diagnosis'
    assert 0 < job['charged_seconds'] < recipe['training_job_seconds']
    assert str(ROOT / 'train_BrainHIVE_functional_alignment_v2.py') in job['command']
    assert '--recipe' in job['command'] and '--resume' not in job['command']
    assert Path(job['command'][job['command'].index('--recipe') + 1]) == recipe_path
    text = expected_log.read_text(encoding='utf-8')
    assert len(text.encode()) <= 16 << 20
    assert re.search(r'brainhive_functional_training_contract_v2\.py["\x27], line 442, in verify_admission', text)
    assert 'snapshot' in text and 'AssertionError' in text
    assert "memory['effective_headroom_bytes'] >= limits['minimum_cgroup_headroom_bytes']" in text
    assert not any(token in text for token in ('OutOfMemoryError', 'CUDA out of memory', 'nonfinite'))
    output = Path(recipe['output_directory'])
    assert output == WORK / 'components/functional_alignment_v2/O'
    assert output.is_dir() and not output.is_symlink()
    names = {'input_provenance.json', 'launch_' + str(job['pid']) + '.json', 'training.jsonl', 'progress.json'}
    actual = list(output.iterdir())
    assert {p.name for p in actual} == names and all(p.is_file() and not p.is_symlink() for p in actual)
    assert len(evidence['output_files']) == len(names)
    assert {Path(item['path']) for item in evidence['output_files']} == set(actual)
    assert sum(p.stat().st_size for p in actual) <= SMALL_FAILED_CAP
    for item in evidence['output_files']:
        check_ref(item)
    provenance = P.read(output / 'input_provenance.json')
    launch = P.read(output / ('launch_' + str(job['pid']) + '.json'))
    assert provenance == {k: v for k, v in launch.items() if k not in ('exclusive_GPU', 'admission_sha256', 'recipe_sha256')}
    assert provenance['arm'] == 'O' and provenance['candidate'] == recipe['candidate']
    assert provenance['recipe_core_sha256'] == C.core_sha(recipe)
    assert provenance['source_sha256'] == recipe['source_sha256']
    assert provenance['input_contract_sha256'] == recipe['input_contract_sha256']
    assert provenance['risk_head_or_gate'] is False and provenance['all_frozen_host_weights'] is True
    assert launch['recipe_sha256'] == C.sha(recipe_path) and launch['admission_sha256'] == recipe['admission_sha256']
    assert launch['exclusive_GPU']['wrapper_pid'] == job['wrapper_pid']
    assert launch['exclusive_GPU']['wrapper_job_name'] == job['name']
    rows = [json.loads(line) for line in (output / 'training.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['updates'] == 64 and rows[0]['seen_queries'] == 1024
    event = rows[0]
    assert P.read(output / 'progress.json') == dict(status='FUNCTIONAL_ALIGNMENT_V2_RUNNING', **event)
    assert event['arm'] == 'O' and event['pass_number'] == 1 and event['expected_updates'] == 24807
    assert event['expected_queries'] == 396900 and event['UNet_forwards'] == 2 and event['mean_gate'] == 1.
    assert event['functional_auxiliary'] == 0 and event['functional_diagnostics'] == {}
    for key in ('cumulative_mean_loss', 'last_loss', 'epsilon_loss', 'token_loss', 'gradient_norm', 'correction_RMS', 'seconds'):
        assert math.isfinite(event[key]) and event[key] >= 0
    return dict(failed_queue=evidence['formal_queue'], failed_job=evidence['fit_job'], failed_log=evidence['fit_log'],
        output_files=evidence['output_files'], original_output=str(output),
        last_logged_updates=64, last_logged_seen_queries=1024,
        actual_uncheckpointed_update_count='UNKNOWN; progress is emitted every64 updates',
        original_charged_seconds=job['charged_seconds'], no_checkpoint_or_final_saved=True)


def cache_records(receipt):
    assert receipt['status'] == 'TARGETED_PRIVATE_RUNTIME_PAGECACHE_RELEASE_VERIFIED_NO_FILE_REMOVAL'
    for key in ('deletion_performed', 'quantization_performed', 'GPU_used', 'headroom_threshold_changed'):
        assert receipt[key] is False, key
    assert receipt['SHA_and_path_and_inode_unchanged'] is True
    assert receipt['original_formal_minimum_headroom_bytes'] == ORIGINAL_HEADROOM
    assert receipt['minimum_restart_headroom_bytes'] == MIN_RESTART_HEADROOM
    assert receipt['restart_headroom_satisfied'] is True
    assert receipt['effective_headroom_after_bytes'] >= MIN_RESTART_HEADROOM
    assert 0 <= time.time() - receipt['issued_epoch'] <= 1800, 'Reclaim proof is stale; preserve it and diagnose again'
    assert receipt['source_sha256'] == C.sha(ROOT / CACHE_HELPER)
    before, after = receipt['cgroup_before'], receipt['cgroup_after']
    assert before['max'] == after['max'] > 0 and after['current'] < before['current']
    assert after['file'] <= before['file'] and after['shmem'] == before['shmem']
    for key in ('oom', 'oom_kill'):
        assert before['events'][key] == after['events'][key] == 0
    closure = load_ref(receipt['original_identity_closure'])
    assert closure['status'] == 'EXACT_ORIGINAL_IDENTITY_COVERAGE_RAM_AND_DURABLE_RUNTIME_CLOSURE_VERIFIED'
    assert closure['core_original_records'] == 55 and closure['evaluator_original_records'] == 25
    assert closure['direct_profile_input_SHA_count'] == 37
    expected = {}
    for stage in closure['stages']:
        for row in stage['files']:
            path = str(row['durable_path'])
            assert path not in expected or expected[path] == row
            expected[path] = row
    assert len(expected) == len(receipt['files']) == 80
    assert len({r['path'] for r in receipt['files']}) == 80
    assert {r['path'] for r in receipt['files']} == set(expected)
    assert receipt['actual_fadvise_calls'] >= 80
    return expected


def verify_and_advise_payloads(records, expected, durable):
    """SHA-read validation itself fills cache; release each verified FD again."""
    advised = []
    assert {r['path'] for r in records} == set(expected) and len(records) == len(expected)
    for row in records:
        path = Path(row['path']); old = expected[row['path']]
        assert path.is_absolute() and not path.is_symlink() and path.is_file()
        assert path.resolve(strict=True) == path and path.is_relative_to(durable)
        assert path != durable and not path.is_relative_to(Path('/dev/shm'))
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            before = os.fstat(descriptor)
            assert stat.S_ISREG(before.st_mode)
            assert (before.st_dev, before.st_ino, before.st_size) == (row['device'], row['inode'], row['bytes'])
            assert before.st_size == old['bytes']
            assert row['sha256_before'] == row['sha256_after'] == old['sha256']
            with os.fdopen(os.dup(descriptor), 'rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert actual == row['sha256_after']
            after = os.fstat(descriptor)
            assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            advised.append(dict(path=str(path), sha256=actual, bytes=before.st_size,
                device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns))
        finally:
            os.close(descriptor)
    return advised


def archive_failed_output(source, destination, records):
    assert source.is_dir() and not source.is_symlink() and not destination.exists()
    assert destination.parent.is_dir() and source.stat().st_dev == destination.parent.stat().st_dev
    assert source.resolve(strict=True) == source.absolute() and destination.parent.resolve(strict=True) == destination.parent.absolute()
    files = list(source.iterdir())
    assert {Path(r['path']) for r in records} == set(files)
    assert len(files) == 4 and sum(p.stat().st_size for p in files) <= SMALL_FAILED_CAP
    for record in records:
        check_ref(record)
    source.rename(destination)  # Atomic same-filesystem rename; no file deletion.
    moved = []
    for record in records:
        path = destination / Path(record['path']).name
        assert path.is_file() and C.sha(path) == record['sha256']
        moved.append(dict(original_path=record['path'], retained_archive=P.ref(path), bytes=path.stat().st_size))
    assert not source.exists()
    return moved


def advise_regular_closed_payloads(paths, roots):
    """Evict clean pages using closed-arm evidence, without reading image bytes."""
    roots = [Path(root).resolve(strict=True) for root in roots]
    assert len(paths) == len({str(path) for path in paths})
    identity = hashlib.sha256(); total = 0
    for path in paths:
        path = Path(path)
        assert path.is_absolute() and path.is_file() and not path.is_symlink()
        resolved = path.resolve(strict=True)
        assert resolved == path and any(resolved != root and resolved.is_relative_to(root) for root in roots)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            before = os.fstat(descriptor)
            assert stat.S_ISREG(before.st_mode)
            facts = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            after = os.fstat(descriptor)
            assert facts == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            identity.update(C.canonical(dict(path=str(path), facts=facts)))
            total += before.st_size
        finally:
            os.close(descriptor)
    return dict(files=len(paths), bytes=total, actual_fadvise_calls=len(paths),
        unchanged_stat_identity_SHA256=identity.hexdigest(), file_contents_read=False,
        deletion_performed=False, bytes_modified=False)


def closed_arm_payloads(item, recipe, entry):
    """Admit only a fully generated, audited, frozen-feature-evaluated past arm."""
    import audit_BrainHIVE_functional_full_V_artifacts_CPU_v1 as A
    assert item['arm'] == recipe['arm'] == entry['arm']
    generated, evaluated = load_ref(item['generated']), load_ref(item['evaluation'])
    assert generated['status'] == Q.GENERATE_DONE and generated['queries'] == 16600
    assert evaluated['status'] == 'FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED'
    for value in (generated, evaluated):
        assert value['arm'] == recipe['arm'] and value['recipe_core_sha256'] == C.core_sha(recipe)
    assert evaluated['queries'] == 16600 and evaluated['actual_full_V16600_evaluation_completed'] is True
    assert evaluated['full_identity_target_seed_noise_chain_verified'] is True
    assert evaluated['complete_low_level_mechanical_coverage'] is True
    assert evaluated['image_sampling'] is False and evaluated['GPU_used'] is False
    context = A.prepare(CONFIG / (recipe['arm'] + '.json'), time.monotonic() + 300)
    assert context['entry'] == entry and context['count'] == 16600
    joined = [json.loads(line) for line in context['manifest'].read_text().splitlines()]
    inference = [json.loads(line) for line in context['inference'].read_text().splitlines()]
    assert len(joined) == len(inference) == 16600
    paths = []
    for index, (before, row) in enumerate(zip(inference, joined)):
        issues, path_ok, identity, path = A.identity_issues(context, before, row, index)
        assert not issues and path_ok
        paths.append(path)
    assert len(set(paths)) == 16600
    features = Path(entry['feature_root'])
    for key, expected in (('features_config', features / 'config.json'),
            ('features_completion', features / 'completion.json'), ('manifest', context['manifest'])):
        assert Path(evaluated[key]['path']) == expected
        check_ref(evaluated[key])
    feature_done = P.read(features / 'completion.json')
    assert feature_done['status'] == 'COMPLETED' and feature_done['queries'] == 16600
    assert feature_done['all_queries_used'] is True and feature_done['query_subsampling'] is False
    assert feature_done['manifest_sha256'] == C.sha(context['manifest'])
    stages = ('DINO', 'EffNet', 'CapSim', 'ObjectF1')
    expected_features = {str(features / stage / (str(start).zfill(8) + '.pt'))
        for stage in stages for start in range(0, 16600, 256)}
    refs = evaluated['feature_shards']
    assert len(refs) == len(expected_features) and {ref['path'] for ref in refs} == expected_features
    assert {str(p) for stage in stages for p in (features / stage).glob('*.pt')} == expected_features
    # The actual evaluation closure already binds every feature/image SHA. No
    # payload is read again here merely to evict its clean filesystem pages.
    feature_paths = [Path(ref['path']) for ref in refs]
    return paths, feature_paths, [Path(part['root']) for part in entry['image_segments']], features


def pre_fit_memory(runner, arm, recipes, entries):
    P.gpu_free()  # Closed previous arms and no active sampler/feature job only.
    before = C.resource_headroom()
    released = []
    for item in runner.state['completed']:
        previous = item['arm']
        assert list(C.ARMS).index(previous) < list(C.ARMS).index(arm)
        images, features, image_roots, feature_root = closed_arm_payloads(item, recipes[previous], entries[previous])
        released.append(dict(arm=previous, generated=item['generated'], evaluation=item['evaluation'],
            images=advise_regular_closed_payloads(images, image_roots),
            features=advise_regular_closed_payloads(features, [feature_root])))
    after = C.resource_headroom()
    report = dict(next_fit_arm=arm, before=before, after=after, closed_arms_only=released,
        minimum_fresh_effective_headroom_bytes=MIN_RESTART_HEADROOM,
        original_formal_minimum_headroom_bytes=ORIGINAL_HEADROOM,
        threshold_changed=False, image_or_feature_bytes_read=False, deletion_performed=False)
    runner.state.setdefault('pre_fit_memory_reviews', []).append(report); runner.save()
    assert after['effective_headroom_bytes'] >= MIN_RESTART_HEADROOM, 'Less than24GiB before fit; stop without starting another uncheckpointed run'
    return report


class RecoveryRunner(Q.Runner):
    def restart_O(self, recipe):
        name, seconds, entry, arguments = second_slot_restart_command(recipe)
        assert not Path(recipe['output_directory']).exists()
        assert not (ROOT / 'audit' / (name + '_job.json')).exists()
        self.state.update(O_restart_kind='from_scratch_same_seed_no_checkpoint',
            O_restart_exact_resume=False, O_restart_registered_slot=2, O_additional_fit_attempts=1)
        self.save()
        return self.run(name, entry, arguments, seconds, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', required=True, type=Path)
    args = parser.parse_args()
    P.assert_host()
    import fcntl
    with (BASE / 'lifecycle_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not STATE.exists() and not ARCHIVE.exists(), 'One diagnosed restart only; retain every previous recovery'
        preparation, plan, recipes = Q.recipe_set()
        authorization = P.read(args.authorization)
        validate_authorization(authorization, recipes['O'])
        auth_pin = P.ref(args.authorization)
        receipt = load_ref(authorization['cache_release_receipt'])
        assert authorization['failure_evidence'] == receipt['failure_evidence']
        failure = verify_failure(recipes['O'], receipt['failure_evidence'], CONFIG / 'O.json')
        assert failure['original_charged_seconds'] + recipes['O']['training_job_seconds'] <= recipes['O']['maximum_cumulative_training_seconds_per_arm']
        expected_payloads = cache_records(receipt)
        validation_advice = verify_and_advise_payloads(receipt['files'], expected_payloads, DURABLE)
        closure = P.read(BASE / 'formal_admission_closure.json')
        assert closure['status'] == 'FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMISSION_ACTUALLY_VERIFIED'
        assert closure['plan'] == preparation['plan']
        for item in closure['recipes']:
            check_ref(item)
        for arm, recipe in recipes.items():
            admitted = C.verify_admission(recipe, 'formal')
            limits = P.read(recipe['admission_receipt'])['resource_limits']
            assert limits['minimum_cgroup_headroom_bytes'] == ORIGINAL_HEADROOM
            assert admitted['memory']['effective_headroom_bytes'] >= MIN_RESTART_HEADROOM
            if arm != 'O':
                assert not Path(recipe['output_directory']).exists()
                assert not any((ROOT / 'audit' / (name + '_job.json')).exists() for name in recipe['formal_job_names'])
            assert not (WORK / 'images' / C.run_name(arm)).exists()
            assert not (WORK / 'audit/functional_postfit_T32_v1' / arm).exists()
        ready = closure['evaluation']['readiness']
        check_ref(ready); P.evaluation_ready(ready['path'], plan)
        P.gpu_free()
        fresh_memory = C.resource_headroom()
        assert fresh_memory['effective_headroom_bytes'] >= MIN_RESTART_HEADROOM
        check_ref(auth_pin); check_ref(authorization['cache_release_receipt'])
        # Preserve original failure proofs and recipe bytes before the rename.
        runner = RecoveryRunner(STATE, 'formal', preparation)
        runner.state.update(status='RUNNING_DIAGNOSED_SINGLE_O_RESTART_AND_FIXED_FULL_LIFECYCLE',
            recovery_controller_source_sha256=C.sha(__file__), authorization=auth_pin,
            cache_release_receipt=authorization['cache_release_receipt'], failure=failure,
            pagecache_validation_fadvise_calls=len(validation_advice), verified_durable_payloads=validation_advice,
            fresh_restart_memory=fresh_memory, original_formal_minimum_headroom_bytes=ORIGINAL_HEADROOM,
            minimum_restart_headroom_bytes=MIN_RESTART_HEADROOM, original_failure_not_erased=True,
            seed_changed=False, recipe_core_changed=False, frozen_sources_changed=False,
            O_restart_is_exact_resume=False, O_failure_retry_allowed=False)
        runner.save()
        try:
            archive = archive_failed_output(Path(recipes['O']['output_directory']), ARCHIVE, failure['output_files'])
            runner.state['failed_O_archive'] = archive; runner.save()
            # All historical metadata remains exactly at its original path.
            for key in ('formal_queue', 'fit_job', 'fit_log'):
                check_ref(receipt['failure_evidence'][key])
            entries = {entry['arm']: entry for entry in plan['arms']}
            for arm, recipe in recipes.items():
                check_ref(auth_pin)
                for item in closure['recipes']:
                    check_ref(item)
                pre_fit_memory(runner, arm, recipes, entries)
                C.verify_admission(recipe, 'formal')
                runner.state['formal_training_started'] = True; runner.save()
                if arm == 'O':
                    code = runner.restart_O(recipe)
                    assert code == 0, 'The one O restart failed; no third slot or automatic retry'
                else:
                    code = runner.gpu(recipe, 'train')
                    if Q.training_can_resume(recipe, code):
                        code = runner.gpu(recipe, 'train', 1)
                    assert code == 0, 'Other-arm training failed beyond the original exact-resume policy'
                core = C.core_sha(recipe)
                final = Q.exact_completion(Path(recipe['output_directory']) / 'completion.json', Q.TRAIN_DONE, arm=arm, core=core)
                postfit = P.postfit(arm)
                runner.state['current_postfit_wrapper'] = P.ref(postfit['path']); runner.save()
                assert runner.gpu(recipe, 'postfit') == 0
                Q.exact_completion(WORK / 'audit/functional_postfit_T32_v1' / arm / 'completion.json', Q.POSTFIT_DONE, arm=arm, count=32)
                code = runner.gpu(recipe, 'generate')
                if Q.generation_can_resume(recipe, code):
                    code = runner.gpu(recipe, 'generate', 1)
                assert code == 0, 'Generation failed beyond the original exact-prefix policy'
                generated = Q.exact_completion(Path(entries[arm]['metadata_root']) / 'completion.json', Q.GENERATE_DONE, arm=arm, core=core, count=16600)
                prefix = 'BrainHIVE_functional_v2_' + arm + '_expanded_lifecycle_' + P.TAG
                runner.cpu(prefix + '_artifact', 'audit_BrainHIVE_functional_full_V_artifacts_CPU_v1.py',
                    ['--recipe', str(CONFIG / (arm + '.json')), '--output', entries[arm]['artifact_audit_root']], 1800)
                artifact = P.read(Path(entries[arm]['artifact_audit_root']) / 'summary.json')
                assert artifact['status'] == 'COMPLETE_FULL_V_ARTIFACT_AUDIT' and artifact['actual_full_V16600_verified'] is True
                assert artifact['complete_technical_coverage'] is True and artifact['full_denominator'] == 16600
                assert not artifact['failures'] and not artifact['source_errors']
                assert runner.gpu(recipe, 'features') == 0
                feature = P.read(Path(entries[arm]['feature_root']) / 'completion.json')
                assert feature['status'] == 'COMPLETED' and feature['queries'] == 16600 and feature['all_queries_used'] is True
                runner.cpu(prefix + '_evaluation', 'run_BrainHIVE_functional_evaluation_CPU_v1.py',
                    ['--recipe', str(CONFIG / (arm + '.json'))], 14400)
                evaluation = Q.exact_completion(Path(entries[arm]['metadata_root']) / 'evaluation_closure.json',
                    'FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED', arm=arm, core=core, count=16600)
                runner.state['completed'].append(dict(arm=arm, final=final, generated=generated, evaluation=evaluation)); runner.save()
            assert [row['arm'] for row in runner.state['completed']] == list(C.ARMS)
            runner.cpu('BrainHIVE_functional_v2_expanded_lifecycle_' + P.TAG + '_fixed_comparison',
                'compare_BrainHIVE_functional_V_CPU_v2.py', ['--recipe', str(CONFIG / 'PN.json')], 1800)
            comparison = Q.exact_completion(Path(plan['comparison']['output_root']) / 'summary.json',
                'ALL_FOUR_FULL16600_V_PAIRED_COMPARISONS_COMPLETE')
            runner.state.update(status='ALL_FOUR_FULL_LIFECYCLES_COMPLETE_AFTER_DOCUMENTED_SINGLE_O_RESTART',
                comparison=comparison, current_stage=None, scientific_goal_complete=False,
                independent_mechanism_confirmation=False, independent_SOTA_established=False)
            runner.save()
        except BaseException as error:
            runner.state.update(status=('RECOVERY_STOP_REQUESTED_WRAPPER_STILL_LIVE' if runner.state.get('unresolved_live_child')
                else 'RECOVERY_FAILED_PRESERVED_NO_FURTHER_AUTOMATIC_RETRY'), error=repr(error))
            runner.save()
            raise


if __name__ == '__main__':
    main()
