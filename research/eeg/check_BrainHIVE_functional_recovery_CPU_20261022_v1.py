"""CPU fixtures for the single explicitly diagnosed O recovery, never GPU work.

Fixtures contain four tiny metadata files and fake job receipts, never model
tensors. No private dataset, real cache eviction, live process, server, original
experiment file or frozen source is changed.
"""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import queue_BrainHIVE_functional_recovery_20261022_v1 as R


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def bound(path):
    return dict(path=str(path), sha256=R.C.sha(path), bytes=path.stat().st_size)


class RecoveryTests(unittest.TestCase):
    def reject(self, action):
        with self.assertRaises((AssertionError, KeyError, OSError)):
            action()

    def failure_fixture(self, base):
        repo, work = base / 'repo', base / 'work'
        lifecycle = repo / 'audit/lifecycle'; lifecycle.mkdir(parents=True)
        config = repo / 'configs/O.json'
        original = R.P.read(R.CONFIG / 'O.json')
        recipe = copy.deepcopy(original)
        out = work / 'components/functional_alignment_v2/O'; out.mkdir(parents=True)
        recipe['output_directory'] = str(out)
        write(config, recipe)
        name = recipe['formal_job_names'][0]
        job_path = repo / 'audit' / (name + '_job.json')
        job = dict(name=name, pid=3551, wrapper_pid=3550, status='FAILED_OR_JOB_TIMEOUT_RESUMABLE',
            exit_code=1, charged_seconds=178.92947624402586,
            command=['SYNTHETIC_PYTHON', str(repo / 'train_BrainHIVE_functional_alignment_v2.py'), '--recipe', str(config)])
        write(job_path, job)
        log = lifecycle / (name + '.log')
        log.write_text('CPU SYNTHETIC TRACEBACK ONLY\n'
            '  File "brainhive_functional_training_contract_v2.py", line 442, in verify_admission\n'
            "    assert memory['effective_headroom_bytes'] >= limits['minimum_cgroup_headroom_bytes']\n"
            'snapshot()\nAssertionError\n')
        queue = dict(status='FORMAL_LIFECYCLE_STOPPED_WITH_PRESERVED_EVIDENCE_NO_AUTOMATIC_RETRY',
            pid=3539, completed=[], current_stage=name,
            attempts=[dict(name=name, exit_code=1, log=str(log), log_sha256=R.C.sha(log))])
        queue_path = lifecycle / 'formal_queue.json'; write(queue_path, queue)
        provenance = dict(arm='O', candidate=recipe['candidate'], recipe_core_sha256=R.C.core_sha(recipe),
            source_sha256=recipe['source_sha256'], input_contract_sha256=recipe['input_contract_sha256'],
            risk_head_or_gate=False, all_frozen_host_weights=True)
        launch = dict(provenance, recipe_sha256=R.C.sha(config), admission_sha256=recipe['admission_sha256'],
            exclusive_GPU=dict(wrapper_pid=3550, wrapper_job_name=name))
        write(out / 'input_provenance.json', provenance); write(out / 'launch_3551.json', launch)
        event = dict(stage='BRAINHIVE_FUNCTIONAL_ALIGNMENT_V2_REAL_OOF_TRAINING', arm='O', pass_number=1,
            updates=64, expected_updates=24807, seen_queries=1024, expected_queries=396900,
            cumulative_mean_loss=.2, last_loss=.2, epsilon_loss=.18, token_loss=.2, functional_auxiliary=0.,
            functional_diagnostics={}, UNet_forwards=2, gradient_norm=.01, correction_RMS=.03, mean_gate=1., seconds=98.)
        (out / 'training.jsonl').write_bytes((json.dumps(event) + '\n').encode())
        write(out / 'progress.json', dict(status='FUNCTIONAL_ALIGNMENT_V2_RUNNING', **event))
        evidence = dict(formal_queue=bound(queue_path), fit_job=bound(job_path), fit_log=bound(log),
            output_files=[bound(p) for p in sorted(out.iterdir())])
        return dict(repo=repo, work=work, lifecycle=lifecycle, recipe=recipe, config=config,
            job_path=job_path, job=job, log=log, queue_path=queue_path, queue=queue, output=out, evidence=evidence)

    def validate_fixture(self, f):
        with patch.object(R, 'ROOT', f['repo']), patch.object(R, 'WORK', f['work']), patch.object(R, 'BASE', f['lifecycle']), patch.object(R.C, 'process_live', lambda pid: False):
            return R.verify_failure(f['recipe'], f['evidence'], f['config'])

    def test_actual_second_registered_slot_is_fresh_same_recipe_without_resume(self):
        recipe = R.P.read(R.CONFIG / 'O.json')
        before = copy.deepcopy(recipe)
        name, seconds, entry, args = R.second_slot_restart_command(recipe)
        self.assertEqual(name, recipe['formal_job_names'][1])
        self.assertEqual(seconds, 86400)
        self.assertEqual(entry, 'train_BrainHIVE_functional_alignment_v2.py')
        self.assertNotIn('--resume', args)
        self.assertEqual(recipe, before)
        self.assertEqual(recipe['training_seed'], 1729)
        self.assertEqual(recipe['expected_updates'], 24807)

    def test_other_arm_cannot_be_relabelled_as_diagnosed_O_restart(self):
        recipe = R.P.read(R.CONFIG / 'PN.json')
        self.reject(lambda: R.second_slot_restart_command(recipe))

    def test_positive_failed_metadata_join_and_atomic_archive_keep_all_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.failure_fixture(Path(directory)); result = self.validate_fixture(f)
            self.assertEqual(result['last_logged_updates'], 64)
            self.assertIn('UNKNOWN', result['actual_uncheckpointed_update_count'])
            archive = f['lifecycle'] / 'preserved_failed_O'
            before = {p.name: R.C.sha(p) for p in f['output'].iterdir()}
            old_queue = R.C.sha(f['queue_path']); old_log = R.C.sha(f['log'])
            mapping = R.archive_failed_output(f['output'], archive, f['evidence']['output_files'])
            self.assertEqual({p.name: R.C.sha(p) for p in archive.iterdir()}, before)
            self.assertEqual(len(mapping), 4)
            self.assertFalse(f['output'].exists())
            self.assertEqual(R.C.sha(f['queue_path']), old_queue)
            self.assertEqual(R.C.sha(f['log']), old_log)

    def test_checkpoint_or_extra_output_forbids_fresh_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.failure_fixture(Path(directory))
            (f['output'] / 'resume.pt').write_bytes(b'CPU_SYNTHETIC_NOT_A_MODEL')
            self.reject(lambda: self.validate_fixture(f))

    def test_different_failure_line_or_OOM_is_not_the_same_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.failure_fixture(Path(directory))
            text = f['log'].read_text().replace('line 442', 'line 443')
            f['log'].write_text(text)
            f['evidence']['fit_log'] = bound(f['log'])
            f['queue']['attempts'][0]['log_sha256'] = R.C.sha(f['log']); write(f['queue_path'], f['queue'])
            f['evidence']['formal_queue'] = bound(f['queue_path'])
            self.reject(lambda: self.validate_fixture(f))
            f['log'].write_text(text.replace('line 443', 'line 442') + '\nCUDA out of memory\n')
            f['evidence']['fit_log'] = bound(f['log'])
            f['queue']['attempts'][0]['log_sha256'] = R.C.sha(f['log']); write(f['queue_path'], f['queue'])
            f['evidence']['formal_queue'] = bound(f['queue_path'])
            self.reject(lambda: self.validate_fixture(f))

    def test_live_previous_fit_forbids_archiving(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.failure_fixture(Path(directory))
            with patch.object(R, 'ROOT', f['repo']), patch.object(R, 'WORK', f['work']), patch.object(R, 'BASE', f['lifecycle']), patch.object(R.C, 'process_live', lambda pid: pid == 3551):
                self.reject(lambda: R.verify_failure(f['recipe'], f['evidence'], f['config']))
            self.assertTrue(f['output'].exists())

    def test_failed_evidence_tampering_is_rejected_before_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.failure_fixture(Path(directory))
            (f['output'] / 'progress.json').write_bytes(b'{"status":"changed"}\n')
            self.reject(lambda: self.validate_fixture(f))

    def test_archive_refuses_existing_destination_and_foreign_record(self):
        with tempfile.TemporaryDirectory() as directory:
            f = self.failure_fixture(Path(directory)); archive = f['lifecycle'] / 'archive'; archive.mkdir()
            self.reject(lambda: R.archive_failed_output(f['output'], archive, f['evidence']['output_files']))
            archive.rmdir()
            records = copy.deepcopy(f['evidence']['output_files']); records[0]['path'] = str(f['log'])
            self.reject(lambda: R.archive_failed_output(f['output'], archive, records))

    def payload_fixture(self, base):
        durable = base / 'durable'; durable.mkdir()
        records, expected = [], {}
        for index in range(2):
            path = durable / (str(index) + '.SYNTHETIC_PUBLIC_PAYLOAD'); path.write_bytes(('fixture ' + str(index)).encode())
            info = path.stat(); digest = R.C.sha(path)
            records.append(dict(path=str(path), bytes=info.st_size, device=info.st_dev, inode=info.st_ino,
                sha256_before=digest, sha256_after=digest))
            expected[str(path)] = dict(bytes=info.st_size, sha256=digest)
        return durable, records, expected

    def test_payload_SHA_verification_releases_only_same_verified_regular_FDs(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            durable, records, expected = self.payload_fixture(Path(directory)); calls = []
            before = {p.name: R.C.sha(p) for p in durable.iterdir()}
            def advice(fd, offset, length, flag):
                calls.append((os.fstat(fd).st_ino, offset, length, flag))
            stack.enter_context(patch.object(os, 'posix_fadvise', advice, create=True))
            stack.enter_context(patch.object(os, 'POSIX_FADV_DONTNEED', 4, create=True))
            checked = R.verify_and_advise_payloads(records, expected, durable)
            self.assertEqual(len(calls), len(records))
            self.assertEqual({v[0] for v in calls}, {r['inode'] for r in records})
            self.assertTrue(all(row[1:] == (0, 0, 4) for row in calls))
            self.assertEqual({p.name: R.C.sha(p) for p in durable.iterdir()}, before)
            self.assertEqual(len(checked), 2)

    def test_payload_outside_scope_corrupt_bytes_or_wrong_inode_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(os, 'posix_fadvise', lambda *args: None, create=True), patch.object(os, 'POSIX_FADV_DONTNEED', 4, create=True):
            base = Path(directory); durable, records, expected = self.payload_fixture(base)
            wrong = copy.deepcopy(records); wrong[0]['inode'] += 1
            self.reject(lambda: R.verify_and_advise_payloads(wrong, expected, durable))
            self.reject(lambda: R.verify_and_advise_payloads(records, expected, durable / 'nested'))
            Path(records[0]['path']).write_bytes(b'changed!')
            self.reject(lambda: R.verify_and_advise_payloads(records, expected, durable))

    def cache_fixture(self, base):
        helper = base / R.CACHE_HELPER; helper.write_bytes(b'CPU_SYNTHETIC_ADVICE_SOURCE')
        closure = base / 'closure.json'
        files = [dict(durable_path=str(base / 'durable' / str(i)), bytes=1, sha256='a' * 64) for i in range(80)]
        write(closure, dict(status='EXACT_ORIGINAL_IDENTITY_COVERAGE_RAM_AND_DURABLE_RUNTIME_CLOSURE_VERIFIED',
            core_original_records=55, evaluator_original_records=25, direct_profile_input_SHA_count=37,
            stages=[dict(files=files)]))
        memory = dict(max=64 << 30, current=48 << 30, file=40 << 30, shmem=15 << 30,
            active_file=20 << 30, inactive_file=20 << 30, events=dict(oom=0, oom_kill=0))
        after = dict(memory, current=30 << 30, file=22 << 30)
        return dict(status='TARGETED_PRIVATE_RUNTIME_PAGECACHE_RELEASE_VERIFIED_NO_FILE_REMOVAL',
            deletion_performed=False, quantization_performed=False, GPU_used=False, headroom_threshold_changed=False,
            SHA_and_path_and_inode_unchanged=True, original_formal_minimum_headroom_bytes=8 << 30,
            minimum_restart_headroom_bytes=24 << 30, restart_headroom_satisfied=True,
            effective_headroom_after_bytes=34 << 30, issued_epoch=time.time(), source_sha256=R.C.sha(helper),
            cgroup_before=memory, cgroup_after=after, original_identity_closure=bound(closure),
            files=[dict(path=r['durable_path']) for r in files], actual_fadvise_calls=160)

    def test_cache_receipt_requires_exact80_coverage_unchanged_threshold_no_oom_and_fresh_time(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(R, 'ROOT', Path(directory)):
            receipt = self.cache_fixture(Path(directory))
            self.assertEqual(len(R.cache_records(receipt)), 80)
            bads = []
            for key, value in [('issued_epoch', time.time() - 1801), ('original_formal_minimum_headroom_bytes', 2 << 30),
                    ('effective_headroom_after_bytes', (24 << 30) - 1), ('deletion_performed', True)]:
                changed = copy.deepcopy(receipt); changed[key] = value; bads.append(changed)
            changed = copy.deepcopy(receipt); changed['files'].pop(); bads.append(changed)
            changed = copy.deepcopy(receipt); changed['cgroup_after']['events']['oom_kill'] = 1; bads.append(changed)
            for changed in bads:
                self.reject(lambda: R.cache_records(changed))

    def test_recovery_runner_is_original_runner_and_O_dispatches_once(self):
        self.assertTrue(issubclass(R.RecoveryRunner, R.Q.Runner))
        with tempfile.TemporaryDirectory() as directory, patch.object(R, 'ROOT', Path(directory)):
            recipe = R.P.read(R.CONFIG / 'O.json'); recipe['output_directory'] = str(Path(directory) / 'not_created')
            runner = object.__new__(R.RecoveryRunner); runner.state = {}; runner.save = lambda: None
            calls = []
            runner.run = lambda *args: calls.append(args) or 1
            self.assertEqual(runner.restart_O(recipe), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], recipe['formal_job_names'][1])
            self.assertNotIn('--resume', calls[0][2])
            self.assertFalse(runner.state['O_restart_exact_resume'])

    def test_postfit_fullV_audit_features_evaluation_comparison_remain_exact_native_entries(self):
        tree = ast.parse(Path(R.__file__).read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        stage_calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr in ('restart_O', 'gpu', 'cpu', 'postfit')]
        stage_calls.sort(key=lambda node: node.lineno)
        names = []
        for call in stage_calls:
            if call.func.attr == 'restart_O': names.append('restart_O')
            elif call.func.attr == 'postfit': names.append('prepare_postfit')
            elif call.func.attr == 'gpu': names.append(ast.literal_eval(call.args[1]))
            elif call.func.attr == 'cpu': names.append(ast.literal_eval(call.args[1]))
        self.assertEqual(names, ['restart_O', 'train', 'train', 'prepare_postfit', 'postfit', 'generate', 'generate',
            'audit_BrainHIVE_functional_full_V_artifacts_CPU_v1.py', 'features',
            'run_BrainHIVE_functional_evaluation_CPU_v1.py', 'compare_BrainHIVE_functional_V_CPU_v2.py'])
        text = Path(R.__file__).read_text()
        self.assertIn("RecoveryRunner(STATE, 'formal', preparation)", text)
        self.assertIn("lifecycle_controller.lock", text)
        self.assertIn("assert not STATE.exists() and not ARCHIVE.exists()", text)

    def test_closed_payload_eviction_reads_no_image_or_feature_content_and_keeps_stats(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, rows, expected = self.payload_fixture(Path(directory))
            paths = [Path(r['path']) for r in rows]; calls = []
            before = {p.name: p.read_bytes() for p in paths}
            stack.enter_context(patch.object(os, 'posix_fadvise', lambda *args: calls.append(args), create=True))
            stack.enter_context(patch.object(os, 'POSIX_FADV_DONTNEED', 4, create=True))
            stack.enter_context(patch.object(R.hashlib, 'file_digest', side_effect=AssertionError('Closed images must not be read to release their page cache')))
            result = R.advise_regular_closed_payloads(paths, [root])
            self.assertEqual(result['files'], 2)
            self.assertEqual(result['actual_fadvise_calls'], 2)
            self.assertFalse(result['file_contents_read'])
            self.assertFalse(result['bytes_modified'])
            self.assertEqual({p.name: p.read_bytes() for p in paths}, before)
            self.reject(lambda: R.advise_regular_closed_payloads(paths, [root / 'foreign_scope']))

    def test_each_arm_prefit_stops_at24GiB_without_relaxing_original8GiB(self):
        class Runner:
            state = dict(completed=[], formal_training_started=False)
            def save(self): pass
        runner = Runner()
        with patch.object(R.P, 'gpu_free', lambda: None), patch.object(R.C, 'resource_headroom',
                side_effect=[dict(effective_headroom_bytes=32 << 30), dict(effective_headroom_bytes=23 << 30)]):
            self.reject(lambda: R.pre_fit_memory(runner, 'PN', {}, {}))
        report = runner.state['pre_fit_memory_reviews'][0]
        self.assertEqual(report['minimum_fresh_effective_headroom_bytes'], 24 << 30)
        self.assertEqual(report['original_formal_minimum_headroom_bytes'], 8 << 30)
        self.assertFalse(runner.state['formal_training_started'])
        self.assertFalse(report['threshold_changed'])

    def test_prefit_eviction_targets_only_closed_earlier_arm(self):
        class Runner:
            def __init__(self): self.state = dict(completed=[dict(arm='O', generated={}, evaluation={})])
            def save(self): pass
        runner = Runner(); calls = []
        with patch.object(R.P, 'gpu_free', lambda: None), patch.object(R.C, 'resource_headroom',
                return_value=dict(effective_headroom_bytes=32 << 30)), patch.object(R, 'closed_arm_payloads',
                return_value=(['image'], ['feature'], ['image_scope'], 'feature_scope')), patch.object(R, 'advise_regular_closed_payloads',
                side_effect=lambda paths, roots: calls.append((paths, roots)) or dict(files=1, bytes=1, actual_fadvise_calls=1)):
            report = R.pre_fit_memory(runner, 'P', {'O': {}}, {'O': {}})
        self.assertEqual([item['arm'] for item in report['closed_arms_only']], ['O'])
        self.assertEqual(calls, [(['image'], ['image_scope']), (['feature'], ['feature_scope'])])
        runner.state['completed'] = [dict(arm='S')]
        with patch.object(R.P, 'gpu_free', lambda: None), patch.object(R.C, 'resource_headroom', return_value=dict(effective_headroom_bytes=32 << 30)):
            self.reject(lambda: R.pre_fit_memory(runner, 'P', {}, {}))

    def test_closed_arm_admission_requires_full16600_and_all260_feature_shards(self):
        import audit_BrainHIVE_functional_full_V_artifacts_CPU_v1 as A
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); metadata = base / 'metadata'; metadata.mkdir()
            feature = base / 'features'; feature.mkdir(); images = base / 'images'; images.mkdir()
            recipe = R.P.read(R.CONFIG / 'O.json'); core = R.C.core_sha(recipe)
            entry = dict(arm='O', feature_root=str(feature), image_segments=[dict(root=str(images))])
            manifest, inference = metadata / 'manifest.jsonl', metadata / 'inference_manifest.jsonl'
            raw = b'{}\n' * 16600; manifest.write_bytes(raw); inference.write_bytes(raw)
            write(feature / 'config.json', dict(CPU_SYNTHETIC_ONLY=True))
            write(feature / 'completion.json', dict(status='COMPLETED', queries=16600, all_queries_used=True,
                query_subsampling=False, manifest_sha256=R.C.sha(manifest)))
            refs = []
            for stage in ('DINO', 'EffNet', 'CapSim', 'ObjectF1'):
                folder = feature / stage; folder.mkdir()
                for start in range(0, 16600, 256):
                    path = folder / (str(start).zfill(8) + '.pt'); path.write_bytes(b'CPU_SYNTHETIC_NOT_TENSOR')
                    refs.append(bound(path))
            generated = metadata / 'completion.json'
            write(generated, dict(status=R.Q.GENERATE_DONE, arm='O', queries=16600, recipe_core_sha256=core))
            evaluated = metadata / 'evaluation_closure.json'
            closure = dict(status='FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED', arm='O',
                recipe_core_sha256=core, queries=16600, actual_full_V16600_evaluation_completed=True,
                full_identity_target_seed_noise_chain_verified=True, complete_low_level_mechanical_coverage=True,
                image_sampling=False, GPU_used=False, features_config=bound(feature / 'config.json'),
                features_completion=bound(feature / 'completion.json'), manifest=bound(manifest), feature_shards=refs)
            write(evaluated, closure)
            item = dict(arm='O', generated=bound(generated), evaluation=bound(evaluated))
            context = dict(entry=entry, count=16600, manifest=manifest, inference=inference)
            def identity(context, before, row, index): return [], True, str(index), images / (str(index) + '.webp')
            with patch.object(A, 'prepare', return_value=context), patch.object(A, 'identity_issues', identity):
                image_paths, feature_paths, roots, feature_root = R.closed_arm_payloads(item, recipe, entry)
            self.assertEqual(len(image_paths), 16600)
            self.assertEqual(len(feature_paths), 260)
            self.assertEqual(roots, [images])
            self.assertEqual(feature_root, feature)
            closure['actual_full_V16600_evaluation_completed'] = False; write(evaluated, closure)
            item['evaluation'] = bound(evaluated)
            self.reject(lambda: R.closed_arm_payloads(item, recipe, entry))


if __name__ == '__main__':
    unittest.main(verbosity=2)
