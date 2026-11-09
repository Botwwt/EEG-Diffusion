"""Portable synthetic FD/copy tests; no Torch, GPU, SSH or live-pipeline reads."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import archive_X01_checkpoints_CPU_20261028_v1 as A


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    root = Path(__file__).resolve().parent; checks = []
    def rejected(function, kind=AssertionError):
        try:
            function()
        except kind:
            return
        raise AssertionError('Expected rejection did not occur')
    def content(path, byte):
        with Path(path).open('wb') as stream:
            for _ in range(48):
                stream.write(bytes([byte]) * (64 << 10))
            stream.flush(); os.fsync(stream.fileno())
        return A.reference(path)
    assert 'torch' not in sys.modules
    with tempfile.TemporaryDirectory(prefix='X01_archiver_CPU_', dir=root) as name:
        work = Path(name).resolve(); assert work.is_relative_to(root)
        output = work / 'archives'; output.mkdir()
        source = work / 'resume.pt'; expected = content(source, 31)
        destination = output / 'checkpoints/audited_001.pt'
        result = A.stable_copy(source, expected['sha256'], expected['bytes'], destination, output)
        assert result['status'] == 'ARCHIVED_EXACT_CPU_AUDITED_CHECKPOINT_BYTES'
        assert A.sha(destination) == A.sha(source) == expected['sha256'] and not os.path.samefile(source, destination)
        checks.append('Exact audited FD SHA copies to independently allocated file with destination readback')
        newer = content(source, 47)
        result = A.stable_copy(source, expected['sha256'], expected['bytes'], output / 'late.pt', output)
        assert result['status'] == 'MISSED_AUDITED_SHA_NO_LONGER_AT_ROLLING_PATH'
        assert result['observed_current_checkpoint']['sha256'] == newer['sha256'] and not (output / 'late.pt').exists()
        checks.append('Late capture records actual current SHA and never fabricates old bytes')
        expected = content(source, 61); replacement = work / 'replacement.pt'; next_value = content(replacement, 73)
        def advance_path():
            if os.name == 'nt':
                # MoveFileEx cannot replace an open destination on this host.
                # Sharing deletion still permits retiring the old name; retain
                # it, then bind the new pathname while the original FD lives.
                os.rename(source, work / 'retired_open_checkpoint.pt')
            os.replace(replacement, source)
        result = A.stable_copy(source, expected['sha256'], expected['bytes'], output / 'renamed.pt', output,
            before_copy=advance_path)
        assert result['status'] == 'ARCHIVED_EXACT_CPU_AUDITED_CHECKPOINT_BYTES'
        assert A.sha(output / 'renamed.pt') == expected['sha256'] and A.sha(source) == next_value['sha256']
        assert result['observed_current_checkpoint']['sha256'] == next_value['sha256']
        checks.append('One open FD preserves old audited bytes across pathname advancement; POSIX atomic replace, Windows two-rename simulation')
        expected = content(source, 83)
        result = A.stable_copy(source, expected['sha256'], expected['bytes'], output / 'mutated.pt', output,
            before_copy=lambda: content(source, 97))
        assert result['status'] == 'MISSED_SOURCE_BYTES_CHANGED_DURING_COPY' and not (output / 'mutated.pt').exists()
        assert Path(result['retained_rejected_partial']['path']).is_file()
        checks.append('In-place source mutation between passes retains rejected partial and yields no audited archive')
        expected = A.reference(source); original = A.MAX_OUTPUT
        try:
            A.MAX_OUTPUT = 1 << 20
            result = A.stable_copy(source, expected['sha256'], expected['bytes'], output / 'too_large.pt', output)
            assert result['status'] == 'NOT_ARCHIVED_2GIB_OUTPUT_CAP' and not (output / 'too_large.pt').exists()
        finally:
            A.MAX_OUTPUT = original
        checks.append('Output cap rejects copy before allocating destination')
        result = A.stable_copy(work / 'absent.pt', 'a' * 64, 1024, output / 'absent.pt', output)
        assert result['status'] == 'MISSED_SOURCE_MISSING_AT_CAPTURE'
        checks.append('Missing source is explicitly missed without creating a checkpoint')
        old = A.sha(destination)
        rejected(lambda: A.stable_copy(source, expected['sha256'], expected['bytes'], destination, output))
        assert A.sha(destination) == old
        checks.append('Existing permanent archive is never overwritten')
        pending = output / 'publication.pending'; pending.write_text('new')
        final = output / 'publication.json'; final.write_text('old')
        rejected(lambda: A.publish_no_overwrite(pending, final), FileExistsError)
        assert pending.read_text() == 'new' and final.read_text() == 'old'
        checks.append('Atomic publication fails without replacing an existing destination')
        fake_proc = work / 'proc'; proc = fake_proc / '240115'; proc.mkdir(parents=True)
        script = root / 'run_ATMS_standard_ordinary_pipeline_CPU_20261028_v1.py'; launch = work / 'launch.json'
        argv = ['python', str(script.resolve()), '--launch-review', str(launch.resolve())]
        (proc / 'stat').write_text('240115 (python (fd test)) ' + ' '.join(['S'] + ['0'] * 18 + ['82331630']))
        (proc / 'cmdline').write_bytes(b'\0'.join(s.encode() for s in argv) + b'\0')
        matched = A.process_sample(240115, 82331630, script, launch, argv, fake_proc)
        assert matched['status'] == 'EXPECTED_PIPELINE_IDENTITY_MATCH'
        assert A.process_sample(240115, 999, script, launch, argv, fake_proc)['status'] == 'PIPELINE_IDENTITY_MISMATCH_STOP_NO_FOLLOW'
        assert A.process_sample(240115, 82331630, script, launch, argv + ['changed'], fake_proc)['status'] == 'PIPELINE_IDENTITY_MISMATCH_STOP_NO_FOLLOW'
        assert A.process_sample(999, 82331630, script, launch, argv, fake_proc)['status'] == 'EXPECTED_PIPELINE_PROCESS_ABSENT'
        checks.append('Synthetic Linux proc parser handles parentheses and guards PID start_ticks, exact argv, and absence')
        original_read_text = Path.read_text; stat_path = (proc / 'stat').resolve(); reads = [0]
        def changing_stat(path, *args, **kwargs):
            if path.resolve() == stat_path:
                reads[0] += 1
                return '240115 (python) ' + ' '.join(['S' if reads[0] == 1 else 'Z'] + ['0'] * 18 + ['82331630'])
            return original_read_text(path, *args, **kwargs)
        try:
            Path.read_text = changing_stat
            assert A.process_sample(240115, 82331630, script, launch, argv, fake_proc)['status'] == 'EXPECTED_PIPELINE_PROCESS_ABSENT'
        finally:
            Path.read_text = original_read_text
        checks.append('Synthetic S-to-Z transition between the two proc reads is an expected exit, never a reused PID')
        audit = work / 'checkpoint_audit_001.json'
        document = dict(status='REAL_ATMS_ORDINARY_COMMITTED_CHECKPOINT_CPU_VERIFIED', checkpoint=A.reference(source),
            source_sha256='b' * 64, CUDA_initialized=False, model_optimizer_RNG_order_coverage_log_prefix_verified=True,
            updates=512, seen=8192, previous_updates=0)
        audit.write_text(json.dumps(document)); A.audit_document(audit, source, 'b' * 64)
        assert A.expected_seen(8269) == 132300 and A.expected_seen(24807) == 396900
        audit.write_text(json.dumps(document | dict(seen=8191)))
        rejected(lambda: A.audit_document(audit, source, 'b' * 64))
        checks.append('Audit schema/source/checkpoint identity and pass-end exposure arithmetic are enforced')
        original_sample = A.process_sample
        def full_run_fixture(label, failure=False, deadline=False):
            fixture = work / label; fixture.mkdir(); audits = fixture / 'audit'; audits.mkdir()
            (fixture / 'formal_train').mkdir()
            live_source = fixture / 'formal_train/resume.pt'; archived_old = content(live_source, 109)
            fake_pipeline = fixture / 'pipeline.py'; fake_pipeline.write_text('# Synthetic fixture only\n')
            fake_launch = fixture / 'launch.json'
            fake_launch.write_text(json.dumps(dict(status='ATMS_STANDARD_ORDINARY_FINITE_PIPELINE_REVIEWED',
                source_pins=[A.reference(fake_pipeline)])))
            fake_spec = fixture / 'spec.json'; fake_spec.write_text('{"synthetic":true}')
            first = document | dict(checkpoint=archived_old, source_sha256=A.sha(fake_pipeline), UTC=A.utc())
            if not deadline:
                (audits / 'checkpoint_audit_001.json').write_text(json.dumps(first))
            latest = content(live_source, 127)
            if not failure and not deadline:
                second = first | dict(checkpoint=latest, updates=1024, seen=16384, previous_updates=512)
                (audits / 'checkpoint_audit_002.json').write_text(json.dumps(second))
                (audits / 'actual_image_screen_completion.json').write_text(json.dumps(dict(
                    status='X01_MATCHED_ATMS_NATIVE_ORDINARY_ACTUAL_IMAGE_SCREEN_CLOSED', UTC=A.utc())))
            elif failure:
                (audits / 'pipeline_failure_fixture.json').write_text(json.dumps(dict(
                    status='X01_PIPELINE_STOPPED_FAILURE_AND_CHECKPOINTS_PRESERVED_NO_AUTOMATIC_RETRY', UTC=A.utc())))
            calls = [0]; expected_argv = ['python', str(fake_pipeline.resolve()), '--launch-review', str(fake_launch.resolve())]
            def fake_process(*_args, **_kwargs):
                calls[0] += 1
                if deadline or calls[0] <= 2:
                    return dict(status='EXPECTED_PIPELINE_IDENTITY_MATCH', pid=240115, start_ticks=82331630,
                        argv=expected_argv, start_epoch=time.time() - 100, UTC=A.utc())
                return dict(status='EXPECTED_PIPELINE_PROCESS_ABSENT', pid=240115, UTC=A.utc())
            A.process_sample = fake_process
            arguments = SimpleNamespace(pipeline_pid=240115, pipeline_start_ticks=82331630,
                pipeline_script=fake_pipeline, launch_review=fake_launch, audit_dir=audits,
                checkpoint=live_source, spec=fake_spec, output=fixture / 'archives',
                max_seconds=1 if deadline else 5, resume=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result = A.run(arguments)
            completions = sorted((arguments.output / 'runs').glob('*/completion.json'))
            summary, _ = A.small_json(completions[-1])
            assert A.sha(live_source) == latest['sha256']
            return result, summary, arguments
        try:
            code, summary, arguments = full_run_fixture('full_success')
            assert code == 0 and summary['archived_serials'] == [2] and summary['missed_serials'] == [1]
            assert summary['terminal_last_pointer']['status'] == 'TERMINAL_POINTER_ALREADY_HAS_MATCHING_CPU_AUDITED_ARCHIVE'
            checks.append('Synthetic full archiver loop records historical MISS, captures current audited slot, and closes only after process absence plus terminal marker')
            arguments.resume = True; original_spec = arguments.spec.read_text()
            arguments.spec.write_text('{"synthetic":"changed"}')
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    A.run(arguments)
            except AssertionError as exc:
                assert 'SHA changed' in str(exc)
            else:
                raise AssertionError('Resumed archive accepted changed prior spec')
            arguments.spec.write_text(original_spec)
            checks.append('Explicit archiver resume revalidates previous records training-spec SHA and rejects drift')
            with contextlib.redirect_stdout(io.StringIO()):
                assert A.run(arguments) == 0
            checks.append('Unchanged explicit resume retains existing permanent files and closes without duplicate archives')
            code, summary, _ = full_run_fixture('full_failure', failure=True)
            assert code == 0 and summary['archived_serials'] == [] and summary['missed_serials'] == [1]
            assert summary['terminal_last_pointer']['status'] == 'UNREVIEWED_TERMINAL_CHECKPOINT_BYTES_PRESERVED'
            assert summary['terminal_last_pointer']['never_count_as_audited_slot']
            assert summary['terminal_last_pointer']['capture']['status'] == 'UNREVIEWED_STABLE_FD_CHECKPOINT_BYTES_COPIED'
            assert summary['terminal_last_pointer']['capture']['expected_SHA_origin'] == 'UNREVIEWED_CURRENT_FILE_OBSERVATION'
            checks.append('Synthetic pipeline failure preserves last unaudited bytes separately and never relabels them CPU-audited')
            code, summary, _ = full_run_fixture('finite_deadline', deadline=True)
            assert code == 2 and summary['status'] == 'ARCHIVER_FINITE_DEADLINE_REACHED'
            assert summary['terminal_last_pointer'] is None
            checks.append('Finite deadline exits without following or controlling still-live pipeline')
        finally:
            A.process_sample = original_sample
        assert A.bytes_used(output) < A.MAX_OUTPUT and A.memory_check() <= A.MAX_RAM
        checks.append('Actual copy tests remain below32MiB peak process RSS and bounded archive allocation')
    assert 'torch' not in sys.modules
    report = dict(status='X01_ARCHIVER_PORTABLE_SYNTHETIC_CPU_COPY_TESTS_PASSED', checks=len(checks), scope=checks,
        source=A.reference(A.__file__), test_source=A.reference(__file__), actual_platform=sys.platform,
        actual_POSIX_atomic_os_replace_with_open_FD_tested=os.name != 'nt',
        actual_Windows_two_rename_FD_rebinding_tested=os.name == 'nt', synthetic_Linux_proc_identity_checked=True,
        actual_Linux_execution_claimed=sys.platform.startswith('linux'), Torch_imported=False,
        GPU_or_SSH_used=False, real_pipeline_or_checkpoint_read=False,
        stream_chunk_bytes=A.CHUNK, maximum_peak_process_RSS_bytes=A.MAX_RAM,
        measured_peak_process_RSS_bytes=A.peak_rss(), archive_copy_semantics_only=True)
    A.atomic(args.output, report); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
