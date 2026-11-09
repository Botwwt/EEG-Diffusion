"""Finite CPU-only archiver for the already running, identity-bound X01 PID.

Audited slots are copied only when bytes read through one stable source FD
match the immutable CPU audit SHA. A missed rolling checkpoint stays MISSED.
An optional terminal last-pointer copy is explicitly UNREVIEWED, never an
audited slot. No Torch, GPU, training control, formal-file writes or deletion.
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
import traceback

SCHEMA = 'X01_CPU_stable_FD_checkpoint_archiver_20261028_v1'
CHUNK = 1 << 20
MAX_RAM = 32 << 20
MAX_OUTPUT = 2 << 30
MAX_SECONDS = 180000
AUDIT_LIMIT = 128
META_RESERVE = 4 << 20
AUDIT_PATTERN = re.compile(r'checkpoint_audit_(\d{3})\.json\Z')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def open_stable(path):
    """Share Windows rename/deletion; POSIX replacement keeps the old FD valid."""
    path = Path(path)
    assert not path.is_symlink(), 'Do not follow a checkpoint-file symlink'
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        import msvcrt
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        handle = create(str(path), 0x80000000, 1 | 2 | 4, None, 3, 0x08000000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            kernel.CloseHandle(handle)
            raise
    else:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    return os.fdopen(fd, 'rb', buffering=0)


def peak_rss():
    proc = Path('/proc/self/status')
    if proc.exists():
        for line in proc.read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) * 1024
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        size = ctypes.c_size_t
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', size), ('WorkingSetSize', size), ('QuotaPeakPagedPoolUsage', size),
                ('QuotaPagedPoolUsage', size), ('QuotaPeakNonPagedPoolUsage', size),
                ('QuotaNonPagedPoolUsage', size), ('PagefileUsage', size), ('PeakPagefileUsage', size)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        value = Counters(); value.cb = ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(value.PeakWorkingSetSize)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def memory_check():
    value = peak_rss()
    assert value <= MAX_RAM, f'Archiver peak RSS exceeds32MiB: {value}'
    return value


def hash_stream(stream):
    result = hashlib.sha256(); count = 0
    while True:
        block = stream.read(CHUNK)
        if not block:
            break
        result.update(block); count += len(block); memory_check()
    return result.hexdigest(), count


def sha(path):
    with open_stable(path) as stream:
        return hash_stream(stream)[0]


def reference(path):
    path = Path(path).resolve()
    with open_stable(path) as stream:
        digest, count = hash_stream(stream)
    return dict(path=str(path), sha256=digest, bytes=count)


def checked(item):
    path = Path(item['path']); actual = reference(path)
    assert actual['sha256'] == item['sha256'], 'SHA changed: ' + str(path)
    if 'bytes' in item:
        assert actual['bytes'] == item['bytes']
    return path


def small_json(path, maximum=1 << 20):
    with open_stable(path) as stream:
        raw = stream.read(maximum + 1)
    assert len(raw) <= maximum, 'Metadata size limit'
    return json.loads(raw), dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def publish_no_overwrite(temporary, destination):
    """The hard link is between two names of our new copy, never the source."""
    assert Path(temporary).parent.resolve() == Path(destination).parent.resolve()
    os.link(temporary, destination)  # Atomic fail-if-exists on both NTFS and Linux.
    Path(temporary).unlink()       # Remove only our now-published temporary name.
    if os.name != 'nt':
        directory = os.open(Path(destination).parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending_' + str(os.getpid()) + '_' + str(time.time_ns()))
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    publish_no_overwrite(temporary, path)


def bytes_used(output):
    seen = set(); total = 0
    for path in Path(output).rglob('*'):
        if path.is_file():
            assert not path.is_symlink(), 'Archive output must contain only our own regular files'
            value = path.stat(); key = (value.st_dev, value.st_ino)
            if key in seen:
                continue
            seen.add(key); total += max(value.st_size, getattr(value, 'st_blocks', 0) * 512)
    return total


def fd_stat(value):
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
        modified_ns=value.st_mtime_ns, links=value.st_nlink)


def current_sample(path):
    try:
        with open_stable(path) as stream:
            first = os.fstat(stream.fileno()); digest, count = hash_stream(stream)
        return dict(path=str(Path(path).resolve()), sha256=digest, bytes=count,
            opened_FD=fd_stat(first), UTC=utc(), exact_path_time_atomicity_claimed=False)
    except FileNotFoundError:
        return dict(path=str(Path(path).resolve()), missing=True, UTC=utc())


def stable_copy(source, expected_sha, expected_bytes, destination, output, before_copy=None, *, audit_bound=True):
    """Two passes on the SAME FD survive pathname os.replace; no retry/reroll."""
    source, destination, output = Path(source), Path(destination), Path(output).resolve()
    assert destination.resolve().is_relative_to(output) and not source.resolve().is_relative_to(output)
    assert re.fullmatch('[0-9a-f]{64}', expected_sha) and expected_bytes >= 0
    assert not destination.exists()
    temporary = destination.with_name(destination.name + '.partial_' + str(time.time_ns()))
    try:
        with open_stable(source) as stream:
            initial = os.fstat(stream.fileno()); assert stat.S_ISREG(initial.st_mode)
            actual_sha, actual_bytes = hash_stream(stream)
            proof = dict(expected_sha256=expected_sha, expected_bytes=expected_bytes,
                expected_SHA_origin='IMMUTABLE_CPU_AUDIT' if audit_bound else 'UNREVIEWED_CURRENT_FILE_OBSERVATION',
                opened_FD_sha256=actual_sha, opened_FD_bytes=actual_bytes, opened_FD=fd_stat(initial))
            if actual_sha != expected_sha or actual_bytes != expected_bytes:
                return dict(status='MISSED_AUDITED_SHA_NO_LONGER_AT_ROLLING_PATH', **proof,
                    observed_current_checkpoint=current_sample(source), archive=None)
            if bytes_used(output) + actual_bytes + META_RESERVE > MAX_OUTPUT:
                return dict(status='NOT_ARCHIVED_2GIB_OUTPUT_CAP', **proof, archive=None)
            if before_copy is not None:
                before_copy()  # Test-only seam: production never supplies a callback.
            stream.seek(0); destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(); copied = 0
            with temporary.open('xb', buffering=0) as target:
                while True:
                    block = stream.read(CHUNK)
                    if not block:
                        break
                    target.write(block); digest.update(block); copied += len(block); memory_check()
                target.flush(); os.fsync(target.fileno())
            final = os.fstat(stream.fileno()); copied_sha = digest.hexdigest()
            proof.update(copied_sha256=copied_sha, copied_bytes=copied, source_FD_after=fd_stat(final))
            if copied_sha != expected_sha or copied != expected_bytes:
                return dict(status='MISSED_SOURCE_BYTES_CHANGED_DURING_COPY', **proof, archive=None,
                    retained_rejected_partial=reference(temporary), observed_current_checkpoint=current_sample(source))
            verified = reference(temporary)
            if verified['sha256'] != expected_sha or verified['bytes'] != expected_bytes:
                return dict(status='MISSED_DESTINATION_BYTES_CHANGED_BEFORE_COMMIT', **proof, archive=None,
                    retained_rejected_partial=verified, observed_current_checkpoint=current_sample(source))
            assert (final.st_dev, final.st_ino) != (temporary.stat().st_dev, temporary.stat().st_ino)
            publish_no_overwrite(temporary, destination)
            archive = reference(destination); assert archive['sha256'] == expected_sha and archive['bytes'] == expected_bytes
            return dict(status='ARCHIVED_EXACT_CPU_AUDITED_CHECKPOINT_BYTES' if audit_bound
                else 'UNREVIEWED_STABLE_FD_CHECKPOINT_BYTES_COPIED', **proof, archive=archive,
                independent_copy_inode=True, observed_current_checkpoint=current_sample(source))
    except FileNotFoundError as exc:
        return dict(status='MISSED_SOURCE_MISSING_AT_CAPTURE', expected_sha256=expected_sha,
            expected_bytes=expected_bytes, error=repr(exc), archive=None, observed_current_checkpoint=current_sample(source))


def process_sample(pid, start_ticks, script, launch_review, expected_argv=None, proc_root=Path('/proc')):
    directory = proc_root / str(pid)
    try:
        text = (directory / 'stat').read_text(); end = text.rfind(')'); assert end > 0
        fields = text[end + 2:].split(); actual_ticks = int(fields[19])
        raw = (directory / 'cmdline').read_bytes(); argv = [p.decode('utf-8') for p in raw.split(b'\0') if p]
        again = (directory / 'stat').read_text(); second_fields = again[again.rfind(')') + 2:].split()
        second_ticks = int(second_fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return dict(status='EXPECTED_PIPELINE_PROCESS_ABSENT', pid=pid, UTC=utc())
    if actual_ticks == second_ticks == start_ticks and (fields[0] == 'Z' or second_fields[0] == 'Z'):
        return dict(status='EXPECTED_PIPELINE_PROCESS_ABSENT', pid=pid, zombie=True, UTC=utc())
    epoch = None
    top = proc_root / 'stat'
    if top.exists() and hasattr(os, 'sysconf'):
        boot = next((int(line.split()[1]) for line in top.read_text().splitlines() if line.startswith('btime ')), None)
        if boot is not None:
            epoch = boot + actual_ticks / os.sysconf('SC_CLK_TCK')
    state = dict(pid=pid, start_ticks=actual_ticks, second_start_ticks=second_ticks, argv=argv, start_epoch=epoch,
        argv_sha256=hashlib.sha256(raw).hexdigest(), UTC=utc())
    valid = actual_ticks == second_ticks == start_ticks and str(Path(script).resolve()) in argv
    if '--launch-review' not in argv:
        valid = False
    else:
        index = argv.index('--launch-review')
        valid = valid and index + 1 < len(argv) and Path(argv[index + 1]).resolve() == Path(launch_review).resolve()
    if expected_argv is not None:
        valid = valid and argv == expected_argv
    state['status'] = 'EXPECTED_PIPELINE_IDENTITY_MATCH' if valid else 'PIPELINE_IDENTITY_MISMATCH_STOP_NO_FOLLOW'
    return state


@contextlib.contextmanager
def output_lock(output):
    path = Path(output) / 'archiver.lock'
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def audits(directory):
    result = {}
    for path in Path(directory).glob('checkpoint_audit_*.json'):
        match = AUDIT_PATTERN.fullmatch(path.name)
        if match:
            number = int(match.group(1)); assert 1 <= number <= AUDIT_LIMIT
            result[number] = path
    return result


def expected_seen(updates):
    assert 0 < updates <= 24807
    complete, offset = divmod(updates, 8269)
    return complete * 132300 + offset * 16


def audit_document(path, checkpoint, pipeline_sha, minimum_epoch=None):
    document, binding = small_json(path, 64 << 10)
    assert document['status'] == 'REAL_ATMS_ORDINARY_COMMITTED_CHECKPOINT_CPU_VERIFIED'
    assert document['source_sha256'] == pipeline_sha and document['CUDA_initialized'] is False
    if minimum_epoch is not None:
        assert dt.datetime.fromisoformat(document['UTC']).timestamp() >= minimum_epoch - 2., 'Audit predates bound PID'
    assert document['model_optimizer_RNG_order_coverage_log_prefix_verified'] is True
    assert Path(document['checkpoint']['path']).resolve() == Path(checkpoint).resolve()
    assert document['seen'] == expected_seen(document['updates'])
    assert 0 <= document['previous_updates'] < document['updates']
    return document, binding


def terminal_markers(audit_dir, minimum_epoch=None):
    paths = []
    success = Path(audit_dir) / 'actual_image_screen_completion.json'
    if success.exists():
        value, binding = small_json(success)
        assert value['status'] == 'X01_MATCHED_ATMS_NATIVE_ORDINARY_ACTUAL_IMAGE_SCREEN_CLOSED'
        if minimum_epoch is None or dt.datetime.fromisoformat(value['UTC']).timestamp() >= minimum_epoch - 2.:
            paths.append(dict(kind='MATCHED_IMAGE_SCREEN_TERMINAL_SUCCESS', binding=binding))
    for path in sorted(Path(audit_dir).glob('pipeline_failure_*.json')):
        value, binding = small_json(path)
        assert value['status'] == 'X01_PIPELINE_STOPPED_FAILURE_AND_CHECKPOINTS_PRESERVED_NO_AUTOMATIC_RETRY'
        if minimum_epoch is None or dt.datetime.fromisoformat(value['UTC']).timestamp() >= minimum_epoch - 2.:
            paths.append(dict(kind='PIPELINE_TERMINAL_FAILURE', binding=binding))
    return paths


def run(args):
    assert 1 <= args.max_seconds <= MAX_SECONDS
    args.output = args.output.resolve(); args.checkpoint = args.checkpoint.resolve()
    args.pipeline_script = args.pipeline_script.resolve(); args.launch_review = args.launch_review.resolve()
    args.audit_dir = args.audit_dir.resolve(); args.spec = args.spec.resolve()
    assert not args.checkpoint.is_relative_to(args.output) and not args.audit_dir.is_relative_to(args.output)
    assert not args.output.is_relative_to(args.checkpoint.parent), 'Never place archive inside formal train output'
    assert args.resume or not args.output.exists()
    args.output.mkdir(parents=True, exist_ok=True); memory_check()
    with output_lock(args.output):
        launch, launch_ref = small_json(args.launch_review)
        assert launch['status'] == 'ATMS_STANDARD_ORDINARY_FINITE_PIPELINE_REVIEWED'
        pipeline_ref = reference(args.pipeline_script)
        pins = {str(Path(item['path']).resolve()): item for item in launch['source_pins']}
        assert pins[str(args.pipeline_script)]['sha256'] == pipeline_ref['sha256']
        config_path = args.output / 'archiver_config.json'
        sample = process_sample(args.pipeline_pid, args.pipeline_start_ticks, args.pipeline_script, args.launch_review)
        config = dict(schema=SCHEMA, source=reference(__file__), pipeline=pipeline_ref, pipeline_launch=launch_ref,
            pipeline_pid=args.pipeline_pid, pipeline_start_ticks=args.pipeline_start_ticks,
            checkpoint_path=str(args.checkpoint), audit_directory=str(args.audit_dir), training_spec_path=str(args.spec),
            maximum_output_bytes=MAX_OUTPUT, maximum_RAM_bytes=MAX_RAM, stream_chunk_bytes=CHUNK,
            maximum_audits=AUDIT_LIMIT, poll_seconds=1, terminal_unreviewed_snapshot_separate=True)
        if config_path.exists():
            existing, _ = small_json(config_path)
            assert existing == config | {'expected_argv': existing['expected_argv'],
                'pipeline_start_epoch': existing['pipeline_start_epoch']}
            config = existing
        else:
            assert sample['status'] == 'EXPECTED_PIPELINE_IDENTITY_MATCH', sample
            assert sample['start_epoch'] is not None, 'Actual Linux /proc boot-time binding required'
            config['expected_argv'] = sample['argv']; config['pipeline_start_epoch'] = sample['start_epoch']
            atomic(config_path, config)
        started = time.monotonic(); stopped = [False]; records = {}; spec_ref = None
        run_dir = args.output / 'runs' / str(time.time_ns()); run_dir.mkdir(parents=True)
        atomic(run_dir / 'control.json', dict(config=reference(config_path), UTC=utc(), pid=os.getpid(),
            max_seconds=args.max_seconds, first_process_observation=sample))
        signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
        for path in sorted((args.output / 'records').glob('audit_*.json')):
            value, _ = small_json(path); number = value['serial']; assert number not in records
            assert value['config'] == reference(config_path)
            checked(value['audit'])
            checked(value['training_spec'])
            if spec_ref is None:
                spec_ref = value['training_spec']
            else:
                assert value['training_spec'] == spec_ref, 'Prior records mixed training spec SHA'
            if value['capture'].get('archive'):
                checked(value['capture']['archive'])
            records[number] = value
        reason = None; last_heartbeat = -60.; stable_absent_polls = 0
        try:
            while True:
                memory_check(); assert bytes_used(args.output) <= MAX_OUTPUT
                checked(config['pipeline']); checked(config['pipeline_launch'])
                sample = process_sample(args.pipeline_pid, args.pipeline_start_ticks, args.pipeline_script,
                    args.launch_review, config['expected_argv'])
                if sample['status'] == 'PIPELINE_IDENTITY_MISMATCH_STOP_NO_FOLLOW':
                    reason = sample['status']; break
                if args.spec.exists():
                    current = reference(args.spec)
                    if spec_ref is not None:
                        assert current == spec_ref, 'The immutable running training spec changed'
                    spec_ref = current
                found = audits(args.audit_dir)
                for number, path in sorted(found.items()):
                    if number in records:
                        continue
                    if stopped[0] or time.monotonic() - started >= args.max_seconds:
                        break
                    copy_process = process_sample(args.pipeline_pid, args.pipeline_start_ticks, args.pipeline_script,
                        args.launch_review, config['expected_argv'])
                    if copy_process['status'] == 'PIPELINE_IDENTITY_MISMATCH_STOP_NO_FOLLOW':
                        sample = copy_process; reason = sample['status']; break
                    document, binding = audit_document(path, args.checkpoint, pipeline_ref['sha256'], config['pipeline_start_epoch'])
                    assert spec_ref is not None, 'A CPU-audited formal checkpoint requires its existing spec'
                    if number > 1 and number - 1 in records:
                        assert document['previous_updates'] == records[number - 1]['updates']
                    expected = document['checkpoint']
                    destination = args.output / 'checkpoints' / f'checkpoint_{number:03d}_{expected["sha256"]}.pt'
                    capture = stable_copy(args.checkpoint, expected['sha256'], int(expected['bytes']), destination, args.output)
                    checked(binding)
                    value = dict(schema=SCHEMA, serial=number, UTC=utc(), status=capture['status'],
                        config=reference(config_path), source=config['source'], pipeline=config['pipeline'],
                        pipeline_launch=config['pipeline_launch'], training_spec=spec_ref, audit=binding,
                        updates=document['updates'], seen=document['seen'], expected_checkpoint=expected,
                        process_observation=copy_process, capture=capture, no_historical_checkpoint_fabricated=True,
                        no_formal_file_or_training_state_modified=True)
                    atomic(args.output / 'records' / f'audit_{number:03d}.json', value); records[number] = value
                    print(json.dumps(dict(serial=number, status=capture['status'], updates=document['updates'],
                        expected_sha256=expected['sha256'], actual_opened_FD_sha256=capture.get('opened_FD_sha256'))), flush=True)
                if reason is not None:
                    break
                markers = terminal_markers(args.audit_dir, config['pipeline_start_epoch'])
                if sample['status'] == 'EXPECTED_PIPELINE_PROCESS_ABSENT':
                    stable_absent_polls += 1
                    if stable_absent_polls >= 2 and set(found).issubset(records):
                        reason = 'PIPELINE_TERMINAL_ALL_EXISTING_AUDITS_PROCESSED' if markers else 'PIPELINE_ABSENT_WITHOUT_TERMINAL_MARKER'
                        break
                else:
                    stable_absent_polls = 0
                elapsed = time.monotonic() - started
                if stopped[0] or elapsed >= args.max_seconds:
                    reason = 'ARCHIVER_SIGNAL_STOPPED' if stopped[0] else 'ARCHIVER_FINITE_DEADLINE_REACHED'
                    break
                if elapsed - last_heartbeat >= 60:
                    print(json.dumps(dict(status='WAITING_OR_ARCHIVING_EXISTING_X01_ONLY', seconds=elapsed,
                        pipeline=sample['status'], processed_audits=len(records), allocated_bytes=bytes_used(args.output))), flush=True)
                    last_heartbeat = elapsed
                time.sleep(min(1., max(0., args.max_seconds - elapsed)))
            last_pointer = None
            if reason in ('PIPELINE_TERMINAL_ALL_EXISTING_AUDITS_PROCESSED',
                    'PIPELINE_ABSENT_WITHOUT_TERMINAL_MARKER') and args.checkpoint.exists():
                observed = current_sample(args.checkpoint)
                matching = [value['capture']['archive'] for value in records.values()
                    if value['capture'].get('archive') and value['capture']['archive']['sha256'] == observed.get('sha256')]
                if matching:
                    last_pointer = dict(status='TERMINAL_POINTER_ALREADY_HAS_MATCHING_CPU_AUDITED_ARCHIVE',
                        observed=observed, archive=matching[-1])
                else:
                    destination = args.output / 'terminal_unreviewed' / ('checkpoint_' + observed['sha256'] + '.pt')
                    if destination.exists():
                        archive = reference(destination)
                        assert archive['sha256'] == observed['sha256'] and archive['bytes'] == observed['bytes']
                        capture = dict(status='PREVIOUS_UNREVIEWED_LAST_OBSERVATION_ALREADY_PRESERVED', archive=archive)
                    else:
                        capture = stable_copy(args.checkpoint, observed['sha256'], observed['bytes'], destination,
                            args.output, audit_bound=False)
                    last_pointer = dict(status='UNREVIEWED_TERMINAL_CHECKPOINT_BYTES_PRESERVED' if capture.get('archive')
                        else 'TERMINAL_POINTER_NOT_PRESERVED_CHANGED_OR_RESOURCE_STOP', observed=observed, capture=capture,
                        CPU_model_optimizer_RNG_audit_established=False, never_count_as_audited_slot=True,
                        observed_after_expected_pipeline_disappeared=True,
                        terminal_marker_present=reason == 'PIPELINE_TERMINAL_ALL_EXISTING_AUDITS_PROCESSED',
                        unknown_or_remaining_child_activity_not_ruled_out=reason == 'PIPELINE_ABSENT_WITHOUT_TERMINAL_MARKER')
            final_audits = audits(args.audit_dir); highest = max(final_audits, default=0)
            gaps = sorted(set(range(1, highest + 1)) - set(final_audits))
            summary = dict(schema=SCHEMA, status=reason, source=reference(__file__), config=reference(config_path),
                pipeline=config['pipeline'], pipeline_launch=config['pipeline_launch'], training_spec=spec_ref,
                expected_pipeline_pid=args.pipeline_pid, expected_pipeline_start_ticks=args.pipeline_start_ticks,
                last_process_observation=sample, terminal_markers=terminal_markers(args.audit_dir, config['pipeline_start_epoch']),
                existing_audit_serials=sorted(final_audits), processed_serials=sorted(records), missing_audit_sequence_numbers=gaps,
                archived_serials=sorted(n for n, value in records.items() if value['capture'].get('archive')),
                missed_serials=sorted(n for n, value in records.items() if not value['capture'].get('archive')),
                records=[reference(args.output / 'records' / f'audit_{n:03d}.json') for n in sorted(records)],
                terminal_last_pointer=last_pointer, peak_RAM_bytes=peak_rss(), allocated_bytes=bytes_used(args.output),
                seconds=time.monotonic() - started, Torch_imported='torch' in sys.modules,
                no_GPU_or_training_commands_started=True, original_pipeline_or_sources_modified=False)
            atomic(run_dir / 'completion.json', summary)
            assert bytes_used(args.output) <= MAX_OUTPUT; memory_check()
            print(json.dumps(dict(status=reason, completion=reference(run_dir / 'completion.json'),
                archived=len(summary['archived_serials']), missed=len(summary['missed_serials']))), flush=True)
            return 0 if reason == 'PIPELINE_TERMINAL_ALL_EXISTING_AUDITS_PROCESSED' else 2
        except BaseException as exc:
            atomic(run_dir / 'failure.json', dict(schema=SCHEMA, status='CPU_ARCHIVER_FAILED_EXISTING_COPIES_PRESERVED',
                error=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic() - started,
                processed_serials=sorted(records), no_retry_or_training_control=True))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-pid', type=int, default=240115)
    parser.add_argument('--pipeline-start-ticks', type=int, default=82331630)
    for name in ('pipeline-script', 'launch-review', 'audit-dir', 'checkpoint', 'spec', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=172800)
    parser.add_argument('--resume', action='store_true')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
