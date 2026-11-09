"""Read-only observation of the admitted replacement-instance lifecycle.

This creates one new immutable observation receipt. It neither starts jobs nor
changes recipes, weights, checkpoints, scientific results, or running state.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    p = Path(path)
    return dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)


def read(path):
    return json.loads(Path(path).read_bytes())


def captured(path):
    p = Path(path)
    raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)), json.loads(raw)


def proc(pid):
    p = Path('/proc') / str(pid)
    if not p.exists():
        return dict(pid=pid, live=False)
    try:
        state = (p / 'stat').read_text().split(') ', 1)[1].split()[0]
        return dict(pid=pid, live=state != 'Z', state=state,
                    cmdline=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return dict(pid=pid, live=False, exited_during_observation=True)


def main():
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    folder = ROOT / 'audit/new_instance_restore_20261022_v2'
    out = folder / ('formal_running_observation_' + stamp + '.json')
    assert not out.exists()
    manifest = ROOT / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json'
    m = read(manifest)
    # This is the same manifest bound by the live outer queue.
    sources = m['source_sha256']
    assert sources
    source_records = []
    iterator = sources.items() if isinstance(sources, dict) else enumerate(sources)
    for name, item in iterator:
        if isinstance(item, str):
            path, expected = ROOT / name, item
        else:
            path = Path(item.get('path', item.get('relative_path', str(name))))
            if not path.is_absolute():
                path = ROOT / path
            expected = item['sha256']
        assert digest(path) == expected, str(path)
        source_records.append(ref(path))
    base = ROOT / 'audit/functional_full_lifecycle_20261022_v1'
    outer_path = folder / 'continuation_queue.json'
    formal_path = base / 'formal_queue.json'
    outer_ref, outer = captured(outer_path)
    formal_ref, formal = captured(formal_path)
    admission_path = base / 'formal_admission_closure.json'
    admission = read(admission_path)
    assert admission['status'] == 'FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMISSION_ACTUALLY_VERIFIED'
    jobs = []
    for p in sorted(ROOT.glob('audit/BrainHIVE_functional_v2_*_expanded_lifecycle_20261022_v1_fit_job.json')):
        receipt, d = captured(p)
        jobs.append(dict(receipt=receipt, record=d, process=proc(d['pid']), wrapper=proc(d['wrapper_pid'])))
    progress = {}
    for arm in ('O', 'P', 'PN', 'S'):
        recipe_path = ROOT / 'configs/functional_full_lifecycle_20261022_v1' / (arm + '.json')
        recipe = read(recipe_path)
        target = Path(recipe['output_directory']) / 'progress.json'
        progress[arm] = dict(recipe=ref(recipe_path), output_directory=recipe['output_directory'],
                             progress_present=target.exists())
        if target.exists():
            target_ref, observed = captured(target)
            progress[arm].update(receipt=target_ref, observed=observed)
    space = {}
    for p in ('/root', '/root/autodl-tmp', '/dev/shm'):
        v = os.statvfs(p)
        space[p] = dict(device=os.stat(p).st_dev, total=v.f_blocks * v.f_frsize,
                        available=v.f_bavail * v.f_frsize)
    result = dict(status='ADMITTED_FULL_FOUR_ARM_LIFECYCLE_ACTUAL_PROCESS_OBSERVATION',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source=ref(__file__),
        cleanup_cancelled_by_latest_user_instruction=True, deletion_performed=False,
        server_hostname=os.uname().nodename,
        git_HEAD=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        source_manifest=ref(manifest), all_bound_source_SHA_verified=len(source_records),
        bound_sources=source_records, formal_admission=ref(admission_path),
        continuation=dict(receipt=outer_ref, observed=outer, process=proc(outer['pid'])),
        lifecycle=dict(receipt=formal_ref, observed=formal, process=proc(formal['pid'])),
        formal_jobs=jobs, progress=progress, filesystems=space,
        GPU_observation=subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip(),
        prior_results=ref(folder / 'migrated_results_verified.json'),
        runtime_closure=ref(folder / 'exact_coverage_closure_core_and_evaluation_attempt1.json'),
        evaluation_probe=ref(ROOT / 'audit/new_instance_evaluation_environment_probe_20261022_v2_attempt1.json'),
        results_table=ref(ROOT / 'results.tsv'), scientific_goal_complete=False,
        new_functional_V_image_gain_established=False,
        interpretation='Time-stamped observation; progress and process snapshots are not completed training or image-quality evidence.')
    out.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + '\n').encode())
    print(json.dumps(dict(path=str(out), sha256=digest(out), UTC=result['UTC'],
        formal_jobs=[dict(pid=x['record']['pid'], status=x['record']['status'], live=x['process']['live']) for x in jobs],
        progress={a: p.get('observed', {}).get('updates') for a, p in progress.items()})))


if __name__ == '__main__':
    main()
