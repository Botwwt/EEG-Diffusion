"""Read-only observation of the separately diagnosed replacement recovery lifecycle.

This creates one new immutable observation receipt. It neither starts jobs nor
changes recipes, weights, checkpoints, scientific results, or running state.
"""
from pathlib import Path
import hashlib
import io
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
    out = folder / ('formal_recovery_observation_' + stamp + '.json')
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
    recovery_path = base / 'formal_recovery_queue.json'
    recovery_ref, recovery = captured(recovery_path)
    recipes = [read(ROOT / 'configs/functional_full_lifecycle_20261022_v1' / (a + '.json')) for a in ('O', 'P', 'PN', 'S')]
    job_paths = [ROOT / 'audit' / (name + '_job.json') for recipe in recipes for name in recipe['formal_job_names']]
    jobs = []
    for p in job_paths:
        if not p.exists():
            continue
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
        checkpoint = target.with_name('resume.pt')
        if checkpoint.exists():
            import torch
            raw = checkpoint.read_bytes()
            saved = torch.load(io.BytesIO(raw), weights_only=False, map_location='cpu')
            assert 0 < saved['updates'] <= 24807
            assert saved['coverage'].shape == (3, 132300)
            assert int(saved['coverage'].sum()) == saved['seen'] == saved['cursor']
            provenance = target.with_name('input_provenance.json')
            assert saved['input_provenance_sha256'] == digest(provenance)
            progress[arm]['saved_checkpoint'] = dict(path=str(checkpoint),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), updates=saved['updates'],
                seen_queries=saved['seen'], cursor=saved['cursor'], seconds=saved['seconds'],
                coverage_shape=list(saved['coverage'].shape), coverage_count=int(saved['coverage'].sum()),
                input_provenance_sha256=saved['input_provenance_sha256'], CPU_decode_verified=True)
    space = {}
    for p in ('/root', '/root/autodl-tmp', '/dev/shm'):
        v = os.statvfs(p)
        space[p] = dict(device=os.stat(p).st_dev, total=v.f_blocks * v.f_frsize,
                        available=v.f_bavail * v.f_frsize)
    result = dict(status='DIAGNOSED_FOUR_ARM_RECOVERY_ACTUAL_PROCESS_OBSERVATION',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source=ref(__file__),
        cleanup_cancelled_by_latest_user_instruction=True, deletion_performed=False,
        server_hostname=os.uname().nodename,
        git_HEAD=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        source_manifest=ref(manifest), all_bound_source_SHA_verified=len(source_records),
        bound_sources=source_records, formal_admission=ref(admission_path),
        continuation=dict(receipt=outer_ref, observed=outer, process=proc(outer['pid'])),
        lifecycle=dict(receipt=formal_ref, observed=formal, process=proc(formal['pid'])),
        recovery=dict(receipt=recovery_ref, observed=recovery, process=proc(recovery['pid'])),
        original_queues_are_preserved_history=True,
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
