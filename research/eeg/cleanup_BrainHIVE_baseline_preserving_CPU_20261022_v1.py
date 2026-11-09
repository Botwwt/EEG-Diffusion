"""Apply a narrow cache-only subset of the old retirement proposal.

No directories, EEG inputs, learned models, generated images or evaluation
outputs are removed. Every retained entry in the old proposal is SHA-checked.
Run once, server only; a refused preflight never proceeds to deletion.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time

ROOT = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
BASE = ROOT / 'audit/baseline_preserving_cleanup_20261022_v1'
OLD_PLAN = ROOT / 'audit/BrainHIVE_legacy_retirement_exact_file_plan_20261022_v2.json.gz'
PREFIX = Path('/root/autodl-tmp/avde_rime_oracle_latent_20261015/cache_v1')
PILOT = Path('/root/autodl-tmp/repeat_precision_pilot_20261017_v1')
TRIAL_LATENTS = PILOT / 'training_trial_latents.npy'
SCOPE = ROOT / 'audit/BrainHIVE_cleanup_user_scope_20261022_v1.json'
MINIMUM_NEW_LIFECYCLE_BYTES = 26655048192


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def disposable(path):
    p = Path(path)
    return p == TRIAL_LATENTS or (
        p.parent == PREFIX and re.fullmatch(r'prefix_[0-9]{5}\.pt', p.name) is not None)


def identity(path):
    p = Path(path)
    assert p.resolve(strict=True) == p and not p.is_symlink(), str(p)
    s = p.lstat()
    assert stat.S_ISREG(s.st_mode), str(p)
    return (s.st_dev, s.st_ino, s.st_size, s.st_nlink, s.st_mtime_ns)


def assert_record(row, content=True):
    current = identity(row['path'])
    assert current[:4] == (row['dev'], row['ino'], row['bytes'], row['links']), row['path']
    if content:
        assert sha(row['path']) == row['sha256'], row['path']
    return current


def live_uses(targets):
    wanted = {str(p) for p in targets}; hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    if os.readlink(fd).removesuffix(' (deleted)') in wanted:
                        hits.append(str(fd))
                except FileNotFoundError:
                    pass
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[-1] in wanted:
                    hits.append(str(proc / 'maps'))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return hits


def self_check():
    assert disposable(PREFIX / 'prefix_00000.pt')
    assert disposable(TRIAL_LATENTS)
    for p in [PREFIX / 'oracle_coordinates.pt', PREFIX / 'manifest.json',
              PREFIX / '../prefix_00000.pt', PREFIX / 'prefix_00000.pt.bak',
              PILOT / 'calibration.npz', PILOT / 'image.png',
              Path('/root/avde_author_training_cache_20261016/preprocessed_bc/sub-01/preprocessed_eeg_training.npy'),
              Path('/root/brainhive_reproduction_20261020_v1/components/bounded_v1/risk/component_final.pt'),
              Path('/root/autodl-tmp/avde_native680_stage2_20261016_v1/training/ar-ckpt-last.pth.EXACT_BS4_ZLIB_v1')]:
        assert not disposable(p), p


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true')
    parser.add_argument('--self-check', action='store_true'); args = parser.parse_args()
    self_check()
    if args.self_check:
        print('CACHE_WHITELIST_AND_BASELINE_EXCLUSIONS_PASSED'); return
    assert Path(__file__).resolve().parent == ROOT and not BASE.exists()
    assert sha(OLD_PLAN) == '25af127952982b12880ca3afbec269a0250ea8425772c974a54b6de25dcdb72c'
    scope = json.loads(SCOPE.read_text())
    assert scope['preserve_potential_future_baselines'] is True
    assert scope['blanket_legacy_retirement_authorized'] is False
    assert not subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              check=True, capture_output=True, text=True).stdout.strip()
    old = json.loads(gzip.decompress(OLD_PLAN.read_bytes()))['files']
    targets = [r for r in old if disposable(r['path'])]
    retained = [r for r in old if not disposable(r['path'])]
    assert len(old) == 113820 and len(targets) == 66
    assert all(r['links'] == 1 for r in targets)
    assert {Path(r['path']).name for r in targets if Path(r['path']).parent == PREFIX} == {
        f'prefix_{start:05d}.pt' for start in range(0, 16540, 256)}
    # Authenticate derivation and retained inputs, without running old fits/generation.
    prefix_manifest = PREFIX / 'manifest.json'; pm = json.loads(prefix_manifest.read_text())
    assert pm['not_inference_gallery'] is True and pm['latent_targets_for_training_and_diagnostics_only'] is True
    declared = {r['path']: r for r in pm['shards']}
    assert len(declared) == 65
    for row in targets:
        if Path(row['path']).parent == PREFIX:
            assert row['sha256'] == declared[row['path']]['sha256']
    trial_manifest = PILOT / 'training_cache_manifest.json'; tm = json.loads(trial_manifest.read_text())
    assert tm['cache']['path'] == str(TRIAL_LATENTS)
    raw = Path('/root/avde_author_training_cache_20261016/preprocessed_bc/sub-01/preprocessed_eeg_training.npy')
    assert sha(raw) == tm['raw_sha256']
    assert sha(tm['encoder']['path']) == tm['encoder']['sha256']
    assert sha(pm['teacher']['path']) == pm['teacher']['sha256']
    scripts = [Path('/root/avde_rime_20261015/prepare_oracle_latent_cache.py'),
               Path('/root/avde_rime_20261015/run_repeat_precision_pilot.py')]
    assert all(p.is_file() for p in scripts)
    protected = {str(p): sha(p) for p in [prefix_manifest, trial_manifest, SCOPE, *scripts,
        ROOT / 'results.tsv', ROOT / 'experiment_ledger.jsonl',
        ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json',
        ROOT / 'configs/BrainHIVE_functional_alignment_v2_preregistration_DRAFT.json']}
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    assert not {r['path'] for r in targets}.intersection(recipe['input_sha256'])
    for name, digest in recipe['source_sha256'].items():
        assert sha(ROOT / name) == digest
        protected[str(ROOT / name)] = digest
    # The historical proposal includes unique baseline files: verify then retain them.
    before = {p: shutil.disk_usage(p).free for p in ['/root', '/root/autodl-tmp']}
    retained_stats = {}; seen = set()
    for i, row in enumerate(old):
        current = assert_record(row, content=False)
        key = current[:2]
        if key not in seen:
            assert sha(row['path']) == row['sha256'], row['path']; seen.add(key)
        if not disposable(row['path']):
            retained_stats[row['path']] = current
        if i % 10000 == 0:
            print('VERIFIED_OLD_PROPOSAL_ENTRIES', i, flush=True)
    assert not live_uses([r['path'] for r in targets])
    BASE.mkdir(mode=0o700)
    plan = dict(status='NARROW_CACHE_SUBSET_ONLY', UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        old_plan_sha256=sha(OLD_PLAN), source_sha256=sha(__file__), scope=scope,
        targets=targets, retained_proposal_entries=len(retained), protected_sha256=protected,
        original_images_or_models_or_EEG_deleted=False, before_free=before,
        rebuilding='Use preserved sources with fresh output paths. Prefix reconstruction needs the pinned public VAR/VQ weights, which may require re-download. No cache byte-exact replay or full rebuild was executed here.',
        prefix_rebuild_argv=['/root/miniconda3/bin/python', str(scripts[0]),
            '--rime-root', '/root/rime_var_20261015', '--oracle-run', str(Path(pm['teacher']['path']).parent),
            '--config', pm['config']['path'], '--out', str(PREFIX) + '_rebuilt_after_20261022'],
        training_latent_rebuild_source=str(scripts[1]),
        training_latent_rebuild_note='Frozen encoder plus preserved raw input, split, metadata and original configuration; do not rerun or overwrite the completed pilot as an automatic recovery step.')
    write_new(BASE / 'plan.json', plan)
    if not args.apply:
        print('PLAN_ONLY_NO_DELETION'); return
    assert not live_uses([r['path'] for r in targets])
    with (BASE / 'deletion_journal.jsonl').open('x') as journal:
        for row in targets:
            assert_record(row)
            assert disposable(row['path'])
            Path(row['path']).unlink()
            journal.write(json.dumps(dict(path=row['path'], sha256=row['sha256'], deleted=True)) + '\n')
            journal.flush(); os.fsync(journal.fileno())
    for path, previous in retained_stats.items():
        assert identity(path) == previous, path
    for path, digest in protected.items():
        assert sha(path) == digest, path
    for folder in (PREFIX, PILOT):
        write_new(folder / 'REBUILDABLE_CACHE_SUBSET_RETIRED_20261022_v1.json', dict(
            status='SELECTED_REPRODUCIBLE_CACHE_PAYLOADS_REMOVED', receipt=str(BASE / 'receipt.json'),
            retained_inputs_models_images_results=True, plan_sha256=sha(BASE / 'plan.json')))
    after = {p: shutil.disk_usage(p).free for p in before}
    receipt = dict(status='BASELINES_PRESERVED_NARROW_CACHE_CLEANUP_COMPLETE',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        source_sha256=sha(__file__), plan_sha256=sha(BASE / 'plan.json'),
        scope_sha256=sha(SCOPE), old_plan_superseded_for_execution=True,
        files_removed=len(targets), allocated_bytes_removed=sum(r['allocated'] for r in targets),
        before_free=before, after_free=after,
        verified_and_retained_old_plan_entries=len(retained),
        all_retained_entries_SHA_verified_before_and_identity_unchanged_after=True,
        protected_source_and_result_files_unchanged=True,
        current_BrainHIVE_chain_outside_retirement_plan=True,
        raw_EEG_unique_weights_generated_images_deleted=0,
        minimum_four_arm_space_with_two_buffers=MINIMUM_NEW_LIFECYCLE_BYTES,
        remaining_aggregate_space_gap=max(0, MINIMUM_NEW_LIFECYCLE_BYTES - sum(after.values())),
        formal_training_admitted=False, formal_training_started=False,
        no_new_GPU_work=True)
    write_new(BASE / 'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
