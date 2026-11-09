"""Retire only explicitly verified duplicate/rebuildable payloads, server only.

Preserve every EEG input, learned checkpoint, original image, metric table,
manifest, log, and current BrainHIVE feature cache. All old image files are
SHA-verified before retiring their reproducible feature shards.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
BASE = ROOT / 'audit/rebuildable_payload_cleanup_20261022_v1_repaired_once'
ARCHIVE = Path('/root/autodl-tmp/mced_research_bundle/external/things_eeg2_direct_v1/training_images.zip')
IMAGES = Path('/root/autodl-tmp/avde_single_task_data_20261016/image_set')
ARMS = ('REF_REG_correct_R4', 'IDEA1_Frozen_correct_R4',
        'IDEA1_RealPaired_correct_R4', 'IDEA1_Conditional_correct_R4')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def live_uses(targets):
    wanted = {str(p) for p in targets}
    hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    if os.readlink(fd).removesuffix(' (deleted)') in wanted:
                        hits.append(str(fd))
                except OSError:
                    pass
            maps = (proc / 'maps').read_text()
            for row in maps.splitlines():
                if row.split(None, 5)[-1] in wanted:
                    hits.append(proc.name)
        except OSError:
            pass
    return hits


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert ROOT == Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
    assert not BASE.exists()
    assert not subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.strip()
    queue = json.loads((ROOT / 'audit/functional_profiles_preparation_20261022_v1/queue_state.json').read_text())
    assert queue['status'] == 'ALL_FOUR_ACTUAL_ENGINEERING_PROFILES_COMPLETE_NO_FORMAL_FIT'
    BASE.mkdir()
    before = {p: shutil.disk_usage(p).free for p in ('/root', '/root/autodl-tmp')}
    protected = {str(ROOT / p): sha(ROOT / p) for p in (
        'configs/BrainHIVE_bounded_main_recipe_v1.json',
        'configs/BrainHIVE_functional_alignment_v2_preregistration_DRAFT.json',
        'experiment_ledger.jsonl', 'results.tsv', 'extract_image_features.py')}
    rows = []
    image_rows = []
    cache_proofs = []
    for arm in ARMS:
        folder = DATA / 'evaluation/V' / arm / 'features'
        config = json.loads((folder / 'config.json').read_text())
        completion = json.loads((folder / 'completion.json').read_text())
        scores = folder.parent / 'metrics/summary.json'
        summary = json.loads(scores.read_text())
        assert completion['status'] == summary['status'] == 'COMPLETED'
        assert completion['queries'] == summary['queries'] == 16600
        manifest = Path(config['manifest'])
        assert sha(manifest) == config['manifest_sha256'] == completion['manifest_sha256']
        manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert len(manifest_rows) == 16600
        for row in manifest_rows:
            path = Path(row['generated_path'])
            assert path.exists() and sha(path) == row['generated_sha256'], str(path)
            image_rows.append(dict(path=str(path), sha256=row['generated_sha256']))
        payloads = sorted(folder.glob('*/*.pt'))
        assert len(payloads) == 260
        for path in payloads:
            assert path.resolve() == path and path.is_relative_to(folder)
            assert path.parent.name in ('DINO', 'EffNet', 'CapSim', 'ObjectF1')
            assert re.fullmatch(r'[0-9]{8}\.pt', path.name) and not path.is_symlink()
            st = path.stat()
            rows.append(dict(path=str(path), sha256=sha(path), bytes=st.st_size,
                             allocated=st.st_blocks * 512, dev=st.st_dev, ino=st.st_ino,
                             reason='Rebuildable frozen-evaluator feature shard of preserved scored images'))
        cache_proofs.append(dict(arm=arm, feature_root=str(folder), manifest=str(manifest),
            manifest_sha256=sha(manifest), scores=str(scores), scores_sha256=sha(scores),
            config_sha256=sha(folder / 'config.json'), completion_sha256=sha(folder / 'completion.json'),
            rebuild_argv=['/root/miniconda3/bin/python', str(ROOT / 'extract_image_features.py'),
                          '--manifest', str(manifest), '--output', str(folder) + '_rebuilt_after_20261022']))
    zip_images = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = IMAGES / info.filename
            assert target.resolve().is_relative_to(IMAGES.resolve()) and target.is_file()
            with archive.open(info) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert target.stat().st_size == info.file_size and sha(target) == digest
            zip_images.append(dict(member=info.filename, preserved_path=str(target), sha256=digest))
    assert len(zip_images) == 16540
    st = ARCHIVE.stat()
    assert not ARCHIVE.is_symlink() and ARCHIVE.resolve() == ARCHIVE
    rows.append(dict(path=str(ARCHIVE), sha256=sha(ARCHIVE), bytes=st.st_size,
        allocated=st.st_blocks * 512, dev=st.st_dev, ino=st.st_ino,
        reason='Duplicate downloadable ZIP; all16540 original JPEG member bytes preserved and SHA-verified in expanded image set'))
    targets = [Path(row['path']) for row in rows]
    assert not live_uses(targets)
    write(BASE / 'preserved_image_identity.json', dict(old_generated=image_rows, original_stimuli=zip_images))
    plan = dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=sha(__file__),
        targets=rows, old_image_cache_proofs=cache_proofs, before_free=before,
        retained_protected_sha256=protected, planned_allocated_bytes=sum(x['allocated'] for x in rows),
        EEG_or_weights_removed=False, original_or_generated_images_removed=False,
        raw_scores_or_manifests_removed=False, current_BrainHIVE_features_removed=False)
    write(BASE / 'plan.json', plan)
    if not args.apply:
        print(json.dumps(dict(status='VERIFIED_PLAN_ONLY', allocated=plan['planned_allocated_bytes'])))
        return
    assert not live_uses(targets)
    for row in rows:
        path = Path(row['path']); st = path.stat()
        assert (st.st_dev, st.st_ino, st.st_size) == (row['dev'], row['ino'], row['bytes'])
        assert sha(path) == row['sha256']
        path.unlink()
    for proof in cache_proofs:
        write(Path(proof['feature_root']) / 'PAYLOADS_RETIRED_REBUILDABLE_20261022_v1.json',
              dict(status='FEATURE_PAYLOAD_RETIRED_AFTER_COMPLETED_EVALUATION',
                   retirement_receipt=str(BASE / 'receipt.json'), **proof))
        assert sha(proof['scores']) == proof['scores_sha256']
        assert sha(proof['manifest']) == proof['manifest_sha256']
    write(ARCHIVE.with_name(ARCHIVE.name + '.DUPLICATE_RETIRED_20261022_v1.json'),
          dict(original_zip_sha256=rows[-1]['sha256'], preserved_images=str(IMAGES / 'training_images'),
               preserved_image_identity=str(BASE / 'preserved_image_identity.json'),
               count=16540, original_JPEG_bytes_preserved=True))
    for path, digest in protected.items():
        assert sha(path) == digest
    after = {p: shutil.disk_usage(p).free for p in before}
    receipt = dict(status='VERIFIED_DUPLICATE_AND_REBUILDABLE_PAYLOAD_CLEANUP_COMPLETE',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), plan_sha256=sha(BASE / 'plan.json'),
        files_retired=len(rows), allocated_bytes_retired=sum(x['allocated'] for x in rows),
        before_free=before, after_free=after,
        observed_free_delta_bytes={p: after[p] - before[p] for p in before},
        old_generated_images_SHA_verified_and_preserved=len(image_rows),
        original_JPEGs_byteverified_and_preserved=len(zip_images),
        EEG_weights_images_scores_manifests_preserved=True,
        current_BrainHIVE_feature_cache_preserved=True, no_new_GPU_jobs=True,
        four_arm_subtotal_still_required=26118177280,
        four_arm_storage_admitted=False)
    write(BASE / 'receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
