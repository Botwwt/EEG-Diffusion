"""Retire exact public-download copies unused by the active Brain-HIVE jobs.

Only the 20 registered ATM downloads and two superseded generic weights are
eligible. Raw EEG, locally learned weights, images and runtime files are excluded.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import time

from relocate_retired_ATM_features_CPU_20261021_v1 import ROOT, active_uses, memory, sha, write

RECEIPT = ROOT / 'audit/unused_public_assets_retirement_20261021_v1.json'
WORK = Path('/root/brainhive_reproduction_20261020_v1')
ATM = Path('/root/atm_author_pretrained_20261020_v2')
RAM = Path('/dev/shm/eeg_diffusion_next_20261018_v1')


def snapshot():
    recipe = json.loads((ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json').read_text())
    manifests = {}
    for arm, run in recipe['run_names'].items():
        p = WORK / 'images' / run / 'inference_manifest.jsonl'
        manifests[arm] = sum(1 for _ in p.open()) if p.exists() else 0
    return dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), memory=memory(),
                free={p: shutil.disk_usage(p).free for p in ['/root', '/root/autodl-tmp', '/dev/shm']},
                image_counts=manifests)


def candidates():
    reg = json.loads((ROOT / 'configs/ATM_public_decoder_registration_20261020_v1.json').read_text())
    result = []
    for subject, item in reg['subjects'].items():
        assert subject in {f'sub-{i:02}' for i in range(1, 11)}
        for kind in ['features', 'prior']:
            p = Path(item[kind])
            assert p.is_relative_to(ATM) and p.resolve() == p
            member = str(p.relative_to(ATM))
            result.append(dict(path=str(p), sha256=item[kind + '_sha256'], storage='ROOT',
                restoration_url='https://huggingface.co/datasets/LidongYang/EEG_Image_decode/resolve/'
                    'dbe34bb2407164f70883c661c76664f0d596e522/' + member,
                reason='Published ATM decoder download; no active ATM decoder job. Brain-HIVE uses its native priors and cached F targets.'))
    assert len(result) == 20
    for member, size, digest, remote in [
        ('clip_vision/model.safetensors', 2528373448,
         '6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030',
         'models/image_encoder/model.safetensors'),
        ('adapter.safetensors', 698391064,
         'ebf05d918348aec7abb02a5e9ecef77e0aaea6914a5c4ea13f50d45eb1681831',
         'sdxl_models/ip-adapter_sdxl_vit-h.safetensors'),
    ]:
        result.append(dict(path=str(RAM / member), sha256=digest, expected_bytes=size, storage='RAM',
            restoration_url='https://huggingface.co/h94/IP-Adapter/resolve/'
                '018e402774aeeddd60609b4ecdb7e298259dc729/' + remote,
            reason='Superseded generic H14/IP weights; native HIVE A/P/IP and current Turbo are separate and retained.'))
    return result


def protected():
    paths = [ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json',
             ROOT / 'queue_BrainHIVE_main_storage_continuation_20261021_v1.py',
             ROOT / 'generate_BrainHIVE_bounded_V_v1.py',
             ROOT / 'brainhive_native_models_v1.py',
             ROOT / 'diagnose_BrainHIVE_native_function_T32_v2.py',
             WORK / 'components/bounded_v1/risk/component_final.pt']
    return {str(p): sha(p, evict=True) for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    if not RECEIPT.exists():
        jobs = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
                if any(x in cmd for x in ['generate_ATM_published_decoder_v1.py',
                                          'run_ATM_published_decoder_v1.py',
                                          'queue_ATM_published_decoder']):
                    jobs.append(dict(pid=proc.name, cmd=cmd))
            except (OSError, ProcessLookupError):
                pass
        assert not jobs, jobs
        entries = candidates()
        for x in entries:
            p = Path(x['path'])
            assert p.resolve() == p
            s = p.lstat()
            assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1, str(p)
            if 'expected_bytes' in x:
                assert s.st_size == x['expected_bytes']
            assert sha(p, evict=True) == x['sha256'], str(p)
            x.update(bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                     dev=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns)
        uses = active_uses([Path(x['path']) for x in entries])
        assert not uses, uses
        report = dict(status='PREPARED_FULL_SHA_AND_NO_FD_OR_MAP_REFERENCE', entries=entries,
            before=snapshot(), protected_before=protected(), active_ATM_jobs=jobs,
            fd_or_map_references=uses, deleted=[], raw_EEG_or_unique_learned_weights_or_images_deleted=0,
            note='Only cached public copies are retired. Metadata, SHA, revisions, URLs, historical reports and source code remain.')
        write(RECEIPT, report)
    report = json.loads(RECEIPT.read_text())
    assert report['status'] == 'PREPARED_FULL_SHA_AND_NO_FD_OR_MAP_REFERENCE'
    if args.apply:
        entries = report['entries']
        assert protected() == report['protected_before']
        assert not active_uses([Path(x['path']) for x in entries])
        for x in entries:
            p = Path(x['path'])
            s = p.lstat()
            assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and p.resolve() == p
            assert (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == (
                x['dev'], x['inode'], x['bytes'], x['mtime_ns'])
            assert sha(p, evict=True) == x['sha256']
            assert not active_uses([p])
            p.unlink()
            report['deleted'].append(x['path'])
            write(RECEIPT, report)
        assert protected() == report['protected_before']
        from brainhive_lifecycle_storage_v1 import verify_remaining
        recipe = json.loads((ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json').read_text())
        report.update(status='EXACT_UNUSED_PUBLIC_COPIES_RETIRED_CURRENT_MAIN_FORECAST_PASSED',
                      after=snapshot(), protected_after=protected(), main_storage_forecast=verify_remaining(recipe))
        write(RECEIPT, report)
    print(json.dumps(dict(status=report['status'], receipt=str(RECEIPT),
        counts={k: sum(x['storage'] == k for x in report['entries']) for k in ['ROOT', 'RAM']},
        allocated_bytes={k: sum(x['allocated_bytes'] for x in report['entries'] if x['storage'] == k) for k in ['ROOT', 'RAM']},
        before=report['before'], after=report.get('after'))))


if __name__ == '__main__':
    main()
