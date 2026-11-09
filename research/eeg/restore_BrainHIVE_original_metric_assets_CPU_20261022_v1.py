"""Restore only missing author-metric assets to private durable storage.

One direct request per missing AlexNet/CLIP payload, exact original identities,
no fallback/retry, no models, no image/score access and no GPU initialization.
Every large durable SHA read ends with targeted clean-page-cache advice.
"""
import argparse
import email.parser
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import time
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
import brainhive_functional_training_contract_v2 as C

ROOT = Path(__file__).resolve().parent
SERVER_ROOT = Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
DURABLE = DATA / 'runtime_durable_original_metrics_20261022_v1'
SHARED_DURABLE = DATA / 'runtime_durable_20261022_v2'
AUDIT = ROOT / 'audit/original_metrics_restoration_20261022_v1'
OLD_RECEIPT = Path('/root/brainhive_reproduction_20261020_v1/audit/original_metric_assets_CPU_v3.json')
CLOSURE = ROOT / 'audit/new_instance_restore_20261022_v2/exact_coverage_closure_core_and_evaluation_attempt1.json'
CLOSURE_SHA = '94bec9f9dfac96e0e873de4e984a546b63a8142a6bd6d80ad26ac80802cbe9e6'
PACKAGE_ROOT = DATA / 'runtime/BrainHIVE_original_metrics_0p24_v1'
MIN_MEMORY, BUFFER, MAX_SECONDS, CHUNK = 12 << 30, 1 << 30, 1800, 4 << 20
ASSETS = {
    'AlexNet': dict(path='/dev/shm/BrainHIVE_original_metrics_20261020_v1/alexnet-owt-7be5be79.pth',
        bytes=244408911, sha256='7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
        url='https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'),
    'CLIP_L14': dict(path='/dev/shm/BrainHIVE_original_metrics_20261020_v1/ViT-L-14.pt',
        bytes=932768134, sha256='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836',
        url='https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'),
    'Inception': dict(path='/root/rime_var_20261015/assets/evaluators/inception_v3_google-0cc3c7bd.pth',
        bytes=108949747, sha256='0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef'),
    'SwAV': dict(path='/root/rime_var_20261015/assets/evaluators/swav_800ep_pretrain.pth.tar',
        bytes=113703565, sha256='029d0b8c2e70bcee3f8beb70cc104ef51585c090224d33fb9c6f51d146cfd1eb'),
    'EffNet': dict(path='/dev/shm/eeg_sota_autoresearch_20261018_v1/semantic_evaluation/efficientnet_b1-c27df63c.pth',
        bytes=31600329, sha256='c27df63ce6eb17ef8bcea58922fd3a254cba910c720f41ee89d64d99fb7a4ddf'),
}
SOURCE_PINS = {
    'brainhive_original_metrics_runtime_v1.py': '4cf240478f5301e262206717ba5ca86573da81aebe40ab2a26f592bdad639caa',
    'configs/BrainHIVE_original_image_metrics_registration_v1.json': '9fb9204a2b8c8418bf8b286d48c213bbb4d0e56a240e1531845a06c77323939d',
    'prepare_BrainHIVE_original_metric_assets_CPU_v3.py': 'bf0b64251cf5714842027d5624a2eabfed78321101d2e8771857cc70ede1976f',
    'install_BrainHIVE_original_metric_CPU_runtime_v1.py': 'dc3f25135eee1438d3d73925624b5b132dab2d1c40fe1e9bc78a96e5f1735ad0',
    'check_BrainHIVE_original_metric_math_CPU_v1.py': 'c95b779c69d805217ec1ab055fcf3a0ba8cb53b0c6dc1ccc03ea079d268cf2d5',
    'reproduction_sources/BrainHIVE_f06db69f/main/utils_eval.py': 'ea59145cd6027c027185c57bd43a17f622bc6ec5eded585c847f671e972c6f5a',
    'reproduction_sources/OpenAI_CLIP_d05afc436d78/clip/model.py': '9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c',
    'reproduction_sources/OpenAI_CLIP_d05afc436d78/clip/clip.py': '9540f200fbf8145479fa655382a56dab048d238cc698b9cbd8df3b6d86d3f1b6',
    'reproduction_sources/OpenAI_CLIP_d05afc436d78/LICENSE': '987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db',
}
WHEEL_PINS = {
    'download_manifest_v1.json': ('BrainHIVE_original_metric_CPU_download_manifest_v1.json', '322b0e2ce53e8074b4764b04527b7965d9e967c0f470ff32a9df7dc44c734740'),
    'requirements_v1.txt': ('BrainHIVE_original_metric_CPU_requirements_v1.txt', '760298c94bce6e80d33f4623fab2aa6b1c658692c1be1b2ba589ab59f74c7fb2'),
    'pip_report_v1.json': ('BrainHIVE_original_metric_CPU_pip_report_v1.json', '9a0a90883a87c160b75dd1642d38935f8f3827dc21e457d38cc0c452c59410e1'),
}


def write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    assert len(raw) <= 128 << 10
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def tick(deadline):
    assert time.monotonic() < deadline, 'One finite 1800-second recovery window'


def validate_old_receipt(value):
    assert value['status'] == 'ALL5_AUTHOR_METRIC_ASSETS_STRICT_LOAD_VERIFIED_CPU'
    assert len(value['assets']) == 5 and {r['name'] for r in value['assets']} == set(ASSETS)
    for row in value['assets']:
        expected = ASSETS[row['name']]
        assert all(row[key] == expected[key] for key in expected)


def persistent_path(path, root):
    path, root = Path(path), Path(root)
    assert path.is_absolute() and root.is_absolute() and '..' not in path.parts
    assert root.resolve(strict=True) == root
    assert path != root and path.is_relative_to(root)
    assert path.resolve(strict=False) == path and not path.is_symlink()
    if path.exists():
        assert path.is_file()
    return path


def hash_and_advise(path, asset):
    """Verify the same regular FD and then release its clean durable pages."""
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and path.resolve() == path
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    may_advise = False
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode) and before.st_size == asset['bytes']
        assert before.st_dev != Path('/dev/shm').stat().st_dev, 'Only durable files receive cache advice'
        may_advise = True
        with os.fdopen(os.dup(fd), 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == asset['sha256']
        after = os.fstat(fd)
        facts = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        assert facts == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        return dict(path=str(path), bytes=before.st_size, sha256=digest, device=before.st_dev,
            inode=before.st_ino, mtime_ns=before.st_mtime_ns, clean_pagecache_advised=True)
    finally:
        try:
            if may_advise:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def download_once(asset, target, partial, deadline, *, _opener=None):
    """One URL attempt. A bad/incomplete partial remains; never adopt or retry it."""
    assert not target.exists() and not target.is_symlink()
    assert not partial.exists() and not partial.is_symlink()
    parsed = urllib.parse.urlsplit(asset['url'])
    assert parsed.scheme == 'https' and parsed.hostname in ('download.pytorch.org', 'openaipublic.azureedge.net')
    tick(deadline)
    opener = _opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(asset['url'], headers={'Accept-Encoding': 'identity'})
    count = 0; digest = hashlib.sha256()
    # Create even an empty partial before a failed request, retaining the attempt.
    with partial.open('xb') as stream:
        with opener.open(request, timeout=min(45., max(.1, deadline - time.monotonic()))) as response:
            assert response.status == 200
            assert response.headers.get('Content-Encoding', 'identity') == 'identity'
            assert int(response.headers['Content-Length']) == asset['bytes']
            final = urllib.parse.urlsplit(response.url)
            assert final.scheme == 'https' and final.hostname in (
                'download.pytorch.org', 'openaipublic.azureedge.net', 'openaipublic.blob.core.windows.net')
            while True:
                tick(deadline)
                block = response.read(min(CHUNK, asset['bytes'] - count + 1))
                if not block:
                    break
                count += len(block)
                assert count <= asset['bytes'], 'Pinned payload size cap'
                stream.write(block); digest.update(block)
        stream.flush(); os.fsync(stream.fileno())
    assert count == asset['bytes'] and digest.hexdigest() == asset['sha256']
    assert not target.exists() and not target.is_symlink()
    partial.rename(target)
    return dict(url=asset['url'], network_attempts=1, bytes_received=count)


def link_missing(logical, target):
    logical, target = Path(logical), Path(target)
    assert logical.is_relative_to('/dev/shm/BrainHIVE_original_metrics_20261020_v1')
    assert logical.name in ('alexnet-owt-7be5be79.pth', 'ViT-L-14.pt')
    logical.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert logical.parent.resolve(strict=True) == logical.parent
    assert target.is_file() and not target.is_symlink()
    if logical.exists() or logical.is_symlink():
        assert logical.is_symlink() and logical.resolve(strict=True) == target
        return False
    logical.symlink_to(target)
    assert logical.resolve(strict=True) == target
    return True


def effnet_durable(closure):
    assert closure['status'] == 'EXACT_ORIGINAL_IDENTITY_COVERAGE_RAM_AND_DURABLE_RUNTIME_CLOSURE_VERIFIED'
    assert closure['core_original_records'] == 55 and closure['evaluator_original_records'] == 25
    asset = ASSETS['EffNet']
    rows = [r for stage in closure['stages'] for r in stage['files']
        if r['sha256'] == asset['sha256'] and r['bytes'] == asset['bytes']
        and r['original_path'] == asset['path']]
    assert len(rows) == 1
    return persistent_path(rows[0]['durable_path'], SHARED_DURABLE)


def package_presence(pins):
    metadata = {}
    for label, (filename, expected) in WHEEL_PINS.items():
        path = ROOT / 'audit' / filename
        assert C.sha(path) == expected
        pins[str(path)] = expected
        metadata[label] = path
    manifest = json.loads(metadata['download_manifest_v1.json'].read_text())
    assert {(r['name'], r['version']) for r in manifest} == {('scikit-image', '0.24.0'), ('lazy-loader', '0.4')}
    assert len(manifest) == 2
    report = json.loads(metadata['pip_report_v1.json'].read_text())
    assert {r['download_info']['archive_info']['hashes']['sha256'] for r in report['install']} == {r['sha256'] for r in manifest}
    lock = metadata['requirements_v1.txt'].read_text()
    assert lock == ''.join(r['url'] + ' --hash=sha256:' + r['sha256'] + '\n' for r in manifest)
    assert PACKAGE_ROOT.resolve(strict=True) == PACKAGE_ROOT
    checks = []
    for package, version, directory in [('scikit-image', '0.24.0', 'scikit_image-0.24.0.dist-info'),
            ('lazy-loader', '0.4', 'lazy_loader-0.4.dist-info')]:
        path = PACKAGE_ROOT / directory / 'METADATA'
        content = path.read_bytes(); header = email.parser.BytesParser().parsebytes(content)
        assert header['Name'].lower().replace('_', '-') == package and header['Version'] == version
        pins[str(path)] = hashlib.sha256(content).hexdigest()
        checks.append(dict(name=package, version=version, metadata=str(path), metadata_sha256=pins[str(path)]))
    for name in ('skimage/__init__.py', 'skimage/metrics/_structural_similarity.py',
            'skimage/color/colorconv.py', 'lazy_loader/__init__.py'):
        path = PACKAGE_ROOT / name
        assert path.is_file() and path.stat().st_size > 0
        pins[str(path)] = C.sha(path)
    return checks


def headroom():
    value = C.resource_headroom()
    assert value['effective_headroom_bytes'] >= MIN_MEMORY, 'Keep12GiB raw headroom while formal GPU training continues'
    return value


def lifecycle_space(missing):
    plan_path = ROOT / 'audit/functional_full_lifecycle_20261022_v1/four_arm_lifecycle_plan.json'
    plan = json.loads(plan_path.read_text())
    # The frozen formal_remaining() verifies generated bytes when a manifest
    # contains rows. This preparation runs during first-arm fitting only, so
    # reject later phases before that helper can read even one result image.
    assert all(not Path(entry['inference_manifest']).exists()
        or not Path(entry['inference_manifest']).read_text().strip() for entry in plan['arms']), \
        'Asset preparation must precede formal image generation; no result image bytes may be read'
    recipes = {r['arm']: json.loads(Path(r['path']).read_text()) for r in plan['recipes']}
    remaining = C.formal_remaining(plan, recipes)
    device = DURABLE.stat().st_dev
    required = sum(v['required'] for v in remaining if Path(v['root']).stat().st_dev == device)
    assert required > 0
    free = shutil.disk_usage(DURABLE).free
    assert free >= required + missing + BUFFER, 'Keep the funded four-arm lifecycle, missing assets, and extra1GiB'
    return dict(plan=dict(path=str(plan_path), sha256=C.sha(plan_path)), formal_remaining=remaining,
        asset_missing_bytes=missing, asset_safety_buffer_bytes=BUFFER, durable_free_bytes=free,
        required_on_durable_filesystem=required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT == SERVER_ROOT and sys.version_info[:2] == (3, 12)
    assert 1 <= args.attempt <= 3 and 'torch' not in sys.modules
    AUDIT.mkdir(parents=True, exist_ok=True)
    DURABLE.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert DURABLE.resolve(strict=True) == DURABLE and DURABLE.stat().st_dev == DATA.stat().st_dev
    assert DURABLE.stat().st_dev != Path('/dev/shm').stat().st_dev
    output = AUDIT / f'persistent_asset_recovery_attempt{args.attempt}.json'
    assert not output.exists() and not output.with_name(output.name + '.pending').exists()
    started = time.monotonic(); deadline = started + MAX_SECONDS
    result = dict(status='ORIGINAL_PAPER_METRIC_ASSETS_CPU_RECOVERY_RUNNING',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(), source_sha256=C.sha(__file__),
        maximum_seconds=MAX_SECONDS, network_retry=False, alternate_source_fallback=False,
        model_construction=False, GPU_used=False, image_values_read=0, score_values_read=0,
        deletion_performed=False, old_receipts_or_source_changed=False, assets=[], network_requests_started=[])
    write(output, result)
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout(_signal, _frame):
        raise TimeoutError('Finite1800-second original-asset recovery boundary; partial files retained')
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        pins = {str(ROOT / name): digest for name, digest in SOURCE_PINS.items()}
        assert all(C.sha(path) == digest for path, digest in pins.items())
        original = json.loads(OLD_RECEIPT.read_text()); validate_old_receipt(original)
        pins[str(OLD_RECEIPT)] = C.sha(OLD_RECEIPT)
        assert C.sha(CLOSURE) == CLOSURE_SHA
        pins[str(CLOSURE)] = CLOSURE_SHA
        closure = json.loads(CLOSURE.read_text())
        source_manifest = ROOT / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json'
        pins[str(source_manifest)] = C.sha(source_manifest)
        sources = json.loads(source_manifest.read_text())['source_sha256']
        assert isinstance(sources, dict) and len(sources) == 29
        for name, digest in sources.items():
            assert Path(name).name == name and C.sha(ROOT / name) == digest
            pins[str(ROOT / name)] = digest
        packages = package_presence(pins)
        result['memory_before'] = headroom()
        targets = {name: persistent_path(DURABLE / Path(ASSETS[name]['path']).name, DURABLE) for name in ('AlexNet', 'CLIP_L14')}
        missing = sum(ASSETS[name]['bytes'] for name, path in targets.items() if not path.exists())
        result.update(source_and_metadata_sha256=pins, package_files_present=packages,
            original29_source_SHA_verified=29, storage_before=lifecycle_space(missing))
        write(output, result)
        for name, asset in ASSETS.items():
            tick(deadline); headroom()
            if name in targets:
                target = targets[name]
                logical = Path(asset['path'])
                if logical.exists() or logical.is_symlink():
                    # Only an already verified exact durable alias can be reused.
                    assert logical.is_symlink() and logical.resolve(strict=True) == target
                received = None
                if not target.exists():
                    missing = sum(ASSETS[n]['bytes'] for n, path in targets.items() if not path.exists())
                    lifecycle_space(missing); headroom()
                    partial = target.with_name(target.name + f'.attempt{args.attempt}.partial')
                    result['network_requests_started'].append(dict(name=name, url=asset['url'], partial_path=str(partial)))
                    write(output, result)
                    received = download_once(asset, target, partial, deadline)
                verified = hash_and_advise(target, asset)
                linked = link_missing(logical, target)
                assert C.sha(logical) == asset['sha256']
                # Re-reading the new alias warms the same durable inode.
                verified = hash_and_advise(target, asset)
                record = dict(name=name, logical_path=str(logical), persistent_asset=verified,
                    newly_created_logical_symlink=linked, downloaded=received is not None, download=received)
            elif name == 'EffNet':
                target = effnet_durable(closure)
                assert C.sha(asset['path']) == asset['sha256']
                record = dict(name=name, logical_path=asset['path'], persistent_asset=hash_and_advise(target, asset),
                    reused_from_original80_asset_closure=True, downloaded=False, newly_created_logical_symlink=False)
            else:
                target = Path(asset['path'])
                assert target.resolve(strict=True) == target and not target.is_symlink()
                record = dict(name=name, logical_path=asset['path'], persistent_asset=hash_and_advise(target, asset),
                    preexisting_private_durable_asset=True, downloaded=False, newly_created_logical_symlink=False)
            result['assets'].append(record)
            result['memory_latest'] = headroom(); write(output, result)
        assert len(result['assets']) == 5 and len(result['network_requests_started']) <= 2
        assert all(C.sha(path) == digest for path, digest in pins.items())
        assert C.sha(__file__) == result['source_sha256'] and 'torch' not in sys.modules
        result.update(status='ALL5_PERSISTENT_ASSET_SHA_READY_CPU_MODEL_PROBE_PENDING',
            end_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seconds=time.monotonic()-started,
            memory_after=headroom(), storage_after=lifecycle_space(0),
            original29_and_original_metric_sources_and_metadata_unchanged=True,
            CPU_strict_model_load_pending=True, GPU_model_probe_pending=True, full_image_evaluation_verified=False,
            package_import_probe_pending=True, formal_training_or_candidate_or_metric_changed=False)
        write(output, result)
        print(json.dumps(dict(status=result['status'], receipt=str(output), sha256=C.sha(output))), flush=True)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(status='ORIGINAL_METRIC_ASSET_RECOVERY_FAILED_PARTIALS_AND_PRIOR_RESULTS_PRESERVED',
            error_type=type(error).__name__, error=str(error)[:600], seconds=time.monotonic()-started)
        write(output, result)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    main()
