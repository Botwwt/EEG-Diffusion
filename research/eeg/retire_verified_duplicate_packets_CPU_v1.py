"""Remove two old transport ZIPs only if every byte survives in its source.

Unique historical versions or unavailable sources retain their entire archive.
Only these two regular files can be removed; no extraction or recursive deletion.
"""
from pathlib import Path
import hashlib
import json
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/avde_rime_oracle_latent_20261015')
NAMES = ['completed_generation_packet_v2.zip', 'completed_nll_packet_v1.zip']


def digest(f):
    return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    start = time.monotonic()
    proof = ROOT/'audit/duplicate_transport_packet_retirement_20261020_v1.json'
    assert not proof.exists()
    assert BASE.resolve() == BASE and BASE.parent == Path('/root/autodl-tmp')
    before = shutil.disk_usage(BASE).free
    records = []
    for name in NAMES:
        path = BASE/name
        assert path.resolve().parent == BASE and not path.is_symlink() and path.is_file()
        with path.open('rb') as f:
            archive_sha = digest(f)
        record = dict(path=str(path), bytes=path.stat().st_size, sha256=archive_sha,
                      status='RETAINED_UNTIL_ALL_SOURCES_VERIFIED', failures=[], verified_members=0)
        with zipfile.ZipFile(path) as z:
            manifest_bytes = z.read('packet_manifest.json')
            manifest = json.loads(manifest_bytes)
            manifest_copy = ROOT/'audit'/f'{name}.packet_manifest.preserved_20261020_v1.json'
            with manifest_copy.open('xb') as f:
                f.write(manifest_bytes)
            record['preserved_manifest'] = str(manifest_copy)
            names = [x['archive_path'] for x in manifest['files']]
            assert len(set(names)) == len(names)
            assert set(z.namelist()) == set(names) | {'packet_manifest.json'}
            for entry in manifest['files']:
                member, source = entry['archive_path'], Path(entry['source'])
                assert z.getinfo(member).file_size == entry['bytes']
                with z.open(member) as f:
                    assert digest(f) == entry['sha256'], ('Archive byte mismatch', member)
                if not source.is_absolute() or not source.is_file():
                    record['failures'].append(dict(member=member, source=str(source), reason='Source unavailable'))
                    continue
                assert source.resolve().is_relative_to('/root')
                with source.open('rb') as f:
                    same = source.stat().st_size == entry['bytes'] and digest(f) == entry['sha256']
                if same:
                    record['verified_members'] += 1
                else:
                    record['failures'].append(dict(member=member, source=str(source), reason='Different historical source bytes'))
            record['members'] = len(names)
        if not record['failures']:
            assert record['verified_members'] == record['members']
            assert path.resolve() == BASE/name
            path.unlink()
            assert not path.exists()
            record['status'] = 'REMOVED_BYTE_VERIFIED_REDUNDANT_TRANSPORT_ZIP'
        else:
            record['status'] = 'RETAINED_ARCHIVE_WITH_UNIQUE_OR_UNAVAILABLE_SOURCE'
        records.append(record)
    result = dict(status='COMPLETED_CONSERVATIVE_PACKET_CLEANUP', records=records,
                  free_bytes_before=before, free_bytes_after=shutil.disk_usage(BASE).free,
                  GPU_used=False, seconds=time.monotonic()-start,
                  original_datasets_checkpoints_images_and_source_deleted=0)
    proof.write_text(json.dumps(result, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage='retire_verified_duplicate_transport_packets',
                               seconds=result['seconds'], GPU_used=False, receipt=str(proof)))+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
