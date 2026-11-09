"""Record completed storage maintenance without changing scientific recipes."""
import json
import os
from pathlib import Path
import shutil
import time

from relocate_retired_ATM_features_CPU_20261021_v1 import ROOT, memory, sha, write
from brainhive_lifecycle_storage_v1 import verify_remaining


def main():
    work = Path('/root/brainhive_reproduction_20261020_v1')
    dest = ROOT / 'audit/BrainHIVE_storage_cleanup_20261021_v2.json'
    assert not dest.exists()
    public_path = ROOT / 'audit/unused_public_assets_retirement_20261021_v1.json'
    shared_path = ROOT / 'audit/BrainHIVE_retired_shared_RAM_relocation_20261021_v1.json'
    packet_path = ROOT / 'audit/BrainHIVE_historical_packet_member_retirement_20261021_v1.json'
    public = json.loads(public_path.read_text())
    shared = json.loads(shared_path.read_text())
    assert public['status'] == 'EXACT_UNUSED_PUBLIC_COPIES_RETIRED_CURRENT_MAIN_FORECAST_PASSED'
    assert shared['status'] == 'BYTE_IDENTICAL_LINK_AND_MAIN_FORECAST_VERIFIED'
    assert len(public['deleted']) == 22
    assert all(not Path(p).exists() for p in public['deleted'])
    assert Path(shared['original']).is_symlink() and sha(shared['original']) == shared['sha256']
    for p, digest in public['protected_before'].items():
        assert sha(p, evict=True) == digest
    recipe = json.loads((ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json').read_text())
    queue_path = work / 'audit/main_storage_continuation_20261021_v1.json'
    queue = json.loads(queue_path.read_text())
    proc = Path('/proc') / str(queue['pid'])
    assert proc.exists() and proc.joinpath('stat').read_text().split()[2] not in ['Z', 'X']
    assert 'queue_BrainHIVE_main_storage_continuation_20261021_v1.py' in proc.joinpath('cmdline').read_bytes().decode(errors='replace')
    counts = {}
    for arm, run in recipe['run_names'].items():
        p = work / 'images' / run / 'inference_manifest.jsonl'
        rows = [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        ids = [x['global_query_index'] for x in rows]
        assert len(ids) == len(set(ids))
        counts[arm] = len(rows)
    assert counts['risk'] > public['before']['image_counts']['risk']
    packet = json.loads(packet_path.read_text()) if packet_path.exists() else None
    assert packet and packet['status'] == 'REDUNDANT_ZIP_CONTAINER_RETIRED_ALL_ORIGINAL_MEMBER_BYTES_PRESERVED'
    assert packet['durable_members_freshly_verified'] == 75 and packet['raw_manifest_freshly_verified']
    report = dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        status='MAINTENANCE_COMPLETE_PROTECTED_SOURCE_HASHES_AND_RUNNING_MAIN_FORECAST_PASSED',
        public_retirement_receipt=str(public_path), shared_relocation_receipt=str(shared_path),
        packet_retirement_receipt=str(packet_path) if packet else None,
        public_ROOT_allocated_bytes_retired=sum(x['allocated_bytes'] for x in public['entries'] if x['storage'] == 'ROOT'),
        public_RAM_allocated_bytes_retired=sum(x['allocated_bytes'] for x in public['entries'] if x['storage'] == 'RAM'),
        shared_ROOT_allocated_bytes_retired=shared['original_allocated_bytes'],
        shared_derived_RAM_bytes_added=shared['bytes'], RAM_is_persistent_backup=False,
        raw_EEG_unique_learned_checkpoints_or_historical_images_deleted=0,
        large_C_or_D_files_written=0, scientific_recipe_or_model_changes=False,
        current_image_counts=counts, main_pid=queue['pid'], main_status=queue['status'],
        free_bytes={p:shutil.disk_usage(p).free for p in ['/root', '/root/autodl-tmp', '/dev/shm']},
        memory=memory(), main_remaining_forecast=verify_remaining(recipe),
        protected_sha256=public['protected_before'],
        deferred_monitor_reconciliation=dict(
            observed_state=json.loads((work / 'audit/deferred_shared_conditions_maintenance_20261021_v1.json').read_text()),
            final_relocation_status=shared['status'],
            note='The deferred observer saw the prepared receipt while this user-authorized immediate helper ran. The completed helper receipt is authoritative; no retry is needed.'))
    if packet:
        report['packet_result'] = packet
        report['persistent_bytes_reclaimed_lower_bound'] = (
            report['public_ROOT_allocated_bytes_retired'] + report['shared_ROOT_allocated_bytes_retired']
            + packet['predicted_minimum_net_reclaimed_bytes'])
    report['net_shm_allocated_bytes_released'] = (report['public_RAM_allocated_bytes_retired']
        - Path(shared['RAM_target']).stat().st_blocks * 512)
    write(dest, report)
    print(json.dumps({k:report[k] for k in ['UTC', 'status', 'current_image_counts', 'free_bytes',
        'public_ROOT_allocated_bytes_retired', 'public_RAM_allocated_bytes_retired',
        'shared_ROOT_allocated_bytes_retired', 'main_remaining_forecast']}))


if __name__ == '__main__':
    main()
