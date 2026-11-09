"""Recheck migrated result images and weights without replacing old evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import verify_BrainHIVE_results_after_expanded_cleanup_CPU_20261022_v1 as previous

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit/new_instance_restore_20261022_v2'


def main():
    source = previous.sha(previous.__file__)
    assert source == '223cd6e33cd2ca72c36cbf9caed523a1b7c611de14210227c150f681433857fd'
    previous.OUT = OUT / 'migrated_results_verified.json'
    previous.main()
    assert previous.sha(previous.__file__) == source
    import shutil
    observations = dict(status='MIGRATED_RESULT_PIXELS_MODELS_AND_FROZEN_SOURCES_REVERIFIED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cleanup_performed=False,
        source_sha256=previous.sha(__file__), original_verifier_sha256=source,
        result_proof_sha256=previous.sha(previous.OUT),
        disk={p: dict(zip(('total', 'used', 'free'), shutil.disk_usage(p))) for p in ('/root', '/root/autodl-tmp', '/dev/shm')},
        GPU_observed=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
        old_SSH_helper_HEAD_sha256=hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:ssh_rpc.py'], cwd=ROOT)).hexdigest(),
        hostname=subprocess.check_output(['hostname'], text=True).strip(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        formal_training_started=False)
    with (OUT / 'instance_observation.json').open('x') as stream:
        json.dump(observations, stream, indent=2); stream.write('\n')
    print(json.dumps(observations), flush=True)


if __name__ == '__main__':
    main()
