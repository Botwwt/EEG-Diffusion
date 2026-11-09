"""Restore frozen evaluator assets and copy them to the expanded private disk."""
import json
from pathlib import Path
import sys

import restore_BrainHIVE_evaluation_runtime_CPU_20261022_v1 as evaluator
from restore_BrainHIVE_new_instance_20261022_v2 import AUDIT, preserve_runtime_files, sha, write


def main():
    output = AUDIT / 'evaluation_driver.json'
    assert not output.exists()
    original = {str(Path(m.__file__)): sha(m.__file__) for m in (evaluator, evaluator.transfer)}
    sys.argv = [str(Path(evaluator.__file__)), '--workers', '32', '--attempt', '1']
    evaluator.main()
    receipt = evaluator.ROOT / 'audit/runtime_evaluation_public_restore_20261022_v1_attempt1.json'
    proof = json.loads(receipt.read_text())
    assert proof['status'] == 'ALL25_ORIGINAL_PUBLIC_EVALUATOR_ASSETS_EXACT_BYTES_RESTORED'
    assert all(sha(p) == h for p, h in original.items())
    durable = preserve_runtime_files(proof['assets'], 'evaluation')
    write(output, dict(status='ALL25_FROZEN_EVALUATOR_ASSETS_HAVE_VERIFIED_DURABLE_COPIES',
        source_sha256=sha(__file__), helper_sha256=original,
        asset_receipt=dict(path=str(receipt), sha256=sha(receipt)),
        durable_receipt=dict(path=str(AUDIT / 'evaluation_durable.json'), sha256=sha(AUDIT / 'evaluation_durable.json')),
        assets=len(durable['files']), GPU_load_verified=False,
        cleanup_performed=False, formal_training_started=False))
    print(output.read_text(), flush=True)


if __name__ == '__main__':
    main()
