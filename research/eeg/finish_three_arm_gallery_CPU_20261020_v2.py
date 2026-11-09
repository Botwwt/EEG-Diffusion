"""Bounded export of saved full three-arm images; no model execution or scoring."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
OUTPUT = DATA/'galleries/IDEA1_Frozen_RealPaired_Conditional_complete_20261020_v2'
CACHE = DATA/'galleries/IDEA1_Conditional_RealPaired_V_complete_20261019T231408Z_v1'
RECEIPT = ROOT/'audit/three_arm_gallery_CPU_delivery_20261020_v2.json'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert not OUTPUT.exists() and not RECEIPT.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    commands = [
        [sys.executable, str(ROOT/'build_idea_comparison_gallery_cached_v1.py'),
         '--output', str(OUTPUT), '--thumbnail-cache-root', str(CACHE)],
        [sys.executable, str(ROOT/'verify_idea_gallery_materials_v3.py'), '--gallery', str(OUTPUT)],
    ]
    state = dict(status='RUNNING_CPU_SAVED_IMAGE_EXPORT', pid=os.getpid(),
                 output=str(OUTPUT), commands=commands, phases=[], GPU_used=False,
                 new_scientific_images=0, new_scientific_scores=0,
                 original_storage_gate_and_1GiB_reserve_unchanged=True,
                 started_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 source_sha256=sha(__file__))
    RECEIPT.write_text(json.dumps(state, indent=2)+'\n')
    try:
        for i, command in enumerate(commands):
            t = time.monotonic()
            phase = dict(command=command, status='RUNNING')
            state['phases'].append(phase)
            RECEIPT.write_text(json.dumps(state, indent=2)+'\n')
            result = subprocess.run(command, cwd=ROOT, check=False, timeout=1200 if i == 0 else 120)
            phase.update(status='COMPLETED' if result.returncode == 0 else 'FAILED',
                         returncode=result.returncode, seconds=time.monotonic()-t)
            if result.returncode:
                raise RuntimeError(f'CPU export phase {i} failed with {result.returncode}')
            RECEIPT.write_text(json.dumps(state, indent=2)+'\n')
        state.update(status='COMPLETED_SAVED_THREE_ARM_GALLERY_AND_INDEPENDENT_ROW_CHECK',
                     completion_sha256=sha(OUTPUT/'completion.json'),
                     context_sha256=sha(OUTPUT/'comparison_context.json'),
                     verification_sha256=sha(OUTPUT/'independent_materials_check_v1.json'),
                     browser_UI_QA_performed=False, scientific_goal_complete=False)
    except Exception as e:
        state.update(status='FAILED_CPU_DELIVERY_NOT_A_SCIENTIFIC_NEGATIVE',
                     error_type=type(e).__name__, error=str(e))
    finally:
        state.update(seconds=time.monotonic()-started,
                     ended_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        RECEIPT.write_text(json.dumps(state, indent=2)+'\n')
        with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage='finish_saved_three_arm_gallery',
                                   seconds=state['seconds'], GPU_used=False,
                                   actual_status=state['status'], receipt=str(RECEIPT)))+'\n')
        print(json.dumps(state), flush=True)
    return 0 if state['status'].startswith('COMPLETED_') else 2


if __name__ == '__main__':
    sys.exit(main())
