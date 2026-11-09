"""Withdraw only the unstarted two-idea queue for the user's mechanism review."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')


def main():
    state_path = WORK / 'audit/two_ideas_queue_v1.json'
    previous = json.loads(state_path.read_text())
    assert previous['pid'] == 192644
    assert previous['status'] == 'WAITING_FOR_EXISTING_MAIN_LIFECYCLE_AND_EXCLUSIVE_GPU'
    assert previous['formal_new_training_started'] is False
    command = Path('/proc/192644/cmdline').read_bytes().replace(b'\0', b' ').decode()
    expected = str(ROOT / 'queue_BrainHIVE_two_ideas_v1.py')
    assert expected in command and '--sequence' in command
    sequence = ROOT / 'configs/BrainHIVE_two_ideas_sequence_v1.json'
    assert hashlib.sha256(sequence.read_bytes()).hexdigest() == '809a16b87d44a1dd28301ff8f41cb0d82d84e6f25c77a31024aeff8f7431d93a'
    assert not (WORK / 'two_ideas_v1/recipes').exists()
    assert not (WORK / 'runs/brainkan_full_v1').exists()
    assert not (WORK / 'components/two_ideas_v1/response_rank_v1').exists()
    receipt_path = ROOT / 'audit/BrainHIVE_two_ideas_withdrawn_before_training_v1.json'
    assert not receipt_path.exists()
    utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    receipt = dict(
        status='WITHDRAWING_UNSTARTED_NEW_QUEUE_FOR_USER_MECHANISM_REVIEW', utc=utc,
        reason='User makes KAN optional and prioritizes solving a demonstrated Brain-HIVE architecture defect.',
        previous_queue=previous, exact_process_command=command,
        protected_existing_main_pid=185943, signal_target_pid=192644,
        new_formal_training_started=False, new_images=0,
        source_recipes_preserved=True, GPU_jobs_stopped=0,
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    os.kill(192644, signal.SIGTERM)
    for _ in range(30):
        process = Path('/proc/192644/stat')
        if not process.exists() or process.read_text().split()[2] == 'Z':
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Queue did not exit; do not claim withdrawal complete')
    os.kill(185943, 0)
    final = dict(
        status='WITHDRAWN_BEFORE_NEW_TRAINING_PENDING_MECHANISM_REVIEW', utc=utc,
        previous_pid=192644, fixed_order_preserved_in_registration=['brainkan_full_v1', 'response_rank_v1'],
        formal_new_training_started=False, automatic_launch_enabled=False,
        previous_state_preserved_at=str(receipt_path), existing_main_preserved=True,
    )
    pending = state_path.with_name(state_path.name + '.pending')
    pending.write_text(json.dumps(final, indent=2) + '\n')
    pending.replace(state_path)
    receipt.update(status='WITHDRAWN_BEFORE_NEW_TRAINING_PENDING_MECHANISM_REVIEW',
                   queue_exit_verified=True, existing_main_live_verified=True, final_queue=final)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
