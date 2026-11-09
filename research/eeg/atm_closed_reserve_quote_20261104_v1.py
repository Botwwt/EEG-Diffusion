"""Explicit terminal-only unused allowance settlement; preserve actual files/floors."""
import os
from pathlib import Path


def allocated(root):
    return sum(p.stat().st_blocks*512 for p in Path(root).rglob('*') if p.is_file())


def verify_and_quote(K, ledger, project, cap):
    assert ledger['status'] == 'REVIEWED_TWO_CLOSED_PROJECT_UNUSED_FUTURE_ALLOWANCES_ONLY'
    assert ledger['legacy_floor_bytes'] == 50 << 30 and ledger['disk_buffer_bytes'] == 8 << 30
    assert ledger['file_deletion'] == 0 and len(ledger['settlements']) == 2
    credit = 0
    for row in ledger['settlements']:
        policy = K.read(K.checked(row['policy']))
        completion = K.read(K.checked(row['completion']))
        launch = K.read(K.checked(row['launch']))
        assert completion['status'] == row['completion_status']
        assert not K.process(launch['PID'], launch['start_ticks'])['live']
        assert allocated(row['project']) == row['actual_allocated_bytes']
        assert policy['maximum_output_bytes'] == row['maximum_output_bytes']
        assert row['unused_future_allowance'] == row['maximum_output_bytes']-row['actual_allocated_bytes']
        for event in completion['events']:
            if event['GPU']:
                j = K.read(K.checked(event['receipt']))
                assert j['status'] == 'COMPLETED' and j['exit_code'] == 0
                assert not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
        credit += row['unused_future_allowance']
    assert credit == ledger['settled_unused_future_bytes']
    previous = ledger['unsettled_snapshot_required_bytes']
    assert previous == ledger['unsettled_full_quote']['required_including_this_project']
    # All non-settled allowances stay at the snapshot value, conservatively:
    # later growth in those directories never causes an unreviewed budget credit.
    prior = previous-credit
    assert prior >= ledger['legacy_floor_bytes']+ledger['disk_buffer_bytes']
    used = allocated(project) if Path(project).exists() else 0
    assert 0 <= used <= cap
    stat = os.statvfs(Path(project).parent)
    free = stat.f_bavail*stat.f_frsize
    required = prior+cap-used
    assert free >= required, dict(free=free, required=required)
    return dict(required_remaining_bytes=prior, required_including_this_project=required,
                free_bytes=free, own_used_bytes=used, own_remaining_bytes=cap-used,
                inherited_unsettled_snapshot_bytes=previous,
                explicitly_settled_closed_unused_future_bytes=credit,
                other_forecasts_held_constant=True, legacy50GiB_and8GiB_buffer_unchanged=True,
                actual_files_deleted=0)
