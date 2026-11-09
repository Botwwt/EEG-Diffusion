"""The fixed wrapper returns0/1; its receipt retains the actual child exit/signal."""
TIMEOUT='Finite registered job timeout; preserve resumable checkpoints'


def verify(wrapper_returncode,receipt,expected_wrapper_pid=None):
    child=receipt['exit_code']
    assert type(child) is int and type(wrapper_returncode) is int
    assert wrapper_returncode==(0 if child==0 else 1)
    assert receipt['status']==('COMPLETED' if child==0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE')
    if expected_wrapper_pid is not None:assert receipt['wrapper_pid']==expected_wrapper_pid
    timed_out=receipt.get('termination_reason')==TIMEOUT
    assert not timed_out or child!=0
    return dict(wrapper_exit_code=wrapper_returncode,child_exit_code=child,
                success=child==0,timed_out=timed_out)
