"""Finite OLD GPU diagnostic followed by CPU audit; zero optimizer updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import importlib.util, subprocess, sys, time, traceback
from pathlib import Path
import enigma_repeat_task_risk_contract_20261105_v1 as C

qpath = Path(sys.argv[1])
c = C.setup(qpath)
q, out = c['q'], c['out']
assert C.ref(__file__) == q['sources']['controller']
spec = importlib.util.spec_from_file_location('pinned_owned_process', C.checked(q['owned_process']))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
started = time.monotonic()
owner = child = None
until = time.monotonic() + 15
while not (qpath.parent / 'actual_launch.json').exists():
    assert time.monotonic() < until
    time.sleep(.05)
launch = C.read(qpath.parent / 'actual_launch.json')
assert launch['PID'] == os.getpid() and C.process(os.getpid())['start_ticks'] == launch['start_ticks']
repo = Path(q['GPU_wrapper']['path']).parent
label = q['job_prefix'] + '_measure'
receipt = repo / 'audit' / (label + '_job.json')
try:
    C.resources(c, starting=True)
    assert time.time() + 1000 < launch['absolute_deadline_epoch'] and not receipt.exists()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    command = [sys.executable, '-B', q['GPU_wrapper']['path'], '--name', label, '--max-seconds',
               str(q['maximum_GPU_job_seconds']), '--', sys.executable, '-B', q['sources']['GPU']['path'], '--policy', str(qpath)]
    with (out / 'measure.log').open('xb') as log:
        owner = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        child = P.capture_wrapper_child(owner, receipt)
        rc = owner.wait(timeout=q['maximum_GPU_job_seconds'] + 35)
    job = C.read(receipt)
    assert rc == 0 and job['status'] == 'COMPLETED' and job['exit_code'] == 0
    assert not C.process(job['pid'])['live'] and not C.process(job['wrapper_pid'])['live']
    owner = child = None
    C.resources(c, starting=True)
    with (out / 'CPU_audit.log').open('xb') as log:
        owner = subprocess.Popen([sys.executable, '-B', q['sources']['CPU']['path'], str(qpath)], cwd=qpath.parent,
                                 env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        cpu_owner = C.process(owner.pid)
        C.write(out / 'CPU_audit_owner.json', cpu_owner)
        rc = owner.wait(timeout=min(900, launch['absolute_deadline_epoch'] - time.time()))
    assert rc == 0 and not C.process(cpu_owner['PID'], cpu_owner['start_ticks'])['live']
    audit = C.read(out / 'independent_CPU_audit.json')
    assert audit['status'] == 'PASS_FIXED240_ORIGINAL_CPU64_FORWARDS_AND_ALL_T_REDUCTIONS'
    C.write(out / 'completion.json', dict(status='COMPLETE_T_ONLY_ALL13230_ALL15_ENCODER_OOF_TASK_RISK_AND_CPU_AUDIT',
        policy=C.ref(qpath), scientific_completion=C.ref(out / 'scientific_completion.json'),
        CPU_audit=C.ref(out / 'independent_CPU_audit.json'), receipt=C.ref(receipt),
        seconds=time.monotonic() - started, absolute_deadline_epoch=launch['absolute_deadline_epoch'],
        new_models=0, new_updates=0, new_images=0, no_official_R80_arrays_read=True,
        no_automatic_followup_training=True, originality_or_SOTA_not_established=True))
except BaseException as exc:
    cleanup = P.stop_owned(owner, child)
    C.write(out / 'controller_failure.json', dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY',
        error=repr(exc), traceback=traceback.format_exc(), cleanup=cleanup, seconds=time.monotonic() - started))
    raise
