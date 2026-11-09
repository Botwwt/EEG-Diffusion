"""Finite profile -> 80 -> conditional 400; no failed-stage retries."""
import argparse, fcntl, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import brainhive_functional_mean_contract_20261103_v1 as G

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-protocol', required=True)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    spec = G.read(args.image_protocol)
    assert spec['status'] == 'REGISTERED_FUNCTIONAL_MEAN_IMAGES_WITH_PAIRING_CONTROLS'
    assert spec['maximum_controller_seconds'] == 7200
    for pin in spec['source_pins']:
        G.checked(pin)
    s = G.setup(G.checked(spec['base_protocol']))
    root, code = Path(s['output_root']), Path(args.image_protocol).parent
    G.checked(spec['statistics_completion']); G.checked(spec['statistics_CPU_audit'])
    G.resources(s, start=True)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if args.preflight:
        assert not (root/'image_queue').exists() and not (root/'profile').exists() and not (root/'generation').exists()
        print(G.json.dumps(dict(status='PASS_COMPLETE_IMAGE_CONTROLLER_PREFLIGHT_NO_CUDA', image_protocol=G.ref(args.image_protocol))))
        return
    destination = root / 'image_queue'
    assert not destination.exists()
    destination.mkdir()
    lock = (destination/'queue.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic(); events = []
    def run(stage, command, seconds, gpu=False, completion=None):
        assert time.monotonic()-started + seconds + 90 < 7200
        G.resources(s, start=gpu)
        if gpu:
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        G.write(destination/(stage+'_started.json'), dict(stage=stage, command=command, UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat()))
        then = time.monotonic()
        with (destination/(stage+'.stdout.log')).open('xb') as stdout, (destination/(stage+'.stderr.log')).open('xb') as stderr:
            child = subprocess.Popen(command, cwd=s['repository_root'], stdout=stdout, stderr=stderr, start_new_session=True)
            G.write(destination/(stage+'_process.json'), dict(PID=child.pid, command=command, stage=stage))
            try:
                rc = child.wait(timeout=seconds+90)
            except subprocess.TimeoutExpired:
                # The wrapper has its own finite child deadline. If it fails to close,
                # terminate only its recorded matching child and this owned wrapper.
                if gpu:
                    receipt = G.read(Path(s['repository_root'])/'audit'/(stage+'_job.json'))
                    pid = receipt['pid']
                    proc = Path('/proc', str(pid), 'cmdline')
                    if proc.exists():
                        actual = proc.read_bytes().split(b'\0')
                        assert str(code/'run_HIVE_functional_mean_images_GPU_20261103_v1.py').encode() in actual
                        assert os.getpgid(pid) == pid
                        os.killpg(pid, signal.SIGTERM)
                else:
                    os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
                raise
        event = dict(stage=stage, returncode=rc, seconds=time.monotonic()-then,
                     stdout=G.ref(destination/(stage+'.stdout.log')), stderr=G.ref(destination/(stage+'.stderr.log')))
        if gpu:
            receipt_path = Path(s['repository_root'])/'audit'/(stage+'_job.json')
            receipt = G.read(receipt_path)
            event['GPU_wrapper_receipt'] = G.ref(receipt_path)
            assert receipt['wrapper_pid'] == child.pid and receipt['status'] == 'COMPLETED' and receipt['exit_code'] == 0, event
        assert rc == 0, event
        if completion is not None:
            assert completion.is_file()
            event['completion'] = G.ref(completion)
        events.append(event)
        G.write(destination/(stage+'_complete.json'), event)
    def gpu(mode, lo, cap):
        stage = 'HIVE_functional_mean_' + (mode if mode == 'profile' else f'images_{lo:03d}_{lo+80:03d}') + '_20261103_v1'
        command = [sys.executable, '-B', str(Path(s['repository_root'])/'run_gpu_job_v2.py'), '--name', stage,
                   '--max-seconds', str(cap), '--', sys.executable, '-B', str(code/'run_HIVE_functional_mean_images_GPU_20261103_v1.py'),
                   '--image-protocol', args.image_protocol, '--mode', mode, '--lo', str(lo), '--hi', str(lo+80), '--job-name', stage, '--max-seconds', str(cap)]
        complete = root/'profile/completion.json' if mode == 'profile' else root/'generation'/f'{lo:03d}_{lo+80:03d}'/'completion.json'
        run(stage, command, cap, gpu=True, completion=complete)
    def score(n):
        run('score'+str(n), [sys.executable, '-B', str(code/'score_HIVE_functional_mean_CPU_20261103_v1.py'),
                            '--image-protocol', args.image_protocol, '--queries', str(n)], 120,
            completion=root/('score'+str(n))/'completion.json')
        return G.read(root/('score'+str(n))/'completion.json')
    try:
        G.write(destination/'started.json', dict(status='FINITE_FUNCTIONAL_MEAN_IMAGE_QUEUE_STARTED', own_PID=os.getpid(),
                UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(), image_protocol=G.ref(args.image_protocol),
                maximum_seconds=7200, automatic_retries=0))
        gpu('profile', 0, 420)
        profile = G.read(root/'profile/completion.json')
        assert profile['status'] == 'PASS_NATIVE_PIXELS_ALL70_REAL_Q_AND_NEW_FIELD_ENGINEERING'
        gpu('generate', 0, 900)
        result = score(80)
        if result['decision']['action'] == 'EXPAND400':
            for lo in (80, 160, 240, 320):
                gpu('generate', lo, 900)
            result = score(400)
        G.resources(s)
        G.write(destination/'completion.json', dict(status='COMPLETE_FIXED_FUNCTIONAL_MEAN_IMAGE_EXPERIMENT',
                image_protocol=G.ref(args.image_protocol), events=events, final_score=G.ref(root/('score'+str(result['queries']))/'completion.json'),
                research_images=result['queries']*6, engineering_images=9, decision=result['decision'],
                seconds=time.monotonic()-started, all_failed_attempts_retained=True, new_training_updates=0,
                independent_image_CPU_audit_pending=True, ordinary_unchanged=True, NEW_contacted=False))
    except BaseException as error:
        G.write(destination/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), events=events,
                seconds=time.monotonic()-started, automatic_retries=0, preserve_all_prefixes=True))
        raise

if __name__ == '__main__':
    main()
