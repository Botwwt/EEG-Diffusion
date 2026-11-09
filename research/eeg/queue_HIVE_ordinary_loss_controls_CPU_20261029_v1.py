"""Finite old-card-only full-T controls. Failure is preserved, not retried."""
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from train_HIVE_ordinary_loss_controls_GPU_20261029_v1 import ref,checked,atomic

def run(path):
    s=json.loads(Path(path).read_bytes());root=Path(s['output_root']);assert not root.exists();root.mkdir()
    active=None;events=[];phase='INIT';start=time.monotonic()
    try:
        assert s['host']=='old'
        for x in s['source_pins']:checked(x)
        for arm in ('F_only','token_only'):
            for segment in range(8):
                phase=f'{arm}_{segment:02d}';job='HIVE_loss_controls_'+phase+'_20261029_v1'
                command=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds','900','--',sys.executable,'-B',str(checked(s['trainer'])),'--protocol',str(path),'--arm',arm,'--segment',str(segment)]
                with (root/(phase+'.stdout.log')).open('xb') as out,(root/(phase+'.stderr.log')).open('xb') as err:
                    active=subprocess.Popen(command,cwd=s['repository_root'],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                    stat=(Path('/proc')/str(active.pid)/'stat').read_text().rsplit(')',1)[1].split()
                    atomic(root/'queue_progress.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,wrapper_PID=active.pid,start_ticks=int(stat[19]),command=command,events=events))
                    rc=active.wait(timeout=930);active=None
                event=dict(phase=phase,exit_code=rc,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());events.append(event);atomic(root/(phase+'_terminal.json'),event);assert rc==0,event
                committed=json.loads((root/arm/f'segment_{segment:02d}.json').read_bytes());checked(committed['checkpoint'])
                if (root/arm/'completion.json').exists():break
            else:raise RuntimeError('Finite eight-segment full-fit allocation exhausted; all committed states retained')
        a=json.loads((root/'F_only/provenance.json').read_bytes());b=json.loads((root/'token_only/provenance.json').read_bytes())
        assert a['initial_model_tensor_sha256']==b['initial_model_tensor_sha256'] and a['orders_sha256']==b['orders_sha256']
        atomic(root/'training_completion.json',dict(status='BOTH_FULL_T_LOSS_MECHANISM_CONTROLS_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),events=events,arms={arm:ref(root/arm/'completion.json') for arm in ('F_only','token_only')},identical_initial_head_and_orders=True,ordinary_unchanged=ref(s['ordinary']['path'])==s['ordinary'],new_research_images=0,seconds=time.monotonic()-start))
    except BaseException as e:
        if active is not None and active.poll() is None:
            try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
            except ProcessLookupError:pass
        atomic(root/'queue_failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-start));raise
if __name__=='__main__':
    def stop(*_):raise InterruptedError('Explicit mechanism-control queue stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
