"""Finite native teacher reference diagnostic only, never auto-fits a model."""
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from run_HIVE_terminal_visual_probe_20261029_v1 import ref,write,checked
def run(path):
    s=json.loads(Path(path).read_bytes());root=Path(s['output_root']);assert not root.exists();root.mkdir();logs=root/'queue';logs.mkdir();active=None;phase='INIT';events=[];start=time.monotonic()
    try:
        for x in s['source_pins']:checked(x)
        for mode in ('generate','features','score'):
            phase=mode;cmd=[sys.executable,'-B',s['program']['path'],'--protocol',str(path),'--mode',mode]
            if mode!='score':cmd=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name','HIVE_terminal_visual_'+mode+'_20261029_v1','--max-seconds','900','--']+cmd
            with (logs/(mode+'.stdout.log')).open('xb') as out,(logs/(mode+'.stderr.log')).open('xb') as err:
                active=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,cwd=s['repository_root']);f=(Path('/proc')/str(active.pid)/'stat').read_text().rsplit(')',1)[1].split();temp=root/'progress.pending';temp.write_text(json.dumps(dict(phase=phase,pid=active.pid,start_ticks=int(f[19]),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');os.replace(temp,root/'progress.json');rc=active.wait(timeout=930);active=None
            event=dict(mode=mode,exit_code=rc);events.append(event);write(logs/(mode+'_terminal.json'),event);assert rc==0,event
        write(root/'completion.json',dict(status='COMPLETE_NATIVE_LAST_STEP_PRIVILEGED_VISUAL_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),events=events,score=ref(root/'score80.json'),research_images=160,ordinary_replays=80,training_updates=0,no_auto_fit=True,seconds=time.monotonic()-start,NEW_contacted=False))
    except BaseException as e:
        if active is not None and active.poll() is None:
            try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
            except ProcessLookupError:pass
        write(root/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-start));raise
if __name__=='__main__':
    def stop(*_):raise InterruptedError('Explicit terminal visual probe stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
