"""Finite OLD-only CPU preparation -> native pixel parity -> paired80/400."""
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from prepare_HIVE_direction_interventions_CPU_20261029_v1 import ref,write
def atomic(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.pending');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)
def run(path):
    s=json.loads(Path(path).read_bytes());assert s['status']=='REGISTERED_ORDINARY_EQUAL_ANGLE_DIRECTION_INTERVENTION' and s['host']=='old'
    root=Path(s['output_root']);assert not root.exists();root.mkdir();logs=root/'queue';logs.mkdir();active=None;phase='CPU_PREPARE';events=[];start=time.monotonic()
    def stage(label,cmd,gpu=True):
        nonlocal active,phase
        phase=label;job='HIVE_direction_'+label+'_20261029_v1'
        if gpu:cmd=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds','900','--']+cmd
        with (logs/(label+'.stdout.log')).open('xb') as out,(logs/(label+'.stderr.log')).open('xb') as err:
            active=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,cwd=s['repository_root']);f=(Path('/proc')/str(active.pid)/'stat').read_text().rsplit(')',1)[1].split();atomic(root/'queue_progress.json',dict(phase=phase,pid=active.pid,start_ticks=int(f[19]),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),events=events));rc=active.wait(timeout=930 if gpu else 180);active=None
        events.append(dict(label=label,exit_code=rc));write(logs/(label+'_terminal.json'),events[-1]);assert rc==0,(label,rc)
    try:
        stage('CPU_prepare',[sys.executable,'-B',s['CPU_preparer']['path'],'--protocol',str(path)],False)
        prepared=ref(root/'prepared/completion.json');execution=dict(status='BOUND_ORDINARY_EQUAL_ANGLE_DIRECTION_IMAGE_EXECUTION',scientific_protocol=ref(path),source_pins=s['source_pins'],D09_completion=s['D09_completion'],prepared_completion=prepared,output_root=str(root/'images'),DINO_weight=s['DINO_weight'])
        ep=root/'image_execution.json';write(ep,execution);out=root/'images';out.mkdir()
        def image_stage(label,mode,n,arms,lo=0,hi=80):
            stage(label,[sys.executable,'-B',s['image_program']['path'],'--execution',str(ep),'--mode',mode,'--stage',str(n),'--lo',str(lo),'--hi',str(hi),'--arms']+arms,mode!='score')
        arms=['population_direction','varying_direction'];image_stage('generate80','generate',80,arms);image_stage('features80','features',80,arms);image_stage('score80','score',80,arms)
        score=json.loads((out/'score80.json').read_bytes());expand=[a for a in arms if score['decisions'][a]['action']=='EXPAND400']
        if expand:
            for lo,hi in ((80,240),(240,400)):image_stage(f'generate{lo:03d}_{hi:03d}','generate',400,expand,lo,hi)
            image_stage('features400','features',400,expand);image_stage('score400','score',400,expand)
        write(root/'completion.json',dict(status='COMPLETE_ORDINARY_EQUAL_ANGLE_DIRECTION_INTERVENTION_FIXED_SCREEN',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),execution=ref(ep),events=events,expanded_arms=expand,stage80=ref(out/'score80.json'),stage400=ref(out/'score400.json') if expand else None,new_model_updates=0,seconds=time.monotonic()-start,NEW_contacted=False))
    except BaseException as e:
        if active is not None and active.poll() is None:
            try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
            except ProcessLookupError:pass
        write(root/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-start));raise
if __name__=='__main__':
    def stop(*_):raise InterruptedError('Explicit direction-intervention stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
