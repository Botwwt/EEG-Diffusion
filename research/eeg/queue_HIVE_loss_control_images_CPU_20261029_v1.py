"""Wait exact full-T fit controller, then fixed paired native images and scores."""
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from train_HIVE_ordinary_loss_controls_GPU_20261029_v1 import ref,atomic
def live(identity):
    try:
        f=(Path('/proc')/str(identity['controller_PID'])/'stat').read_text().rsplit(')',1)[1].split();return f[0] not in ('Z','X') and int(f[19])==identity['start_ticks']
    except FileNotFoundError:return False
def run(path):
    s=json.loads(Path(path).read_bytes());t=json.loads(Path(s['training_protocol']['path']).read_bytes());root=Path(s['output_root']);root.mkdir();logs=root/'queue';logs.mkdir();start=time.monotonic();active=None;events=[];phase='WAIT_EXACT_FIT_QUEUE'
    def stage(label,args,gpu=True):
        nonlocal active,phase
        phase=label;job='HIVE_loss_images_'+label+'_20261029_v1';cmd=[sys.executable,'-B',s['image_program']['path'],'--execution',str(path)]+args
        if gpu:cmd=[sys.executable,'-B',str(Path(t['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds','900','--']+cmd
        with (logs/(label+'.stdout.log')).open('xb') as out,(logs/(label+'.stderr.log')).open('xb') as err:
            active=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,cwd=t['repository_root']);f=(Path('/proc')/str(active.pid)/'stat').read_text().rsplit(')',1)[1].split();atomic(root/'queue_progress.json',dict(phase=phase,pid=active.pid,start_ticks=int(f[19]),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),events=events));rc=active.wait(timeout=930);active=None
        events.append(dict(label=label,exit_code=rc));atomic(logs/(label+'_terminal.json'),events[-1]);assert rc==0,(label,rc)
    try:
        atomic(root/'queue_progress.json',dict(phase=phase,wait_identity=s['training_controller'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        while live(s['training_controller']):assert time.monotonic()-start<18000;time.sleep(5)
        train=Path(t['output_root']);assert not (train/'queue_failure.json').exists();done=json.loads((train/'training_completion.json').read_bytes());assert done['status']=='BOTH_FULL_T_LOSS_MECHANISM_CONTROLS_COMPLETE'
        # Check the complete final model, moments and coverage before images.
        sys.path.insert(0,t['repository_root']);import torch
        torch.set_num_threads(2)
        for arm in ('F_only','token_only'):
            c=json.loads((train/arm/'completion.json').read_bytes());last=torch.load(c['checkpoint']['path'],weights_only=False,map_location='cpu');final=torch.load(c['component']['path'],weights_only=True,map_location='cpu')
            assert ref(c['checkpoint']['path'])==c['checkpoint'] and ref(c['component']['path'])==c['component']
            assert last['coverage'].shape==(3,132300) and bool(last['coverage'].all()) and last['updates']==24807 and last['seen']==396900
            assert all(torch.equal(last['model'][k],final['model'][k]) for k in final['model'])
            assert {int(v['step']) for v in last['optimizer']['state'].values()}=={24807}
        atomic(root/'full_training_CPU_audit.json',dict(status='BOTH_FINAL_HEADS_ADAM_STEPS_AND_ALL3x132300_COVERAGE_CPU_CHECKED',training=ref(train/'training_completion.json'),GPU_initialized=torch.cuda.is_initialized()));assert not torch.cuda.is_initialized()
        arms=['F_only','token_only'];common=['--stage','80','--arms']+arms
        stage('generate80',['--mode','generate','--lo','0','--hi','80']+common);stage('features80',['--mode','features']+common);stage('score80',['--mode','score']+common,False)
        score=json.loads((root/'score80.json').read_bytes());expand=[a for a in arms if score['decisions'][a]['action']=='EXPAND400']
        if expand:
            common=['--stage','400','--arms']+expand
            for lo in range(80,400,80):stage(f'generate{lo:03d}_{lo+80:03d}',['--mode','generate','--lo',str(lo),'--hi',str(lo+80)]+common)
            stage('features400',['--mode','features']+common);stage('score400',['--mode','score']+common,False)
        atomic(root/'completion.json',dict(status='BOTH_LOSS_CONTROL_IMAGES_AND_FIXED_SCREEN_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution=ref(path),events=events,expanded_arms=expand,stage80=ref(root/'score80.json'),stage400=ref(root/'score400.json') if expand else None,seconds=time.monotonic()-start,NEW_contacted=False))
    except BaseException as e:
        if active is not None and active.poll() is None:
            try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
            except ProcessLookupError:pass
        atomic(root/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-start));raise
if __name__=='__main__':
    def stop(*_):raise InterruptedError('Explicit image controller stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);p=argparse.ArgumentParser();p.add_argument('--execution',required=True);run(p.parse_args().execution)
