"""Wait for the exact live training controller, then run the fixed ten-arm screen."""
import argparse,datetime,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import brainhive_measurement_images_20261028_v1 as U

def process(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,state=s[0],start_ticks=int(s[19]),argv=(p/'cmdline').read_bytes().decode().replace('\0',' '))
    except FileNotFoundError:return None
def atomic(path,v):
    tmp=path.with_name(path.name+'.pending');assert not tmp.exists();U.write(tmp,v);os.replace(tmp,path)

def run(path):
    start=time.monotonic();execution,policy,train,base,queries=U.load_execution(path)
    Q=U.EVAL/'queue';assert not Q.exists();Q.mkdir(parents=True)
    steps=[];active=None;phase='WAIT_EXACT_TRAINING_CONTROLLER'
    def progress(**kw):atomic(Q/'progress.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,completed_steps=steps,**kw))
    try:
        launch=U.read(U.checked(execution['training_launch']));deadline=datetime.datetime.fromisoformat(launch['UTC'])+datetime.timedelta(seconds=train['maximum_controller_seconds'])
        while not (U.B/'training_queue_v1/completion.json').exists():
            assert not (U.B/'training_queue_v1/failure.json').exists(),'Training failure retained; do not dispatch images'
            p=process(launch['pid'])
            if not p or p['state'] in ['Z','X','x']:
                assert (U.B/'training_queue_v1/completion.json').exists(),'Original training handle terminal without successful completion; do not restart'
                break
            assert p['start_ticks']==launch['start_ticks'] and 'queue_BrainHIVE_measurement_training_CPU_20261028_v1.py' in p['argv']
            assert datetime.datetime.now(datetime.timezone.utc)<deadline,'Original training deadline not reset by this waiting queue'
            progress(status='D39_WAITING_CONFIRMED_ORIGINAL_TRAINING_PROCESS',actual_training_process=p,original_training_deadline_UTC=deadline.isoformat())
            time.sleep(15)
        U.complete_fits(execution)
        # The training controller must leave before evaluation can claim the
        # GPU lock. A completion JSON alone is not a process-lifecycle proof.
        end=time.monotonic()+120
        while True:
            p=process(launch['pid'])
            if not p or p['state'] in ['Z','X','x'] or p['start_ticks']!=launch['start_ticks']:break
            assert time.monotonic()<end;time.sleep(2)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        with (U.ROOT/'audit/GPU_JOB.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
        eval_start=time.monotonic();phase='PREPARE_AND_VALIDATE_ALL400x10'
        def stage(label,command,seconds):
            nonlocal active,phase
            assert time.monotonic()-eval_start<18000
            for x in execution['source_pins']:U.checked(x)
            assert shutil.disk_usage(U.EVAL).free>8<<30
            phase=label;progress(status='D39_FIXED_IMAGE_PIPELINE_EXECUTING',command=command)
            U.write(Q/(label+'_started.json'),dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command))
            began=time.monotonic()
            with (Q/(label+'_stdout.log')).open('xb') as out,(Q/(label+'_stderr.log')).open('xb') as err:
                active=subprocess.Popen(command,cwd=U.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                rc=active.wait(timeout=seconds);active=None
            v=dict(label=label,command=command,exit_code=rc,seconds=time.monotonic()-began,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
            U.write(Q/(label+'_terminal.json'),v);steps.append(v);assert rc==0,(label,rc)
        def cpu(label,source,extra,seconds=600):
            stage(label,[sys.executable,'-B',str(U.ROOT/source),'--execution',str(path),*extra],seconds)
        def gpu(label,mode,count,lo=0,hi=20):
            name='D39_'+label+'_20261028_v1'
            command=[sys.executable,'-B',str(U.ROOT/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),'--name',name,'--max-seconds','900','--',
                sys.executable,'-B',str(U.ROOT/'generate_BrainHIVE_measurement_images_GPU_20261028_v1.py'),'--execution',str(path),'--mode',mode,
                '--stage',str(count),'--query-start',str(lo),'--query-end',str(hi),'--job-name',name]
            stage(label,command,945)
        cpu('prepare400x10','prepare_BrainHIVE_measurement_images_CPU_20261028_v1.py',[])
        for count,begin in [(80,0),(400,80)]:
            if count==400:
                previous=U.read(U.EVAL/'screen80/score/completion.json')
                if previous['decision']['action']!='EXPAND400':break
                U.stage_gate(path,400)
            for lo in range(begin,count,20):gpu(f'images{count}_{lo:03d}_{lo+20:03d}','generate',count,lo,lo+20)
            cpu('pixel_audit'+str(count),'score_BrainHIVE_measurement_images_CPU_20261028_v1.py',['--stage',str(count),'--mode','audit_images'])
            gpu('features'+str(count),'encode',count)
            cpu('score'+str(count),'score_BrainHIVE_measurement_images_CPU_20261028_v1.py',['--stage',str(count),'--mode','score'])
            cpu('independent'+str(count),'audit_BrainHIVE_measurement_images_CPU_20261028_v1.py',['--stage',str(count)])
            final=count
        done=U.read(U.EVAL/f'screen{final}/score/completion.json');audit=U.read(U.EVAL/f'screen{final}/score/independent_CPU_audit.json')
        assert audit['decision']==done['decision'];U.evaluation_budget()
        v=dict(status='D39_FOUR_FITS_AND_ALL_TEN_FIXED_IMAGE_ARMS_ACTUALLY_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          execution=U.ref(path),training_completion=U.ref(U.B/'training_queue_v1/completion.json'),final_stage=final,decision=done['decision'],
          final_score=U.ref(U.EVAL/f'screen{final}/score/completion.json'),independent_CPU=U.ref(U.EVAL/f'screen{final}/score/independent_CPU_audit.json'),
          research_images=final*10,steps=steps,ordinary_unchanged=True,no_official_or_V_unlocked=True,seconds=time.monotonic()-start)
        U.write(Q/'completion.json',v);print(json.dumps(v),flush=True)
    except BaseException as e:
        if active is not None and active.poll() is None:
            import signal
            try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
            except ProcessLookupError:pass
        U.write(Q/'failure.json',dict(status='D39_FIXED_IMAGE_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',phase=phase,error=repr(e),traceback=traceback.format_exc(),completed_steps=steps,seconds=time.monotonic()-start))
        raise

if __name__=='__main__':
    import signal
    def interrupted(signum,frame):raise InterruptedError('Explicit image-controller stop: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);run(p.parse_args().execution)
