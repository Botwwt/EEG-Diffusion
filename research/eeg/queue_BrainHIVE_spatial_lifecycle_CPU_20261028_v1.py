"""D40 finite serial profile -> three fits -> eight matched image arms.

Wait for the exact D39 image controller's successful terminal lifecycle.
No GPU initialization while waiting, no restart after a scientific/engineering failure.
"""
import argparse,datetime,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_spatial_images_20261028_v1 import read,sha,ref,checked,write,ROOT,B

def process(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,state=s[0],start_ticks=int(s[19]),argv=(p/'cmdline').read_bytes().decode().replace('\0',' '))
    except FileNotFoundError:return None

def atomic(p,v):
    t=p.with_name(p.name+'.pending');write(t,v);os.replace(t,p)

def run(path):
    start=time.monotonic();s=read(path);assert s['status']=='D40_COMPLETE_SPATIAL_VALIDATION_LIFECYCLE_BOUND_WAIT_D39'
    A=ROOT/'audit/spatial_innovation_20261028_v1';Q=B/'lifecycle_v1';assert not Q.exists();Q.mkdir(parents=True)
    active=None;phase='WAIT_D39_COMPLETE_IMAGE_LIFECYCLE';steps=[]
    def pins():
        for x in s['source_pins']:checked(x)
        assert time.monotonic()-start<85000 and shutil.disk_usage(B).free>12<<30
    def stage(label,cmd,seconds):
        nonlocal phase,active
        pins();phase=label;atomic(Q/'progress.json',dict(phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,steps=steps))
        with (Q/(label+'.stdout.log')).open('xb') as out,(Q/(label+'.stderr.log')).open('xb') as err:
            active=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            info=dict(pid=active.pid,process=process(active.pid),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd)
            if label=='training':
                info['start_ticks']=info['process']['start_ticks'];write(A/'actual_training_queue_launch_v1.json',info)
            rc=active.wait(timeout=seconds);active=None
        v=dict(label=label,exit_code=rc,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());write(Q/(label+'_terminal.json'),v);steps.append(v);assert rc==0,(label,rc)
    try:
        previous=read(checked(s['previous_image_launch']));previous_pid=previous['pid'];ticks=previous['process']['start_ticks']
        deadline=datetime.datetime.fromisoformat(previous['UTC'])+datetime.timedelta(seconds=39600)
        oldB=Path(s['previous_candidate_output']);done=oldB/'factorial_images_v1/queue/completion.json'
        while True:
            pins();assert not (oldB/'training_queue_v1/failure.json').exists() and not (oldB/'factorial_images_v1/queue/failure.json').exists(),'D39 failed: preserve its failure; do not dispatch D40'
            p=process(previous_pid);same=p and p['start_ticks']==ticks and p['state'] not in ['Z','X','x']
            if done.exists() and not same:break
            assert same,'Exact previous controller terminal without complete successful image lifecycle'
            assert 'queue_BrainHIVE_measurement_images_CPU_20261028_v1.py' in p['argv'] and datetime.datetime.now(datetime.timezone.utc)<deadline
            atomic(Q/'progress.json',dict(status='D40_ACTUALLY_QUEUED_WAITING_D39_TRAINING_IMAGES_AND_SCORING',phase=phase,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_process=p,new_GPU_jobs=0))
            time.sleep(20)
        previous_done=read(done);assert previous_done['status']=='D39_FOUR_FITS_AND_ALL_TEN_FIXED_IMAGE_ARMS_ACTUALLY_CLOSED'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        ps=checked(s['profile_spec']);profile=read(ps);job=profile['GPU_job_name']
        stage('GPU_engineering',[sys.executable,'-B',str(ROOT/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),'--name',job,'--max-seconds','900','--',sys.executable,'-B',str(ROOT/'profile_BrainHIVE_spatial_innovation_GPU_20261028_v1.py'),'--spec',str(ps)],945)
        proof=read(Path(profile['output'])/'completion.json')
        assert proof['status']=='D40_REAL_NATIVE_EEG_CORE_GRADIENT_AND_PARITY_ENGINEERING_PASSED' and proof['temporary_updates']==6 and proof['zero_update_T_pixel_parity_exact']
        assert proof['peak_GPU_bytes']<20<<30 and proof['peak_extra_CPU_bytes']<10<<30
        ts=checked(s['training_spec'])
        stage('training',[sys.executable,'-B',str(ROOT/'queue_BrainHIVE_spatial_training_CPU_20261028_v1.py'),'--spec',str(ts)],21600)
        # Execution fields refer to the actual training process receipt; the
        # scientific policy, source pins, all eight arms, and gates were locked now.
        v=dict(s['image_execution_template']);v['training_launch']=ref(A/'actual_training_queue_launch_v1.json')
        ep=ROOT/'configs/spatial_innovation_20261028_v1/image_execution_v1.json';write(ep,v)
        stage('eight_arm_images',[sys.executable,'-B',str(ROOT/'queue_BrainHIVE_spatial_images_CPU_20261028_v1.py'),'--execution',str(ep)],18060)
        final=B/'factorial_images_v1/queue/completion.json';read(final)
        write(Q/'completion.json',dict(status='D40_THREE_FITS_EIGHT_IMAGE_ARMS_SCORES_AND_CPU_AUDITS_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(path),previous_lifecycle=ref(done),GPU_profile=ref(Path(profile['output'])/'completion.json'),image_completion=ref(final),steps=steps,seconds=time.monotonic()-start))
    except BaseException as e:
        if active is not None and active.poll() is None:
            try:os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=10)
            except ProcessLookupError:pass
        write(Q/'failure.json',dict(status='D40_LIFECYCLE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',phase=phase,error=repr(e),traceback=traceback.format_exc(),steps=steps,seconds=time.monotonic()-start));raise

if __name__=='__main__':
    def stop(*_):raise InterruptedError('Explicit D40 stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
