"""Hand over only the recorded coordinator after its current worker exits normally."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def proc(owner,fragment=None):
    p=Path('/proc',str(owner['PID']))
    try:st=(p/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return None
    assert int(st[19])==owner['start_ticks'],'PID reused; refuse signal'
    if fragment and st[0] not in ('Z','X'):
        for _ in range(20):
            try:command=(p/'cmdline').read_bytes()
            except FileNotFoundError:return None
            if command:
                assert fragment.encode() in command;break
            # /proc/stat and cmdline are not an atomic snapshot at exit.
            time.sleep(.01)
            try:st=(p/'stat').read_text().rsplit(')',1)[1].split()
            except FileNotFoundError:return None
            assert int(st[19])==owner['start_ticks']
            if st[0] in ('Z','X'):return st[0]
        else:raise AssertionError('Live process has persistently empty cmdline; refuse signal')
    return st[0]


def identity(pid):
    st=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return dict(PID=pid,start_ticks=int(st[19]))


def suspend(owner,fragment):
    assert proc(owner,fragment) in ('S','R')
    os.kill(owner['PID'],signal.SIGSTOP)
    for _ in range(50):
        if proc(owner,fragment)=='T':return
        time.sleep(.02)
    raise RuntimeError('Exact old coordinator did not enter stopped state')


def terminate(owner,fragment):
    state=proc(owner,fragment)
    if state in (None,'Z','X'):return
    os.kill(owner['PID'],signal.SIGTERM)
    if state=='T':os.kill(owner['PID'],signal.SIGCONT)
    for _ in range(100):
        if proc(owner,fragment) in (None,'Z','X'):return
        time.sleep(.1)
    raise RuntimeError('Exact old coordinator did not terminate; no replacement launch')


def normal_receipt(job,wrapper_pid,segment):
    assert job['wrapper_pid']==wrapper_pid and job['status']=='COMPLETED' and job['exit_code']==0
    assert segment['status'] in ('RESUMABLE_FIXED_NATIVE_EEG_SEGMENT_SOFT_LIMIT','COMPLETE_FIXED_EPOCH3_NATIVE_EEG_MODEL')
    assert segment['updates_this_segment']>0


def checks(output):
    """Real Linux signals on own short fixtures, plus damaged-state refusals."""
    assert os.name=='posix';output=Path(output);output.mkdir();cases=[]
    with tempfile.TemporaryDirectory(dir=output) as temp:
        p=Path(temp);stub=p/'own_signal_fixture.py'
        stub.write_text('import subprocess,sys,time\nfrom pathlib import Path\nchild=subprocess.Popen([sys.executable,"-c","import time;time.sleep(0.35)"])\nPath(sys.argv[1]).write_text(str(child.pid))\nchild.wait()\nPath(sys.argv[2]).write_text("normal child finished")\ntime.sleep(30)\n')
        owner=subprocess.Popen([sys.executable,str(stub),str(p/'child'),str(p/'done')],start_new_session=True)
        try:
            for _ in range(100):
                if (p/'child').exists():break
                time.sleep(.01)
            ident=identity(owner.pid);suspend(ident,'own_signal_fixture.py');cases.append('exact_parent_stopped')
            child_finished=False
            for _ in range(100):
                ch=Path('/proc',(p/'child').read_text())
                if not ch.exists() or (ch/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':
                    child_finished=True;break
                time.sleep(.01)
            assert child_finished and proc(ident)=='T' and not (p/'done').exists();cases.append('existing_child_ran_to_exit_while_parent_stopped')
            try:proc(ident|dict(start_ticks=ident['start_ticks']+1),'own_signal_fixture.py')
            except AssertionError:cases.append('reused_PID_refused_without_signal')
            else:raise AssertionError('Wrong ticks accepted')
            terminate(ident,'own_signal_fixture.py');owner.wait(timeout=5);cases.append('stopped_parent_retired_without_signalling_child')
        finally:
            if owner.poll() is None:os.killpg(owner.pid,signal.SIGKILL);owner.wait(timeout=5)
        for bad in (dict(status='FAILED_OR_JOB_TIMEOUT_RESUMABLE',exit_code=0,wrapper_pid=3),dict(status='COMPLETED',exit_code=-15,wrapper_pid=3),dict(status='COMPLETED',exit_code=0,wrapper_pid=4)):
            try:normal_receipt(bad,3,dict(status='RESUMABLE_FIXED_NATIVE_EEG_SEGMENT_SOFT_LIMIT',updates_this_segment=2))
            except AssertionError:cases.append('failed_or_wrong_wrapper_refused')
            else:raise AssertionError('Damaged receipt accepted')
        normal_receipt(dict(status='COMPLETED',exit_code=0,wrapper_pid=3),3,dict(status='RESUMABLE_FIXED_NATIVE_EEG_SEGMENT_SOFT_LIMIT',updates_this_segment=2));cases.append('normal_soft_boundary_accepted')
        spec=importlib.util.spec_from_file_location('priority_scheduler_fixture',Path(__file__).with_name('queue_HIVE_native_direction_first_subject_CPU_20261103_v1.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        r=p/'segments';r.mkdir();assert mod.next_segment(r)==0
        (r/'segment_000').mkdir();(r/'segment_000/completion.json').write_text('{}');assert mod.next_segment(r)==1;cases.append('next_unused_segment_preserves_prior')
        (r/'segment_002').mkdir()
        try:mod.next_segment(r)
        except AssertionError:cases.append('missing_segment_or_failure_refused')
        else:raise AssertionError('Gap accepted')
    record=dict(status='PASS_REAL_LINUX_OWN_PARENT_CHILD_NORMAL_BOUNDARY_AND_REFUSAL_CHECKS',cases=cases,groups=len(cases),new_GPU_jobs=0,new_research_images=0)
    (output/'completion.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-protocol');p.add_argument('--checks-output');a=p.parse_args()
    if a.checks_output:return checks(a.checks_output)
    q=json.loads(Path(a.control_protocol).read_bytes());sys.path.insert(0,str(Path(q['training_protocol']['path']).parent))
    import brainhive_native_direction_full_contract_20261103_v1 as C
    C.checked(q['training_protocol']);s,e,old,base,G,H=C.setup(q['training_protocol']['path'])
    code=Path(q['code_root']);root=Path(q['output_root']);stage='PREFLIGHT';suspended=False;retired=False
    original=q['original_training_owner'];fragment='queue_HIVE_native_direction_full_T_CPU_20261103_v1.py'
    def stop(sig,frame):raise SystemExit('Boundary handoff signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def write(name,value):G.write(root/name,value|dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    try:
        for pin in q['source_pins']:C.checked(pin)
        assert not (code/'actual_launch.json').exists() and not (root/'queue').exists()
        cpuproof=C.read(root/'handoff_CPU_checks/completion.json');assert cpuproof['status']=='PASS_REAL_LINUX_OWN_PARENT_CHILD_NORMAL_BOUNDARY_AND_REFUSAL_CHECKS'
        imagepin=C.ref(q['image_protocol_path']);imagespec=C.read(q['image_protocol_path']);imageout=Path(imagespec['output_root'])
        preflight=C.read(imageout/'CPU_preflight/completion.json');assert preflight['protocol']==imagepin and preflight['status']=='PASS_CPU_SOURCE_MATH_REAL10_PARENT_PROBES_AND_FIXED_GALLERY_POLICY_NOT_POSTFIT_GPU_RESULT'
        assert preflight['new_research_images']==0 and not (imageout/'queue').exists()
        oldimage=C.read(C.checked(q['original_postfit_protocol']));oldout=Path(oldimage['output_root']);assert C.read(oldout/'queue/progress.json')['stage']=='WAIT_SAME_ORIGINAL_TRAINING_OWNER'
        assert all(not (oldout/k).exists() for k in ('conditions','profile','generation','score80','score400'))
        assert time.time()<q['whole_deadline_epoch']-1200 and proc(original,fragment) in ('S','R')
        assert proc(q['original_postfit_owner'],'queue_HIVE_native_direction_postfit_CPU_20261103_v1.py') in ('S','R')
        before=C.resources(s,e,False);write('handoff_intent.json',dict(status='PREPARED_EXACT_ORIGINAL_COORDINATOR_BOUNDARY_TRANSFER',control_protocol=C.ref(a.control_protocol),image_protocol=imagepin,old_training_owner=original,old_postfit_owner=q['original_postfit_owner'],new_GPU_jobs=0,resources=before))
        stage='QUIESCE_OLD_COORDINATOR_ONLY';suspend(original,fragment);suspended=True
        current=C.read(Path(s['output_root'])/'queue/current.json');jp=Path(base['repository_root'])/'audit'/(current['name']+'_job.json')
        # Popen and current.json are separate operations in the old source.
        # Refuse the tiny unrecorded-child interval instead of missing a worker.
        live_children=[]
        children=Path('/proc',str(original['PID']),'task',str(original['PID']),'children').read_text().split()
        for pid in children:
            p=Path('/proc',pid)
            try:st=(p/'stat').read_text().rsplit(')',1)[1].split()
            except FileNotFoundError:continue
            if st[0] not in ('Z','X'):live_children.append(int(pid))
        assert set(live_children).issubset({current['PID']}),'Old coordinator has an unrecorded live child; resume it without takeover'
        job=C.read(jp);assert job['wrapper_pid']==current['PID'];wrapper=identity(current['PID']) if Path('/proc',str(current['PID'])).exists() else None;child=dict(PID=job['pid'])
        if wrapper:proc(wrapper,'run_gpu_job_v2.py')
        write('handoff_waiting_boundary.json',dict(status='OLD_COORDINATOR_ONLY_STOPPED_EXISTING_GPU_WORKER_UNCHANGED',current=current,wrapper=wrapper,child=child,initial_job=C.ref(jp),maximum_boundary_wait_seconds=1040))
        end=time.monotonic()+1040
        while wrapper is not None and proc(wrapper,'run_gpu_job_v2.py') not in (None,'Z','X'):
            assert time.monotonic()<end and proc(original,fragment)=='T'
            time.sleep(1)
        job=C.read(jp);sp=Path(s['output_root'])/'models'/current['subject']/current['arm']/f"segment_{current['segment']:03d}"/'completion.json';done=C.read(sp)
        normal_receipt(job,current['PID'],done);assert done['protocol']==q['training_protocol'] and done['subject']==current['subject'] and done['arm']==current['arm']
        if Path('/proc',str(child['PID'])).exists():assert Path('/proc',str(child['PID']),'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X')
        checkpoint=C.read(Path(s['output_root'])/'models'/current['subject']/current['arm']/'resume_index.json');C.checked(checkpoint['checkpoint'])
        C.resources(s,e,True)
        stage='RETIRE_ONLY_IDLE_OLD_COORDINATORS'
        terminate(q['original_postfit_owner'],'queue_HIVE_native_direction_postfit_CPU_20261103_v1.py')
        terminate(original,fragment);retired=True;suspended=False
        handoff=dict(status='ORIGINAL_CONTROLLER_RETIRED_AFTER_OWN_WORKER_NORMAL_SOFT_OR_FINAL_BOUNDARY',control_protocol=C.ref(a.control_protocol),training_protocol=q['training_protocol'],old_training_owner=original,old_postfit_owner=q['original_postfit_owner'],adopted_segment=C.ref(sp),normal_job=C.ref(jp),current=current,checkpoint=checkpoint,
                     training_source_unchanged=True,no_training_worker_interrupted=True,old_training_failure=C.ref(Path(s['output_root'])/'queue/failure.json'),old_waiter_failure=C.ref(oldout/'queue/failure.json'),administrative_retirement_not_scientific_negative=True,new_GPU_jobs_before_transfer=0)
        write('handoff_completion.json',handoff);stage='START_ONE_CONTINUATION_COORDINATOR'
        review=dict(status='ADMITTED_STAGEWISE_SCHEDULE_ONLY_SAME_TRAINING_AND_IMAGE_SCIENCE',control_protocol=C.ref(a.control_protocol),image_protocol=imagepin,CPU_preflight=C.ref(imageout/'CPU_preflight/completion.json'),real_Linux_checks=C.ref(root/'handoff_CPU_checks/completion.json'),handoff=C.ref(root/'handoff_completion.json'),original_absolute_deadline_unchanged=True,resources=C.resources(s,e,True),new_research_images=0)
        write('actual_conditional_review.json',review)
        cmd=[sys.executable,'-B',str(code/'queue_HIVE_native_direction_first_subject_CPU_20261103_v1.py'),'--protocol',a.control_protocol];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        with (code/'queue.stdout.log').open('xb') as stdout,(code/'queue.stderr.log').open('xb') as stderr:owner=subprocess.Popen(cmd,cwd=code,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        launch=dict(status='STAGEWISE_NATIVE_EEG_COORDINATOR_ACTUALLY_LAUNCHED_AFTER_NORMAL_BOUNDARY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=C.ref(a.control_protocol),image_protocol=imagepin,handoff=C.ref(root/'handoff_completion.json'),review=C.ref(root/'actual_conditional_review.json'),owner=identity(owner.pid),command=cmd,whole_deadline_epoch=q['whole_deadline_epoch'],NEW_contacted=False)
        G.write(code/'actual_launch.json',launch);print(json.dumps(launch),flush=True)
    except BaseException as error:
        if suspended and not retired and proc(original,fragment)=='T':os.kill(original['PID'],signal.SIGCONT)
        write('handoff_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),stage=stage,old_controller_retired=retired,old_controller_resumed_after_failed_pre_retirement_handoff=not retired,no_automatic_retry=True));raise


if __name__=='__main__':main()
