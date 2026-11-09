"""Source-bound continuation after owned-output cache repair; reuse3 exact folds."""
import argparse,datetime,fcntl,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import brainhive_exposed_OOF_assets_contract_CPU_20261101_v1 as G

def main(path):
    s=G.read(path);tasks,_=G.load_scope(s);root=Path(s['output_root']);code=Path(path).parent
    a=G.read(root/'training_admission.json');assert a['status']=='ALL20_TRAIN_ONLY_ASSET_FITS_REVIEWED_AFTER_ACTUAL_PROFILE' and a['protocol']==G.ref(path)
    recovery=G.read(root/'recovery1_review.json')
    assert recovery['protocol']==G.ref(path) and recovery['original_training_admission']==G.ref(root/'training_admission.json')
    for p in a['execution_sources']+s['source_pins']+recovery['new_sources']:G.checked(p)
    assert a['actual_profile']==G.ref(root/'engineering/completion.json')
    lock=(root/'RECOVERY1_CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    events=root/'lifecycle_recovery1';events.mkdir(exist_ok=False)
    assert not (root/'recovery1_controller_completion.json').exists() and not (root/'recovery1_controller_failure.json').exists()
    begin=time.monotonic();completed=[];phase='START'
    def run(label,command,seconds,gpu):
        nonlocal phase
        phase=label
        seconds=min(seconds,int(recovery['original_controller_deadline_epoch']-time.time())-65)
        assert seconds>0
        for source in a['execution_sources']+s['source_pins']+recovery['new_sources']:G.checked(source)
        job=s['job_prefix']+'_recovery1_'+label if gpu else None;cmd=list(command)
        if gpu:
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==G.GPU
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            cmd=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds',str(seconds),'--']+cmd
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
        with (events/(label+'.stdout.log')).open('xb') as so,(events/(label+'.stderr.log')).open('xb') as se:
            tick=time.monotonic();p=subprocess.Popen(cmd,stdout=so,stderr=se,stdin=subprocess.DEVNULL,env=env,cwd=s['repository_root'],start_new_session=True)
            G.atomic(root/'recovery1_controller_progress.json',dict(PID=os.getpid(),phase=phase,active_PID=p.pid,job_name=job,completed_tasks=len(completed),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),replace=True)
            try:rc=p.wait(timeout=seconds+65)
            except subprocess.TimeoutExpired:
                if gpu:
                    receipt=G.read(Path(s['repository_root'])/'audit'/f'{job}_job.json');assert receipt['wrapper_pid']==p.pid and receipt['command']==command
                    child=Path('/proc',str(receipt['pid']))
                    if child.exists():
                        fields=(child/'stat').read_text().rsplit(')',1)[1].split()
                        actual=(child/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
                        assert int(fields[1])==p.pid and actual==[v.encode() for v in command]
                        os.killpg(receipt['pid'],signal.SIGTERM)
                        try:p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            if child.exists():os.killpg(receipt['pid'],signal.SIGKILL)
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise TimeoutError('Owned finite asset task timed out; preserve states, no automatic retry')
        event=dict(label=label,command=command,exit_code=rc,seconds=time.monotonic()-tick,GPU=gpu)
        if gpu:
            receipt=G.read(Path(s['repository_root'])/'audit'/f'{job}_job.json')
            assert receipt['wrapper_pid']==p.pid and receipt['command']==command and receipt['exit_code']==rc
            event['GPU_receipt']=G.ref(Path(s['repository_root'])/'audit'/f'{job}_job.json')
        G.atomic(events/(label+'.json'),event);completed.append(event);assert rc==0
    try:
        for task_id in sorted(tasks):
            if task_id in recovery['reused_task_ids']:
                for pin in recovery['reused_task_files'][task_id]:G.checked(pin)
                continue
            assert not (root/'tasks'/task_id).exists(), 'No implicit continuation of a partial fold'
            label=task_id.replace('/','__')
            run('fit_'+label,[sys.executable,'-B',str(code/'run_HIVE_exposed_OOF_assets_GPU_20261101_v1.py'),'--protocol',str(path),'--mode','fit','--task',task_id],900,True)
            run('audit_'+label,[sys.executable,'-B',str(code/'audit_HIVE_exposed_OOF_assets_CPU_20261101_v1.py'),'--protocol',str(path),'--mode','fold','--task',task_id],1800,False)
            run('cache_after_'+label,[sys.executable,'-B',str(code/'release_HIVE_owned_OOF_output_cache_CPU_20261101_v1.py'),'--protocol',str(path),'--label','recovery1_after_'+label,'--task',task_id],180,False)
        run('aggregate_all165400',[sys.executable,'-B',str(code/'audit_HIVE_exposed_OOF_assets_CPU_20261101_v1.py'),'--protocol',str(path),'--mode','aggregate'],1800,False)
        bank=G.read(root/'input_bank_completion.json');assert bank['all20_folds_and_all165400_join_values_verified']
        G.atomic(root/'recovery1_controller_completion.json',dict(status='COMPLETE_TRAIN_ONLY_NATIVE20FOLD165400_INPUT_LIFECYCLE_AFTER_CACHE_RECOVERY',recovery_review=G.ref(root/'recovery1_review.json'),reused_task_ids=recovery['reused_task_ids'],protocol=G.ref(path),tasks=completed,input_bank=G.ref(root/'input_bank_completion.json'),seconds=time.monotonic()-begin,official_test_payloads_opened=False,final_component_or_benchmark_admitted=False,NEW_contacted=False,SOTA=False))
    except BaseException as ex:
        G.atomic(root/'recovery1_controller_failure.json',dict(phase=phase,error=repr(ex),traceback=traceback.format_exc(),completed_tasks=completed,seconds=time.monotonic()-begin,no_automatic_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
