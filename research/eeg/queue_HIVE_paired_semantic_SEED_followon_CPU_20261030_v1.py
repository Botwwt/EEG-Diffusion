"""Independent finite CPU waiter; no GPU until exact upstream success/support."""
import argparse,datetime,fcntl,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import ARMS,IMAGE_ARMS,atomic,checked,read,ref

def exact_live(pid,ticks):
    try:f=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return f[0] not in ('Z','X') and int(f[19])==ticks
def main(path):
    s=read(path);assert s['status']=='REGISTERED_CONDITIONAL_ALL7_FULL400_SEED_AND_LOWLEVEL_FOLLOWON'
    for pin in s['source_pins']:checked(pin)
    root=Path(s['output_root']);root.mkdir(exist_ok=True);events=root/'lifecycle';events.mkdir(exist_ok=True);lock=(root/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'controller_completion.json').exists() and not (root/'controller_failure.json').exists()
    started=time.monotonic();tasks=[];phase='WAIT_EXACT_UPSTREAM';up=Path(s['upstream_output_root']);handle=s['upstream_controller'];code=Path(path).parent
    def progress(**fields):atomic(root/'controller_progress.json',dict(phase=phase,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields),replace=True)
    def run(label,command,seconds,gpu=False):
        nonlocal phase
        phase=label;assert time.monotonic()-started<s['controller_max_seconds'];assert not (events/f'{label}.json').exists();cmd=list(command);job=s['job_prefix']+'_'+label if gpu else None
        if gpu:
            assert not exact_live(handle['PID'],handle['start_ticks'])
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            cmd=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds',str(seconds),'--']+cmd
        env=os.environ.copy()
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        else:env.pop('CUDA_VISIBLE_DEVICES',None)
        with (events/f'{label}.stdout.log').open('xb') as so,(events/f'{label}.stderr.log').open('xb') as se:
            begin=time.monotonic();p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=so,stderr=se,cwd=s['repository_root'],start_new_session=True);progress(active_PID=p.pid,job_name=job,completed_tasks=len(tasks))
            try:rc=p.wait(timeout=seconds+65)
            except subprocess.TimeoutExpired:
                if gpu:
                    receipt=read(Path(s['repository_root'])/'audit'/f'{job}_job.json');assert receipt['wrapper_pid']==p.pid and receipt['command']==command;child=receipt['pid'];proc=Path('/proc',str(child))
                    if proc.exists():
                        assert str(command[2]).encode() in (proc/'cmdline').read_bytes();os.killpg(child,signal.SIGTERM);time.sleep(2)
                        if proc.exists():os.killpg(child,signal.SIGKILL)
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise RuntimeError('Finite followon task timeout; prefix retained, no failed job retry')
        record=dict(label=label,exit_code=rc,seconds=time.monotonic()-begin,command=command,gpu=gpu)
        if gpu:record['job_receipt']=ref(Path(s['repository_root'])/'audit'/f'{job}_job.json')
        atomic(events/f'{label}.json',record);tasks.append(record);assert rc==0
    try:
        while True:
            assert time.monotonic()-started<s['maximum_wait_seconds']
            alive=exact_live(handle['PID'],handle['start_ticks'])
            if (up/'controller_failure.json').exists():
                atomic(root/'controller_completion.json',dict(status='WITHDRAWN_UPSTREAM_FAILED_NO_SECONDARY_GPU',upstream_failure=ref(up/'controller_failure.json'),GPU_jobs=0));return
            if (up/'controller_completion.json').exists() and not alive:
                result=read(up/'controller_completion.json');assert result['registration']==s['upstream_protocol']
                if result['final_audit']['stage']!=400 or result['final_audit']['decision']['action']!='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION':
                    atomic(root/'controller_completion.json',dict(status='WITHDRAWN_PRIMARY_NOT_PROMOTED_NO_SECONDARY_RESCUE',upstream_completion=ref(up/'controller_completion.json'),GPU_jobs=0));return
                break
            assert alive, 'Exact upstream handle missing without success/failure; never infer completion or restart'
            progress(upstream_exact_live=True,upstream_controller=handle,GPU_jobs=0);time.sleep(30)
        from run_HIVE_paired_semantic_SEED_followon_20261030_v1 import promote
        e=read(checked(s['upstream_image_execution']));promote(s,Path(e['output_root']));py=[sys.executable,'-B'];program=str(code/'run_HIVE_paired_semantic_SEED_followon_20261030_v1.py')
        run('prepare400_exactbytes',py+[program,'--protocol',str(path),'--mode','prepare'],1800)
        for arm in ('target',)+IMAGE_ARMS:
            for stage in ('EffNet','CapSim','ObjectF1'):
                done=False
                for slot in range(3):
                    run(f'{arm}_{stage}_{slot}',py+[program,'--protocol',str(path),'--mode','features','--arm',arm,'--stage',stage,'--slot',str(slot)],900,True)
                    result=read(root/'attempts'/f'{arm}_{stage}_{slot}'/'completion.json');done=result['full_stage']
                    if done:break
                assert done,'Finite feature prefix slots exhausted'
        run('score_all2800',py+[program,'--protocol',str(path),'--mode','score'],1800)
        run('independent_all2800_CPU_audit',py+[str(code/'audit_HIVE_paired_semantic_SEED_CPU_20261030_v1.py'),'--protocol',str(path)],1800)
        atomic(root/'controller_completion.json',dict(status='COMPLETE_CONDITIONAL_ALL7_FULL400_SECONDARY_EVALUATION',protocol=ref(path),tasks=tasks,independent_audit=ref(root/'independent_secondary_CPU_audit.json'),next_action=read(root/'independent_secondary_CPU_audit.json')['next_action'],new_training_or_images=0,seconds=time.monotonic()-started,NEW_contacted=False,SOTA=False))
    except BaseException as ex:atomic(root/'controller_failure.json',dict(phase=phase,error=repr(ex),traceback=traceback.format_exc(),tasks=tasks,no_automatic_retry=True));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
