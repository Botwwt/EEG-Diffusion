"""Finite OLD-only training and image lifecycle. Failed tasks never auto-retry."""
import argparse,datetime,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import ARMS,atomic,checked,read,ref

def main(path):
    s=read(path);assert s['status']=='ADMITTED_FULL_T_EEG_P_PAIRED_SEMANTIC_STUDY_V1' and s['host']=='old' and s['training_seed']==1729
    for x in s['source_pins']+s['payloads']:checked(x)
    code=Path(path).parent;root=Path(s['output_root']);root.mkdir(exist_ok=True);events=root/'lifecycle';events.mkdir(exist_ok=True)
    lock=(root/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'controller_completion.json').exists() and not (root/'controller_failure.json').exists()
    begin=time.monotonic();tasks=[];phase='START';active=None
    def run(label,command,seconds,gpu=False):
        nonlocal active,phase
        assert time.monotonic()-begin<s['controller_max_seconds'];phase=label
        assert not (events/f'{label}.json').exists();cmd=list(command);job=None
        if gpu:
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            job=s['job_prefix']+'_'+label;cmd=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds',str(seconds),'--']+cmd
        env=os.environ.copy()
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        with (events/f'{label}.stdout.log').open('xb') as so,(events/f'{label}.stderr.log').open('xb') as se:
            tick=time.monotonic();p=subprocess.Popen(cmd,stdout=so,stderr=se,stdin=subprocess.DEVNULL,env=env,cwd=s['repository_root'],start_new_session=True);active=p
            atomic(root/'controller_progress.json',dict(phase=phase,PID=os.getpid(),active_PID=p.pid,job_name=job,completed_tasks=len(tasks),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),replace=True)
            try:rc=p.wait(timeout=seconds+65)
            except subprocess.TimeoutExpired:
                if gpu:
                    receipt=read(Path(s['repository_root'])/'audit'/f'{job}_job.json');assert receipt['wrapper_pid']==p.pid and receipt['command']==command
                    child=receipt['pid'];proc=Path('/proc',str(child))
                    if proc.exists():
                        assert str(command[2]).encode() in (proc/'cmdline').read_bytes();os.killpg(child,signal.SIGTERM)
                        time.sleep(2)
                        if proc.exists():os.killpg(child,signal.SIGKILL)
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise RuntimeError('Finite task deadline exceeded; preserved prefix, no automatic retry')
        active=None;record=dict(label=label,command=command,exit_code=rc,seconds=time.monotonic()-tick,gpu=gpu)
        if gpu:
            receipt=read(Path(s['repository_root'])/'audit'/f'{job}_job.json');assert receipt['wrapper_pid']==p.pid and receipt['command']==command
            record['GPU_job']=ref(Path(s['repository_root'])/'audit'/f'{job}_job.json');assert receipt['exit_code']==rc
        atomic(events/f'{label}.json',record);tasks.append(record);assert rc==0
    py=[sys.executable,'-B'];image_execution=code/'image_execution.json'
    try:
        for arm in ARMS:
            for slot in range(s['maximum_segments_per_arm']):
                if (root/arm/'completion.json').exists():break
                run(f'fit_{arm}_{slot:03d}',py+[str(code/'train_HIVE_paired_semantic_EEG_P_GPU_20261030_v1.py'),'--protocol',str(path),'--arm',arm,'--segment',str(slot)],900,True)
            assert (root/arm/'completion.json').exists(), 'Registered finite segment slots exhausted'
        auditor=str(code/'audit_HIVE_paired_semantic_study_CPU_20261030_v1.py');image=str(code/'run_HIVE_paired_semantic_images_20261030_v1.py')
        run('fit_CPU_audit',py+[auditor,'--protocol',str(path),'--mode','fit'],1800)
        run('export400',py+[image,'--execution',str(image_execution),'--mode','export'],900,True)
        run('tokens400_CPU_audit',py+[auditor,'--protocol',str(image_execution),'--mode','tokens'],1800)
        image_root=Path(read(image_execution)['output_root'])
        for stage,ranges in [(80,[(0,80)]),(400,[(80,160),(160,240),(240,320),(320,400)])]:
            for lo,hi in ranges:
                full=False
                for slot in range(3):
                    run(f'images{stage}_{lo:03d}_{hi:03d}_{slot}',py+[image,'--execution',str(image_execution),'--mode','generate','--stage',str(stage),'--lo',str(lo),'--hi',str(hi),'--slot',str(slot)],900,True)
                    done=read(image_root/f'generate_{stage}_{lo:03d}_{hi:03d}_{slot}'/'completion.json');full=done['full_range']
                    if full:break
                assert full,'Registered finite image prefix slots exhausted'
            run(f'features{stage}',py+[image,'--execution',str(image_execution),'--mode','features','--stage',str(stage)],900,True)
            run(f'score{stage}',py+[image,'--execution',str(image_execution),'--mode','score','--stage',str(stage)],1800)
            run(f'images{stage}_CPU_audit',py+[auditor,'--protocol',str(image_execution),'--mode','images','--stage',str(stage)],1800)
            result=read(image_root/f'independent_images{stage}_CPU_audit.json')
            if stage==80 and result['decision']['action']=='STOP_INSTANCE':break
        atomic(root/'controller_completion.json',dict(status='COMPLETE_FIXED_FULL_T_AND_NATIVE_IMAGE_STUDY',registration=ref(path),tasks=tasks,final_audit=result,seconds=time.monotonic()-begin,NEW_contacted=False,SOTA=False))
    except BaseException as ex:
        # run() owns/cleans only its exact subprocess on timeout; ordinary failures
        # have already exited. No historical queue or unrelated PID is touched.
        atomic(root/'controller_failure.json',dict(phase=phase,error=repr(ex),traceback=traceback.format_exc(),tasks=tasks,seconds=time.monotonic()-begin,no_automatic_retry=True));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
