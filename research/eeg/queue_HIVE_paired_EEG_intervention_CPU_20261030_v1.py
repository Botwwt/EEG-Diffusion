"""Finite dependent diagnostic; does not alter either upstream experiment."""
import argparse,datetime,fcntl,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import atomic,checked,read,ref
def live(h):
    try:f=Path('/proc',str(h['PID']),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return f[0] not in ('Z','X') and int(f[19])==h['start_ticks']
def main(path):
    s=read(path);assert s['status']=='REGISTERED_CONDITIONAL_FROZEN_EEG_INPUT_INTERVENTION400'
    for p in s['source_pins']:checked(p)
    root=Path(s['output_root']);root.mkdir(exist_ok=True);events=root/'lifecycle';events.mkdir(exist_ok=True);lock=(root/'CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'controller_completion.json').exists() and not (root/'controller_failure.json').exists();start=time.monotonic();phase='WAIT_EXACT_BOTH_UPSTREAM';tasks=[];code=Path(path).parent
    def progress(**v):atomic(root/'controller_progress.json',dict(phase=phase,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**v),replace=True)
    def run(label,command,seconds,gpu=False):
        nonlocal phase
        phase=label;assert time.monotonic()-start<s['controller_max_seconds'];job=s['job_prefix']+'_'+label if gpu else None;cmd=list(command)
        if gpu:
            assert not any(live(h) for h in s['upstream_handles'])
            assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            cmd=[sys.executable,'-B',str(Path(s['repository_root'])/'run_gpu_job_v2.py'),'--name',job,'--max-seconds',str(seconds),'--']+cmd
        env=os.environ.copy()
        if gpu:env.pop('CUDA_VISIBLE_DEVICES',None)
        else:env['CUDA_VISIBLE_DEVICES']=''
        with (events/f'{label}.stdout.log').open('xb') as so,(events/f'{label}.stderr.log').open('xb') as se:
            tick=time.monotonic();p=subprocess.Popen(cmd,env=env,cwd=s['repository_root'],stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True);progress(active_PID=p.pid,job_name=job,completed_tasks=len(tasks))
            try:rc=p.wait(timeout=seconds+65)
            except subprocess.TimeoutExpired:
                if gpu:
                    r=read(Path(s['repository_root'])/'audit'/f'{job}_job.json');assert r['wrapper_pid']==p.pid and r['command']==command;c=Path('/proc',str(r['pid']))
                    if c.exists():
                        assert str(command[2]).encode() in (c/'cmdline').read_bytes();os.killpg(r['pid'],signal.SIGTERM);time.sleep(2)
                        if c.exists():os.killpg(r['pid'],signal.SIGKILL)
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise RuntimeError('Finite frozen diagnostic task timeout, no automatic retry')
        value=dict(label=label,exit_code=rc,seconds=time.monotonic()-tick,gpu=gpu,command=command)
        if gpu:value['job_receipt']=ref(Path(s['repository_root'])/'audit'/f'{job}_job.json')
        atomic(events/f'{label}.json',value);tasks.append(value);assert rc==0
    try:
        while True:
            assert time.monotonic()-start<s['maximum_wait_seconds'];completed=[]
            for name,handle,folder in zip(('core','secondary'),s['upstream_handles'],(s['upstream_output_root'],s['secondary_output_root'])):
                folder=Path(folder);alive=live(handle)
                if (folder/'controller_failure.json').exists():atomic(root/'controller_completion.json',dict(status='WITHDRAWN_UPSTREAM_FAILURE_NO_DIAGNOSTIC_GPU',upstream=name,failure=ref(folder/'controller_failure.json'),new_GPU_jobs=0));return
                if (folder/'controller_completion.json').exists():
                    result=read(folder/'controller_completion.json')
                    if name=='core' and (result['final_audit']['stage']!=400 or result['final_audit']['decision']['action']!='SUPPORTED_FOR_COMPLETE_MULTISUBJECT_EVALUATION'):
                        atomic(root/'controller_completion.json',dict(status='WITHDRAWN_CORE_PRIMARY_NOT_PROMOTED_NO_DIAGNOSTIC_RESCUE',new_GPU_jobs=0));return
                    if name=='secondary' and result['status']!='COMPLETE_CONDITIONAL_ALL7_FULL400_SECONDARY_EVALUATION':atomic(root/'controller_completion.json',dict(status='WITHDRAWN_SECONDARY_NOT_COMPLETE_NO_DIAGNOSTIC_GPU',new_GPU_jobs=0));return
                    completed.append(not alive)
                else:assert alive,'Exact upstream handle missing without terminal result; never restart';completed.append(False)
            if all(completed):break
            progress(upstream_exact_live=[live(h) for h in s['upstream_handles']],new_GPU_jobs=0);time.sleep(30)
        py=[sys.executable,'-B'];program=str(code/'run_HIVE_paired_EEG_intervention_20261030_v1.py');run('prepare',py+[program,'--protocol',str(path),'--mode','prepare'],1800)
        for lo in range(0,400,80):
            full=False
            for slot in range(3):
                run(f'images_{lo}_{lo+80}_{slot}',py+[program,'--protocol',str(path),'--mode','generate','--lo',str(lo),'--hi',str(lo+80),'--slot',str(slot)],900,True)
                full=read(root/'attempts'/f'generate_{lo}_{lo+80}_{slot}'/'completion.json')['full_range']
                if full:break
            assert full,'Registered finite image prefix slots exhausted'
        run('features',py+[program,'--protocol',str(path),'--mode','features'],900,True)
        run('score',py+[program,'--protocol',str(path),'--mode','score'],1800)
        run('independent_CPU_audit',py+[program,'--protocol',str(path),'--mode','audit'],1800)
        atomic(root/'controller_completion.json',dict(status='COMPLETE_FIXED400_FROZEN_EEG_INPUT_DIAGNOSIS',protocol=ref(path),tasks=tasks,independent_audit=ref(root/'independent_CPU_audit.json'),new_models=0,new_research_images=1600,upstream_decisions_unchanged=True,SOTA=False))
    except BaseException as ex:atomic(root/'controller_failure.json',dict(phase=phase,error=repr(ex),traceback=traceback.format_exc(),tasks=tasks,no_automatic_retry=True));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
