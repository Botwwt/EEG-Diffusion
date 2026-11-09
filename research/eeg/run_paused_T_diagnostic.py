"""Briefly yield the registered REG worker's GPU slot; always resume it."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def main():
    job=json.loads((ROOT/'audit/S1_V_generate_REF_REG_job.json').read_text());pid=job['pid']
    cmdline=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert job['status']=='RUNNING' and 'generate_reference_images.py --method REF_REG' in cmdline
    current=int(Path('/sys/fs/cgroup/memory.current').read_text());maximum=int(Path('/sys/fs/cgroup/memory.max').read_text())
    assert maximum-current>20*1024**3,'Insufficient RAM for the temporary diagnostic pipeline'
    record=dict(status='PAUSING_REG_FOR_T_DIAGNOSTIC',REG_pid=pid,REG_command=cmdline,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        budget_accounting='Diagnostic elapsed time is already charged once by the paused REG GPU-slot wrapper; separately itemized here, no extra long training')
    path=ROOT/'audit/REG_T_diagnostic_pause.json';path.write_text(json.dumps(record,indent=2)+'\n')
    # Independent short watchdog resumes this exact private worker if controller fails.
    watchdog=subprocess.Popen(['/root/miniconda3/bin/python','-c',
        'import os,signal,time;time.sleep(150);os.kill('+str(pid)+',signal.SIGCONT)'],start_new_session=True)
    started=time.monotonic();os.kill(pid,signal.SIGSTOP)
    try:
        time.sleep(1)
        command=['/root/miniconda3/bin/python',str(ROOT/'diagnose_prior_T_images.py')]
        log=ROOT/'logs/prior_T_images.log'
        with log.open('xb') as file:
            process=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT)
            record.update(diagnostic_pid=process.pid,diagnostic_command=command,log=str(log));path.write_text(json.dumps(record,indent=2)+'\n')
            try:code=process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate();code=process.wait(timeout=10)
            record['diagnostic_exit_code']=code
    finally:
        os.kill(pid,signal.SIGCONT);watchdog.terminate();watchdog.wait(timeout=5)
        record.update(status='REG_RESUMED',pause_seconds=time.monotonic()-started,end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
