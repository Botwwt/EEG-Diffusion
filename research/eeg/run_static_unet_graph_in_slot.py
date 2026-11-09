"""Brief bounded diagnostic inside the already-accounted REG GPU allocation.

Only the verified own REG worker is stopped; finally and an independent watchdog
resume it. The diagnostic uses a private frozen model; all formal state survives.
"""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def write(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def main():
    job=json.loads((ROOT/'audit/S1_V_generate_REF_REG_attempt2_job.json').read_text());pid=job['pid']
    cmd=Path('/proc')/str(pid)/'cmdline';assert job['status']=='RUNNING'
    text=cmd.read_bytes().replace(b'\0',b' ').decode();assert str(ROOT/'generate_reference_images.py')+' --method REF_REG' in text
    current=int(Path('/sys/fs/cgroup/memory.current').read_text());limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
    assert limit-current>20*1024**3,'Private graph probe needs20GiB measured RAM headroom'
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>16*1024,'Private graph probe needs16GiB measured GPU headroom'
    receipt=ROOT/'audit/REG_static_unet_graph_cost_slot.json';assert not receipt.exists()
    record=dict(status='PREPARED_BOUNDED_SLOT_DIAGNOSTIC',REG_pid=pid,REG_command=text,
        start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        budget_accounting='All elapsed time is charged ONCE inside existing REG wrapper; this is not an additional parallel GPU allocation')
    write(receipt,record)
    # If this controller dies, kill only this exact graph-probe child then resume the
    # exact private REG worker. All paths/PIDs are read from our own receipt.
    guard_code='''from pathlib import Path
import os,json,signal,time
time.sleep(145)
r=Path(%r)
d=json.loads(r.read_text());child=d.get('diagnostic_pid')
try:
 if child:
  p=Path('/proc')/str(child)/'cmdline'
  if p.exists() and b'profile_static_unet_graph.py' in p.read_bytes():
   os.kill(child,signal.SIGTERM);time.sleep(5)
   if p.exists() and b'profile_static_unet_graph.py' in p.read_bytes():os.kill(child,signal.SIGKILL)
except (FileNotFoundError,ProcessLookupError):pass
finally:
 try:
  p=Path('/proc')/str(d['REG_pid'])/'cmdline'
  if p.exists() and b'generate_reference_images.py' in p.read_bytes():os.kill(d['REG_pid'],signal.SIGCONT)
 except (FileNotFoundError,ProcessLookupError):pass
''' % str(receipt)
    watchdog=subprocess.Popen(['/root/miniconda3/bin/python','-c',guard_code],start_new_session=True)
    started=time.monotonic();process=None;os.kill(pid,signal.SIGSTOP)
    try:
        time.sleep(1)
        command=['/root/miniconda3/bin/python',str(ROOT/'profile_static_unet_graph.py')]
        log=ROOT/'logs/static_unet_graph_cost.log'
        with log.open('xb') as file:
            process=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT)
            record.update(status='RUNNING_STATIC_UNET_GRAPH_COST',diagnostic_pid=process.pid,command=command,log=str(log),watchdog_pid=watchdog.pid)
            write(receipt,record)
            try:code=process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:code=process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();code=process.wait(timeout=10)
            record['diagnostic_exit_code']=code
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=10)
        os.kill(pid,signal.SIGCONT);watchdog.terminate();watchdog.wait(timeout=5)
        record.update(status='REG_RESUMED_AFTER_STATIC_UNET_GRAPH_COST',pause_seconds=time.monotonic()-started,
            end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write(receipt,record)
    assert record.get('diagnostic_exit_code')==0,record
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
