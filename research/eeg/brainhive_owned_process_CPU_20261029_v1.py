"""Cleanup only directly launched groups and their pinned wrapper child."""
import json,os,signal,subprocess,time
from pathlib import Path
def identity(pid):
    p=Path('/proc')/str(pid)
    if not (p/'stat').exists():return None
    try:
        f=(p/'stat').read_text().rsplit(')',1)[1].split()
        if f[0] in ('Z','X'):return None
        return dict(pid=pid,start_ticks=int(f[19]),cmdline=(p/'cmdline').read_bytes().hex(),pgid=os.getpgid(pid))
    except (FileNotFoundError,ProcessLookupError):return None
def capture_wrapper_child(owner,receipt):
    start=time.monotonic();parent=identity(owner.pid);assert parent is not None
    while time.monotonic()-start<10:
        if owner.poll() is not None:return None
        p=Path(receipt)
        if p.exists():
            try:row=json.loads(p.read_bytes())
            except json.JSONDecodeError:time.sleep(.05);continue
            assert row['wrapper_pid']==owner.pid;child=identity(row['pid'])
            if child is not None:
                assert child['pgid']==child['pid'] and child['start_ticks']>=parent['start_ticks'];return child
        time.sleep(.05)
    raise RuntimeError('Cannot pin newly launched wrapper child')
def stop_owned(owner,pinned_child=None):
    events=[]
    if pinned_child is not None and identity(pinned_child['pid'])==pinned_child:
        os.killpg(pinned_child['pid'],signal.SIGTERM);events.append('TERM exact pinned own wrapper child');start=time.monotonic()
        while time.monotonic()-start<30 and identity(pinned_child['pid'])==pinned_child:time.sleep(.1)
        if identity(pinned_child['pid'])==pinned_child:os.killpg(pinned_child['pid'],signal.SIGKILL);events.append('KILL exact pinned own wrapper child')
    if owner is not None and owner.poll() is None:
        # A GPU wrapper should commit its failure ledger after its child exits.
        if pinned_child is not None:
            try:owner.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        if owner.poll() is None:
            os.killpg(owner.pid,signal.SIGTERM)
            try:owner.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(owner.pid,signal.SIGKILL);owner.wait(timeout=10)
    return events
