"""Non-overwriting WWT copy of independently hashed learned weights/Adam/RNG states."""
import hashlib,json,os,shutil,socket,time
from pathlib import Path,PurePosixPath
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
root=Path('/dfs/share/maleiLab/wwt/eeg');start=time.monotonic()
audit=json.loads((root/'runs/HIVE_native_independent_audit_5096674/completion.json').read_text())
assert audit['status']=='PASS_INDEPENDENT_ALL14880_CPU_FORWARD_OOF_AND_FULL_SCORE_AUDIT'
directory=root/'backups/HIVE_native_sub03_5096542';directory.mkdir(parents=True,mode=0o700,exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
files=[]
for r in audit['backup_files']:
    parts=PurePosixPath(r['path']).parts;assert parts[:2]==('runs','HIVE_native_sub03_5096542') and len(parts)==4
    assert parts[2] in ['full','oof0','oof1'] and parts[3] in ['brain_final.pt','final_training_state.pt']
    source=root/r['path'];assert source.stat().st_size==r['bytes'] and sha(source)==r['sha256']
    dest=directory/parts[2]/parts[3];dest.parent.mkdir(exist_ok=True,mode=0o700);assert not dest.exists()
    temp=dest.with_suffix('.pending');assert not temp.exists();shutil.copyfile(str(source),str(temp))
    with temp.open('rb') as f:os.fsync(f.fileno())
    assert temp.stat().st_size==r['bytes'] and sha(temp)==r['sha256'];temp.replace(dest);dest.chmod(0o400)
    assert source.stat().st_ino!=dest.stat().st_ino
    files.append(dict(source=r['path'],backup=str(dest.relative_to(root)),bytes=r['bytes'],sha256=r['sha256'],different_inode=True))
result=dict(status='COMPLETE_SIX_IMMUTABLE_WWT_MODEL_STATE_COPIES_SHA256_PASS',job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),
    files=files,total_bytes=sum(r['bytes'] for r in files),seconds=time.monotonic()-start,
    same_storage_system=True,offsite_backup=False,local_D_backup_declined_due_to_disk_floor=True,new_updates=0,new_images=0)
p=directory/'completion.json';temp=p.with_suffix('.pending')
with temp.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
temp.replace(p);p.chmod(0o400);print(json.dumps(result),flush=True)
