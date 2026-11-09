"""Advise away clean pages of one pinned, unused training input; no file mutation."""
import hashlib,os,time
from pathlib import Path

SOURCE={'path':'/root/autodl-tmp/atm_published_training_trials_20261016_v2/preprocessed/sub-01/preprocessed_eeg_training.npy','bytes':4168083005,'sha256':'c54cf2e8a4f0c122d79bb7d98ddf04698f992d292f3e26ae59bed97230944f63'}

def headroom():
    return int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())

def release(pin):
    assert pin==SOURCE
    p=Path(pin['path']);assert p.resolve(strict=True)==p and not p.is_symlink()
    started=time.monotonic();before=headroom();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    def sha():
        os.lseek(fd,0,os.SEEK_SET)
        with os.fdopen(os.dup(fd),'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    try:
        first=os.fstat(fd);assert first.st_size==pin['bytes'];assert sha()==pin['sha256']
        os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        assert sha()==pin['sha256'] and identity(first)==identity(os.fstat(fd))
        os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    finally:os.close(fd)
    return dict(status='PINNED_CLEAN_TRAINING_INPUT_ADVISED_BYTES_AND_IDENTITY_UNCHANGED',source=pin,
        inode=list(identity(first)),cgroup_headroom_before=before,cgroup_headroom_after=headroom(),
        seconds=time.monotonic()-started,deleted_files=0,global_drop_caches=False,other_projects_touched=False)
