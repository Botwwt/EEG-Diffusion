"""Bounded old-to-new T-only visual feature cache transfer; credentials only stdin/RAM."""

import ctypes,errno,fcntl,hashlib,inspect,json,os,socket,stat,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

import signal
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite900s public runtime transfer deadline')))
signal.alarm(900)
req=json.loads(sys.stdin.buffer.readline())
old_password=req.pop('old_password')
started=time.monotonic();stage='VALIDATE';transport=None;sftp=None;results=[];install=None
runtime=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/spatial_inputs_transfer_20261028_v1')
roots=tuple(map(Path,('/root/brainhive_reproduction_20261020_v1','/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1','/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1','/dev/shm','/root/mced_brain_dinov3_s_v1')))

def need(ok,msg):
 if not ok:raise ValueError(msg)
def emit(event,**data):print(json.dumps(dict(event=event,**data),allow_nan=False),flush=True)
def no_symlinks(path):
 path=Path(path)
 for candidate in (path,*path.parents):
  need(not candidate.is_symlink(),'Symlink path/ancestor refused: '+str(candidate))
 return path
def bounded_path(value):
 need(isinstance(value,str) and '\\' not in value,'Invalid POSIX asset path')
 p=PurePosixPath(value);need(p.is_absolute() and '..' not in p.parts and str(p)==value,'Non-normalized absolute path')
 need('kan_icml' not in value.lower() and '.ssh' not in p.parts,'Unrelated/private project path refused')
 need(not any(v in value.lower() for v in ('id_rsa','id_ed25519','private_key','password','credential','.pem','.key')),'Credential path refused')
 path=no_symlinks(Path(value));need(any(path.is_relative_to(root) and path!=root for root in roots),'Outside three authorized roots')
 return path
def digest(path,limit=None):
 h=hashlib.sha256();remaining=limit
 with Path(path).open('rb') as f:
  while remaining is None or remaining:
   block=f.read((4<<20) if remaining is None else min(4<<20,remaining))
   if not block:break
   h.update(block)
   if remaining is not None:remaining-=len(block)
 need(remaining in (None,0),'Prefix shorter than declared')
 return h.hexdigest()
def write_new_json(path,value):
 path=no_symlinks(path);path.parent.mkdir(parents=True,exist_ok=True);no_symlinks(path)
 with path.open('x',encoding='utf-8') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def rename_without_overwrite(source,destination):
 libc=ctypes.CDLL(None,use_errno=True)
 need(hasattr(libc,'renameat2'),'Atomic no-replace renameat2 unavailable; pending retained')
 fn=libc.renameat2;fn.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint);fn.restype=ctypes.c_int
 result=fn(-100,os.fsencode(source),-100,os.fsencode(destination),1)
 if result:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),str(destination))
 dfd=os.open(str(destination.parent),os.O_DIRECTORY)
 try:os.fsync(dfd)
 finally:os.close(dfd)
def source_prefix(item,count):
 # Hash the old prefix beside its disk, avoiding an extra prefix download.
 code="import hashlib,json,sys;from pathlib import Path;r=json.loads(sys.stdin.buffer.readline());p=Path(r['source']);assert p.stat().st_size==r['bytes'];h=hashlib.sha256();left=r['prefix_bytes'];f=p.open('rb')\nwhile left:\n b=f.read(min(left,4194304));assert b;h.update(b);left-=len(b)\nf.close();print(json.dumps({'prefix_sha256':h.hexdigest()}))"
 import shlex
 channel=transport.open_session(timeout=30);channel.settimeout(120)
 channel.exec_command('/root/miniconda3/bin/python -B -c '+shlex.quote(code))
 channel.sendall((json.dumps(dict(source=item['source'],bytes=item['bytes'],prefix_bytes=count))+'\n').encode());channel.shutdown_write()
 out=channel.makefile('rb').read();err=channel.makefile_stderr('rb').read();rc=channel.recv_exit_status();channel.close()
 need(rc==0,'Old-host prefix hash failed: '+err.decode('utf-8','replace')[-1000:])
 return json.loads(out)['prefix_sha256']

try:
 need(hashlib.sha256(req['manifest_text'].encode('utf-8')).hexdigest()==req['manifest_sha256'],'Manifest byte SHA mismatch')
 manifest=json.loads(req['manifest_text']);need(manifest['status']=='D40_TRUE63_SPATIAL_INPUT_TRANSFER_PLAN_BOUND','Incomplete materialized manifest')
 files=manifest['files'];need(16<=len(files)<=24,'Expected latent receipt and whole nativeT latent array')
 seen={}
 for item in files:
  source=bounded_path(item['source']);target=bounded_path(item['destination'])
  need(target.is_relative_to(Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1/candidates/D40_spatial_innovation/CPU_v1')),'Unexpected nativeT destination');binding=(item['bytes'],item['sha256']);need(str(target) not in seen or seen[str(target)]==binding,'Conflicting duplicate public destination');seen[str(target)]=binding
  need(type(item['bytes']) is int and item['bytes']>=0,'Invalid materialized byte count')
  need(len(item['sha256'])==64 and all(c in '0123456789abcdef' for c in item['sha256']),'Invalid expected SHA256')
 no_symlinks(runtime);runtime.mkdir(parents=True,exist_ok=True);no_symlinks(runtime)
 lock=(runtime/'direct_pull.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt_path=bounded_path(req['remote_receipt'])
 # Keep all packages/dependencies in this task-specific target directory.
 stage='PARAMIKO_RUNTIME';packages=runtime/'packages';no_symlinks(packages);sys.path.insert(0,str(packages));sys.path.insert(0,'/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/crossdisciplinary_transfer_runtime_20261028_v1/packages')
 try:
  import paramiko
  need('max_concurrent_requests' in inspect.signature(paramiko.SFTPFile.prefetch).parameters,'Required prefetch capability missing')
 except (ImportError,ValueError):
  emit('INSTALL_TASK_ONLY_PARAMIKO',package='paramiko==3.5.1')
  command=[sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input','--only-binary=:all:','--ignore-installed','--target',str(packages),'paramiko==3.5.1']
  p=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
  install=dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
  write_new_json(runtime/('pip_install_'+str(time.time_ns())+'.json'),install)
  need(p.returncode==0,'Task-only Paramiko installation failed; exact command/log retained')
  for key in list(sys.modules):
   if key=='paramiko' or key.startswith('paramiko.'):del sys.modules[key]
  import importlib;importlib.invalidate_caches();import paramiko
 stage='DIRECT_NEW_TO_OLD_SSH';emit('CONNECTING_SOURCE_DIRECT',host='CLOUD_LOGIN_HOST',port=23526)
 source_socket=socket.create_connection(('CLOUD_LOGIN_HOST',23526),timeout=25)
 transport=paramiko.Transport(source_socket,default_window_size=64<<20,default_max_packet_size=32768)
 transport.banner_timeout=30;transport.auth_timeout=30;transport.connect(username='root',password=old_password)
 source_host_key_sha256=hashlib.sha256(transport.get_remote_server_key().asbytes()).hexdigest();need(source_host_key_sha256==req['expected_source_host_key_sha256'],'Original authorized source SSH host key changed')
 transport.set_keepalive(10)
 sftp=paramiko.SFTPClient.from_transport(transport,window_size=64<<20,max_packet_size=32768);sftp.get_channel().settimeout(120)
 emit('DIRECT_SOURCE_CONNECTED',paramiko=paramiko.__version__,window_bytes=64<<20,prefetch_requests=req['prefetch'])
 for number,item in enumerate(files,1):
  stage='ASSET_'+str(number);target=bounded_path(item['destination']);pending=no_symlinks(target.with_name(target.name+'.transfer_pending'))
  no_symlinks(target);remote_stat=sftp.stat(item['source']);need(remote_stat.st_size==item['bytes'] and stat.S_ISREG(remote_stat.st_mode),'Source is not expected regular file/size')
  if target.exists():
   need(target.is_file() and target.stat().st_size==item['bytes'] and digest(target)==item['sha256'],'Existing target differs; no overwrite: '+str(target))
   result=dict(path=str(target),status='ALREADY_IDENTICAL_SHA256',bytes=item['bytes'],sha256=item['sha256'],resumed_prefix_bytes=0,transferred_bytes=0)
  else:
   no_symlinks(target.parent);target.parent.mkdir(parents=True,exist_ok=True);no_symlinks(target.parent)
   offset=pending.stat().st_size if pending.exists() else 0;need(offset<=item['bytes'],'Pending file exceeds expected asset size')
   if pending.exists():need(pending.is_file(),'Pending is not a regular file')
   old_prefix=source_prefix(item,offset);local_prefix=digest(pending) if offset else hashlib.sha256(b'').hexdigest()
   need(old_prefix==local_prefix,'Old/new pending prefix SHA differs; preserved without overwrite')
   need((pending.stat().st_size if pending.exists() else 0)==offset,'Pending changed while prefix verified; old relay may still be running')
   emit('PREFIX_VERIFIED',file=str(target),prefix_bytes=offset,total_bytes=item['bytes'])
   flags=os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW
   if not pending.exists():flags|=os.O_EXCL
   fd=os.open(str(pending),flags,0o600)
   downloaded=0;last=time.monotonic();file_start=last
   try:
    need(os.fstat(fd).st_size==offset,'Pending changed before opening')
    with os.fdopen(fd,'ab',buffering=1<<20) as output:
     with sftp.open(item['source'],'rb',bufsize=0) as source:
      source.seek(offset);source.prefetch(file_size=item['bytes'],max_concurrent_requests=req['prefetch'])
      remaining=item['bytes']-offset
      while remaining:
       chunk=source.read(min(1<<20,remaining));need(bool(chunk),'Unexpected EOF from source SFTP')
       output.write(chunk);remaining-=len(chunk);downloaded+=len(chunk)
       if time.monotonic()-last>=10:
        output.flush();emit('TRANSFER_PROGRESS',file=str(target),prefix_bytes=offset,transferred_bytes=downloaded,total_bytes=item['bytes'],MiB_per_second=downloaded/(1<<20)/max(time.monotonic()-file_start,1e-9));last=time.monotonic()
     output.flush();os.fsync(output.fileno())
   except BaseException:
    try:os.close(fd)
    except OSError:pass
    raise
   no_symlinks(pending);no_symlinks(target)
   need(pending.stat().st_size==item['bytes'] and digest(pending)==item['sha256'],'Completed pending byte/SHA mismatch; file preserved')
   rename_without_overwrite(pending,target)
   result=dict(path=str(target),status='DIRECT_SFTP_TRANSFERRED_ALL_BYTES_SHA256_VERIFIED',bytes=item['bytes'],sha256=item['sha256'],resumed_prefix_bytes=offset,transferred_bytes=downloaded,seconds=time.monotonic()-file_start)
  results.append(result);emit('FILE_COMPLETE',number=number,file=result['path'],bytes=result['bytes'],resumed_prefix_bytes=result['resumed_prefix_bytes'])
 stage='FINAL_COMPLETE_PUBLIC_RUNTIME_SHA256'
 for item in files:
  target=bounded_path(item['destination']);need(target.stat().st_size==item['bytes'] and digest(target)==item['sha256'],'Final complete-manifest SHA mismatch')
 receipt=dict(status='D40_TRUE63_SPATIAL_INPUT_DIRECT_TRANSFER_SHA_VERIFIED',UTC=datetime.now(timezone.utc).isoformat(),manifest_sha256=req['manifest_sha256'],files=results,total_asset_bytes=sum(i['bytes'] for i in files),transferred_bytes=sum(i['transferred_bytes'] for i in results),resumed_prefix_bytes=sum(i['resumed_prefix_bytes'] for i in results),seconds=time.monotonic()-started,source_host='CLOUD_LOGIN_HOST',source_port=23526,source_host_key_sha256=source_host_key_sha256,destination_host=socket.gethostname(),window_bytes=64<<20,prefetch_requests=req['prefetch'],paramiko=paramiko.__version__,task_only_dependency_install=install is not None,credentials_saved=False,local_machine_large_file_bytes=0,GPU_initialized=False,new_training=0)
 if receipt_path.exists():
  old=read_json=json.loads(receipt_path.read_text());need(old['manifest_sha256']==receipt['manifest_sha256'] and old['status']==receipt['status'],'Existing receipt differs; no overwrite')
  receipt=old
 else:write_new_json(receipt_path,receipt)
 emit('COMPLETE',remote_receipt=str(receipt_path),receipt=receipt)
except BaseException as exc:
 reason=str(exc).replace(old_password,'[redacted]') if old_password else str(exc)
 failure=dict(status='DIRECT_OLD_TO_NEW_PULL_FAILED_PREFIXES_PRESERVED',UTC=datetime.now(timezone.utc).isoformat(),stage=stage,error_type=type(exc).__name__,reason=reason[:1500],completed_files=results,seconds=time.monotonic()-started,credentials_saved=False,GPU_initialized=False,no_automatic_retry=True)
 try:
  if runtime.exists():write_new_json(runtime/('failure_'+str(time.time_ns())+'.json'),failure)
 except BaseException:pass
 emit('FAILED',receipt=failure);sys.exit(1)
finally:
 old_password=None;req.clear()
 if sftp is not None:sftp.close()
 if transport is not None:transport.close()
