from pathlib import Path
import sys,json,shutil,tarfile,os,time,subprocess
r=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
sys.path.insert(0,str(r))
from cleanup_unrelated_server import live_uses,disk
p=Path('/root/atm_paper_env_20261016_v1')
assert p.is_dir() and not p.is_symlink() and p.resolve()==Path('/root/atm_paper_env_20261016_v1')
a=r/'audit/storage_cleanup_20261018_v1'
assert json.loads((a/'replacement_environment_check.json').read_text())['status']=='REPLACEMENT_RUNTIME_IMPORTS_PASSED'
assert json.loads((r/'audit/full_atms_profile.json').read_text())['status']=='PROFILE_COMPLETED'
uses=live_uses([p]);assert not uses,uses
before=disk()
records=[]
for root,dirs,files in os.walk(p,followlinks=False):
 for name in files:
  f=Path(root)/name;s=f.lstat();records.append(dict(path=str(f),bytes=s.st_size,symlink=f.is_symlink()))
(a/'old_environment_file_inventory.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
with tarfile.open(a/'old_environment_metadata.tar.gz','x:gz') as ar:
 for f in p.rglob('*'):
  if f.is_symlink() or not f.is_file():continue
  if f.name in ['METADATA','WHEEL','INSTALLER','direct_url.json','pyvenv.cfg'] or f.suffix=='.pth':
   if f.stat().st_size<1024**2:ar.add(f,arcname=str(f).removeprefix('/'),recursive=False)
assert p.resolve()==Path('/root/atm_paper_env_20261016_v1') and not live_uses([p])
shutil.rmtree(p)
assert not p.exists()
after=disk()
receipt=dict(status='COMPLETED',removed=str(p),reason='Redundant Torch 2.5 standalone environment; actual full ATM forward/backward and generator/evaluator imports passed on retained Torch 2.3 runtime',before=before,after=after,freed_bytes={k:after[k]['free']-before[k]['free'] for k in after},local_large_files_created=0)
(a/'environment_cleanup_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
