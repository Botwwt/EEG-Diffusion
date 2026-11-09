"""Install a SHA-verified native HIVE runtime on the authorized second host.

Only explicitly bundled paths and new public-asset links may be created.
Existing differing files are never replaced. No GPU is initialized.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parent
PERSISTENT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
WORK=Path('/root/brainhive_reproduction_20261020_v1')
PUBLIC=PERSISTENT/'native_HIVE_runtime_20261028_v1'
RUNTIME_ROOTS=(Path('/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages'),
 Path('/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages'),
 Path('/root/mced_representation_runtime_v1/lib/python3.12/site-packages'),
 Path('/root/mced_eeg_brainhive_parquet_packages_v1'),
 PERSISTENT/'analysis_runtime/accelerate_0342_20261025_v3')
START=time.monotonic()


def memory():
    values={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):values[line.split(':')[0]]=int(line.split()[1])*1024
    return values


BASE=memory()['VmRSS']


def check():
    assert time.monotonic()-START<600,'Finite CPU installation deadline'
    assert memory()['VmHWM']-BASE<=10<<30,'Original extra CPU cap'


def sha(path):
    check()
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def checked(item):
    path=Path(item.get('path',item.get('destination')))
    assert path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    return path


def atomic_new(source,destination):
    fn=ctypes.CDLL(None,use_errno=True).renameat2
    fn.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint);fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(source),-100,os.fsencode(destination),1):
        raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),str(destination))


def target_path(value):
    pure=PurePosixPath(value)
    assert pure.is_absolute() and '..' not in pure.parts and str(pure)==value and '\\' not in value
    path=Path(value)
    roots=(ROOT,WORK/'assets/prior_H14',WORK/'audit/functional_T_pixel_replay_v2',*RUNTIME_ROOTS)
    assert any(path.is_relative_to(root) and path!=root for root in roots) or path==WORK/'audit/native_asset_verification.json',value
    assert not any(parent.is_symlink() for parent in (path,*path.parents)),value
    return path


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def install(plan_path,receipt_path,output):
    assert not output.exists();output.mkdir(parents=True)
    plan=read(plan_path);receipt=read(receipt_path)
    assert plan['status']=='NATIVE_HIVE_SECOND_HOST_TRANSFER_PLAN_ALL_DONOR_SHAS_VERIFIED'
    assert receipt['status']=='NATIVE_HIVE_PUBLIC_RUNTIME_ASSETS_DIRECT_TRANSFER_ALL_SHA_VERIFIED'
    assert receipt['manifest_sha256']==sha(plan_path)
    assert len(plan['files'])==len(receipt['files'])==21
    for item in plan['files']:
        assert Path(item['destination']).is_relative_to(PUBLIC)
        checked(item)
    bundle=next(x for x in plan['files'] if x['kind']=='EXPLICIT_RUNTIME_SOURCE_BUNDLE')
    manifest=next(x for x in plan['files'] if x['kind']=='BUNDLE_MEMBER_SHA_MANIFEST')
    files=read(checked(manifest))['files']
    assert 1<=len(files)<20000 and sum(x['bytes'] for x in files)<512<<20
    expected={x['member']:x for x in files};assert len(expected)==len(files)
    for item in files:
        path=target_path(item['destination'])
        assert 0<=item['bytes']<=128<<20 and item['member'].startswith('payload/')
        if path.exists():assert path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    installed=[]
    with tarfile.open(checked(bundle),'r:gz') as archive:
        seen=set()
        for member in archive:
            check();assert member.isfile() and member.name in expected and member.name not in seen
            seen.add(member.name);item=expected[member.name];assert member.size==item['bytes']
            payload=archive.extractfile(member).read();assert len(payload)==item['bytes'] and hashlib.sha256(payload).hexdigest()==item['sha256']
            target=target_path(item['destination'])
            if target.exists():status='EXISTING_IDENTICAL_BYTES_PRESERVED'
            else:
                target.parent.mkdir(parents=True,exist_ok=True);target_path(str(target))
                pending=target.with_name(target.name+'.HIVE_second_card_pending');assert not pending.exists()
                with pending.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
                atomic_new(pending,target);status='NEW_BUNDLED_FILE_INSTALLED'
            assert sha(target)==item['sha256'];installed.append(dict(path=str(target),sha256=item['sha256'],status=status))
        assert seen==set(expected)
    links=[]
    for item in plan['files']:
        if item['kind']!='PINNED_PUBLIC_ASSET':continue
        target=checked(item);logical=Path(item['logical_path'])
        assert logical.is_relative_to(Path('/dev/shm/eeg_diffusion_next_20261018_v1')) or logical.is_relative_to(WORK/'assets/prior_H14')
        assert not any(parent.is_symlink() for parent in logical.parents)
        logical.parent.mkdir(parents=True,exist_ok=True)
        if logical.is_symlink():assert logical.resolve()==target.resolve()
        elif logical.exists():assert logical.is_file() and sha(logical)==item['sha256']
        else:logical.symlink_to(target)
        assert sha(logical)==item['sha256'];links.append(dict(logical_path=str(logical),persistent_target=str(target),sha256=item['sha256']))
    runtime_manifest=ROOT/'audit/diagnostic_memory_recovery_20261025_v3/isolated_runtime_manifest.json'
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=dict(path=str(runtime_manifest),sha256=sha(runtime_manifest))))
    import torch
    import numpy as np
    import diffusers,transformers,accelerate,safetensors
    from brainhive_native_models_v1 import prior_models
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    assert torch.__version__=='2.3.0+cu121' and np.__version__=='1.26.4' and accelerate.__version__=='0.34.2'
    models=prior_models('cpu',include_attention=True)
    counts={name:sum(p.numel() for p in model.parameters()) for name,model in models.items()}
    assert counts['attn_adapter']==340787200 and counts['proj']==8400896
    assert all(not parameter.requires_grad and parameter.device.type=='cpu' and torch.isfinite(parameter).all() for model in models.values() for parameter in model.parameters())
    assert not torch.cuda.is_initialized();check()
    modules=[]
    for module in [torch,np,diffusers,transformers,accelerate,safetensors]:
        modules.append(dict(name=module.__name__,version=getattr(module,'__version__',None),file=module.__file__,sha256=sha(module.__file__)))
    for item in files:assert sha(target_path(item['destination']))==item['sha256']
    value=dict(status='ADDITIONAL_HOST_NATIVE_HIVE_RUNTIME_INSTALLED_AND_STRICT_CPU_PRIORS_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               source_sha256=sha(__file__),transfer_plan_sha256=sha(plan_path),transfer_receipt_sha256=sha(receipt_path),
               installed_files=len(installed),existing_identical_files=sum(x['status']=='EXISTING_IDENTICAL_BYTES_PRESERVED' for x in installed),
               public_assets=links,strict_loaded_parameter_counts=counts,runtime_modules=modules,
               original_source_and_weight_bytes_preserved=True,GPU_initialized=False,new_training_updates=0,new_images=0,
               base_Torch_or_NumPy_or_environment_modified=False,per_file_SHA=installed,
               seconds=time.monotonic()-START,peak_extra_CPU_bytes=memory()['VmHWM']-BASE)
    write(output/'completion.json',value)
    print(json.dumps({k:value[k] for k in ['status','installed_files','strict_loaded_parameter_counts','seconds','peak_extra_CPU_bytes']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert args.output.resolve().is_relative_to(PUBLIC.resolve())
    try:install(args.plan,args.receipt,args.output)
    except BaseException as error:
        if args.output.is_dir():write(args.output/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-START,new_GPU_jobs=0,new_training_updates=0))
        raise

if __name__=='__main__':main()
