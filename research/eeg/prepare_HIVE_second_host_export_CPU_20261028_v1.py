"""Bounded CPU-only export of existing native HIVE public assets and code.

No model fitting, GPU use, result deletion, or changes to donor sources.
Public model bytes are transferred separately; the bundle contains only
explicit runtime roots, their source dependencies, and old T replay evidence.
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')
PERSISTENT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
START=time.monotonic()
MAX_SECONDS=300
MAX_BYTES=1<<30
RUNTIME_ROOTS=(Path('/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages'),
 Path('/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages'),
 Path('/root/mced_representation_runtime_v1/lib/python3.12/site-packages'),
 Path('/root/mced_eeg_brainhive_parquet_packages_v1'),
 PERSISTENT/'analysis_runtime/accelerate_0342_20261025_v3')
EXCLUDED_PREFIXES=('torch','nvidia','triton','numpy','sympy','mpmath','networkx')


def check():
    assert time.monotonic()-START<MAX_SECONDS,'Finite CPU export deadline'


def sha(path):
    check()
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def ref(path):
    p=Path(path)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def ordinary_files(directory):
    for base,dirs,files in os.walk(directory,followlinks=False):
        dirs[:]=[name for name in sorted(dirs) if name not in ['__pycache__','.git']]
        for name in sorted(files):
            check();path=Path(base)/name
            assert not path.is_symlink(),str(path)
            if path.suffix!='.pyc':yield path


def source_dependencies():
    pending=['brainhive_native_models_v1','brainhive_component_generator_v1',
             'brainhive_bounded_component_v1','brainhive_paired_study_contract_20261025_v1',
             'run_BrainHIVE_frozen_function_low_memory_20261025_v3',
             'generate_BrainHIVE_neural_clean_fastscreen_20261027_v1',
             'run_BrainHIVE_two_host_GPU_job_20261028_v1']
    seen=set()
    while pending:
        module=pending.pop()
        if module in seen:continue
        path=ROOT/(module+'.py');assert path.is_file();seen.add(module)
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                top=name.split('.')[0]
                if (ROOT/(top+'.py')).is_file() and top not in seen:pending.append(top)
    return [ROOT/(name+'.py') for name in sorted(seen)]


def prepare(output):
    assert output.resolve().is_relative_to(PERSISTENT.resolve())
    assert not output.exists();output.mkdir(parents=True)
    assets=read(ROOT/'configs/BrainHIVE_runtime_restore_manifest_20261022_v1.json')['assets']
    public=[];records={}
    for item in assets:
        logical=Path(item['path']);assert logical.is_file(),str(logical)
        binding=ref(logical);assert binding['sha256']==item['sha256']
        destination=PERSISTENT/'native_HIVE_runtime_20261028_v1/public_assets'/binding['sha256']/logical.name
        public.append(dict(source=str(logical.resolve()),destination=str(destination),logical_path=str(logical),
                           bytes=binding['bytes'],sha256=binding['sha256'],kind='PINNED_PUBLIC_ASSET'))
    for runtime in RUNTIME_ROOTS:
        assert runtime.is_dir(),str(runtime)
        for child in sorted(runtime.iterdir()):
            if child.name.startswith(EXCLUDED_PREFIXES) or child.name=='__pycache__':continue
            assert not child.is_symlink(),str(child)
            for path in ([child] if child.is_file() else ordinary_files(child)):
                if path.suffix!='.pyc':records[str(path)]=path
    chosen=source_dependencies()+[ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json',
         ROOT/'configs/research_parallel_authorization_20261028_v1.json',
         ROOT/'configs/ordinary_frozen_base_20261028_v1.json',
         ROOT/'audit/diagnostic_memory_recovery_20261025_v3/isolated_runtime_manifest.json',
         WORK/'audit/native_asset_verification.json']
    for name in ['proj','fusion_encoder','attn_adapter']:chosen.append(WORK/'assets/prior_H14'/name/'config.json')
    author=ROOT/'reproduction_sources/BrainHIVE_f06db69f'
    chosen += [author/name for name in ['_fetch_receipt.json','main/layers.py','main/models_brain.py','main/models_adapter.py']]
    replay=WORK/'audit/functional_T_pixel_replay_v2'
    assert replay.is_dir()
    chosen += list(ordinary_files(replay))
    for path in chosen:
        assert path.is_file() and not path.is_symlink(),str(path)
        records[str(path)]=path
    files=[]
    for number,path in enumerate(sorted(records.values(),key=str)):
        binding=ref(path)
        files.append(dict(source=str(path),destination=str(path),bytes=binding['bytes'],sha256=binding['sha256'],member='payload/'+str(number).zfill(6)))
    assert len(files)<20000 and sum(x['bytes'] for x in files)<512<<20
    archive=output/'native_runtime_sources_and_T_replay.tar.gz'
    with archive.open('xb') as disk:
        with gzip.GzipFile(fileobj=disk,mode='wb',compresslevel=1,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as tar:
                for item in files:
                    check();path=Path(item['source']);assert sha(path)==item['sha256']
                    metadata=tarfile.TarInfo(item['member']);metadata.size=item['bytes'];metadata.mode=0o644
                    with path.open('rb') as stream:tar.addfile(metadata,stream)
                    assert sha(path)==item['sha256']
        disk.flush();os.fsync(disk.fileno())
    assert archive.stat().st_size<MAX_BYTES
    runtime_manifest=output/'bundle_files.json'
    write(runtime_manifest,dict(status='EXPLICIT_NATIVE_HIVE_RUNTIME_SOURCE_BUNDLE',files=files,
                               runtime_roots=list(map(str,RUNTIME_ROOTS)),excluded_prefixes=list(EXCLUDED_PREFIXES),
                               raw_EEG_or_original_target_images_selected=False))
    transfers=public+[dict(source=str(archive),destination=str(PERSISTENT/'native_HIVE_runtime_20261028_v1/transfer/native_runtime_sources_and_T_replay.tar.gz'),
                          bytes=archive.stat().st_size,sha256=sha(archive),kind='EXPLICIT_RUNTIME_SOURCE_BUNDLE'),
                     dict(source=str(runtime_manifest),destination=str(PERSISTENT/'native_HIVE_runtime_20261028_v1/transfer/bundle_files.json'),
                          bytes=runtime_manifest.stat().st_size,sha256=sha(runtime_manifest),kind='BUNDLE_MEMBER_SHA_MANIFEST')]
    total=sum(x['bytes'] for x in transfers);assert total<10<<30
    done=dict(status='NATIVE_HIVE_SECOND_HOST_TRANSFER_PLAN_ALL_DONOR_SHAS_VERIFIED',source=ref(__file__),
              files=transfers,bundle=ref(archive),bundle_members=ref(runtime_manifest),
              transferred_file_count=len(transfers),total_bytes=total,runtime_uncompressed_bytes=sum(x['bytes'] for x in files),
              no_existing_donor_sources_changed=True,new_GPU_jobs=0,new_training_updates=0,
              source_public_weight_bytes=sum(x['bytes'] for x in public),
              excludes_Torch_NumPy_CUDA_base=True,seconds=time.monotonic()-START)
    write(output/'transfer_plan.json',done)
    print(json.dumps({k:done[k] for k in ['status','transferred_file_count','total_bytes','runtime_uncompressed_bytes','seconds']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Explicit CPU export termination')))
    try:prepare(args.output)
    except BaseException as error:
        if args.output.is_dir():
            write(args.output/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-START,new_GPU_jobs=0,new_training_updates=0))
        raise

if __name__=='__main__':main()
