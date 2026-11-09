"""Preserve retired sample objects and prove exact original-byte restoration.

The historical sample tensor is a view retaining all16 simulation draws in its
storage, although its logical value is just the first draw. Keep that value and
all metadata in compact files, and preserve every learned channel/model, source,
input and report. Retire original cache files only after all40 have been rebuilt
byte-for-byte from the saved models and registered query RNG. No image changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from native_error_channel import NativeErrorChannel,stream_seed

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
OLD=DATA/'error_validation/idea_first_ATMS_patience50_v1'
ARCHIVE=DATA/'archived_validation_samples_compact_v1'
MANIFEST=ARCHIVE/'exact_restore_manifest.json'
ARMS=['Clean','Iso','Moment','Conditional']
SCRATCH=Path('/dev/shm/EEG_retired_sample_restore_proof_20261020_v1')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write(path,value):
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def identity(path):
    path=Path(path)
    assert path.resolve()==path and path.is_file() and not path.is_symlink()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def equal(left,right):
    if isinstance(left,torch.Tensor):assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert list(left)==list(right)
        for key in left:equal(left[key],right[key])
    else:assert left==right


def dependencies():
    plan_path=ROOT/'configs/idea_first_error_validation_plan_v1.json'
    plan=json.loads(plan_path.read_text());assert plan['seed']==1729 and plan['simulation_draws']==16
    assert sha(ROOT/'native_error_channel.py')==plan['source_sha256']['native_error_channel.py']
    cfg=json.loads((ROOT/'configs/idea_first_native_cache_v1.json').read_text())
    bank_path=Path(cfg['condition_cache_output_root'])/'C_I_D.pt'
    bank=torch.load(bank_path,weights_only=True,map_location='cpu')
    assert bank['C_I'].shape==(1650,1024)
    return plan,bank,[identity(plan_path),identity(ROOT/'native_error_channel.py'),identity(bank_path)]


def regenerate(sample,subject,arm,plan,bank,model,neighbors):
    values=[];sources=[]
    assert sample['plan_sha256']==sha(ROOT/'configs/idea_first_error_validation_plan_v1.json')
    assert sample['simulation_draws']==16 and sample['all_draws_reproducible_from_saved_model_and_registered_query_stream']
    assert sample['query_ids']==[f'{subject}/{s}/R4' for s in bank['stimulus_ids']]
    for draw in range(16):
        seeds=[stream_seed(1729,plan['version'],subject,s,'R4','error-draw',draw) for s in bank['stimulus_ids']]
        value,metadata=model.sample(arm,bank['C_I'],bank['concept_ids'],seeds,neighbors)
        values.append(value)
        if arm=='Conditional':sources.append(metadata['source_indices'])
    assert sources==sample['source_residual_indices']
    sim=torch.stack(values,dim=1)
    result=sample.copy();result['C_simulated_first_draw']=sim[:,0]
    equal(sample,result)
    return result


def audit():
    assert not ARCHIVE.exists() and not SCRATCH.exists(),'Preserve existing audit/cache'
    assert OLD.resolve()==OLD
    assert shutil.disk_usage('/dev/shm').free>1<<30
    assert shutil.disk_usage(DATA).free>700<<20
    ARCHIVE.mkdir();SCRATCH.mkdir()
    torch.set_num_threads(2);started=time.monotonic()
    plan,bank,base_dependencies=dependencies()
    report=dict(status='AUDITING_EXACT_ORIGINAL_RESTORATION_NO_ORIGINALS_REMOVED',entries=[],
        dependencies=base_dependencies,old_root=str(OLD),retired_files=0,
        unchanged_images=True,unchanged_learned_weights=True,unchanged_EEG=True)
    write(MANIFEST,report)
    for subject in [f'sub-{i:02}' for i in range(1,11)]:
        folder=OLD/subject
        model_path=folder/'error_channel.pt';neighbors_path=folder/'D_nearest_other_T_concepts.pt'
        model=NativeErrorChannel(torch.load(model_path,weights_only=True,map_location='cpu')['state'])
        neighbors=torch.load(neighbors_path,weights_only=True,map_location='cpu')
        report['dependencies']+=[identity(model_path),identity(neighbors_path)]
        for arm in ARMS:
            original=folder/(arm+'_D_validation_samples.pt')
            row=identity(original)
            sample=torch.load(original,weights_only=True,map_location='cpu')
            tensor=sample['C_simulated_first_draw']
            assert tensor.shape==(1650,1024) and tensor.dtype==torch.float32
            assert tensor.untyped_storage().nbytes()==16*tensor.numel()*tensor.element_size()
            compact=sample.copy();compact['C_simulated_first_draw']=tensor.clone().contiguous()
            destination=ARCHIVE/'compact'/subject/original.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            torch.save(compact,destination)
            equal(sample,torch.load(destination,weights_only=True,map_location='cpu'))
            restored=regenerate(compact,subject,arm,plan,bank,model,neighbors)
            scratch=SCRATCH/original.name
            assert not scratch.exists()
            torch.save(restored,scratch)
            restored_sha=sha(scratch)
            assert restored_sha==row['sha256'],'Exact bytes not reproduced; originals MUST stay: '+str(original)
            row.update(subject=subject,arm=arm,compact=identity(destination),
                logical_tensor_shape=list(tensor.shape),logical_tensor_bytes=tensor.numel()*tensor.element_size(),
                original_tensor_storage_bytes=tensor.untyped_storage().nbytes(),
                exact_original_bytes_actually_restored_and_SHA_verified=True,
                restored_sha256=restored_sha)
            report['entries'].append(row);write(MANIFEST,report)
            assert scratch.resolve().parent==SCRATCH;scratch.unlink()
            del sample,compact,restored,tensor
        print(json.dumps(dict(stage='EXACT_CACHE_RESTORE_AUDIT',subject=subject,verified_files=len(report['entries']),seconds=time.monotonic()-started)),flush=True)
    assert len(report['entries'])==40
    report.update(status='ALL40_EXACT_ORIGINAL_BYTES_REBUILT_AND_COMPACT_LOGICAL_VALUES_VERIFIED',
        old_bytes=sum(r['bytes'] for r in report['entries']),compact_bytes=sum(r['compact']['bytes'] for r in report['entries']),
        seconds=time.monotonic()-started)
    write(MANIFEST,report)
    assert SCRATCH.resolve()==SCRATCH and not list(SCRATCH.iterdir());SCRATCH.rmdir()


def assert_unused(paths):
    wanted={str(p) for p in paths}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            for handle in (process/'fd').iterdir():
                try:assert os.readlink(handle) not in wanted,'Target open by PID '+process.name
                except (FileNotFoundError,PermissionError):pass
            text=(process/'maps').read_text(errors='replace')
            assert not any(p in text for p in wanted),'Target mapped by PID '+process.name
        except (FileNotFoundError,PermissionError):pass


def apply():
    report=json.loads(MANIFEST.read_text())
    assert report['status']=='ALL40_EXACT_ORIGINAL_BYTES_REBUILT_AND_COMPACT_LOGICAL_VALUES_VERIFIED'
    assert len(report['entries'])==40 and report['retired_files']==0
    for dep in report['dependencies']:assert identity(dep['path'])==dep
    paths=[]
    for row in report['entries']:
        path=Path(row['path']);assert path.resolve()==path and path.parent==OLD/row['subject']
        assert path.name==row['arm']+'_D_validation_samples.pt' and row['arm'] in ARMS
        assert identity(path)=={k:row[k] for k in ['path','bytes','sha256']}
        assert identity(row['compact']['path'])==row['compact']
        assert row['exact_original_bytes_actually_restored_and_SHA_verified']
        paths.append(path)
    assert_unused(paths)
    before=shutil.disk_usage(DATA).free
    for row in report['entries']:
        path=Path(row['path'])
        assert sha(path)==row['sha256'] and sha(row['compact']['path'])==row['compact']['sha256']
        receipt=path.with_name(path.name+'.ARCHIVED_COMPACT_v1.json')
        assert not receipt.exists()
        write(receipt,dict(status='EXACT_RESTORE_PROVEN_COMPACT_ARCHIVE_PRESERVED',manifest=str(MANIFEST),
            original=row,restore_command=f'{sys.executable} {__file__} --mode restore --subject {row["subject"]} --arm {row["arm"]}',
            note='Original derived cache retired; learned model, input bank, full recipe and logical sample retained. Restore original bytes before historical hash checks.'))
        path.unlink()
        report['retired_files']+=1;write(MANIFEST,report)
    report.update(status='ALL40_DERIVED_CACHE_FILES_RETIRED_WITH_EXACT_RESTORE_PROOF',
        observed_free_increase_bytes=shutil.disk_usage(DATA).free-before)
    write(MANIFEST,report);print(json.dumps({k:v for k,v in report.items() if k not in ['entries','dependencies']}),flush=True)


def restore(subject,arm):
    report=json.loads(MANIFEST.read_text());row=next(x for x in report['entries'] if x['subject']==subject and x['arm']==arm)
    for dep in report['dependencies']:assert identity(dep['path'])==dep
    original=Path(row['path']);assert original.parent==OLD/subject and not original.exists()
    assert identity(row['compact']['path'])==row['compact']
    plan,bank,_=dependencies();torch.set_num_threads(2)
    model=NativeErrorChannel(torch.load(original.parent/'error_channel.pt',weights_only=True,map_location='cpu')['state'])
    neighbors=torch.load(original.parent/'D_nearest_other_T_concepts.pt',weights_only=True,map_location='cpu')
    sample=torch.load(row['compact']['path'],weights_only=True,map_location='cpu')
    restored=regenerate(sample,subject,arm,plan,bank,model,neighbors)
    temp=original.parent/'original_restore_pending_v1';assert not temp.exists();temp.mkdir()
    rebuilt=temp/original.name;torch.save(restored,rebuilt)
    assert sha(rebuilt)==row['sha256'];rebuilt.rename(original);temp.rmdir()
    print('EXACT_ORIGINAL_CACHE_RESTORED',str(original),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['audit','apply','restore'],required=True)
    parser.add_argument('--subject');parser.add_argument('--arm',choices=ARMS);args=parser.parse_args()
    if args.mode=='audit':audit()
    elif args.mode=='apply':apply()
    else:restore(args.subject,args.arm)
