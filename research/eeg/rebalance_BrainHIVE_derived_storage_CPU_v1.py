"""Rebalance byte-verified derived caches to existing private RAM storage.

Raw EEG, learned checkpoint files, and original generated images stay durable.
This does not increase any filesystem quota or use a shared/public mount.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import ROOT,WORK,sha,write,torch

DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
RAM=Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2')


def candidates():
    rows=[]
    def add(path,kind):
        if path.is_file() and not path.is_symlink():rows.append(dict(path=str(path),kind=kind,bytes=path.stat().st_size,
            device=path.stat().st_dev,inode=path.stat().st_ino,links=path.stat().st_nlink))
    for p in (WORK/'assets').rglob('*'):
        if p.is_file() and p.stat().st_size>1<<20:add(p,'pinned_public_prior_asset')
    for p in (WORK/'runs').rglob('conditions.pt'):add(p,'BrainHIVE_checkpoint_derived_condition_export')
    for p in (DATA/'runs').rglob('encoder_native_all_train_rows.pt'):add(p,'old_ATMS_checkpoint_derived_condition_export')
    for folder,kind in [('conditions','old_ATMS_paired_condition_cache'),('latents','old_Base512_T_image_latent_cache')]:
        for p in (DATA/folder).rglob('*.pt'):add(p,kind)
    # These gallery pixels are derived renderings; unique generated originals
    # remain under DATA/images, untouched. All HTML and identity metadata stay.
    original_inodes={(p.stat().st_dev,p.stat().st_ino) for p in (DATA/'images').rglob('*') if p.is_file()}
    for p in (DATA/'galleries').rglob('*'):
        if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png','.webp']:
            # A gallery hardlink to a durable original already shares its blocks.
            # Keep it untouched rather than spending RAM for zero disk savings.
            if (p.stat().st_dev,p.stat().st_ino) not in original_inodes:add(p,'historical_gallery_derivative_not_original_generation')
    for folder in ['/root/mced_eeg_brainhive_teacher_confirmation_samples_v1','/root/mced_eeg_native_conditional_prior_samples_v1']:
        for p in Path(folder).rglob('*.pt'):add(p,'old_frozen_prior_derived_sample_cache')
    assert len({r['path'] for r in rows})==len(rows)
    return rows


def public_ram_retirement():
    base=Path('/dev/shm/eeg_sota_autoresearch_20261018_v1/sdxl_base_46216598')
    receipt=json.loads((base/'completion.json').read_text())
    assert receipt['status']=='COMPLETE_OFFICIAL_SNAPSHOT'
    rows=[]
    for row in receipt['files']:
        p=Path(row['path'])
        if p.suffix=='.safetensors' and p.stat().st_size>50<<20:
            rel=p.relative_to(base).as_posix();assert sha(p)==row['sha256']
            rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=row['sha256'],
                public_restore_url=f"https://huggingface.co/{receipt['repository']}/resolve/{receipt['revision']}/{rel}"))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    plan_path=ROOT/'audit/BrainHIVE_derived_storage_rebalance_plan_v11.json'
    if not args.apply:
        assert not plan_path.exists();items=candidates();retire=public_ram_retirement()
        physical={}
        for row in items:physical.setdefault((row['device'],row['inode']),row['bytes'])
        by_kind={kind:sum(r['bytes'] for r in items if r['kind']==kind) for kind in sorted({r['kind'] for r in items})}
        plan=dict(status='PLANNED_BYTE_IDENTICAL_RAM_RELOCATION_OF_REGENERABLE_CACHES',
            source_sha256=sha(__file__),items=items,public_RAM_retirement=retire,
            logical_bytes_by_kind=by_kind,maximum_RAM_copy_bytes=sum(physical.values()),
            existing_RAM_free=shutil.disk_usage('/dev/shm').free,
            retired_public_RAM_bytes=sum(r['bytes'] for r in retire),
            original_generated_images_and_raw_EEG_and_learned_checkpoints_preserved=True,
            RAM_is_volatile_not_a_durable_backup=True,purchases_or_quota_changes=False)
        assert plan['existing_RAM_free']+plan['retired_public_RAM_bytes']>plan['maximum_RAM_copy_bytes']+(2<<30)
        write(plan_path,plan);print(json.dumps({k:v for k,v in plan.items() if k not in ['items','public_RAM_retirement']}),flush=True);return
    plan=json.loads(plan_path.read_text());assert plan['source_sha256']==sha(__file__)
    job=ROOT/'audit/BrainHIVE_bounded_component_engineering_batch16_v2_job.json'
    assert json.loads(job.read_text())['status']=='COMPLETED','Wait for the finite engineering job before moving loaders'
    out=ROOT/'audit/BrainHIVE_derived_storage_rebalance_v11.json';assert not out.exists()
    RAM.mkdir(mode=0o700,exist_ok=True)
    owner=RAM/'owner_plan_sha256.txt'
    if owner.exists():assert owner.read_text().strip()==sha(plan_path)
    else:owner.write_text(sha(plan_path)+'\n')
    before_path=ROOT/'audit/BrainHIVE_derived_storage_before_v11.json'
    if not before_path.exists():write(before_path,{p:shutil.disk_usage(p)._asdict() for p in ['/root','/root/autodl-tmp','/dev/shm']})
    before=json.loads(before_path.read_text())
    # Only fully pinned, publicly recoverable Base weights are retired. Turbo,
    # the native prior and the fixed semantic evaluators are retained.
    retirement=ROOT/'audit/BrainHIVE_retired_public_Base_RAM_v11.json'
    if retirement.exists():assert json.loads(retirement.read_text())['files']==plan['public_RAM_retirement']
    else:
        for row in plan['public_RAM_retirement']:
            p=Path(row['path']);assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256']
        write(retirement,dict(files=plan['public_RAM_retirement'],purpose='Free existing RAM for derived caches; current parent is Turbo'))
    for row in plan['public_RAM_retirement']:
        p=Path(row['path'])
        if p.exists():assert sha(p)==row['sha256'];p.unlink()
    journal=ROOT/'audit/BrainHIVE_derived_storage_rebalance_progress_v11.jsonl'
    completed=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    done={x['path']:x for x in completed};shared={(x['device'],x['inode']):Path(x['RAM_target']) for x in completed};skipped=[]
    for number,row in enumerate(plan['items']):
        p=Path(row['path'])
        if str(p) in done:assert p.is_symlink() and sha(p)==done[str(p)]['sha256'];continue
        assert p.is_file() and not p.is_symlink()
        stat=p.stat();assert stat.st_size==row['bytes'] and stat.st_ino==row['inode'] and stat.st_dev==row['device']
        # Checkpoints and fits are not eligible under any condition.
        assert p.name not in ['brain_final.pt','encoder_best.pt','error_channel.pt','ar-ckpt-last.pth']
        if p.suffix=='.pt':
            obj=torch.load(p,weights_only=True,map_location='cpu',mmap=True)
            if isinstance(obj,dict) and any(k in obj for k in ['model','state_dict','optimizer']):
                skipped.append(dict(path=str(p),reason='Contains learned/optimizer state; retained durable'));continue
            if row['kind']=='old_ATMS_checkpoint_derived_condition_export':
                assert set(obj)=={'condition','array_rows','native_visual_sha256','encoder_sha256','training_role'}
                matches=[q for q in p.parent.glob('*.pt') if q!=p and q.is_file() and not q.is_symlink() and q.stat().st_size<50<<20 and sha(q)==obj['encoder_sha256']]
                if not matches:
                    skipped.append(dict(path=str(p),reason='No exact durable learned encoder match; keep cache durable'));continue
            del obj
        digest=sha(p);identity=(row['device'],row['inode'])
        if identity in shared:target=shared[identity];assert sha(target)==digest
        else:
            target=RAM/'files'/p.relative_to('/root');target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.pending');assert not target.exists() and not temporary.exists()
            shutil.copyfile(p,temporary);os.chmod(temporary,0o600);assert sha(temporary)==digest
            temporary.replace(target);shared[identity]=target
        link=p.with_name(p.name+'.RAM_LINK_PENDING_v10');assert not link.exists();link.symlink_to(target)
        os.replace(link,p);assert sha(p)==digest
        record=row|dict(sha256=digest,RAM_target=str(target),original_path_retained_as_symlink=True,
            RAM_is_volatile=True,artifact_is_regenerable_cache=True)
        with journal.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        completed.append(record)
        if (number+1)%100==0:print(json.dumps(dict(stage='CACHE_RAM_REBALANCE',completed=len(completed),scanned=number+1,expected=len(plan['items']))),flush=True)
    after={p:shutil.disk_usage(p)._asdict() for p in before}
    write(out,dict(status='BYTE_IDENTICAL_DERIVED_CACHE_RAM_RELOCATION_COMPLETE',before=before,after=after,
        completed=completed,skipped=skipped,public_RAM_retirement=str(retirement),
        unique_generated_images_removed=0,raw_EEG_files_removed=0,learned_checkpoint_files_removed=0,
        source_sha256=sha(__file__),plan_sha256=sha(plan_path),
        limitation='RAM copies disappear on server restart. Original generated images, raw EEG, learned models, manifests and source stay durable; derived caches must then be recomputed or restored from pinned public assets.'))
    print(json.dumps(dict(status='CACHE_REBALANCE_COMPLETE',files=len(completed),skipped=len(skipped),before=before,after=after)),flush=True)


if __name__=='__main__':main()
