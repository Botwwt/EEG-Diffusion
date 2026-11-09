"""Release only this stage's clean output pages; prove all file bytes retained."""
import argparse,datetime,hashlib,json,os,re,sys,time
from pathlib import Path

def release(path,label,task_id=None):
    assert re.fullmatch(r'[A-Za-z0-9_-]+',label)
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(Path(path).parent))
    import brainhive_exposed_OOF_assets_contract_CPU_20261101_v1 as G
    s=G.read(path);protocol=G.ref(path);root=Path(s['output_root'])
    assert s['purpose']=='native_full1654_train_only_OOF_assets'
    assert root.resolve(strict=True)==root and root.parent==Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
    for p in s['source_pins']:G.checked(p)
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    from brainhive_functional_training_contract_v2 import resource_headroom
    plans={t['task_id']:t for t in G.read(G.checked(s['tasks']))['tasks']}
    if task_id is not None:assert task_id in plans
    before=resource_headroom();started=time.monotonic();folders=[];expected={}
    for key in sorted(plans):
        folder=root/'tasks'/key
        if task_id is not None and task_id!=key:continue
        audit=folder/'independent_CPU_audit.json'
        if not audit.exists():
            assert task_id is None;continue
        a=G.read(audit);assert a['status']=='PASS_INDEPENDENT_FULL25_EPOCH_AND8270_HELD_NATIVE_OOF_AUDIT' and a['protocol']==protocol
        folders.append(folder)
        for pin in (a['final_model'],a['raw_conditions'],a['export_array_rows']):expected[str(Path(pin['path']))]=pin['sha256']
        for commit_path in (folder/'epochs').glob('epoch-*/commit.json'):
            commit=G.read(commit_path);expected[str(commit_path.parent/'resume.pt')]=commit['resume_sha256']
    if task_id is None:
        engineering=root/'engineering';c=G.read(engineering/'completion.json');assert c['protocol']==protocol
        folders.append(engineering)
        for name in ('raw_conditions','export_array_rows'):
            p=c[name];expected[str(Path(p['path']))]=p['sha256']
    assert folders
    records=[];seen=set();bytes_advised=0
    for folder in folders:
        for p in sorted(folder.rglob('*')):
            assert not p.is_symlink() and p.resolve(strict=True).is_relative_to(root)
            if not p.is_file() or p.suffix not in ('.pt','.npy'):continue
            v=p.stat();identity=(v.st_dev,v.st_ino)
            if identity in seen:continue
            seen.add(identity)
            fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                first=os.fstat(fd)
                with os.fdopen(os.dup(fd),'rb') as f:sha_before=hashlib.file_digest(f,'sha256').hexdigest()
                if str(p) in expected:assert sha_before==expected[str(p)]
                os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                os.lseek(fd,0,os.SEEK_SET)
                with os.fdopen(os.dup(fd),'rb') as f:sha_after=hashlib.file_digest(f,'sha256').hexdigest()
                last=os.fstat(fd)
                assert sha_before==sha_after
                assert (first.st_dev,first.st_ino,first.st_size,first.st_mtime_ns)==(last.st_dev,last.st_ino,last.st_size,last.st_mtime_ns)
                os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            finally:os.close(fd)
            records.append(dict(path=str(p),bytes=v.st_size,sha256=sha_before,inode=list(identity),all_bytes_and_identity_unchanged=True))
            bytes_advised+=v.st_size
            assert time.monotonic()-started<120
    after=resource_headroom();reviews=root/'cache_reviews';reviews.mkdir(exist_ok=True)
    result=dict(status='COMPLETE_OWNED_CLEAN_OUTPUT_CACHE_ADVICE_ALL_BYTES_RETAINED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=protocol,label=label,task_id=task_id,
        before=before,after=after,headroom_delta_bytes=after['effective_headroom_bytes']-before['effective_headroom_bytes'],original24GiB_floor_met_after=after['effective_headroom_bytes']>=s['minimum_CPU_headroom_bytes'],
        files=records,unique_binary_files=len(records),bytes_advised=bytes_advised,files_deleted_or_moved=0,global_drop_caches_used=False,other_projects_touched=False,
        training_math_data_seed_budget_precision_thresholds_changed=False,new_models_or_updates=0,new_GPU_jobs=0,seconds=time.monotonic()-started)
    G.atomic(reviews/(label+'.json'),result)
    print(json.dumps(dict(status=result['status'],review=G.ref(reviews/(label+'.json')),before_headroom=before['effective_headroom_bytes'],after_headroom=after['effective_headroom_bytes'],floor_met=result['original24GiB_floor_met_after'],unique_files=len(records),bytes_advised=bytes_advised,seconds=result['seconds'])))

if __name__=='__main__':
    if len(sys.argv)==1:
        q=json.loads(sys.stdin.readline());assert q['host']=='old'
        p=Path(q['protocol']['path']);assert p.stat().st_size==q['protocol']['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==q['protocol']['sha256']
        release(str(p),q['label'],q.get('task'))
    else:
        p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--label',required=True);p.add_argument('--task');a=p.parse_args();release(a.protocol,a.label,a.task)
