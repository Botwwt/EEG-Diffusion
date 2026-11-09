"""Separate train-only asset contract; never relaxes the legacy official gates."""
import csv,datetime,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path

STATUS='SOURCE_REVIEWED_EXPOSED_BENCHMARK_TRAIN_ONLY_OOF_ASSETS_V1'
GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
SUBJECTS=tuple(f'sub-{i:02d}' for i in range(1,11))
TOKEN=object()

def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(pin):
    p=Path(pin['path']);assert p.is_file() and not p.is_symlink(),str(p)
    a=ref(p);assert all(a[k]==v for k,v in pin.items()),str(p);return p
def atomic(p,v,replace=False):
    p=Path(p);assert replace or not p.exists()
    temporary=p.with_name(p.name+'.pending.'+str(os.getpid()))
    with temporary.open('xb') as f:f.write((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    if replace:os.replace(temporary,p)
    else:os.link(temporary,p)
def allocated(root):
    seen=set();total=0
    for p in Path(root).rglob('*'):
        assert not p.is_symlink()
        if p.is_file():
            v=p.stat();identity=(v.st_dev,v.st_ino)
            if identity not in seen:seen.add(identity);total+=getattr(v,'st_blocks',math.ceil(v.st_size/512))*512
    return total

def validate_scope(s,tasks,images,ids):
    assert s['status']==STATUS and s['host']=='old' and s['GPU_UUID']==GPU
    assert s['training_seed']==1729 and s['official200_previously_exposed'] is True
    assert s['official_results_used_for_previous_direction'] is True and s['independent_H'] is False
    assert s['official_test_payloads_allowed'] is False and s['final_component_or_benchmark_admitted'] is False
    assert s['purpose']=='native_full1654_train_only_OOF_assets'
    assert s['epochs']==25 and s['batch_size']==1024 and s['updates_per_task']==225
    assert s['expected_OOF_queries']==165400 and s['expected_tasks']==20
    assert len(s['subjects'])==10 and set(s['subjects'])==set(SUBJECTS)
    assert set(s['subject_source_paths'])==set(SUBJECTS)
    assert all(Path(s['subject_source_paths'][sub]).name==sub+'_train_mean4.pt' for sub in SUBJECTS)
    train={int(r['array_row']):r for r in images if r['dataset_split']=='train'}
    assert len(train)==len(ids)==16540 and set(train)==set(range(16540))
    assert len({r['concept_id'] for r in train.values()})==1654
    assert [Path(train[i]['stimulus_id']).stem for i in range(16540)]==ids
    assert len(tasks['tasks'])==20
    seen=set();result={};exports={subject:set() for subject in SUBJECTS};concept_fold={}
    for t in tasks['tasks']:
        key=t['task_id'];subject=t['subject'];held=t['held_fold']
        assert subject in SUBJECTS and held in (0,1) and key==subject+'/official_oof'+str(held)
        assert key not in seen;seen.add(key)
        fit=t['training_array_rows'];out=t['exported_OOF_array_rows']
        assert len(fit)==len(set(fit))==8270 and len(out)==len(set(out))==8270
        assert not set(fit)&set(out) and set(fit)|set(out)==set(train)
        fc={train[i]['concept_id'] for i in fit};ec={train[i]['concept_id'] for i in out}
        assert len(fc)==len(ec)==827 and not fc&ec
        assert fc==set(t['training_concepts']) and ec==set(t['held_concepts'])
        assert concept_fold.get(held,ec)==ec;concept_fold[held]=ec
        assert (t['training_seed'],t['epochs'],t['batch_size'],t['expected_updates'],t['expected_training_exposures'],t['final_batch_size'])==(1729,25,1024,225,206750,78)
        assert t['parameter_updates']==0 and not t['existing_protected_T_OOF_reused']
        assert all(t[k] is None for k in ('checkpoint_path','checkpoint_sha256','condition_cache_path','condition_cache_sha256'))
        assert not exports[subject]&set(out);exports[subject].update(out);result[key]=t
    assert all(v==set(train) for v in exports.values())
    return result,train

def load_scope(s):
    tasks=read(checked(s['tasks']))
    with checked(s['image_manifest']).open(encoding='utf8',newline='') as f:images=list(csv.DictReader(f))
    return validate_scope(s,tasks,images,read(checked(s['visual_IDs'])))

def assert_input_aliases(s):
    # Source files still must be regular files. Public data/weight cache aliases
    # are pinned separately to the real immutable payload; never accepted by a
    # blanket symlink exception in checked().
    for binding in s['input_aliases']:
        logical=Path(binding['logical_path']);canonical=Path(binding['canonical']['path'])
        assert logical.resolve(strict=True)==canonical and not canonical.is_symlink()
        assert canonical.is_file() and canonical.stat().st_size==binding['canonical']['bytes']

def resources(s):
    from brainhive_functional_training_contract_v2 import resource_headroom
    memory=resource_headroom();assert memory['effective_headroom_bytes']>=s['minimum_CPU_headroom_bytes']>=24<<30
    root=Path(s['output_root']);used=allocated(root) if root.exists() else 0
    assert used<=s['maximum_output_bytes']==52<<30
    required=s['maximum_output_bytes']-used+s['legacy_storage_floor_bytes']+s['minimum_disk_buffer_bytes']
    free=shutil.disk_usage(root if root.exists() else root.parent).free;assert free>=required
    return dict(effective_CPU_headroom_bytes=memory['effective_headroom_bytes'],used_output_bytes=used,free_bytes=free,remaining_stage_plus_legacy_and_buffer_bytes=required,not_physical_reservation=True)

def preflight(path,mode,task_id):
    s=read(path);assert mode in ('profile','fit')
    for p in s['source_pins']:checked(p)
    tasks,images=load_scope(s);assert task_id in tasks
    assert_input_aliases(s)
    bindings=read(checked(s['bindings']))
    assert all(bindings['subjects'][sub]['train_path']==s['subject_source_paths'][sub] and Path(s['subject_source_paths'][sub]).resolve(strict=True)==Path(s['subjects'][sub]['path']) for sub in SUBJECTS)
    preparation=read(Path(path).parent/'actual_preparation_review.json')
    assert preparation['status']=='SOURCE_DATA_SCOPE_AND_STORAGE_PROFILE_PREPARATION_COMPLETE_NO_FIT_ADMISSION'
    assert preparation['protocol']==ref(path) and preparation['legacy_gates_not_modified'] is True
    checks=read(checked(preparation['CPU_checks']))
    assert checks['status']=='PASS_TRAIN_ONLY_EXPOSURE_AWARE_FULL_SCOPE_CPU_CHECKS' and checks['count']==10
    import sys
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(path).parent))
    pid=os.getppid();command=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'run_gpu_job_v2.py' in command and s['job_prefix'] in command
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==GPU
    before=resources(s)
    for p in s['common_payloads']+[s['subjects'][tasks[task_id]['subject']]]:checked(p)
    if mode=='fit':
        admission=read(Path(s['output_root'])/'training_admission.json')
        assert admission['status']=='ALL20_TRAIN_ONLY_ASSET_FITS_REVIEWED_AFTER_ACTUAL_PROFILE'
        assert admission['protocol']==ref(path) and admission['actual_profile']==ref(Path(s['output_root'])/'engineering/completion.json')
        profile=read(checked(admission['actual_profile']))
        assert profile['status']=='PASS_ACTUAL_TRAIN_ONLY_OOF_ASSET_ENGINEERING' and profile['temporary_updates']==9
        assert profile['learned_weights_saved'] is False and profile['official_test_payloads_loaded'] is False
    return dict(token=TOKEN,s=s,tasks=tasks,task=tasks[task_id],images=images,before=before,protocol=ref(path),mode=mode)
