"""Truthful exposed-benchmark fullscope comparison; legacy gates stay intact."""
import datetime,hashlib,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
ARMS=('bounded_F','Euclidean_F')
GPU='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
STATUS='REVIEWED_EXPOSED_FULL1654_TWO_F_HEADS_AND_COMPLETE_R80_LIFECYCLE'

def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def check(pin):
    p=Path(pin['path']);assert p.is_file() and not p.is_symlink() and ref(p)==pin
    return p
def write(p,value,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert replace or not p.exists()
    tmp=p.with_name(p.name+'.pending.'+str(os.getpid()));assert not tmp.exists()
    with tmp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    if replace:os.replace(tmp,p)
    else:os.link(tmp,p)
def seed(namespace):return int(hashlib.sha256(('1729|BrainHIVE_bounded_v1|'+namespace).encode()).hexdigest()[:16],16)%(2**63-1)
def rss():return int(next(v.split()[1] for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('VmHWM:')))*1024
def allocated(root):
    seen=set();total=0
    for p in Path(root).rglob('*'):
        assert not p.is_symlink()
        if p.is_file():
            s=p.stat();key=s.st_dev,s.st_ino
            if key not in seen:seen.add(key);total+=s.st_blocks*512
    return total
def validate_scope(s):
    assert s['status']==STATUS and s['host']=='old' and s['GPU_UUID']==GPU and s['arms']==list(ARMS)
    assert s['official200_previously_exposed'] is True and s['official_results_used_for_previous_direction'] is True and s['independent_H'] is False
    assert s['method']==dict(seed=1729,queries=165400,images=16540,subjects=10,passes=3,batch=16,updates=31014,exposures=496200,last_batch=8,parameters=525568,lr=1e-4,weight_decay=.01,gradient_clip=1.,strength=1.,radius=.25,objective='.1 mean(1-cos(raw_corrected_c,F))',fixed_last=True)
    assert s['R80']==dict(queries=2000,subjects=10,candidates=200,resolution=512,UNet_steps=4,CFG=0,IP_scale=1,images_per_query=1)
    assert s['maximum_output_bytes']==12<<30 and s['legacy_floor_bytes']==50<<30 and s['disk_buffer_bytes']==8<<30
    assert s['minimum_CPU_headroom_bytes']==24<<30 and s['maximum_extra_CPU_bytes']==10<<30 and s['maximum_GPU_bytes']==20<<30
    assert s['numerical_guard']==dict(atol=1e-5,rtol=5e-6)
    return s
def validate(s):
    validate_scope(s)
    for pin in s['source_pins']:check(pin)
    profile=read(check(s['actual_profile']));assert profile['status']=='PASS_ACTUAL_FULL1654_F_GEOMETRY_ENGINEERING_ONLY' and not profile['learned_weights_saved']
    assert profile['input_bank']==s['input_bank'] and [a['temporary_updates'] for a in profile['arms']]==[4,4]
    consumer=read(check(s['consumer_review']));assert consumer['status']=='PASS_ACTUAL_ALL165400_NATIVE_RAW_JOIN_AND_ALL20_OWN_AFFINE_CONSUMER'
    assert consumer['input_bank']==s['input_bank'] and consumer['asset_protocol']==s['asset_protocol']
    assert check(s['ordinary']).stat().st_size==2104936 and s['ordinary']['sha256']=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
    return s
def setup(path):
    s=validate(read(path));review=read(check(s['registration_CPU_review']))
    assert review['status']=='PASS_SOURCE_SCOPE_COMPLETE_LIFECYCLE_CPU_REVIEW' and review['input_bank']==s['input_bank'] and review['source_pins']==s['source_pins']
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));sys.path.insert(0,str(Path(path).parent))
    return s
def resources(s,training_start=False):
    from brainhive_functional_training_contract_v2 import resource_headroom
    m=resource_headroom();floor=s['minimum_CPU_headroom_bytes'] if training_start else 10<<30
    assert m['effective_headroom_bytes']>=floor
    root=Path(s['output_root']);used=allocated(root) if root.exists() else 0
    assert used<=s['maximum_output_bytes']
    required=s['maximum_output_bytes']-used+s['legacy_floor_bytes']+s['disk_buffer_bytes'];free=shutil.disk_usage(root if root.exists() else root.parent).free
    assert free>=required
    return dict(effective_CPU_headroom_bytes=m['effective_headroom_bytes'],minimum_phase_headroom_bytes=floor,used_output_bytes=used,free_bytes=free,required_remaining_bytes=required,not_physical_reservation=True)
def wrapper(s):
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_bytes().decode()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==GPU
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def model_class(arm):
    assert arm in ARMS
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl
    return BoundedConditionAdapter if arm=='bounded_F' else EuclideanConditionControl
def load_final(s,arm,torch):
    out=Path(s['output_root'])/arm;done=read(out/'fit_completion.json');audit=read(out/'fit_CPU_audit.json')
    assert done['status']=='COMPLETE_FULL1654_F_HEAD_THREE_PASSES' and audit['status']=='PASS_FULL1654_F_HEAD_STATE_AND_ALL165400_FORWARD_AUDIT'
    assert done['arm']==audit['arm']==arm and done['updates']==31014 and done['exposures']==496200
    saved=torch.load(check(done['component']),weights_only=True,map_location='cpu')
    assert saved['arm']==arm and saved['seed']==1729 and saved['strength']==1. and saved['updates']==31014
    model=model_class(arm)().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True)
    return model,done,audit
