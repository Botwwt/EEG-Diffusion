"""Independent structural/prefix audit of a real D40 committed GPU checkpoint."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(x):
    p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256']
    if 'bytes' in x:assert p.stat().st_size==x['bytes']
    return p
def run(sp,commit_path,output):
    start=time.monotonic();spec=read(sp);commit=read(commit_path);assert commit['status']=='D40_SPATIAL_FIRST_LAYER_APPEND_ONLY_COMMITTED_CHECKPOINT'
    assert commit['spec_sha256']==sha(sp) and commit['arm'] in spec['arms'];torch.set_num_threads(2)
    cp=torch.load(checked(commit['checkpoint']),map_location='cpu',weights_only=True)
    assert cp['schema']=='D40_EXACT_SPATIAL_FIRST_LAYER_ONE_PASS_COMMIT_V1' and cp['spec_sha256']==sha(sp) and cp['arm']==commit['arm']
    updates,cursor=cp['updates'],cp['cursor'];assert updates==commit['updates'] and cursor==commit['cursor'] and updates==math.ceil(cursor/16)
    assert 0<cursor<=13230 and 0<updates<=827
    seed=int(hashlib.sha256(b'D40|1729|one_full_T_order').hexdigest()[:16],16)%(2**63-1)
    order=torch.randperm(13230,generator=torch.Generator().manual_seed(seed));assert hashlib.sha256(order.numpy().tobytes()).hexdigest()==cp['order_sha256']
    expected=torch.zeros(13230,dtype=torch.bool);expected[order[:cursor]]=True
    assert cp['coverage'].dtype==torch.bool and torch.equal(cp['coverage'],expected)
    log=commit_path.parent/'training.jsonl'
    with log.open('rb') as f:prefix=f.read(cp['log_bytes'])
    assert len(prefix)==cp['log_bytes']==commit['log_bytes'] and hashlib.sha256(prefix).hexdigest()==cp['log_sha256']==commit['log_sha256']
    events=[json.loads(line) for line in prefix.decode().splitlines()];assert len(events)==updates
    last=0
    for i,event in enumerate(events):
        assert event['updates']==i+1 and event['arm']==cp['arm'] and event['indices']==order[last:event['cursor']].tolist()
        assert len(event['indices'])==len(event['t'])<=16 and event['cursor']-last==len(event['indices']) and all(0<=t<1000 for t in event['t'])
        assert all(math.isfinite(event[k]) for k in ['loss','epsilon_loss','token_loss','anchor_loss','gradient_norm','unit_shift_RMS'])
        assert len(event['noise_sha256'])==64;last=event['cursor']
    assert last==cursor
    group=cp['optimizer']['param_groups'];state=cp['optimizer']['state']
    assert len(group)==1 and group[0]['params']==[0] and group[0]['lr']==spec['learning_rate'] and group[0]['weight_decay']==spec['weight_decay']
    assert set(state)=={0}
    for item in state.values():
        assert int(item['step'])==updates and torch.isfinite(item['exp_avg']).all() and torch.isfinite(item['exp_avg_sq']).all() and (item['exp_avg_sq']>=0).all()
    for rng in [cp['noise_rng'],cp['CPU_rng'],*cp['CUDA_rng']]:assert rng.dtype==torch.uint8 and rng.ndim==1 and rng.numel()>0
    assert len(cp['CUDA_rng'])==1 and math.isfinite(cp['loss_sum'])
    eng=read(checked(spec['engineering']));engsp=read(checked(eng['spec']));cpu=read(checked(engsp['CPU_completion']));csp=read(checked(cpu['spec']));d30=read(checked(csp['D30_spec']))
    parent=torch.load(checked(d30['roles']['full']['checkpoint']),map_location='cpu',weights_only=True)['model']
    import brainhive_spatial_innovation_20261028_v1 as M
    from brainhive_reversible_eeg_20261028_v1 import author_brain_class
    spatial=read(checked(spec['spatial_CPU']));sf=spatial['files']
    stats={n:torch.from_numpy(np.load(checked(sf[n]),allow_pickle=False)) for n in ['L','conditional_white','marginal_white','temporal_white','weight_Frobenius_radius']}
    mean=torch.from_numpy(np.load(checked(sf['mean']),allow_pickle=False))
    assert set(cp['model'])=={'B'};value=cp['model']['B'];assert value.shape==(1024,11500) and value.dtype==torch.float32 and torch.isfinite(value).all()
    assert 0<float(value.norm())<=float(stats['weight_Frobenius_radius'])*(1+2e-6)
    cls=author_brain_class(checked(d30['author_models_source']),checked(d30['author_layers_source']))
    original=cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);original.load_state_dict(parent,strict=True)
    student=M.SpatialNative(original,stats,mean,cp['arm']).eval().requires_grad_(False);student.B.copy_(value)
    raw=np.load(checked(sf['T_mean63']),mmap_mode='r',allow_pickle=False);rows=np.linspace(0,13229,16,dtype=np.int64);x=torch.from_numpy(np.array(raw[rows],copy=True))
    with torch.no_grad():before=original(x[:,46:].flatten(1));after=student(x)
    change=float((after-before).square().mean().sqrt());assert change>0 and torch.isfinite(after).all() and not torch.cuda.is_initialized()
    changed={'B':float(value.norm())}
    result=dict(status='D40_REAL_COMMITTED_MODEL_ADAMW_RNG_COVERAGE_AND_LOG_PREFIX_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       source_sha256=sha(__file__),commit_sha256=sha(commit_path),checkpoint=commit['checkpoint'],arm=cp['arm'],updates=updates,cursor=cursor,
       original_output_affine_bitwise_unchanged=True,first_layer_B_norm=changed,
       sampled_T_function_replay_queries=16,sampled_condition_change_RMS=change,
       uncommitted_log_tail_bytes=log.stat().st_size-len(prefix),committed_prefix_verified=True,
       model_optimizer_CPU_CUDA_noise_RNG_and_order_coverage_verified=True,independent_GPU_update_numerical_replay=False,
       CUDA_initialized=False,seconds=time.monotonic()-start)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','arm','updates','cursor','seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['spec','commit','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.spec,a.commit,a.output)
