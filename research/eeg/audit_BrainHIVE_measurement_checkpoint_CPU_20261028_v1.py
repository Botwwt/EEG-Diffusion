"""Independent structural/prefix audit of a real D39 committed GPU checkpoint."""
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
    start=time.monotonic();spec=read(sp);commit=read(commit_path);assert commit['status']=='D39_NATIVE_CORE_APPEND_ONLY_COMMITTED_CHECKPOINT'
    assert commit['spec_sha256']==sha(sp) and commit['arm'] in spec['arms'];torch.set_num_threads(2)
    cp=torch.load(checked(commit['checkpoint']),map_location='cpu',weights_only=True)
    assert cp['schema']=='D39_EXACT_NATIVE_CORE_ONE_PASS_COMMIT_V1' and cp['spec_sha256']==sha(sp) and cp['arm']==commit['arm']
    updates,cursor=cp['updates'],cp['cursor'];assert updates==commit['updates'] and cursor==commit['cursor'] and updates==math.ceil(cursor/16)
    assert 0<cursor<=13230 and 0<updates<=827
    seed=int(hashlib.sha256(b'D39|1729|one_full_T_order').hexdigest()[:16],16)%(2**63-1)
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
    assert len(group)==1 and group[0]['params']==[0,1,2,3] and group[0]['lr']==spec['learning_rate'] and group[0]['weight_decay']==spec['weight_decay']
    assert set(state)=={0,1,2,3}
    for item in state.values():
        assert int(item['step'])==updates and torch.isfinite(item['exp_avg']).all() and torch.isfinite(item['exp_avg_sq']).all() and (item['exp_avg_sq']>=0).all()
    for rng in [cp['noise_rng'],cp['CPU_rng'],*cp['CUDA_rng']]:assert rng.dtype==torch.uint8 and rng.ndim==1 and rng.numel()>0
    assert len(cp['CUDA_rng'])==1 and math.isfinite(cp['loss_sum'])
    eng=read(checked(spec['engineering']));engsp=read(checked(eng['spec']));cpu=read(checked(engsp['CPU_completion']));csp=read(checked(cpu['spec']));d30=read(checked(csp['D30_spec']))
    parent=torch.load(checked(d30['roles']['full']['checkpoint']),map_location='cpu',weights_only=True)['model']
    assert set(parent)==set(cp['model'])
    changed={}
    for name,value in cp['model'].items():
        assert value.shape==parent[name].shape and value.dtype==parent[name].dtype and torch.isfinite(value).all()
        if name in ['block.layers.0.norm.weight','block.layers.0.norm.bias']:assert torch.equal(value,parent[name])
        else:changed[name]=float((value-parent[name]).norm());assert changed[name]>0
    rawref=next(x for x in cpu['files'] if x['path'].endswith('/T13230_mean4.npy'))
    raw=np.load(checked(rawref),mmap_mode='r',allow_pickle=False);rows=np.linspace(0,13229,16,dtype=np.int64);x=torch.from_numpy(np.array(raw[rows],copy=True))
    def forward(model):
        h=F.linear(x,model['proj_in.weight'],model['proj_in.bias'])
        z=h+F.linear(F.gelu(F.linear(h,model['block.layers.0.ff.w1.weight'])),model['block.layers.0.ff.w2.weight'])
        return F.layer_norm(z,(1024,),model['block.layers.0.norm.weight'],model['block.layers.0.norm.bias'],1e-6)
    with torch.no_grad():before=forward(parent);after=forward(cp['model'])
    change=float((after-before).square().mean().sqrt());assert change>0 and torch.isfinite(after).all() and not torch.cuda.is_initialized()
    result=dict(status='D39_REAL_COMMITTED_MODEL_ADAMW_RNG_COVERAGE_AND_LOG_PREFIX_CPU_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       source_sha256=sha(__file__),commit_sha256=sha(commit_path),checkpoint=commit['checkpoint'],arm=cp['arm'],updates=updates,cursor=cursor,
       original_output_affine_bitwise_unchanged=True,all4_native_core_weight_change_norms=changed,
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
