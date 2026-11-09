"""CPU recheck of original full fit state and finite independently captured traces."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=digest)
def check(x):assert ref(x['path'])==x;return Path(x['path'])


def run(path,script_root):
    q=read(path);assert q['status']=='REGISTERED_SAME_FIXED_HEAD_FP32_TRACE_ENGINEERING_ONLY'
    for x in q['source_pins']+q['input_pins']:check(x)
    sys.path.insert(0,q['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=q['runtime_manifest']));sys.path.insert(0,str(script_root))
    import numpy as np
    import torch
    from brainhive_FP32_trace_audit_CPU_20261030_v1 import verify,merge,bit_equal
    from brainhive_component_data_v1 import load_training
    torch.set_num_threads(2);assert not torch.cuda.is_initialized();started=time.monotonic()
    root=Path(q['output_root']);done=read(root/'completion.json');assert done['status']=='PASS_FINITE_CAPTURED_FP32_LAYER_CHECKS_ORIGINAL_FULL64_GUARD_STILL_FAILED'
    assert done['T']['queries']==132300 and done['R80']['queries']==2000 and done['R80']['batches']==2000
    assert not done['original_full64_guard_passed'] and done['component']==q['component']
    original=read(check(q['original_study']));fit_root=Path(original['output_root'])/'fit';fit=read(fit_root/'completion.json')
    assert fit['protocol']==q['original_study'] and fit['component']==q['component']
    state=torch.load(check(fit['component']),map_location='cpu',weights_only=True);saved=torch.load(check(fit['checkpoint']),map_location='cpu',weights_only=False)
    assert state['updates']==saved['updates']==24807 and state['seen_queries']==saved['seen']==396900
    assert state['seed']==saved['seed']==1729 and saved['coverage'].shape==(3,132300) and saved['coverage'].dtype==torch.bool
    assert bool(saved['coverage'].all()) and int(saved['coverage'].sum())==396900
    assert state['protocol_sha256']==saved['protocol_sha256']==q['original_study']['sha256']
    assert all(torch.equal(v,state['model'][k]) for k,v in saved['model'].items())
    opt=saved['optimizer'];ids=opt['param_groups'][0]['params'];assert len(ids)==len(opt['state'])==len(state['model'])==4
    for index,value in zip(ids,state['model'].values()):
        item=opt['state'][index];assert int(item['step'])==24807
        for key in ('exp_avg','exp_avg_sq'):assert item[key].shape==value.shape and torch.isfinite(item[key]).all()
    assert saved['torch_rng'].dtype==torch.uint8 and saved['cuda_rng'] and all(x.dtype==torch.uint8 for x in saved['cuda_rng'])
    orders=[]
    for i in range(3):
        seed=int(hashlib.sha256(('1729|BrainHIVE_bounded_v1|pass'+str(i)).encode()).hexdigest()[:16],16)%(2**63-1)
        order=torch.randperm(132300,generator=torch.Generator().manual_seed(seed));orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
    assert orders==saved['orders_sha256']==fit['orders_sha256']==read(check(original['bounded_F_training_provenance']))['orders_sha256']
    calibration=read(check(fit['calibration']));assert calibration==state['calibration']
    assert calibration['strengths']==[0.,.25,.5,.75,1.] and calibration['selected_strength']==q['selected_strength']
    choice=min(range(5),key=lambda i:(calibration['Dfit_token_errors'][i],i));assert calibration['strengths'][choice]==q['selected_strength']
    weights={k:v.numpy() for k,v in state['model'].items()};probe=np.load(check(done['T_probe']),allow_pickle=False)
    indices=probe['index'];assert indices.shape==(320,) and len(set(indices.tolist()))==320
    assert all(bit_equal(probe[k],value) for k,value in weights.items())
    inputs=load_training('cpu');assert inputs['input_contract_sha256']==q['input_contract']['sha256']
    assert bit_equal(probe['c'],inputs['condition'][indices].numpy())
    aids=inputs['affine_ids'][indices];assert bit_equal(probe['g'],inputs['weight'][aids].numpy()) and bit_equal(probe['b'],inputs['bias'][aids].numpy())
    assert all(int((aids==aid).sum())==16 for aid in range(20))
    T=verify({k:probe[k] for k in ('u','z','h','raw','output')},weights,probe['c'],probe['g'],probe['b'],.25*q['selected_strength']);del inputs
    source=read(check(q['R80_input_completion']));bank=np.load(check(source['bank']),allow_pickle=False);trace=np.load(check(done['R80_trace']),allow_pickle=False)
    c=bank['conditions'].reshape(2000,1024);g=np.repeat(bank['gammas'],200,axis=0);b=np.repeat(bank['betas'],200,axis=0)
    reports=[]
    for start in range(0,2000,100):
        sl=slice(start,start+100);reports.append(verify({k:trace[k][sl] for k in ('u','z','h','raw','output')},weights,c[sl],g[sl],b[sl],.25*q['selected_strength']))
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_INDEPENDENT_CPU_REPLAY_T320_AND_ALL2000_R80_CAPTURED_FP32_TRACES',numeric_protocol=ref(path),numeric_completion=ref(root/'completion.json'),original_fit_completion=ref(fit_root/'completion.json'),component=q['component'],training_seed=1729,selected_strength=q['selected_strength'],full_checkpoint_model_Adam_RNG_orders_and_all396900_coverage_verified=True,future_bit_exact_optimization_not_claimed=True,original_fullT_FP64_guard_passed=False,original_guard_not_reclassified=True,T320=T,R80=merge(reports),new_GPU_calls=0,training_updates=0,research_images=0,seconds=time.monotonic()-started)
    p=root/'independent_CPU_replay_and_state.json';assert not p.exists();p.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--numeric-protocol',required=True);a=p.parse_args();run(a.numeric_protocol,Path(__file__).parent)
