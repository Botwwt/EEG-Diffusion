"""Full T checkpoint/coverage and independent raw-residual replay; no images."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def ref(p):
    p=Path(p)
    with p.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest)
def read(p): return json.loads(Path(p).read_bytes())
def checked(item): assert ref(item['path']) == item; return Path(item['path'])


def run(path):
    s=read(path); assert s['status']=='REGISTERED_SINGLE1729_EUCLIDEAN_F_GEOMETRY_FULL_LIFECYCLE'
    assert s['training_seed']==1729 and not s['multiple_training_seeds_allowed']
    for item in s['source_pins']+s['input_pins']: checked(item)
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    import torch, numpy as np
    from scipy.special import erf
    from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl
    from brainhive_component_data_v1 import load_training
    torch.set_num_threads(2); assert not torch.cuda.is_initialized(); started=time.monotonic()
    root=Path(s['output_root']); out=root/'fit'; done=read(out/'completion.json')
    assert done['status']=='COMPLETE_ALL10_TRUE_OOF_THREE_PASSES_EUCLIDEAN_F_FIXED_FINAL'
    assert done['protocol']==ref(path) and done['seed']==1729 and done['updates']==24807 and done['query_exposures']==396900
    saved=torch.load(checked(done['checkpoint']),weights_only=False,map_location='cpu')
    final=torch.load(checked(done['component']),weights_only=True,map_location='cpu')
    calibration=read(checked(done['calibration']))
    assert final['geometry']=='UNBOUNDED_AFFINE_RESIDUAL' and final['arm']=='Euclidean_F'
    assert final['seed']==saved['seed']==1729 and final['updates']==saved['updates']==24807
    assert saved['seen']==final['seen_queries']==396900 and saved['coverage'].shape==(3,132300)
    assert saved['coverage'].dtype==torch.bool and bool(saved['coverage'].all())
    assert int(saved['coverage'].sum())==396900 and final['calibration']==calibration
    assert all(torch.equal(v,final['model'][k]) for k,v in saved['model'].items())
    parameters=list(final['model'].values()); opt=saved['optimizer']; ids=opt['param_groups'][0]['params']
    assert len(ids)==len(opt['state'])==len(parameters)==4
    for index,parameter in zip(ids,parameters):
        state=opt['state'][index]; assert int(state['step'])==24807
        for key in ('exp_avg','exp_avg_sq'):
            assert state[key].shape==parameter.shape and torch.isfinite(state[key]).all()
    assert saved['torch_rng'].dtype==torch.uint8 and saved['cuda_rng'] and all(x.dtype==torch.uint8 for x in saved['cuda_rng'])
    orders=[]
    for i in range(3):
        seed=int(hashlib.sha256(('1729|BrainHIVE_bounded_v1|pass'+str(i)).encode()).hexdigest()[:16],16)%(2**63-1)
        order=torch.randperm(132300,generator=torch.Generator().manual_seed(seed))
        orders.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
    assert orders==saved['orders_sha256']==done['orders_sha256']==read(checked(s['bounded_F_training_provenance']))['orders_sha256']
    assert calibration['strengths']==[0.,.25,.5,.75,1.]
    values=calibration['Dfit_token_errors']; selected=min(range(5),key=lambda i:(values[i],i))
    assert calibration['selected_strength']==calibration['strengths'][selected]
    inputs=load_training('cpu'); assert inputs['input_contract_sha256']==s['input_contract']['sha256']
    m=EuclideanConditionControl().eval().requires_grad_(False); m.load_state_dict(final['model'],strict=True)
    m.radius=.25*calibration['selected_strength']
    weights={k:v.numpy().astype(np.float64) for k,v in final['model'].items()}
    maximum=scaled_max=error2=actual2=0.; ratios=[]
    with torch.no_grad():
        for start in range(0,132300,512):
            sl=slice(start,start+512); c=inputs['condition'][sl]; aid=inputs['affine_ids'][sl]
            gamma=inputs['weight'][aid]; beta=inputs['bias'][aid]
            actual=m(c,gamma,beta)[0].numpy(); a,b,g=[x.numpy().astype(np.float64) for x in (c,beta,gamma)]
            u=(a-b)/g; z=u@weights['trunk.0.weight'].T+weights['trunk.0.bias']
            h=.5*z*(1+erf(z/np.sqrt(2))); direction=h@weights['direction.weight'].T+weights['direction.bias']
            expected=a+g*m.radius*direction; error=np.abs(actual-expected)
            maximum=max(maximum,float(error.max())); scaled_max=max(scaled_max,float((error/(1+np.abs(expected))).max()))
            error2+=float(np.square(error).sum()); actual2+=float(np.square(actual.astype(np.float64)).sum())
            ratios.extend((np.linalg.norm((actual-b)/g,axis=1)/np.linalg.norm(u,axis=1)).tolist())
            assert np.isfinite(actual).all() and torch.equal(m(c,gamma,beta,enabled=False)[0],c)
    relative=(error2/actual2)**.5
    # Fixed before observing this control: allow FP32 absolute plus relative
    # rounding on an intentionally unbounded-amplitude residual.
    assert scaled_max<2e-6 and relative<1e-5
    for item in (s['immutable_ordinary'],s['bounded_F_component']): checked(item)
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_FULL132300_EUCLIDEAN_F_STATE_COVERAGE_DFIT_AND_NUMPY_FORWARD_AUDIT',
        protocol=ref(path), fit_completion=ref(out/'completion.json'), queries=132300, parameters=525568,
        updates=24807, exposures=396900, Adam_state_parameters_verified=4,
        model_optimizer_RNG_coverage_serialization_verified=True, future_optimizer_trajectory_bit_equality_not_claimed=True,
        orders_sha256=orders, selected_strength=calibration['selected_strength'],
        NumPy64_actualCPU_max_abs_error=maximum, scaled_absolute_error_max=scaled_max, relative_L2_error=relative,
        numeric_limits=dict(abs_plus_relative_scaled_max=2e-6,relative_L2=1e-5),
        all_queries_disable_exact=True, standardized_norm_ratio_quantiles=np.quantile(ratios,[0,.5,1]).tolist(),
        normative_norm_constraint_not_imposed=True, native_generation_not_yet_verified=True,
        effect_scores_read=False, new_GPU_calls=0, seconds=time.monotonic()-started)
    dest=out/'independent_CPU_audit.json'; assert not dest.exists(); dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--protocol',required=True); args=p.parse_args(); run(args.protocol)
