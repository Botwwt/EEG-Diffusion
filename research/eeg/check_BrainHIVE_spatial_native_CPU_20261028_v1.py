"""Actual native T32 first-layer zero, disable, gradient, and folded replay."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import argparse,datetime,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_spatial_images_20261028_v1 as U
import brainhive_spatial_innovation_20261028_v1 as M

def run(sp,output):
    start=time.monotonic();torch.set_num_threads(2);s=U.read(sp)
    eng=U.read(U.checked(s['engineering']));es=U.read(U.checked(eng['spec']))
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=es['runtime_manifest']))
    from brainhive_reversible_eeg_20261028_v1 import author_brain_class
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from prepare_BrainHIVE_spatial_images_CPU_20261028_v1 import native_np,ordinary_np
    cpu=U.read(U.checked(es['CPU_completion']));cs=U.read(U.checked(cpu['spec']));d30=U.read(U.checked(cs['D30_spec']))
    sf=U.read(U.checked(s['spatial_CPU']))['files'];raw=np.load(U.checked(sf['T_mean63']),mmap_mode='r',allow_pickle=False)
    stats={n:torch.from_numpy(np.load(U.checked(sf[n]),allow_pickle=False)) for n in ['L','conditional_white','marginal_white','temporal_white','weight_Frobenius_radius']}
    mean=torch.from_numpy(np.load(U.checked(sf['mean']),allow_pickle=False));rows=np.linspace(0,13229,32,dtype=np.int64);x=torch.from_numpy(np.array(raw[rows],copy=True))
    cls=author_brain_class(U.checked(d30['author_models_source']),U.checked(d30['author_layers_source']))
    original=cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False)
    parent=torch.load(U.checked(d30['roles']['full']['checkpoint']),weights_only=True,map_location='cpu')['model'];original.load_state_dict(parent,strict=True)
    frozen=U.read(U.checked(es['ordinary_spec']));old=torch.load(U.checked(frozen['checkpoint']),weights_only=True,map_location='cpu');ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);ordinary.load_state_dict(old['model'])
    gamma=parent['block.layers.0.norm.weight'];beta=parent['block.layers.0.norm.bias'];checks=[]
    os_={k:v.numpy().astype(np.float64) for k,v in old['model'].items()}
    for arm in M.ARMS:
        model=M.SpatialNative(original,stats,mean,arm).eval();assert model.B.numel()==11776000
        with torch.no_grad():
            c=original(x[:,46:].flatten(1));assert torch.equal(model(x),c) and torch.equal(model(x,enabled=False),c)
        torch.manual_seed(1729)
        with torch.no_grad():model.B.normal_(0,1e-4)
        model.project();v=model(x);v.square().mean().backward();assert model.B.grad is not None and torch.isfinite(model.B.grad).all() and model.B.grad.norm()>0
        assert all(p.grad is None for p in original.parameters())
        with torch.no_grad():
            o=ordinary(v,gamma.expand_as(v),beta.expand_as(v),gate_mode='fixed')[0]
            folded=model.folded_state();sn={k:t.numpy().astype(np.float64) for k,t in folded.items()}
            cnp=native_np(x.numpy().reshape(32,-1).astype(np.float64),sn)
            onp=ordinary_np(cnp,gamma.numpy().astype(np.float64),beta.numpy().astype(np.float64),os_)
            ce=float(np.abs(cnp-v.numpy()).max());oe=float(np.abs(onp-o.numpy()).max())
            assert ce<1e-3 and oe<1e-3
        checks.append(dict(arm=arm,zero_and_disabled_exact=True,trainable_parameters=11776000,gradient_norm=float(model.B.grad.norm()),native_folded_NumPy64_max_error=ce,ordinary_NumPy64_max_error=oe))
    assert not torch.cuda.is_initialized()
    U.write(output,dict(status='D40_REAL_NATIVE_T32_ALL3_FIRST_LAYER_ZERO_DISABLE_GRADIENT_AND_FOLDED_NUMPY_REPLAY_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,T_queries=32,temporary_random_weights_discarded=True,new_GPU_jobs=0,new_research_fits=0,seconds=time.monotonic()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.spec,a.output)
