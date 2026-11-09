"""OLD-only resource contract and classical Gaussian-overlap construction."""
import datetime,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

DOMAINS=('ordinary_DDPM','Gaussian_midpoint','harmonic_overlap')
GEOMETRY=('separation','log_qi_over_qj','responsibility_i','responsibility_j','log_affinity','log_geometric_weight')

def logcosh(x):return np.logaddexp(x,-x)-np.log(2.)
def draw_axial(rng,separation):
    assert np.isfinite(separation) and separation>=0
    for attempt in range(1,1000001):
        value=float(rng.normal());uniform=float(rng.random())
        if np.log(max(uniform,np.finfo(float).tiny))<=-float(logcosh(.5*separation*value)):
            return value,attempt
    raise RuntimeError('Finite rejection bound; no alternate proposal or silent row drop')

def setup(path):
    spec=json.loads(Path(path).read_bytes());sys.path.insert(0,str(Path(spec['base_protocol']['path']).parent))
    import brainhive_functional_mean_contract_20261103_v1 as G
    base=G.setup(G.checked(spec['base_protocol']))
    assert spec['status']=='REGISTERED_FROZEN_NATIVE_EEG_OVERLAP_GRADIENT_PROFILE' and spec['seed']==1729
    assert spec['maximum_output_bytes']==512<<20 and spec['GPU_bytes']==20<<30 and spec['extra_CPU_bytes']==10<<30
    assert spec['full_chunks']==[list(range(i,i+20)) for i in range(0,80,20)] and spec['engineering_queries']==[0,72]
    assert spec['full_raw_parameter_queries']==[0,72] and spec['full_raw_parameter_time_indices']==[0]
    for pin in spec['source_pins']:G.checked(pin)
    return spec,base,G

def resources(spec,base,G,start=False):
    original=G.resources(base,start=start);remaining=0;usage={}
    groups=spec['upstream_output_groups']+[dict(name='this_profile',cap=512<<20,roots=[spec['output_root'],spec['code_root']])]
    allroots=[Path(v) for group in groups for v in group['roots']];assert len(allroots)==len(set(allroots))
    for group in groups:
        used=0
        for name in group['roots']:
            directory=Path(name)
            if not directory.exists():continue
            for child in directory.rglob('*'):
                assert not child.is_symlink()
                if child.is_file():used+=child.stat().st_blocks*512
        assert used<=group['cap'],group['name'];remaining+=max(0,group['cap']-used)
        usage[group['name']]=dict(used=used,cap=group['cap'])
    assert original['external']['free_bytes']>=original['required_with_all_prior_commitments']+remaining
    return dict(existing=original,groups=usage,additional_remaining_commitments=remaining)

def numpy_checks():
    from scipy.integrate import quad
    rng=np.random.default_rng(1729);results=[]
    for d in (1.,10.,100.):
        draws=np.array([draw_axial(rng,d)[0]*d for _ in range(12000)])
        f=lambda ell:np.exp(-ell*ell/(2*d*d)-float(logcosh(ell/2)))
        normal=quad(f,-np.inf,np.inf,epsabs=1e-10)[0]
        variance=quad(lambda ell:ell*ell*f(ell),-np.inf,np.inf,epsabs=1e-9)[0]/normal
        assert abs(draws.mean())<.12 and abs(np.mean(draws*draws)-variance)<.07*max(variance,1.)
        results.append(dict(separation=d,samples=len(draws),empirical_mean=float(draws.mean()),empirical_second_moment=float(np.mean(draws*draws)),quadrature_second_moment=variance))
    # A midpoint mean does not make each random draw equally plausible for both images.
    assert abs(10*2.)>1 and abs(np.exp(-float(logcosh(10))))<1e-3
    u=rng.normal(size=31);v=rng.normal(size=31);r=.27
    error=abs(r*np.mean(u*u)+(1-r)*np.mean(v*v)-np.mean((r*u+(1-r)*v)**2)-r*(1-r)*np.mean((u-v)**2))
    assert error<1e-12
    return dict(status='PASS_OVERLAP_REJECTION_DISTRIBUTION_QUADRATURE_AND_WEIGHTED_RISK',cases=results,weighted_risk_error=error,
                classical_bridge_density_not_novel_or_unbiased_new_objective=True)

def torch_toy_checks():
    import torch
    torch.manual_seed(1729)
    model=torch.nn.Sequential(torch.nn.Linear(5,7),torch.nn.GELU(),torch.nn.Linear(7,3)).double()
    condition=torch.randn(2,5,dtype=torch.float64,requires_grad=True);prediction=model(condition);truth=torch.randn(2,3,dtype=torch.float64)
    r=.31;errors=prediction-truth
    marginal=(r*errors[0]+(1-r)*errors[1]).square().mean()
    contrast=((prediction[0]-prediction[1])-(truth[0]-truth[1])).square().mean()
    total=r*errors[0].square().mean()+(1-r)*errors[1].square().mean()
    params=tuple(model.parameters())+(condition,)
    gm=torch.autograd.grad(marginal,params,retain_graph=True);gc=torch.autograd.grad(contrast,params,retain_graph=True);gr=torch.autograd.grad(total,params)
    error=max(float((a+r*(1-r)*b-c).abs().max()) for a,b,c in zip(gm,gc,gr));assert error<1e-12
    originals=[p.detach().clone() for p in model.parameters()];flat=torch.cat([v.flatten() for v in gc[:-1]]);norm=float(flat.norm());direction=flat/norm;values=[]
    with torch.no_grad():
        for sign in (-1.,1.):
            cursor=0
            for parameter,original in zip(model.parameters(),originals):
                n=parameter.numel();parameter.copy_(original+sign*1e-5*direction[cursor:cursor+n].reshape_as(parameter));cursor+=n
            trial=model(condition);values.append(float(((trial[0]-trial[1])-(truth[0]-truth[1])).square().mean()))
        for parameter,original in zip(model.parameters(),originals):parameter.copy_(original)
    fd=(values[1]-values[0])/2e-5;assert abs(fd-norm)<1e-7
    return dict(status='PASS_NONLINEAR_CPU64_PARAMETER_INPUT_GRADIENT_IDENTITY_AND_FINITE_DIFFERENCE',gradient_identity_max_error=error,
                finite_difference_error=abs(fd-norm),temporary_weights_restored_exact=True,no_EEG_or_native_GPU_effect_claim=True)
