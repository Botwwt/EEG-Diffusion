"""CPU operator/adjoint/local-QP checks; no real EEG efficacy or GPU claim."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
import eeg_margin_matrix_free_20261106_v233 as M
from check_eeg_cross_host_gpu_runtime_CPU_20261106_v231 import serial

def flat(values):return torch.cat([x.reshape(-1) for x in values])

def checks():
    torch.set_num_threads(2);rows=[]
    for dtype in (torch.float64,torch.float32):
        for rescale in (False,True):
            torch.manual_seed(1729)
            net=nn.Sequential(nn.Linear(5,9),nn.GELU(),nn.Linear(9,7)).to(dtype).eval()
            original={n:v.detach().clone() for n,v in net.state_dict().items()}
            x=torch.randn(4,5,dtype=dtype);gallery=F.normalize(torch.randn(12,7,dtype=dtype),dim=-1)
            ids=torch.arange(4);groups=torch.arange(12)//3
            raw=net(x);params=tuple(net.parameters())
            margin,J,_,_=serial(raw,gallery,ids,groups,params)
            scale=(torch.linspace(.7,1.4,8,dtype=dtype) if rescale else torch.ones(8,dtype=dtype))
            J/=scale.double()[:,None]
            enc,theta=M.functional_encoder_snapshot(net,list(dict(net.named_parameters())),(x,))
            operator=M.MatrixFreeMarginMap(enc,theta,gallery,ids,groups,scale)
            assert torch.equal(margin,operator.margins)
            direction=tuple(torch.randn_like(p) for p in theta);v=flat(direction).double()
            w=torch.randn(8,dtype=dtype)
            actual_jv=operator.forward(direction).double();actual_jtw=flat(operator.transpose(w)).double()
            actual_gram=operator.gram(w).double()
            errors=dict(Jv=float((actual_jv-J@v).abs().max()),JT_w=float((actual_jtw-J.T@w.double()).abs().max()),
                        gram=float((actual_gram-J@(J.T@w.double())).abs().max()),
                        adjoint=float(abs(actual_jv@w.double()-v@actual_jtw)))
            tolerance=3e-12 if dtype==torch.float64 else 2e-5
            assert max(errors.values())<tolerance,(dtype,rescale,errors)
            assert all(torch.equal(v,original[n]) for n,v in net.state_dict().items())
            rows.append(dict(dtype=str(dtype),positive_row_scaling=rescale,errors=errors))
            if dtype!=torch.float64 or rescale:continue
            # A nontrivial feasible local QP. This is NOT nonlinear training feasibility.
            proposal=tuple(.01*torch.randn_like(p) for p in theta);pvec=flat(proposal)
            feasible=.002*torch.randn_like(pvec);rhs=J@feasible-.0001
            explicit=J@J.T;b=J@pvec-rhs
            def objective(lam,implicit):
                vector=torch.tensor(lam,dtype=dtype)
                hv=operator.gram(vector) if implicit else explicit@vector
                return float(.5*vector@hv+b@vector),(hv+b).numpy()
            solutions=[]
            for implicit in (False,True):
                result=minimize(lambda t:objective(t,implicit),[0.]*8,jac=True,bounds=[(0,None)]*8,
                                method='L-BFGS-B',options={'maxiter':2000,'ftol':0.,'gtol':1e-12,'maxls':100})
                lam=torch.tensor(result.x,dtype=dtype)
                delta=(flat(operator.closest_step(proposal,lam)) if implicit else pvec+J.T@lam)
                slack=J@delta-rhs
                violation=float(torch.clamp(-slack,min=0).max());complementarity=float((lam*slack).abs().max())
                assert violation<1e-8 and complementarity<1e-8,(implicit,violation,complementarity)
                solutions.append(dict(delta=delta,iterations=int(result.nit),reported_success=bool(result.success),
                                      violation=violation,complementarity=complementarity))
            gap=float((solutions[0]['delta']-solutions[1]['delta']).abs().max());assert gap<1e-8,gap
            rows.append(dict(local_QP_delta_max_abs_error=gap,
                             explicit={k:v for k,v in solutions[0].items() if k!='delta'},
                             implicit={k:v for k,v in solutions[1].items() if k!='delta'}))
    # Forward inputs are snapshots: subsequent external mutation must not alter maps.
    old=operator.forward(direction);x.add_(10.)
    assert torch.equal(operator.forward(direction),old)
    return dict(status='PASS_SYNTHETIC_CPU_MATRIX_FREE_JVP_VJP_GRAM_ADJOINT_AND_LOCAL_QP_ONLY',
                tests=rows,actual_GPU_or_full_host_tested=False,new_updates=0,new_images=0,
                nonlinear_finite_feasibility_or_original_trajectory_proved=False)

if __name__=='__main__':print(json.dumps(checks(),allow_nan=False))
