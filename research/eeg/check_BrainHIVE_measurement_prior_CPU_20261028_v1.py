"""JVP, coordinate precision, matched controls and local closed-form checks."""
import argparse,datetime,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
import brainhive_measurement_prior_20261028_v1 as M

def run(output):
    torch.set_num_threads(2);torch.manual_seed(1729)
    D=12;A=torch.randn(D,D,dtype=torch.float64);N=A@A.T
    Q=M.normalized_precision(N);assert abs(float(Q.trace())/D-1)<1e-12
    p=torch.randperm(D);Qp=Q[p][:,p]
    assert torch.allclose(torch.linalg.eigvalsh(Q),torch.linalg.eigvalsh(Qp),atol=1e-12)
    u=torch.randn(7,D);v=torch.randn(7,D)
    assert torch.allclose(M.anchor_penalty(u,v,torch.eye(D),'isotropic'),M.anchor_penalty(u,v,torch.eye(D),'measurement'))
    x=torch.randn(7,D,requires_grad=True);loss=M.anchor_penalty(x,x.detach().clone(),Q,'measurement');g=torch.autograd.grad(loss,x)[0]
    assert loss==0 and g.count_nonzero()==0
    J=torch.randn(21,D,dtype=torch.float64);r=torch.randn(21,dtype=torch.float64)
    delta=M.local_quadratic_map(J,r,Q);d=delta.detach().requires_grad_()
    value=(r+J@d).square().sum()+.5*d@Q@d/(D*.25**2)
    error=float(torch.autograd.grad(value,d)[0].abs().max());assert error<1e-11
    R=4;z=torch.randn(9,R,5,dtype=torch.float64);centered=z-z.mean(1,keepdim=True)
    mean_cov=torch.einsum('nrd,nre->de',centered,centered)/(len(z)*R*(R-1))
    dirs=centered/math.sqrt(R*(R-1))
    assert torch.allclose(mean_cov,dirs.flatten(0,1).T@dirs.flatten(0,1)/len(z),atol=1e-14)
    # Dimensions in the pure native JVP are fixed; compare a small batch with AD.
    state={'proj_in.weight':torch.randn(1024,4250,dtype=torch.float64)/math.sqrt(4250),
      'proj_in.bias':torch.randn(1024,dtype=torch.float64)*.01,
      'block.layers.0.ff.w1.weight':torch.randn(1024,1024,dtype=torch.float64)/32,
      'block.layers.0.ff.w2.weight':torch.randn(1024,1024,dtype=torch.float64)/32}
    x=torch.randn(2,4250,dtype=torch.float64);directions=torch.randn(2,4,4250,dtype=torch.float64)*.01
    unit,actual=M.unit_and_jvp(x,directions,state)
    def forward(xx):
        h=F.linear(xx,state['proj_in.weight'],state['proj_in.bias'])
        z=h+F.linear(F.gelu(F.linear(h,state['block.layers.0.ff.w1.weight'])),state['block.layers.0.ff.w2.weight'])
        return F.layer_norm(z,(1024,),eps=1e-6)
    jerr=0.
    assert torch.allclose(unit,forward(x),atol=1e-12)
    for t in range(4):
        _,truth=torch.autograd.functional.jvp(forward,x,directions[:,t]);jerr=max(jerr,float((truth-actual[:,t]).abs().max()))
    assert jerr<1e-12 and not torch.cuda.is_initialized()
    result=dict(status='D39_MEASUREMENT_PRIOR_JVP_PRECISION_AND_LOCAL_MAP_MATH_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests=7,
      exact_local_normal_equation_gradient_error=error,native_JVP_max_error=jerr,CUDA_initialized=False)
    with Path(output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
