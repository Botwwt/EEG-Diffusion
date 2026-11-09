"""Algebra checks of whitening, native weight coupling, and working-noise cap."""
import argparse,datetime,json
from pathlib import Path
import torch
import brainhive_spatial_innovation_20261028_v1 as M

def run(output):
    torch.set_num_threads(2);g=torch.Generator().manual_seed(1729);D=torch.float64
    C,P,T,H=7,3,5,8;E=C-P
    a=torch.randn(C,C,generator=g,dtype=D);s=a@a.T+.2*torch.eye(C,dtype=D)
    b=torch.randn(T,T,generator=g,dtype=D);t=b@b.T+.2*torch.eye(T,dtype=D);t*=s.trace()/C/(t.trace()/T)
    W=torch.randn(H,P*T,generator=g,dtype=D);bias=torch.randn(H,generator=g,dtype=D);mean=torch.randn(C,T,generator=g,dtype=D)
    stats=M.geometry(s,t,W,posterior=P);S=stats['working_spatial'];Theta=stats['working_temporal'];L=stats['L']
    A=torch.cat([torch.eye(E,dtype=D),-L],1)
    cross=A@S[:,E:];conditional=A@S@A.T
    assert cross.abs().max()<1e-12 and torch.allclose(conditional,stats['conditional_covariance'],atol=1e-12,rtol=0)
    lw=stats['conditional_white'];rw=stats['temporal_white']
    assert torch.allclose(lw@conditional@lw.T,torch.eye(E,dtype=D),atol=1e-12,rtol=0)
    assert torch.allclose(rw@Theta@rw.T,torch.eye(T,dtype=D),atol=1e-12,rtol=0)
    x=torch.randn(11,C,T,generator=g,dtype=D);B=torch.randn(H,E*T,generator=g,dtype=D)
    B*=stats['weight_Frobenius_radius']/B.norm()
    expected=x[:,E:].flatten(1)@W.T+bias+M.innovation(x,mean,L,lw,rw)@B.T
    full,fb=M.folded_first_layer(W,bias,B,mean,L,lw,rw)
    actual=x.flatten(1)@full.T+fb;fold=float((actual-expected).abs().max());assert fold<1e-12
    K=torch.kron(S,Theta);original=torch.cat([torch.zeros(H,E*T,dtype=D),W],1)
    baseline=torch.trace(original@K@original.T);energy=torch.trace(full@K@full.T)
    assert torch.allclose(baseline,stats['original_hidden_working_noise_energy'],atol=1e-10,rtol=1e-12)
    assert torch.allclose(energy,baseline+B.square().sum(),atol=1e-10,rtol=1e-12)
    assert energy<=baseline*(1+.25**2)*(1+1e-12)
    zero,zb=M.folded_first_layer(W,bias,torch.zeros_like(B),mean,L,lw,rw)
    assert torch.equal(zero,original) and torch.equal(zb,bias)
    v=dict(status='D40_SCHUR_WHITENING_FOLDING_AND_WORKING_NOISE_BOUND_ALGEBRA_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      checks=8,maximum_fold_error=fold,conditional_cross_noise_max=float(cross.abs().max()),
      actual_empirical_noise_covariance_is_not_assumed_separable=True,new_GPU_jobs=0,new_fits=0)
    with Path(output).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    print(json.dumps(v))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
