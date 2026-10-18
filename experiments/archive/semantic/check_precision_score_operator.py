"""Independent analytic non-Gaussian test of the actual anisotropic FD operator."""
import argparse
import json
from pathlib import Path
import torch
from precision_score_math import precision_residual


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False)
    torch.set_num_threads(2);dtype=torch.float64
    Sigma=torch.tensor([[.8,.25],[.25,.6]],dtype=dtype);L=torch.linalg.cholesky(Sigma)
    C=torch.tensor([[1.3,.1,.35,-.1],[.1,1.1,.15,.25],[.35,.15,.9,.2],[-.1,.25,.2,.8]],dtype=dtype)
    means=torch.tensor([[1.,-.5,.7,-.4],[-.8,.6,-.5,.3]],dtype=dtype);logs=torch.tensor([.4,.6],dtype=dtype).log()
    class Analytic(torch.nn.Module):
        def forward(self,values,t,v):
            full=C.expand(len(v),-1,-1)+torch.nn.functional.pad(v[:,:,None]*Sigma,(2,0,2,0))
            inv=torch.linalg.inv(full);diff=values[:,None,:]-means[None,:,:]
            scores=-torch.einsum('bmi,bij->bmj',diff,inv)
            weight=torch.softmax(-.5*torch.einsum('bmi,bij,bmj->bm',diff,inv,diff)+logs[None,:],dim=1)
            score=(weight[:,:,None]*scores).sum(1)
            # Same VP velocity interpretation as production, a=.6,sigma=.8.
            velocity=-.8/.6*(score[:,:2]+values[:,:2])
            ez=-v.sqrt()*(score[:,2:]@L)
            return torch.cat([velocity,ez],-1)
    g=torch.Generator().manual_seed(92);x=torch.randn((4,2),dtype=dtype,generator=g);z=torch.randn((4,2),dtype=dtype,generator=g)
    v=torch.tensor([.8,1.,1.4,1.8],dtype=dtype);t=torch.full((4,),500,dtype=torch.long)
    errors=[]
    for h in [.08,.04,.02]:
        r=precision_residual(Analytic(),x,z,t,v,L,spatial_h=h,variance_h=h,drift_h=h)
        errors.append(float(r.square().mean().sqrt()))
    assert errors[2]<errors[1]<errors[0] and errors[2]<.003,errors
    # Gradients through the regularizer must reach both native output heads.
    torch.manual_seed(0)
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(5,16),torch.nn.SiLU(),torch.nn.Linear(16,4))
        def forward(self,x,t,v):return self.net(torch.cat([x,v],-1))
    model=Tiny().double();r=precision_residual(model,x,z,t,v,L,spatial_h=.2)
    r.square().mean().backward();gradient=model.net[-1].weight.grad
    assert torch.isfinite(gradient).all() and gradient[:2].abs().sum()>0 and gradient[2:].abs().sum()>0
    result=dict(status='completed',scope='synthetic-pass only',actual_production_operator_tested=True,
        analytic_nonGaussian_velocity_FD_RMSE=errors,anisotropic_cholesky_orientation_checked=True,
        gradients_reach_visual_and_EEG_heads=True,real_EEG_or_generation_effect_proven=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
