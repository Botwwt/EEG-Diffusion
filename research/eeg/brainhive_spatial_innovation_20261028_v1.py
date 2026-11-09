"""D40 native first-projection expansion from true additional EEG electrodes.

Schur complements and matrix-normal whitening are classical. Their use here
is a working noise model, not a claim of calibrated EEG or image improvement.
"""
import torch
from torch import nn
from torch.nn import functional as F
ARMS=('marginal','measurement','permuted_measurement')
RHO=.25

def symmetric_power(x,power):
    v,U=torch.linalg.eigh((x+x.T)/2);assert v.min()>0
    return (U*v.pow(power)[None])@U.T

def geometry(spatial,temporal,W0,posterior=17):
    spatial=spatial.double();temporal=temporal.double();C=len(spatial);T=len(temporal);E=C-posterior
    scale=spatial.trace()/C;assert scale>0
    S=.9*spatial+.1*scale*torch.eye(C,dtype=torch.float64)
    Theta=.9*temporal/scale+.1*torch.eye(T,dtype=torch.float64)
    PP=S[E:,E:];EP=S[:E,E:];EE=S[:E,:E]
    L=torch.linalg.solve(PP,EP.T).T;conditional=EE-L@EP.T
    temporal_white=symmetric_power(Theta,-.5)
    marginal_white=symmetric_power(EE,-.5);conditional_white=symmetric_power(conditional,-.5)
    w=W0.double().reshape(len(W0),posterior,T)
    whitened=torch.matmul(symmetric_power(PP,.5),w)@symmetric_power(Theta,.5)
    energy=whitened.square().sum();radius=RHO*energy.sqrt()
    return dict(working_spatial=S,working_temporal=Theta,L=L,conditional_covariance=conditional,
      marginal_white=marginal_white,conditional_white=conditional_white,temporal_white=temporal_white,
      original_hidden_working_noise_energy=energy,weight_Frobenius_radius=radius)

def innovation(x,mean,L,left,right):
    E=len(L);centered=x-mean
    residual=centered[:,:E]-torch.matmul(L,centered[:,E:])
    return (torch.matmul(left,residual)@right).flatten(1)

def folded_first_layer(W0,b0,B,mean,L,left,right):
    H=len(W0);E=len(L);T=mean.shape[1]
    WE=torch.matmul(left.T,B.reshape(H,E,T))@right.T
    WP=W0.reshape(H,mean.shape[0]-E,T)-torch.einsum('het,ep->hpt',WE,L)
    mu=mean[:E]-L@mean[E:]
    bias=b0-(WE*mu[None]).sum((1,2))
    return torch.cat([WE.flatten(1),WP.flatten(1)],1),bias

class SpatialNative(nn.Module):
    def __init__(self,parent,stats,mean,arm):
        super().__init__();assert arm in ARMS
        self.parent=parent.eval().requires_grad_(False);self.arm=arm
        left=stats['marginal_white'] if arm=='marginal' else stats['conditional_white']
        L=torch.zeros_like(stats['L']) if arm=='marginal' else stats['L']
        for name,value in [('mean',mean),('L',L),('left',left),('right',stats['temporal_white']),('radius',stats['weight_Frobenius_radius'])]:
            self.register_buffer(name,value.detach().float().clone())
        self.B=nn.Parameter(torch.zeros(parent.proj_in.out_features,len(L)*mean.shape[1]))
        assert self.B.numel()==11776000
    def forward(self,x,innovation_source=None,enabled=True):
        assert x.ndim==3 and x.shape[1:]==(63,250)
        original=x[:,46:].flatten(1)
        if not enabled:return self.parent(original)
        z=innovation(x if innovation_source is None else innovation_source,self.mean,self.L,self.left,self.right)
        hidden=self.parent.proj_in(original)+F.linear(z,self.B)
        return self.parent.block(hidden)
    @torch.no_grad()
    def project(self):
        norm=self.B.norm();scale=torch.clamp(self.radius/(norm+1e-30),max=1.)
        self.B.mul_(scale)
        assert self.B.norm()<=self.radius*(1+2e-6)
        return float(scale),float((self.B.norm()/self.radius).square())
    def folded_state(self):
        state={k:v.detach().cpu().clone() for k,v in self.parent.state_dict().items()}
        w,b=folded_first_layer(state['proj_in.weight'].double(),state['proj_in.bias'].double(),self.B.detach().cpu().double(),
          self.mean.cpu().double(),self.L.cpu().double(),self.left.cpu().double(),self.right.cpu().double())
        state['proj_in.weight']=w.float();state['proj_in.bias']=b.float();return state
