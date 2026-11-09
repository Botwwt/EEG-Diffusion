"""Native EEG function anchor: train the EEG network itself under a fixed angular cap."""
import copy
import torch
from torch import nn
from PKU_eeg_candidate_core_20261109_v266 import make_model as native_model,posterior_indices,seed,negative_cycle,sha
CASES=['C_positive','C_pair','C_reverse']

def constrain(candidate,anchor,weight,bias,rho=.25):
    u0=(anchor.float()-bias.float())/weight.float();u=(candidate.float()-bias.float())/weight.float();d=u-u0;d=d-d.mean(-1,keepdim=True)
    base_norm=u0.norm(dim=-1,keepdim=True);r2=base_norm.square();assert (r2>1e-8).all() and (weight.abs()>1e-6).all()
    d=d-(d*u0).sum(-1,keepdim=True)/r2*u0
    delta=rho*r2.sqrt()*d/(r2+d.square().sum(-1,keepdim=True)).sqrt()
    scale=base_norm/(u0+delta).norm(dim=-1,keepdim=True)
    result=anchor.float()+weight.float()*((scale-1)*u0+scale*delta)
    return result

class AnchoredEEGEncoder(nn.Module):
    def __init__(self,device='cuda'):
        super().__init__();self.candidate=native_model('A_positive',{},device=device);self.anchor=copy.deepcopy(self.candidate).requires_grad_(False).eval();self.rho=.25
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==6450176
    def forward(self,posterior):
        candidate=self.candidate(posterior)
        with torch.no_grad():anchor=self.anchor(posterior)
        weight=self.anchor.block.layers[0].norm.weight;bias=self.anchor.block.layers[0].norm.bias
        return constrain(candidate,anchor,weight,bias,self.rho)

def make_model(case,stats=None,device='cuda'):
    assert case in CASES
    return AnchoredEEGEncoder(device).eval()

def encode(model,case,eeg,post,wrong=None):
    assert case in CASES and wrong is None
    return model(eeg[:,post,:])
