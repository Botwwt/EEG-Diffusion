"""Two registered EEG-side candidates; known mathematical tools, unproven efficacy."""
import copy,hashlib,json
from pathlib import Path
import torch
from torch import nn
from PKU_HIVE_native_classes_20261109_v239 import classes
R=Path('/dfs/share/maleiLab/wwt/eeg')
CASES=['A_positive','A_pair','A_reverse','B_residual','B_raw','B_shuffle']
def seed(n):return int(hashlib.sha256(('1729|EEG_candidates_v266|'+n).encode()).hexdigest()[:16],16)%(2**63-1)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def posterior_indices():
    layout=json.loads((R/'data/recovery_20261108_v236/sub-03_layout.json').read_text())
    policy=json.loads((R/'code/HIVE_native_sub03_protocol_v239.json').read_text())
    p=[layout['channels'].index(n) for n in policy['electrodes']];n=[i for i in range(63) if i not in p]
    assert len(p)==17 and len(n)==46 and len(set(p+n))==63
    return p,n
def negative_cycle(ids):
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text())
    assert ids==split['image_rows']['T']
    identity=json.loads((R/'data/full_T_recovery_20261108_v238/T_identities.json').read_text());assert [r['array_row'] for r in identity['T']]==ids
    groups={}
    for i,row in enumerate(identity['T']):groups.setdefault(row['concept_id'],[]).append(i)
    assert len(groups)==1323 and all(len(v)==10 for v in groups.values())
    neg=torch.empty(13230,dtype=torch.long)
    for positions in groups.values():
        for j,i in enumerate(positions):neg[i]=positions[(j+1)%10]
    assert (neg!=torch.arange(13230)).all() and len(neg.unique())==13230
    gen=torch.Generator().manual_seed(1729);keys=sorted(groups);which=torch.randperm(len(keys),generator=gen)[:64].tolist();probe=[i for j in which for i in groups[keys[j]][:4]]
    assert len(probe)==256
    return neg,probe
@torch.no_grad()
def spatial_statistics(eeg,post,other):
    mu=eeg.mean(0);CPP=torch.zeros(17,17,device=eeg.device,dtype=torch.float64);CNP=torch.zeros(46,17,device=eeg.device,dtype=torch.float64)
    for lo in range(0,len(eeg),128):
        x=(eeg[lo:lo+128]-mu).double();p=x[:,post,:].permute(1,0,2).reshape(17,-1);n=x[:,other,:].permute(1,0,2).reshape(46,-1)
        CPP+=p@p.T;CNP+=n@p.T
    CPP/=len(eeg)*250;CNP/=len(eeg)*250;ridge=.001*CPP.trace()/17;assert ridge>0
    B=torch.linalg.solve(CPP+ridge*torch.eye(17,device=eeg.device,dtype=torch.float64),CNP.T).T
    error=(CNP-B@CPP-ridge*B).abs().max();assert float(error/(1+CNP.abs().max()))<1e-10
    return dict(mu=mu.cpu(),B=B.float().cpu(),post=post,other=other,CPP=CPP.cpu(),CNP=CNP.cpu(),ridge=float(ridge),normal_equation_scaled_error=float(error/(1+CNP.abs().max())),scope='T EEG only, no image or D/V statistics')
class SpatialInnovationEncoder(nn.Module):
    def __init__(self,base,stats,raw_control=False):
        super().__init__();self.base=base.eval().requires_grad_(False)
        self.register_buffer('mu',stats['mu'].clone());self.register_buffer('B',torch.zeros_like(stats['B']) if raw_control else stats['B'].clone())
        self.register_buffer('post',torch.tensor(stats['post'],dtype=torch.long));self.register_buffer('other',torch.tensor(stats['other'],dtype=torch.long))
        self.innovation=nn.Linear(11500,1024,bias=False);nn.init.zeros_(self.innovation.weight)
    def residual(self,eeg):
        x=eeg-self.mu;return x[:,self.other,:]-torch.einsum('np,bpt->bnt',self.B,x[:,self.post,:])
    def forward(self,eeg,residual_eeg=None):
        post=eeg[:,self.post,:].reshape(len(eeg),4250).contiguous();r=self.residual(eeg if residual_eeg is None else residual_eeg).reshape(len(eeg),11500).contiguous()
        return self.base.block(self.base.proj_in(post)+self.innovation(r))
    def folded_projection(self):
        U=self.innovation.weight.reshape(1024,46,250);W=torch.zeros(1024,63,250,device=U.device,dtype=U.dtype)
        W[:,self.other,:]=U;W[:,self.post,:]=self.base.proj_in.weight.reshape(1024,17,250)-torch.einsum('ont,np->opt',U,self.B)
        mean=self.mu[self.other,:]-self.B@self.mu[self.post,:]
        bias=self.base.proj_in.bias-self.innovation.weight@mean.reshape(-1)
        return W.reshape(1024,15750),bias
def make_model(case,stats,device='cuda'):
    assert case in CASES
    state=torch.load(R/'runs/HIVE_native_sub03_5096542/full/brain_final.pt',map_location='cpu',weights_only=True)['model']
    base=classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1);base.load_state_dict(state,strict=True);base.eval()
    if case.startswith('A'):
        base.requires_grad_(True)
        for m in base.modules():
            if isinstance(m,nn.LayerNorm):m.requires_grad_(False)
        model=base;expected=6450176
    else:model=SpatialInnovationEncoder(base,stats,raw_control=case=='B_raw');expected=11776000
    model=model.to(device).eval();assert sum(p.numel() for p in model.parameters() if p.requires_grad)==expected
    return model
def encode(model,case,eeg,post,wrong=None):
    if case.startswith('A'):return model(eeg[:,post,:])
    return model(eeg,residual_eeg=wrong)
