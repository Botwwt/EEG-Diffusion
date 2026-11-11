"""EEG backbones and RNG/BN-correct full-batch gradient caching. Slurm only."""
import ast,copy,hashlib,json,math,os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID'),'Numerical EEG modules run on Slurm compute nodes only'
import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from PKU_downstream_core_20261109_v303 import sha,write,save
R=Path('/dfs/share/maleiLab/wwt/eeg');ROOT=R/'runs/EEG_backbones_v322'
CASES=['MLP17','ATM63','Joint63','Separated63','Joint17'];ROLES=['full','oof0','oof1']

def author_atm_scope():
    from einops.layers.torch import Rearrange
    import logging
    scope=dict(torch=torch,nn=nn,F=F,np=np,math=math,Tensor=Tensor,Rearrange=Rearrange,sqrt=math.sqrt,logging=logging,dist=torch.distributed)
    selections=[('Embed',None),('SelfAttention_Family',['FullAttention','AttentionLayer']),('Transformer_EncDec',['EncoderLayer','Encoder']),('loss',None),('atms',None)]
    pins=json.loads((R/'code/EEG_backbone_author_pins_v322.json').read_text())
    for stem,names in selections:
        p=R/'code'/('ATM_author_'+stem+'_v322.py');assert sha(p)==pins[p.name]
        nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (names is None or n.name in names)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
    return scope

class PatchBlock(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(128);self.qkv=nn.Linear(128,384);self.out=nn.Linear(128,128);self.norm2=nn.LayerNorm(128)
        self.ff=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,128),nn.Dropout(.1))
    def forward(self,x,allowed=None):
        n,l,d=x.shape;q,k,v=self.qkv(self.norm1(x)).reshape(n,l,3,4,32).permute(2,0,3,1,4).unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=.1 if self.training else 0.).transpose(1,2).reshape(n,l,d)
        x=x+F.dropout(self.out(a),p=.1,training=self.training);return x+self.ff(self.norm2(x))

class PatchEEG(nn.Module):
    """Nonlinear channel-time interactions before one final global compression."""
    def __init__(self,channels,mode):
        super().__init__();assert channels in [17,63] and mode in ['joint','separated'];self.channels=channels;self.mode=mode
        self.patch=nn.Linear(25,128);self.channel_pos=nn.Parameter(torch.zeros(1,1,channels,128));self.time_pos=nn.Parameter(torch.zeros(1,10,1,128))
        self.blocks=nn.ModuleList([PatchBlock(),PatchBlock()]);self.input_proj=nn.Linear(1280,1024);self.res=nn.Sequential(nn.GELU(),nn.Linear(1024,1024),nn.GELU(),nn.Dropout(.1),nn.Linear(1024,1024));self.out_norm=nn.LayerNorm(1024)
        t=torch.arange(10).repeat_interleave(channels);c=torch.arange(channels).repeat(10)
        self.register_buffer('space_allowed',t[:,None]==t[None,:],persistent=False);self.register_buffer('time_allowed',c[:,None]==c[None,:],persistent=False)
    def forward(self,x):
        assert x.shape[1:]==(self.channels,250)
        h=self.patch(x.reshape(len(x),self.channels,10,25).transpose(1,2))+self.channel_pos+self.time_pos
        h=h.flatten(1,2)
        for i,block in enumerate(self.blocks):h=block(h,None if self.mode=='joint' else self.space_allowed if i==0 else self.time_allowed)
        h=h.reshape(len(x),10,self.channels,128).mean(2).reshape(len(x),1280);h=self.input_proj(h);return self.out_norm(h+self.res(h))

def make(case):
    assert case in CASES
    if case=='MLP17':
        from PKU_HIVE_native_classes_20261109_v239 import classes
        return classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1)
    if case=='ATM63':return author_atm_scope()['ATMS'](num_subjects=10)
    return PatchEEG(17 if case=='Joint17' else 63,'separated' if case=='Separated63' else 'joint')

def encode(model,case,x,post):
    if case in ['MLP17','Joint17']:x=x[:,post,:]
    if case=='ATM63':return model(x,torch.full((len(x),),3,dtype=torch.long,device=x.device))
    return model(x)

def affine(model,case):
    m=model.block.layers[0].norm if case=='MLP17' else model.proj_eeg[2] if case=='ATM63' else model.out_norm
    return dict(gamma=m.weight.detach().cpu().clone(),beta=m.bias.detach().cpu().clone())

def rng_state(device):return dict(CPU=torch.get_rng_state(),CUDA=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
def set_rng(state,device):
    torch.set_rng_state(state['CPU'])
    if device.type=='cuda':torch.cuda.set_rng_state(state['CUDA'],device)
def buffers(model):return {k:v.detach().clone() for k,v in model.named_buffers()}
def restore_buffers(model,state):
    with torch.no_grad():
        for k,v in model.named_buffers():v.copy_(state[k])

def nce(raw,target,scale):
    z=F.normalize(raw.float(),dim=-1);y=F.normalize(target.float(),dim=-1);logits=z@y.T*scale.exp();labels=torch.arange(len(raw),device=raw.device)
    return .5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))

def cached_backward(model,forward,x,scale,target_fn,micro=128,mixed=True):
    """One full1024 InfoNCE, same stochastic microforwards, BN updated only once."""
    device=x.device;chunks=[];targets=[];replay=[]
    for lo in range(0,len(x),micro):
        hi=min(len(x),lo+micro);replay.append((lo,hi,rng_state(device),buffers(model)))
        with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=mixed):raw=forward(x[lo:hi])
        chunks.append(raw.detach().float().clone())
        with torch.no_grad():targets.append(target_fn(lo,hi).detach().float())
    final_rng=rng_state(device);final_buffers=buffers(model)
    leaf=torch.cat(chunks).requires_grad_(True);loss=nce(leaf,torch.cat(targets),scale);loss.backward();g=leaf.grad.detach();assert torch.isfinite(g).all()
    for lo,hi,state,buf in replay:
        set_rng(state,device);restore_buffers(model,buf)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=mixed):raw=forward(x[lo:hi])
        assert torch.equal(raw.detach().float(),leaf.detach()[lo:hi]),'Stochastic/BN cache replay mismatch'
        raw.backward(g[lo:hi].to(raw.dtype))
    restore_buffers(model,final_buffers);set_rng(final_rng,device)
    return float(loss.detach())

def input_contract():
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());T=split['image_rows']['T']
    layout=json.loads((R/'data/recovery_20261108_v236/sub-03_layout.json').read_text());postnames=['P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2'];post=[layout['channels'].index(v) for v in postnames]
    import csv
    with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:rows={int(r['array_row']):r for r in csv.DictReader(f) if r['dataset_split']=='train'}
    folds=torch.tensor([split['oof_fold_by_concept'][rows[i]['concept_id']] for i in T]);assert len(T)==13230 and set(folds.tolist())=={0,1}
    return split,T,folds,post
