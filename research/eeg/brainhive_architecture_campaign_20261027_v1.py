"""A fixed architecture campaign: replacements, not extra condition adapters."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

METHODS={
 'D21_full63_mlp':dict(channels=63,output='F1024',architecture='native BrainMLP with15750 inputs; one1024 residual block',matched_control=None,complete400_as_control=True),
 'D22_full63_multiscale_transformer':dict(channels=63,output='F1024',architecture='three depthwise temporal scales, spatial mixing, two temporal Transformer layers',matched_control='D21_full63_mlp',complete400_as_control=False),
 'D23_wide17_vector':dict(channels=17,output='F1024',architecture='2048-wide three-block EEG residual backbone;1024 nativeF head',matched_control=None,complete400_as_control=True),
 'D24_wide17_direct_IP':dict(channels=17,output='IP4x2048',architecture='same2048-wide three-block EEG backbone; direct four native IP tokens; native A/P bypassed',matched_control='D23_wide17_vector',complete400_as_control=False),
 'D25_deterministic_Euler4':dict(channels=17,output='old_ordinary_F1024',architecture='replace ancestral Euler with deterministic Euler; four native UNet evaluations',matched_control=None,complete400_as_control=False),
}
SUBJECTS=tuple(f'sub-{i:02d}' for i in range(1,11))
TRAIN_METHODS=tuple(x for x in METHODS if x!='D25_deterministic_Euler4')
SCHEMA='BrainHIVE_architecture_replacements_fixed_campaign_20261027_v1'

class Residual2048(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(2048);self.first=nn.Linear(2048,2048);self.second=nn.Linear(2048,2048);self.dropout=nn.Dropout(.1)
    def forward(self,x):return x+self.dropout(self.second(F.gelu(self.first(self.norm(x)))))

class WideEEG(nn.Module):
    def __init__(self,direct_tokens=False,gamma=None,beta=None):
        super().__init__();self.direct_tokens=direct_tokens
        self.input=nn.Linear(4250,2048);self.input_norm=nn.LayerNorm(2048);self.dropout=nn.Dropout(.1)
        self.blocks=nn.Sequential(*[Residual2048() for _ in range(3)])
        self.head=nn.Linear(2048,8192 if direct_tokens else 1024)
        if direct_tokens:
            assert gamma is not None and beta is not None and gamma.shape==beta.shape==(2048,)
            self.register_buffer('token_gamma',gamma.detach().float().clone());self.register_buffer('token_beta',beta.detach().float().clone())
        else:self.output_norm=nn.LayerNorm(1024)
    def forward(self,x):
        assert x.shape[1:]==(17,250)
        h=self.dropout(F.gelu(self.input_norm(self.input(x.flatten(1)))));h=self.blocks(h);out=self.head(h)
        if self.direct_tokens:
            return F.layer_norm(out.reshape(len(x),4,2048),(2048,),self.token_gamma.to(out.dtype),self.token_beta.to(out.dtype),1e-5)
        return self.output_norm(out)

class MultiScaleSpatialTemporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales=nn.ModuleList([nn.Conv1d(63,63,kernel_size=k,stride=5,padding=k//2,groups=63) for k in [5,25,65]])
        self.spatial=nn.Conv1d(189,512,1)
        self.temporal=nn.Sequential(nn.Conv1d(512,512,5,stride=2,padding=2,groups=512),nn.GELU(),nn.Conv1d(512,512,1))
        self.position=nn.Parameter(torch.zeros(1,25,512));self.input_norm=nn.LayerNorm(512)
        layer=nn.TransformerEncoderLayer(d_model=512,nhead=8,dim_feedforward=2048,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.pool_query=nn.Parameter(torch.zeros(512));self.head=nn.Linear(512,1024);self.output_norm=nn.LayerNorm(1024)
        nn.init.normal_(self.position,std=.01);nn.init.normal_(self.pool_query,std=.01)
    def forward(self,x):
        assert x.shape[1:]==(63,250)
        y=torch.cat([F.gelu(layer(x)) for layer in self.scales],dim=1);assert y.shape[1:]==(189,50)
        y=self.temporal(F.gelu(self.spatial(y))).transpose(1,2);assert y.shape[1:]==(25,512)
        y=self.transformer(self.input_norm(y)+self.position)
        weight=(y.float()*self.pool_query.float()[None,None]).sum(-1).div(512**.5).softmax(1).to(y.dtype)
        return self.output_norm(self.head((y*weight[:,:,None]).sum(1)))

def make_model(method,native_scope,gamma=None,beta=None):
    assert method in TRAIN_METHODS
    if method=='D21_full63_mlp':return native_scope['BrainMLP'](63*250,1024,num_layers=1,dropout=.1)
    if method=='D22_full63_multiscale_transformer':return MultiScaleSpatialTemporal()
    return WideEEG(method=='D24_wide17_direct_IP',gamma,beta)

def teacher_seed(subject,epoch,batch):
    # Isolate the frozen F dropout stream from architecture-dependent EEG RNG.
    s=f'architecture_campaign_frozen_F|1729|{subject}|{epoch}|{batch}'
    return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)%(2**63-1)

def loss(prediction,target,logit_scale):
    assert prediction.shape==target.shape
    a=F.normalize(prediction.flatten(1),dim=-1);b=F.normalize(target.flatten(1),dim=-1)
    logits=a@b.T*logit_scale.exp();label=torch.arange(len(a),device=a.device)
    return .5*(F.cross_entropy(logits,label)+F.cross_entropy(logits.T,label))

def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value

def save_tensor(path,value):
    path=path if hasattr(path,'with_name') else __import__('pathlib').Path(path);pending=path.with_name(path.name+'.pending')
    torch.save(cpu(value),pending)
    with pending.open('rb') as f:__import__('os').fsync(f.fileno())
    pending.replace(path)
