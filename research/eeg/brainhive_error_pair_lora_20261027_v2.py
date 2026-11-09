"""D20: low-rank changes to native IP K/V; original projections stay frozen.

The logistic denoising-error preference is a proxy on Turbo, not an exact
likelihood ratio or posterior. Original state files are never modified.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('paired_error','positive_only','sameconcept_wrong_error')
RANK=2
RELATIVE_WEIGHT_CAP=.05
BETA=100.

class LowRankIP(nn.Module):
    def __init__(self,base):
        super().__init__();assert isinstance(base,nn.Linear) and base.bias is None
        self.base=base.requires_grad_(False)
        self.down=nn.Parameter(torch.empty(RANK,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.up=nn.Parameter(torch.zeros(base.out_features,RANK,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down,a=math.sqrt(5));self.enabled=True
        self.original_norm=float(base.weight.detach().float().norm());self.legacy_fp16_cast=False
    def forward(self,x):
        y=self.base(x)
        if not self.enabled:return y
        # During training pass FP32 master parameters DIRECTLY into BF16
        # autocast. A prior FP16 cast inserts an FP16 gradient edge that may
        # underflow even though the actual linear kernels use BF16.
        if torch.is_autocast_enabled() and not self.legacy_fp16_cast:
            z=F.linear(F.linear(x,self.down),self.up)
        else:
            z=F.linear(F.linear(x,self.down.to(x.dtype)),self.up.to(x.dtype))
        return y+z.to(y.dtype)
    @torch.no_grad()
    def project(self):
        ratio=float((self.up@self.down).norm())/self.original_norm
        if ratio>RELATIVE_WEIGHT_CAP:self.up.mul_(RELATIVE_WEIGHT_CAP/ratio)
        return min(ratio,RELATIVE_WEIGHT_CAP)

def install(unet):
    replacements=[]
    for name,module in list(unet.named_modules()):
        if type(module).__name__=='IPAttnProcessor2_0':
            for key in ['to_k_ip','to_v_ip']:
                original=getattr(module,key);new=LowRankIP(original);setattr(module,key,new)
                replacements.append((name+'.'+key,new))
    assert len(replacements)==140
    return replacements

def enabled(layers,value):
    for _,module in layers:module.enabled=value

def new_parameters(layers):return [p for _,m in layers for p in [m.down,m.up]]

def new_state(layers):return {name:{'down':m.down.detach().cpu().clone(),'up':m.up.detach().cpu().clone()} for name,m in layers}

@torch.no_grad()
def load_state(layers,state):
    assert set(state)=={n for n,_ in layers}
    for name,m in layers:
        m.down.copy_(state[name]['down']);m.up.copy_(state[name]['up'])

def objective(student,reference,noise,arm):
    assert arm in ARMS and student.shape==reference.shape==noise.shape and len(student)%2==0
    n=len(student)//2
    errors=(student.float()-noise.float()).square().flatten(1).mean(1)
    ref_errors=(reference.float()-noise.float()).square().flatten(1).mean(1)
    positive=errors[:n].mean();change=errors-ref_errors
    margin=BETA*(change[:n]-change[n:]);preference=F.softplus(margin).mean()
    retention=(student.float()-reference.float()).square().mean()
    loss=positive+.1*retention+(preference if arm!='positive_only' else 0.*preference)
    return loss,dict(positive_MSE=positive,negative_MSE=errors[n:].mean(),reference_positive_MSE=ref_errors[:n].mean(),reference_negative_MSE=ref_errors[n:].mean(),preference_loss=preference,preference_margin=margin.mean(),reference_retention=retention)

def cpu_math_check():
    # Directly check the sign of the comparison: reducing preferred error or
    # increasing rejected error must reduce its logistic term.
    delta=torch.tensor([0.,0.],requires_grad=True)
    v=F.softplus(BETA*(delta[0]-delta[1]));v.backward()
    assert delta.grad[0]>0 and delta.grad[1]<0
    torch.manual_seed(1729);base=nn.Linear(2048,320,bias=False).eval().requires_grad_(False)
    layer=LowRankIP(base);x=torch.randn(2,4,2048)
    assert torch.equal(layer(x),base(x));layer(x).square().mean().backward()
    assert layer.up.grad is not None and layer.up.grad.norm()>0 and base.weight.grad is None
    with torch.no_grad():layer.up.normal_();layer.project()
    assert (layer.up@layer.down).norm()/layer.original_norm<=RELATIVE_WEIGHT_CAP+1e-6
    layer.enabled=False;assert torch.equal(layer(x),base(x))
    return {'preference_gradient_sign':True,'zero_and_disabled_projection_exact':True,'new_up_gradient_nonzero':True,'original_projection_gradient_absent':True,'relative_Frobenius_cap':RELATIVE_WEIGHT_CAP}
