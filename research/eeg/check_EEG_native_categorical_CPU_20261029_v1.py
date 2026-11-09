"""Exact normalized-mixture/gradient and native-class sampling contracts."""
import datetime,hashlib,json
from pathlib import Path
import torch
from torch import nn
from EEG_native_categorical_core_20261029_v1 import EEGNativeClassPrior,mixture_terms,conditional_loss,draw_native_class
torch.set_num_threads(2);torch.manual_seed(1729);checks={}
P=torch.tensor([[.7,.2,.1],[.15,.6,.25],[.1,.25,.65]],dtype=torch.float64) # class by observed symbol
logits=torch.tensor([[.2,-.3,.1],[.1,.3,-.1]],dtype=torch.float64,requires_grad=True);targets=torch.tensor([0,2]);ell=P[:,targets].T.log()
ev,r=mixture_terms(logits,ell);manual=(logits.softmax(-1)*P[:,targets].T).sum(-1).log()
checks['native_mixture_evidence_matches_direct_probability_sum']=torch.allclose(ev,manual,atol=1e-14,rtol=1e-14)
grad=torch.autograd.grad(-ev.mean(),logits)[0]
checks['exact_gradient_is_prior_minus_closed_teacher']=torch.allclose(grad,(logits.softmax(-1)-r)/2,atol=1e-14,rtol=1e-14)
checks['closed_teacher_rows_normalized']=torch.allclose(r.sum(-1),torch.ones(2,dtype=torch.float64),atol=1e-14)
total=sum((logits.softmax(-1)*P[:,j]).sum(-1) for j in range(3));checks['mixture_is_normalized_over_visual_outcomes']=torch.allclose(total,torch.ones(2,dtype=torch.float64),atol=1e-14)
z1=torch.zeros(2,3,dtype=torch.float64,requires_grad=True);z2=z1.detach().clone().requires_grad_(True)
a,_=conditional_loss(z1,ell,'EEG_coupled_native_class');b,_=conditional_loss(z2,ell,'visual_only_native_class')
checks['uniform_initial_main_and_control_gradients_exactly_match']=torch.allclose(torch.autograd.grad(a,z1)[0],torch.autograd.grad(b,z2)[0],atol=1e-14,rtol=1e-14)
class Encoder(nn.Module):
    def __init__(self):super().__init__();self.body=nn.Linear(5,7);self.head=nn.Linear(7,1024)
    def forward(self,x):return self.head(self.body(x).tanh()),None,None
p=EEGNativeClassPrior(Encoder()).eval();x=torch.randn(3,5);scores=p(x)
checks['EEG_initialization_uniform_native_support']=torch.equal(scores,torch.zeros(3,1000))
optimizer=torch.optim.AdamW(p.parameters(),lr=.01)
for _ in range(3):
    optimizer.zero_grad();loss,_=conditional_loss(p(x),torch.randn(3,1000).abs().neg(),'EEG_coupled_native_class');loss.backward();optimizer.step()
checks['real_EEG_backbone_receives_gradient_after_head_learning']=p.encoder.body.weight.grad is not None and bool(p.encoder.body.weight.grad.norm()>0)
g1=torch.Generator().manual_seed(1729);g2=torch.Generator().manual_seed(1729);a=draw_native_class(p(x),generator=g1);b=draw_native_class(p(x),generator=g2)
checks['single_native_draw_seed_exact_and_in_valid_support']=torch.equal(a,b) and bool((a>=0).all() and (a<1000).all())
try:mixture_terms(scores,torch.ones_like(scores));checks['invalid_likelihood_rejected']=False
except ValueError:checks['invalid_likelihood_rejected']=True
checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),checks
L=Path(__file__).resolve().parent;out=L/'audit/EEG_native_categorical_20261029_v1';out.mkdir();result=dict(status='PASS_EXACT_NATIVE_CLASS_MIXTURE_CPU_CONTRACT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,source_sha256=hashlib.sha256((L/'EEG_native_categorical_core_20261029_v1.py').read_bytes()).hexdigest(),real_EEG_or_generator_weights=False,mathematical_originality_claimed=False)
(out/'CPU_actual_v1.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
