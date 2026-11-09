"""Matched two-pass EEG training with canonical combined stochastic backward.

BN changes exactly once. Stochastic CE+P uses the native single-graph sum on
the second same-mask pass: adding separately rounded gradients perturbs nearly
zero BN-preceding bias gradients through Adam. Deployment P has its own eval graph.
"""
ARMS=('nce_control','stochastic_P','deployment_P')

def symmetric_ce(net,c,target):
 import torch
 import torch.nn.functional as F
 s=net.logit_scale.exp().clamp_max(100)*F.normalize(c,dim=1)@F.normalize(target,dim=1).T
 y=torch.arange(len(c),device=c.device)
 return (F.cross_entropy(s,y)+F.cross_entropy(s.T,y))/2

def set_seed(device,seed):
 import torch
 if device.type=='cuda':torch.cuda.manual_seed(seed)
 else:torch.manual_seed(seed)

def collect(net,x,target,project,arm,seed):
 import torch
 assert arm in ARMS
 ps=list(net.parameters());named=list(net.named_parameters())
 original_buffers={n:b.detach().clone() for n,b in net.named_buffers()}
 net.train();set_seed(x.device,seed)
 ce_output=net(x,torch.ones(len(x),device=x.device,dtype=torch.long))
 ce=symmetric_ce(net,ce_output,target);ce_grad=torch.autograd.grad(ce,ps,allow_unused=True)
 ce_buffers={n:b.detach().clone() for n,b in net.named_buffers()}
 with torch.no_grad():
  for n,b in net.named_buffers():b.copy_((ce_buffers if arm=='deployment_P' else original_buffers)[n])
 net.train(arm!='deployment_P');set_seed(x.device,seed)
 p_output=net(x,torch.ones(len(x),device=x.device,dtype=torch.long))
 p_loss=(project(p_output)-project(target).detach()).square().mean()
 if arm=='stochastic_P':
  ce_second=symmetric_ce(net,p_output,target)
  assert torch.equal(ce.detach(),ce_second.detach())
  grads=list(torch.autograd.grad(ce_second+p_loss,ps,allow_unused=True))
  p_grad=[None if g is None else g-(torch.zeros_like(g) if gc is None else gc) for g,gc in zip(grads,ce_grad)]
 else:
  p_grad=torch.autograd.grad(p_loss,ps,allow_unused=True)
  grads=[]
  for p,gc,gp in zip(ps,ce_grad,p_grad):
   if arm=='nce_control':grads.append(gc);continue
   if gc is None and gp is None:grads.append(None);continue
   grads.append((torch.zeros_like(p) if gc is None else gc)+(torch.zeros_like(p) if gp is None else gp))
 with torch.no_grad():
  for n,b in net.named_buffers():b.copy_(ce_buffers[n])
 for n,b in net.named_buffers():
  if n.endswith('num_batches_tracked'):assert int(b)==int(original_buffers[n])+1
 if arm!='deployment_P':assert torch.equal(ce_output.detach(),p_output.detach())
 norm_ce=norm_p=dot=0.
 for (name,p),gc,gp in zip(named,ce_grad,p_grad):
  if name=='logit_scale':continue
  a=torch.zeros_like(p) if gc is None else gc.detach();b=torch.zeros_like(p) if gp is None else gp.detach()
  norm_ce+=float(a.double().square().sum());norm_p+=float(b.double().square().sum());dot+=float((a.double()*b.double()).sum())
 net.train()
 return grads,dict(arm=arm,CE=float(ce.detach()),P=float(p_loss.detach()),lambda_P=0. if arm=='nce_control' else 1.,
  mapper_CE_norm=norm_ce**.5,mapper_P_norm=norm_p**.5,mapper_dot=dot,
  mapper_cosine=dot/(norm_ce*norm_p)**.5 if norm_ce*norm_p>0 else None,
  P_gradient_diagnostic='joint-minus-CE in FP32' if arm=='stochastic_P' else 'direct',
  BN_updates_per_batch=1,EEG_forward_calls=2,EEG_derivative_calls=2,
  stochastic_condition_SHA_required_equal=arm!='deployment_P')

def apply_step(net,opt,grads):
 import torch
 opt.zero_grad(set_to_none=True)
 for p,g in zip(net.parameters(),grads):p.grad=None if g is None else g.clone()
 n=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.));assert n==n and n>0
 opt.step()
 return n
