"""Training-only response targets. No teacher or extra module at inference.

The main target is the minimum response displacement preserving the updated
EEG teacher's visual-distance contrasts. The control has identical displacement
energy AND identical initial repair error, with a seeded orthogonal direction.
Classical projection and a matched control, not a novelty or efficacy claim.
"""
def response(c,Q,w):
 import torch
 import torch.nn.functional as F
 p=F.linear(c,w['P_weight'],w['P_bias']).reshape(-1,4,2048)
 p=F.layer_norm(p,(2048,),w['P_gamma'],w['P_beta'],1e-5)
 h,d=Q.shape[-3],Q.shape[-1]
 k=F.linear(p,w['K']).reshape(-1,4,h,d).transpose(1,2)
 v=F.linear(p,w['V']).reshape(-1,4,h,d).transpose(1,2)
 return (torch.softmax(Q@k.transpose(-1,-2)/d**.5,-1)@v).flatten(1)

def targets(old,updated,visual,random_direction,eigen_relative=1e-12):
 import torch
 assert old.shape==updated.shape and visual.ndim==2 and visual.shape[1]==old.shape[1]
 # All target construction is detached and FP64. Teacher weights never update.
 with torch.no_grad():
  a=old.double();delta=updated.double()-a
  D=visual.double()-visual.double().mean(0,keepdim=True)
  gram=D@D.T;lam,U=torch.linalg.eigh(gram)
  active=lam>lam.max()*eigen_relative
  assert int(active.sum())>0 and float(lam.min())>=-1e-10*float(lam.max())
  pinv=(U[:,active]/lam[active])@U[:,active].T
  projection=((delta@D.T)@pinv)@D
  dd=delta.square().sum(1);pp=projection.square().sum(1)
  beta=torch.where(dd>0,pp/dd.clamp_min(1e-300),torch.zeros_like(dd))
  assert float(beta.max())<1+1e-7 and float(beta.min())>=0
  beta=beta.clamp(0,1)
  u=random_direction.double()
  u=u-((u*delta).sum(1)/dd.clamp_min(1e-300))[:,None]*delta
  un=u.square().sum(1).sqrt();assert bool((un>1e-12).all())
  u=u/un[:,None]
  matched=beta[:,None]*delta+(beta*(1-beta)*dd).clamp_min(0).sqrt()[:,None]*u
  scale=D.square().mean();assert float(scale)>1e-12
  proof=dict(rank=int(active.sum()),teacher_count=len(D),response_dimension=D.shape[1],
   minimum_eigen=float(lam.min()),maximum_eigen=float(lam.max()),
   projection_idempotence_relative=float(((projection@D.T)@pinv@D-projection).norm()/projection.norm().clamp_min(1e-300)),
   teacher_contrast_relative_residual=float(((projection-delta)@D.T).norm()/(delta@D.T).norm().clamp_min(1e-300)),
   orthogonality_relative=float(abs(((delta-projection)*projection).sum(1)).max()/dd.max().clamp_min(1e-300)),
   control_energy_relative_error=float(abs(matched.square().sum(1)-pp).max()/dd.max().clamp_min(1e-300)),
   control_initial_error_relative_error=float(abs((matched-delta).square().sum(1)-(projection-delta).square().sum(1)).max()/dd.max().clamp_min(1e-300)),
   mean_retained_energy_fraction=float(beta.mean()),scale=float(scale),
   control_teacher_contrast_relative_error=float(((matched-delta)@D.T).norm()/(delta@D.T).norm().clamp_min(1e-300)))
  for key in ['projection_idempotence_relative','teacher_contrast_relative_residual','orthogonality_relative','control_energy_relative_error','control_initial_error_relative_error']:
   assert proof[key]<1e-5,(key,proof[key])
  return dict(main=(a+projection).to(old.dtype),matched_direction=(a+matched).to(old.dtype)),scale.to(old.dtype),proof

def loss(value,target,scale):
 return (value-target.detach()).square().mean()/scale.detach()

def prepare_student(net):
 # Deployment mode is intentional: fixed BN and no dropout; gradients still flow.
 net.eval()
 active=[]
 for name,p in net.named_parameters():
  use=not (name.startswith('proj_eeg.') or name=='logit_scale')
  p.requires_grad_(use)
  if use:active.append(p)
 assert active
 return active
