"""Fixed complete-generator EEG targets and their matched controls.

No image target is accepted by either normal model forward. Likelihoods enter
only the separate training_loss function. HMM/mixtures are classical tools.
"""
import torch
from torch import nn
from EEG_full_native_chain_objective_20261102_v1 import complete_log_evidence
ARMS=('global_full','global_coarse','chain_full','independent_full','visual_global_full','image_prior_full')
def family(arm):
 if arm not in ARMS:raise ValueError('One registered complete study arm required')
 return 'chain' if arm=='chain_full' else 'independent' if arm=='independent_full' else 'global'
class EEGDistribution(nn.Module):
 def __init__(self,encoder,arm):
  super().__init__();self.arm=arm;self.family=family(arm);self.encoder=encoder
  if arm=='image_prior_full':raise ValueError('Image prior must use the EEG-independent class')
  if not isinstance(encoder.head,nn.Linear):raise ValueError('Audited EEG output projection required')
  outputs=10009 if self.family!='global' else 1000
  head=nn.Linear(encoder.head.in_features,outputs);nn.init.zeros_(head.weight);nn.init.zeros_(head.bias);encoder.head=head
  if hasattr(encoder,'logit_scale'):encoder.logit_scale.requires_grad_(False)
 def forward(self,eeg,*,input_chans):
  raw=self.encoder(eeg,input_chans=input_chans)
  if isinstance(raw,tuple):
   if len(raw)!=3 or raw[1] is not None or raw[2] is not None:raise ValueError('Visual/text side inputs are not normal EEG inference')
   raw=raw[0]
  if self.family=='global':return raw[:,None,:].expand(-1,10,-1),raw.new_zeros(len(raw),9)
  return raw[:,:10000].reshape(len(raw),10,1000),raw[:,10000:]
class ImagePrior(nn.Module):
 def __init__(self):
  super().__init__();self.arm='image_prior_full';self.family='global';self.logits=nn.Parameter(torch.zeros(1000))
 def forward(self,eeg,*,input_chans=None):
  n=len(eeg);return self.logits[None,None].expand(n,10,1000),self.logits.new_zeros(n,9)
def training_loss(logits,stay,emissions,arm):
 mode=family(arm)
 if emissions.requires_grad:raise ValueError('Frozen precomputed image likelihoods cannot receive gradients')
 if emissions.shape[1:]!=(10,1000) or logits.shape!=emissions.shape:raise ValueError('All10 scales/1000 classes remain available in the bank')
 count=2 if arm=='global_coarse' else 10
 evidence=complete_log_evidence(logits[:,:count],stay[:,:count-1],emissions[:,:count],mode=mode)
 if arm=='visual_global_full':
  teacher=emissions.double().sum(1).softmax(-1).detach()
  loss=-(teacher*logits[:,0].double().log_softmax(-1)).sum(-1).mean()
 else:loss=-evidence.mean()
 return loss,dict(observed_negative_log_evidence_nats=-evidence.detach(),optimized_objective='fixed_visual_posterior_cross_entropy' if arm=='visual_global_full' else 'negative_exact_marginal_log_evidence',optimized_scales=count)
def sampling_parameters(logits,stay,arm):
 if logits.ndim!=3 or logits.shape[1:]!=(10,1000) or stay.shape!=(len(logits),9):raise ValueError('Complete EEG-only output required')
 f=family(arm)
 value=stay.double()
 rho=torch.ones_like(value) if f=='global' else torch.zeros_like(value) if f=='independent' else value.sigmoid()
 return logits.double(),rho
