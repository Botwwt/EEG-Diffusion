"""DRAFT change to the existing EEG output target, not a post-condition adapter.

Normal inference reads EEG only. Neither image emissions nor targets enter
this module. Head and latent-chain utilities are not a confirmed innovation.
"""
import torch
from torch import nn
class EEGNativeScalePrior(nn.Module):
    def __init__(self,encoder,*,family='chain'):
        super().__init__()
        if family not in ('chain','independent','global'):raise ValueError('Registered latent family required')
        if not isinstance(encoder.head,nn.Linear):raise ValueError('Audited native EEG linear output required')
        self.encoder=encoder;self.family=family
        outputs=1000 if family=='global' else 10009
        head=nn.Linear(encoder.head.in_features,outputs)
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias);encoder.head=head
        if hasattr(encoder,'logit_scale'):encoder.logit_scale.requires_grad_(False)
    def forward(self,eeg,**kwargs):
        value=self.encoder(eeg,**kwargs)
        if isinstance(value,tuple):value=value[0]
        if self.family=='global':
            return value[:,None,:].expand(-1,10,-1),value.new_zeros(len(value),9)
        logits=value[:,:10000].reshape(len(value),10,1000)
        return logits,value[:,10000:]
