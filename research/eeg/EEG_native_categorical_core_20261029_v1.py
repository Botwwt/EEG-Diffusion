"""EEG probabilities over the generator's fixed native1000 condition support.

For frozen native log-likelihood ell_i(k), the exact conditional evidence is
logsumexp(log pi_EEG(k)+ell_i(k)). Its latent posterior is a closed softmax;
gradient wrt EEG logits is pi-r. These are classical mixture identities.
"""
import torch
from torch import nn

class EEGNativeClassPrior(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        if not isinstance(encoder.head,nn.Linear):raise ValueError('Audited EEG linear head required')
        head=nn.Linear(encoder.head.in_features,1000);nn.init.zeros_(head.weight);nn.init.zeros_(head.bias);encoder.head=head
        if hasattr(encoder,'logit_scale'):encoder.logit_scale.requires_grad_(False)
    def forward(self,eeg,**kwargs):
        result=self.encoder(eeg,**kwargs)
        if isinstance(result,tuple):result=result[0]
        if result.shape!=(len(eeg),1000):raise ValueError('1000 native latent-condition logits required')
        return result

def mixture_terms(logits,log_likelihood):
    if logits.ndim!=2 or logits.shape!=log_likelihood.shape or logits.shape[1]<2:raise ValueError('Paired BxK logits/native likelihoods required')
    if not torch.isfinite(logits).all() or not torch.isfinite(log_likelihood).all() or (log_likelihood>0).any():raise ValueError('Finite logits and nonpositive native log likelihood required')
    log_prior=logits.log_softmax(-1);joint=log_prior+log_likelihood
    return joint.logsumexp(-1),joint.softmax(-1)

def conditional_loss(logits,log_likelihood,arm):
    log_evidence,posterior=mixture_terms(logits,log_likelihood)
    if arm=='EEG_coupled_native_class':loss=-log_evidence.mean()/5
    elif arm=='visual_only_native_class':loss=-(log_likelihood.softmax(-1).detach()*logits.log_softmax(-1)).sum(-1).mean()/5
    else:raise ValueError('Fixed main/control arm required')
    return loss,dict(native_coarse_NLL_nats_per_query=-log_evidence.detach(),posterior=posterior.detach())

@torch.no_grad()
def draw_native_class(logits,*,generator):
    if logits.ndim!=2 or logits.shape[1]!=1000 or not torch.isfinite(logits).all():raise ValueError('Finite1000-class logits required')
    # One uniform value per query allows exactly shared latent randomness
    # between methods; no best-class/mean projection or image selection.
    u=torch.rand((len(logits),1),device=logits.device,dtype=logits.dtype,generator=generator)
    return (logits.softmax(-1).cumsum(-1)<u).sum(-1).clamp_max(999)
