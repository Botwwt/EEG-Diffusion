"""Complete image likelihood under EEG-controlled native latent states.

Classical scaled HMM probability/autograd, not a novelty or efficacy claim.
Image likelihood tensors are training-only, never normal EEG inference inputs.
"""
import torch
import torch.nn.functional as F

def complete_log_evidence(logits,stay_logits,emissions,*,mode='chain'):
    if logits.ndim!=3 or emissions.shape!=logits.shape or logits.shape[-1]<2:
        raise ValueError('Matching B x complete-scales x all-classes inputs required')
    B,S,K=logits.shape
    if stay_logits.shape!=(B,S-1):raise ValueError('One transition logit per adjacent scale')
    if not all(torch.isfinite(x).all() for x in (logits,stay_logits,emissions)):
        raise ValueError('Finite complete inputs required')
    if mode not in ('chain','global','independent'):raise ValueError('Registered latent family required')
    logpi=logits.double().log_softmax(-1)
    ell=emissions.double()
    # A query/scale constant affects evidence, but cannot alter EEG gradients.
    center=ell.max(-1,keepdim=True).values.detach();relative=ell-center
    if mode=='global':
        # One native class for the whole image; first prior supplies its mass.
        return (logpi[:,0]+relative.sum(1)).logsumexp(-1)+center.squeeze(-1).sum(-1)
    if mode=='independent':
        return (logpi+relative).logsumexp(-1).sum(-1)+center.squeeze(-1).sum(-1)
    logstay=F.logsigmoid(stay_logits.double());logreset=F.logsigmoid(-stay_logits.double())
    value=logpi[:,0]+relative[:,0];scale=value.logsumexp(-1);alpha=value-scale[:,None];normalizers=[scale]
    for k in range(1,S):
        predicted=torch.logaddexp(logstay[:,k-1,None]+alpha,logreset[:,k-1,None]+logpi[:,k])
        value=predicted+relative[:,k];scale=value.logsumexp(-1)
        alpha=value-scale[:,None];normalizers.append(scale)
    return torch.stack(normalizers,1).sum(-1)+center.squeeze(-1).sum(-1)

def full_sequence_loss(logits,stay_logits,emissions,*,mode='chain'):
    # One common normalization for comparisons; do not change scale with 5 vs680.
    evidence=complete_log_evidence(logits,stay_logits,emissions,mode=mode)
    return -evidence.mean(),dict(negative_log_evidence_nats_per_query=-evidence.detach())
