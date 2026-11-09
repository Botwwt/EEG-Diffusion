"""D17 privileged cross-brain within-concept soft supervision, absent at inference."""
import hashlib
import torch
import torch.nn.functional as F

SCHEMA='BrainHIVE_D17_crossbrain_privileged_soft_supervision_20261027_v1'
ARMS=('visual_only','crossbrain_soft','shuffled_crossbrain_soft')

def teacher_groups(subject):
    other=[i for i in range(10) if i!=subject]
    assert len(other)==9 and subject not in other
    return tuple(other[:4]),tuple(other[4:])

def centered_crossbrain_scores(a,b):
    assert a.shape==b.shape and a.ndim==3 and a.shape[1:]==(10,16)
    a=F.normalize(a.double()-a.double().mean(1,keepdim=True),dim=-1)
    b=F.normalize(b.double()-b.double().mean(1,keepdim=True),dim=-1)
    return (.5*(a@b.transpose(1,2)+b@a.transpose(1,2))).float()

def probabilities(scores):
    assert scores.ndim==3 and scores.shape[1:]==(10,10)
    result=(scores/.1).softmax(-1)
    assert torch.isfinite(result).all() and torch.allclose(result.sum(-1),torch.ones_like(result[:,:,0]),atol=1e-6)
    return result

def within_concept_permutation(concept):
    offset=1+int(hashlib.sha256(('1729|D17-neural-row-control|'+concept).encode()).hexdigest()[:8],16)%9
    return (torch.arange(10)+offset)%10

def student_losses(condition,visual,soft_targets,logit_scale):
    assert condition.shape==visual.shape and condition.ndim==2 and condition.shape[1]==1024 and len(condition)%10==0
    c=F.normalize(condition.float(),dim=-1);f=F.normalize(visual.float(),dim=-1)
    # Registered training uses true complete concept groups, every stimulus once.
    logits=c@f.T*logit_scale.exp().clamp(max=100.)
    labels=torch.arange(len(c),device=c.device)
    nce=.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
    local=torch.bmm(c.reshape(-1,10,1024),f.reshape(-1,10,1024).transpose(1,2))/.1
    assert soft_targets.shape==local.shape
    logp=F.log_softmax(local,dim=-1)
    kl=(soft_targets*(soft_targets.clamp_min(1e-20).log()-logp)).sum(-1).mean()
    fine=F.cross_entropy(local.reshape(-1,10),torch.arange(10,device=c.device).repeat(len(c)//10))
    return nce,kl,fine

def self_test():
    torch.set_num_threads(2);torch.manual_seed(1729)
    for si in range(10):
        a,b=teacher_groups(si);assert len(a)==4 and len(b)==5 and set(a).isdisjoint(b) and si not in a+b
    a=torch.randn(8,10,16,dtype=torch.float64);score=centered_crossbrain_scores(a,a)
    assert torch.equal(score.argmax(-1),torch.arange(10).expand(8,-1)) and torch.allclose(score,score.transpose(1,2),atol=1e-7)
    shifted=centered_crossbrain_scores(a+17.,a-23.);assert torch.allclose(score,shifted,atol=1e-6)
    p=probabilities(score);perm=within_concept_permutation('fixed');wrong=p[:,perm]
    assert (perm!=torch.arange(10)).all()
    # Exact row permutation preserves row entropy as a multiset and column sums.
    assert torch.allclose(p.sum(1),wrong.sum(1),atol=1e-6)
    entropy=lambda x:-(x*x.clamp_min(1e-20).log()).sum(-1)
    assert torch.allclose(entropy(p).sort(-1).values,entropy(wrong).sort(-1).values)
    c=torch.randn(20,1024,requires_grad=True);v=torch.randn(20,1024);soft=torch.ones(2,10,10)/10
    nce,kl,fine=student_losses(c,v,soft,torch.tensor(2.6592));(nce+.1*kl).backward()
    assert torch.isfinite(c.grad).all() and c.grad.norm()>0 and kl>=-1e-6
    local=F.normalize(c.detach(),dim=-1).reshape(2,10,1024)@F.normalize(v,dim=-1).reshape(2,10,1024).transpose(1,2)/.1
    own=local.softmax(-1);_,zero,_=student_losses(c.detach(),v,own,torch.tensor(2.6592));assert abs(zero)<1e-6
    return dict(status='D17_NEURAL_GROUP_EXCLUSION_CENTERING_SOFT_KL_AND_CONTROL_CPU_CHECKS_PASSED',checks=6,CUDA_initialized=torch.cuda.is_initialized())
