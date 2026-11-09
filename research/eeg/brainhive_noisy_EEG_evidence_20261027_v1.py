"""D13 balanced noisy joint/product EEG evidence; not the old VAE p/q study."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SCHEMA='BrainHIVE_noisy_F_EEG_joint_product_evidence_20261027_v1'
ARMS=('c_only','raw_EEG','shuffled_raw_train')
TIMES=(.125,.375,.625,.875)


class Evidence(nn.Module):
    def __init__(self):
        super().__init__(); self.subject=nn.Embedding(10,16)
        self.c=nn.Linear(1040,256); self.raw=nn.Linear(4250,256); self.x=nn.Linear(1024,256)
        self.pair=nn.Linear(5*256+32+16,256)
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(256),nn.Linear(256,512),nn.SiLU(),nn.Linear(512,256)) for _ in range(3)])
        self.out=nn.Linear(256,1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer('frequencies',torch.exp(torch.linspace(0.,math.log(16.),16)))

    def forward(self,x,t,c,b,subject,arm):
        assert arm in ARMS
        c=F.normalize(c.float(),dim=-1)*32
        slot=c.repeat(1,5)[:,:4250] if arm=='c_only' else b
        phase=t[:,None]*self.frequencies[None]*(2*math.pi)
        s=self.subject(subject); a=self.x(x); brain=self.c(torch.cat([c,s],-1))+self.raw(slot)
        h=self.pair(torch.cat([a,brain,a*brain,a.square(),brain.square(),phase.sin(),phase.cos(),s],-1))
        for block in self.blocks: h=h+block(h)/math.sqrt(3.)
        # x_0 is independent Gaussian in BOTH classes, so r_0=0 by construction.
        return t*self.out(h).squeeze(-1)


def product_indices(subjects,stimuli=13230,generator=None):
    # Sampling with replacement includes self-pairs: exact empirical product,
    # not a hard-negative/complement distribution with a changed denominator.
    return subjects*stimuli+torch.randint(stimuli,subjects.shape,device=subjects.device,generator=generator)


def arrays(positive,negative):
    p=torch.as_tensor(positive,dtype=torch.float64,device='cpu'); n=torch.as_tensor(negative,dtype=torch.float64,device='cpu')
    return dict(log_loss=(.5*(F.softplus(-p)+F.softplus(n))).numpy(),
        Brier=(.5*((p.sigmoid()-1).square()+n.sigmoid().square())).numpy())


def auc(labels,scores):
    order=np.argsort(scores,kind='stable'); sorted_scores=scores[order]; ranks=np.empty(len(order),np.float64)
    lo=0
    while lo<len(order):
        hi=lo+1
        while hi<len(order) and sorted_scores[hi]==sorted_scores[lo]: hi+=1
        ranks[order[lo:hi]]=(lo+1+hi)/2; lo=hi
    pos=np.asarray(labels,dtype=bool); n1=int(pos.sum()); n0=len(pos)-n1
    return float((ranks[pos].sum()-n1*(n1+1)/2)/(n1*n0))


def metrics(positive,negative):
    p=np.asarray(positive,np.float64).reshape(-1); n=np.asarray(negative,np.float64).reshape(-1)
    values=arrays(p,n); logits=np.concatenate([p,n]); labels=np.concatenate([np.ones(len(p)),np.zeros(len(n))])
    probability=torch.as_tensor(logits,device='cpu').sigmoid().numpy(); bins=[]; ece=0.
    for k in range(10):
        mask=(probability>=k/10)&(probability<(k+1)/10 if k<9 else probability<=1)
        if mask.any():
            conf=float(probability[mask].mean()); observed=float(labels[mask].mean()); count=int(mask.sum())
            bins.append(dict(bin=k,count=count,probability=conf,observed_positive=observed)); ece+=count/len(labels)*abs(conf-observed)
    return dict(log_loss=float(values['log_loss'].mean()),Brier=float(values['Brier'].mean()),
        AUROC=auc(labels,logits),ECE10=float(ece),calibration_bins=bins,balanced_class_prior=.5)


def interval(values,queries):
    values=np.asarray(values,np.float64); assert values.shape==(len(queries),)
    concepts=sorted({q['concept_id'] for q in queries})
    grouped=np.asarray([values[[i for i,q in enumerate(queries) if q['concept_id']==c]].mean() for c in concepts])
    draws=np.random.default_rng(1729).integers(0,len(grouped),(1000,len(grouped)))
    return dict(difference=float(values.mean()),descriptive_concept_bootstrap95=np.quantile(grouped[draws].mean(1),[.025,.975]).tolist(),
        positive_concepts=int((grouped>0).sum()),concepts=len(grouped),draws=1000,seed=1729)


def readout(logits,retrieval,queries):
    # logits: partition -> arm -> [N, four t, positive/product].
    result={}; improvements={}
    for split,arms in logits.items():
        result[split]={arm:dict(pooled=metrics(z[:,:,0],z[:,:,1]),
            by_time={str(t):metrics(z[:,j,0],z[:,j,1]) for j,t in enumerate(TIMES)}) for arm,z in arms.items()}
        qs=queries[split]; raw=arrays(arms['raw_EEG'][:,:,0],arms['raw_EEG'][:,:,1]); comparisons={}
        for base in ('c_only','shuffled_raw_train'):
            other=arrays(arms[base][:,:,0],arms[base][:,:,1])
            comparisons[base]={key:interval((other[key]-raw[key]).mean(1),qs) for key in raw}
        prior=math.log(2)-arrays(arms['c_only'][:,:,0],arms['c_only'][:,:,1])['log_loss'].mean(1)
        comparisons['c_only_vs_constant']=interval(prior,qs); improvements[split]=comparisons
    fit=improvements['Dfit']; check=improvements['Dcheck']; means=result['Dcheck']
    base_support=check['c_only_vs_constant']['difference']>=.005 and check['c_only_vs_constant']['descriptive_concept_bootstrap95'][0]>0
    raw_support=all(check[b]['log_loss']['difference']>=.005 and check[b]['log_loss']['descriptive_concept_bootstrap95'][0]>0
        and check[b]['Brier']['difference']>0 and fit[b]['log_loss']['difference']>0 for b in ('c_only','shuffled_raw_train'))
    raw_support=raw_support and means['raw_EEG']['pooled']['log_loss']<math.log(2)-.005
    raw_support=raw_support and retrieval['Dcheck']['raw_EEG']['top1']>=.12 and retrieval['Dcheck']['raw_EEG']['top1']>=retrieval['Dcheck']['c_only']['top1']
    action=('RAW_EEG_EVIDENCE_SUPPORTS_SEPARATE_SAMPLING_TEST' if raw_support else
        'BASE_C_ONLY_EVIDENCE_SUPPORTS_SEPARATE_TEST_NO_RAW_INCREMENT' if base_support else 'STOP_NO_HELD_PREDICTIVE_EVIDENCE')
    return dict(status='COMPLETE_D13_FIXED_NOISY_EEG_EVIDENCE_READOUT',classification=result,paired_improvements=improvements,
        sameconcept_retrieval=retrieval,raw_EEG_increment_supported=bool(raw_support),base_c_pairing_supported=bool(base_support),
        decision=dict(action=action,automatic_image_admission=False),
        endpoints_not_used_as_neural_evidence=True,all_four_registered_times_equally_weighted=True,
        source_domain='standardized full1024 F, not VAE image latent',old_B_study_completed=False,
        actual_image_or_EEG_reconstruction_gain_established=False,independent_confirmation=False,SOTA_established=False)


def self_test():
    torch.set_num_threads(2); torch.manual_seed(1729); dtype=torch.float64
    eye=torch.eye(2,dtype=dtype); H=torch.tensor([[1.,.2],[.3,1.]],dtype=dtype)
    R=torch.tensor([[.4,.15],[.15,.6]],dtype=dtype); b=torch.tensor([.4,-.7],dtype=dtype)
    def posterior(h,r,obs):
        cov=eye-h.T@torch.linalg.solve(h@h.T+r,h)
        mean=h.T@torch.linalg.solve(h@h.T+r,obs); return mean,cov
    mf,Sf=posterior(H,R,b); mc,Sc=posterior(H[:1],R[:1,:1],b[:1]); errors=[]
    for time in (.05,.125,.375,.625,.875,.95):
        x=torch.tensor([.31,-.28],dtype=dtype,requires_grad=True); t=torch.tensor(time,dtype=dtype)
        q=(t*t+(1-t)**2)*eye
        def lognormal(mu,cov):
            z=x-mu; return -.5*(z@torch.linalg.solve(cov,z)+torch.linalg.slogdet(cov)[1])
        cf=t*t*Sf+(1-t)**2*eye; cc=t*t*Sc+(1-t)**2*eye
        dc=lognormal(t*mc,cc)-lognormal(torch.zeros(2,dtype=dtype),q)
        df=lognormal(t*mf,cf)-lognormal(torch.zeros(2,dtype=dtype),q)
        gc=torch.autograd.grad(dc,x,retain_graph=True)[0]; gf=torch.autograd.grad(df,x,retain_graph=True)[0]
        sq=-torch.linalg.solve(q,x); sf=-torch.linalg.solve(cf,x-t*mf); sc=-torch.linalg.solve(cc,x-t*mc)
        errors.append(float((gf-gc-(sf-sc)).abs().max()))
        # Direct conditional mean velocity of X=tY+(1-t)Z, compared to score conversion.
        direct=mf+(t*Sf-(1-t)*eye)@torch.linalg.solve(cf,x-t*mf)
        converted=x/t+(1-t)/t*(sq+gf)
        errors.append(float((direct-converted).abs().max()))
        increment=torch.autograd.grad(df-dc,x)[0]
        errors.append(float((increment-(sf-sc)).abs().max()))
    assert max(errors)<1e-10
    # Additional measurement is correlated with c, but has no information about Y given c.
    var_c=1.4; alpha=.7; var_eta=.6
    yy=torch.tensor([[1.,0.],[0.,1.]],dtype=dtype)
    yz=torch.tensor([[1.,alpha],[0.,0.]],dtype=dtype)
    zz=torch.tensor([[var_c,alpha*var_c],[alpha*var_c,alpha*alpha*var_c+var_eta]],dtype=dtype)
    z=torch.tensor([.4,-.2],dtype=dtype)
    full_cov=yy-yz@torch.linalg.solve(zz,yz.T); full_mean=yz@torch.linalg.solve(zz,z)
    base_cov=yy-yz[:,:1]@torch.linalg.solve(zz[:1,:1],yz[:,:1].T)
    base_mean=yz[:,:1]@torch.linalg.solve(zz[:1,:1],z[:1])
    sufficient_error=max(float((full_cov-base_cov).abs().max()),float((full_mean-base_mean).abs().max()))
    assert sufficient_error<1e-12
    subjects=torch.tensor([0]*128+[1]*128); k=product_indices(subjects,4,torch.Generator().manual_seed(1729))
    assert torch.equal(k//4,subjects) and any((k%4)==torch.arange(256)%4)
    null=metrics(np.zeros(12),np.zeros(12)); assert abs(null['log_loss']-math.log(2))<1e-12 and null['Brier']==.25 and null['AUROC']==.5
    head=Evidence().eval().requires_grad_(False)
    with torch.no_grad(): head.out.weight.normal_(std=.05)
    x=torch.randn(2,1024,requires_grad=True); c=torch.randn(2,1024); raw=torch.randn(2,4250); s=torch.zeros(2,dtype=torch.long)
    zero=head(x,torch.zeros(2),c,raw,s,'raw_EEG'); assert torch.equal(zero,torch.zeros(2))
    assert torch.equal(torch.autograd.grad(zero.sum(),x)[0],torch.zeros_like(x))
    a=head(x,torch.ones(2)*.5,c,raw,s,'c_only'); altered=head(x,torch.ones(2)*.5,c,raw+100,s,'c_only'); assert torch.equal(a,altered)
    grad=torch.autograd.grad(head(x,torch.ones(2)*.5,c,raw,s,'raw_EEG').sum(),x)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    return dict(status='D13_GAUSSIAN_SUFFICIENCY_CORRELATED_NOISE_PRODUCT_AND_X_GRADIENT_CPU_CHECKS_PASSED',
        checks=8,maximum_Gaussian_score_and_velocity_error=max(errors),correlated_sufficient_condition_error=sufficient_error,
        t0_neutrality_is_structural_not_EEG_efficacy=True,parameters_per_arm=sum(p.numel() for p in head.parameters()))
