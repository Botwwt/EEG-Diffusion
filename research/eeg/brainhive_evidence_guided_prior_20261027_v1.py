"""D14 marginal F flow plus frozen D13 c evidence, stable Gaussian-path guidance."""
import math
import numpy as np
import torch
from torch import nn
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_conditional_flow_20261027_v1 as OLD

SCHEMA='BrainHIVE_F_marginal_prior_and_frozen_c_evidence_20261027_v1'
ARMS=('guided_c','guided_shuffled_c','marginal_prior')


class Marginal(nn.Module):
    def __init__(self):
        super().__init__(); self.state=nn.Linear(1024,512)
        self.time=nn.Sequential(nn.Linear(32,512),nn.SiLU(),nn.Linear(512,512))
        self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(512),nn.Linear(512,1024),nn.SiLU(),nn.Linear(1024,512)) for _ in range(3)])
        self.out=nn.Linear(512,1024); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer('frequencies',torch.exp(torch.linspace(0.,math.log(16.),16)))
    def forward(self,x,t):
        phase=t[:,None]*self.frequencies[None]*(2*math.pi)
        h=self.state(x)+self.time(torch.cat([phase.sin(),phase.cos()],-1))
        for block in self.blocks: h=h+block(h)/math.sqrt(3.)
        return self.out(h)


def unscaled_evidence(head,x,t,c,subjects):
    """Observe original h before r=t*h, without editing the frozen D13 forward."""
    seen=[]; handle=head.out.register_forward_hook(lambda module,args,out:seen.append(out))
    try: r=head(x,t,c,None,subjects,'c_only')
    finally: handle.remove()
    assert len(seen)==1
    h=seen[0].squeeze(-1)
    return h,r


def sample(prior,head,initial,c,subjects,mean,scale,weight,bias,steps=32):
    assert steps==32; x=initial.detach().clone().float(); traces=[]; states=[]; gradients=[]
    on_cuda=x.is_cuda
    for k in range(steps):
        t=torch.full((len(x),),k/steps,device=x.device)
        with torch.no_grad(),torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=on_cuda):
            velocity=prior(x,t).float()
        guide=torch.zeros_like(x)
        if head is not None:
            with torch.enable_grad(),torch.autocast(x.device.type,enabled=False):
                leaf=x.detach().requires_grad_(True); h,r=unscaled_evidence(head,leaf,t,c,subjects)
                guide=torch.autograd.grad(h.sum(),leaf)[0].detach()
            # Exact removal of t/t for this path: (1-t)/t * grad(r)=(1-t)*grad(h).
            # No division by zero, sigmoid gradient, guidance sweep or clipping.
            guide=guide*(1-k/steps)
            traces.append(torch.stack([r.detach(),h.detach(),guide.norm(dim=-1),velocity.norm(dim=-1)],-1).cpu().numpy())
        assert torch.isfinite(velocity).all() and torch.isfinite(guide).all()
        if k in (4,12,20,28):
            states.append(x.detach().cpu().numpy()); gradients.append(guide.cpu().numpy())
        x=(x+(velocity+guide)/steps).detach(); assert torch.isfinite(x).all()
    with torch.no_grad(): final=OLD.project_F(x,mean,scale,weight,bias)
    return final,x,dict(states=np.stack(states,1),guidance=np.stack(gradients,1),
        diagnostics=np.stack(traces,1) if traces else np.zeros((len(x),32,4),np.float32))


def score(generated,targets,baseline,queries,gallery,stage):
    assert stage in (80,400) and generated.shape==(stage,3,768)
    references={arm:U.S.metric_arrays(baseline[:stage,j],targets,queries,gallery) for j,arm in enumerate(U.ARMS)}
    candidates={arm:U.S.metric_arrays(generated[:,j],targets,queries,gallery) for j,arm in enumerate(ARMS)}
    contrasts={arm:U.S.compare(candidates['guided_c'],value,queries) for arm,value in references.items()}
    controls={arm:U.S.compare(candidates['guided_c'],candidates[arm],queries) for arm in ARMS[1:]}
    decision=U.S.decision(contrasts['risk'],str(stage))
    if stage==400 and decision['action']=='PROMISING_FOR_FULL_EVALUATION':
        if not all(U.S.decision(v,'400')['action']=='PROMISING_FOR_FULL_EVALUATION' for v in contrasts.values()):
            decision=decision|dict(action='STOP_INSTANCE',reason='NOT_PROMISING_AGAINST_ALL3_FIXED_ARMS')
        elif not all(v['DINO2AFC']['difference']>0 and v['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 for v in controls.values()):
            decision=decision|dict(action='STOP_INSTANCE',reason='NO_GUIDED_C_INCREMENT_OVER_BOTH_REGISTERED_CONTROLS')
    return dict(status='COMPLETE_D14_FIXED_MARGINAL_PRIOR_AND_C_EVIDENCE_READOUT',stage=stage,queries=stage,concepts=stage//10,gallery_targets=400,
        means={a:{k:float(x.mean()) for k,x in v.items()} for a,v in (references|candidates).items()},
        guided_minus_each_baseline=contrasts,guided_minus_registered_controls=controls,decision=decision,
        frozen_D13_c_only_head=True,raw_EEG_increment_claim=False,small_prior_velocity_evaluations_per_query=32,
        small_evidence_input_gradient_evaluations_per_guided_query=32,conditional_samples_per_query=1,UNet_evaluations_per_query=4,
        exact_posterior_established=False,normal_inference_target_access=False,independent_confirmation=False,SOTA_established=False)
