"""D12 full-rank conditional visual prior; no UNet training or best-of-N."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_visual_support_20261027_v1 as OLD

SCHEMA = 'BrainHIVE_full1024_conditional_flow_20261027_v1'
ARMS = ('conditional_flow', 'shuffled_EEG_flow')


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.subject = nn.Embedding(10, 16)
        self.context = nn.Sequential(nn.Linear(1040, 512), nn.SiLU(), nn.Linear(512, 512))
        self.state = nn.Linear(1024, 512)
        self.time = nn.Sequential(nn.Linear(32, 512), nn.SiLU(), nn.Linear(512, 512))
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 1024),
            nn.SiLU(), nn.Linear(1024, 512)) for _ in range(3)])
        self.out = nn.Linear(512, 1024)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer('frequencies', torch.exp(torch.linspace(0., math.log(16.), 16)))

    def condition(self, c, subject):
        # Fixed input coordinate convention, independent of visual targets.
        c = F.normalize(c.float(), dim=-1) * 32
        return self.context(torch.cat([c, self.subject(subject)], -1))

    def velocity(self, x, t, context):
        phase = t[:, None] * self.frequencies[None] * (2*math.pi)
        h = self.state(x) + context + self.time(torch.cat([phase.sin(), phase.cos()], -1))
        for block in self.blocks: h = h + block(h) / math.sqrt(3.)
        return self.out(h)

    def forward(self, x, t, c, subject): return self.velocity(x, t, self.condition(c, subject))


def project_F(x, mean, scale, weight, bias):
    u = x.float() * scale + mean
    return F.layer_norm(u, (1024,), weight, bias, eps=1e-6)


def query_noise(queries):
    values = []; seeds = []
    for q in queries:
        seed = int(hashlib.sha256(f'D12_conditional_noise|{q["image_seed"]}'.encode()).hexdigest()[:16],16) % (2**63-1)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        values.append(torch.randn(1024, generator=generator)); seeds.append(seed)
    return torch.stack(values), seeds


@torch.no_grad()
def sample(model, initial, c, subjects, mean, scale, weight, bias, steps=32):
    assert steps == 32
    x = initial.clone().float(); context = model.condition(c, subjects)
    for k in range(steps):
        t = torch.full((len(x),), k/steps, device=x.device)
        x = x + model.velocity(x, t, context).float()/steps
    raw = x*scale+mean
    return project_F(x, mean, scale, weight, bias), raw


def score(generated, targets, baseline, queries, gallery, stage):
    assert stage in (80,400) and generated.shape == (stage,2,768)
    original = {arm: U.S.metric_arrays(baseline[:stage,j],targets,queries,gallery) for j,arm in enumerate(U.ARMS)}
    candidate = {arm: U.S.metric_arrays(generated[:,j],targets,queries,gallery) for j,arm in enumerate(ARMS)}
    contrasts = {arm: U.S.compare(candidate[ARMS[0]], value, queries) for arm,value in original.items()}
    paired = U.S.compare(candidate[ARMS[0]],candidate[ARMS[1]],queries)
    decision = U.S.decision(contrasts['risk'],str(stage))
    if stage == 400 and decision['action'] == 'PROMISING_FOR_FULL_EVALUATION':
        if not all(U.S.decision(value,'400')['action'] == 'PROMISING_FOR_FULL_EVALUATION' for value in contrasts.values()):
            decision = decision | dict(action='STOP_INSTANCE',reason='NOT_PROMISING_AGAINST_ALL3_FROZEN_ARMS')
        elif paired['DINO2AFC']['difference'] <= 0 or paired['DINO2AFC']['descriptive_concept_bootstrap95'][0] <= 0:
            decision = decision | dict(action='STOP_INSTANCE',reason='NO_PAIRED_EEG_INCREMENT_WITH_IDENTICAL_CONDITION_NOISE')
    return dict(status='COMPLETE_D12_FIXED_FULL1024_FLOW_READOUT',stage=stage,queries=stage,concepts=stage//10,gallery_targets=400,
        means={arm:{key:float(x.mean()) for key,x in value.items()} for arm,value in (original|candidate).items()},
        flow_minus_each_baseline=contrasts,flow_minus_shuffled_EEG=paired,decision=decision,
        risk_primary_fixed_in_D09_Dfit=True,normal_inference_target_access=False,exclusions=0,
        conditional_prior_samples_per_query=1,small_flow_velocity_evaluations_per_query=32,UNet_evaluations_per_query=4,
        independent_confirmation=False,SOTA_established=False)


def self_test():
    torch.set_num_threads(2)
    q=[dict(image_seed=i+10) for i in range(4)]
    z,seeds=query_noise(q); z2,seeds2=query_noise(q)
    assert torch.equal(z,z2) and seeds==seeds2 and len(set(seeds))==4
    x=torch.randn(3,1024); mean=torch.randn(1024); weight=torch.rand(1024)+.5; bias=torch.randn(1024)
    out=project_F(x,mean,.75,weight,bias)
    u=(out-bias)/weight
    assert u.mean(-1).abs().max()<1e-6 and (u.square().mean(-1)-1).abs().max()<1e-5
    class Constant:
        def condition(self,c,s): return c
        def velocity(self,x,t,context): return context
    c=torch.randn_like(x)
    sampled,raw=sample(Constant(),x,c,torch.zeros(3,dtype=torch.long),mean,.75,weight,bias)
    assert torch.allclose(raw,(x+c)*.75+mean,atol=1e-5,rtol=1e-5)
    assert torch.allclose(sampled,project_F(x+c,mean,.75,weight,bias),atol=1e-5,rtol=1e-5)
    return dict(status='D12_CPU_FIXED_NOISE_FULLRANK_AFFINE_AND_EULER_CHECKS_PASSED',checks=4)
