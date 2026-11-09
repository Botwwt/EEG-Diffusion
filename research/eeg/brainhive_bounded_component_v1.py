"""A bounded direction correction in the host's actual LayerNorm coordinates.

This is the first BrainHIVE component hypothesis, not a claimed general method.
Risk is predicted conditional error, not physiological EEG signal quality.
"""
import torch
from torch import nn
import torch.nn.functional as F


class BoundedConditionAdapter(nn.Module):
    def __init__(self,dimension=1024,hidden=256,predict_risk=False,radius=.25):
        super().__init__()
        assert dimension==1024 and hidden==256 and radius==.25
        self.radius=radius;self.predict_risk=predict_risk
        self.trunk=nn.Sequential(nn.Linear(dimension,hidden),nn.GELU())
        self.direction=nn.Linear(hidden,dimension)
        self.risk=nn.Linear(hidden,1) if predict_risk else None
        nn.init.zeros_(self.direction.weight);nn.init.zeros_(self.direction.bias)
        if self.risk is not None:
            nn.init.zeros_(self.risk.weight);nn.init.zeros_(self.risk.bias)

    def forward(self,condition,weight,bias,*,enabled=True,gate_mode='native',calibration=(1.,0.),risk_permutation=None):
        assert condition.ndim==2 and condition.shape[-1]==1024
        assert weight.shape==bias.shape==condition.shape
        assert torch.all(weight.abs()>1e-6)
        if not enabled:return condition,{}
        # Undo only this real host's learned output affine. No replacement of the
        # raw native condition by a unit vector is fed to A/P.
        u=(condition-bias)/weight
        h=self.trunk(u)
        d=self.direction(h);d=d-d.mean(-1,keepdim=True)
        base_norm=u.norm(dim=-1,keepdim=True)
        r2=base_norm.square()
        assert torch.all(r2>1e-8)
        d=d-(d*u).sum(-1,keepdim=True)/r2*u
        risk=2*torch.sigmoid(self.risk(h)).squeeze(-1) if self.risk is not None else None
        if gate_mode=='native':
            if risk is None:gate=torch.ones(len(condition),device=condition.device)
            else:
                calibrated=(calibration[0]*risk.detach()+calibration[1]).clamp(0,2)
                gate=1-calibrated/2
        elif gate_mode=='fixed':gate=torch.ones(len(condition),device=condition.device)
        elif gate_mode=='shuffled':
            assert risk is not None and risk_permutation is not None
            assert sorted(risk_permutation.tolist())==list(range(len(condition)))
            calibrated=(calibration[0]*risk.detach()[risk_permutation]+calibration[1]).clamp(0,2)
            gate=1-calibrated/2
        else:raise ValueError(gate_mode)
        delta=self.radius*r2.sqrt()*d/(r2+d.square().sum(-1,keepdim=True)).sqrt()*gate[:,None]
        scale=base_norm/(u+delta).norm(dim=-1,keepdim=True)
        # Written as an exact residual so zero-init and explicit disable preserve
        # the original float32 bytes instead of re-normalizing the parent path.
        correction=weight*((scale-1)*u+scale*delta)
        result=condition+correction
        assert torch.isfinite(result).all()
        return result,dict(risk=risk,gate=gate,correction=correction,
            standardized_delta_norm=delta.norm(dim=-1),standardized_input_norm=r2.sqrt().squeeze(-1))


def risk_target(base_tokens,real_tokens):
    return (1-F.cosine_similarity(base_tokens.float(),real_tokens.float(),dim=-1)).mean(-1).detach()
