"""D16: native encoder response view, cross-attention, bounded native-c correction."""
import math
import torch
from torch import nn
import torch.nn.functional as F
import brainhive_cross_encoder_20261027_v1 as OLD

SCHEMA='BrainHIVE_native_encoder_responses_bounded_cross_attention_20261027_v1'
ARMS=('c_only','final_ATMS','paired_responses','shuffled_responses')


def four_responses_from_native(model,features,native):
    assert tuple(features.shape[1:])==(36,40) and native.shape==(len(features),1024)
    n=len(features);variants=features[:,None].expand(n,4,36,40).clone()
    mean=features.mean(1,keepdim=True)
    for group in range(4):variants[:,group,9*group:9*(group+1)]=mean
    changed=model.proj_eeg(variants.reshape(4*n,1440)).reshape(n,4,1024).float()
    response=native.float()[:,None]-changed
    assert torch.isfinite(response).all();return response


def observe_responses(model,x,subject_ids):
    seen=[];handle=model.enc_eeg[0].register_forward_hook(lambda module,args,out:seen.append(out))
    try:native=model(x,subject_ids).float()
    finally:handle.remove()
    assert len(seen)==1
    return native,four_responses_from_native(model,seen[0],native)


def bounded_direction(u,d):
    # Same registered native-sphere radius and exact-residual geometry as ordinary.
    d=d.float();d=d-d.mean(-1,keepdim=True);base_norm=u.norm(dim=-1,keepdim=True);r2=base_norm.square()
    assert (r2>1e-8).all();d=d-(d*u).sum(-1,keepdim=True)/r2*u
    delta=.25*r2.sqrt()*d/(r2+d.square().sum(-1,keepdim=True)).sqrt()
    scale=base_norm/(u+delta).norm(dim=-1,keepdim=True)
    correction=(scale-1)*u+scale*delta
    return correction,dict(used_radius=(delta.norm(dim=-1)/(.25*r2.sqrt().squeeze(-1))).detach())


class ResponseAdapter(nn.Module):
    def __init__(self):
        super().__init__();self.query=nn.Linear(1024,128);self.token=nn.Linear(1024,128)
        self.position=nn.Parameter(torch.randn(5,128)*.01)
        self.attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.norm=nn.LayerNorm(128)
        self.block=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
        self.out=nn.Linear(128,1024);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,c_features,context,u):
        assert context.shape==(len(u),5,1024)
        q=self.query(c_features).unsqueeze(1);tokens=self.token(context)+self.position[None]
        # Five context tokens: explicit attention weights keep the deterministic
        # math path for both training gradients and independent CPU replay.
        h=self.norm(q+self.attention(q,tokens,tokens,need_weights=True)[0]).squeeze(1)
        h=h+self.block(h)/math.sqrt(2.)
        correction,stats=bounded_direction(u,self.out(h))
        return correction,stats


def contexts(c,a,responses,permutation):
    assert responses.shape==(len(c),4,1024)
    return dict(c_only=c[:,None].expand(-1,5,-1),final_ATMS=a[:,None].expand(-1,5,-1),
        paired_responses=torch.cat([a[:,None],responses],1),
        shuffled_responses=torch.cat([a[:,None],responses[permutation]],1))


def local_loss(prediction,target):
    assert len(prediction)%10==0 and target.shape==prediction.shape
    p=F.normalize(prediction.float(),dim=-1).reshape(-1,10,1024)
    y=F.normalize(target.float(),dim=-1).reshape(-1,10,1024)
    logits=torch.bmm(p,y.transpose(1,2))/.1
    labels=torch.arange(10,device=prediction.device).expand(len(p),-1)
    return F.cross_entropy(logits.reshape(-1,10),labels.reshape(-1))


def readout(values,linear_baselines,subjects,concepts):
    primary=values['paired_responses'];references={a:v for a,v in values.items() if a!='paired_responses'}|linear_baselines
    contrasts={a:OLD.paired_difference(v,primary,subjects,concepts) for a,v in references.items()}
    means={a:{k:float(v.mean()) for k,v in x.items()} for a,x in (values|linear_baselines).items()}
    supported=True
    for arm,effect in contrasts.items():
        supported&=effect['MSE']['difference']>=.005*means[arm]['MSE'] and effect['MSE']['descriptive_subject_concept_bootstrap95'][0]>0
        supported&=effect['within10_log_loss']['difference']>0 and effect['within10_log_loss']['descriptive_subject_concept_bootstrap95'][0]>0
        supported&=means['paired_responses']['within10_top1']>=means[arm]['within10_top1']
    return dict(status='D16_COMPLETE_ALL10_FULL_DFIT_RESPONSE_RESIDUAL_READOUT',means=means,reference_minus_paired_error_contrasts=contrasts,
        prediction_gate_passed=bool(supported),decision=dict(action='SUPPORTS_SEPARATELY_REGISTERED_NATIVE_IMAGE_TEST' if supported else 'STOP_FIXED_D16_INSTANCE',
            no_automatic_image_admission=True,no_hyperparameter_sweep=True),Dcheck_effects_read=False,actual_image_gain_established=False,
        independent_confirmation=False,SOTA_established=False)
