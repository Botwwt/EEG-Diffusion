"""Generator-native conditional variational supervision, not a CLIP adapter.

Y is the complete discrete VQ sequence. The frozen native prior defines
p_G(Y|z); image-specific q_i(z) and EEG p_theta(z|E) minimize a single token
ELBO. Both terms are in nats, divided together by the token count.
"""
import torch
import torch.nn.functional as F
from torch import nn
from eeg_multiscale_prefix_20261028_v1 import EEGPrefixDecoder

class NativeIdentityCondition(nn.Identity):
    def __init__(self,width):super().__init__();self.in_features=self.out_features=width

class NativeConditionDecoder(EEGPrefixDecoder):
    def __init__(self,decoder,class_embedding):
        super().__init__(decoder)
        width=decoder.C
        if class_embedding.ndim!=2 or class_embedding.shape[1]!=width or len(class_embedding)<2:raise ValueError('Native pretrained class conditions required')
        # The introduced random EEG projection is removed; all transferred
        # pretrained VAR tensors keep their original values and coordinates.
        decoder.embed_proj=NativeIdentityCondition(width)
        self.register_buffer('native_class_embedding',class_embedding.detach().clone())
        decoder.cond_drop_rate=0.;self.eval().requires_grad_(False)
    def forward(self,previous_image_features,condition):
        if self.training or self.decoder.training:raise RuntimeError('The native visual prior must remain eval; gradients may flow through z')
        return super().forward(previous_image_features,condition,eeg_memory=None)
    @torch.no_grad()
    def generate(self,condition=None,*,seed=1729,guidance=0.,top_k=600):
        if self.training:raise RuntimeError('Frozen prior must remain eval')
        d=self.decoder
        if condition is None:condition=self.native_class_embedding[-1:]
        if condition.ndim!=2 or condition.shape[-1]!=d.C:raise ValueError('Expected native condition BxD')
        B=len(condition);cond=torch.cat((condition,self.native_class_embedding[-1:].expand(B,-1)))
        conditioning=d.shared_ada_lin(cond);position=d.lvl_embed(d.lvl_1L)+d.pos_1LC
        token=cond[:,None].expand(-1,d.first_l,-1)+d.pos_start+position[:,:d.first_l]
        accumulated=cond.new_zeros(B,d.Cvae,d.patch_nums[-1],d.patch_nums[-1]);rng=torch.Generator(device=condition.device).manual_seed(seed);end=0
        with self._cached_attention():
            for k,side in enumerate(d.patch_nums):
                hidden=token
                for block in d.blocks:hidden=block(hidden,conditioning,None)
                logits=d.get_logits(hidden,cond);amount=float(guidance)*k/(len(d.patch_nums)-1);logits=(1+amount)*logits[:B]-amount*logits[B:]
                count=min(max(int(top_k),0),logits.shape[-1])
                if count:logits=logits.masked_fill(logits<logits.topk(count,dim=-1).values[...,-1:],-torch.inf)
                idx=torch.multinomial(logits.softmax(-1).reshape(-1,logits.shape[-1]),1,generator=rng).reshape(B,side*side)
                values=d.vae_quant_proxy[0].embedding(idx).transpose(1,2).reshape(B,d.Cvae,side,side)
                accumulated,following=d.vae_quant_proxy[0].get_next_autoregressive_input(k,len(d.patch_nums),accumulated,values);end+=side*side
                if k+1<len(d.patch_nums):token=(d.word_embed(following.flatten(2).transpose(1,2))+position[:,end:end+d.patch_nums[k+1]**2]).repeat(2,1,1)
        return d.vae_proxy[0].fhat_to_img(accumulated).add(1).mul(.5)

def native_moments(class_embedding):
    values=class_embedding[:-1].detach();center=values.mean(0);scale=values.std(0,unbiased=False).clamp_min(1e-4)
    return center,scale

class EEGGaussianPrior(nn.Module):
    """Train the real EEG backbone and a Gaussian mean/log-scale output head."""
    def __init__(self,encoder,center,scale):
        super().__init__();self.encoder=encoder;self.width=len(center)
        if not isinstance(encoder.head,nn.Linear) or center.shape!=scale.shape or not (scale>0).all():raise ValueError('Audited EEG linear head and native coordinate moments required')
        head=nn.Linear(encoder.head.in_features,2*self.width);nn.init.zeros_(head.weight);nn.init.zeros_(head.bias);encoder.head=head
        if hasattr(encoder,'logit_scale'):encoder.logit_scale.requires_grad_(False)
        self.register_buffer('native_center',center.detach().clone());self.register_buffer('native_logscale',scale.detach().log().clone())
    def forward(self,eeg,**kwargs):
        raw=self.encoder(eeg,**kwargs)
        if isinstance(raw,tuple):raw=raw[0]
        if raw.shape!=(len(eeg),2*self.width):raise ValueError('Expected EEG Gaussian parameters')
        mean_delta,scale_delta=raw.split(self.width,dim=-1)
        return self.native_center+self.native_logscale.exp()*mean_delta,self.native_logscale+scale_delta.clamp(-4,4)

class ImageVariationalBank(nn.Module):
    """Training-only per-image posterior; absent from normal EEG inference."""
    def __init__(self,images,center,scale):
        super().__init__();self.mean_delta=nn.Parameter(torch.zeros(images,len(center),device=center.device));self.logscale_delta=nn.Parameter(torch.zeros_like(self.mean_delta))
        self.register_buffer('native_center',center.detach().clone());self.register_buffer('native_logscale',scale.detach().log().clone())
    def forward(self,indices):
        return self.native_center+self.native_logscale.exp()*self.mean_delta[indices],self.native_logscale+self.logscale_delta[indices].clamp(-4,4)

def diagonal_KL(q_mean,q_logscale,p_mean,p_logscale):
    if not(q_mean.shape==q_logscale.shape==p_mean.shape==p_logscale.shape) or q_mean.ndim!=2:raise ValueError('Matching BxD Gaussian parameters required')
    value=(p_logscale-q_logscale)+.5*((2*(q_logscale-p_logscale)).exp()+((q_mean-p_mean)*(-p_logscale).exp()).square()-1)
    return value.sum(-1)

def token_ELBO(logits,targets,q_mean,q_logscale,p_mean,p_logscale):
    if logits.ndim!=3 or targets.shape!=logits.shape[:2] or q_mean.shape[0]!=len(targets):raise ValueError('Full paired token sequence required')
    nll=F.cross_entropy(logits.float().transpose(1,2),targets,reduction='none').sum(-1)
    kl=diagonal_KL(q_mean,q_logscale,p_mean,p_logscale)
    loss=(nll+kl).mean()/targets.shape[1]
    return loss,dict(token_NLL_nats_per_query=nll.detach(),KL_nats_per_query=kl.detach(),normalization_tokens=targets.shape[1])

def reparameterize(mean,logscale,*,generator):
    return mean+logscale.exp()*torch.randn(mean.shape,device=mean.device,dtype=mean.dtype,generator=generator)
