"""Same-noise image-effect objective; classical contrastive/reward tools.

This module supplies a NEW engineering path, not permission to resume the old
terminal, formal-D, latent-teacher or multi-seed jobs. No image-gain or firstness
claim follows from its algebra. Import after selecting the pinned runtime.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def paired_objective(features, targets, *, temperature=.07, effect_enabled=True):
    assert features.shape==targets.shape and features.ndim==2 and len(features)==2
    assert math.isfinite(temperature) and temperature>0
    g=F.normalize(features.float(),dim=-1)
    v=F.normalize(targets.detach().float(),dim=-1)
    absolute=1-(g*v).sum(dim=-1).mean()
    effect=((g[0]-g[1])*(v[0]-v[1])).sum()
    pairing=temperature*F.softplus(-effect/temperature)
    return absolute+(pairing if effect_enabled else pairing*0),dict(
        absolute=absolute,pairing=pairing,effect=effect,features=g,targets=v)


def native_four_step_latent(pipe, prompts, tokens, seed, *, gradient=False):
    """Use the actual pipeline body and scheduler, not an Euler surrogate.

    Checkpoint only UNet forward. Ancestral random draws occur outside the
    checkpoint; replaying backward must not draw new scheduler noise.
    """
    original=pipe.unet.forward
    observations=[]
    import hashlib,inspect
    module=inspect.getmodule(type(pipe.scheduler))
    randn=module.randn_tensor
    def observed_randn(*args,**kwargs):
        value=randn(*args,**kwargs)
        observations.append(hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        return value
    def checkpointed(*args,**kwargs):
        return checkpoint(original,*args,use_reentrant=False,**kwargs)
    module.randn_tensor=observed_randn
    if gradient:pipe.unet.forward=checkpointed
    kwargs=dict(height=512,width=512,
        generator=torch.Generator(device='cuda').manual_seed(seed),
        prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],
        cross_attention_kwargs={'ip_hidden_states':tokens},
        num_inference_steps=4,guidance_scale=0.,output_type='latent')
    try:
        if gradient:result=type(pipe).__call__.__wrapped__(pipe,**kwargs).images
        else:
            with torch.no_grad():result=pipe(**kwargs).images
    finally:
        pipe.unet.forward=original
        module.randn_tensor=randn
    assert result.shape==(1,4,64,64) and len(observations)==4
    assert [float(t) for t in pipe.scheduler.timesteps]==[999.,749.,499.,249.]
    return result,observations


def CLIP_pixels(image):
    image=F.interpolate(image,size=(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    mean=image.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
    std=image.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
    return (image-mean)/std


def image_feature(pipe,critic,latent,*,gradient):
    def decode(value):return pipe.vae.decode(value.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
    image=checkpoint(decode,latent,use_reentrant=False) if gradient else decode(latent)
    pixels=CLIP_pixels((image*.5+.5).clamp(0,1))
    features=checkpoint(critic,pixels,use_reentrant=False) if gradient else critic(pixels)
    return F.normalize(features.float(),dim=-1)
