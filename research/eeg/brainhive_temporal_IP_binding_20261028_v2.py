"""Bind D37 residual attention to frozen native IP without changing its path.

The original processor still computes the whole native result. Temporary hooks
capture its real Q and image K; only an independently projected residual is added.
"""
import torch
from torch import nn
import torch.nn.functional as F
from brainhive_temporal_conditional_attention_20261028_v1 import attention_delta


class TemporalContext:
    def __init__(self):
        self.enabled=False
        self.detail=None
        self.mode='conditional'
        self.calls={}
        self.records={}

    def configure(self,detail,mode='conditional',enabled=True):
        assert mode in ['conditional','flat_residual']
        assert detail is None or detail.shape[1:]==(4,5,2048)
        self.detail=detail
        self.mode=mode
        self.enabled=enabled
        self.calls={}
        self.records={}


class TemporalIPProcessor(nn.Module):
    def __init__(self,original,context,name):
        super().__init__()
        assert type(original).__name__=='IPAttnProcessor2_0'
        self.original=original
        self.context=context
        self.name=name

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,
                 attention_mask=None,temb=None,ip_hidden_states=None,*args,**kwargs):
        ctx=self.context
        if not ctx.enabled:
            return self.original(attn,hidden_states,encoder_hidden_states,
                                 attention_mask,temb,ip_hidden_states,*args,**kwargs)
        assert ip_hidden_states is not None and ctx.detail is not None
        assert not attn.training and not self.original.training
        assert isinstance(attn.to_out[0],nn.Linear)
        assert isinstance(attn.to_out[1],nn.Dropout) and not attn.to_out[1].training
        captured={}
        def capture(name):
            def hook(module,inputs,output):
                assert name not in captured
                captured[name]=output
            return hook
        hooks=[attn.to_q.register_forward_hook(capture('q')),
               self.original.to_k_ip.register_forward_hook(capture('k'))]
        try:
            base=self.original(attn,hidden_states,encoder_hidden_states,
                               attention_mask,temb,ip_hidden_states,*args,**kwargs)
        finally:
            for hook in hooks:hook.remove()
        q=captured['q'];k=captured['k'];B,N,inner=q.shape;H=attn.heads;D=inner//H
        assert inner%H==0 and k.shape==(B,4,inner)
        assert ctx.detail.shape==(B,4,5,2048)
        q=q.reshape(B,N,H,D).transpose(1,2)
        if attn.norm_q is not None:q=attn.norm_q(q)
        k=k.reshape(B,4,H,D).transpose(1,2)
        detail=ctx.detail.to(device=k.device,dtype=ip_hidden_states.dtype)
        dk=self.original.to_k_ip(detail).reshape(B,4,5,H,D).permute(0,3,1,2,4)
        dv=self.original.to_v_ip(detail).reshape(B,4,5,H,D).permute(0,3,1,2,4)
        first=self.name not in ctx.records
        result=attention_delta(q,k,dk,dv,mode=ctx.mode,query_chunk=256,
                               return_probabilities=first)
        if first:
            delta,prob=result
            root=(q.float()@k.float().transpose(-1,-2)/(D**.5)).softmax(-1)
            ctx.records[self.name]=dict(heads=H,head_dimension=D,query_positions=N,
                query_dtype=str(q.dtype),delta_RMS=float(delta.float().square().mean().sqrt()),
                native_marginal_RMS_error=float((prob['joint'].sum(-1)-root).square().mean().sqrt()),
                native_marginal_max_error=float((prob['joint'].sum(-1)-root).abs().max()))
        else:delta=result
        delta=delta.transpose(1,2).reshape(B,N,inner)
        # Native projection bias is already in base. Do not add it twice.
        delta=F.linear(delta,attn.to_out[0].weight,bias=None)*self.original.scale
        delta=attn.to_out[1](delta)
        if hidden_states.ndim==4:
            delta=delta.transpose(-1,-2).reshape_as(hidden_states)
        delta=delta/attn.rescale_output_factor
        assert delta.shape==base.shape and torch.isfinite(delta).all()
        ctx.calls[self.name]=ctx.calls.get(self.name,0)+1
        return base+delta


def bind_temporal_IP(unet,context):
    originals=dict(unet.attn_processors)
    replacement={name:TemporalIPProcessor(value,context,name)
                 if type(value).__name__=='IPAttnProcessor2_0' else value
                 for name,value in originals.items()}
    assert sum(isinstance(value,TemporalIPProcessor) for value in replacement.values())==70
    before=sum(p.numel() for p in unet.parameters())
    unet.set_attn_processor(dict(replacement))
    assert sum(p.numel() for p in unet.parameters())==before
    unet.eval()
    return originals



def sample_temporal_observed(pipe,prompts,tokens,seed,torch):
    """Same native pipeline/RNG, distinguishing base from extra temporal K/V."""
    import hashlib
    import inspect
    initial=[];prepared=[];noise=[];counts=[0,0,0,0,0,0]
    processors=[p.original if isinstance(p,TemporalIPProcessor) else p
                for p in pipe.unet.attn_processors.values()]
    processors=[p for p in processors if type(p).__name__=='IPAttnProcessor2_0']
    assert len(processors)==70
    def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def unet_hook(module,args,result):
        counts[0]+=1
        if not initial:initial.append(digest(args[0]))
        assert torch.isfinite(result[0]).all()
    def projection_hook(base_slot,extra_slot):
        def hook(module,args):
            value=args[0]
            if value.shape==(1,4,2048):
                assert value.data_ptr()==tokens.data_ptr();counts[base_slot]+=1
            else:
                assert value.shape==(1,4,5,2048);counts[extra_slot]+=1
        return hook
    def vae_hook(module,args,result):assert torch.isfinite(result).all();counts[5]+=1
    handles=[pipe.unet.register_forward_hook(unet_hook),pipe.vae.decoder.register_forward_hook(vae_hook)]
    for processor in processors:
        handles.extend([processor.to_k_ip.register_forward_pre_hook(projection_hook(1,3)),
                        processor.to_v_ip.register_forward_pre_hook(projection_hook(2,4))])
    prepare=pipe.prepare_latents
    def observed_prepare(*args,**kwargs):
        value=prepare(*args,**kwargs);assert value.shape==(1,4,64,64) and torch.isfinite(value).all()
        prepared.append(digest(value));return value
    module=inspect.getmodule(type(pipe.scheduler));native=module.randn_tensor;original_step=pipe.scheduler.step
    def observed_noise(*args,**kwargs):
        value=native(*args,**kwargs);noise.append(dict(sha256=digest(value),shape=list(value.shape),dtype=str(value.dtype)));return value
    pipe.prepare_latents=observed_prepare;module.randn_tensor=observed_noise
    try:
        image=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),
            prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':tokens},
            num_inference_steps=4,guidance_scale=0.,output_type='pil').images[0].convert('RGB')
    finally:
        pipe.prepare_latents=prepare;module.randn_tensor=native
        for handle in handles:handle.remove()
    assert counts[:3]==[4,280,280] and counts[5]==1
    assert counts[3]==counts[4] and counts[3] in [0,280]
    assert len(prepared)==1 and len(noise)==4 and pipe.scheduler.step==original_step and image.size==(512,512)
    return image,dict(initial_UNet_input_sha256=initial[0],initial_noise_latents_sha256=prepared[0],
        UNet_NFE=counts[0],native_IP_K_calls=counts[1],native_IP_V_calls=counts[2],
        additional_temporal_K_calls=counts[3],additional_temporal_V_calls=counts[4],
        ancestral_noise_observations=noise,scheduler_step_signature_unmodified=True,extra_UNet_evaluations=0,
        noise_observation='actual native prepare_latents return, scheduler-scaled; no RNG replay proxy')
