"""D25 independently switchable sampler. Existing ordinary code is never imported.

The pipeline and original scheduler.step remain intact. At the same native input
we observe a second, missing-raw-condition UNet prediction; its response direction
controls only the covariance of the scheduler's ancestral Gaussian draw.
"""
import hashlib,inspect
import brainhive_conditional_noise_geometry_20261028_v1 as V

def sample(pipe,prompts,tokens,reference_tokens,seed,torch,mode,capture_arrays=False,enabled=True):
    if not enabled:
        import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N
        image,proof=N.sample_observed(pipe,prompts,tokens,seed,torch)
        return image,proof|dict(sampler_enabled=False,primary_UNet_evaluations=4,reference_UNet_evaluations=0),[]
    assert mode in V.MODES and tokens.shape==reference_tokens.shape==(1,4,2048)
    assert type(pipe.scheduler).__name__=='EulerAncestralDiscreteScheduler' and pipe.scheduler.config.prediction_type=='epsilon'
    assert pipe.scheduler.config.timestep_spacing=='trailing'
    def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    module=inspect.getmodule(type(pipe.scheduler));original_randn=module.randn_tensor;original_forward=pipe.unet.forward;original_prepare=pipe.prepare_latents;original_step=pipe.scheduler.step
    rng=torch.Generator(device='cuda').manual_seed(seed);phase=[None];field=[None];first=[];initial=[];raw_noises=[];used_noises=[];records=[];arrays=[];handles=[]
    counts={'primary':0,'reference':0,'primary_K':0,'primary_V':0,'reference_K':0,'reference_V':0,'VAE':0}
    def prepare(*args,**kwargs):
        value=original_prepare(*args,**kwargs);assert value.shape==(1,4,64,64);initial.append(digest(value));return value
    def forward(latent,timestep,*args,**kwargs):
        before=digest(latent)
        if not first:first.append(before)
        assert kwargs['cross_attention_kwargs']['ip_hidden_states'].data_ptr()==tokens.data_ptr()
        phase[0]='primary';counts['primary']+=1;result=original_forward(latent,timestep,*args,**kwargs)
        conditional=result[0] if isinstance(result,tuple) else result.sample
        assert torch.isfinite(conditional).all() and digest(latent)==before
        refkwargs=dict(kwargs);refkwargs['cross_attention_kwargs']=dict(kwargs['cross_attention_kwargs'],ip_hidden_states=reference_tokens)
        state=rng.get_state();phase[0]='reference';counts['reference']+=1;reference=original_forward(latent,timestep,*args,**refkwargs)
        absent=reference[0] if isinstance(reference,tuple) else reference.sample
        assert torch.equal(rng.get_state(),state) and digest(latent)==before and torch.isfinite(absent).all()
        field[0]=conditional.float()-absent.float();phase[0]=None
        return result
    def key_hook(mod,args):
        assert phase[0] in ('primary','reference');expected=tokens if phase[0]=='primary' else reference_tokens
        assert args[0].shape==(1,4,2048) and args[0].data_ptr()==expected.data_ptr();counts[phase[0]+'_K']+=1
    def value_hook(mod,args):
        assert phase[0] in ('primary','reference');expected=tokens if phase[0]=='primary' else reference_tokens
        assert args[0].shape==(1,4,2048) and args[0].data_ptr()==expected.data_ptr();counts[phase[0]+'_V']+=1
    def vae_hook(mod,args,result):assert torch.isfinite(result).all();counts['VAE']+=1
    def noise(*args,**kwargs):
        z=original_randn(*args,**kwargs);assert field[0] is not None and z.shape==field[0].shape==(1,4,64,64)
        assert len(raw_noises)==counts['primary']-1
        used=V.transform_ancestral_noise(z,field[0],mode);raw_noises.append(dict(sha256=digest(z),shape=list(z.shape),dtype=str(z.dtype)));used_noises.append(dict(sha256=digest(used),shape=list(used.shape),dtype=str(used.dtype)))
        v,valid=V.unit_direction(field[0],mode);dot=(used.float()*v).sum(1);den=used.float().norm(dim=1).clamp_min(1e-20)
        idx=pipe.scheduler.step_index;sigma=pipe.scheduler.sigmas[idx];next_sigma=pipe.scheduler.sigmas[idx+1];sigma_up=(next_sigma**2*(sigma**2-next_sigma**2)/sigma**2).sqrt()
        records.append(dict(step=int(idx),sigma_up=float(sigma_up),field_sha256=digest(field[0]),unit_direction_sha256=digest(v),zero_direction_pixels=int((~valid).sum()),field_norm_mean=float(field[0].norm(dim=1).mean()),used_noise_norm_mean=float(used.float().norm(dim=1).mean()),relative_protected_dot_RMS=float((dot/den).square().mean().sqrt()),raw_and_used_bitwise_equal=torch.equal(z,used)))
        if capture_arrays:arrays.append(dict(raw_noise=z.detach().cpu(),field=field[0].detach().cpu(),used_noise=used.detach().cpu()))
        return used
    try:
        handles.append(pipe.vae.decoder.register_forward_hook(vae_hook))
        processors=[p for p in pipe.unet.attn_processors.values() if type(p).__name__=='IPAttnProcessor2_0'];assert len(processors)==70
        for processor in processors:handles.extend([processor.to_k_ip.register_forward_pre_hook(key_hook),processor.to_v_ip.register_forward_pre_hook(value_hook)])
        pipe.prepare_latents=prepare;pipe.unet.forward=forward;module.randn_tensor=noise
        image=pipe(height=512,width=512,generator=rng,prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='pil').images[0].convert('RGB')
    finally:
        pipe.prepare_latents=original_prepare;pipe.unet.forward=original_forward;module.randn_tensor=original_randn
        for handle in handles:handle.remove()
    assert pipe.scheduler.step==original_step and len(initial)==1 and len(raw_noises)==4
    assert counts==dict(primary=4,reference=4,primary_K=280,primary_V=280,reference_K=280,reference_V=280,VAE=1)
    assert image.size==(512,512)
    return image,dict(sampler_enabled=True,mode=mode,initial_noise_latents_sha256=initial[0],initial_UNet_input_sha256=first[0],ancestral_noise_observations=raw_noises,actual_used_noise_observations=used_noises,noise_geometry_steps=records,
        UNet_NFE=8,primary_UNet_evaluations=4,reference_UNet_evaluations=4,native_IP_K_calls=560,native_IP_V_calls=560,denoising_steps=4,VAE_decodes=1,extra_UNet_evaluations=4,scheduler_step_signature_unmodified=True,
        conditional_prediction_returned_unmodified=True,no_additional_random_draws=True,reference_UNet_did_not_advance_query_RNG=True,reference_condition='declared reference tokens; normal inference uses native A/P of missing raw1024-vector',sampling_mean_function_unmodified=True,normal_inference_target_access=False),arrays
