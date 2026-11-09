"""Verified native BrainHIVE generator, with gradients only through new inputs."""
import json
import logging
import os
from pathlib import Path

from brainhive_native_models_v1 import *


def load_native(training=False):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from diffusers import StableDiffusionXLPipeline
    class SelfAttentionNotice(logging.Filter):
        def filter(self,record):
            return record.getMessage()!="cross_attention_kwargs ['ip_hidden_states'] are not expected by AttnProcessor2_0 and will be ignored."
    logging.getLogger('diffusers.models.attention_processor').addFilter(SelfAttentionNotice())
    registration=json.loads((ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json').read_text())
    payload=next(x for x in registration['generator_assets'] if 'sdxl_turbo_fp16' in x['path'])
    assert sha(payload['path'])==payload['sha256']
    model=prior_models('cpu',include_attention=True)
    pipe=StableDiffusionXLPipeline.from_single_file(payload['path'],
        config=str(Path(registration['generator_asset_root'])/'sdxl_config'),
        torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    assert type(pipe.scheduler).__name__=='EulerAncestralDiscreteScheduler'
    assert pipe.scheduler.config.timestep_spacing=='trailing'
    assert pipe.scheduler.config.prediction_type=='epsilon'
    attention=model['attn_adapter'].to(dtype=torch.float16)
    assert tuple(pipe.unet.attn_processors)==tuple(attention.attn_processor_keys)
    attention.bind_unet(pipe.unet)
    assert sum(p.numel() for p in attention.parameters())==340787200
    for module in pipe.unet.modules():
        if type(module).__name__=='Attention':
            for key in ['norm_q','norm_k']:
                if not hasattr(module,key):setattr(module,key,None)
    pipe.to('cuda')
    for module in pipe.components.values():
        if isinstance(module,nn.Module):module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    adapter=model['fusion_encoder'].adapter.cuda().eval().requires_grad_(False)
    projector=model['proj'].cuda().eval().requires_grad_(False)
    dtype=torch.bfloat16 if training else torch.float16
    # Keep the original FP16 weight bytes. Training autocast may compute in
    # BF16, but FP16->BF16->FP16 would permanently round frozen host weights.
    adapter.to(dtype=torch.float16);projector.to(dtype=torch.float16)
    if training:
        assert float(pipe.unet.config.dropout)==0
        pipe.unet.train();pipe.unet.enable_gradient_checkpointing()
        pipe.vae.to(dtype=torch.float32);pipe.vae.enable_slicing()
    with torch.no_grad():
        prompt,_,pooled,_=pipe.encode_prompt('',device='cuda',do_classifier_free_guidance=False)
    # These fixed tensors must be ordinary no-grad tensors, not inference-mode
    # tensors, because frozen UNet layers may save them to backpropagate inputs.
    prompt=prompt.detach().clone().to(dtype);pooled=pooled.detach().clone().to(dtype)
    time_ids=pipe._get_add_time_ids((512,512),(0,0),(512,512),dtype=dtype,
        text_encoder_projection_dim=pooled.shape[-1]).cuda()
    receipt=dict(generator_sha256=payload['sha256'],generator_revision=payload['revision'],
        native_prior_revision='135a284109e71f5cde3d3e6d5dee6268a3c7987a',
        loader_source_sha256=sha(__file__),native_A_P_IP_frozen=True,
        frozen_UNet=True,forward_autocast_dtype=str(dtype),frozen_weights_dtype='torch.float16',inference_resolution=512,
        inference_steps=4,inference_CFG=0,inference_IP_scale=1)
    return pipe,adapter,projector,(prompt,pooled,time_ids),receipt


def denoise(pipe,prompts,noisy,timesteps,tokens):
    prompt,pooled,time_ids=prompts;n=len(noisy)
    assert tokens.shape==(n,4,2048)
    return pipe.unet(noisy.to(pipe.unet.dtype),timesteps,
        encoder_hidden_states=prompt.expand(n,-1,-1),
        added_cond_kwargs=dict(text_embeds=pooled.expand(n,-1),time_ids=time_ids.expand(n,-1)),
        cross_attention_kwargs={'ip_hidden_states':tokens},return_dict=False)[0]
