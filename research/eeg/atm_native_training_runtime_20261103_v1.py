"""Frozen native h94/Turbo runtime shared by actual-data profile and later fit."""
import hashlib
from pathlib import Path
import torch

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def checked(pin):
    assert sha(pin['path'])==pin['sha256'];return Path(pin['path'])

def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def load_host(policy, output):
    from diffusers import StableDiffusionXLPipeline,DDPMScheduler
    from transformers import CLIPImageProcessor
    from safetensors import safe_open
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    audit=ConvertedWeightAudit(torch,Path(output));audit.install()
    try:
        pipe=StableDiffusionXLPipeline.from_single_file(str(checked(policy['Turbo'])),config=policy['Turbo_config_root'],
            torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    finally:audit.restore()
    proof=audit.complete()
    pipe.register_modules(feature_extractor=CLIPImageProcessor())
    pipe.load_ip_adapter(str(checked(policy['h94']).parent),subfolder='',weight_name=Path(policy['h94']['path']).name,
        image_encoder_folder=None,torch_dtype=torch.float16,local_files_only=True)
    pipe.set_ip_adapter_scale(1.);pipe.to('cuda');pipe.set_progress_bar_config(disable=True)
    for component in pipe.components.values():
        if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
    assert type(pipe.scheduler).__name__=='EulerAncestralDiscreteScheduler' and pipe.scheduler.config.timestep_spacing=='trailing'
    projection=pipe.unet.encoder_hid_proj.image_projection_layers[0]
    mapping={'image_embeds.weight':'image_proj.proj.weight','image_embeds.bias':'image_proj.proj.bias',
             'norm.weight':'image_proj.norm.weight','norm.bias':'image_proj.norm.bias'}
    count=0
    with safe_open(policy['h94']['path'],framework='pt',device='cpu') as source:
        for name,value in projection.state_dict().items():assert tensor_sha(value)==tensor_sha(source.get_tensor(mapping[name]).half())
        index=1
        for name,processor in pipe.unet.attn_processors.items():
            if name.endswith('attn1.processor'):continue
            for kind in ('to_k_ip','to_v_ip'):
                assert tensor_sha(processor.state_dict()[kind+'.0.weight'])==tensor_sha(source.get_tensor(f'ip_adapter.{index}.{kind}.weight').half());count+=1
            index+=2
    assert count==140
    with torch.no_grad():prompt,_,pooled,_=pipe.encode_prompt('',device='cuda',do_classifier_free_guidance=False)
    prompt=prompt.detach().clone();pooled=pooled.detach().clone()
    timeids=pipe._get_add_time_ids((512,512),(0,0),(512,512),dtype=prompt.dtype,text_encoder_projection_dim=pooled.shape[-1]).cuda()
    # Match original ordinary: derive training betas from this actual Turbo host.
    diffusion=DDPMScheduler.from_config(pipe.scheduler.config)
    assert diffusion.config.prediction_type=='epsilon' and diffusion.config.num_train_timesteps==1000
    return pipe,projection,prompt,pooled,timeids,diffusion,proof

def ordinary_loss(pipe,projection,prompt,pooled,timeids,diffusion,changed,teacher,z0,noise,t):
    count=len(changed);z=diffusion.add_noise(z0,noise,t)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=pipe.unet(z.to(pipe.unet.dtype),t,encoder_hidden_states=prompt.expand(count,-1,-1),
            added_cond_kwargs={'text_embeds':pooled.expand(count,-1),'time_ids':timeids.expand(count,-1),
                               'image_embeds':[changed[:,None].half()]},return_dict=False)[0]
        tokens=projection(changed.half())
        with torch.no_grad():teacher_tokens=projection(teacher.half())
        epsilon=(prediction.float()-noise).square().mean()
        token=(1-torch.nn.functional.cosine_similarity(tokens.float(),teacher_tokens.float(),dim=-1)).mean()
        return epsilon+.1*token,epsilon,token
