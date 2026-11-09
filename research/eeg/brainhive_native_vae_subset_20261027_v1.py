"""Byte-bound native VAE-only loader, no UNet/text/prior model allocation."""
from types import SimpleNamespace
import torch
import brainhive_gain_calibration_20261027_v1 as U

def state_sha(state):
    import hashlib
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load(completion,device='cuda'):
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from safetensors.torch import load_file
    saved=U.read(completion);assert saved['status']=='NATIVE_VAE_SUBSET_EXACT_ORIGINAL_WEIGHTS_AND_EXISTING_IMAGE_DOMAIN_BANK_BOUND'
    config=U.read(U.checked(saved['VAE_config']));state=load_file(str(U.checked(saved['VAE_weights'])),device='cpu')
    assert state_sha(state)==saved['original_FP16_VAE_state_sha256']
    vae=AutoencoderKL.from_config(config).eval().requires_grad_(False);vae.load_state_dict(state,strict=True)
    assert sum(p.numel() for p in vae.parameters())==83653863
    assert state_sha({key:value.to(torch.float16) for key,value in vae.state_dict().items()})==saved['original_FP16_VAE_state_sha256']
    del state;vae.to(device=device,dtype=torch.float32);vae.enable_slicing()
    assert 2**(len(config['block_out_channels'])-1)==8
    pipe=SimpleNamespace(vae=vae,image_processor=VaeImageProcessor(vae_scale_factor=8))
    return pipe,dict(native_VAE_weights_exact=True,original_FP16_VAE_state_sha256=saved['original_FP16_VAE_state_sha256'],
        parameters=83653863,no_UNet_text_or_prior_model_allocated=True,FP32_widening=True,VAE_slicing=True)
