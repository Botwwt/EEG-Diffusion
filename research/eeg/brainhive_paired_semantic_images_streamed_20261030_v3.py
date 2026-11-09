"""Equivalent per-image VAE/critic checkpointing under the same memory cap."""
import torch
from brainhive_paired_semantic_effect_20261030_v1 import image_feature

def image_features(context,latent,*,gradient):
    # VAE slicing inside one large checkpoint retains both decode paths during
    # recomputation. Separate checkpoints bound recomputation to one image.
    assert latent.ndim==4 and len(latent)==2
    return torch.cat([image_feature(context['pipe'],context['critic'],part,gradient=gradient) for part in latent.split(1)],dim=0)
