"""Native CLIP-coordinate responses to internal ATMS temporal-feature block changes."""
import torch

SCHEMA='BrainHIVE_ATMS_native_projection_four_internal_block_responses_20261027_v1'


def native_block_responses(model,features):
    """Keep original head and its learned affine; no hidden-basis mixing across models.

    features is original PatchEmbedding output [N,36,40]. Each of four9-position
    groups is replaced by that query's feature mean across all36 positions.
    This is an internal-feature intervention, not physical EEG-time ablation.
    """
    assert features.ndim==3 and tuple(features.shape[1:])==(36,40)
    n=len(features);variants=features[:,None].expand(n,4,36,40).clone()
    reference=features.mean(1,keepdim=True)
    for group in range(4):variants[:,group,group*9:(group+1)*9]=reference
    base=model.proj_eeg(features.reshape(n,1440)).float()
    changed=model.proj_eeg(variants.reshape(n*4,1440)).reshape(n,4,1024).float()
    response=base[:,None]-changed
    assert torch.isfinite(base).all() and torch.isfinite(response).all()
    return base,response
