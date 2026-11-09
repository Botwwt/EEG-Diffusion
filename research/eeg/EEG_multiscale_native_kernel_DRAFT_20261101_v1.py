"""DRAFT frozen native per-scale kernels with fresh current-class prefixes.

This changes no pretrained tensor and learns nothing. Normal generation accepts
sampled latent class indices, not target images, gallery labels or VQ answers.
Teacher scoring is a separate API. Full-size checkpoint/cost/effect admission is
absent; CPU random-model tests do not establish a real EEG method.
"""
import hashlib
import torch
import torch.nn.functional as F


def _no_cache(native):
    d=native.decoder
    if native.training or d.training or d.prog_si!=-1:raise ValueError('Frozen eval decoder without progressive training required')
    if any(block.attn.caching or block.attn.cached_k is not None or block.attn.cached_v is not None for block in d.blocks):
        raise RuntimeError('A class-dependent native prefix cache is present; do not reinterpret it under another class')


def stage_logits(native,previous_features,condition,stage):
    """All current-stage logits under one native fixed-class prefix kernel."""
    _no_cache(native);d=native.decoder
    if type(stage) is not int or not 0<=stage<len(d.patch_nums):raise ValueError('Registered complete scale required')
    begin,end=d.begin_ends[stage]
    if previous_features.shape!=(len(condition),end-d.first_l,d.Cvae):raise ValueError('Only complete causal prefix features required')
    value=native.marginal_logits(previous_features,condition,tokens=end)
    _no_cache(native)
    return value[:,begin:end]


def complete_scale_log_likelihoods(native,previous_features,true_codes,condition):
    """Teacher-only likelihoods: no token subset or mean-of-token probability."""
    _no_cache(native);d=native.decoder
    if true_codes.shape!=(len(condition),d.L) or true_codes.dtype!=torch.long:raise ValueError('Complete true VQ sequence required for scoring')
    if previous_features.shape!=(len(condition),d.L-d.first_l,d.Cvae):raise ValueError('Complete shifted teacher features required')
    if not ((true_codes>=0)&(true_codes<d.V)).all():raise ValueError('Invalid true VQ code')
    full=native(previous_features,condition)
    # Keep native FP32 log-softmax, accumulate all scale sums in FP64 so
    # regrouping680 token terms does not introduce avoidable FP32 drift.
    logprob=F.log_softmax(full.float(),dim=-1).gather(-1,true_codes[:,:,None]).squeeze(-1).double()
    value=torch.stack([logprob[:,lo:hi].sum(-1) for lo,hi in d.begin_ends],dim=1)
    if value.shape!=(len(condition),len(d.patch_nums)) or not torch.isfinite(value).all():raise FloatingPointError('Invalid per-scale native likelihood')
    _no_cache(native);return value


@torch.no_grad()
def draw_class_chain(logits,stay_probability,*,generator):
    """Private latent RNG; consume both stay and refresh draws at every stage."""
    if logits.ndim!=3 or not torch.isfinite(logits).all():raise ValueError('EEG latent logits must be BxSxK')
    B,S,K=logits.shape
    if S<1 or K<2 or stay_probability.shape!=(B,S-1):raise ValueError('Matching class-chain dimensions required')
    if not torch.isfinite(stay_probability).all() or not ((stay_probability>=0)&(stay_probability<=1)).all():raise ValueError('Invalid stay probabilities')
    if generator is None:raise ValueError('An explicit private latent generator is required')
    before=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()
    pi=logits.softmax(-1)
    initial=torch.rand((B,1),device=logits.device,dtype=logits.dtype,generator=generator)
    stays=torch.rand((B,S-1),device=logits.device,dtype=logits.dtype,generator=generator)
    refresh=torch.rand((B,S-1),device=logits.device,dtype=logits.dtype,generator=generator)
    initial_class=(pi[:,0].cumsum(-1)<initial).sum(-1).clamp_max(K-1)
    values=[initial_class]
    for k in range(1,S):
        next_class=(pi[:,k].cumsum(-1)<refresh[:,k-1:k]).sum(-1).clamp_max(K-1)
        values.append(torch.where(stays[:,k-1]<stay_probability[:,k-1],values[-1],next_class))
    indices=torch.stack(values,dim=1)
    return indices,dict(initial_uniform=initial,stay_uniform=stays,refresh_uniform=refresh,
        private_latent_rng_before=before,private_latent_rng_after=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
        uniform_draws_per_query=2*S-1,no_best_of_N=True)


@torch.no_grad()
def generate_class_chain(native,class_indices,*,image_seed,top_k=600,keep_tensor_trace=False):
    """One ancestral generation, recomputing the current-class causal prefix."""
    _no_cache(native);d=native.decoder
    if class_indices.ndim!=2 or class_indices.dtype!=torch.long or class_indices.shape[1]!=len(d.patch_nums):raise ValueError('One sampled class per scale required')
    B,S=class_indices.shape;K=len(native.native_class_embedding)-1
    if not ((class_indices>=0)&(class_indices<K)).all():raise ValueError('Native non-null latent class indices required')
    device=native.native_class_embedding.device
    if class_indices.device!=device:raise ValueError('Class indices and frozen native condition table must share device')
    image_rng=torch.Generator(device=device).manual_seed(image_seed)
    accumulated=native.native_class_embedding.new_zeros(B,d.Cvae,d.patch_nums[-1],d.patch_nums[-1])
    previous=[];records=[]
    try:
        for k,side in enumerate(d.patch_nums):
            condition=torch.cat((native.native_class_embedding[class_indices[:,k]],native.native_class_embedding[-1:].expand(B,-1)))
            features=torch.cat(previous,dim=1) if previous else accumulated.new_empty(B,0,d.Cvae)
            logits=stage_logits(native,features.repeat(2,1,1),condition,k)
            conditional=logits[:B]  # The registered candidate uses CFG0; no unregistered guidance interpolation.
            count=min(max(int(top_k),0),conditional.shape[-1])
            if count:conditional=conditional.masked_fill(conditional<conditional.topk(count,dim=-1).values[...,-1:],-torch.inf)
            codes=torch.multinomial(conditional.softmax(-1).reshape(-1,d.V),1,generator=image_rng).reshape(B,side*side)
            item=dict(scale=k,classes=class_indices[:,k].detach().cpu().tolist(),tokens=side*side,
                sampled_codes_SHA256=hashlib.sha256(codes.cpu().numpy().tobytes()).hexdigest(),
                logits_SHA256=hashlib.sha256(logits.contiguous().cpu().numpy().tobytes()).hexdigest(),
                prefix_tokens=sum(p*p for p in d.patch_nums[:k+1]),prefix_recomputed_under_current_class=True)
            if keep_tensor_trace:item.update(codes=codes.detach().clone(),logits=logits.detach().clone())
            records.append(item)
            embedding=d.vae_quant_proxy[0].embedding(codes).transpose(1,2).reshape(B,d.Cvae,side,side)
            accumulated,following=d.vae_quant_proxy[0].get_next_autoregressive_input(k,S,accumulated,embedding)
            if k+1<S:previous.append(following.flatten(2).transpose(1,2))
        image=d.vae_proxy[0].fhat_to_img(accumulated).add(1).mul(.5)
        _no_cache(native)
        return image,dict(scales=S,actual_generated_tokens=sum(x['tokens'] for x in records),records=records,
            one_image_per_query=True,CFG=0,full_prefix_transformer_calls=S,native_cached_sampling_used=False,
            normal_inference_target_access=False,extra_learned_parameters=0,
            image_rng_final_SHA256=hashlib.sha256(image_rng.get_state().cpu().numpy().tobytes()).hexdigest())
    finally:
        # This path never enables a cache. Fail closed if unrelated code mutated one.
        _no_cache(native)
