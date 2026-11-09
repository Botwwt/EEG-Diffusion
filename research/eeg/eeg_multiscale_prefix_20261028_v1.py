"""Research prototype: persistent EEG memory in the author AVDE/VAR decoder.

Uses the pinned author's existing projections and transformer blocks. No new
learned layer is introduced here. This is an untrained architectural experiment,
not a claim of novelty or reconstruction benefit. Upstream provenance is in
reproduction_sources/direction_switch_20261028_v1/*/source_inventory.json.
"""
from contextlib import contextmanager

import torch
from torch import nn


class EEGPrefixDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    @staticmethod
    def centered_memory(global_eeg, patch_eeg):
        if patch_eeg.ndim != 3 or global_eeg.shape != (patch_eeg.shape[0], patch_eeg.shape[2]):
            raise ValueError('Expected global BxD and EEG-only patch memory BxNxD')
        return global_eeg[:, None] + patch_eeg - patch_eeg.mean(1, keepdim=True)

    def _attention_mask(self, count, end, device, dtype):
        image = self.decoder.attn_bias_for_masking[:, :, :end, :end].to(device=device, dtype=dtype)
        if not count:
            return image
        result = torch.full((1, 1, count+end, count+end), -torch.inf, device=device, dtype=dtype)
        # The EEG memory cannot read any image token, including during teacher
        # forcing. Otherwise an image could leak back through the memory bank.
        result[:, :, :count, :count] = 0
        result[:, :, count:, :count] = 0
        result[:, :, count:, count:] = image
        return result

    def _inputs(self, previous_image_features, global_eeg, memory):
        d = self.decoder
        B = global_eeg.shape[0]
        if global_eeg.ndim != 2 or global_eeg.shape[1] != d.embed_proj.in_features:
            raise ValueError('Invalid global EEG shape')
        if previous_image_features.shape != (B, d.L-d.first_l, d.Cvae):
            raise ValueError('Training requires the complete shifted image-token sequence')
        if memory is not None and (memory.ndim != 3 or memory.shape[0] != B or memory.shape[2] != global_eeg.shape[1]):
            raise ValueError('Invalid EEG memory shape')
        if d.prog_si != -1:
            raise ValueError('Progressive subscale training is not registered in this prototype')
        if self.training and d.cond_drop_rate:
            keep = torch.rand(B, 1, device=global_eeg.device) >= d.cond_drop_rate
            global_eeg = global_eeg * keep
            if memory is not None:
                memory = memory * keep[:, :, None]
        condition = d.embed_proj(global_eeg)
        first = condition[:, None].expand(-1, d.first_l, -1) + d.pos_start
        image = torch.cat((first, d.word_embed(previous_image_features.float())), dim=1)
        image = image + (d.lvl_embed(d.lvl_1L) + d.pos_1LC)
        if memory is None:
            return image, condition, 0
        return torch.cat((d.embed_proj(memory), image), dim=1), condition, memory.shape[1]

    def forward(self, previous_image_features, global_eeg, *, eeg_memory=None):
        """Teacher-forced TRAINING/scoring; image targets are not inference inputs."""
        d = self.decoder
        hidden, condition, n = self._inputs(previous_image_features, global_eeg, eeg_memory)
        # Multiplication follows the actual active autocast policy of the caller.
        dtype = (hidden.new_ones(2, 2) @ hidden.new_ones(2, 2)).dtype
        hidden = hidden.to(dtype)
        conditioning = d.shared_ada_lin(condition).to(dtype)
        mask = self._attention_mask(n, d.L, hidden.device, dtype)
        for block in d.blocks:
            hidden = block(hidden, conditioning, mask)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError('Fused residual layout needs a separately verified implementation')
        return d.get_logits(hidden[:, n:].float(), condition)

    @contextmanager
    def _cached_attention(self):
        for block in self.decoder.blocks:
            block.attn.kv_caching(True)
        try:
            yield
        finally:
            for block in self.decoder.blocks:
                block.attn.kv_caching(False)

    @torch.no_grad()
    def generate(self, global_eeg, *, eeg_memory=None, seed=1729, guidance=4.0, top_k=900):
        """Free generation from EEG only; no labels, gallery, image or VQ target API."""
        if self.training:
            raise RuntimeError('Call eval() before EEG-only generation')
        d = self.decoder
        B = global_eeg.shape[0]
        if global_eeg.ndim != 2 or global_eeg.shape[1] != d.embed_proj.in_features:
            raise ValueError('Invalid global EEG shape')
        if eeg_memory is not None and (eeg_memory.ndim != 3 or eeg_memory.shape[0] != B or eeg_memory.shape[2] != global_eeg.shape[1]):
            raise ValueError('Invalid EEG memory shape')
        condition = d.embed_proj(torch.cat((global_eeg, torch.zeros_like(global_eeg))))
        conditioning = d.shared_ada_lin(condition)
        position = d.lvl_embed(d.lvl_1L) + d.pos_1LC
        token = condition[:, None].expand(-1, d.first_l, -1) + d.pos_start + position[:, :d.first_l]
        accumulated = condition.new_zeros(B, d.Cvae, d.patch_nums[-1], d.patch_nums[-1])
        generator = torch.Generator(device=global_eeg.device).manual_seed(seed)
        end = 0
        with self._cached_attention():
            # Prime the cache with EEG-only keys/values once. Its hidden state
            # never reads ground-truth or generated image tokens.
            if eeg_memory is not None:
                prefix = d.embed_proj(torch.cat((eeg_memory, torch.zeros_like(eeg_memory))))
                for block in d.blocks:
                    prefix = block(prefix, conditioning, None)
            for scale, side in enumerate(d.patch_nums):
                hidden = token
                for block in d.blocks:
                    hidden = block(hidden, conditioning, None)
                logits = d.get_logits(hidden, condition)
                amount = guidance * scale / (len(d.patch_nums)-1)
                logits = (1+amount)*logits[:B] - amount*logits[B:]
                k = min(max(int(top_k), 0), logits.shape[-1])
                if k:
                    threshold = logits.topk(k, dim=-1).values[..., -1:]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                sampled = torch.multinomial(logits.softmax(-1).reshape(-1, logits.shape[-1]), 1, generator=generator).reshape(B, side*side)
                values = d.vae_quant_proxy[0].embedding(sampled).transpose(1, 2).reshape(B, d.Cvae, side, side)
                accumulated, following = d.vae_quant_proxy[0].get_next_autoregressive_input(scale, len(d.patch_nums), accumulated, values)
                end += side*side
                if scale+1 < len(d.patch_nums):
                    token = d.word_embed(following.flatten(2).transpose(1, 2))
                    token = (token + position[:, end:end+d.patch_nums[scale+1]**2]).repeat(2, 1, 1)
        return d.vae_proxy[0].fhat_to_img(accumulated).add(1).mul(.5)


def paired_scale_evidence(logits_correct, logits_control, targets, scale_sizes):
    """Diagnostic paired log-likelihood difference, not an information estimate.

Both calls must use the exact same image prefix. Only held-out concepts may
support an EEG information claim; teacher-forced gains still need real images.
"""
    import torch.nn.functional as F
    if logits_correct.shape != logits_control.shape or targets.shape != logits_correct.shape[:2]:
        raise ValueError('Paired logit/target shape mismatch')
    if sum(scale_sizes) != targets.shape[1]:
        raise ValueError('Scale coverage mismatch')
    right = F.log_softmax(logits_correct.float(), -1).gather(-1, targets[..., None]).squeeze(-1)
    control = F.log_softmax(logits_control.float(), -1).gather(-1, targets[..., None]).squeeze(-1)
    return torch.stack([part.mean(1) for part in torch.split(right-control, scale_sizes, dim=1)], dim=1)
