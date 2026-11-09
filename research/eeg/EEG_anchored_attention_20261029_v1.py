"""Untrained anchored EEG attention using only existing AVDE parameters.

Native image attention keeps its own softmax normalization. A separate EEG
response subtracts the response of the same global-condition anchor before
the native output projection. There are no learned layers, gates or positions
owned by this wrapper. This is a hypothesis, not an efficacy/novelty claim.
"""
import torch
import torch.nn.functional as F
from torch import nn
from eeg_multiscale_prefix_20261028_v1 import EEGPrefixDecoder


class EEGAnchoredAttentionDecoder(EEGPrefixDecoder):
    def __init__(self, decoder):
        super().__init__(decoder)
        for block in decoder.blocks:
            if block.attn.attn_drop != 0 or not isinstance(block.attn.proj_drop, nn.Identity):
                raise ValueError('This version requires zero native attention/projection dropout')
            if block.fused_add_norm_fn is not None:
                raise ValueError('Fused residual layouts require a separate verified implementation')

    def _validate_memory(self, global_eeg, memory):
        if global_eeg.ndim != 2 or global_eeg.shape[1] != self.decoder.embed_proj.in_features:
            raise ValueError('Expected EEG-only global BxD condition')
        if memory is not None and (memory.ndim != 3 or memory.shape[0] != global_eeg.shape[0]
                or memory.shape[2] != global_eeg.shape[1] or memory.shape[1] < 1):
            raise ValueError('Expected EEG-only BxMxD memory, M>=1')

    @staticmethod
    def _anchor_only(global_eeg, memory):
        return memory is None or torch.equal(memory, global_eeg[:, None].expand_as(memory))

    def _memory_embeddings(self, global_eeg, memory, condition):
        # The algebraic residual form makes H=c exactly zero before LayerNorm.
        anchor = condition[:, None].expand(-1, memory.shape[1], -1)
        delta = F.linear(memory-global_eeg[:, None], self.decoder.embed_proj.weight, None)
        return anchor, anchor+delta

    @staticmethod
    def _native_condition_parts(block, conditioning):
        if block.shared_aln:
            return (block.ada_gss+conditioning).unbind(2)
        return block.ada_lin(conditioning).view(-1, 1, 6, block.C).unbind(2)

    def _prepare_layer(self, block, conditioning, embeddings):
        anchor, memory = embeddings
        _, _, scale1, _, shift1, _ = self._native_condition_parts(block, conditioning)
        # Same shape, same normalization and same conditional affine for both
        # paths. Raw centered EEG does not remain centered after this transform.
        u0 = block.ln_wo_grad(anchor).mul(scale1.add(1)).add(shift1)
        ue = block.ln_wo_grad(memory).mul(scale1.add(1)).add(shift1)
        du = ue-u0
        attention = block.attn
        width = block.C
        key_weight = attention.mat_qkv.weight[width:2*width]
        value_weight = attention.mat_qkv.weight[2*width:]
        key_delta = F.linear(du, key_weight, None)
        value_delta = F.linear(du, value_weight, None)
        B, M, _ = key_delta.shape
        # Key normalization in the native implementation is per HEAD, not over
        # the entire embedding. The correction below uses that exact geometry.
        if attention.attn_l2_norm:
            key_anchor = F.linear(u0, key_weight, attention.zero_k_bias).view(B, M, attention.num_heads, attention.head_dim)
            raw_delta = F.linear(du, key_weight, None).view_as(key_anchor)
            key_delta = F.normalize(key_anchor+raw_delta, dim=-1)-F.normalize(key_anchor, dim=-1)
        else:
            key_delta = key_delta.view(B, M, attention.num_heads, attention.head_dim)
        value_delta = value_delta.view(B, M, attention.num_heads, attention.head_dim)
        return key_delta.transpose(1, 2), value_delta.transpose(1, 2)

    @staticmethod
    def _innovation(image_input, attention, prepared):
        B, L, C = image_input.shape
        # Recompute the exact native query projection, including its bias. This
        # costs computation but introduces no parameter or native-cache change.
        qkv = F.linear(image_input, attention.mat_qkv.weight,
            torch.cat((attention.q_bias, attention.zero_k_bias, attention.v_bias)))
        query = qkv.view(B, L, 3, attention.num_heads, attention.head_dim)[:, :, 0].transpose(1, 2)
        if attention.attn_l2_norm:
            multiplier = attention.scale_mul_1H11.clamp_max(attention.max_scale_mul).exp()
            query = F.normalize(query, dim=-1).mul(multiplier)
        key_delta, value_delta = prepared
        # A common key shift cancels in softmax, including after native key
        # normalization. A common value cancels in the anchored difference.
        innovation = F.scaled_dot_product_attention(query, key_delta, value_delta,
            dropout_p=0., scale=attention.scale).transpose(1, 2).reshape(B, L, C)
        return F.linear(innovation, attention.proj.weight, None)

    def _block(self, block, hidden, conditioning, image_mask, prepared):
        if prepared is None:
            return block(hidden, conditioning, image_mask)
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self._native_condition_parts(block, conditioning)
        image_input = block.ln_wo_grad(hidden).mul(scale1.add(1)).add_(shift1)
        native_image_response = block.attn(image_input, attn_bias=image_mask)
        innovation = self._innovation(image_input, block.attn, prepared)
        hidden = hidden+block.drop_path((native_image_response+innovation).mul_(gamma1))
        hidden = hidden+block.drop_path(block.ffn(block.ln_wo_grad(hidden).mul(scale2.add(1)).add_(shift2)).mul(gamma2))
        return hidden

    def forward(self, previous_image_features, global_eeg, *, eeg_memory=None):
        self._validate_memory(global_eeg, eeg_memory)
        if self._anchor_only(global_eeg, eeg_memory):
            # Explicit bypass preserves native arithmetic and random draws.
            return super().forward(previous_image_features, global_eeg, eeg_memory=None)
        d = self.decoder
        if previous_image_features.shape != (global_eeg.shape[0], d.L-d.first_l, d.Cvae) or d.prog_si != -1:
            raise ValueError('Only complete shifted native scale sequences are supported')
        if self.training and d.cond_drop_rate:
            keep = torch.rand(global_eeg.shape[0], 1, device=global_eeg.device) >= d.cond_drop_rate
            global_eeg = global_eeg*keep
            eeg_memory = eeg_memory*keep[:, :, None]
        condition = d.embed_proj(global_eeg)
        first = condition[:, None].expand(-1, d.first_l, -1)+d.pos_start
        hidden = torch.cat((first, d.word_embed(previous_image_features.float())), 1)
        hidden = hidden+(d.lvl_embed(d.lvl_1L)+d.pos_1LC)
        dtype = (hidden.new_ones(2, 2)@hidden.new_ones(2, 2)).dtype
        hidden = hidden.to(dtype);conditioning = d.shared_ada_lin(condition).to(dtype)
        embeddings = tuple(x.to(dtype) for x in self._memory_embeddings(global_eeg, eeg_memory, condition))
        image_mask = d.attn_bias_for_masking.to(device=hidden.device, dtype=dtype)
        for block in d.blocks:
            hidden = self._block(block, hidden, conditioning, image_mask,
                self._prepare_layer(block, conditioning, embeddings))
        return d.get_logits(hidden.float(), condition)

    @torch.no_grad()
    def generate(self, global_eeg, *, eeg_memory=None, seed=1729, guidance=4., top_k=900):
        if self.training:
            raise RuntimeError('Call eval() before EEG-only generation')
        self._validate_memory(global_eeg, eeg_memory)
        if self._anchor_only(global_eeg, eeg_memory):
            return super().generate(global_eeg, eeg_memory=None, seed=seed, guidance=guidance, top_k=top_k)
        d = self.decoder;B = global_eeg.shape[0]
        global_pair = torch.cat((global_eeg, torch.zeros_like(global_eeg)))
        memory_pair = torch.cat((eeg_memory, torch.zeros_like(eeg_memory)))
        condition = d.embed_proj(global_pair);conditioning = d.shared_ada_lin(condition)
        embeddings = self._memory_embeddings(global_pair, memory_pair, condition)
        prepared = [self._prepare_layer(block, conditioning, embeddings) for block in d.blocks]
        position = d.lvl_embed(d.lvl_1L)+d.pos_1LC
        token = condition[:, None].expand(-1, d.first_l, -1)+d.pos_start+position[:, :d.first_l]
        accumulated = condition.new_zeros(B, d.Cvae, d.patch_nums[-1], d.patch_nums[-1])
        generator = torch.Generator(device=global_eeg.device).manual_seed(seed);end = 0
        # The native cache contains IMAGE keys/values only; EEG K/V are static
        # per layer in the local prepared list and never read any image token.
        with self._cached_attention():
            for scale, side in enumerate(d.patch_nums):
                hidden = token
                for block, memory_kv in zip(d.blocks, prepared):
                    hidden = self._block(block, hidden, conditioning, None, memory_kv)
                logits = d.get_logits(hidden, condition);amount = guidance*scale/(len(d.patch_nums)-1)
                logits = (1+amount)*logits[:B]-amount*logits[B:]
                k = min(max(int(top_k), 0), logits.shape[-1])
                if k:
                    logits = logits.masked_fill(logits < logits.topk(k, dim=-1).values[..., -1:], -torch.inf)
                indices = torch.multinomial(logits.softmax(-1).reshape(-1, logits.shape[-1]), 1, generator=generator).reshape(B, side*side)
                values = d.vae_quant_proxy[0].embedding(indices).transpose(1, 2).reshape(B, d.Cvae, side, side)
                accumulated, following = d.vae_quant_proxy[0].get_next_autoregressive_input(scale, len(d.patch_nums), accumulated, values)
                end += side*side
                if scale+1 < len(d.patch_nums):
                    token = d.word_embed(following.flatten(2).transpose(1, 2))
                    token = (token+position[:, end:end+d.patch_nums[scale+1]**2]).repeat(2, 1, 1)
        return d.vae_proxy[0].fhat_to_img(accumulated).add(1).mul(.5)
