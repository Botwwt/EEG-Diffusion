"""Exact coarse-token marginal of the frozen causal prior, plus row-local Adam.

The first two complete scales are fixed prospectively (1+4 tokens). This is a
hypothesis motivated by the previous recipe's diagnostic, not an EEG bandwidth
theorem. Integrating all later normalized causal factors leaves this marginal.
"""
import torch
from EEG_native_conditional_variational_core_20261029_v1 import NativeConditionDecoder


class NativeMarginalDecoder(NativeConditionDecoder):
    def marginal_logits(self, previous_image_features, condition, tokens=5):
        d = self.decoder
        boundaries = torch.tensor([x*x for x in d.patch_nums]).cumsum(0).tolist()
        if tokens not in boundaries or self.training or d.training:
            raise ValueError('Frozen prior and a complete scale boundary required')
        if condition.ndim != 2 or condition.shape[1] != d.C:
            raise ValueError('Native BxD condition required')
        if previous_image_features.shape != (len(condition), tokens-d.first_l, d.Cvae):
            raise ValueError('Only the causal coarse prefix may enter this target')
        first = condition[:, None].expand(-1, d.first_l, -1) + d.pos_start
        hidden = torch.cat((first, d.word_embed(previous_image_features.float())), 1)
        hidden = hidden + (d.lvl_embed(d.lvl_1L)+d.pos_1LC)[:, :tokens]
        dtype = (hidden.new_ones(2, 2) @ hidden.new_ones(2, 2)).dtype
        hidden = hidden.to(dtype)
        conditioning = d.shared_ada_lin(condition).to(dtype)
        mask = d.attn_bias_for_masking[:, :, :tokens, :tokens].to(dtype)
        for block in d.blocks:
            hidden = block(hidden, conditioning, mask)
        return d.get_logits(hidden.float(), condition)


class RowGaussianAdam:
    """Only the visited image rows update, including their bias-correction clock.

    A dense Adam on an indexed global Parameter would keep moving unvisited
    rows with old momentum. This explicit optimizer avoids that coupling.
    """
    def __init__(self, images, width, device):
        self.value = torch.zeros(images, 2, width, device=device)
        self.first = torch.zeros_like(self.value)
        self.second = torch.zeros_like(self.value)
        self.count = torch.zeros(images, device=device, dtype=torch.long)

    def leaf(self, ids):
        if ids.ndim != 1 or len(ids.unique()) != len(ids):
            raise ValueError('Distinct one-dimensional image identities required')
        return self.value[ids].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def step(self, ids, leaf, lr=.03):
        grad = leaf.grad
        if grad is None or not torch.isfinite(grad).all():
            raise FloatingPointError('Finite posterior gradient required')
        # Clip independently per query: batch composition must not alter its
        # variational E step. The training loss is a mean, so undo its batch
        # factor before the per-image optimizer.
        grad = grad * len(ids)
        norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        grad = grad * (10./norm).clamp_max(1.)[:, None, None]
        self.count[ids] += 1
        m = .9*self.first[ids] + .1*grad
        v = .999*self.second[ids] + .001*grad.square()
        # Python/Torch Adam computes these scalar powers in double precision.
        count = self.count[ids].to(torch.float64)[:, None, None]
        bias1 = (1-.9**count).to(grad.dtype)
        bias2 = (1-.999**count).to(grad.dtype)
        self.value[ids] -= lr*(m/bias1)/((v/bias2).sqrt()+1e-8)
        self.first[ids] = m
        self.second[ids] = v

    def state_dict(self):
        return {key:getattr(self,key) for key in ('value','first','second','count')}

    def load_state_dict(self, state):
        if set(state) != {'value','first','second','count'}:
            raise ValueError('Complete row optimizer state required')
        for key, value in state.items():
            assert getattr(self,key).shape == value.shape
            getattr(self,key).copy_(value)


def posterior_parameters(leaf, center, scale):
    return center+scale*leaf[:,0], scale.log()+leaf[:,1].clamp(-4,4)
