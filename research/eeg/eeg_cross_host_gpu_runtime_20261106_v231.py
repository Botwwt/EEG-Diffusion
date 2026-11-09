"""Unadmitted acceleration prototypes; never imported by sealed experiments.

These are classical chain-rule/VJP and request-local memoization optimizations,
not a verified third research contribution. Numerical trajectory equivalence and
end-to-end speed are explicitly unverified. Import/run Torch only on OLD server.
"""
import torch
from torch import nn
from torch.nn import functional as F


def factorized_margin_jacobian(raw, gallery, own_indices, gallery_groups,
                               parameters, *, chunk_size=8, guard=lambda: None):
    """Same within/global hard-negative margins, analytically factored pullback.

    Args are explicit to avoid hardcoded EEG hosts, concepts, or gallery sizes.
    `raw` is a differentiable [queries, semantic_dimension] encoder output;
    `gallery` is the SAME already normalized, frozen visual gallery as reference.
    Ties use torch.argmax's first-index rule, matching the serial implementation.
    Returns detached margins and a FLOAT64 Jacobian in within-then-global order.
    This changes floating-point evaluation order. Never silently replace a fit.
    """
    if raw.ndim != 2 or gallery.ndim != 2 or raw.shape[1] != gallery.shape[1]:
        raise ValueError('incompatible encoder/gallery shape')
    if not raw.requires_grad or gallery.requires_grad or not parameters:
        raise ValueError('need differentiable encoder and frozen visual gallery')
    if chunk_size < 1 or own_indices.shape != (len(raw),):
        raise ValueError('invalid chunk or identity shape')
    if gallery_groups.shape != (len(gallery),):
        raise ValueError('explicit gallery groups required')
    params = tuple(parameters)
    batch, dimension = raw.shape
    with torch.no_grad():
        norms = raw.norm(dim=-1, keepdim=True)
        if not bool(torch.isfinite(raw).all() and (norms > 1e-12).all()):
            raise ValueError('degenerate/nonfinite encoder output: use reference')
        yy = F.normalize(raw, dim=-1)
        scores = yy @ gallery.T
        ii = torch.arange(batch, device=raw.device)
        own = scores[ii, own_indices].clone()
        scores[ii, own_indices] = -torch.inf
        same = gallery_groups[own_indices, None] == gallery_groups[None, :]
        if not bool((same.sum(-1) >= 2).all()):
            raise ValueError('within-concept negatives absent')
        within = scores.masked_fill(~same, -torch.inf).argmax(-1)
        global_ = scores.argmax(-1)
        margins = torch.cat((own - scores[ii, within], own - scores[ii, global_]))
        rows = torch.arange(2 * batch, device=raw.device) % batch
        negatives = torch.cat((within, global_))
        difference = gallery[own_indices[rows]] - gallery[negatives]
        yrows = yy[rows]
        # d[(z/||z||).(v_pos-v_neg)]/dz. Hard-negative index is piecewise fixed.
        cotangents = (difference - yrows * (yrows * difference).sum(-1, keepdim=True)) / norms[rows]
    result = torch.empty((2 * batch, sum(p.numel() for p in params)),
                         dtype=torch.float64, device=raw.device)
    for lo in range(0, 2 * batch, chunk_size):
        guard()
        hi = min(lo + chunk_size, 2 * batch)
        seeds = raw.new_zeros((hi - lo, batch, dimension))
        seeds[torch.arange(hi - lo, device=raw.device), rows[lo:hi]] = cotangents[lo:hi]
        gradients = torch.autograd.grad(raw, params, grad_outputs=seeds,
                                       is_grads_batched=True, retain_graph=True)
        offset = 0
        for p, g in zip(params, gradients):
            result[lo:hi, offset:offset + p.numel()] = g.reshape(hi - lo, -1).double()
            offset += p.numel()
    return margins, result, {'within_negative': within, 'global_negative': global_,
                             'chunk_size': chunk_size, 'queries': batch}


def _tensor_key(x):
    if not isinstance(x, torch.Tensor):
        return None
    try:
        version = x._version
    except RuntimeError:  # inference tensors have no mutation counter: do not cache.
        return None
    return (x.data_ptr(), version, tuple(x.shape), tuple(x.stride()),
            x.storage_offset(), x.dtype, x.device)


class _MemoizedFrozenModule(nn.Module):
    def __init__(self, original, owner):
        super().__init__()
        self.original = original
        object.__setattr__(self, 'owner', owner)
        self.train(False)
        self.clear()

    def clear(self):
        self.key = self.input_reference = self.output = self.output_key = None

    def forward(self, *args, **kwargs):
        o = self.owner
        # No detached caching in training, even when only an upstream input trains.
        if (torch.is_grad_enabled() or self.original.training or kwargs or len(args) != 1
                or any(p.requires_grad for p in self.original.parameters())):
            o.bypasses += 1
            return self.original(*args, **kwargs)
        xkey = _tensor_key(args[0])
        state = tuple(_tensor_key(v) for v in (*self.original.parameters(), *self.original.buffers()))
        if xkey is None or any(v is None for v in state):
            o.bypasses += 1
            return self.original(*args, **kwargs)
        # Autocast state is part of the computation. No cross-query/timestep guesses.
        enabled = torch.is_autocast_enabled()
        dtype = torch.get_autocast_gpu_dtype() if enabled else None
        key = (xkey, state, enabled, dtype)
        if key == self.key and _tensor_key(self.output) == self.output_key:
            o.hits += 1
            return self.output
        o.misses += 1
        value = self.original(*args, **kwargs)
        value_key = _tensor_key(value)
        old_bytes = 0 if self.output is None else self.output.numel() * self.output.element_size()
        new_bytes = 0 if value_key is None else value.numel() * value.element_size()
        if value_key is not None and o.bytes - old_bytes + new_bytes <= o.maximum_bytes:
            o.bytes += new_bytes - old_bytes
            self.key, self.input_reference, self.output, self.output_key = key, args[0], value, value_key
        else:
            o.bytes -= old_bytes
            self.clear()
        return value


def ip_cache_targets(unet):
    """Support HIVE scalar K/V and ATM ModuleList K/V without altering attention."""
    targets = []
    for name, module in unet.named_modules():
        parts = name.split('.')
        if isinstance(module, nn.Linear) and any(p in ('to_k_ip', 'to_v_ip') for p in parts):
            targets.append(name)
        # ATM recomputes its deterministic image projection inside each UNet call.
        elif len(parts) >= 2 and parts[-2] == 'image_projection_layers' and parts[-1].isdigit():
            targets.append(name)
    # The projection is a separate branch, never an ancestor of K/V modules.
    if not targets or any(a != b and b.startswith(a + '.') for a in targets for b in targets):
        raise ValueError('unsupported/overlapping cache target structure')
    return targets


class FrozenConditionRequest:
    """ONE generation, no_grad only. Restores original module objects on exit.

    No hashes/labels/subject guesses identify inputs. Retained tensor references,
    storage+mutation versions, module versions and dtype identify actual calls.
    Untracked .data writes and concurrent mutation are outside this contract.
    Cache misses recompute normally; there is no approximate feature reuse.
    """
    def __init__(self, unet, maximum_bytes=8 << 20):
        self.unet, self.maximum_bytes = unet, maximum_bytes
        self.hits = self.misses = self.bypasses = self.bytes = 0
        self.replacements = []

    def __enter__(self):
        if self.replacements or self.unet.training:
            raise ValueError('one nonnested eval request required')
        names = ip_cache_targets(self.unet)
        try:
            for name in names:
                parent_name, _, leaf = name.rpartition('.')
                parent = self.unet.get_submodule(parent_name) if parent_name else self.unet
                original = parent._modules[leaf]
                wrapper = _MemoizedFrozenModule(original, self)
                parent._modules[leaf] = wrapper
                self.replacements.append((parent, leaf, original, wrapper))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for parent, leaf, original, wrapper in reversed(self.replacements):
            assert parent._modules[leaf] is wrapper
            parent._modules[leaf] = original
            wrapper.clear()
        self.replacements.clear()
        self.bytes = 0
        return False
