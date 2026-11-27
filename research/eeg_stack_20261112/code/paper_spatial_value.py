"""Separate coarse spatial bias from token-guided variation in image attention."""

from functools import partial
from math import isqrt

from torch.nn import functional as F


def project_spatial_band(features, band):
    """Keep the mean or residual of each disjoint 2 by 2 attention cell."""
    batch, locations, channels = features.shape
    side = isqrt(locations)
    if side * side != locations or side % 2:
        raise ValueError("Image attention must have an even square grid")
    cells = features.reshape(batch, side // 2, 2, side // 2, 2, channels)
    mean = cells.mean((2, 4), keepdim=True)
    if band == "coarse":
        return mean.expand_as(cells).reshape_as(features)
    if band == "detail":
        return (cells - mean).reshape_as(features)
    raise ValueError(f"Unexpected attention band: {band}")


class SpatialImageValue:
    """Add one projection of A(V_new − V_native) before the native output map."""

    def __init__(self, attention, value_layer):
        processor = attention.processor
        self.reference = None
        self.band = None
        self.heads = attention.heads
        self.query_norm = attention.norm_q
        self.scale = processor.scale
        self.value_layer = value_layer
        self.activations = {}
        attention.to_q.register_forward_hook(partial(self._capture, "query"))
        processor.to_k_ip.register_forward_hook(partial(self._capture, "key"))
        processor.to_v_ip.register_forward_hook(partial(self._capture, "value"))
        attention.to_out[0].register_forward_pre_hook(self._add_image_residual)

    def _capture(self, name, module, inputs, output):
        if self.reference is not None:
            self.activations[name] = output

    def _heads(self, values):
        return values.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def _add_image_residual(self, module, inputs):
        if self.reference is None:
            return None
        query, key, original = (self.activations.pop(name) for name in ("query", "key", "value"))
        replacement = F.linear(self.reference, self.value_layer.weight, self.value_layer.bias)
        if replacement.shape != original.shape:
            raise ValueError("Value replacement must preserve the native token axes")
        query = self._heads(query)
        if self.query_norm is not None:
            query = self.query_norm(query)
        delta = F.scaled_dot_product_attention(query, self._heads(key), self._heads(replacement - original))
        output = inputs[0]
        delta = delta.transpose(1, 2).flatten(2).float()
        residual = project_spatial_band(delta, self.band).to(output.dtype)
        return (output + self.scale * residual,)
