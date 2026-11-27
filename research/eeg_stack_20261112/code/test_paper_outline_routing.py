from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from paper_outline_routing import OutlineRouter, outline_latents, spatial_delta


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.norm_q = None
        self.to_q = nn.Linear(6, 6, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(6, 6)])
        self.processor = nn.Module()
        self.processor.to_k_ip = nn.Linear(6, 6, bias=False)
        self.processor.to_v_ip = nn.Linear(6, 6, bias=False)
        self.processor.scale = .8

    def forward(self, q, tokens):
        projected = self.to_q(q)
        k = self.processor.to_k_ip(tokens)
        v = self.processor.to_v_ip(tokens)
        shape = lambda x: x.reshape(len(x), -1, 2, 3).transpose(1, 2)
        relative = F.scaled_dot_product_attention(shape(projected), shape(k), shape(v))
        relative = relative.transpose(1, 2).reshape_as(projected)
        return self.to_out[0](projected + self.processor.scale * relative)


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FakeAttention()

    def forward(self, q, token):
        return self.attention(q, token)


def example():
    torch.manual_seed(71)
    model = FakeUNet().eval()
    q, tokens = torch.randn(2, 16, 6), torch.randn(2, 4, 6)
    shape = torch.randn(2, 1, 8, 8).sigmoid()
    return model, q, tokens, shape


def test_gate_is_spatially_centered_and_constant_mask_is_identity():
    _, _, _, shape = example()
    delta = spatial_delta(shape, 16, .9)
    assert delta.shape == (2, 16, 1)
    torch.testing.assert_close(delta.mean(1), torch.zeros(2, 1), atol=1e-7, rtol=0)
    assert torch.count_nonzero(spatial_delta(torch.full_like(shape, .37), 16, .9)) == 0


def test_router_updates_only_relative_value_in_first_two_of_four_steps():
    model, q, tokens, field = example()
    native = model(q, tokens)
    layer = model.attention
    v = layer.processor.to_v_ip(tokens)
    k = layer.processor.to_k_ip(tokens)
    query = layer.to_q(q)
    shape = lambda x: x.reshape(len(x), -1, 2, 3).transpose(1, 2)
    ip = F.scaled_dot_product_attention(shape(query), shape(k), shape(v)).transpose(1, 2).reshape_as(query)
    update = layer.processor.scale * spatial_delta(field, 16, .9) * (ip - v.mean(1, keepdim=True))
    expected = native + F.linear(update, layer.to_out[0].weight)
    with OutlineRouter(model) as router:
        with router.batch(field):
            first, second, third, fourth = [model(q, tokens) for _ in range(4)]
        with router.batch(torch.ones_like(field)):
            flat = [model(q, tokens) for _ in range(4)]
        torch.testing.assert_close(first, expected)
        assert not torch.equal(first, native) and torch.equal(first, second)
        assert torch.equal(third, native) and torch.equal(fourth, native)
        assert all(torch.equal(image, native) for image in flat)
    assert torch.equal(model(q, tokens), native)


def test_failed_four_step_contract_resets_router_for_the_next_batch():
    model, q, tokens, field = example()
    with OutlineRouter(model) as router:
        with pytest.raises(ValueError, match="four"):
            with router.batch(field):
                model(q, tokens)
        with router.batch(field):
            for _ in range(4):
                result = model(q, tokens)
        assert torch.equal(result, model(q, tokens))

def test_stencil_latent_keeps_location_and_removes_channel_offsets():
    class Vae:
        config = SimpleNamespace(scaling_factor=.5)

        def encode(self, pixels):
            assert pixels.min() >= -.601 and pixels.max() <= .601
            compressed = F.avg_pool2d(pixels[:, :1], 8).repeat(1, 4, 1, 1) + .24
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: compressed))

    mask = torch.zeros(5, 1, 20, 20)
    mask[:, :, :, 10:] = 1.
    out = outline_latents(mask, Vae())
    assert out.shape == (5, 4, 64, 64)
    torch.testing.assert_close(out.mean((2, 3)), torch.zeros(5, 4), atol=1e-7, rtol=0)
    assert (out[:, :, :, 34:] > 0).all() and (out[:, :, :, :30] < 0).all()
