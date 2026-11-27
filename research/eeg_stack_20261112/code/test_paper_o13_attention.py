from types import SimpleNamespace

import torch

from paper_o13_attention import LateAttentionUNet, O13ImageAttention, compose_features, token_midpoint
from test_paper_spatial_value import SmallImageAttention


def test_equal_cost_response_mix_and_token_mix_match_their_definitions():
    torch.manual_seed(113)
    attention = SmallImageAttention().double().eval()
    states = torch.randn(2, 16, 6, dtype=torch.float64)
    base, other = torch.randn(2, 2, 4, 6, dtype=torch.float64)
    mixed = token_midpoint(base, other)
    native = attention(states, base)
    alternate = attention(states, other)
    token_control = attention(states, mixed)

    route = O13ImageAttention(attention)
    route.reference, route.weight = other, .5
    torch.testing.assert_close(attention(states, base), (native + alternate) / 2)
    route.reference, route.weight = mixed, 1.
    torch.testing.assert_close(attention(states, base), token_control)
    assert not torch.allclose((native + alternate) / 2, token_control)

    route.reference = base
    torch.testing.assert_close(attention(states, base), native, rtol=0, atol=0)


def test_unet_keeps_early_steps_and_zero_updates_bitwise_native():
    torch.manual_seed(105)
    attention = SmallImageAttention().eval()
    route = O13ImageAttention(attention)
    states = torch.randn(2, 16, 6)
    base, other = torch.randn(2, 2, 4, 6)

    def native(sample, timestep, *, cross_attention_kwargs):
        return SimpleNamespace(sample=attention(sample, cross_attention_kwargs["ip_hidden_states"]))

    native.dtype = states.dtype
    kwargs = {"cross_attention_kwargs": {"ip_hidden_states": base}}
    expected = native(states, torch.tensor([499, 499]), **kwargs).sample
    for timestep in (999, 749, 499, 249):
        time = torch.full((2,), timestep)
        zero = LateAttentionUNet(native, (route,), other, 0.)
        identity = LateAttentionUNet(native, (route,), base, .5)
        torch.testing.assert_close(zero(states, time, **kwargs).sample, expected, rtol=0, atol=0)
        torch.testing.assert_close(identity(states, time, **kwargs).sample, expected, rtol=0, atol=0)
        active = LateAttentionUNet(native, (route,), other, .5)
        output = active(states, time, **kwargs).sample
        if timestep in (999, 749):
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        else:
            assert not torch.allclose(output, expected)
        assert route.reference is None and route.activations == {}


def test_exchange_swaps_the_two_semantic_views_only():
    h14 = torch.arange(48, dtype=torch.float32).reshape(24, 2)
    b32 = h14 + 100
    recipient_layout = h14 + 200
    donors = torch.roll(torch.arange(len(h14)), len(h14) // 2)

    def vision(**views):
        return torch.cat(tuple(views.values()), dim=1)

    values = compose_features(vision, ("h14", "b32", "layout"), h14, b32, recipient_layout, donors)
    torch.testing.assert_close(values[:, :2], h14[donors])
    torch.testing.assert_close(values[:, 2:4], b32[donors])
    torch.testing.assert_close(values[:, 4:], recipient_layout)
