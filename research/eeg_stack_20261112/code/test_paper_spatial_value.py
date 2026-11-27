import torch
from torch import nn
from torch.nn import functional as F

from paper_spatial_value import SpatialImageValue, project_spatial_band


class SmallImageAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.norm_q = nn.LayerNorm(3)
        self.to_q = nn.Linear(6, 6)
        self.to_out = nn.ModuleList([nn.Linear(6, 6)])
        self.processor = nn.Module()
        self.processor.scale = .75
        self.processor.to_k_ip = nn.Linear(6, 6, bias=False)
        self.processor.to_v_ip = nn.Linear(6, 6, bias=False)

    def forward(self, states, tokens, values=None):
        def heads(x):
            return x.unflatten(-1, (self.heads, 3)).transpose(1, 2)

        query = self.norm_q(heads(self.to_q(states)))
        key = heads(self.processor.to_k_ip(tokens))
        value = heads(self.processor.to_v_ip(tokens if values is None else values))
        output = F.scaled_dot_product_attention(query, key, value)
        return self.to_out[0](states + self.processor.scale * output.transpose(1, 2).flatten(2))


def test_grid_mean_and_residual_are_complementary():
    value = torch.arange(32., dtype=torch.float64).reshape(1, 16, 2)
    rough = project_spatial_band(value, "coarse")
    detail = project_spatial_band(value, "detail")
    torch.testing.assert_close(rough + detail, value)
    torch.testing.assert_close(rough.reshape(1, 4, 4, 2)[0, :2, :2, 0], torch.full((2, 2), 5., dtype=value.dtype))
    torch.testing.assert_close(project_spatial_band(detail, "coarse"), torch.zeros_like(value))


def test_attention_corrections_recombine_into_full_value_replacement():
    torch.manual_seed(45)
    attention = SmallImageAttention().eval()
    states = torch.randn(2, 16, 6)
    original, substitute = torch.randn(2, 4, 6), torch.randn(2, 4, 6)
    with torch.inference_mode():
        native = attention(states, original)
        replaced = attention(states, original, substitute)
        route = SpatialImageValue(attention, attention.processor.to_v_ip)
        torch.testing.assert_close(attention(states, original), native, rtol=0, atol=0)
        route.reference = substitute
        route.band = "coarse"
        coarse = attention(states, original)
        route.band = "detail"
        detail = attention(states, original)
        torch.testing.assert_close(coarse + detail - native, replaced)
        assert not torch.allclose(native, coarse) and not torch.allclose(native, detail)
        route.reference = None
        torch.testing.assert_close(attention(states, original), native, rtol=0, atol=0)
