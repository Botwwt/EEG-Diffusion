import pytest
import torch

from paper_spatial_photo import compose_crops, position_weights, routed_attention


def test_positions_follow_image_raster_order_and_sum_to_one():
    windows = position_weights(64)
    identity = torch.eye(4)
    assert torch.equal(windows[[0, 7, 56, 63]], identity)
    torch.testing.assert_close(windows.sum(-1), torch.ones(64))
    assert torch.equal(position_weights(1), torch.full((1, 4), .25))
    with pytest.raises(ValueError, match="square"):
        position_weights(8)


def test_equal_crops_make_no_attention_update_and_global_blend_ignores_order():
    g = torch.Generator().manual_seed(40)
    query = torch.randn(2, 3, 16, 5, generator=g)
    key, value = (torch.randn(2, 3, 4, 5, generator=g) for _ in range(2))
    same_k, same_v = (v[:, None].expand(-1, 4, -1, -1, -1) for v in (key, value))
    assert torch.count_nonzero(routed_attention(query, key, value, same_k, same_v, position_weights(16))) == 0
    crop_k, crop_v = (torch.randn(2, 4, 3, 4, 5, generator=g) for _ in range(2))
    flipped = routed_attention(query, key, value, crop_k.flip(1), crop_v.flip(1), torch.full((16, 4), .25))
    normal = routed_attention(query, key, value, crop_k, crop_v, torch.full((16, 4), .25))
    torch.testing.assert_close(normal, flipped)
    first = routed_attention(query, key, value, crop_k, crop_v, position_weights(16))
    last = routed_attention(query, key, value, crop_k.flip(1), crop_v.flip(1), position_weights(16))
    assert not torch.equal(first[:, :, 0], last[:, :, 0])


def test_composition_preserves_each_photo_direction_and_calibrates_crop_norms():
    base = torch.tensor([[5., 0.], [0., 2.]])
    displacement = torch.zeros(2, 4, 2)
    displacement[0, 1, 1] = 5.
    crops = compose_crops(base, displacement)
    torch.testing.assert_close(crops.norm(dim=-1), torch.full((2, 4), 32.))
    torch.testing.assert_close(crops[0, [0, 2, 3]], torch.tensor([[32., 0.]]).expand(3, -1))
    assert crops[0, 1, 1] > 0 and torch.equal(crops[1], torch.tensor([[0., 32.]]).expand(4, -1))
