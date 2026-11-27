import torch
from torch.nn import functional as F

from paper_ijepa_context import context_targets, patch_indices, patch_pool


def test_quadrants_partition_the_image_and_exclude_hidden_tokens():
    visible, hidden = patch_indices(16)
    assert visible.shape == (4, 192) and hidden.shape == (4, 64)
    assert torch.equal(hidden.flatten().sort().values, torch.arange(256))
    for context, target in zip(visible, hidden, strict=True):
        assert torch.equal(torch.cat((context, target)).sort().values, torch.arange(256))


def test_masked_prediction_errors_follow_the_target_positions():
    torch.manual_seed(2)
    tokens = torch.randn(2, 16, 3)

    def visible_only(pixel_tokens, masks):
        return torch.cat([pixel_tokens.gather(1, m[..., None].expand(-1, -1, 3)) for m in masks])

    def predict(context, mask_visible, mask_target):
        assert mask_visible.shape == (8, 12)
        return context.mean(1, keepdim=True).expand(-1, 4, -1)

    own, error = context_targets(tokens, visible_only, lambda x: x, predict, side=4)
    visible, hidden = patch_indices(4)
    torch.testing.assert_close(own, F.layer_norm(tokens, (3,)))
    for context, target in zip(visible, hidden, strict=True):
        prediction = tokens[:, context].mean(1, keepdim=True).expand(-1, 4, -1)
        expected = F.smooth_l1_loss(prediction, own[:, target], reduction="none").mean(-1)
        torch.testing.assert_close(error[:, target], expected)


def test_photo_specific_weights_follow_the_map_source_without_swapping_the_photo():
    patches = torch.tensor([[[1.], [0.]], [[0.], [4.]]])
    errors = torch.tensor([[8., 0.], [0., 8.]])
    average, scales = torch.full((2,), 4.), torch.ones(2)
    pool, weights = patch_pool(patches, errors, average, scales, torch.tensor([1, 0]))
    torch.testing.assert_close(weights["same-object-map"], weights["photo-map"].flip(0))
    torch.testing.assert_close(weights["mean-map"], torch.full((2, 2), .5))
    torch.testing.assert_close(weights["photo-map"].sum(1), torch.ones(2))
    assert pool["photo-map"][0, 0] > pool["mean-map"][0, 0] > pool["same-object-map"][0, 0]
    assert pool["photo-map"][1, 0] > pool["mean-map"][1, 0] > pool["same-object-map"][1, 0]


def test_foreground_limits_every_arm_to_current_photo_and_handles_empty_masks():
    tokens = torch.tensor([[[1.], [8.], [12.]], [[2.], [5.], [9.]]])
    errors = torch.tensor([[0., 90., 20.], [0., 8., 0.]])
    foreground = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    average, scales = torch.zeros(3), torch.ones(3)
    result, weights = patch_pool(tokens, errors, average, scales, torch.tensor([1, 0]), foreground=foreground)

    for arm, weight in weights.items():
        torch.testing.assert_close(weight[0], torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(weight.sum(1), torch.ones(2))
        assert result[arm][0, 0] == 1.
        assert torch.isfinite(result[arm]).all()
