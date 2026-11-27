import torch

from paper_foreground_targets import matched_conditions, patch_representations


def test_patch_views_share_tokens_and_shift_only_the_mask():
    patches = torch.zeros(2, 4, 2)
    patches[:, 0::2, 0] = 1.
    patches[:, 1::2, 1] = 1.
    mask = torch.zeros(2, 2, 2)
    mask[0, :, 0] = 1.
    actual = patch_representations(patches, mask)
    torch.testing.assert_close(actual["foreground"][0], torch.tensor([1., 0.]))
    torch.testing.assert_close(actual["shifted"][0], torch.tensor([0., 1.]))
    torch.testing.assert_close(actual["uniform"][0], torch.tensor([.5, .5]).sqrt())
    torch.testing.assert_close(actual["foreground"][1], actual["uniform"][1])


def test_conditions_match_norms_and_exchange_only_the_foreground_readout():
    native = torch.tensor([[1., 0.], [2., -1.]])
    region = {
        "foreground": torch.tensor([[0., 3.], [0., 5.]]),
        "uniform": torch.tensor([[1., 1.], [1., 1.]]),
        "shifted": torch.tensor([[0., -1.], [0., -3.]]),
    }
    actual = matched_conditions(native, region, [1, 0], 2.)
    for name in ("uniform", "shifted", "foreground-wrong"):
        torch.testing.assert_close(actual[name].norm(dim=1), actual["foreground"].norm(dim=1))
    torch.testing.assert_close(actual["foreground-wrong"][0] / actual["foreground-wrong"][0, 1],
                               torch.tensor([.2, 1.]))
    torch.testing.assert_close(actual["foreground"][0], torch.tensor([.5, 1.5]))
