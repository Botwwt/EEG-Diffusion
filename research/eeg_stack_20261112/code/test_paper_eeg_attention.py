from copy import deepcopy

import pytest
import torch
from torch import nn

from paper_eeg_attention import ImageProjectionUpdate, merge_image_updates, response_loss, without_updates


def test_initial_update_reproduces_the_native_projection():
    layer = nn.Linear(6, 5, bias=False)
    update = ImageProjectionUpdate(layer, rank=3)
    queries = torch.randn(2, 4, 6)
    assert not layer.weight.requires_grad
    assert torch.equal(update(queries), layer(queries))
    assert update.up.requires_grad and update.down.requires_grad


def test_merge_matches_the_learned_projection_and_restores_the_base_even_on_error():
    layer = nn.Linear(6, 5, bias=False)
    image_projection = ImageProjectionUpdate(layer, rank=3)
    with torch.no_grad():
        image_projection.up.normal_(std=.1)
    original = deepcopy(layer.state_dict())
    x = torch.randn(3, 4, 6)
    before = image_projection(x)
    with pytest.raises(RuntimeError, match="interrupted"):
        with merge_image_updates({"v": layer}, {"v": image_projection.export()}):
            torch.testing.assert_close(layer(x), before)
            raise RuntimeError("interrupted")
    torch.testing.assert_close(layer.weight, original["weight"], rtol=0, atol=0)


def test_oracle_pass_disables_updates_and_then_restores_them():
    layer = nn.Linear(3, 2, bias=False)
    update = ImageProjectionUpdate(layer, rank=2)
    with torch.no_grad():
        update.up.fill_(1)
    x = torch.randn(4, 3)
    assert not torch.allclose(update(x), layer(x))
    with without_updates({"k": update}):
        torch.testing.assert_close(update(x), layer(x), rtol=0, atol=0)
    assert update.enabled is True


def test_response_loss_does_not_explode_when_true_and_mean_teacher_agree():
    null = torch.zeros(2, 1, 2, 2)
    signal = torch.ones_like(null)
    assert response_loss(signal, signal, null) == 0
    assert response_loss(null, signal, null) == pytest.approx(1.)
    assert torch.isfinite(response_loss(signal, null, null))


def test_rollout_returns_the_state_before_the_requested_step():
    from paper_eeg_attention import rollout_conditioning_input

    class Schedule:
        init_noise_sigma = 2.

        def set_timesteps(self, count, device):
            assert count == 4 and device.type == "cpu"
            self.timesteps = torch.tensor([999, 749, 499, 249])

        def scale_model_input(self, sample, t):
            return sample / 2

        def step(self, prediction, t, sample, generator, return_dict):
            assert return_dict is False
            return (sample - prediction,)

    calls = []

    def predict(value, time):
        calls.append(time)
        return value

    epsilon = torch.tensor([6.])
    output = rollout_conditioning_input(Schedule(), epsilon, 499, predict, torch.Generator())
    assert calls == [999, 749]
    torch.testing.assert_close(output, torch.tensor([1.5]))
