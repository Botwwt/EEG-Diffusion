import numpy as np
import torch

from paper_cbramod_repeats import encode_repeats, signal_scale, to_200hz


def test_resampling_preserves_four_presentations_and_constant_values():
    original = np.full((2, 4, 17, 250), 2., dtype=np.float32)
    sampled = to_200hz(original)
    assert sampled.shape == (2, 4, 17, 200)
    np.testing.assert_allclose(sampled, 2., atol=.003)
    x = torch.from_numpy(sampled)
    center, scale = signal_scale(x)
    assert center.shape == scale.shape == (1, 1, 17, 1)
    assert torch.isfinite(scale).all() and (scale > 0).all()


def test_joint_and_independent_encoders_have_the_same_per_trial_stem():
    class Recording:
        def __init__(self):
            self.stem_shapes = []
            self.encoder_shapes = []

        def patch_embedding(self, x):
            self.stem_shapes.append(x.shape)
            return x * x

        def encoder(self, x):
            self.encoder_shapes.append(x.shape)
            return x + x.mean(2, keepdim=True) ** 2

    x = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(4))
    network = Recording()
    direct = encode_repeats(network, x, "joint")
    independent = encode_repeats(network, x, "separate")
    repeated = encode_repeats(network, x[:, [2, 0, 3, 1]], "joint")
    assert network.stem_shapes == [(8, 3, 1, 5)] * 3
    assert network.encoder_shapes == [(2, 3, 4, 5), (8, 3, 1, 5), (2, 3, 4, 5)]
    torch.testing.assert_close(direct, repeated)
    assert not torch.equal(direct, independent)
