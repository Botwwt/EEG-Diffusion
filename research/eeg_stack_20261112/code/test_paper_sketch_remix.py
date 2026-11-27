from types import SimpleNamespace
import unittest

import torch

from exp6_draft_intra4 import rollout_4step_step_ips


class RecordingDenoiser:
    dtype = torch.float32

    def __init__(self):
        self.inputs = []

    def __call__(self, latent, *args, **kwargs):
        self.inputs.append(latent.clone())
        return SimpleNamespace(sample=torch.ones_like(latent))


class SketchRemixTests(unittest.TestCase):
    def test_terminal_resume_preserves_the_exact_euler_state(self):
        schedule = SimpleNamespace(timesteps=torch.tensor([999, 749, 499, 249]),
                                   sigmas=torch.tensor([2., 1., .5, .25, 0.]))
        sketch = torch.ones(1, 4, 8, 8)
        noise = torch.full_like(sketch, .25)
        condition = torch.zeros(1, 1, 1)
        pooled = torch.zeros(1, 1)
        args = (schedule, [condition] * 4, sketch, noise, [noise * i for i in range(4)],
                condition, pooled, pooled, .375, (8, .11, .02))
        full, prefix, terminal = RecordingDenoiser(), RecordingDenoiser(), RecordingDenoiser()

        expected = rollout_4step_step_ips(full, *args)
        state = rollout_4step_step_ips(prefix, *args, stop_step=3)
        resumed = rollout_4step_step_ips(terminal, *args, start_step=3, start_latent=state)

        self.assertEqual((len(prefix.inputs), len(terminal.inputs)), (3, 1))
        torch.testing.assert_close(terminal.inputs[0], full.inputs[-1], rtol=0, atol=0)
        torch.testing.assert_close(resumed, expected, rtol=0, atol=0)

    def test_disabling_remix_preserves_the_sketch_initialization(self):
        schedule = SimpleNamespace(timesteps=torch.arange(4), sigmas=torch.tensor([2., 1., .5, .25, 0.]))
        sketch = torch.ones(1, 4, 2, 2)
        noise = torch.full_like(sketch, .25)
        zero = torch.zeros_like(sketch)
        condition = torch.zeros(1, 1, 1)
        pooled = torch.zeros(1, 1)
        original, changed = RecordingDenoiser(), RecordingDenoiser()
        args = (schedule, [condition] * 4, sketch, noise, [zero] * 4,
                condition, pooled, pooled, .375, None)

        rollout_4step_step_ips(original, *args)
        rollout_4step_step_ips(changed, *args, remix_initial_layout=False)

        torch.testing.assert_close(changed.inputs[0], original.inputs[0])
        torch.testing.assert_close(changed.inputs[1], torch.zeros_like(sketch))
        self.assertGreater(original.inputs[1].abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
