"""Batched execution of the native 50-step DDPM prior, with query-specific RNG.

The published Pipe.generate calls Tensor.item() on a batch of timesteps and
therefore only accepts one query. This implementation passes the shared scalar
timestep to the unchanged scheduler. It is an execution adapter, not a new model.
Real single-query/batch numerical parity is required before any cache is admitted.
"""
import torch
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor


@torch.no_grad()
def sample_native_prior(prior, conditions, seeds, *, return_states=False):
    assert conditions.ndim == 2 and conditions.shape[1] == 1024
    assert len(conditions) == len(seeds) and len(conditions) > 0
    assert all(isinstance(seed, int) and 0 <= seed < 2**63 for seed in seeds)
    assert not prior.training and all(not p.requires_grad for p in prior.parameters())
    device = next(prior.parameters()).device
    assert next(prior.parameters()).dtype == torch.float32
    conditions = conditions.to(device=device, dtype=torch.float32)
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    scheduler = DDPMScheduler()
    scheduler.set_timesteps(50, device=device)
    assert scheduler.config.clip_sample and scheduler.config.prediction_type == 'epsilon'
    state = randn_tensor((len(conditions), 1024), generator=generators, device=device, dtype=torch.float32)
    states = [state.detach().clone()] if return_states else None
    for timestep in scheduler.timesteps:
        time = torch.ones(len(state), device=device, dtype=torch.float32) * timestep
        conditional = prior(state, time, conditions)
        missing = prior(state, time, None)
        epsilon = missing + 5. * (conditional - missing)
        state = scheduler.step(epsilon, int(timestep.item()), state, generator=generators).prev_sample
        if return_states:
            states.append(state.detach().clone())
    assert torch.isfinite(state).all()
    return state, states
