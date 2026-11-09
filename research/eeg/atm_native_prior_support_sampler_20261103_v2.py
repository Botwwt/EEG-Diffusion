"""One configuration comparison of the unchanged native DDPM prior.

This is a diagnostic, not a new model. The original clipped sampler stays frozen.
"""
import torch
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor


@torch.no_grad()
def sample(prior, conditions, seeds, *, clip_sample, trace=False):
    assert isinstance(clip_sample, bool) and isinstance(trace, bool)
    assert conditions.ndim == 2 and conditions.shape[1] == 1024
    assert len(conditions) == len(seeds) > 0
    assert all(isinstance(v, int) and 0 <= v < 2**63 for v in seeds)
    assert not prior.training and all(not p.requires_grad for p in prior.parameters())
    device = next(prior.parameters()).device
    assert next(prior.parameters()).dtype == torch.float32
    conditions = conditions.to(device=device, dtype=torch.float32)
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    scheduler = DDPMScheduler(clip_sample=clip_sample)
    scheduler.set_timesteps(50, device=device)
    assert scheduler.config.prediction_type == 'epsilon'
    assert scheduler.config.variance_type == 'fixed_small'
    state = randn_tensor((len(conditions), 1024), generator=generators,
                        device=device, dtype=torch.float32)
    states = [state[:2].detach().cpu().numpy()] if trace else None
    predictions, noises = [], []
    for timestep in scheduler.timesteps:
        time = torch.ones(len(state), device=device, dtype=torch.float32) * timestep
        conditional = prior(state, time, conditions)
        missing = prior(state, time, None)
        epsilon = missing + 5. * (conditional - missing)
        if trace:
            clones = [torch.Generator(device=device).set_state(g.get_state()) for g in generators[:2]]
            variance_noise = (randn_tensor(state[:2].shape, generator=clones, device=device,
                             dtype=state.dtype) if int(timestep.item()) > 0 else torch.zeros_like(state[:2]))
            predictions.append(epsilon[:2].detach().cpu().numpy())
            noises.append(variance_noise.detach().cpu().numpy())
        state = scheduler.step(epsilon, int(timestep.item()), state,
                               generator=generators).prev_sample
        assert torch.isfinite(state).all()
        if trace:
            states.append(state[:2].detach().cpu().numpy())
    return state, (dict(states=states, predictions=predictions, variance_noises=noises,
                       timesteps=scheduler.timesteps.cpu().numpy(),
                       alphas_cumprod=scheduler.alphas_cumprod.cpu().numpy()) if trace else None)
