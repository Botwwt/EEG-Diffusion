"""Conditional mean and mode of a visual Gaussian mixture given an EEG likelihood."""

import torch


def conditional_mixture_estimates(masses, centers, covariance, precision, information, refinements=24):
    """Condition a tied-covariance mixture on exp(info @ q - q @ precision @ q / 2).

    `information` has one row per query. Search for the joint density mode from
    every conditional component mean, rather than choosing a component by mass.
    """
    prior_precision = torch.linalg.inv(covariance)
    posterior_precision = prior_precision + precision
    prior_information = centers @ prior_precision
    component_information = information[:, None, :] + prior_information
    conditional_centers = torch.linalg.solve(posterior_precision, component_information.mT).mT

    component_bias = masses.log() - 0.5 * (centers * prior_information).sum(dim=-1)
    component_log_mass = component_bias + 0.5 * (component_information * conditional_centers).sum(dim=-1)
    conditional_mean = (component_log_mass.softmax(dim=-1)[..., None] * conditional_centers).sum(dim=1)

    candidates = conditional_centers
    for _ in range(refinements):
        logits = candidates @ component_information.mT + component_bias
        candidates = logits.softmax(dim=-1) @ conditional_centers

    logits = candidates @ component_information.mT + component_bias
    log_density = logits.logsumexp(dim=-1) - 0.5 * ((candidates @ posterior_precision) * candidates).sum(dim=-1)
    best = log_density.argmax(dim=-1)
    conditional_mode = candidates[torch.arange(len(best), device=best.device), best]
    return conditional_mean, conditional_mode


def replace_relative_coordinates(tokens, center, analysis, synthesis, coordinates, residual_strength):
    """Write a fixed coordinate vector and keep the specified fraction of the remaining offset."""
    tokens = tokens.float()
    common = tokens.mean(dim=1, keepdim=True)
    offset = (tokens - common).flatten(1) - center
    remainder = offset - (offset @ analysis) @ synthesis.mT
    relative = center + coordinates @ synthesis.mT + residual_strength * remainder
    return common + relative.reshape_as(tokens)
