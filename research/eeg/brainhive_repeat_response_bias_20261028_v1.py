"""Classical delete-one-repeat jackknife, as a host-independent math prototype.

This module neither trains nor loads an EEG encoder or a generator.  It uses the
four *physical* repeats supplied by its caller; it does not fabricate repeats.
For a smooth fixed-Q response T and iid centered errors, the classical estimate

    4 T(mean4) - 3 mean_r T(mean_except_r)

cancels the leading 1/n expectation bias.  The coefficient is derived from that
assumption, not fitted here.  Actual EEG repeats have NOT been established as iid
or centered.  Correct algebra does not establish lower variance, lower MSE,
whole-generator unbiasedness, new algorithmic novelty, or better images.

All operations are deterministic, consume no RNG, and have no host dependency.
CPU verification is separate.  Zero-valued EEG/responses are legitimate; empty
axes, wrong repeat counts, non-floating tensors, and non-finite values are not.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import torch
from torch import Tensor


REPEATS = 4
M3_SUBSETS = tuple(combinations(range(REPEATS), 3))
M2_SUBSETS = tuple(combinations(range(REPEATS), 2))


@dataclass(frozen=True)
class RepeatSubsetMeans:
    """m4=[B,...], m3=[4,B,...], m2=[6,B,...], lexicographic subsets.

    M3_SUBSETS order is (012, 013, 023, 123), hence the deleted-repeat order is
    (3, 2, 1, 0).  Consumers must not silently assume deleted order (0, 1, 2, 3).
    """

    m4: Tensor
    m3: Tensor
    m2: Tensor
    m3_subsets: tuple = M3_SUBSETS
    m2_subsets: tuple = M2_SUBSETS


def _finite_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a real floating dtype")
    if value.numel() == 0 or any(d == 0 for d in value.shape):
        raise ValueError(f"{name} must not have an empty axis")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def subset_means(trials: Tensor) -> RepeatSubsetMeans:
    """Return all R4/R3/R2 means of trials [B,4,...] without resampling.

    Identical physical repeats return exactly their original values, including
    signed zero, rather than introducing a divide-by-three rounding difference.
    """

    _finite_tensor(trials, "trials")
    if trials.ndim < 2 or trials.shape[1] != REPEATS:
        raise ValueError("trials must have shape [B,4,...], repeat axis=1")
    same = (trials == trials[:, :1]).flatten(start_dim=1).all(dim=1)
    batch_mask = same.reshape((trials.shape[0],) + (1,) * (trials.ndim - 2))
    m4 = torch.where(batch_mask, trials[:, 0], trials.mean(dim=1))

    def all_means(subsets: tuple) -> Tensor:
        result = torch.stack([trials[:, indices, ...].mean(dim=1) for indices in subsets])
        return torch.where(batch_mask.unsqueeze(0), m4.unsqueeze(0), result)

    m3, m2 = all_means(M3_SUBSETS), all_means(M2_SUBSETS)
    for name, value in (("m4", m4), ("m3", m3), ("m2", m2)):
        _finite_tensor(value, name)
    return RepeatSubsetMeans(m4=m4, m3=m3, m2=m2)


def _response_pair(base: Tensor, loo_response: Tensor) -> None:
    _finite_tensor(base, "base")
    _finite_tensor(loo_response, "loo_response")
    if loo_response.shape != (REPEATS,) + tuple(base.shape):
        raise ValueError("loo_response must have shape [4, *base.shape]")
    if base.device != loo_response.device or base.dtype != loo_response.dtype:
        raise ValueError("base and loo_response must have the same device and dtype")


def jackknife_response(
    base: Tensor,
    loo_response: Tensor,
    *,
    direction: Literal["debias", "opposite"] = "debias",
) -> Tensor:
    """Apply the classical leading-bias counterterm to a fixed-Q response.

    base is T(mean4); loo_response[r] is T(mean_except_r), with the SAME Q.
    The opposite-sign control has the identical displacement norm.  No clipping,
    normalization, learned gate, RNG, or response-strength selection is applied.
    All-identical responses return the original base tensor object exactly.
    """

    _response_pair(base, loo_response)
    if direction not in ("debias", "opposite"):
        raise ValueError("direction must be 'debias' or 'opposite'")
    equal = (loo_response == base.unsqueeze(0)).all(dim=0)
    if bool(equal.all()):
        return base
    mean_loo = torch.where(equal, base, loo_response.mean(dim=0))
    counterterm = 3.0 * (mean_loo - base)
    _finite_tensor(counterterm, "counterterm")
    result = base - counterterm if direction == "debias" else base + counterterm
    result = torch.where(equal, base, result)
    _finite_tensor(result, "response result")
    return result


def opposite_response(base: Tensor, loo_response: Tensor) -> Tensor:
    """Same-magnitude, opposite-sign control; not a second fitted model."""

    return jackknife_response(base, loo_response, direction="opposite")


def response_summary(base: Tensor, loo_response: Tensor) -> dict:
    """Descriptive magnitudes only; these are not efficacy or SNR estimates."""

    corrected = jackknife_response(base, loo_response)
    b = base.detach().to(dtype=torch.float64)
    delta = corrected.detach().to(dtype=torch.float64) - b
    base_rms = float(b.square().mean().sqrt())
    correction_rms = float(delta.square().mean().sqrt())
    return {
        "base_rms": base_rms,
        "correction_rms": correction_rms,
        "correction_to_base_rms": correction_rms / base_rms if base_rms > 0 else None,
        "max_abs_correction": float(delta.abs().max()),
        "count": base.numel(),
        "descriptive_only": True,
    }
