"""Explicit metric definitions for paired retrieval/generation experiments.

These portable utilities are release-time refactors, not the byte-identical
historical evaluators. In particular, query bootstrap keeps the gallery fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _scores(similarities: ArrayLike, targets: ArrayLike) -> tuple[NDArray[np.float64], NDArray]:
    scores = np.asarray(similarities, dtype=np.float64)
    truth = np.asarray(targets)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("Expected a finite, nonempty query-by-gallery matrix")
    if truth.shape != (len(scores),) or not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("One integer target index is required per query")
    if np.any(truth < 0) or np.any(truth >= scores.shape[1]):
        raise ValueError("Target index outside gallery")
    return scores, truth


def retrieval_top1(similarities: ArrayLike, targets: ArrayLike) -> NDArray[np.float64]:
    """Top-1 per query, with deterministic first-index tie breaking (numpy argmax)."""
    scores, truth = _scores(similarities, targets)
    return (scores.argmax(axis=1) == truth).astype(np.float64)


def image_two_way(similarities: ArrayLike, targets: ArrayLike) -> NDArray[np.float64]:
    """Per-query pairwise identification against every non-target candidate.

    Half credit for tied scores; unrelated candidates only. This metric is
    different from gallery Top-1 and from EEG-embedding retrieval.
    """
    scores, truth = _scores(similarities, targets)
    if scores.shape[1] < 2:
        raise ValueError("At least one distractor is required")
    matched = scores[np.arange(len(scores)), truth, None]
    wins = (matched > scores).astype(float) + 0.5 * (matched == scores)
    wins[np.arange(len(scores)), truth] = 0
    return wins.sum(axis=1) / (scores.shape[1] - 1)


@dataclass(frozen=True)
class PairedInterval:
    difference: float
    lower: float
    upper: float
    units: int
    repetitions: int
    seed: int
    resampling_unit: str = "query; gallery fixed"


def paired_query_bootstrap(
    treatment: ArrayLike,
    control: ArrayLike,
    *,
    repetitions: int = 20000,
    seed: int = 0,
) -> PairedInterval:
    """Paired percentile interval; seeds/repeats must first be grouped per query.

    This interval is conditional on the fixed subjects, trained models and
    gallery. It is not a population or multi-subject confidence statement.
    """
    a, b = np.asarray(treatment, dtype=float), np.asarray(control, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("At least two paired query values are required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Missing/invalid queries must be reported explicitly")
    if not isinstance(repetitions, int) or repetitions < 100:
        raise ValueError("Use at least 100 resamples")
    delta = a - b
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    # Bounded temporary allocation for large query sets.
    for start in range(0, repetitions, 256):
        end = min(start + 256, repetitions)
        ids = rng.integers(0, len(delta), size=(end - start, len(delta)))
        estimates[start:end] = delta[ids].mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return PairedInterval(float(delta.mean()), float(lo), float(hi), len(delta), repetitions, seed)
