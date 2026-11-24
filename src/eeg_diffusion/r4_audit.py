"""Reconstruct the R4 s375 CLIP Intra10 comparison from participant records."""

from __future__ import annotations

import json
from itertools import product
from math import isclose, sqrt
from pathlib import Path
from statistics import mean, stdev

_RESULTS = Path("research/eeg_stack_20261112/results")
_N_IMAGES = 500
_ALL = tuple(range(1, 11))
_HELDOUT = (1, 3, 4, 6, 7, 10)
_SELECTED = "M1a + D2"
_CONTROLS = ("M1a + SwapDiff", "M1a + Arm1b", "Naive D2")
# Two-sided 95% Student-t critical values for the fixed participant groups.
_T95 = {6: 2.5705818366, 10: 2.2621571629}
_EXP7 = "exp7_heldout6_and_all10_results.json"
_SOURCES = {
    _EXP7: {
        "Naive Arm1b": ("all10_per_sub", "Arm1b_RepIntra_CondLyr", "s375"),
        "Naive SwapDiff": ("all10_per_sub", "SwapDiff_Arm3b", "s375"),
        "Naive D2": ("all10_per_sub", "D2_Compete4_Cond", "s375"),
    },
    "exp8c_m1a_heldout6_and_all10_results.json": {
        "M1a + SwapDiff": ("m1a_arm3b_all10", "s375"),
        _SELECTED: ("m1a_d2_all10", "s375"),
    },
    "exp8d_m1a_arm1b_symmetric_results.json": {"M1a + Arm1b": ("m1a_arm1b_all10", "s375")},
    "exp9b_m2_and_m1m2_all10_results.json": {
        "Naive M2a": ("m2a_naive_all10",),
        "M1a + M2a": ("m1a_m2a_all10",),
    },
}


def _unique_keys(pairs: list[tuple]) -> dict:
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("Duplicate JSON key in R4 evidence")
    return values


def _participant_hits(records: dict, arm: str) -> dict[str, int]:
    if not isinstance(records, dict) or set(records) != {f"sub-{s:02d}" for s in _ALL}:
        raise ValueError(f"Unexpected or missing participant for {arm}")
    hits = {}
    for subject, metrics in records.items():
        try:
            rate = metrics["clip"]["top1"]
        except (KeyError, TypeError) as err:
            raise ValueError(f"Missing CLIP Top-1 for {arm}/{subject}") from err
        if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not 0 <= rate <= 1:
            raise ValueError(f"Invalid Top-1 rate for {arm}/{subject}")
        count = round(rate * _N_IMAGES)
        if not isclose(rate * _N_IMAGES, count, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"Top-1 rate gives a fractional hit for {arm}/{subject}")
        hits[subject] = count
    return hits


def _paired(deltas: list[int]) -> dict:
    n = len(deltas)
    delta_pp = mean(deltas) * 100 / _N_IMAGES
    width = _T95[n] * stdev(deltas) / sqrt(n) * 100 / _N_IMAGES
    # Integer hit differences avoid floating-point ties in the exact enumeration.
    observed = abs(sum(deltas))
    extreme = sum(
        abs(sum(sign * d for sign, d in zip(signs, deltas, strict=True))) >= observed
        for signs in product((1, -1), repeat=n)
    )
    return {
        "difference_pp": delta_pp,
        "ci95_pp": [delta_pp - width, delta_pp + width],
        "participants_improved": sum(d > 0 for d in deltas),
        "participants_tied": deltas.count(0),
        "p_sign_flip": extreme / 2**n,
    }


def verify_r4_results(root: Path) -> dict:
    """Check participant IDs and integer hit counts for the eight reported arms."""
    hits = {}
    for filename, arms in _SOURCES.items():
        data = json.loads(
            (root / _RESULTS / filename).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_keys,
        )
        if not isinstance(data, dict):
            raise ValueError(f"Expected an R4 evidence object in {filename}")
        if filename == _EXP7:
            for label, expected in (("heldout_subs", _HELDOUT), ("dev_subs", (2, 5, 8, 9))):
                members = data.get(label)
                if (
                    not isinstance(members, list)
                    or any(type(s) is not int for s in members)
                    or sorted(members) != list(expected)
                ):
                    raise ValueError(f"Unexpected participant split for {label}")
        for arm, keys in arms.items():
            records = data
            for key in keys:
                if not isinstance(records, dict) or key not in records:
                    raise ValueError(f"Missing {key} in {filename} for {arm}")
                records = records[key]
            hits[arm] = _participant_hits(records, arm)

    splits = {}
    for split, ids in (("all10", _ALL), ("heldout6", _HELDOUT)):
        subjects = [f"sub-{s:02d}" for s in ids]
        queries = len(subjects) * _N_IMAGES
        totals = {arm: sum(records[s] for s in subjects) for arm, records in hits.items()}
        splits[split] = {
            "participants": subjects,
            "queries": queries,
            "arms": {arm: {"hits": n, "top1": n / queries} for arm, n in totals.items()},
            "paired_vs_selected": {
                control: _paired([hits[_SELECTED][s] - hits[control][s] for s in subjects])
                for control in _CONTROLS
            },
        }
    return {
        "metric": "R4 s375 free-generation OpenAI CLIP ViT-L/14 Intra10 Top-1",
        "selected": _SELECTED,
        "images_per_participant": _N_IMAGES,
        "sources": [(_RESULTS / source).as_posix() for source in _SOURCES],
        "paired_method": "Selected minus control, participant-paired 95% t interval "
        "and two-sided exact sign-flip p-value.",
        "splits": splits,
    }
