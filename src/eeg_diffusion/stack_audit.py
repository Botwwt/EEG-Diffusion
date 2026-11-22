"""Recompute published Top-1 counts from the November stack's subject table."""

from __future__ import annotations

import re
from math import isclose, sqrt
from pathlib import Path
from statistics import mean, stdev

STACK_LOG = Path("research/eeg_stack_20261112/results/o12_o13_o14_info_ceiling_xbrain.txt")
_COLUMNS = {
    "O10": "O10",
    "O11nb": "O11_noBigG",
    "O12ct": "O12_ctrl_enc",
    "O12nb": "O12_noBigG",
    "O13ct": "O13_ctrl_xsub",
    "O13": "O13",
    "O14nb": "O14_noBigG",
    "O14": "O14",
    "O14sh": "O14_ctrl_shuf",
    "O14s4": "O14_sel4",
}
_RANK_ARMS = {"O12_noBigG", "O13_ctrl_xsub", "O13", "O14_noBigG", "O14", "O14_sel4"}
_SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 11))
_IMAGES_PER_SUBJECT = 200 * 4
_ROUNDING_TOLERANCE = 0.00005 + 1e-12
_T95_DF9 = 2.262157


def _hit_counts(log: str) -> dict[str, dict[str, int]]:
    _, heading, rest = log.partition("=== Per-Subject Official CLIP 200-Way Top-1")
    if not heading:
        raise ValueError("Missing stack subject table")
    table = rest.split("\n===", 1)[0].splitlines()
    if len(table) < 2:
        raise ValueError("Missing stack table header")
    header = table[1].split("|")
    if len(header) != 5 or " ".join(header[1:4]).split() != list(_COLUMNS):
        raise ValueError("Unexpected stack table column order")
    subjects: dict[str, dict[str, int]] = {}
    reported_means: list[str] | None = None
    for line in table[2:]:
        label, *groups = line.split("|")
        label = label.strip().removesuffix("*")
        if label != "MEAN" and not label.startswith("sub-"):
            continue
        if len(groups) != 4:
            raise ValueError(f"Invalid stack table row: {label}")
        values = [value for group in groups[:3] for value in group.split()]
        if len(values) != len(_COLUMNS):
            raise ValueError(f"Invalid stack table column count: {label}")
        if label == "MEAN":
            if reported_means is not None:
                raise ValueError("Duplicate stack mean row")
            reported_means = values
            continue

        if label not in _SUBJECTS or label in subjects:
            raise ValueError(f"Unexpected or duplicate stack participant: {label}")
        counts = {}
        for arm, value in zip(_COLUMNS.values(), values, strict=True):
            rate = float(value)
            if not 0 <= rate <= 1:
                raise ValueError(f"Invalid Top-1 rate: {label}/{arm}")
            hits = round(rate * _IMAGES_PER_SUBJECT)
            if abs(rate - hits / _IMAGES_PER_SUBJECT) > _ROUNDING_TOLERANCE:
                raise ValueError(f"Top-1 rate does not match an integer count: {label}/{arm}")
            counts[arm] = hits
        subjects[label] = counts

    if set(subjects) != set(_SUBJECTS) or reported_means is None:
        raise ValueError("Incomplete stack participant or mean rows")
    for arm, reported in zip(_COLUMNS.values(), reported_means, strict=True):
        measured = mean(row[arm] for row in subjects.values()) / _IMAGES_PER_SUBJECT
        if not isclose(measured, float(reported), abs_tol=_ROUNDING_TOLERANCE, rel_tol=0.0):
            raise ValueError(f"Stack mean differs from its participant rows: {arm}")
    return subjects


def _verify_rank_counts(log: str, totals: dict[str, int], total_images: int) -> None:
    _, heading, rest = log.partition("=== 8,000-Sample Paired Rank Audit")
    if not heading:
        raise ValueError("Missing published stack rank audit")
    rank_rows = rest.partition("Paired Flips:")[0]
    pattern = r"^[ \t]*(O\w+)[^|\n]*\| Top-1=(\d\.\d{4})\s+\(\s*(\d+)/(\d+)\)"
    seen = set()
    for arm, rate, hits, images in re.findall(pattern, rank_rows, re.MULTILINE):
        if arm not in _RANK_ARMS:
            continue
        exact_rate = totals[arm] / total_images
        if (
            arm in seen
            or int(images) != total_images
            or int(hits) != totals[arm]
            or abs(float(rate) - exact_rate) > _ROUNDING_TOLERANCE
        ):
            raise ValueError(f"Published rank count differs from participant rows: {arm}")
        seen.add(arm)
    if seen != _RANK_ARMS:
        raise ValueError("Incomplete published stack rank counts")


def verify_stack_results(root: Path) -> dict:
    """Check the table arithmetic and compute ten-participant paired t intervals."""
    log = (root / STACK_LOG).read_text(encoding="utf-8-sig")
    counts = _hit_counts(log)
    total_images = len(_SUBJECTS) * _IMAGES_PER_SUBJECT
    totals = {arm: sum(row[arm] for row in counts.values()) for arm in _COLUMNS.values()}
    _verify_rank_counts(log, totals, total_images)
    paired = []
    for treatment, control in (
        ("O12_noBigG", "O12_ctrl_enc"),
        ("O13", "O13_ctrl_xsub"),
        ("O14_noBigG", "O13"),
        ("O14_noBigG", "O10"),
    ):
        gaps = [
            100 * (row[treatment] - row[control]) / _IMAGES_PER_SUBJECT for row in counts.values()
        ]
        difference = mean(gaps)
        width = _T95_DF9 * stdev(gaps) / sqrt(len(gaps))
        paired.append(
            {
                "treatment": treatment,
                "control": control,
                "difference_pp": difference,
                "ci95_pp": [difference - width, difference + width],
                "participants_improved": sum(gap > 0 for gap in gaps),
            }
        )
    return {
        "source": STACK_LOG.as_posix(),
        "participants": len(_SUBJECTS),
        "images_per_participant": _IMAGES_PER_SUBJECT,
        "top1_hits": totals,
        "top1": {arm: hits / total_images for arm, hits in totals.items()},
        "paired": paired,
        "scope": "Arithmetic and descriptive participant-paired intervals for the fixed gallery.",
    }
