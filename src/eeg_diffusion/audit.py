"""Integrity and arithmetic checks; no EEG, weight or image-feature replay."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON artifact, accepting a legacy byte-order mark."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_manifest(root: Path, manifest: dict[str, Any]) -> int:
    """Verify unique, repository-contained file paths against their SHA-256."""
    root = root.resolve()
    failures: list[str] = []
    seen: set[str] = set()
    for record in manifest["files"]:
        relative = record["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes the repository")
        if relative in seen:
            raise ValueError(f"Duplicate manifest path: {relative}")
        seen.add(relative)
        if not path.is_file():
            failures.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(relative)
    if failures:
        raise ValueError("Artifact hash mismatch: " + ", ".join(failures))
    return len(seen)


def verify_release(root: str | Path) -> dict[str, Any]:
    """Check frozen sources, released evidence, and two displayed contrasts.

    Intervals and report-transcribed values are not recomputed from aggregates.
    Verification of these files cannot establish upstream data independence.
    """
    root = Path(root).resolve()
    source_count = verify_manifest(root, read_json(root / "provenance/source_manifest.json"))
    evidence_count = verify_manifest(root, read_json(root / "provenance/evidence_manifest.json"))
    summary = read_json(root / "results/summary.json")
    retrieval = read_json(root / "results/retrieval.json")
    expected = [row for row in retrieval["evaluations"] if row["role"] == "full_raw"]
    if summary["retrieval"]["rows"] != expected:
        raise ValueError("Summary disagrees with released source aggregates")
    grouped: dict[str, list[float]] = {}
    for row in expected:
        if row["split"] != "official_test200":
            raise ValueError("Mixed evaluation splits")
        grouped.setdefault(row["method"], []).append(row["top1"])
    means = {name: sum(values) / len(values) for name, values in grouped.items()}
    delta_pp = 100 * (means["middle12_joint"] - means["final_joint"])

    image = read_json(root / "results/image_pairing.json")
    arms = {row["arm"]: row for row in image["final"]}
    paired = next(
        item
        for item in image["paired_comparisons"]
        if item["control"] == "native_full_target_cyclic_shift"
    )
    delta = (
        arms["native_full_target"]["DINO_2AFC"]
        - arms["native_full_target_cyclic_shift"]["DINO_2AFC"]
    )
    if not math.isclose(delta, paired["metrics"]["DINO_2AFC"]["mean_difference"], abs_tol=1e-12):
        raise ValueError("Paired image difference inconsistent")
    return {
        "source_assets_verified": source_count,
        "evidence_files_verified": evidence_count,
        "retrieval_seed_means": means,
        "retrieval_difference_pp": delta_pp,
        "image_pairing_difference": delta,
        "scope": "File integrity and aggregate arithmetic; no EEG/weight/image-feature replay.",
    }
