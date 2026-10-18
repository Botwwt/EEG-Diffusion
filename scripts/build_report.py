"""Rebuild the HTML and source-data CSV without training or inference."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def artifacts() -> dict[Path, str]:
    """Render fixed templates from released aggregates, with no external calls."""
    summary = load("summary.json")
    pairing = load("image_pairing.json")
    comparison = next(
        item
        for item in pairing["paired_comparisons"]
        if item["control"] == "native_full_target_cyclic_shift"
    )["metrics"]["DINO_2AFC"]
    interval = comparison["paired_image_bootstrap95"]
    means = {
        method: sum(row["top1"] for row in summary["retrieval"]["rows"] if row["method"] == method)
        / 2
        for method in ["final_fixed", "final_joint", "middle12_joint"]
    }
    replacements = {
        "{{PAIR_DELTA}}": f"{comparison['mean_difference']:+.4f}",
        "{{PAIR_CI}}": f"[{interval[0]:+.4f}, {interval[1]:+.4f}]",
        "{{RETRIEVAL_MEAN}}": f"{100 * means['middle12_joint']:.1f}",
        "{{RETRIEVAL_DELTA}}": f"{100 * (means['middle12_joint'] - means['final_joint']):.1f}",
    }
    page = (ROOT / "scripts/report.template.html").read_text(encoding="utf-8")
    for token, value in replacements.items():
        page = page.replace(token, value)
    if "{{" in page:
        raise ValueError("Unresolved report template value")

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "experiment",
            "arm",
            "training_seed",
            "metric",
            "value",
            "unit",
            "ci95_lower",
            "ci95_upper",
            "uncertainty_scope",
        ]
    )
    for row in summary["retrieval"]["rows"]:
        writer.writerow(
            [
                "retrieval_mean80",
                row["method"],
                row["seed"],
                "Top-1",
                row["top1"],
                "proportion",
                "",
                "",
                "Individual fixed training seed; not a confidence interval",
            ]
        )
    for row in pairing["final"]:
        writer.writerow(
            [
                "image_pairing_32",
                row["arm"],
                "fixed single seed",
                "DINOv3_2AFC",
                row["DINO_2AFC"],
                "proportion",
                "",
                "",
                "Aggregate mean; intervals stored in image_pairing.json",
            ]
        )
    for row in summary["v_parameterization"]["comparisons"]:
        writer.writerow(
            [
                "v_parameterization_32",
                row["comparison"],
                "two fixed seeds",
                "DINO2AFC_difference",
                row["difference"],
                "proportion",
                *row["ci95"],
                "Archived descriptive paired interval; transcribed precision",
            ]
        )
    return {ROOT / "docs/index.html": page, ROOT / "results/figure_data.csv": buffer.getvalue()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    for path, content in artifacts().items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                failures.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if failures:
        raise SystemExit("Stale report artifacts: " + ", ".join(failures))
    print(
        "Report artifacts match released data." if args.check else "Built HTML and source-data CSV."
    )


if __name__ == "__main__":
    main()
