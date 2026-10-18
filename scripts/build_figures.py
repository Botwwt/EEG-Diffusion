"""Export the report's statistical figures as editable SVG and vector PDF.

Data are released aggregates. Dots in panel a are fixed training seeds; they are
not confidence intervals. No unavailable query-level data are reconstructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = ["results/summary.json", "results/image_pairing.json"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_figures() -> None:
    """Check recorded figure/data hashes without relying on renderer versions."""
    manifest = json.loads((ROOT / "provenance/figure_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"] + manifest["data"] + manifest["builder"]:
        path = (ROOT / record["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Figure or source changed: {record['path']}")
    print("Figure and source-data hashes verified.")


def build_figures() -> None:
    """Render at 174-mm width, with panel labels and embedded PDF fonts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager

    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        font_family = "Arial"
    except ValueError:
        font_family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "svg.fonttype": "none",
            "svg.hashsalt": "eeg-diffusion-v0.2",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "savefig.facecolor": "white",
        }
    )
    summary = json.loads((ROOT / DATA_FILES[0]).read_text(encoding="utf-8"))
    pairing = json.loads((ROOT / DATA_FILES[1]).read_text(encoding="utf-8"))
    destination = ROOT / "docs/assets"
    destination.mkdir(exist_ok=True)
    exported: list[Path] = []

    def save(figure, name: str) -> None:
        for extension in ["svg", "pdf"]:
            target = destination / f"{name}.{extension}"
            metadata = (
                {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None}
            )
            figure.savefig(target, metadata=metadata)
            if extension == "svg":
                text = (
                    "\n".join(
                        line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()
                    )
                    + "\n"
                )
                target.write_text(text, encoding="utf-8", newline="\n")
            exported.append(target)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 79 / 25.4))
    figure.subplots_adjust(left=0.195, right=0.965, bottom=0.24, top=0.80, wspace=0.97)
    labels = ["Final layer\nfixed head", "Final layer\njoint head", "Layer 12\njoint head"]
    methods = ["final_fixed", "final_joint", "middle12_joint"]
    ax = axes[0]
    for index, method in enumerate(methods):
        values = [
            100 * row["top1"]
            for row in sorted(summary["retrieval"]["rows"], key=lambda r: r["seed"])
            if row["method"] == method
        ]
        y = 2 - index
        ax.plot(values, [y + 0.09, y - 0.09], color="#a8b2b8", linewidth=0.7, zorder=1)
        ax.scatter(
            values[0],
            y + 0.09,
            s=22,
            marker="o",
            color="#35677c",
            linewidths=0.7,
            label="Seed 1" if index == 0 else None,
            zorder=3,
        )
        ax.scatter(
            values[1],
            y - 0.09,
            s=22,
            marker="o",
            facecolor="white",
            edgecolor="#35677c",
            linewidths=0.9,
            label="Seed 2" if index == 0 else None,
            zorder=3,
        )
        ax.text(max(values) + 4, y, f"{values[0]:g} / {values[1]:g}", va="center", fontsize=7)
    ax.set(
        yticks=[2, 1, 0],
        yticklabels=labels,
        xlim=(0, 112),
        ylim=(-0.5, 2.5),
        xticks=[0, 50, 100],
        xlabel="EEG retrieval Top-1 (%)",
    )
    ax.tick_params(axis="y", length=0, pad=8)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.04),
        ncols=2,
        frameon=False,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0,
    )
    ax.text(-0.68, 1.26, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)

    ax = axes[1]
    arms = {row["arm"]: row for row in pairing["final"]}
    conditions = ["native_full_target", "constant_training_mean", "native_full_target_cyclic_shift"]
    labels = ["Correct EEG", "Training mean", "Shuffled EEG"]
    colors = ["#35677c", "#8d979b", "#8d979b"]
    for index, (condition, color) in enumerate(zip(conditions, colors, strict=True)):
        value = arms[condition]["DINO_2AFC"]
        ax.scatter(value, 2 - index, s=26, color=color, zorder=3)
        ax.text(value + 0.055, 2 - index, f"{value:.4f}", va="center", fontsize=7)
    ax.set(
        yticks=[2, 1, 0],
        yticklabels=labels,
        xlim=(0, 1.12),
        ylim=(-0.5, 2.5),
        xticks=[0, 0.5, 1],
        xlabel="Generated-image DINOv3 2AFC",
    )
    ax.tick_params(axis="y", length=0, pad=8)
    ax.text(-0.69, 1.26, "b", transform=ax.transAxes, fontweight="bold", fontsize=8)
    ax.text(-0.02, 1.08, "32 fixed queries · one trained model", transform=ax.transAxes, fontsize=7)
    for ax in axes:
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_color("#7a8388")
    save(figure, "figure-1")

    figure, ax = plt.subplots(figsize=(174 / 25.4, 61 / 25.4))
    figure.subplots_adjust(left=0.37, right=0.94, bottom=0.30, top=0.88)
    rows = summary["v_parameterization"]["comparisons"]
    for index, row in enumerate(rows):
        y = len(rows) - index - 1
        low, high = row["ci95"]
        value = row["difference"]
        ax.errorbar(
            value,
            y,
            xerr=np.array([[value - low], [high - value]]),
            fmt="o",
            color="#35677c" if index else "#262d31",
            markersize=4,
            linewidth=0.9,
            capsize=2.5,
            capthick=0.8,
        )
    ax.axvline(0, color="#aeb6ba", linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set(
        yticks=[2, 1, 0],
        yticklabels=[row["comparison"] for row in rows],
        xlim=(-0.15, 0.41),
        ylim=(-0.6, 2.6),
        xticks=[-0.1, 0, 0.1, 0.2, 0.3, 0.4],
        xlabel="Paired difference in generated-image DINO2AFC",
    )
    ax.tick_params(axis="y", length=0, pad=12)
    ax.spines["bottom"].set_color("#7a8388")
    save(figure, "figure-2")
    manifest = {
        "renderer": f"matplotlib {matplotlib.__version__}",
        "font_family": font_family,
        "body_font_pt": 7,
        "panel_label_pt": 8,
        "width_mm": 174,
        "pdf_fonts": "Embedded TrueType (type 42)",
        "svg_text": "Editable text; no rasterized data marks",
        "uncertainty": "Figure 1a: fixed seeds; 1b: aggregate points. Figure 2: archived paired 95% intervals.",
        "data": [{"path": name, "sha256": digest(ROOT / name)} for name in DATA_FILES],
        "builder": [{"path": "scripts/build_figures.py", "sha256": digest(Path(__file__))}],
        "files": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in exported
        ],
    }
    (ROOT / "provenance/figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("Exported two statistical figures as SVG and PDF.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_figures()
    else:
        build_figures()


if __name__ == "__main__":
    main()
