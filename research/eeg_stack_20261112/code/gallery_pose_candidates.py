"""Show all 24 objects at a fixed sub-04 row after ranking the archived eight-image bank."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from score_pose_candidates import PANELS, PUBLIC, check, read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=Path("outputs/photo-structure/h14_bank"))
    p.add_argument("--selection", type=Path, default=Path("outputs/pose-candidate"))
    p.add_argument("--references", type=Path, default=Path("outputs/object-fidelity"))
    p.add_argument("--out", type=Path, default=Path("research/eeg_stack_20261112/gallery"))
    a = p.parse_args()
    report = read(PUBLIC / "pose_candidate.json")
    check(
        sha(a.selection / "selections.json") == report["selection_seal_sha256"],
        "Wrong set of EEG selections",
    )
    seal = read(a.selection / "selections.json")
    for panel in PANELS:
        reg = read(a.references / (panel + "_ablation.json"))
        rows = np.array(reg["rows"])
        labels = np.array(reg["words"])[reg["truth"]]
        key = panel + "/sub-04"
        path = a.selection / key / "choices.npz"
        oracle = a.selection / key / "oracle_evaluation_only.npz"
        check(
            sha(path) == seal["cases"][key]
            and sha(oracle) == report["oracle_and_diversity"][key]["oracle_sha256"],
            "Changed visual selections",
        )
        with np.load(path, allow_pickle=False) as z, np.load(oracle, allow_pickle=False) as d:
            check(
                np.array_equal(z["rows"], rows) and np.array_equal(d["rows"], rows),
                "Gallery order differs",
            )
            words = z["words"]
            h = z["candidate_hashes"]
            arms = ["fixed0", "mean", "old_h14", "joint", "wrong", "oracle"]
            indices = np.stack([z[k] if k != "oracle" else d["index"] for k in arms])
        base = a.bank / "renders" / panel
        names = {r["sha256"]: r["path"] for r in read(base / "settings.json")["records"]}
        tile = 120
        width = len(arms) * tile + 10
        height = tile + 36
        out = Image.new("RGB", (2 * width, 6 * height), "white")
        draw = ImageDraw.Draw(out)
        for ordinal, w in enumerate(sorted(set(labels))):
            i = np.flatnonzero(rows == rows[labels == w].min()).item()
            left = (ordinal % 2) * width
            top = (ordinal // 2) * height
            draw.text(
                (left, top),
                f"Viewed: {w}  |  EEG: {words[i]}  |  sub-04 row {rows[i]}",
                fill="black",
            )
            captions = ["First", "Mean", "H14 EEG", "Joint EEG", "Wrong EEG", "DINO oracle"]
            for j, title in enumerate(captions):
                choice = indices[j, i]
                sig = h[i, choice]
                src = base / names[sig]
                check(sha(src) == sig, "Generated gallery image changed")
                with Image.open(src) as im:
                    out.paste(
                        im.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS),
                        (left + j * tile, top + 14),
                    )
                draw.text((left + j * tile, top + tile + 14), f"{title}  #{choice}", fill="black")
        dest = a.out / f"pose_candidate_sub04_{panel}.jpg"
        dest.parent.mkdir(exist_ok=True, parents=True)
        out.save(dest, "JPEG", quality=91, optimize=True)
        print(dest, sha(dest))


if __name__ == "__main__":
    main()
