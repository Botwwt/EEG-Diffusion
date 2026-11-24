"""Reproduce the fixed native H14 projection radius from 32 non-panel training images."""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_within_photo_embedding import images
from PIL import Image
from readout_external_images import MODEL_SHA, h14_pixels, load_h14
from render_within_photo_ip import check, sha


def run(args):
    import torch

    plan = json.loads(args.design.read_bytes())
    ids = np.asarray(plan["generic_radius_rows"])
    panel_classes = []
    for panel in ("registered", "replicate"):
        reg = args.packets / "sub-04" / panel / "conditions_512.npz"
        check(sha(reg) == plan["packets_sha256"][f"sub-04/{panel}"], "Locked packet differs")
        with np.load(reg, allow_pickle=False) as z:
            classes = z["training_classes"]
            panel_classes.extend(np.unique(z["rows"] // 10))
    chosen = np.sort(
        np.random.default_rng(plan["radius_seed"]).choice(
            (classes[:, None] * 10 + np.arange(10)).ravel(), 32, replace=False
        )
    )
    check(
        np.array_equal(chosen, ids) and not np.intersect1d(ids // 10, panel_classes).size,
        "Unexpected radius calibration photos",
    )
    check(
        sha(args.train_zip) == "b91edf1a316142406a2ae28bdbb673712f1d5c38f57c0aec9dcfb5da446cabf1",
        "Wrong training photograph archive",
    )
    with (args.source / "atm_train_rows.csv").open(encoding="utf-8-sig", newline="") as f:
        table = list(csv.DictReader(f))
    bank = images(args.source)
    pixels = []
    with ZipFile(args.train_zip) as z:
        for i in ids:
            record = table[i]
            check(int(record["train_row"]) == i, "Wrong source row")
            with Image.open(
                io.BytesIO(
                    z.read(
                        f"training_images/{record['official_folder']}/{record['stimulus_stem']}.jpg"
                    )
                )
            ) as im:
                pixels.append(h14_pixels(im))
    model, mu, sd = load_h14(args.assets, "cuda")
    vectors = []
    with torch.inference_mode():
        for lo in range(0, len(ids), 4):
            x = (
                torch.from_numpy(np.stack(pixels[lo : lo + 4]))
                .permute(0, 3, 1, 2)
                .to("cuda")
                .float()
                / 255
            )
            vectors.extend(
                model(pixel_values=((x - mu) / sd).to(model.dtype))
                .image_embeds.float()
                .cpu()
                .numpy()
            )
    v = np.stack(vectors)
    norms = np.linalg.norm(v, axis=1)
    cosines = ((v / norms[:, None]) * bank[ids]).sum(1)
    check(
        np.median(cosines) > 0.95 and np.min(cosines) > 0.85,
        "Raw encoder directions do not match the published H14 training images",
    )
    result = dict(
        median_norm=float(np.median(norms)),
        generic_rows=ids.tolist(),
        norms=norms.tolist(),
        cosines_to_author=cosines.tolist(),
        model_sha256=MODEL_SHA,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print("Training-image radius", result["median_norm"], "median cosine", np.median(cosines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        (
            "design",
            "research/eeg_stack_20261112/results/object_fidelity/within_photo_render_design.json",
        ),
        ("packets", "outputs/photo-structure/within_photo"),
        ("assets", "outputs/external-object-anchors/assets"),
        ("source", "outputs/official-object-readout/sub-03"),
        ("train-zip", "outputs/photo-structure/training_images.zip"),
    ):
        p.add_argument("--" + name, type=Path, default=Path(default))
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
