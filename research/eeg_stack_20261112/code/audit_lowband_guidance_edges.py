"""Post-render 512px sharpness diagnostic for support starts and single-step lowband guidance."""

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import ZIP_SHA, table
from PIL import Image
from register_fresh_photo_panel import freeze, read, sha
from render_official_words import crop512
from render_within_photo_ip import check
from score_reconstruction_jpegs import registered


def gradients(image):
    luma = np.asarray(image, dtype=float) @ np.array([0.299, 0.587, 0.114]) / 255.0
    return np.r_[np.diff(luma, axis=0).ravel(), np.diff(luma, axis=1).ravel()]


def describe(vals, labels):
    labels = np.asarray(labels)
    c, idx = np.unique(labels, return_inverse=True)
    count = np.bincount(idx)
    w = np.random.default_rng(260570).multinomial(len(c), [1 / len(c)] * len(c), size=10000)
    arms = list(vals)
    v = np.array([vals[a] for a in arms])
    by = np.stack([v[:, idx == i].sum(1) for i in range(len(c))])
    boot = (w @ by) / (w @ count)[:, None]
    i = arms.index("native")
    j = arms.index("own")
    return dict(
        means=dict(zip(arms, v.mean(1).tolist(), strict=True)),
        own_minus_native=dict(
            delta=float((v[j] - v[i]).mean()),
            ci95=np.quantile(boot[:, j] - boot[:, i], [0.025, 0.975]).tolist(),
        ),
    )


def run(a):
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    rec = table(a)
    z = a.data / "photo-structure/training_images.zip"
    check(sha(z) == ZIP_SHA, "Unexpected reference archive")
    output = {}
    with ZipFile(z) as ar:
        for sub, exp, arms in (
            ("sub-04", "support-view", ("native", "mean", "medoid", "own", "wrong")),
            ("sub-05", "lowband-guidance", ("native", "prior", "own", "wrong")),
        ):
            root = a.data / exp / sub
            src = (
                a.data
                / (
                    "trial-object-encoder/personal12"
                    if sub == "sub-04"
                    else f"trial-object-replication/{sub}"
                )
                / "render_inputs/renders/personal12"
                / sub
            )
            cfg = read(root / "renders/settings.json")
            ncfg = read(src / "settings.json")
            reg = registered(
                a.data / "object-head-confirmation/evaluation/references" / sub, "personal12"
            )
            with np.load(
                a.data
                / f"object-head-confirmation/evaluation/pixels/{sub}/original_cls/personal12.npz",
                allow_pickle=False,
            ) as cache:
                check(
                    np.array_equal(cache["rows"], reg["rows"])
                    and str(cache["registration_sha"]) == reg["sha"],
                    "Changed reference order",
                )
                source_hashes = cache["jpeg_sha256"]
            energy = {k: [] for k in ("original", *arms)}
            error = {k: [] for k in arms}
            sim = {k: [] for k in arms}
            for j, row in enumerate(reg["rows"]):
                r = rec[row]
                check(r["unique_id"] == reg["labels"][j], "Original word mismatch")
                raw = ar.read(f"training_images/{r['official_folder']}/{r['stimulus_stem']}.jpg")
                check(
                    hashlib.sha256(raw).hexdigest() == source_hashes[j], "Reference image changed"
                )
                with Image.open(io.BytesIO(raw)) as im:
                    g = gradients(crop512(im.convert("RGB")))
                ref_energy = float(np.sqrt(np.mean(g * g)))
                energy["original"].append(ref_energy)
                for k in arms:
                    if k == "native":
                        f = src / "images/own" / f"{j:03}.jpg"
                        h = ncfg["jpeg_sha256"]["own"][j]
                        if exp == "lowband-guidance":
                            check(
                                h == cfg["jpeg_sha256"]["native"][j],
                                "Native guidance pixels changed",
                            )
                    else:
                        f = root / "renders" / k / f"{j:03}.jpg"
                        h = cfg["jpeg_sha256"][k][j]
                    check(sha(f) == h, "Rendered photo changed")
                    with Image.open(f) as im:
                        check(im.size == (512, 512), "Wrong output resolution")
                        d = gradients(im.convert("RGB"))
                    if k == "native":
                        native = d
                    power = float(np.sqrt(np.mean(d * d)))
                    energy[k].append(power)
                    error[k].append(abs(power - ref_energy))
                    sim[k].append(
                        float((native @ d) / max(np.linalg.norm(native) * np.linalg.norm(d), 1e-14))
                    )
            output[sub] = dict(
                n=len(reg["rows"]),
                objects=len(np.unique(reg["labels"])),
                source_settings_sha256=sha(src / "settings.json"),
                edited_settings_sha256=sha(root / "renders/settings.json"),
                reference_mean_edge_rms=float(np.mean(energy.pop("original"))),
                edge_rms=describe(energy, reg["labels"]),
                absolute_reference_energy_error=describe(error, reg["labels"]),
                cosine_of_gradients_to_native=describe(sim, reg["labels"]),
            )
    freeze(
        a.out,
        dict(
            timing="Descriptive test specified after the primary results. Separate participants, menus and interventions.",
            scorer_sha256=sha(Path(__file__)),
            methods="Full 512px center crop. Luminance in [0,1], horizontal and vertical forward differences over every adjacent pair. Per-image RMS gradient magnitude, then average over 120 photos. Absolute difference from the original edge RMS and cosine to the native generated gradient vector are separate diagnostics. Larger edge energy may arise from texture, noise or compression. Seed 260570, 10000 whole-object paired draws.",
            by_subject=output,
        ),
    )
    print({s: d["edge_rms"]["means"] for s, d in output.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "research/eeg_stack_20261112/results/object_fidelity/lowband_guidance_edges.json"
        ),
    )
    run(p.parse_args())
