"""Isolate word and visual-condition errors using explicitly labeled reference-input diagnostics."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from render_within_photo_ip import check, draw_packets, packet, read, save, sha

ARMS = ("reference_word", "reference_image", "reference_both")


def prepare(a):
    from fit_support_structure import unit
    from fit_within_photo_embedding import images
    from score_reconstruction_jpegs import registered

    rows, folds, words, donors, support, v, radius, receipt = packet(a.base, a.subject, "fresh")
    reg = registered(a.references, "fresh")
    check(np.array_equal(rows, reg["rows"]), "Reference-photo row mismatch")
    pic = unit(images(a.source)[rows].astype(float)).astype(np.float32)
    a.root.mkdir(parents=True, exist_ok=True)
    dest = a.root / "diagnostic.npz"
    np.savez(
        dest,
        rows=rows,
        folds=folds,
        words=words,
        truth=reg["labels"],
        donors=donors,
        support=support,
        eeg=v["own"],
        reference=pic,
    )
    shutil.copyfile(a.base / "radius.json", a.root / "radius.json")
    save(
        a.root / "design.json",
        dict(
            subject=a.subject,
            study="Developmental capability check on the 120 already scored fresh-panel queries. Every new arm deliberately supplies a reference label, the H/14 embedding of the exact viewed picture, or both. None is an EEG reconstruction.",
            arms=ARMS,
            radius=radius,
            radius_sha256=sha(a.root / "radius.json"),
            registration_sha256=reg["sha"],
            eeg_packet_sha256=receipt,
            input_sha256=sha(dest),
            driver_sha256=sha(Path(__file__)),
        ),
    )


def render(a):
    d = read(a.root / "design.json")
    path = a.root / "diagnostic.npz"
    check(
        d["subject"] == a.subject
        and sha(path) == d["input_sha256"]
        and sha(a.root / "radius.json") == d["radius_sha256"]
        and d["driver_sha256"] == sha(Path(__file__)),
        "Diagnostic inputs changed",
    )
    packets = {}
    with np.load(path, allow_pickle=False) as z:
        for name in ARMS:
            reference_word = name in ("reference_word", "reference_both")
            reference_image = name in ("reference_image", "reference_both")
            matrix = z["reference"] if reference_image else z["eeg"]
            words = z["truth"] if reference_word else z["words"]
            check(
                np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5),
                "Invalid visual condition",
            )
            packets[name] = (
                z["rows"],
                z["folds"],
                words,
                z["donors"],
                z["support"],
                {name: matrix},
                d["radius"],
                d["input_sha256"],
            )
    draw_packets(a, packets)


def score(a):
    import zipfile

    from PIL import Image
    from refine_registered_color import metrics, thumbnail
    from score_fresh_photo import reference_pixels
    from score_object_panel import score_object_panel
    from score_within_photo_dino import FILES, encoder, originals, reference

    d = read(a.root / "design.json")
    check(
        sha(a.references / "fresh_ablation.json") == d["registration_sha256"], "Wrong registration"
    )
    locations = {"eeg_only": a.base / "renders/fresh" / a.subject}
    locations.update({name: a.root / "renders" / name / a.subject for name in ARMS})
    encode = encoder(a)
    tab = reference(a)
    with zipfile.ZipFile(a.originals) as archive:
        reg, ref, hashes = originals(a, encode, tab, archive, "fresh")
        targets, _ = reference_pixels(tab, archive, reg["rows"], hashes, set())
    truth = np.array(reg["words"])[reg["truth"]]
    eegwords = read(locations["eeg_only"] / "settings.json")["predicted_words"]
    correct = np.asarray(eegwords) == truth
    records = {}
    receipts = {}
    for name, loc in locations.items():
        settings = read(loc / "settings.json")
        arm = "own" if name == "eeg_only" else name
        check(
            settings["rows"] == reg["rows"]
            and np.array_equal(settings["seeds"], 923761 + np.asarray(reg["rows"]))
            and settings["packet_sha256"]
            == d["eeg_packet_sha256" if name == "eeg_only" else "input_sha256"],
            "Source order or condition differs",
        )
        words = truth if name in ("reference_word", "reference_both") else eegwords
        check(np.array_equal(settings["predicted_words"], words), "Unexpected prompts")
        pictures = []
        for i, digest in enumerate(settings["jpeg_sha256"][arm]):
            src = loc / "images" / arm / f"{i:03d}.jpg"
            check(sha(src) == digest, "Wrong output JPEG")
            with Image.open(src) as im:
                pictures.append(im.convert("RGB"))
        check(len(pictures) == 120, "Incomplete capability check")
        cos = encode(pictures).astype(float) @ ref.astype(float).T
        gallery = score_object_panel(cos, reg["rows"], truth)["per_photo"]
        pixel = np.stack(
            [metrics(thumbnail(im), target) for im, target in zip(pictures, targets, strict=True)]
        )
        records[name] = dict(
            intra10=np.array([v["intra10_top1"] for v in gallery]),
            rgb_rmse32=pixel[:, 1],
            luma_pearson32=pixel[:, 2],
        )
        receipts[name] = sha(loc / "settings.json")
    groups = {}
    for tag, mask in {
        "all": np.ones(120, bool),
        "eeg_word_correct": correct,
        "eeg_word_wrong": ~correct,
    }.items():
        stats = {}
        _, ind, counts = np.unique(truth[mask], return_inverse=True, return_counts=True)
        draws = np.random.default_rng(260526).integers(len(counts), size=(10000, len(counts)))
        for name, measures in records.items():
            stats[name] = {}
            for metric, raw in measures.items():
                diff = (raw - records["eeg_only"][metric])[mask]
                sums = np.bincount(ind, weights=diff, minlength=len(counts))
                stats[name][metric] = dict(
                    mean=float(raw[mask].mean()),
                    delta=float(diff.mean()),
                    paired_ci95=np.quantile(
                        sums[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]
                    ).tolist(),
                )
                if metric == "intra10":
                    stats[name][metric]["credit"] = float(raw[mask].sum())
        groups[tag] = dict(n=int(sum(mask)), arms=stats)
    baseline = read(a.base / "pixel-scores/fresh" / a.subject / "per_query.json")
    check(
        np.array_equal(
            records["eeg_only"]["intra10"], [r["intra10"]["own"] for r in baseline["images"]]
        ),
        "Baseline ten-photo scores must replay per row",
    )
    save(
        a.release,
        dict(
            subject=a.subject,
            design=d,
            settings_sha256=receipts,
            dino_sha256=FILES["model.safetensors"],
            groups=groups,
            interpretation="Only eeg_only reconstructs from EEG and permitted calibration. The remaining three use information about the held-out photo solely to localize loss of identity in this frozen pipeline. Intervals resample the represented objects within one participant.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "render", "score"])
    for key, default in dict(
        root="outputs/photo-conditioning-diagnostic",
        base="outputs/fresh-photo",
        references="outputs/fresh-photo/references",
        source="outputs/official-object-readout/sub-03",
        model="outputs/photo-structure/dino/facebook-dinov2-small",
        originals="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
        release="outputs/photo-conditioning-diagnostic/report.json",
    ).items():
        p.add_argument("--" + key, type=Path, default=Path(default))
    for key in ("archive", "adapter", "dictionary"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--subject", default="sub-04", choices=["sub-04"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int)
    a = p.parse_args()
    if a.batch_size is None:
        a.batch_size = 8 if a.stage == "render" else 16
    a.out = a.root
    {"prepare": prepare, "render": render, "score": score}[a.stage](a)
