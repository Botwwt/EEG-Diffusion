"""Compute the two potential images for each frozen EEG query using the native sampler."""

import argparse
import csv
import shutil
from copy import copy
from pathlib import Path

import numpy as np
from register_fresh_photo_panel import read, sha
from render_fresh_mix_images import PANELS
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA, check, draw_packets, packet

LOCK = "14fb4ac078af918dc8f3dd033248adeaa028c22dafd3c4907c68e5528f902f3d"
SUB = "sub-04"
FLAGS = ("original", "prior", "matched", "wrong")


def controls(root, case):
    spec = read(root / "protocol.json")
    design = read(root / "worker.json")
    check(
        sha(root / "protocol.json") == LOCK == design["protocol_sha256"], "Prompt protocol differs"
    )
    data = packet(root / "input", SUB, case)
    path = root / "flags" / (case + ".npz")
    check(
        sha(path) == design["flags"][case] and data[-1] == design["conditions"][case],
        "Changed directions or flags",
    )
    with np.load(path, allow_pickle=False) as z:
        check(
            all(
                np.array_equal(z[k], v)
                for k, v in zip(
                    ("rows", "folds", "predicted_words", "donor_rows", "support_rows"),
                    data[:5],
                    strict=True,
                )
            )
            and np.array_equal(z["seeds"], 923761 + data[0]),
            "Prompt flags do not match EEG query",
        )
        flags = {k: z["flags_" + k] for k in FLAGS}
        check(
            all(v.shape == (120,) and v.dtype == np.bool_ for v in flags.values())
            and not flags["original"].any()
            and flags["prior"].sum() == flags["matched"].sum(),
            "Invalid controls",
        )
    return spec, design, data, flags


def potential(a):
    spec, design, data, _ = controls(a.root, a.panel)
    check(
        sha(a.dictionary) == DICTIONARY and sha(a.adapter) == ADAPTER_SHA,
        "Changed native generator assets",
    )
    # The underlying sampler sees the original concept IDs, ordered directions and seeds in both runs.
    # Changing the displayed dictionary name changes only the strings passed to the text encoder.
    words = set(data[2])
    with a.dictionary.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        entries = list(reader)
    check(
        len(entries) == 1854 and words <= {r["unique_id"] for r in entries}, "Incomplete vocabulary"
    )
    root = a.root
    sourced = root / "input"
    for k in ("native_audit", "texture"):
        cfg = copy(a)
        cfg.subject = SUB
        cfg.batch_size = 8
        cfg.root = root / k
        cfg.root.mkdir(exist_ok=True, parents=True)
        for name in ("design.json", "radius.json"):
            shutil.copyfile(sourced / name, cfg.root / name)
        if k == "texture":
            cfg.dictionary = cfg.root / "prompts.csv"
            with cfg.dictionary.open("w", encoding="utf-8", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=cols, lineterminator="\n")
                writer.writeheader()
                for row in entries:
                    v = dict(row)
                    if row["unique_id"] in words:
                        v["word"] += spec["suffix"]
                    writer.writerow(v)
        n = 8 if k == "native_audit" else 120
        pack = (*[v[:n] for v in data[:5]], {"own": data[5]["own"][:n]}, data[6], data[7])
        draw_packets(cfg, {a.panel: pack})
        settings = read(cfg.root / "renders" / a.panel / SUB / "settings.json")
        check(
            settings["turbo_sha256"] == TURBO
            and settings["adapter_sha256"] == ADAPTER_SHA
            and settings["seeds"] == (923761 + data[0][:n]).tolist(),
            "Sampler differed",
        )
        if k == "native_audit":
            check(
                settings["jpeg_sha256"]["own"] == design["baseline_jpeg_sha256"][a.panel][:8],
                "Native batch differs from released pixels",
            )


def name(a):
    from eval_instance_starts import load_visual_scorer
    from run_prospective_readout import frozen_npz
    from score_official_objects import load_dictionary, load_tokenizer, text_bank, unit
    from score_reconstruction_jpegs import CLIP_SHA, raw_features
    from score_support_view import pictures

    _, _, data, _ = controls(a.root, a.panel)
    base = a.root / "texture/renders" / a.panel / SUB
    file = base / "settings.json"
    cfg = read(file)
    check(
        sha(a.dictionary) == DICTIONARY and cfg["packet_sha256"] == data[-1],
        "Wrong naming vocabulary or images",
    )
    model, h = load_visual_scorer(a.archive.resolve())
    check(h == CLIP_SHA, "Changed naming encoder")
    names, common, inv, _, _ = load_dictionary(a.dictionary)
    text = text_bank(
        model,
        load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer"),
        common,
        16,
    )
    imgs = pictures(base / "images/own", cfg["jpeg_sha256"]["own"])
    s = (unit(raw_features(model, imgs.__getitem__, 120, 16)) @ text.T)[:, inv]
    frozen_npz(
        base / "naming.npz",
        dict(
            rows=data[0], names=np.array(names), scores=s, model_sha256=h, settings_sha256=sha(file)
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("render", "name"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--panel", choices=PANELS, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    a = p.parse_args()
    (potential if a.action == "render" else name)(a)
