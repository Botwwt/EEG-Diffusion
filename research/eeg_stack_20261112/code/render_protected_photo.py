"""Redraw the three registered mix13 conditions and name the sealed JPEGs offline."""

import argparse
from pathlib import Path

import numpy as np
from register_fresh_photo_panel import read, sha
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA, check, draw_packets

LOCK = "db55d06bf5b913f20c07e64592f3366169f82da352802faeee7031334c076267"
SUB, PANEL = "sub-04", "mix13"
ARMS = ("native", "own", "wrong")


def packet(root):
    meta = read(root / "design.json")
    path = root / "conditions.npz"
    check(
        sha(root / "protocol.json") == meta["registration_sha256"] == LOCK
        and sha(path) == meta["packet_sha256"]
        and sha(root / "radius.json") == meta["radius_sha256"]
        and sha(root / "native_settings.json")
        == read(root / "protocol.json")["source_sha256"]["native_settings"],
        "Changed protected study",
    )
    original = read(root / "native_settings.json")
    with np.load(path, allow_pickle=False) as z:
        check(
            set(z.files) == {"rows", "folds", "words", "donors", "support", "radius", *ARMS},
            "Unexpected condition fields",
        )
        rows, folds, words, donors, support = (
            z[k] for k in ("rows", "folds", "words", "donors", "support")
        )
        items = {k: z[k] for k in ARMS}
        radius = float(z["radius"])
    check(
        original["batch_size"] == 8
        and original["subject"] == SUB
        and original["panel"] == PANEL
        and original["turbo_sha256"] == TURBO
        and original["adapter_sha256"] == ADAPTER_SHA
        and original["dictionary_sha256"] == DICTIONARY
        and len(set(rows)) == 120
        and support.shape == (120, 8)
        and set(rows) == set(donors)
        and all(
            np.array_equal(original[k], v)
            for k, v in [
                ("rows", rows),
                ("folds", folds),
                ("predicted_words", words),
                ("donor_rows", donors),
                ("support_rows", support),
                ("seeds", 923761 + rows),
            ]
        )
        and original["radius"] == radius == read(root / "radius.json")["median_norm"]
        and original["driver_sha256"] == sha(Path(__file__).with_name("render_within_photo_ip.py")),
        "Archived generator or row order changed",
    )
    lookup = {r: i for i, r in enumerate(rows)}
    idx = np.array([lookup[d] for d in donors])
    check(
        np.array_equal(words, words[idx])
        and np.array_equal(folds, folds[idx])
        and np.array_equal(support, support[idx])
        and all(
            len(set(s)) == 8 and set(s) <= set(rows[folds != f])
            for f, s in zip(folds, support, strict=True)
        )
        and np.array_equal(items["own"][rows == donors], items["wrong"][rows == donors]),
        "Invalid within-word/within-fold control",
    )
    check(
        all(
            v.dtype == np.float32
            and v.shape == (120, 1024)
            and np.allclose(np.linalg.norm(v, axis=1), 1, atol=2e-6, rtol=0)
            for v in items.values()
        ),
        "Invalid image directions",
    )
    return rows, folds, words, donors, support, items, radius, sha(path)


def verified(root):
    rows, folds, words, donors, support, items, rad, h = packet(root)
    folder = root / "renders" / PANEL / SUB
    cfg = read(folder / "settings.json")
    historic = read(root / "native_settings.json")
    check(
        all(
            cfg[k] == historic[k]
            for k in (
                "batch_size",
                "subject",
                "panel",
                "adapter_sha256",
                "turbo_sha256",
                "dictionary_sha256",
                "driver_sha256",
                "radius_sha256",
                "rows",
                "folds",
                "predicted_words",
                "donor_rows",
                "support_rows",
                "seeds",
                "radius",
            )
        )
        and cfg["packet_sha256"] == h
        and cfg["design_sha256"] == sha(root / "design.json")
        and set(cfg["jpeg_sha256"]) == set(ARMS)
        and cfg["jpeg_sha256"]["native"] == historic["jpeg_sha256"]["own"],
        "Historical settings or per-row JPEG replay failed",
    )
    for arm in ARMS:
        hashes = cfg["jpeg_sha256"][arm]
        check(
            len(hashes) == 120
            and all(
                sha(folder / "images" / arm / f"{i:03d}.jpg") == v for i, v in enumerate(hashes)
            ),
            "Incomplete or changed JPEGs",
        )
    for left, right in (("own", "native"), ("wrong", "native"), ("own", "wrong")):
        same = np.all(items[left] == items[right], axis=1)
        check(
            np.array_equal(
                np.asarray(cfg["jpeg_sha256"][left])[same],
                np.asarray(cfg["jpeg_sha256"][right])[same],
            ),
            "Equal directions yielded different pixels",
        )
    return folder, cfg


def name(a):
    from eval_instance_starts import load_visual_scorer
    from run_prospective_readout import frozen_npz
    from score_official_objects import load_dictionary, load_tokenizer, text_bank, unit
    from score_reconstruction_jpegs import CLIP_SHA, raw_features
    from score_support_view import pictures

    folder, cfg = verified(a.root)
    net, h = load_visual_scorer(a.archive.resolve())
    check(h == CLIP_SHA and sha(a.dictionary) == DICTIONARY, "Naming assets changed")
    vocab, common, inv, _, _ = load_dictionary(a.dictionary)
    text = text_bank(
        net, load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer"), common, 16
    )
    arrays = {}
    for arm in ARMS:
        pics = pictures(folder / "images" / arm, cfg["jpeg_sha256"][arm])
        arrays[arm] = (unit(raw_features(net, pics.__getitem__, 120, 16)) @ text.T)[:, inv]
    frozen_npz(
        folder / "naming.npz",
        dict(
            names=np.array(vocab),
            model_sha256=h,
            settings_sha256=sha(folder / "settings.json"),
            **arrays,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "render", "name"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument("--adapter", type=Path)
    a = p.parse_args()
    data = packet(a.root)
    if a.action == "render":
        check(
            sha(a.dictionary) == DICTIONARY and sha(a.adapter) == ADAPTER_SHA,
            "Incorrect generator assets",
        )
        a.batch_size, a.subject = 8, SUB
        draw_packets(a, {PANEL: data})
        verified(a.root)
    if a.action == "name":
        name(a)
