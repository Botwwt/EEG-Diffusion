"""Render and name one immutable EEG or diagnostic input packet on an offline GPU."""

import argparse
from pathlib import Path

import numpy as np
from register_fresh_photo_panel import read, sha
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA, check, draw_packets

LOCK = "abc04a2c4e64136aba49b5d858a66a5a392da72f78f9817bc5e4386f40dd251d"
SUB = "sub-04"
PANEL = "personal12"
ARMS = {"eeg": ("native", "matched", "prior", "wrong"), "oracle": ("reference", "foil")}


def verify(root, stage):
    meta = read(root / "design.json")
    file = root / "conditions.npz"
    check(
        sha(root / "protocol.json") == meta["registration_sha256"] == LOCK
        and meta["stage"] == stage
        and meta["packet_sha256"] == sha(file)
        and meta["radius_sha256"] == sha(root / "radius.json"),
        "Different sealed packet",
    )
    with np.load(file, allow_pickle=False) as z:
        rows, folds, words, donors, support = (
            z[k] for k in ("rows", "folds", "words", "donors", "supports")
        )
        check(
            len(set(rows)) == 120
            and set(folds) == set(range(5))
            and support.shape == (120, 8)
            and len(donors) == 120
            and set(donors) <= set(rows),
            "Malformed row metadata",
        )
        # Donor matching can use a non-bijective rule; for this existing panel the direction map is frozen.
        lookup = {r: i for i, r in enumerate(rows)}
        j = np.array([lookup[r] for r in donors])
        check(
            np.array_equal(words, words[j])
            and np.array_equal(folds, folds[j])
            and np.array_equal(support, support[j]),
            "Wrong matched donors",
        )
        check(
            set(z.files)
            == {"rows", "folds", "words", "donors", "supports", "radius", *ARMS[stage]},
            "Unexpected arm or metadata in this worker",
        )
        values = {k: z[k] for k in ARMS[stage]}
        for v in values.values():
            check(
                v.shape == (120, 1024)
                and v.dtype == np.float32
                and np.allclose(np.linalg.norm(v, axis=1), 1, atol=2e-6, rtol=0),
                "Wrong visual direction",
            )
        radius = float(z["radius"])
        check(radius == read(root / "radius.json")["median_norm"], "Visual radius changed")
    return rows, folds, words, donors, support, values, radius, sha(file)


def render(a):
    data = verify(a.root, a.stage)
    check(
        sha(a.dictionary) == DICTIONARY and sha(a.adapter) == ADAPTER_SHA,
        "Wrong text or visual adapter",
    )
    a.batch_size = 8
    a.subject = SUB
    draw_packets(a, {PANEL: data})
    folder = a.root / "renders" / PANEL / SUB
    cfg = read(folder / "settings.json")
    plan = read(a.root / "design.json")
    check(
        all(cfg[k] == v for k, v in plan["generator"].items())
        and cfg["turbo_sha256"] == TURBO
        and cfg["seeds"] == (data[0] + 923761).tolist(),
        "Generator settings differ",
    )
    if a.stage == "eeg":
        check(cfg["jpeg_sha256"]["native"] == plan["native_hashes"], "Native images do not replay")
    return folder, cfg


def name(a):
    from eval_instance_starts import load_visual_scorer
    from run_prospective_readout import frozen_npz
    from score_official_objects import load_dictionary, load_tokenizer, text_bank, unit
    from score_reconstruction_jpegs import CLIP_SHA, raw_features
    from score_support_view import pictures

    verify(a.root, a.stage)
    folder = a.root / "renders" / PANEL / SUB
    cfg = read(folder / "settings.json")
    net, h = load_visual_scorer(a.archive.resolve())
    check(h == CLIP_SHA and sha(a.dictionary) == DICTIONARY, "Different naming assets")
    vocab, common, inv, _, _ = load_dictionary(a.dictionary)
    text = text_bank(
        net, load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer"), common, 16
    )
    arrays = {}
    for k in ARMS[a.stage]:
        pics = pictures(folder / "images" / k, cfg["jpeg_sha256"][k])
        arrays[k] = (unit(raw_features(net, pics.__getitem__, 120, 16)) @ text.T)[:, inv]
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
    p.add_argument("action", choices=("render", "name"))
    p.add_argument("--stage", choices=ARMS, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    a = p.parse_args()
    (render if a.action == "render" else name)(a)
