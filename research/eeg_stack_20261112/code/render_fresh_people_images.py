"""Run a subject-specific paired generation and naming worker for the fresh-people replication."""

import argparse
import io
import tarfile
from copy import copy
from pathlib import Path

import numpy as np
from package_personal_confirmation import scientific_code
from register_fresh_photo_panel import read, sha
from render_fresh_mix_images import METHODS, PANELS
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA, check, draw_packets, packet
from run_prospective_readout import frozen_npz

SUBJECTS = ("sub-05", "sub-01")
PUB = Path(__file__).resolve().parents[1] / "results/object_fidelity"
IMAGE_LOCK = "25322f9614c5a4e380445e8f19a5d842e8eda7bcec7ed59b802ccc87d93db762"
BLIND_SHA = "d20939ac1fcf44320bdc5ea967e193726fbca81d7ade4513720cb6ac79d0b347"


def gate(protocol):
    image = protocol / "fresh_people_images_design.json"
    result = protocol / "fresh_people_words.json"
    check(sha(image) == IMAGE_LOCK, "Image protocol changed")
    d = read(image)
    check(sha(result) == d["word_results_sha256"], "Word score differs")
    r = read(result)
    p = r["populations"]["primary"]
    check(
        r["design_sha256"] == d["word_design_sha256"]
        and r["advance"]
        and set(p["paired"]) == {"legacy", "learned_wrong"}
        and all(u["ci95_pp"][0] > 0 for u in p["paired"].values())
        and all(p["by_subject"][s]["learned"] >= p["by_subject"][s]["legacy"] for s in SUBJECTS),
        "Word gate did not pass",
    )
    return r


def case(root, sub, menu):
    out = {}
    for m in METHODS:
        base = root / m
        rows, folds, words, donors, ss, items, radius, digest = packet(base, sub, menu)
        file = base / "evaluations" / sub / menu / "word_predictions.json"
        w = read(file)
        check(
            w["subject"] == sub
            and w["panel"] == menu
            and np.array_equal(w["rows"], rows)
            and np.array_equal(w["folds"], folds)
            and np.array_equal(np.array(w["words"])[w["predicted"]], words),
            "Wrong word predictions",
        )
        out[m] = rows, folds, words, donors, ss, {"own": items["own"]}, radius, digest
    old, new = out.values()
    same = old[2] == new[2]
    check(
        same.any()
        and np.array_equal(old[0], new[0])
        and np.array_equal(old[1], new[1])
        and old[6] == new[6]
        and sha(root / "legacy/radius.json") == sha(root / "learned/radius.json")
        and np.array_equal(old[4][same], new[4][same])
        and old[5]["own"][same].tobytes() == new[5]["own"][same].tobytes(),
        "Unpaired rows, scale or conditions",
    )
    return out


def folder(root, person, menu, method):
    return root / method / "renders" / menu / person


def verified(root, sub, menu, both):
    cfgs = {}
    for m, items in both.items():
        path = folder(root, sub, menu, m)
        c = read(path / "settings.json")
        rows, folds, words, donors, support, _, radius, signature = items
        check(
            c["subject"] == sub
            and c["panel"] == menu
            and c["packet_sha256"] == signature
            and c["design_sha256"] == sha(root / m / "design.json")
            and c["radius_sha256"] == sha(root / m / "radius.json")
            and c["driver_sha256"] == sha(Path(__file__).with_name("render_within_photo_ip.py"))
            and c["batch_size"] == 8
            and c["radius"] == radius
            and c["adapter_sha256"] == ADAPTER_SHA
            and c["turbo_sha256"] == TURBO
            and c["dictionary_sha256"] == DICTIONARY,
            "Generator or receipt changed",
        )
        check(
            all(
                np.array_equal(c[k], v)
                for k, v in dict(
                    rows=rows,
                    folds=folds,
                    predicted_words=words,
                    donor_rows=donors,
                    support_rows=support,
                    seeds=923761 + rows,
                ).items()
            ),
            "Query/support row changed",
        )
        h = c["jpeg_sha256"]
        check(
            set(h) == {"own"}
            and len(h["own"]) == len(rows) == 120
            and all(sha(path / "images/own" / f"{i:03d}.jpg") == v for i, v in enumerate(h["own"])),
            "Frozen JPEG validation failed",
        )
        cfgs[m] = c
    a, b = cfgs.values()
    same = np.array(a["predicted_words"]) == b["predicted_words"]
    identical = np.array(a["jpeg_sha256"]["own"]) == b["jpeg_sha256"]["own"]
    check(np.all(identical[same]), "Independent same-word draws failed")
    ix = np.flatnonzero(same)[np.argmin(np.array(a["rows"])[same])]
    return cfgs, dict(
        same_word=int(same.sum()),
        same_word_identical=int(identical[same].sum()),
        different_word=int((~same).sum()),
        different_word_identical=int(identical[~same].sum()),
        smallest_same_word_row=a["rows"][ix],
        rerender_sha256=a["jpeg_sha256"]["own"][ix],
    )


def clip(a, both):
    from eval_instance_starts import load_visual_scorer
    from score_official_objects import load_dictionary, load_tokenizer, text_bank, unit
    from score_reconstruction_jpegs import CLIP_SHA, raw_features
    from score_support_view import pictures

    cfg, parity = verified(a.root, a.subject, a.panel, both)
    model, digest = load_visual_scorer(a.archive.resolve())
    check(digest == CLIP_SHA and sha(a.dictionary) == DICTIONARY, "Changed naming assets")
    names, common, inverse, _, _ = load_dictionary(a.dictionary)
    tok = load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer")
    text = text_bank(model, tok, common, 16)
    arrays = {}
    for m in METHODS:
        base = folder(a.root, a.subject, a.panel, m)
        pics = pictures(base / "images/own", cfg[m]["jpeg_sha256"]["own"])
        arrays[m] = (unit(raw_features(model, pics.__getitem__, 120, 16)) @ text.T)[:, inverse]
        arrays[m + "_settings_sha256"] = sha(base / "settings.json")
    frozen_npz(
        a.root / "scores" / a.panel / "clip_logits.npz",
        dict(
            rows=np.array(cfg["legacy"]["rows"]),
            names=np.array(names),
            model_sha256=digest,
            dictionary_sha256=DICTIONARY,
            **arrays,
        ),
    )
    print(a.subject, a.panel, parity, flush=True)


def package(a):
    gate(PUB)
    members = scientific_code("render_fresh_people_images")
    members.update(
        {
            "protocol/" + n: PUB / n
            for n in ("fresh_people_images_design.json", "fresh_people_words.json")
        }
    )
    a.out.parent.mkdir(exist_ok=True, parents=True)
    temp = a.out.with_suffix(".partial")
    with tarfile.open(temp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, path in sorted(members.items()):
            buf = path.read_bytes()
            i = tarfile.TarInfo(name)
            i.size = len(buf)
            tar.addfile(i, io.BytesIO(buf))
    if a.out.exists():
        check(sha(temp) == sha(a.out), "Stored code archive differs")
        temp.unlink()
    else:
        temp.replace(a.out)
    print(a.out, sha(a.out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("package", "check", "render", "clip"))
    p.add_argument("--root", type=Path)
    p.add_argument("--subject", choices=SUBJECTS)
    p.add_argument("--panel", choices=PANELS)
    p.add_argument("--protocol", type=Path, default=PUB)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument(
        "--out", type=Path, default=Path("outputs/fresh-people/cluster/fresh_people_image_code.tar")
    )
    a = p.parse_args()
    if a.stage == "package":
        package(a)
        return
    check(a.root and a.subject and a.panel, "Participant, panel and condition root required")
    gate(a.protocol)
    both = case(a.root, a.subject, a.panel)
    if a.stage == "check":
        print(a.subject, a.panel, "conditions verified")
        return
    if a.stage == "clip":
        clip(a, both)
        return
    for m in METHODS:
        arg = copy(a)
        arg.root = a.root / m
        arg.batch_size = 8
        draw_packets(arg, {a.panel: both[m]})
    print(verified(a.root, a.subject, a.panel, both)[1])


if __name__ == "__main__":
    main()
