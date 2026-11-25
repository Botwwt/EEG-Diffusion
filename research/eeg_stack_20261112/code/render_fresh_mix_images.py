"""Generate and byte-compare four menus for two frozen EEG word readouts."""

import argparse
from copy import copy
from pathlib import Path

import numpy as np
from render_within_photo_ip import check, draw_packets, packet, read, sha

PANELS = ("mix13", "mix14", "mix23", "mix24")
METHODS = ("legacy", "learned")
SUBJECT = "sub-04"
LOCK = "7fb74944ab647d9931a971c4814a09cd1650d84f890bbb1adcfc587c236a7c6d"


def condition_pair(root, panel):
    combos = {}
    for arm in METHODS:
        base = root / arm
        info = packet(base, SUBJECT, panel)
        rows, folds, words, donors, support, items, radius, digest = info
        file = base / "evaluations" / SUBJECT / panel / "word_predictions.json"
        if file.exists():
            w = read(file)
            check(
                np.array_equal(w["rows"], rows)
                and np.array_equal(w["folds"], folds)
                and np.array_equal(np.array(w["words"])[w["predicted"]], words),
                "Frozen word manifest disagrees",
            )
        combos[arm] = (rows, folds, words, donors, support, {"own": items["own"]}, radius, digest)
    old, new = [combos[arm] for arm in METHODS]
    same = old[2] == new[2]
    check(
        sha(root / "learned/radius.json") == sha(root / "legacy/radius.json")
        and old[6] == new[6]
        and np.array_equal(old[0], new[0])
        and np.array_equal(old[1], new[1])
        and same.any(),
        "Row pairing, scale or fold differs",
    )
    check(
        np.array_equal(old[4][same], new[4][same])
        and old[5]["own"][same].tobytes() == new[5]["own"][same].tobytes(),
        "Same predicted word changed the H14 input or eight supports",
    )
    return combos


def folder(root, arm, panel):
    return root / arm / "renders" / panel / SUBJECT


def verify_images(root, panel, combos):
    settings = {}
    for arm, data in combos.items():
        base = folder(root, arm, panel)
        cfg = read(base / "settings.json")
        rows, folds, words, donors, support, _, rad, digest = data
        check(
            all(
                np.array_equal(cfg[k], v)
                for k, v in dict(
                    rows=rows,
                    folds=folds,
                    predicted_words=words,
                    donor_rows=donors,
                    support_rows=support,
                    seeds=923761 + rows,
                ).items()
            )
            and cfg["radius"] == rad
            and cfg["packet_sha256"] == digest
            and cfg["subject"] == SUBJECT
            and cfg["panel"] == panel
            and cfg["design_sha256"] == sha(root / arm / "design.json")
            and cfg["radius_sha256"] == sha(root / arm / "radius.json")
            and cfg["driver_sha256"] == sha(Path(__file__).with_name("render_within_photo_ip.py")),
            "Changed generation recipe",
        )
        hashes = cfg["jpeg_sha256"]
        check(
            set(hashes) == {"own"}
            and len(hashes["own"]) == 120
            and all(
                sha(base / "images/own" / f"{i:03d}.jpg") == h for i, h in enumerate(hashes["own"])
            ),
            "Sealed JPEG differs",
        )
        settings[arm] = cfg
    old, new = [settings[arm] for arm in METHODS]
    check(
        all(
            old[k] == new[k]
            for k in (
                "adapter_sha256",
                "turbo_sha256",
                "dictionary_sha256",
                "batch_size",
                "driver_sha256",
                "seeds",
            )
        )
        and old["batch_size"] == 8,
        "Generator changed between methods",
    )
    rows = np.array(old["rows"])
    same = np.array(old["predicted_words"]) == new["predicted_words"]
    equal = np.array(old["jpeg_sha256"]["own"]) == new["jpeg_sha256"]["own"]
    check(np.all(equal[same]), "Independent same-word renders differ")
    first = int(np.flatnonzero(same)[np.argmin(rows[same])])
    return settings, dict(
        same_word=int(same.sum()),
        same_word_identical=int(equal[same].sum()),
        different_word=int((~same).sum()),
        different_word_identical=int(equal[~same].sum()),
        smallest_same_word_row=int(rows[first]),
        independent_redraw_sha256=old["jpeg_sha256"]["own"][first],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--panel", choices=PANELS, required=True)
    p.add_argument("--archive", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    combos = condition_pair(a.root, a.panel)
    if a.check_only:
        print(a.panel, "checked both original condition packets")
        return
    for m in METHODS:
        args = copy(a)
        args.root = a.root / m
        args.batch_size = 8
        args.subject = SUBJECT
        draw_packets(args, {a.panel: combos[m]})
    _, checklist = verify_images(a.root, a.panel, combos)
    print(a.panel, checklist, flush=True)


if __name__ == "__main__":
    main()
