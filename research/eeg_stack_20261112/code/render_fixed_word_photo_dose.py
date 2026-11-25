"""Verify and draw the fixed-word 1-, 3- or 9-observer EEG photo conditions."""

import argparse
from pathlib import Path

import numpy as np
from render_within_photo_ip import ARMS, check, check_donor_geometry, draw_packets, read, sha

PEOPLE = ("sub-04", "sub-05", "sub-01", "sub-02", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10")
GROUPS = {"single-1": PEOPLE[:1], "cohort-3": PEOPLE[:3], "cohort-9": PEOPLE}
PANELS = ("registered", "replicate")


def packet(root, subject, panel):
    design = read(root / "design.json")
    frozen = root / "study_design.json"
    radius = read(root / "radius.json")
    check(
        design["study_design_sha256"] == sha(frozen)
        and read(frozen)["nested_contributors"][subject] == list(GROUPS[subject])
        and design["radius_sha256"] == sha(root / "radius.json")
        and radius["generic_rows"] == design["generic_radius_rows"]
        and np.median(radius["cosines_to_author"]) > 0.95
        and radius["median_norm"] == np.median(radius["norms"]),
        "Registered cohort or generic image projection radius changed",
    )
    file = root / "conditions" / subject / panel / "conditions_128.npz"
    check(sha(file) == design["packets_sha256"][f"{subject}/{panel}"], "Incorrect frozen packet")
    with np.load(file, allow_pickle=False) as z:
        rows, folds, words, donors, support = (
            z[k] for k in ("rows", "folds", "predicted_words", "donor_rows", "support_rows")
        )
        vectors = {k: z[k] for k in ARMS}
        angles = z["recipient_angle_radians"]
        check(
            z["contributing_subjects"].tolist() == list(GROUPS[subject])
            and int(z["physical_trials_per_photo"]) == 4 * len(GROUPS[subject])
            and z["training_classes"].shape == (128,)
            and len(np.unique(z["training_classes"])) == 128
            and not np.intersect1d(rows // 10, z["training_classes"]).size
            and z["study_design_sha256"].item() == sha(frozen)
            and z["visual_sha256"].item() == design["visual_sha256"]
            and z["native_packet_sha256"].item()
            == read(frozen)["sources"]["native_packets"][panel],
            "Wrong cohort, calibration objects or baseline evidence",
        )
    check(
        rows.shape == folds.shape == words.shape == donors.shape == (120,)
        and len(np.unique(rows)) == 120
        and set(donors) == set(rows)
        and set(folds) == set(range(5))
        and support.shape == (120, 8),
        "Malformed row registration",
    )
    lookup = {v: i for i, v in enumerate(rows)}
    donor = np.array([lookup[v] for v in donors])
    check(
        np.array_equal(folds, folds[donor])
        and np.array_equal(words, words[donor])
        and np.array_equal(support, support[donor])
        and all(
            len(set(s)) == 8 and set(s) <= set(rows[folds != f])
            for f, s in zip(folds, support, strict=True)
        ),
        "Active query fold entered supports or wrong donor crossed a word/fold",
    )
    check(
        all(
            v.shape == (120, 1024) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
            for v in vectors.values()
        ),
        "Invalid H14 vectors",
    )
    check_donor_geometry(vectors, donor, angles)
    return rows, folds, words, donors, support, vectors, float(radius["median_norm"]), sha(file)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--subject", choices=GROUPS, required=True)
    p.add_argument("--panels", choices=PANELS, nargs="+", default=PANELS)
    p.add_argument("--batch-size", type=int, choices=(8,), default=8)
    p.add_argument("--check-only", action="store_true")
    for k in ("archive", "adapter", "dictionary"):
        p.add_argument("--" + k, type=Path)
    a = p.parse_args()
    data = {panel: packet(a.root, a.subject, panel) for panel in a.panels}
    if not a.check_only:
        if not all((a.archive, a.adapter, a.dictionary)):
            p.error("The renderer needs --archive, --adapter and --dictionary")
        draw_packets(a, data)
