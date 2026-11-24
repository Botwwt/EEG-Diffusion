"""Replay the image-level and paired decisions and publish compact H14 IP-Adapter results."""

import argparse
from pathlib import Path

import numpy as np
from render_within_photo_ip import ARMS, PANELS, check, packet, read, save, sha
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_within_photo_ip import estimate


def same_stat(actual, stored):
    if set(actual) != set(stored):
        return False
    return all(
        (v is None and stored[k] is None)
        or (
            v is not None
            and stored[k] is not None
            and np.allclose(v, stored[k], rtol=0, atol=1e-12)
        )
        for k, v in actual.items()
    )


def summaries(stored, raw, mask, labels, is_credit):
    actual = {}
    for arm in ARMS:
        v = np.asarray(raw[arm])
        m = estimate(v[mask], labels[mask])
        check(same_stat(m, stored["arms"][arm]), "Reported arm and independent row replay differ")
        actual[arm] = m
    deltas = {}
    for arm in ("uniform", "wrong"):
        d = estimate((np.asarray(raw["own"]) - raw[arm])[mask], labels[mask])
        check(
            same_stat(d, stored["own_minus"][arm]),
            "Reported contrast and independent row replay differ",
        )
        deltas[arm] = dict(delta=d["mean"], ci95=d["ci95"])
    return dict(
        n=int(mask.sum()),
        objects=len(np.unique(labels[mask])),
        arms={
            a: {
                "mean": v["mean"],
                "ci95": v["ci95"],
                **({"credit": v["credit"]} if is_credit else {}),
            }
            for a, v in actual.items()
        },
        own_minus=deltas,
    )


def case(args, sub, panel, c, score):
    reg = registered(args.references, panel)
    rows, folds, words, donors, support, _, radius, signature = packet(args.root, sub, panel)
    marker = args.root / "renders" / panel / sub / "settings.json"
    cfg = read(marker)
    with np.load(
        args.root / "conditions" / sub / panel / "conditions_512.npz", allow_pickle=False
    ) as data:
        check(str(data["registration_sha256"]) == reg["sha"], "Registered reference mismatch")
    check(
        sha(marker) == c["settings_sha256"]
        and cfg["packet_sha256"] == signature
        and cfg["design_sha256"] == score["design_sha256"]
        and cfg["radius_sha256"] == sha(args.root / "radius.json")
        and cfg["radius"] == radius
        and cfg["dictionary_sha256"] == score["dictionary_sha256"]
        and cfg["adapter_sha256"] == read(args.root / "design.json")["adapter_sha256"]
        and cfg["subject"] == sub
        and cfg["panel"] == panel
        and cfg["driver_sha256"] == sha(args.root / "render_within_photo_ip.py"),
        "Frozen render source changed",
    )
    for key, expected in (
        ("rows", rows),
        ("folds", folds),
        ("predicted_words", words),
        ("donor_rows", donors),
    ):
        check(
            np.array_equal(c[key], expected) and np.array_equal(cfg[key], expected),
            "Frozen identities disagree",
        )
    wi = {word: i for i, word in enumerate(reg["vocab"])}
    check(
        np.array_equal(c["truth"], reg["labels"])
        and np.array_equal(reg["rows"], rows)
        and np.array_equal(reg["folds"], folds)
        and np.array_equal(cfg["support_rows"], support)
        and np.array_equal(support, reg["calibration"][folds, [wi[w] for w in words]])
        and np.array_equal(cfg["seeds"], 923761 + rows)
        and set(cfg["jpeg_sha256"]) == set(ARMS),
        "Original rows, supports, seeds, or arms disagree",
    )
    for arm in ARMS:
        hs = cfg["jpeg_sha256"][arm]
        paths = [marker.parent / "images" / arm / f"{i:03d}.jpg" for i in range(len(rows))]
        check(
            len(hs) == len(paths) == 120
            and all(sha(path) == h for path, h in zip(paths, hs, strict=True)),
            "Frozen output JPEG fingerprint changed",
        )
    truth = reg["truth"]
    labels = reg["labels"]
    pos = {r: i for i, r in enumerate(rows)}
    dj = np.array([pos[r] for r in donors])
    ii = np.arange(len(rows))
    ps = []
    for k in range(12):
        for f in range(5):
            t = np.flatnonzero((truth == k) & (folds == f))
            if words[t[0]] == words[t[1]]:
                ps.append(t)
    pairs = np.asarray(ps, dtype=int).reshape(-1, 2)
    i, j = pairs.T
    check(
        np.array_equal(rows[pairs], np.asarray(c["pair_rows"]).reshape(-1, 2)),
        "Two-photo pairing differs",
    )
    groups = {
        "same_word": np.ones(len(i), bool),
        "correct_word": words[i] == labels[i],
        "correct_word_reciprocal_donors": (words[i] == labels[i]) & (dj[i] == j) & (dj[j] == i),
    }
    check(
        np.array_equal(groups["correct_word"], c["correct_word_pair"])
        and np.array_equal(groups["correct_word_reciprocal_donors"], c["reciprocal_correct_pair"]),
        "Wrong eligible pair subsets",
    )
    row_groups = {
        "all": np.ones(len(rows), bool),
        "correct_word": labels == words,
        "same_truth_word_fold_distinct_donor": (dj != ii) & (labels == labels[dj]),
    }
    row_metrics = {}
    for key, mask in row_groups.items():
        selected = (
            ("full1854", "closed12", "intra10", "cosine_to_original")
            if key == "all"
            else (
                ("own_minus_donor_cosine",)
                if key.startswith("same_truth")
                else ("full1854", "intra10")
            )
        )
        row_metrics[key] = {
            m: summaries(
                c["row_metrics"][key][m],
                c["per_row"][m],
                mask,
                truth,
                m in ("full1854", "closed12", "intra10"),
            )
            for m in selected
        }
    pair_metrics = {
        key: {
            m: summaries(
                c["pair_metrics"][key][m],
                c["per_pair"][m],
                mask,
                truth[i],
                m != "mean_cosine_margin",
            )
            for m in ("assignment", "mean_cosine_margin", "two_choice")
        }
        for key, mask in groups.items()
    }
    return dict(
        images_per_arm=120,
        settings_sha256=sha(marker),
        condition_packet_sha256=cfg["packet_sha256"],
        reg_sha256=reg["sha"],
        row_metrics=row_metrics,
        pair_metrics=pair_metrics,
    )


def run(args):
    design = read(args.root / "design.json")
    cohort = getattr(args, "subjects", None) or design["cohort"]
    check(
        len(set(cohort)) == len(cohort) and set(cohort) <= set(design["cohort"]),
        "Unknown or duplicate render participant",
    )
    radius = read(args.root / "radius.json")
    cases = {}
    hashes = {}
    scorer = None
    for source in args.scores:
        r = read(source)
        check(
            r["clip_sha256"] == CLIP_SHA and r["design_sha256"] == sha(args.root / "design.json"),
            "Wrong independent evaluator or locked design",
        )
        incoming = {k: r[k] for k in ("dictionary_sha256", "manifest_sha256", "scorer_sha256")}
        check(
            incoming["scorer_sha256"] == sha(args.root / "score_within_photo_ip.py")
            and (scorer is None or scorer == incoming),
            "Scoring sources changed",
        )
        scorer = incoming
        hashes[source.name] = sha(source)
        for key, c in r["cases"].items():
            p, s = key.split("/")
            check(
                s in cohort and p in PANELS and key not in cases,
                "Duplicate or unregistered case",
            )
            cases[key] = case(args, s, p, c, r)
    check(set(cases) == {f"{p}/{s}" for s in cohort for p in PANELS}, "Incomplete cohort")
    query_count = sum(case_data["images_per_arm"] for case_data in cases.values())
    original_count = len(set().union(*(registered(args.references, p)["rows"] for p in PANELS)))
    shared = (
        "These cases use the same photos in both participants."
        if len(cohort) == 2
        else "The audited participant panels use the same registered photographic identities."
    )
    return dict(
        design_sha256=sha(args.root / "design.json"),
        **({"rendered_subjects": list(cohort)} if set(cohort) != set(design["cohort"]) else {}),
        radius_sha256=sha(args.root / "radius.json"),
        native_h14_radius=radius["median_norm"],
        generic_h14_cosine_range=[
            min(radius["cosines_to_author"]),
            float(np.median(radius["cosines_to_author"])),
            max(radius["cosines_to_author"]),
        ],
        independent_clip_sha256=CLIP_SHA,
        score_file_sha256=hashes,
        **scorer,
        participant_photo_queries=query_count,
        unique_originals=original_count,
        generated_jpegs=query_count * len(ARMS),
        comparison=f"A single registered R4 query and its predicted word condition one generated JPEG per arm, with matched per-photo noise. {shared} Endpoint pairs hold true class and fold fixed and require both predictions to name the correct class. The reciprocal subset also requires the wrong arm to exchange exactly those two EEG residuals.",
        units="Credit counts split ties; cosine margin is per pair. Paired percentile 95% intervals resample represented true object classes within each panel and participant. Intra10 follows the prior separately centered panel gallery; pair decisions use raw unit CLIP vectors. The recipient's row seed is held fixed across arms, so a uniform condition can distinguish pairs by chance.",
        cases=cases,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/photo-structure/h14_residual_render"))
    p.add_argument("--references", type=Path, default=Path("outputs/object-fidelity"))
    p.add_argument("--scores", type=Path, nargs="+", required=True)
    p.add_argument("--subjects", nargs="+", help="Participants specified for the image experiment")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    save(a.out, run(a))
