"""Apply the frozen sub-04 background readout to 512 additional objects and package gated prompts."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from fit_background_attribute import METHODS, SUPPORT, policy, relative
from measure_background_attribute import DESIGN as FIRST
from measure_background_confirmation import DESIGN, LOCK, PANELS, check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz


def annotation(a, kind, obj):
    path = a.out / "annotations" / kind / f"{obj:04d}.npz"
    with np.load(path, allow_pickle=False) as z:
        check(
            str(z["design"]) == LOCK
            and str(z["measure_source"])
            == sha(Path(__file__).with_name("measure_background_confirmation.py")),
            "Annotation mismatch",
        )
        return {k: z[k] for k in ("rows", "labels", "background")}


def heads(a, d):
    path = a.data / "eeg-background-attribute/model.npz"
    check(
        sha(path) == d["sources"]["frozen_model"]
        and sha(FIRST) == d["sources"]["base_design"]
        and sha(Path(__file__).with_name("fit_background_attribute.py"))
        == d["sources"]["fit_source"],
        "Changed initial fit or source",
    )
    with np.load(path, allow_pickle=False) as z:
        return z["heads"], z["scales"]


def montage(a, d):
    path = a.data / "trial-object-inputs/sub-04/generic512.json"
    ch = a.data / "object-transfer/halves/sub-04.npz"
    check(
        sha(path) == d["sources"]["generic512"] and sha(ch) == read(FIRST)["inputs"]["montage"],
        "Wrong montage",
    )
    with np.load(ch, allow_pickle=False) as z:
        names = z["channels"].tolist()
    allnames = read(path)["channel_names"]
    return allnames, names


def features(raw, chans, posterior):
    check(
        raw.ndim == 4 and raw.shape[1:] == (4, 63, 125) and raw.dtype == np.float32,
        "Unexpected trial shape or units",
    )
    idx = [chans.index(k) for k in posterior]
    check(len(idx) == 17 and len(set(chans)) == 63, "Expected 17 posterior electrodes of 63")
    x = np.take(raw, idx, axis=2)[:, :, :, 15:115].astype(float)
    return x.mean(1).reshape(len(x), 17, 20, 5).mean(-1).reshape(len(x), 340)


def predict(a, d):
    obj = np.asarray(d["objects"])
    path = a.data / "scale-trial/extra/receipt.json"
    full = a.data / "double-calibration/design.json"
    check(
        sha(path) == d["sources"]["extra_eeg"]
        and sha(full) == d["sources"]["full_class_registration"],
        "Wrong 512 extra physical registration",
    )
    meta = read(path)
    old = read(FIRST)
    check(
        len(set(obj)) == len(obj) == 512
        and set(meta["object_ids"]) == set(obj) == set(read(full)["extra_objects"])
        and set(obj).isdisjoint(
            old["fit_classes"] + old["calibration_classes"] + old["evaluation_classes"]
        ),
        "Expected 512 distinct untouched objects",
    )
    head, scale = heads(a, d)
    chans, post = montage(a, d)
    out = []
    for c in obj:
        file = path.parent / f"{c:04d}.npy"
        rec = meta["blocks"][str(c)]
        check(sha(file) == rec["output_sha256"], "Physical block hash mismatch")
        x = features(np.load(file, allow_pickle=False), chans, post)
        pred = (relative(x[None])[0] @ head) * scale
        at = annotation(a, "generic", c)
        check(np.array_equal(at["rows"], 10 * c + np.arange(10)), "Incomplete generic annotation")
        prior = np.array([at["labels"][indices].mean() for indices in SUPPORT])
        out.append(
            np.clip(
                np.column_stack(
                    (
                        prior,
                        prior + pred[:, 0],
                        prior + pred[np.arange(10) ^ 1, 0],
                        prior + pred[:, 1],
                    )
                ),
                0.01,
                0.99,
            )
        )
    frozen_npz(
        a.out / "blind_predictions.npz",
        dict(
            rows=obj[:, None] * 10 + np.arange(10),
            probabilities=np.stack(out),
            methods=METHODS,
            support_ordinals=SUPPORT,
            model_sha256=d["sources"]["frozen_model"],
            design=LOCK,
        ),
    )


def score(a, d):
    ids = np.array(d["objects"])
    source = a.out / "blind_predictions.npz"
    with np.load(source, allow_pickle=False) as z:
        p = z["probabilities"]
        check(
            str(z["design"]) == LOCK
            and np.array_equal(z["rows"], ids[:, None] * 10 + np.arange(10))
            and str(z["model_sha256"]) == d["sources"]["frozen_model"],
            "Changed sealed generic predictions",
        )
    data = [annotation(a, "generic", i) for i in ids]
    y = np.stack([z["labels"] for z in data])
    known = y != 0.5
    err = (p - y[:, :, None]) ** 2
    draws = np.random.default_rng(d["seed"]).integers(len(ids), size=(10000, len(ids)))

    def est(val, mask):
        sums = np.where(mask, val, 0).sum(1)
        ns = mask.sum(1)
        n = int(ns.sum())
        den = ns[draws].sum(1)
        ix = den > 0
        if n == 0:
            return dict(n=0, objects=0, mean=None, ci95=None)
        return dict(
            n=n,
            objects=int(np.count_nonzero(ns)),
            mean=float(sums.sum() / n),
            ci95=np.quantile(sums[draws].sum(1)[ix] / den[ix], [0.025, 0.975]).tolist(),
        )

    brier = {k: est(err[:, :, i], known) for i, k in enumerate(METHODS)}
    gain = {k: est(err[:, :, i] - err[:, :, 1], known) for i, k in enumerate(METHODS) if i != 1}
    direction = {}
    for name, flag in (("light", y == 1), ("texture", y == 0)):
        direction[name] = dict(
            brier_gain_vs_wrong=est(err[:, :, 2] - err[:, :, 1], flag),
            hits={
                k: est(((p[:, :, j] - 0.5) * (2 * y - 1) > 0) + 0.5 * (p[:, :, j] == 0.5), flag)
                for j, k in enumerate(METHODS)
            },
        )
    yd = y[:, ::2] - y[:, 1::2]
    contrast = np.abs(yd) == 1
    margins = yd[:, :, None] * (p[:, ::2] - p[:, 1::2])
    pairs = {
        k: est((margins[:, :, j] > 0) + 0.5 * (margins[:, :, j] == 0), contrast)
        for j, k in enumerate(METHODS)
    }
    action = policy(p)
    outcomes = {}
    for k, value, target in (("light", 1, 1), ("texture", -1, 0)):
        attempted = action == value
        outcomes[k] = dict(
            attempted=int(attempted.sum()),
            unknown=int((attempted & ~known).sum()),
            precision=est(y == target, attempted & known),
            false=int((attempted & known & (y != target)).sum()),
        )
    ep = pairs["eeg"]
    op = outcomes["texture"]["precision"]
    gates = dict(
        three_brier_lower_bounds=all(v["ci95"][0] > 0 for v in gain.values()),
        at_least_20_pairs=ep["n"] >= 20,
        pair_lower_above_chance=ep["ci95"] is not None and ep["ci95"][0] > 0.5,
        texture_coverage_precision=op["n"] >= 20 and op["mean"] >= 0.70,
        texture_gain_vs_wrong=direction["texture"]["brier_gain_vs_wrong"]["mean"] > 0,
    )
    result = dict(
        subject="sub-04",
        n=int(y.size),
        objects=len(ids),
        design_sha256=LOCK,
        source_sha256=sha(Path(__file__)),
        prediction_sha256=sha(source),
        model_sha256=d["sources"]["frozen_model"],
        annotation_driver_sha256=sha(
            Path(__file__).with_name("measure_background_confirmation.py")
        ),
        clean_light=int((y == 1).sum()),
        textured=int((y == 0).sum()),
        uncertain=int((~known).sum()),
        insufficient_background=int(sum((z["background"][:, 0] < 0.15).sum() for z in data)),
        brier=brier,
        paired_brier_gain=gain,
        opposite_pairs=pairs,
        by_direction=direction,
        actions=outcomes,
        gates=gates,
        gate_passed=all(gates.values()),
    )
    freeze(a.out / "summary.json", result)
    if a.release:
        freeze(a.release, result)
    print("Confirmation", result["gate_passed"], gain, ep, outcomes, flush=True)


def package(a, d):
    summary = read(a.out / "summary.json")
    check(
        summary["design_sha256"] == LOCK
        and summary["gate_passed"]
        and summary["prediction_sha256"] == sha(a.out / "blind_predictions.npz"),
        "Confirmation gate not met",
    )
    root = a.data / "fresh-mix/render_inputs/learned"
    target = a.data / "fresh-mix/eeg/sub-04.npz"
    check(
        sha(root / "design.json") == d["sources"]["target_render_design"]
        and sha(root / "radius.json") == d["sources"]["target_radius"]
        and sha(target) == d["sources"]["target_source"],
        "Fresh mix design or EEG changed",
    )
    rendered = read(root / "design.json")
    head, scales = heads(a, d)
    _, post = montage(a, d)
    out = a.out / "target"
    with np.load(target, allow_pickle=False) as z:
        source_rows = z["rows"]
        x = features(z["trials"], z["channels"].tolist(), post)
        check(
            np.all(np.diff(source_rows) > 0)
            and np.array_equal(post, z["posterior_channels"].tolist()),
            "Wrong target rows or posterior montage",
        )
    traces = {}
    for panel in PANELS:
        p = root / "conditions/sub-04" / panel / "conditions_512.npz"
        st = root / "renders" / panel / "sub-04/settings.json"
        check(sha(p) == rendered["packets_sha256"]["sub-04/" + panel], "Changed H14 packet")
        with np.load(p, allow_pickle=False) as z:
            rows, folds, support, donor, words = [
                z[k] for k in ("rows", "folds", "support_rows", "donor_rows", "predicted_words")
            ]
        cfg = read(st)
        seeds = 923761 + rows
        check(
            cfg["packet_sha256"] == sha(p)
            and np.array_equal(cfg["rows"], rows)
            and np.array_equal(cfg["seeds"], seeds)
            and np.array_equal(cfg["predicted_words"], words)
            and np.array_equal(cfg["support_rows"], support),
            "Existing original branch disagrees",
        )
        lookup = {int(row): j for j, row in enumerate(rows)}
        di = np.array([lookup[int(row)] for row in donor])
        check(
            np.array_equal(words, words[di])
            and np.array_equal(folds, folds[di])
            and np.array_equal(support, support[di])
            and sorted(di) == list(range(len(di))),
            "Invalid same-word/fold donor",
        )
        prior = np.empty(len(rows))
        own = np.empty(len(rows))
        wrong = np.empty(len(rows))
        # Give each active fold a fresh dictionary containing only rows used as its allowed supports.
        for f in range(5):
            query = np.flatnonzero(folds == f)
            legal = np.unique(support[query])
            check(
                not np.intersect1d(legal, rows[query]).size, "Held-out image in support dictionary"
            )
            codes = {}
            for ob in np.unique(legal // 10):
                stored = annotation(a, "supports", ob)
                ix = np.isin(stored["rows"], legal)
                codes.update(zip(stored["rows"][ix], stored["labels"][ix], strict=True))
            check(set(codes) == set(legal), "Missing legal support annotation")
            sq = np.searchsorted(source_rows, support[query])
            rq = np.searchsorted(source_rows, rows[query])
            dq = np.searchsorted(source_rows, donor[query])
            check(
                np.array_equal(source_rows[sq], support[query])
                and np.array_equal(source_rows[rq], rows[query])
                and np.array_equal(source_rows[dq], donor[query]),
                "EEG alignment failed",
            )
            prior[query] = [np.mean([codes[int(r)] for r in su]) for su in support[query]]
            mean = x[sq].mean(1)
            own[query] = np.clip(
                prior[query] + ((x[rq] - mean) @ head[:, 0]) * scales[0], 0.01, 0.99
            )
            wrong[query] = np.clip(
                prior[query] + ((x[dq] - mean) @ head[:, 0]) * scales[0], 0.01, 0.99
            )
        prior = prior.clip(0.01, 0.99)
        matched = (own <= 0.25) & (own - prior <= -0.10)
        fake = (wrong <= 0.25) & (wrong - prior <= -0.10)
        ranking = sorted(
            range(len(rows)),
            key=lambda i: (
                prior[i],
                hashlib.sha256(f"260701/{panel}/{rows[i]}".encode()).hexdigest(),
            ),
        )
        base = np.zeros(len(rows), dtype=bool)
        base[ranking[: int(matched.sum())]] = True
        check(
            np.array_equal(fake, matched[di]) and base.sum() == matched.sum(),
            "Mismatch in control intervention budget",
        )
        dest = out / (panel + ".npz")
        frozen_npz(
            dest,
            dict(
                rows=rows,
                folds=folds,
                seeds=seeds,
                predicted_words=words,
                support_rows=support,
                donor_rows=donor,
                source_packet_sha256=sha(p),
                source_settings_sha256=sha(st),
                probability=own,
                wrong_probability=wrong,
                prior_probability=prior,
                flags_original=np.zeros(len(rows), bool),
                flags_prior=base,
                flags_matched=matched,
                flags_wrong=fake,
                design=LOCK,
            ),
        )
        traces[panel] = dict(
            sha256=sha(dest),
            source_h14_sha256=sha(p),
            source_settings_sha256=sha(st),
            rows=len(rows),
            matched=int(matched.sum()),
            wrong=int(fake.sum()),
            prior=int(base.sum()),
            same_trigger_matched_wrong=int((fake & matched).sum()),
            nonself=int((donor != rows).sum()),
        )
    freeze(
        out / "receipt.json",
        dict(
            design=LOCK,
            model_sha256=d["sources"]["frozen_model"],
            confirmation_sha256=sha(a.out / "summary.json"),
            target_eeg_sha256=sha(target),
            code_sha256=sha(Path(__file__)),
            radius_sha256=d["sources"]["target_radius"],
            suffix=" with a visibly varied, textured background",
            base_template="A photo of {readable}.",
            suffix_template="A photo of {readable} with a visibly varied, textured background.",
            panels=traces,
        ),
    )
    print("Target switches sealed", traces, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["predict", "score", "package"])
    ap.add_argument("--data", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eeg-background-confirmation"))
    ap.add_argument("--release", type=Path)
    args = ap.parse_args()
    check(sha(DESIGN) == LOCK, "Changed confirmation registration")
    {"predict": predict, "score": score, "package": package}[args.stage](args, read(DESIGN))
