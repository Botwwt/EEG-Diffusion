"""Independently evaluate saved H14-conditioned JPEGs with the archived OpenAI CLIP-L/14."""

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from render_within_photo_ip import ARMS, PANELS, check, packet, read, save, sha
from score_object_panel import score_object_panel
from score_official_objects import load_dictionary, load_tokenizer, text_bank, topk_credit, unit
from score_personal_photo_layout import original_records
from score_reconstruction_jpegs import CLIP_SHA, originals, raw_features, registered


def estimate(v, labels):
    v = np.asarray(v, dtype=float)
    labels = np.asarray(labels)
    classes, ix, n = np.unique(labels, return_inverse=True, return_counts=True)
    o = dict(
        n=len(v),
        n_true_objects=len(classes),
        credit=float(v.sum()),
        mean=float(v.mean()) if len(v) else None,
        ci95=None,
    )
    if len(classes) > 1:
        sums = np.bincount(ix, weights=v)
        draws = np.random.default_rng(260509).integers(len(classes), size=(10000, len(classes)))
        o["ci95"] = np.quantile(sums[draws].sum(1) / n[draws].sum(1), [0.025, 0.975]).tolist()
    return o


def paired_table(values, use, classes):
    return {
        "arms": {arm: estimate(v[use], classes[use]) for arm, v in values.items()},
        "own_minus": {
            k: estimate((values["own"] - values[k])[use], classes[use])
            for k in ("uniform", "wrong")
        },
    }


def summarize_features(x, y, reg, words, donors, ids, inverse, bank):
    truth, folds, labels = reg["truth"], reg["folds"], reg["labels"]
    pos = {r: i for i, r in enumerate(reg["rows"])}
    dj = np.asarray([pos[r] for r in donors])
    nq = len(words)
    ii = np.arange(nq)
    pairs = []
    for k in range(12):
        for f in range(5):
            t = np.flatnonzero((truth == k) & (folds == f))
            check(len(t) == 2, "Unexpected fold")
            if words[t[0]] == words[t[1]]:
                pairs.append(t)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    i, j = pairs.T
    photo_classes = truth[i]
    masks = {
        "same_word": np.ones(len(i), bool),
        "correct_word": words[i] == labels[i],
        "correct_word_reciprocal_donors": (words[i] == labels[i]) & (dj[i] == j) & (dj[j] == i),
    }
    image_metrics = {}
    pair_metrics = {}
    name_to_id = {w: k for k, w in enumerate(ids)}
    actual = np.asarray([name_to_id[w] for w in labels])
    subset = np.unique(actual)
    angular_y = unit(np.asarray(y, dtype=np.float64))
    for a, v in x.items():
        logits = (unit(v) @ bank.T)[:, inverse]
        angular = unit(np.asarray(v, dtype=np.float64)) @ angular_y.T
        centered = unit(v - v.mean(0)) @ unit(y - y.mean(0)).T
        gallery = score_object_panel(centered, reg["rows"], labels)
        rows = dict(
            full1854=topk_credit(logits, 1)[ii, actual],
            closed12=topk_credit(logits[:, subset], 1)[ii, np.searchsorted(subset, actual)],
            intra10=np.array([r["intra10_top1"] for r in gallery["per_photo"]]),
            cosine_to_original=angular.diagonal(),
            own_minus_donor_cosine=angular[ii, ii] - angular[ii, dj],
        )
        for m, vec in rows.items():
            image_metrics.setdefault(m, {})[a] = vec
        left = angular[i, i] - angular[i, j]
        right = angular[j, j] - angular[j, i]
        margin = (left + right) / 2

        def credit(z):
            return (z > 1e-12).astype(float) + 0.5 * (abs(z) <= 1e-12)

        for m, vec in {
            "assignment": credit(margin),
            "mean_cosine_margin": margin,
            "two_choice": (credit(left) + credit(right)) / 2,
        }.items():
            pair_metrics.setdefault(m, {})[a] = vec
    distinct = (dj != ii) & (labels == labels[dj])
    row_masks = {
        "all": np.ones(nq, bool),
        "correct_word": words == labels,
        "same_truth_word_fold_distinct_donor": distinct,
    }
    return dict(
        pair_rule="One entry per two held-out originals of the same true class and fold for which predicted words agree. Raw unit-normalized L/14 cosine. Assignment credits correct one-to-one coupling vs exchanged pairing, ties split.",
        row_metrics={
            g: {m: paired_table(vec, mask, truth) for m, vec in image_metrics.items()}
            for g, mask in row_masks.items()
        },
        pair_metrics={
            g: {m: paired_table(vec, mask, photo_classes) for m, vec in pair_metrics.items()}
            for g, mask in masks.items()
        },
        rows=reg["rows"].tolist(),
        truth=labels.tolist(),
        predicted_words=words.tolist(),
        folds=folds.tolist(),
        donor_rows=donors.tolist(),
        per_row={m: {a: v.tolist() for a, v in vec.items()} for m, vec in image_metrics.items()},
        pair_rows=reg["rows"][pairs].tolist(),
        correct_word_pair=masks["correct_word"].tolist(),
        reciprocal_correct_pair=masks["correct_word_reciprocal_donors"].tolist(),
        per_pair={m: {a: v.tolist() for a, v in vec.items()} for m, vec in pair_metrics.items()},
    )


def run(args):
    import torch
    from eval_instance_starts import load_visual_scorer

    torch.set_grad_enabled(False)
    records_by_panel = {p: registered(args.reference_dir, p) for p in PANELS}
    # Require every JPEG run requested in this evaluation to be frozen before reading original pixels.
    registered_cases = {}
    for sub in args.subjects:
        for panel, reg in records_by_panel.items():
            rows, folds, words, donors, support, _, _, signature = packet(args.root, sub, panel)
            dest = args.root / "renders" / panel / sub
            marker = dest / "settings.json"
            cfg = read(marker)
            wi = {w: i for i, w in enumerate(reg["vocab"])}
            check(
                np.array_equal(rows, reg["rows"])
                and np.array_equal(folds, reg["folds"])
                and np.array_equal(support, reg["calibration"][folds, [wi[w] for w in words]])
                and cfg["packet_sha256"] == signature
                and cfg["design_sha256"] == sha(args.root / "design.json")
                and np.array_equal(cfg["predicted_words"], words)
                and np.array_equal(cfg["rows"], rows)
                and np.array_equal(cfg["donor_rows"], donors)
                and set(cfg["jpeg_sha256"]) == set(ARMS),
                "Inputs and completed JPEG run differ",
            )
            registered_cases[sub, panel] = dest, marker, cfg, words, donors
    full_rows = np.concatenate([reg["rows"] for reg in records_by_panel.values()])
    full_labels = np.concatenate([reg["labels"] for reg in records_by_panel.values()])
    records, manifest_sha = original_records(args.archive, full_rows.tolist(), full_labels.tolist())
    model, model_sha = load_visual_scorer(args.archive.resolve())
    check(model_sha == CLIP_SHA, "Incorrect L/14 evaluator")
    ids, common, inverse, _, _ = load_dictionary(args.dictionary)
    tok = load_tokenizer(args.archive / "assets/native_generator_v243/turbo/tokenizer")
    bank = text_bank(model, tok, common, args.batch_size)
    out = dict(
        clip_sha256=model_sha,
        dictionary_sha256=sha(args.dictionary),
        manifest_sha256=manifest_sha,
        design_sha256=sha(args.root / "design.json"),
        scorer_sha256=sha(Path(__file__)),
        cases={},
    )
    for panel, reg in records_by_panel.items():
        refs = originals(args.archive, records, reg["rows"])
        y = raw_features(model, refs.__getitem__, len(refs), args.batch_size)
        for sub in args.subjects:
            dest, marker, cfg, words, donors = registered_cases[sub, panel]
            features = {}
            for a in ARMS:

                def get(i, folder=dest, arm=a, hashes=cfg["jpeg_sha256"][a]):
                    raw = (folder / "images" / arm / f"{i:03d}.jpg").read_bytes()
                    check(
                        hashlib.sha256(raw).hexdigest() == hashes[i],
                        "JPEG differs from frozen hash",
                    )
                    with Image.open(io.BytesIO(raw)) as im:
                        check(
                            im.size == (512, 512) and im.format == "JPEG", "Incorrect output JPEG"
                        )
                        return im.convert("RGB")

                features[a] = raw_features(model, get, len(words), args.batch_size)
            c = summarize_features(features, y, reg, words, donors, ids, inverse, bank)
            c["settings_sha256"] = sha(marker)
            out["cases"][panel + "/" + sub] = c
            print(
                panel,
                sub,
                {a: v["credit"] for a, v in c["row_metrics"]["all"]["full1854"]["arms"].items()},
                {
                    a: (v["credit"], v["n"])
                    for a, v in c["pair_metrics"]["correct_word"]["assignment"]["arms"].items()
                },
                flush=True,
            )
    save(args.out, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "archive", "dictionary", "reference-dir", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument(
        "--subjects",
        nargs="+",
        choices=("sub-01", "sub-04", "sub-05"),
        default=["sub-04", "sub-05"],
    )
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    run(args)
