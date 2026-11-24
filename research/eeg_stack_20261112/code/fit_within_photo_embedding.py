"""Decode within-object H/14 photo differences with frozen 128- and 512-concept EEG calibration."""

import argparse
import hashlib
import itertools
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from decode_eeg_layout import eeg_design
from fit_personal_session_metric import load
from fit_support_structure import H14_SHA, population, unit
from scipy.linalg import eigh

SEED = 260439
GRID = tuple(itertools.product((0.1, 0.5, 2.0), (0.0, 0.5, 1.0)))
PANELS = ("registered", "replicate")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The selection digest is embedded in generator conditions and the confirmation design.
    ending = "\r\n" if path.name in ("selection.json", "strict_pairs.json") else "\n"
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8", newline=ending
    )


def images(source):
    """Memory map the author's verified raw H/14 tensor. Normalize only explicitly requested rows."""
    population(SimpleNamespace(source=source))
    path = source / "assets/text_train.pt"
    if sha(path) != H14_SHA:
        raise ValueError("Unexpected official H/14 embedding file")
    with zipfile.ZipFile(path) as z:
        (member,) = [e for e in z.infolist() if e.filename.endswith("/data/1")]
    if member.compress_type != zipfile.ZIP_STORED or member.file_size != 16540 * 1024 * 4:
        raise ValueError("Unexpected format for row-indexed visual embeddings")
    with path.open("rb") as stream:
        stream.seek(member.header_offset)
        header = stream.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid tensor archive header")
        name, extra = struct.unpack("<HH", header[26:30])
        offset = stream.tell() + name + extra
    return np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(16540, 1024))


def training(a, subject, size, bank):
    root = a.expanded / "features"
    meta = read(root / (subject + ".json"))
    classes = np.asarray(meta["original_concepts" if size == 128 else "training_concepts"])
    if (
        meta["subject"] != subject
        or len(set(classes)) != size
        or not meta["old_128_design_replay_exact"]
    ):
        raise ValueError("Invalid generic training receipt")
    features = []
    for c in classes:
        with np.load(root / subject / f"{c:04d}.npz", allow_pickle=False) as z:
            x = z["features"]
            if (
                int(z["start"]) != 10 * c
                or z["raw_sha"].item() != meta["raw_block_sha256"][str(c)]
                or x.shape != (10, 1003)
                or hashlib.sha256(x.tobytes()).hexdigest() != z["feature_sha"].item()
            ):
                raise ValueError("Training EEG block has changed")
            features.append(x)
    x = np.stack(features)
    if size == 512 and hashlib.sha256(x.tobytes()).hexdigest() != meta["train_design_sha256"]:
        raise ValueError("Concatenated 512-concept feature buffer differs from the frozen design")
    return (
        classes,
        x.astype(np.float64),
        unit(bank[classes[:, None] * 10 + np.arange(10)].astype(np.float64)),
    )


def bases(x, y, q):
    # Remove each generic class mean before estimating cross-modal covariance.
    dx = (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])
    dy = (y - y.mean(1, keepdims=True)).reshape(-1, y.shape[-1])
    scale = dx.std(0).clip(1e-6)
    dx /= scale
    n = len(dx)
    lx, vx = eigh(dx.T @ dx / n, check_finite=False)
    ly, vy = eigh(dy.T @ dy / n, check_finite=False)
    return (
        ((q - x.mean((0, 1))) / scale) @ vx,
        lx.clip(0),
        vy,
        (ly / ly.mean()).clip(0),
        vx.T @ (dx.T @ dy / n) @ vy,
    )


def predict(b, setting):
    q, lx, vy, ly, cross = b
    ridge, white = setting
    return ((q / (lx + ridge)) @ cross) * (1 + ly) ** (-white)


def generative(prior, residual):
    # Both terms have the units of the published unit-norm image embedding.
    return unit(prior + residual)


def select(a, bank):
    # Both sizes use a single configuration chosen on the original sub-04 128 only.
    # The 384 extra classes can affect final coefficient estimation but not selection.
    classes, x, y = training(a, "sub-04", 128, bank)
    folds = np.array_split(np.random.default_rng(SEED).permutation(len(classes)), 4)
    counts = np.zeros((4, len(GRID)), float)
    for f, ev in enumerate(folds):
        fit = np.setdiff1d(np.arange(len(classes)), ev)
        b = bases(x[fit], y[fit], x[ev])
        yr = y[ev] @ b[2]
        sy = yr.sum(1)
        for m, setting in enumerate(GRID):
            pred = predict(b, setting)
            sp = pred.sum(1)
            for i, j in itertools.combinations(range(10), 2):
                prior = (sy - yr[:, i] - yr[:, j]) / 8
                center = (sp - pred[:, i] - pred[:, j]) / 8
                ai = generative(prior, pred[:, i] - center)
                aj = generative(prior, pred[:, j] - center)
                advantage = np.sum((ai - aj) * (yr[:, i] - yr[:, j]), axis=1)
                counts[f, m] += np.sum((advantage > 1e-12) + 0.5 * (abs(advantage) <= 1e-12))
    best = int(np.argmax(counts.sum(0)))
    save(
        a.out / "selection.json",
        dict(
            subject="sub-04",
            n_generic_classes=128,
            n_generic_photos=1280,
            n_paired_decisions=128 * 45,
            selection_seed=SEED,
            selected=GRID[best],
            visual_sha256=H14_SHA,
            training_receipt_sha256=sha(a.expanded / "features/sub-04.json"),
            generic_class_folds=[classes[v].tolist() for v in folds],
            grid=[
                dict(
                    ridge=g[0],
                    visual_whitening=g[1],
                    fold_credits=counts[:, i].tolist(),
                    correct_equivalent=float(counts[:, i].sum()),
                )
                for i, g in enumerate(GRID)
            ],
            endpoint="Cosine assignment of two held-out photos from the same generic object, with the other eight supplying the prior. Four disjoint concept folds, all 45 pairs per class. Ties get half credit.",
        ),
    )
    print("Locked", GRID[best], float(counts[:, best].sum()), "/", 128 * 45, flush=True)


def infer(a, subject, bank):
    config = read(a.out / "selection.json")
    if (
        config["subject"] != "sub-04"
        or config["training_receipt_sha256"] != sha(a.expanded / "features/sub-04.json")
        or tuple(config["selected"]) not in GRID
        or config["visual_sha256"] != H14_SHA
    ):
        raise ValueError("Incorrect frozen generic configuration")
    path = a.inputs / "halves" / (subject + ".npz")
    testrows, _, _, mean = load(path)
    xq = eeg_design(mean, np.arange(len(mean)), np.arange(17)).astype(np.float64)
    panel_docs = {}
    for panel in PANELS:
        registration = a.references / (panel + "_ablation.json")
        wordfile = a.expanded / panel / subject / "evaluation.json"
        reg = read(registration)
        words = read(wordfile)
        rows, folds, vocab = [np.asarray(reg[k]) for k in ("rows", "folds", "words")]
        if (
            words["subject"] != subject
            or any(words[k] != reg[k] for k in ("rows", "folds", "words"))
            or words["target_sha256"] != sha(path)
        ):
            raise ValueError("Frozen predicted words do not match EEG or panel")
        ids = np.asarray(words["results"]["expanded_r4"]["predicted"])
        supports = np.asarray(reg["calibration_rows_by_fold_and_word"])
        if supports.shape != (5, 12, 8) or ids.shape != (120,) or np.any((ids < 0) | (ids > 11)):
            raise ValueError("Invalid predicted support class")
        for f in range(5):
            if (
                set(supports[f].ravel()) != set(rows[folds != f])
                or np.unique(supports[f]).size != 96
            ):
                raise ValueError("Query photo entered its calibration fold")
        panel_docs[panel] = (
            rows,
            folds,
            vocab[ids],
            supports[folds, ids],
            sha(registration),
            sha(wordfile),
        )
    for size in (128, 512):
        classes, x, y = training(a, subject, size, bank)
        if np.intersect1d(classes, testrows // 10).size:
            raise ValueError("A registered concept entered generic training")
        basis = bases(x, y, xq)
        residual = predict(basis, config["selected"]) @ basis[2].T
        for panel, (rows, folds, words, support, reg_sha, word_sha) in panel_docs.items():
            n = len(rows)
            own = np.empty((n, 1024), np.float32)
            prior = np.zeros_like(own)
            lookup = {int(r): i for i, r in enumerate(testrows)}
            for fold in range(5):
                ix = np.flatnonzero(folds == fold)
                allowed = support[ix]
                if np.intersect1d(allowed, rows[ix]).size:
                    raise ValueError("Request for a query embedding at inference")
                # The only images accessed in this fold are the eight allowed supports of the predicted class.
                image_mean = unit(bank[allowed].astype(np.float64)).mean(1)
                eeg_mean = residual[[[lookup[r] for r in g] for g in allowed]].mean(1)
                own[ix] = generative(image_mean, residual[[lookup[r] for r in rows[ix]]] - eeg_mean)
                prior[ix] = unit(image_mean)
            donor = np.arange(n)
            for f in range(5):
                for w in np.unique(words[folds == f]):
                    group = np.flatnonzero((folds == f) & (words == w))
                    donor[group] = np.roll(group, 1)
            out = a.out / subject / panel / f"conditions_{size}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out,
                rows=rows,
                folds=folds,
                predicted_words=words,
                own=own,
                uniform=prior,
                wrong=own[donor],
                donor_rows=rows[donor],
                support_rows=support,
                selected=config["selected"],
                training_classes=classes,
                selection_sha256=sha(a.out / "selection.json"),
                target_eeg_sha256=sha(path),
                registration_sha256=reg_sha,
                word_prediction_sha256=word_sha,
                visual_sha256=H14_SHA,
            )
            print("Frozen", out, flush=True)


def uncertainty(values, labels):
    v = np.asarray(values, float)
    groups = np.asarray(labels)
    unique = np.unique(groups)
    sums = np.array([v[groups == c].sum() for c in unique])
    ns = np.array([(groups == c).sum() for c in unique])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(unique), (10000, len(unique)))
    rates = sums[draws].sum(1) / ns[draws].sum(1)
    flips = (
        np.array(list(itertools.product((-1, 1), repeat=len(unique))))
        if len(unique) <= 12
        else rng.choice((-1, 1), size=(100000, len(unique)))
    )
    extreme = np.sum(flips @ sums >= sums.sum() - 1e-10)
    p = extreme / len(flips) if len(unique) <= 12 else (1 + extreme) / (1 + len(flips))
    return dict(
        n=len(v),
        n_concepts=len(unique),
        sum=float(v.sum()),
        mean=float(v.mean()),
        ci95=np.quantile(rates, [0.025, 0.975]).tolist(),
        cluster_signflip_one_sided_p=float(p),
    )


def statistics(records):
    labels = [(p["panel"] + "/" + p["object"]) for p in records]
    data = {}
    for size in (128, 512):
        name = str(size)
        for arm in ("own", "uniform", "wrong", "paired_swap"):
            vals = np.array([r[name][arm] for r in records])
            s = uncertainty(vals - 0.5, labels)
            data[f"{name}_{arm}"] = dict(
                correct_equivalent=float(vals.sum()),
                pairs=len(vals),
                accuracy=float(vals.mean()),
                accuracy_ci95=(0.5 + np.array(s["ci95"])).tolist(),
                class_signflip_vs_half_p=s["cluster_signflip_one_sided_p"],
            )
    for a, b in (("own", "wrong"),):
        delta = [r["512"][a] - r["512"][b] for r in records]
        data["own_minus_frozen_wrong_512"] = uncertainty(delta, labels)
    data["512_minus_128"] = uncertainty(
        [r["512"]["own"] - r["128"]["own"] for r in records], labels
    )
    return data


def score(a, subject, bank):
    all_pairs = []
    hashes = {}
    for panel in PANELS:
        reg = read(a.references / (panel + "_ablation.json"))
        truth = np.array(reg["words"])[reg["truth"]]
        rows = np.asarray(reg["rows"])
        # The true query image embeddings are first indexed after both model sizes' generator inputs exist.
        packets = {}
        for size in (128, 512):
            path = a.out / subject / panel / f"conditions_{size}.npz"
            with np.load(path, allow_pickle=False) as z:
                packet = {
                    k: z[k]
                    for k in (
                        "rows",
                        "folds",
                        "predicted_words",
                        "own",
                        "uniform",
                        "wrong",
                        "donor_rows",
                        "selection_sha256",
                    )
                }
            if not np.array_equal(packet["rows"], rows) or str(packet["selection_sha256"]) != sha(
                a.out / "selection.json"
            ):
                raise ValueError("Scoring registration does not match frozen conditions")
            packets[str(size)] = packet
            hashes[path.relative_to(a.out).as_posix()] = sha(path)
        z = packets["512"]
        if any(
            not np.array_equal(z[k], packets["128"][k])
            for k in ("rows", "predicted_words", "uniform", "folds", "donor_rows")
        ):
            raise ValueError(
                "128 and 512 comparisons must share their prior, predicted word, rows and donors"
            )
        actual = unit(bank[rows].astype(np.float64))
        for f in range(5):
            for word in np.unique(truth):
                i, j = np.flatnonzero((z["folds"] == f) & (truth == word))
                if z["predicted_words"][i] != word or z["predicted_words"][j] != word:
                    continue
                pair = dict(
                    panel=panel, object=str(word), fold=f, rows=[int(rows[i]), int(rows[j])]
                )
                for k, v in packets.items():
                    delta = actual[i] - actual[j]
                    margins = {
                        arm: float((v[arm][i].astype(float) - v[arm][j]) @ delta / 2)
                        for arm in ("own", "uniform", "wrong")
                    }
                    margins["paired_swap"] = -margins["own"]
                    if abs(margins["uniform"]) > 1e-12:
                        raise ValueError("Uniform prior must tie within each strict pair")
                    pair[k] = {
                        a: (1.0 if s > 1e-12 else 0.0 if s < -1e-12 else 0.5)
                        for a, s in margins.items()
                    }
                    pair[k]["cosine_margins"] = margins
                all_pairs.append(pair)
    out = a.out / subject / "strict_pairs.json"
    report = dict(
        subject=subject,
        condition_sha256=hashes,
        endpoint="Mean cosine advantage for matched versus crossed assignment of two held-out images of the same true object and fold, both words correctly predicted. Half credit for ties.",
        wrong_control="wrong uses frozen cyclic EEG donors within predicted word/fold without true labels. paired_swap reverses the two scored queries and is the algebraic complement of own.",
        uncertainty="10,000 percentile bootstrap draws resample concept clusters. One-sided class sign flips are exhaustive up to 12 classes and sampled 100,000 times otherwise. Rates are fractions.",
        per_panel={p: statistics([r for r in all_pairs if r["panel"] == p]) for p in PANELS},
        combined=statistics(all_pairs),
        per_pair=all_pairs,
    )
    save(out, report)
    print(
        subject,
        [
            (
                k,
                v["512_own"]["correct_equivalent"],
                v["512_own"]["pairs"],
                v["128_own"]["correct_equivalent"],
                v["512_wrong"]["correct_equivalent"],
            )
            for k, v in report["per_panel"].items()
        ],
        flush=True,
    )


def release(a, bank):
    lock = read(a.out / "selection.json")
    selection_sha = sha(a.out / "selection.json")
    extra = a.out / "sub-01_confirmation_design.json"
    if read(extra)["selection_sha256"] != selection_sha:
        raise ValueError("Additional participant did not use the frozen selection")
    result = dict(
        endpoint="Joint H/14 embedding assignment of two held-out photos to two EEG predictions. Both photos must share the true object and fold and both previously predicted words must be correct. Chance is 0.5; ties get half credit.",
        single_query_diagnostic="Secondary descriptive check: score each EEG independently against the same two candidate images, with no one-to-one assignment constraint. These scores and the paired endpoint are conditional on the two alternatives and correct EEG words. Neither is unconstrained single-image retrieval or an image generation score.",
        control="Uniform is the same eight allowed support images for both queries. Frozen wrong EEG cycles predictions within the predicted word and fold without using truth. Pair reversal at scoring exchanges the two query predictions and gives the algebraic complement of own.",
        calibration="Matched 128 versus 512 generic non-panel objects, ten photos each. Otherwise identical four-trial posterior EEG features, ridge setting, visual metric, predicted words, and eight labeled supports. Hyperparameters were selected using only generic classes.",
        inference="Unit-norm H/14 prediction = normalized sum of the eight-image mean and a class-centered EEG ridge residual. Within-fold support membership uses only the frozen expanded_r4 word.",
        uncertainty="Per participant: percentile 95% intervals resample whole concepts with all eligible pairs 10,000 times. One-sided sign flips use full concept clusters (exact for at most 12, otherwise 100,000 draws, plus-one correction). CIs are fractions; repeated participants see the same panel photos.",
        generic_selection=dict(
            subject="sub-04",
            generic_classes=128,
            settings=lock["selected"],
            pairs=lock["n_paired_decisions"],
            distinct_photos=lock["n_generic_photos"],
            correct_equivalent=max(v["correct_equivalent"] for v in lock["grid"]),
            sha256=selection_sha,
        ),
        h14_asset_sha256=H14_SHA,
        additional_confirmation_lock_sha256=sha(extra),
        participants={},
    )
    for sub, role in (
        ("sub-04", "development"),
        ("sub-05", "first confirmation"),
        ("sub-01", "additional confirmation"),
    ):
        path = a.out / sub / "strict_pairs.json"
        report = read(path)
        condition_hashes = {k.replace("\\", "/"): v for k, v in report["condition_sha256"].items()}
        panels = {}
        for panel in PANELS:
            rel = f"{sub}/{panel}/conditions_512.npz"
            cond = a.out / rel
            if (
                sha(cond) != condition_hashes[rel]
                or sha(a.out / f"{sub}/{panel}/conditions_128.npz")
                != condition_hashes[f"{sub}/{panel}/conditions_128.npz"]
            ):
                raise ValueError("Frozen generator inputs differ from the paired evaluation")
            with np.load(cond, allow_pickle=False) as z:
                if str(z["selection_sha256"]) != selection_sha:
                    raise ValueError("Hyperparameter selection differs")
                lookup = {int(r): i for i, r in enumerate(z["rows"])}
                vectors = {k: z[k].astype(float) for k in ("own", "uniform", "wrong")}
            pairs = [r for r in report["per_pair"] if r["panel"] == panel]
            single = {k: 0.0 for k in vectors}
            for rec in pairs:
                row = rec["rows"]
                i, j = [lookup[r] for r in row]
                y = unit(bank[row].astype(np.float64))
                diff = y[0] - y[1]
                for k, v in vectors.items():
                    margins = np.array([v[i] @ diff, -v[j] @ diff])
                    single[k] += float(np.sum((margins > 1e-12) + 0.5 * (abs(margins) <= 1e-12)))
            panels[panel] = dict(
                eligible_concepts=len({r["object"] for r in pairs}),
                possible_registered_pairs=60,
                eligible_pairs=len(pairs),
                **compact(report["per_panel"][panel]),
                single_query_two_alternative=dict(queries=2 * len(pairs), correct_credit=single),
            )
        result["participants"][sub] = dict(
            role=role,
            panels=panels,
            combined=dict(
                eligible_pairs=len(report["per_pair"]),
                possible_registered_pairs=120,
                **compact(report["combined"]),
            ),
            score_report_sha256=sha(path),
            conditions_sha256=condition_hashes,
        )
    save(a.release_file, result)


def compact(stats):
    delta = stats["512_minus_128"]
    own = stats["512_own"]
    return dict(
        paired_top1_credit={
            k: stats[k]["correct_equivalent"]
            for k in ("128_own", "512_own", "512_uniform", "512_wrong", "512_paired_swap")
        },
        own512_accuracy=own["accuracy"],
        own512_accuracy_ci95=own["accuracy_ci95"],
        own512_vs_chance_one_sided_p=own["class_signflip_vs_half_p"],
        own512_minus_128=dict(
            delta_rate=delta["mean"],
            ci95=delta["ci95"],
            one_sided_p=delta["cluster_signflip_one_sided_p"],
        ),
        own512_minus_wrong=dict(
            delta_rate=stats["own_minus_frozen_wrong_512"]["mean"],
            ci95=stats["own_minus_frozen_wrong_512"]["ci95"],
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("select", "infer", "score", "release"))
    p.add_argument("--source", type=Path, default=Path("outputs/official-object-readout/sub-03"))
    p.add_argument("--expanded", type=Path, default=Path("outputs/photo-structure/expanded"))
    p.add_argument("--inputs", type=Path, default=Path("outputs/object-transfer"))
    p.add_argument("--references", type=Path, default=Path("outputs/object-fidelity"))
    p.add_argument("--out", type=Path, default=Path("outputs/photo-structure/within_photo"))
    p.add_argument("--subject", choices=("sub-04", "sub-05", "sub-01"), default="sub-04")
    p.add_argument(
        "--release-file",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/object_fidelity/within_photo_embedding.json",
    )
    a = p.parse_args()
    bank = images(a.source)
    if a.stage == "select":
        select(a, bank)
    elif a.stage == "infer":
        infer(a, a.subject, bank)
    elif a.stage == "score":
        score(a, a.subject, bank)
    else:
        release(a, bank)


if __name__ == "__main__":
    main()
