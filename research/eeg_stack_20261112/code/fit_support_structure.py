"""Fit photo-specific EEG affinities among the eight authorized support photographs."""

import argparse
import csv
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from decode_eeg_layout import CHANNELS, digest, load_json, save_json
from prepare_registered_subjects import ROWS_SHA
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata

H14_SHA = "b8156300f59d058600fc9a2c9e1586c5fbd4a113f236031acc64c6511d56b432"
PHOTO_SHA = "b91edf1a316142406a2ae28bdbb673712f1d5c38f57c0aec9dcfb5da446cabf1"
SPLIT_SEED = 260437
RANDOM_SEED = 260438
RIDGE = 0.3
PANELS = ("registered", "replicate")


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def population(args):
    csv_path = args.source / "atm_train_rows.csv"
    if digest(csv_path) != ROWS_SHA:
        raise ValueError("Public ATM photo row CSV differs from the pinned release")
    with csv_path.open(encoding="utf8", newline="") as stream:
        table = list(csv.DictReader(stream))
    if [int(r["train_row"]) for r in table] != list(range(16540)):
        raise ValueError("Public photo rows are not in published order")
    return table


def visual(args):
    p = args.source / "assets/text_train.pt"
    if digest(p) != H14_SHA:
        raise ValueError("Wrong author H/14 feature archive")
    with ZipFile(p) as z:
        (key,) = [k for k in z.namelist() if k.endswith("/data/1")]
        h = np.frombuffer(z.read(key), "<f4").reshape(16540, 1024)
    banks = {"h14": unit(h)}
    hashes = {"h14": H14_SHA}
    if args.structure:
        with np.load(args.structure, allow_pickle=False) as f:
            rows, a = f["rows"], f["features"]
        if (
            rows.dtype.kind not in "iu"
            or len(set(rows)) != len(rows)
            or len(rows) != len(a)
            or not np.isfinite(a).all()
            or a.ndim != 2
            or min(rows) < 0
            or max(rows) >= len(h)
        ):
            raise ValueError("Malformed row-indexed structure feature bank")
        indexed = np.full((len(h), a.shape[1]), np.nan, np.float32)
        indexed[rows] = unit(a)
        banks["structure"] = indexed
        hashes["structure"] = digest(args.structure)
    return banks, hashes


def assemble(banks, rows):
    views = [b[rows].astype(np.float64) for b in banks.values()]
    result = np.concatenate(views, axis=-1) / np.sqrt(len(views))
    if not np.isfinite(result).all():
        raise ValueError("Missing image feature for a required row")
    return result


def eeg(args, table, generic):
    subject = args.subject
    path = args.inputs / ("generic" if generic else "halves") / (subject + ".npz")
    with np.load(path, allow_pickle=False) as z:
        rows, ch = z["rows"], z["channels"].tolist()
        halves = z["half_eeg"]
        if (
            str(z["subject"]) != subject
            or ch != CHANNELS
            or list(z["half_names"]) != ["early", "late"]
            or halves.shape != (2, 1280 if generic else 240, 17, 250)
            or not np.allclose(z["times"], np.arange(250) / 250, atol=1e-7, rtol=0)
            or np.any(np.diff(rows) <= 0)
            or not np.isfinite(halves).all()
        ):
            raise ValueError(f"Unexpected R2 EEG axes in {path}")
        if generic:
            if z["unique_ids"].tolist() != [table[i]["unique_id"] for i in rows] or z[
                "stimulus_stems"
            ].tolist() != [table[i]["stimulus_stem"] for i in rows]:
                raise ValueError("Generic EEG and visual row/stem identities differ")
        else:
            with np.load(args.inputs / (subject + ".npz"), allow_pickle=False) as native:
                if (
                    not np.array_equal(rows, native["rows"])
                    or native["unique_ids"].tolist() != [table[i]["unique_id"] for i in rows]
                    or native["stimulus_stems"].tolist()
                    != [table[i]["stimulus_stem"] for i in rows]
                ):
                    raise ValueError("Target EEG and visual row/stem identities differ")
    x = halves[..., 15:115].reshape(2, len(rows), 17, 20, 5).mean(-1, dtype=np.float64)
    return rows, x.reshape(2, len(rows), -1), digest(path)


def train(x, y):
    # Within each generic object all ten photos share their early/late session pair.
    # Center each session view separately; fit both R2 views and their R4 average.
    x = x - x.mean(2, keepdims=True)
    centered = np.concatenate([x, x.mean(0, keepdims=True)], axis=0).reshape(-1, x.shape[-1])
    scale = centered.std(0).clip(1e-8)
    z = centered / scale
    target = (y - y.mean(1, keepdims=True)).reshape(-1, y.shape[-1])
    gram = z.T @ z
    gram.flat[:: len(scale) + 1] += RIDGE * len(z)
    w = cho_solve(cho_factor(gram), z.reshape(3, len(target), -1).sum(0).T @ target)
    return w / scale[:, None]


def logits(queries, supports, pics):
    # Arrays are N,2,D for queries; N,2,8,D for supports; N,8,D for photos.
    q = queries - supports.mean(axis=2)
    s = supports - supports.mean(axis=2, keepdims=True)
    yc = pics - pics.mean(1, keepdims=True)
    crossmodal = np.einsum("nd,nsd->ns", q.mean(1), yc)
    # Cross-session signed inner products suppress direct same-session EEG noise.
    paired = (
        np.einsum("nd,nsd->ns", q[:, 0], s[:, 1]) + np.einsum("nd,nsd->ns", q[:, 1], s[:, 0])
    ) / 2
    return np.stack([crossmodal, paired], axis=1)


def soft(logit):
    standardized = (logit - logit.mean(-1, keepdims=True)) / logit.std(-1, keepdims=True).clip(
        1e-12
    )
    exp = np.exp(standardized - standardized.max(-1, keepdims=True))
    return exp / exp.sum(-1, keepdims=True)


def mix_weights(gram, dot):
    # Exact constrained least squares fit on the triangle a>=0,b>=0,a+b<=1.
    candidates = [np.zeros(2)]
    for i in range(2):
        one = np.zeros(2)
        one[i] = np.clip(dot[i] / max(gram[i, i], 1e-12), 0, 1)
        candidates.append(one)
    v = np.array([1.0, -1])
    a = np.clip(v @ (dot - gram[:, 1]) / max(v @ gram @ v, 1e-12), 0, 1)
    candidates.append(np.array([a, 1 - a]))
    both = np.linalg.pinv(gram) @ dot
    if np.all(both >= 0) and both.sum() <= 1:
        candidates.append(both)
    return min(candidates, key=lambda c: c @ gram @ c - 2 * c @ dot)


def calibrate(x, y):
    classes = np.random.default_rng(SPLIT_SEED).permutation(len(y))
    gram, dot, energy = np.zeros((2, 2)), np.zeros(2), 0.0
    for held in np.array_split(classes, 4):
        keep = np.setdiff1d(np.arange(len(y)), held)
        w = train(x[:, keep], y[keep])
        z = (x[:, held] @ w).transpose(1, 2, 0, 3)
        for k in range(10):
            s = np.flatnonzero(np.arange(10) // 2 != k // 2)
            photos = y[held][:, s]
            score = logits(z[:, k], z[:, s].transpose(0, 2, 1, 3), photos)
            diff = np.einsum("nms,nsd->nmd", soft(score) - 1 / 8, photos)
            residual = y[held, k] - photos.mean(1)
            gram += np.einsum("nid,njd->ij", diff, diff)
            dot += np.einsum("nid,nd->i", diff, residual)
            energy += np.sum(residual**2)
    gate = mix_weights(gram, dot)
    loss = energy + gate @ gram @ gate - 2 * gate @ dot
    return gate, dict(
        generic_folds=4,
        seed=SPLIT_SEED,
        held_out_generic_queries=1280,
        uniform_squared_error=float(energy / 1280),
        gated_squared_error=float(loss / 1280),
        gate_crossmodal_cross_session=gate.tolist(),
        gram=gram.tolist(),
        target_dot=dot.tolist(),
    )


def onehot(i):
    return np.eye(8)[i]


def generate(args):
    table = population(args)
    banks, hashes = visual(args)
    rows, train_eeg, gh = eeg(args, table, True)
    target, target_eeg, th = eeg(args, table, False)
    eligible = np.setdiff1d(np.arange(1654), np.unique(target // 10))
    selected = np.sort(np.random.default_rng(260431).choice(eligible, 128, replace=False))
    if not np.array_equal(rows, (selected[:, None] * 10 + np.arange(10)).ravel()):
        raise ValueError("Generic training object draw changed")
    y = assemble(banks, rows).reshape(128, 10, -1)
    x = train_eeg.reshape(2, 128, 10, -1)
    gate, calibration = calibrate(x, y)
    w = train(x, y)
    calibrated = (target_eeg @ w).transpose(1, 0, 2)
    index = {r: i for i, r in enumerate(target)}
    reports = {}
    for panel in PANELS:
        spec = args.references / (panel + "_ablation.json")
        path = args.inputs / "personal" / panel / args.subject / "evaluation.json"
        ref, personal = load_json(spec), load_json(path)
        if (
            personal["subject"] != args.subject
            or personal["generic_sha256"] != gh
            or personal["target_sha256"] != th
            or any(ref[k] != personal[k] for k in ("rows", "words", "truth", "folds"))
        ):
            raise ValueError("Personal object words, source hashes, or target registration changed")
        q, folds, vocab = (np.array(ref[k]) for k in ("rows", "folds", "words"))
        predicted = np.array(personal["results"]["personal_r4"]["predicted"])
        groups = np.array(ref["calibration_rows_by_fold_and_word"])
        support = groups[folds, predicted]
        if set(q) - set(target) or groups.shape != (5, 12, 8) or len(set(q)) != 120:
            raise ValueError("Unexpected query or calibration IDs")
        for f in range(5):
            if set(groups[f].ravel()) != set(q) - set(q[folds == f]):
                raise ValueError("Fold includes held-out photos")
            for j, g in enumerate(groups[f]):
                if len(set(g)) != 8 or any(table[r]["unique_id"] != vocab[j] for r in g):
                    raise ValueError("Incorrect support class or duplicated photo")
        qi = [index[r] for r in q]
        si = np.array([[index[r] for r in group] for group in support])
        # Target image access here is restricted to each query's eight registered supports.
        ys = assemble(banks, support)
        direct = logits(calibrated[qi], calibrated[si].transpose(0, 2, 1, 3), ys)
        branches = soft(direct)
        weights = np.einsum("m,nms->ns", gate, branches) + max(0.0, 1 - gate.sum()) / 8
        affinity = (
            np.einsum("m,nms->ns", gate, branches) / gate.sum()
            if gate.sum() > 0
            else branches[:, 0]
        )
        central = np.einsum("nsd,nd->ns", ys, ys.mean(1)).argmax(1)
        random = [np.random.default_rng([RANDOM_SEED, int(r)]).integers(8) for r in q]
        out = args.out / panel / args.subject
        out.mkdir(parents=True, exist_ok=True)
        tmp, final = out / "selection.partial.npz", out / "selection.npz"
        np.savez(
            tmp,
            rows=q,
            folds=folds,
            predicted_words=vocab[predicted],
            support_rows=support,
            weights=weights,
            scores=affinity,
            weights_nearest=onehot(affinity.argmax(1)),
            weights_central=onehot(central),
            weights_random=onehot(random),
            raw_scores=direct,
            weights_crossmodal=branches[:, 0],
            weights_cross_session=branches[:, 1],
        )
        tmp.replace(final)
        reports[panel] = digest(final)
        save_json(
            out / "selection.json",
            dict(
                subject=args.subject,
                panel=panel,
                selection_sha256=reports[panel],
                generic_sha256=gh,
                target_sha256=th,
                reference_sha256=digest(spec),
                personal_sha256=digest(path),
                image_features_sha256=hashes,
                train_rows_sha256=ROWS_SHA,
                eeg="17 posterior channels; 60 to <460 ms, nonoverlapping 20 ms signed means",
                ridge=RIDGE,
                calibration=calibration,
                random_seed=RANDOM_SEED,
                script_sha256=digest(Path(__file__)),
            ),
        )
    print(args.subject, calibration, reports, flush=True)


def assess(args):
    table = population(args)
    banks, hashes = visual(args)
    views = {**{k: {k: v} for k, v in banks.items()}, "joint": banks}
    for panel in PANELS:
        out = args.out / panel / args.subject
        path = out / "selection.npz"
        receipt = load_json(out / "selection.json")
        if (
            digest(path) != receipt["selection_sha256"]
            or hashes != receipt["image_features_sha256"]
        ):
            raise ValueError("Selection or score feature bank changed")
        with np.load(path, allow_pickle=False) as z:
            rows, supports, words, folds = (
                z[k] for k in ("rows", "support_rows", "predicted_words", "folds")
            )
            weights = {
                k.removeprefix("weights_"): z[k] for k in z.files if k.startswith("weights_")
            }
            weights.update(gated=z["weights"], uniform=np.full((len(rows), 8), 0.125))
        labels = np.array([table[r]["unique_id"] for r in rows])
        donor = np.arange(len(rows))
        for fold in np.unique(folds):
            for name in np.unique(words):
                ids = np.flatnonzero((folds == fold) & (words == name))
                donor[ids] = np.roll(ids, 1)
        if not np.array_equal(supports, supports[donor]):
            raise ValueError("Swapped queries have different allowed supports")
        weights["same_word_fold_swap"] = weights["gated"][donor]
        result = {}
        joint_query_measures = {}
        for key, bank in views.items():
            actual, refs = assemble(bank, rows), assemble(bank, supports)
            similarities = np.einsum("nd,nsd->ns", actual, refs)
            oracle = onehot(similarities.argmax(1))
            ranks = rankdata(-similarities, axis=1, method="average")
            best = similarities == similarities.max(1, keepdims=True)
            metrics = {}
            for arm, p in {**weights, "evaluation_oracle": oracle}.items():
                pred = np.einsum("ns,nsd->nd", p, refs)
                metrics[arm] = np.stack(
                    [
                        np.sum((actual - pred) ** 2, axis=1),
                        np.einsum("nd,nd->n", unit(pred), actual),
                        (p * similarities).sum(1),
                        (p * ranks).sum(1),
                        (p * best).sum(1),
                    ],
                    axis=1,
                )
            if key == "joint":
                joint_query_measures = {name: values.tolist() for name, values in metrics.items()}
            result[key] = {}
            for scope, mask in {
                "all": np.ones(len(rows), bool),
                "predicted_object_correct": labels == words,
                "eligible_for_swap": donor != np.arange(len(rows)),
                "same_truth_swap": (donor != np.arange(len(rows))) & (labels[donor] == labels),
            }.items():
                groups = np.unique(labels[mask])
                if len(groups) == 0:
                    result[key][scope] = dict(n=0, n_classes=0, means={}, gated_minus={})
                    continue
                counts = np.array([(mask & (labels == g)).sum() for g in groups])
                boot = np.random.default_rng(260439).integers(
                    len(groups), size=(10000, len(groups))
                )
                means = {a: v[mask].mean(0).tolist() for a, v in metrics.items()}
                paired = {}
                for arm in ("uniform", "central", "random", "nearest", "same_word_fold_swap"):
                    diff = metrics["gated"] - metrics[arm]
                    sums = np.stack([diff[mask & (labels == g)].sum(0) for g in groups])
                    vals = sums[boot].sum(1) / counts[boot].sum(1)[:, None]
                    paired[arm] = dict(
                        delta=diff[mask].mean(0).tolist(),
                        ci95=np.quantile(vals, [0.025, 0.975], axis=0).T.tolist()
                        if len(groups) > 1
                        else None,
                    )
                result[key][scope] = dict(
                    n=int(mask.sum()), n_classes=len(groups), means=means, gated_minus=paired
                )
        save_json(
            out / "diagnostics.json",
            dict(
                subject=args.subject,
                panel=panel,
                selection_sha256=digest(path),
                metric_order=[
                    "squared_feature_error",
                    "prototype_cosine",
                    "expected_support_cosine",
                    "expected_true_similarity_rank_of_support",
                    "probability_assigned_to_most_similar_support",
                ],
                rows=rows.tolist(),
                donor_rows=rows[donor].tolist(),
                per_query_joint=joint_query_measures,
                bootstrap="10000 draws resampling represented true object classes, seed 260439",
                views=result,
            ),
        )
        print(
            args.subject,
            panel,
            "gated - uniform",
            result["joint"]["all"]["gated_minus"]["uniform"],
            flush=True,
        )


def extract(args):
    from PIL import Image
    from scipy.ndimage import sobel

    table = population(args)
    source_sha = digest(args.photo_zip)
    if source_sha != PHOTO_SHA:
        raise ValueError("Expected the pinned THINGS-EEG2 training photo ZIP")
    with (
        np.load(args.inputs / "generic" / (args.subject + ".npz")) as gen,
        np.load(args.inputs / (args.subject + ".npz")) as tar,
    ):
        rows = np.union1d(gen["rows"], tar["rows"])
    vectors = []
    with ZipFile(args.photo_zip) as z:
        by_stem = {}
        for name in z.namelist():
            p = Path(name)
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not any(
                t.startswith(".") for t in p.parts
            ):
                key = (p.parent.name, p.stem)
                if key in by_stem:
                    raise ValueError(f"Duplicate archived stimulus {key}")
                by_stem[key] = name
        for r in rows:
            record = table[r]
            key = (record["official_folder"], record["stimulus_stem"])
            with Image.open(io.BytesIO(z.read(by_stem[key]))) as im:
                im = im.convert("RGB")
                w, h = im.size
                size = (512, int(512 * h / w)) if w <= h else (int(512 * w / h), 512)
                im = im.resize(size, Image.Resampling.BILINEAR)
                left, top = round((size[0] - 512) / 2), round((size[1] - 512) / 2)
                im = (
                    im.crop((left, top, left + 512, top + 512))
                    .convert("L")
                    .resize((64, 64), Image.Resampling.BOX)
                )
                gray = np.asarray(im, dtype=np.float64) / 255
            intensity = gray.reshape(16, 4, 16, 4).mean((1, 3))
            dx, dy = (sobel(gray, axis=a, mode="reflect") for a in (1, 0))
            angle = np.arctan2(dy, dx) % np.pi
            bins = np.floor(angle * 4 / np.pi).astype(int) % 4
            energy = np.hypot(dx, dy)
            edges = np.stack(
                [(energy * (bins == b)).reshape(8, 8, 8, 8).mean((1, 3)) for b in range(4)]
            )
            vectors.append(np.r_[unit((intensity - intensity.mean()).ravel()), unit(edges.ravel())])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, rows=rows, features=unit(np.asarray(vectors)).astype(np.float32))
    save_json(
        args.out.with_suffix(".json"),
        dict(
            n=len(rows),
            source_sha256=source_sha,
            features_sha256=digest(args.out),
            train_rows_sha256=ROWS_SHA,
            source_name=args.photo_zip.name,
            feature="Equal normalized 16x16 centered grayscale layout and 8x8 four-bin unsigned Sobel orientation histograms",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "score", "extract-structure"):
        p = sub.add_parser(command)
        p.add_argument("--inputs", type=Path, default=Path("outputs/object-transfer"))
        p.add_argument(
            "--source", type=Path, default=Path("outputs/official-object-readout/sub-03")
        )
        p.add_argument("--subject", required=True)
        p.add_argument("--out", type=Path, required=True)
        if command == "extract-structure":
            p.add_argument("--photo-zip", type=Path, required=True)
        else:
            p.add_argument("--references", type=Path, default=Path("outputs/object-fidelity"))
            p.add_argument("--structure", type=Path)
    args = parser.parse_args()
    {"fit": generate, "score": assess, "extract-structure": extract}[args.command](args)


if __name__ == "__main__":
    main()
