"""Score frozen support-guided images against registered, unseen originals on CPU."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image
from render_official_words import crop512
from scipy.ndimage import gaussian_filter, sobel

ZIP_SHA = "b91edf1a316142406a2ae28bdbb673712f1d5c38f57c0aec9dcfb5da446cabf1"
CSV_SHA = "dafc55af7366974feae0e11418680990ff8d74707ec5b89e398fd5a26567c473"
SIZE = 128
SEED = 260433


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(valid, message):
    if not valid:
        raise ValueError(message)


def luminance(im):
    rgb = np.asarray(im.resize((SIZE, SIZE), Image.Resampling.LANCZOS), dtype=np.float64)
    return (rgb @ [0.299, 0.587, 0.114]) / 255


def structural_features(images):
    """Locally normalized SSIM inputs plus spatial histograms of unsigned gradients."""
    mu = gaussian_filter(images, sigma=(0, 1.5, 1.5), truncate=3.5)
    var = gaussian_filter(images**2, sigma=(0, 1.5, 1.5), truncate=3.5) - mu**2
    gx = np.stack([sobel(im, axis=1) for im in images])
    gy = np.stack([sobel(im, axis=0) for im in images])
    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) % np.pi) * (6 / np.pi)
    hist = []
    for orientation in range(6):
        distance = abs(angle - orientation)
        weight = (1 - np.minimum(distance, 6 - distance)).clip(0, 1)
        cell = (mag * weight).reshape(len(images), 8, 16, 8, 16).mean((2, 4))
        hist.append(cell - cell.mean((1, 2), keepdims=True))
    descriptor = np.stack(hist, axis=-1).reshape(len(images), -1)
    descriptor /= np.linalg.norm(descriptor, axis=1, keepdims=True).clip(1e-12)
    return descriptor, mu, var


def ssim(a, b, ma, mb, va, vb):
    cov = gaussian_filter(a * b, sigma=(0, 1.5, 1.5), truncate=3.5) - ma * mb
    structure = (2 * cov + 0.03**2) / (va + vb + 0.03**2)
    score = (2 * ma * mb + 0.01**2) / (ma**2 + mb**2 + 0.01**2) * structure
    return [v[:, 5:-5, 5:-5].mean((1, 2)) for v in (score, structure)]


def peers(truth, folds, words):
    peer = np.full(len(truth), -1)
    for i, (t, f, w) in enumerate(zip(truth, folds, words, strict=True)):
        idx = np.flatnonzero((truth == t) & (folds == f) & (words == w))
        if len(idx) == 2:
            peer[i] = idx[idx != i].item()
    return peer


def instance(name, similarity, peer):
    diag = similarity.diagonal()
    best = similarity.max(1)
    within10 = (diag == best) / (similarity == best[:, None]).sum(1)
    valid = peer >= 0
    idx = np.flatnonzero(valid)
    margin = np.full(len(peer), np.nan)
    margin[valid] = diag[valid] - similarity[idx, peer[valid]]
    # A reciprocal pair counts once in the mean even though the same assignment credit
    # is attached to both rows. Complete pairs always belong to the same true class.
    advantage = (margin[valid] + margin[peer[valid]]) / 2
    assignment = np.full_like(margin, np.nan)
    assignment[valid] = (advantage > 1e-12) + 0.5 * (abs(advantage) <= 1e-12)
    return {
        name: diag,
        f"{name}_intra10": within10,
        f"{name}_own_minus_peer": margin,
        f"{name}_pair_assignment": assignment,
    }


def vectors(generated, originals, ref_features, truth, peer):
    h, mu, var = structural_features(generated)
    ref, ref_mu, ref_var = ref_features
    n = len(truth)
    by_class = {c: np.flatnonzero(truth == c) for c in np.unique(truth)}
    options = np.stack([by_class[c] for c in truth])
    inside = truth[:, None] == truth[None, :]
    edges = np.where(inside, h @ ref.T, -np.inf)
    matrices = [np.full((n, n), -np.inf), np.full((n, n), -np.inf)]
    for indices in options.T:
        values = ssim(generated, originals[indices], mu, ref_mu[indices], var, ref_var[indices])
        for matrix, value in zip(matrices, values, strict=True):
            matrix[np.arange(n), indices] = value
    return (
        instance("luma_ssim", matrices[0], peer)
        | instance("contrast_structure", matrices[1], peer)
        | instance("edge_layout_cosine", edges, peer)
    )


def estimate(values, labels):
    keep = np.isfinite(values)
    v, label = values[keep], labels[keep]
    classes, inverse, counts = np.unique(label, return_inverse=True, return_counts=True)
    out = dict(
        n=len(v), n_objects=len(classes), mean=float(v.mean()) if len(v) else None, ci95=None
    )
    if len(classes) >= 2:
        by_class = np.bincount(inverse, weights=v)
        draws = np.random.default_rng(SEED).integers(len(classes), size=(10000, len(classes)))
        interval = by_class[draws].sum(1) / counts[draws].sum(1)
        out["ci95"] = np.quantile(interval, [0.025, 0.975]).tolist()
    return out


def original_pixels(args, rows, labels):
    check(sha(args.train_rows) == CSV_SHA, "Expected the pinned official training row/stem map")
    check(sha(args.train_zip) == ZIP_SHA, "Expected the pinned official training ZIP")
    with args.train_rows.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    check([int(r["train_row"]) for r in records] == list(range(16540)), "Wrong training row order")
    receipt = json.loads(args.photo_report.read_bytes())
    registered = json.loads(args.registration.read_bytes())
    ref_hashes = receipt["original_sha256"]
    check(len(ref_hashes) == len(set(ref_hashes)) == len(rows), "Missing original checksums")
    check(
        registered["rows"] == rows.tolist(), "Photo identities differ from registered panel order"
    )
    with ZipFile(args.train_zip) as z:
        members = {}
        for name in z.namelist():
            p = PurePosixPath(name)
            if p.suffix.lower() in (".jpeg", ".jpg", ".png") and "__MACOSX" not in p.parts:
                key = (p.parent.name, p.stem)
                check(key not in members, "Duplicate official stimulus")
                members[key] = name
        images = []
        for row, word, expected in zip(rows, labels, ref_hashes, strict=True):
            rec = records[row]
            check(rec["unique_id"] == word, "Wrong object for stimulus row")
            blob = z.read(members[rec["official_folder"], rec["stimulus_stem"]])
            check(hashlib.sha256(blob).hexdigest() == expected, f"Wrong original for row {row}")
            with Image.open(io.BytesIO(blob)) as im:
                images.append(luminance(crop512(im.convert("RGB"))))
    return np.stack(images)


def score_case(
    args, rows, folds, truth, words, donor, image_files, expected_jpegs, extra_scores=None
):
    """Score a locked set of JPEGs; donor[i] indexes the other EEG query used for row i."""
    row_ids = np.arange(len(rows))
    check(all(a.shape == (120,) for a in (rows, folds, truth, words, donor)), "Wrong query axes")
    check(len(set(rows)) == 120 and set(donor) <= set(row_ids), "Wrong rows or donor indices")
    check(
        all(np.sum((truth == c) & (folds == f)) == 2 for c in set(truth) for f in range(5))
        and len(set(truth)) == 12,
        "Unbalanced fold",
    )
    check(
        np.array_equal(words[donor], words) and np.array_equal(folds[donor], folds),
        "Donor differs in fold or predicted word",
    )
    check(
        set((args.primary, args.swap, args.baseline)) <= set(image_files),
        "Missing primary, wrong-EEG, or baseline arm",
    )
    originals = original_pixels(args, rows, truth)
    ref = structural_features(originals)
    peer = peers(truth, folds, words)
    swapped = (donor != row_ids) & (truth[donor] == truth)
    check(np.all(peer[swapped] == donor[swapped]), "Swapped donor differs from held-out photo peer")
    check(np.all(peer[peer >= 0] != row_ids[peer >= 0]), "A query is its own peer")
    masks = dict(
        all=np.ones(len(rows), bool),
        predicted_object_correct=words == truth,
        same_object_word_fold_peer=peer >= 0,
        verified_wrong_eeg_same_object=swapped,
        verified_wrong_eeg_same_object_and_correct_word=swapped & (words == truth),
    )
    scores, receipts = {}, {}
    for arm, paths in image_files.items():
        check(len(paths) == len(rows) and len(set(paths)) == len(paths), f"Missing JPEGs for {arm}")
        image_hashes, generated = [], []
        for i, path in enumerate(paths):
            blob = path.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if expected_jpegs is not None:
                check(digest == expected_jpegs[arm][i], f"Frozen JPEG changed in {arm}/{i}")
            with Image.open(io.BytesIO(blob)) as im:
                check(
                    im.format == "JPEG" and im.mode == "RGB" and im.size == (512, 512),
                    f"Expected 512-square RGB JPEG in {arm}/{i}",
                )
                generated.append(luminance(im))
            image_hashes.append(digest)
        receipts[arm] = hashlib.sha256("".join(image_hashes).encode("ascii")).hexdigest()
        scores[arm] = vectors(np.stack(generated), originals, ref, truth, peer)
        if extra_scores is not None:
            scores[arm].update(extra_scores(arm, peer))

    def summaries(arm, mask, compare=None):
        return {
            key: estimate(
                (arr if compare is None else arr - scores[compare][key])[mask], truth[mask]
            )
            for key, arr in scores[arm].items()
        }

    groups = {}
    for name, active in masks.items():
        # Restrict the joint two-query assignment to complete reciprocal pairs inside this mask.
        paired = (peer >= 0) & active & active[np.maximum(0, peer)]

        def report(arm, compare=None, active=active, paired=paired):
            details = summaries(arm, active, compare)
            for key in [k for k in details if k.endswith("_pair_assignment")]:
                v = scores[arm][key]
                if compare is not None:
                    v = v - scores[compare][key]
                details[key] = estimate(v[paired], truth[paired])
                details[key]["n_pairs"] = int(paired.sum() // 2)
            return details

        controls = (
            [args.swap]
            if name.startswith("verified_wrong")
            else [arm for arm in image_files if arm not in (args.primary, args.swap)]
        )
        shown = [
            arm for arm in image_files if arm != args.swap or name.startswith("verified_wrong")
        ]
        groups[name] = dict(
            n_rows=int(active.sum()),
            n_classes=len(set(truth[active])),
            arm_means={arm: report(arm) for arm in shown},
            primary_minus={control: report(args.primary, control) for control in controls},
        )
        extra = getattr(args, "extra_comparisons", ())
        if extra and name in ("all", "predicted_object_correct"):
            groups[name]["additional_paired"] = {f"{a}_minus_{b}": report(a, b) for a, b in extra}
    return dict(groups=groups, ordered_jpeg_hash_digest=receipts)


def evaluate(args):
    settings_path = args.renders / "settings.json"
    cfg = json.loads(settings_path.read_bytes())
    reg = json.loads(args.registration.read_bytes())
    prior = json.loads(args.baseline_results.read_bytes())
    rows, folds, words, donors = (
        np.asarray(cfg[k]) for k in ("rows", "folds", "predicted_words", "donor_rows")
    )
    check(
        rows.dtype.kind in "iu" and donors.dtype.kind in "iu" and set(donors) == set(rows),
        "Expected an in-panel permutation of EEG query IDs",
    )
    check(
        cfg["selection_sha256"] == sha(args.selection)
        and cfg["ablation_sha256"] == sha(args.registration)
        and cfg["conditions_sha256"] == prior["condition_sha"]
        and cfg["baseline_results_sha256"] == sha(args.baseline_results),
        "Renderer provenance does not match selection, registration or reference render",
    )
    truth = np.asarray(reg["words"])[reg["truth"]]
    check(
        np.array_equal(rows, reg["rows"])
        and np.array_equal(folds, reg["folds"])
        and np.array_equal(rows, prior["rows"])
        and np.array_equal(truth, prior["labels"])
        and np.array_equal(words, prior["predicted_words"]),
        "Incorrect render identities",
    )
    with np.load(args.selection, allow_pickle=False) as packet:
        for key in ("rows", "folds", "support_rows", "predicted_words"):
            check(np.array_equal(cfg[key], packet[key]), f"Render and selector differ in {key}")
        check(
            np.allclose(cfg["weights"], packet["weights"], rtol=1e-6, atol=1e-7),
            "Render and selector differ in matched weights",
        )
    weights = np.asarray(cfg["weights"])
    random = np.asarray(cfg["random_weights"])
    check(
        weights.shape == random.shape == (120, 8)
        and np.min(weights) >= 0
        and np.allclose(weights.sum(1), 1)
        and np.allclose(np.sort(weights, axis=1), np.sort(random, axis=1)),
        "Random control changes the support weight distribution",
    )
    support = np.asarray(cfg["support_rows"])
    permissible = np.asarray(reg["calibration_rows_by_fold_and_word"])
    wi = {w: i for i, w in enumerate(reg["words"])}
    check(
        support.shape == (len(rows), 8)
        and all(
            set(support[i]) == set(permissible[f, wi[w]])
            and not set(support[i]) & set(rows[folds == f])
            for i, (f, w) in enumerate(zip(folds, words, strict=True))
        ),
        "Support rows overlap held-out images or are outside predicted object's calibration set",
    )
    position = {v: i for i, v in enumerate(rows)}
    donor = np.array([position[v] for v in donors])
    expected_swapped = []
    for i, j in enumerate(donor):
        donor_weights = dict(zip(support[j], weights[j], strict=True))
        check(set(donor_weights) == set(support[i]), "Donor calibration photos differ")
        expected_swapped.append([donor_weights[r] for r in support[i]])
    check(
        np.allclose(cfg["swapped_weights"], expected_swapped, rtol=1e-6, atol=1e-7)
        and np.array_equal(cfg["swap_eligible"], rows != donors),
        "Wrong-EEG render does not use the registered donor weights",
    )
    arms = list(cfg["jpeg_sha256"])
    check(
        {"baseline", "matched", "mean", "swapped", "random"} <= set(arms)
        and all(len(cfg["jpeg_sha256"][arm]) == len(rows) for arm in arms),
        "Missing arm or JPEG hashes",
    )
    image_files = {
        arm: [args.renders / "images" / arm / f"{i:03d}.jpg" for i in range(len(rows))]
        for arm in arms
    }
    output = score_case(args, rows, folds, truth, words, donor, image_files, cfg["jpeg_sha256"])
    return dict(
        query_count=len(rows),
        panel=args.panel,
        subject=args.subject,
        primary_arm=args.primary,
        same_word_fold_wrong_eeg_arm=args.swap,
        frozen_reference_arm=args.baseline,
        endpoints={
            "luma_ssim": "Grayscale SSIM on fixed 128-square images with a Gaussian 11-square window, sigma 1.5 and population covariance.",
            "contrast_structure": "Same local SSIM contrast and covariance term with the local mean-luminance factor omitted.",
            "edge_layout_cosine": "Cosine of spatially centered unsigned Sobel orientation histograms using six directions and eight-by-eight spatial cells. This is related to the structural selector target.",
            "intra10": "Tie-split retrieval among ten originals of the true class, including eight calibration photos and two held-out photos.",
            "own_minus_peer": "Similarity to the true original minus similarity to the other held-out photo of the same object, fold and predicted word.",
            "pair_assignment": "Across two held-out photos sharing true object, fold and predicted word, compare the sum of the two correct similarities with the swapped assignment. Chance is 0.5. Only complete pairs count.",
            "wrong_eeg": "The primary-versus-swapped comparison restricts rows to distinct donor EEG queries sharing true object, fold and predicted word.",
            "uncertainty": "95% paired bootstrap percentile intervals over 10,000 resamples of represented true objects, keeping their photos and reciprocal pairs together. Comparisons use identical eligible rows.",
        },
        sources_sha256=dict(
            training_zip=ZIP_SHA,
            training_rows_csv=CSV_SHA,
            photo_report=sha(args.photo_report),
            registration=sha(args.registration),
            render_settings=sha(settings_path),
            baseline_results=sha(args.baseline_results),
            selection=sha(args.selection),
        ),
        **output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("registered", "replicate"), required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--renders", type=Path, required=True, help="One case with settings.json and images/ARM/"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--train-zip", type=Path, default=Path("outputs/photo-structure/training_images.zip")
    )
    parser.add_argument(
        "--train-rows",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    )
    for key in ("registration", "photo-report", "selection", "baseline-results"):
        parser.add_argument("--" + key, type=Path)
    parser.set_defaults(primary="matched", swap="swapped", baseline="baseline")
    args = parser.parse_args()
    root = Path("outputs/object-transfer")
    args.registration = args.registration or Path(
        f"outputs/object-fidelity/{args.panel}_ablation.json"
    )
    args.photo_report = (
        args.photo_report or root / "photo_residual" / args.panel / "sub-04/report.json"
    )
    args.selection = (
        args.selection
        or root / "support_structure_joint" / args.panel / args.subject / "selection.npz"
    )
    args.baseline_results = (
        args.baseline_results
        or root / "personal_render/personal_conditions" / args.panel / args.subject / "results.json"
    )
    result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".partial")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(f"Scored {args.subject} {args.panel} ({result['query_count']} queries): {args.out}")


if __name__ == "__main__":
    main()
