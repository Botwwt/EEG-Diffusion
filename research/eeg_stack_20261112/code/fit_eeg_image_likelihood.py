"""Fit a stimulus-dependent EEG compatibility score using genuine repeat errors.

python fit_eeg_image_likelihood.py --base outputs/official-object-readout/sub-03 --out OUTPUT
Use scorer = ImageLikelihood.load(OUTPUT / "likelihood.npz") and
scorer.score(query_eeg, per_query_candidate_visual_H14). Higher is better.
Input shapes [B,1024] and [B,K,1024], or [1024] and [K,1024] for a single
query. Image features must be OpenCLIP ViT-H-14 laion2b_s32b_b79k image_embeds.
Official test images enter only optional, separately identified oracle-gallery evaluation.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

PCA_DIM = 96
SEED = 20261127
GAMMAS = (0.0, 0.25, 1.0, 2.0)
MIX = (0.0, 0.5, 1.0)


def unit(x):
    length = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(length).all() or np.any(length == 0):
        raise ValueError("Nonfinite or zero-length feature")
    return x / length


def image_features(base, part):
    from official_object_readout import ASSETS, acquire

    name = f"text_{part}.pt"
    path = acquire(base / "assets" / name, ASSETS[name])
    with ZipFile(path) as z:
        (storage,) = [s for s in z.namelist() if s.endswith("/data/1")]
        return unit(np.frombuffer(z.read(storage), dtype="<f4").reshape(-1, 1024))


def repeat_variance(trials, means, basis):
    """Delta-method sampling variance of the normalized four-repeat EEG average."""
    out = np.empty((len(trials), basis.shape[1]), np.float32)
    for start in range(0, len(trials), 256):
        sl = slice(start, start + 256)
        avg = means[sl]
        norm = np.linalg.norm(avg, axis=1, keepdims=True).clip(1e-7)
        direction = avg / norm
        residual = trials[sl] - avg[:, None, :]
        radial = np.einsum("nrd,nd->nr", residual, direction)
        tangent = (residual @ basis - radial[:, :, None] * (direction @ basis)[:, None, :]) / norm[
            :, None, :
        ]
        out[sl] = np.square(tangent).sum(axis=1) / (4 * 3)
    return out


def inner(a, b):
    return np.einsum("...d,...d->...", a, b)


class ImageLikelihood:
    def __init__(self, values):
        self.values = values

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files})

    def prepare_visual(self, features):
        v = self.values
        x = unit(features.reshape(-1, 1024)).reshape(features.shape)
        centered = x - v["visual_center"]
        return x, centered @ v["mean_weight"], centered @ v["variation_weight"]

    def terms(self, queries, images, expected, variation, gamma):
        v = self.values
        projected = (queries - v["eeg_center"]) @ v["basis"]
        ratio = np.clip(1 + gamma * variation / v["global_variance"], 0.25, 4.0)
        # Constant global log determinant removed from every candidate. The
        # candidate-dependent log determinant remains and penalizes large variance.
        diff = np.square(projected[:, None, :] - expected)
        ll = -0.5 * np.mean(diff / (ratio * v["global_variance"]) + np.log(ratio), axis=-1)
        cosine = inner(queries[:, None, :], images)
        return cosine, ll

    def score(self, query_eeg, per_query_candidate_visual_H14, mode="selected"):
        v = self.values
        query = np.asarray(query_eeg, dtype=np.float32)
        features = np.asarray(per_query_candidate_visual_H14, dtype=np.float32)
        single = query.ndim == 1
        if single:
            query, features = query[None, :], features[None, :, :]
        if (
            query.ndim != 2
            or query.shape[1] != 1024
            or features.ndim != 3
            or features.shape[0] != len(query)
            or features.shape[2] != 1024
        ):
            raise ValueError("Expected [B,1024] EEG and [B,K,1024] image embeddings")
        options = {
            "selected": (float(v["gamma"]), float(v["mix"])),
            "global": (0.0, float(v["mix"])),
            "direct": (0.0, 0.0),
            "gaussian_global": (0.0, 1.0),
            "gaussian_image": (1.0, 1.0),
        }
        gamma, mix = options[mode]
        imgs, pred, change = self.prepare_visual(features)
        cosine, ll = self.terms(unit(query), imgs, pred, change, gamma)
        score = (1 - mix) * cosine / v["cosine_scale"] + mix * ll / v["likelihood_scale"]
        return score[0] if single else score


def fit_model(x, means, all_trials, groups):
    idx = groups.ravel()
    raw = means[idx]
    query = unit(raw)
    center = query.mean(0)
    q = query - center
    _, basis = np.linalg.eigh((q.T @ q).astype(np.float64))
    basis = np.ascontiguousarray(basis[:, -PCA_DIM:].astype(np.float32))
    z = q @ basis
    repeat = repeat_variance(all_trials[idx], raw, basis)
    vc = x[idx].mean(0)
    xc = x[idx] - vc
    eig, u = np.linalg.eigh((xc.T @ xc).astype(np.float64))
    eig = eig.clip(0)
    avg_eig = eig.mean()

    def ridge(y, strength):
        reg = strength * avg_eig
        w = (u @ ((u.T @ (xc.T @ y)) / (eig[:, None] + reg))).astype(np.float32)
        return w, np.sum(eig / (eig + reg))

    wm, df = ridge(z, 1.0)
    mu = xc @ wm
    # GCV diagonal approximation to out-of-sample variance of the whole
    # normalized EEG residual: conditional-mean error + four-repeat noise.
    # A centered repeat component modifies this total only on supported axes.
    v0 = np.square(z - mu).mean(0) / (1 - (df + 1) / len(z)) ** 2
    v0 = v0.clip(1e-9).astype(np.float32)
    wv, _ = ridge(repeat - repeat.mean(0), 5.0)
    other = np.roll(np.arange(len(idx)).reshape(-1, 10), 1, axis=1).ravel()
    cosine_scale = (inner(query, x[idx]) - inner(query, x[idx][other])).std()
    l0 = np.square(z - mu) / v0
    l1 = np.square(z - mu[other]) / v0
    ll_scale = (0.5 * (l1 - l0).mean(1)).std()
    return ImageLikelihood(
        {
            "visual_center": vc,
            "eeg_center": center,
            "basis": basis,
            "mean_weight": wm,
            "global_variance": v0,
            "variation_weight": wv,
            "repeat_mean": repeat.mean(0),
            "cosine_scale": np.float32(max(cosine_scale, 1e-6)),
            "likelihood_scale": np.float32(max(ll_scale, 1e-6)),
        }
    )


def stats(scores):
    diag = np.diagonal(scores, axis1=1, axis2=2)
    top = (scores.argmax(axis=2) == np.arange(scores.shape[-1])[None, :]).mean(1)
    i, j = np.triu_indices(scores.shape[-1], 1)
    margin = diag[:, i] + diag[:, j] - scores[:, i, j] - scores[:, j, i]
    swap = ((margin > 0) + 0.5 * (margin == 0)).mean(1)
    return top, swap


def assess(model, x, means, trials, held):
    v = model.values
    idx = held.ravel()
    n = len(held)
    pred_img, mu, var = model.prepare_visual(x[idx])
    query = unit(means[idx]).reshape(n, 10, -1)
    img = pred_img.reshape(n, 10, -1)
    proj = (query - v["eeg_center"]) @ v["basis"]
    mu = mu.reshape(n, 10, -1)
    var = var.reshape(n, 10, -1)
    cosine = query @ img.transpose(0, 2, 1) / v["cosine_scale"]
    sq = np.square(proj[:, :, None, :] - mu[:, None, :, :])
    mixtures = {}
    for gamma in GAMMAS:
        ratio = np.clip(1 + gamma * var / v["global_variance"], 0.25, 4)
        ll = -0.5 * np.mean(
            sq / (ratio[:, None, :, :] * v["global_variance"]) + np.log(ratio[:, None, :, :]),
            axis=-1,
        )
        ll /= v["likelihood_scale"]
        for mix in MIX:
            mixtures[(gamma, mix)] = stats((1 - mix) * cosine + mix * ll)
    # Same-concept shuffled variance leaves mean estimates and EEG untouched.
    ratio = np.clip(1 + np.roll(var, 1, axis=1) / v["global_variance"], 0.25, 4)
    placebo_ll = -0.5 * np.mean(
        sq / (ratio[:, None, :, :] * v["global_variance"]) + np.log(ratio[:, None, :, :]), axis=-1
    )
    mixtures["placebo"] = stats(placebo_ll / v["likelihood_scale"])
    obs = repeat_variance(trials[idx], means[idx], v["basis"]).reshape(var.shape)

    def centered(a):
        return a - a.mean(1, keepdims=True)

    vp, vt = centered(var), centered(obs)
    a, b = vp.ravel(), vt.ravel()
    within_corr = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    variance_r2 = float(
        1
        - np.square(obs - v["repeat_mean"] - var).sum(dtype=np.float64)
        / max(np.square(obs - v["repeat_mean"]).sum(dtype=np.float64), 1e-12)
    )
    return mixtures, {
        "observed_repeat_variance_r2_vs_train_mean": variance_r2,
        "within_concept_prediction_observation_corr": within_corr,
        "mean_abs_change_vs_total_global_variance": float(
            np.mean(np.abs(var) / v["global_variance"])
        ),
    }


def summarize(result):
    return {"intra10": float(result[0].mean()), "swap": float(result[1].mean())}


def query_gallery_eval(model, base):
    ref = image_features(base, "test")
    eeg = np.load(base / "eeg_r80.npy")
    modes = {}
    for name in ("direct", "global", "selected", "gaussian_global", "gaussian_image"):
        parts = []
        for start in range(0, len(eeg), 8):
            q = eeg[start : start + 8]
            parts.append(model.score(q, np.broadcast_to(ref, (len(q), *ref.shape)), name))
        matrix = np.concatenate(parts)
        rank = (matrix > matrix.diagonal()[:, None]).sum(1) + 1
        modes[name] = {
            "top1": int((rank == 1).sum()),
            "top5": int((rank <= 5).sum()),
            "median_rank": float(np.median(rank)),
            "query_to_candidate_2afc": float(np.mean((len(ref) - rank) / (len(ref) - 1))),
        }
    return modes


def main():
    from fit_official_eeg_anchors import TRAIN_EEG
    from official_object_readout import ASSETS, acquire, tensor_storage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--evaluate-reference-gallery",
        action="store_true",
        help="Evaluate frozen scorer on true test photos only after export",
    )
    args = parser.parse_args()
    base = args.base
    x = image_features(base, "train")
    trials = tensor_storage(acquire(base / "assets/eeg_train.pt", TRAIN_EEG), 66160, False).reshape(
        16540, 4, 1024
    )
    means = trials.mean(axis=1, dtype=np.float32)
    # Source labels are used for data splits and scoring only.
    path = acquire(base / "assets/metadata.npy", ASSETS["metadata.npy"])
    label = np.asarray(np.load(path, allow_pickle=True).item()["train_img_concepts"])
    groups = np.arange(len(label)).reshape(-1, 10)
    if (
        len(groups) != 1654
        or len(np.unique(label)) != len(groups)
        or not (label[groups] == label[groups[:, 0], None]).all()
    ):
        raise ValueError("Training groups differ from original concept/image order")
    fold = np.empty(len(groups), dtype=int)
    fold[np.random.default_rng(SEED).permutation(len(groups))] = np.arange(len(groups)) % 5
    dev = []
    calibration = []
    for k in (1, 2, 3, 4):
        train = groups[(fold != 0) & (fold != k)]
        model = fit_model(x, means, trials, train)
        scores, rep = assess(model, x, means, trials, groups[fold == k])
        dev.append(scores)
        calibration.append(rep)
        print(f"Training calibration fold {k}/4 finished", flush=True)

    def merged(k):
        return tuple(np.concatenate([v[k][axis] for v in dev]) for axis in (0, 1))

    def criteria(v):
        return (v[0].mean(), v[1].mean())

    mix = max(MIX, key=lambda m: criteria(merged((0.0, m))))
    gamma = max(GAMMAS, key=lambda g: criteria(merged((g, mix))))
    model = fit_model(x, means, trials, groups[fold != 0])
    scores, rep = assess(model, x, means, trials, groups[fold == 0])
    chosen = scores[gamma, mix]
    global_scores = scores[0.0, mix]
    delta = chosen[0] - global_scores[0]
    resampled = (
        np.random.default_rng(SEED + 1).choice(delta, (5000, len(delta)), replace=True).mean(1)
    )
    report = {
        "training_validation_seed": SEED,
        "pca_dim": PCA_DIM,
        "query_feature": "unit-norm four encoded trial mean; source sub-03",
        "selection": "Folds 1..4 concept OOF; maximize within-concept photo Top1, tiebreak swap, then simpler setting. Choose mix on global first and fix it for gamma.",
        "mixture_grid": list(MIX),
        "variance_grid": list(GAMMAS),
        "selected_mix": mix,
        "selected_gamma": gamma,
        "development": {f"gamma{g}_mix{m}": summarize(merged((g, m))) for g in GAMMAS for m in MIX},
        "dev_repeat_checks": calibration,
        "locked_fold0": {
            "concepts": int((fold == 0).sum()),
            "direct": summarize(scores[0.0, 0.0]),
            "global": summarize(global_scores),
            "selected": summarize(chosen),
            "pure_global_gaussian": summarize(scores[0.0, 1.0]),
            "pure_image_gaussian_gamma1": summarize(scores[1.0, 1.0]),
            "same_concept_shuffled_variation_gamma1": summarize(scores["placebo"]),
            "selected_minus_global_top1": float(delta.mean()),
            "concept_bootstrap_paired_95": np.quantile(resampled, [0.025, 0.975]).tolist(),
            "selected_minus_global_swap": float((chosen[1] - global_scores[1]).mean()),
            "repeat_error_check": rep,
        },
        "interpretation": "Heteroscedastic diagonal Gaussian compatibility in 96 train-EEG PCA coordinates. Repeat variance uses four actual encoded trials and a first-order normalization approximation. R80 uses a different averaging order and has no calibrated physical likelihood here.",
        "scope": "The frozen ATM encoder was originally trained on these official training images. Fold-0 and inner OOF split only the downstream likelihood fit. Official R80 reference-gallery evaluation after the export is a closed-set diagnostic; it is not generation.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    model = fit_model(x, means, trials, groups)
    model.values.update(mix=np.float32(mix), gamma=np.float32(gamma), format_revision=np.int32(1))
    np.savez(args.out / "likelihood.npz", **model.values)
    reload = ImageLikelihood.load(args.out / "likelihood.npz")
    np.testing.assert_allclose(
        reload.score(means[0], x[:4]), model.score(means[:1], x[None, :4])[0], rtol=5e-5, atol=1e-5
    )
    (args.out / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps(report["locked_fold0"]), flush=True)
    if args.evaluate_reference_gallery:
        oracle = query_gallery_eval(reload, base)
        (args.out / "evaluation_only_official_gallery.json").write_text(
            json.dumps(oracle, indent=2) + "\n", encoding="utf8"
        )
        print(json.dumps(oracle), flush=True)


if __name__ == "__main__":
    main()
