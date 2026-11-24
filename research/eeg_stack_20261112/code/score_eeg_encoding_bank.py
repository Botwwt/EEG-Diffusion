"""Fit a participant-specific forward image-to-EEG encoder and select a generated JPEG."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from decode_eeg_layout import eeg_design
from fit_personal_session_metric import load
from fit_shared_photo_eeg import (
    SUBJECTS,
    center,
    flat,
    load_photos,
    read,
    require,
    sha,
    unit,
    write,
)
from fit_support_structure import H14_SHA
from fit_within_photo_embedding import images
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from score_reconstruction_jpegs import registered
from select_within_photo_candidates import PANELS, bank_metadata, candidate_features
from select_within_photo_candidates import unit as selection_unit

ARMS = ("encoding", "wrong_eeg", "mean_eeg", "old_h14", "old_uniform", "fixed0")


def design(a):
    bank = a.public / "generated_bank_inputs.json"
    old = a.public / "generated_bank_dino.json"
    receipts = {s: sha(a.expanded / "features" / (s + ".json")) for s in SUBJECTS}
    return dict(
        protocol="Post-hoc development test on the two previous generated-image banks, fixed before new selection or scoring.",
        subjects=list(SUBJECTS),
        panels=list(PANELS),
        generator="Same existing eight generated JPEGs per object and fold. 960 candidates shared by all three people. Each query can use only its frozen prospective-at-the-time predicted object, fold, eight supports and registered donor.",
        model="One encoder per participant. Train on 512 generic objects / 5120 photos. For unit H14 training image directions and the 1003 posterior 17-channel personal R4 EEG design, subtract each object's ten-photo average. Scale each feature by its training residual SD (floor 1e-6). Fit image to EEG ridge with penalty 2*n. Estimate the covariance of the training EEG residuals about the fitted encoder; use one half empirical covariance and one half its diagonal. No candidate or evaluation photo, word, EEG or metric selects a parameter.",
        selection="For each synthetic candidate, subtract the unnormalized mean of the eight permitted unit support H14 directions, apply train-only scaling and fitted encoder. Subtract mean EEG of those same eight supports from the participant's query EEG and scale with their train-only SD. Select the maximum Gaussian log score (common shrinkage covariance): score=e'P*mu-0.5*mu'P*mu. Smallest index wins ties. No EEG from another participant.",
        controls="Same eight JPEGs and allowed supports for every arm: wrong_eeg uses the registered same-word/same-fold donor; mean_eeg sets residual EEG e=0, holding the scoring rule fixed. old_h14 and old_uniform exactly replay the previously frozen H14 residual and support-mean picks; fixed0 always chooses index zero.",
        evaluation="Freeze all 720 choices per arm before opening evaluation-only DINO scores. Raw unit DINOv2-S/14 cosine ten-photo Top1, full registered denominators; secondary own-original versus best other margin, per-query own-vs-same-fold sibling, correct-word subset and distinct same-truth-and-word donor subgroup. The two panels and subjects are fixed. Pool all 24 object means and resample those classes synchronously with all subjects, 10000 times, seed 260530.",
        gate="Further investigation only if the lower paired 95% bootstrap bound for DINO ten-photo Top1 is positive against EACH of wrong_eeg, mean_eeg, and old_h14 on all 720 decisions. No tuning on the development panels; do not call this external confirmation.",
        old_bank_receipt_sha256=sha(bank),
        bank_design_sha256=sha(a.bank / "design.json"),
        historical_dino_report_sha256=sha(old),
        dino_cache_sha256=read(old)["feature_file_sha256"],
        generic_h14_sha256=H14_SHA,
        generic_training_receipts=receipts,
    )


def encoder(a, s, visual):
    classes = np.array(read(a.expanded / "features" / (s + ".json"))["training_concepts"])
    require(len(np.unique(classes)) == 512, "Incomplete generic training classes")
    x = flat(center(unit(visual[10 * classes[:, None] + np.arange(10)].astype(float))))
    y = flat(center(load_photos(a, s, classes)))
    sd_x = x.std(0).clip(1e-6)
    sd_y = y.std(0).clip(1e-6)
    x /= sd_x
    y /= sd_y
    n = len(x)
    w = cho_solve(
        cho_factor(x.T @ x + 2 * n * np.eye(x.shape[1]), check_finite=False),
        x.T @ y,
        check_finite=False,
    )
    eps = y - x @ w
    cov = eps.T @ eps / n
    chol = np.linalg.cholesky(0.5 * (cov + np.diag(np.diag(cov))))
    # In the whitened output space the shared-covariance log likelihood is a dot minus half a squared norm.
    w = solve_triangular(chol, w.T, lower=True, check_finite=False).T
    return classes, sd_x, sd_y, w, chol


def select(a):
    visual = images(a.source)
    old = read(a.public / "generated_bank_dino.json")["subjects"]
    banks = {}
    for panel in PANELS:
        folder, records, words, prior, support, meta = bank_metadata(a, panel)
        cache_args = SimpleNamespace(out=a.bank / "selected", device="cuda", batch_size=8)
        require(
            (cache_args.out / "features" / (panel + ".npz")).is_file(),
            "Need the archived H14 candidate cache",
        )
        features, signature = candidate_features(cache_args, panel, folder, records, meta, [])
        banks[panel] = (records, words, prior, support, features, signature)
    receipts = {}
    for s in SUBJECTS:
        classes, sx, sy, w, chol = encoder(a, s, visual)
        source = a.inputs / "halves" / (s + ".npz")
        rawrows, _, _, mean = load(source)
        eeg = eeg_design(mean, np.arange(len(mean)), np.arange(17)).astype(float)
        lookup = {r: i for i, r in enumerate(rawrows)}
        require(eeg.shape == (240, 1003), "Incomplete R4 EEG")
        receipts[s] = {}
        for panel in PANELS:
            records, bank_words, prior, cal, features, feature_sha = banks[panel]
            p = a.conditions / s / panel / "conditions_512.npz"
            cfgfile = a.bank / "selected/renders" / panel / s / "settings.json"
            cfg = read(cfgfile)
            historical = old[s]["panels"][panel]
            require(
                sha(cfgfile) == historical["settings_sha256"]
                and sha(p) == historical["packet_sha256"] == cfg["packet_sha256"]
                and cfg["candidate_feature_sha256"] == feature_sha,
                "Archived choices changed",
            )
            with np.load(p, allow_pickle=False) as z:
                rows, folds, words, donors, support = (
                    z[k] for k in ("rows", "folds", "predicted_words", "donor_rows", "support_rows")
                )
                native, uniform = z["own"].astype(float), z["uniform"].astype(float)
                require(
                    str(z["target_eeg_sha256"]) == sha(source)
                    and set(z["training_classes"]) == set(classes)
                    and not np.intersect1d(classes, rows // 10).size,
                    "Participant EEG or training set mismatch",
                )
            require(
                all(
                    np.array_equal(v, cfg[k])
                    for v, k in (
                        (rows, "rows"),
                        (folds, "folds"),
                        (words, "predicted_words"),
                        (donors, "donor_rows"),
                        (support, "support_rows"),
                    )
                ),
                "Case alignment differs",
            )
            cols = np.array([bank_words.tolist().index(word) for word in words])
            dirs = selection_unit(features[folds, cols])
            prior_unit = selection_unit(prior[folds, cols])
            off = np.array([[lookup[r] for r in use] for use in support])
            ix = np.array([lookup[r] for r in rows])
            pos = {r: i for i, r in enumerate(rows)}
            di = np.array([pos[r] for r in donors])
            require(
                np.array_equal(np.sort(support, axis=1), np.sort(cal[folds, cols], axis=1))
                and np.array_equal(folds, folds[di])
                and np.array_equal(words, words[di])
                and np.array_equal(support, support[di]),
                "Wrong donor or calibration object",
            )
            for f in range(5):
                require(
                    not np.intersect1d(rows[folds == f], support[folds == f]).size,
                    "A held-out photo leaked into the supports",
                )
            center_visual = unit(visual[support].astype(float)).mean(1)
            require(
                np.allclose(unit(center_visual), uniform, atol=1e-6, rtol=0),
                "Support H14 reference mismatch",
            )
            observed = solve_triangular(
                chol, ((eeg[ix] - eeg[off].mean(1)) / sy).T, lower=True, check_finite=False
            ).T
            mu = ((dirs - center_visual[:, None, :]) / sx) @ w
            penalty = -0.5 * np.sum(np.square(mu), axis=-1)
            old_scores = np.einsum(
                "nkd,nd->nk",
                selection_unit(dirs - prior_unit[:, None]),
                selection_unit(selection_unit(native) - prior_unit),
            )
            ranking = dict(
                encoding=np.einsum("nkd,nd->nk", mu, observed) + penalty,
                wrong_eeg=np.einsum("nkd,nd->nk", mu, observed[di]) + penalty,
                mean_eeg=penalty,
                old_h14=old_scores,
                old_uniform=np.einsum("nkd,nd->nk", dirs, prior_unit),
            )
            picks = {k: v.argmax(1) for k, v in ranking.items()}
            picks["fixed0"] = np.zeros(len(rows), int)
            require(
                np.array_equal(picks["wrong_eeg"], picks["encoding"][di])
                and all(
                    np.array_equal(picks[k], cfg["candidate_indices"][oldk])
                    for k, oldk in (
                        ("old_h14", "own"),
                        ("old_uniform", "uniform"),
                        ("fixed0", "fixed0"),
                    )
                ),
                "Legacy or exchanged-EEG selections differ",
            )
            hashes = np.array([r["sha256"] for r in records]).reshape(5, 12, 8)[folds, cols]
            dest = a.out / panel / s / "choices.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                dest, rows=rows, words=words, folds=folds, donor_rows=donors, hashes=hashes, **picks
            )
            receipts[s][panel] = dict(
                choices_sha256=sha(dest),
                candidate_h14_sha256=feature_sha,
                prior_packet_sha256=sha(p),
                old_selection_sha256=sha(cfgfile),
                source_eeg_sha256=sha(source),
            )
    write(
        a.out / "choices.json",
        dict(
            design_sha256=sha(a.out / "design.json"),
            driver_sha256=sha(Path(__file__)),
            cases=receipts,
        ),
    )
    print("Frozen all 720 queries for all six methods before reading DINO cache", flush=True)


def score(a):
    lock = read(a.out / "choices.json")
    require(lock["design_sha256"] == sha(a.out / "design.json"), "Selection lock mismatch")
    path = a.bank / "dino_bank.npz"
    old = read(a.public / "generated_bank_dino.json")
    require(sha(path) == old["feature_file_sha256"], "DINO cache differs")
    values = {k: [] for k in ARMS}
    masks = {k: [] for k in ("all", "correct_word", "same_truth_distinct_donor")}
    cluster = []
    individual = {}
    head_counts = []
    with np.load(path, allow_pickle=False) as z:
        hashes = z["jpeg_sha256"]
        ix = {h: i for i, h in enumerate(hashes)}
        require(
            len(ix) == 960
            and str(z["model_sha"]) == old["model_sha256"]
            and str(z["backend"]) == "dino",
            "Wrong independent scorer",
        )
        generated = unit(z["features"].astype(float))
        for panel in PANELS:
            r = registered(a.references, panel)
            rows, labels, folds = r["rows"], r["labels"], r["folds"]
            target = unit(z[panel].astype(float))
            require(np.array_equal(z[panel + "_rows"], rows), "Original DINO row mismatch")
            for s in SUBJECTS:
                dest = a.out / panel / s / "choices.npz"
                require(
                    sha(dest) == lock["cases"][s][panel]["choices_sha256"],
                    "Selection changed after scoring started",
                )
                with np.load(dest, allow_pickle=False) as packet:
                    require(
                        np.array_equal(packet["rows"], rows)
                        and np.array_equal(packet["folds"], folds),
                        "Different images",
                    )
                    word = packet["words"]
                    donor = packet["donor_rows"]
                    row_index = {v: i for i, v in enumerate(rows)}
                    di = np.array([row_index[v] for v in donor])
                    masks["correct_word"].extend((labels == word).tolist())
                    masks["all"].extend(np.ones(len(rows), bool))
                    masks["same_truth_distinct_donor"].extend(
                        ((labels == labels[di]) & (rows != donor)).tolist()
                    )
                    cluster.extend([panel + "/" + str(k) for k in labels])
                    ind = {}
                    for arm in ARMS:
                        chosen = packet["hashes"][np.arange(len(rows)), packet[arm]]
                        scores = generated[[ix[h] for h in chosen]] @ target.T
                        obs = []
                        for i in range(len(rows)):
                            gallery = np.flatnonzero(labels == labels[i])
                            comp = gallery[gallery != i]
                            rival = comp[folds[comp] == folds[i]].item()
                            val = scores[i, i]
                            best = scores[i, gallery].max()
                            d = val - scores[i, rival]
                            obs.append(
                                [
                                    float(val == best) / np.sum(scores[i, gallery] == best),
                                    val - scores[i, comp].max(),
                                    float(d > 1e-12) + 0.5 * float(abs(d) <= 1e-12),
                                    val,
                                ]
                            )
                        matrix = np.array(obs)
                        values[arm].append(matrix)
                        ind[arm] = (
                            int(matrix[:, 0].sum())
                            if matrix[:, 0].sum().is_integer()
                            else float(matrix[:, 0].sum())
                        )
                        if arm in ("old_h14", "old_uniform", "fixed0"):
                            hist = {"old_h14": "own", "old_uniform": "uniform", "fixed0": "fixed0"}[
                                arm
                            ]
                            require(
                                old["subjects"][s]["panels"][panel]["metrics"][
                                    "within_class_photo"
                                ]["arms"][hist]["credit"]
                                == ind[arm],
                                "Old independent Top1 score differs",
                            )
                    individual[panel + "/" + s] = ind
                    head_counts.append(np.mean(packet["encoding"] == packet["old_h14"]))
    labels = np.array(cluster)
    vocab = np.unique(labels)
    require(len(vocab) == 24, "Expected 24 evaluation objects")
    boot = np.random.default_rng(260530).integers(24, size=(10000, 24))
    matrices = {k: np.vstack(v) for k, v in values.items()}

    def summary(v, cls):
        den = np.array([(cls == k).sum() for k in vocab])[boot].sum(1)
        total = np.stack([v[cls == k].sum(0) for k in vocab])[boot].sum(1)
        ci = np.percentile(total[den > 0] / den[den > 0, None], [2.5, 97.5], axis=0)
        return {
            k: dict(mean=float(v[:, i].mean()), ci95=ci[:, i].tolist())
            for i, k in enumerate(
                (
                    "intra10_top1",
                    "self_minus_best_rival_cosine",
                    "same_fold_two_choice",
                    "target_cosine",
                )
            )
        }

    table = {}
    for name, m in masks.items():
        m = np.array(m)
        cls = labels[m]
        by = {k: summary(v[m], cls) for k, v in matrices.items()}
        diff = {
            k: summary((matrices["encoding"] - v)[m], cls)
            for k, v in matrices.items()
            if k != "encoding"
        }
        table[name] = dict(
            n=int(m.sum()),
            credits={k: float(v[m, 0].sum()) for k, v in matrices.items()},
            arms=by,
            encoding_minus=diff,
        )
    passed = all(
        table["all"]["encoding_minus"][k]["intra10_top1"]["ci95"][0] > 0
        for k in ("wrong_eeg", "mean_eeg", "old_h14")
    )
    result = dict(
        design_sha256=sha(a.out / "design.json"),
        choice_receipt_sha256=sha(a.out / "choices.json"),
        dino_file_sha256=sha(path),
        driver_sha256=sha(Path(__file__)),
        separate_cases=individual,
        groups=table,
        passed_gate=bool(passed),
        encoded_choice_matches_old_h14=float(np.mean(head_counts)),
        n_unique_candidates=960,
        n_distinct_photos=240,
        n_participant_queries=720,
        scope="Two previously investigated 12-concept panels, three fixed participants, one of exactly eight shared synthetic images selected per query. DINO and original galleries were accessed after freezing choices.",
    )
    write(a.out / "result.json", result)
    print("DINO Top1 credits", table["all"]["credits"], "passed", passed, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "select", "score"))
    for k, v in dict(
        out="outputs/eeg-encoding-bank",
        bank="outputs/photo-structure/h14_bank",
        expanded="outputs/photo-structure/expanded",
        source="outputs/official-object-readout/sub-03",
        conditions="outputs/photo-structure/within_photo",
        inputs="outputs/object-transfer",
        public="research/eeg_stack_20261112/results/object_fidelity",
        references="outputs/object-fidelity",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    a = p.parse_args()
    d = design(a)
    path = a.out / "design.json"
    if a.stage == "register":
        write(path, d)
        print("Design SHA256", sha(path))
    else:
        require(read(path) == d, "Frozen inputs/model changed")
        (select if a.stage == "select" else score)(a)


if __name__ == "__main__":
    main()
