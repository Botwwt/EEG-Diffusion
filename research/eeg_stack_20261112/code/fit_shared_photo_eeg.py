"""Estimate a within-object EEG subspace predictable across calibration participants."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from fit_support_structure import H14_SHA
from fit_within_photo_embedding import images
from scipy.linalg import cho_factor, cho_solve, eigh, solve_triangular

SUBJECTS = ("sub-01", "sub-04", "sub-05")
SEED = 260526


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(p, d):
    b = (json.dumps(d, indent=2, allow_nan=False) + "\n").encode("utf-8")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.name == "design.json" and p.exists():
        require(p.read_bytes() == b, "Registration differs from disk")
    p.write_bytes(b)


def registration(a):
    paths = {s: a.expanded / "features" / (s + ".json") for s in SUBJECTS}
    metas = {s: read(p) for s, p in paths.items()}
    sets = read(a.exclusions)["excluded_concept_ids"]
    ids = sorted(sets["personal_generic_512"])
    panels = {k: sorted(sets[k + "_objects"]) for k in ("registered", "replicate")}
    for k in ("fresh", "prospective"):
        panels[k] = sorted(read(a.public / (k + "_panel_design.json"))["selected_concept_ids"])
    used = set().union(*panels.values())
    require(
        len(used) == 48 and len(ids) == 512 and not used.intersection(ids),
        "Panel overlap or wrong class inventory",
    )
    for s, m in metas.items():
        require(
            m["subject"] == s
            and m["old_128_design_replay_exact"]
            and sorted(m["training_concepts"]) == ids
            and len(m["raw_block_sha256"]) >= 512
            and len(set(m["training_concepts"])) == 512,
            "Generic row inventory mismatch",
        )
    old = set.union(*(set(m["original_concepts"]) for m in metas.values()))
    gd = read(a.public / "ip_denoiser_gradient_design.json")
    # Preserve a new split outside old H14/gain development, UNet pilot and radius calibration.
    prior = (
        old
        | set(gd["eval_classes"])
        | set(gd["train_classes"])
        | set(sets["generic_visual_radius"])
    )
    eligible = sorted(set(ids) - prior)
    require(len(eligible) >= 96, "Insufficient independent generic objects")
    held = sorted(np.random.default_rng(SEED).choice(eligible, 96, replace=False).tolist())
    train = sorted(set(ids) - set(held))
    require(len(train) == 416, "Invalid training size")
    return dict(
        protocol="Generic photo-level EEG-response prediction across participants, supporting EEG available only for each participant's own eight calibration images. No target-panel waveform or image is used.",
        subjects=list(SUBJECTS),
        train_classes=train,
        test_classes=held,
        common_generic=512,
        eligible_test_classes=len(eligible),
        excluded_panels=panels,
        split_seed=SEED,
        source_row_mapping="Global THINGS original training row = 10*class + photo offset, 0..9; verified for all three physical feature stores.",
        estimators={
            "direct": "Established ridge=2.0, unit H14 targets, no output whitening. Both source and target subtract training object mean before residual fitting.",
            "shared": "Rank-64 reduced-rank ridge predicting both other participants' concatenated within-object standardized EEG from the current participant's standardized EEG. Each teacher enters at scale 1/sqrt(2). Source Gram regularization=0.5*n. Fit ridge=2.0 on unit-SD reduced scores to train H14 residual, then mix residual predictions 0.5 direct + 0.5 reduced.",
            "sham": "Identical model with each other participant's ten photo rows independently cyclically shifted by a nonzero offset per training object, fixing seed 260527. Class, EEG marginals and model dimensions are preserved.",
            "visual_rank": "Capacity check, using training visual residuals directly as the reduced-rank supervisory target and the identical rank/ridge/mixture.",
        },
        endpoint="For each of 96 unseen generic objects, evaluate all 45 within-class distinct photo pairs. Predict each photo's H14 unit direction from the person's EEG relative to their eight calibration EEGs and mean H14 of those same supports. Credit one-to-one assignment against the two original H14s, ties 0.5. Primary statistic averages the 45 credits within object and then the three fixed participants. The raw own-minus-swapped H14 cosine margin is secondary.",
        uncertainty="Resample 96 objects in synchrony across the three people and all models, 10000 times, seed 260528. Report paired percentile intervals and subject-specific counts. 45 pairs per object overlap, participants are fixed.",
        gate="Advance only if the lower paired 95% bound for shared minus direct AND shared minus shuffled are positive on the primary three-participant object-balanced assignment, the mean advantage over the capacity-matched visually supervised head is nonnegative and every participant's point estimate versus direct is positive. Otherwise stop. No target or diffusion sampling.",
        receipts={
            **{s: sha(p) for s, p in paths.items()},
            "old_exclusions": sha(a.exclusions),
            "h14_tensor": H14_SHA,
            **{
                k + "_panel": sha(a.public / (k + "_panel_design.json"))
                for k in ("fresh", "prospective")
            },
            "gradient_registration": sha(a.public / "ip_denoiser_gradient_design.json"),
        },
    )


def load_photos(a, s, classes):
    root = a.expanded / "features"
    meta = read(root / (s + ".json"))
    arrays = []
    for c in classes:
        with np.load(root / s / f"{c:04d}.npz", allow_pickle=False) as z:
            x = z["features"]
            require(
                int(z["start"]) == 10 * c
                and x.shape == (10, 1003)
                and x.dtype == np.float32
                and z["raw_sha"].item() == meta["raw_block_sha256"][str(c)]
                and hashlib.sha256(x.tobytes()).hexdigest() == z["feature_sha"].item()
                and np.isfinite(x).all(),
                f"Misregistered physical features {s} / {c}",
            )
            arrays.append(x)
    return np.stack(arrays).astype(np.float64)


def center(x):
    return x - x.mean(1, keepdims=True)


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    require(np.isfinite(norm).all() and (norm > 1e-10).all(), "Undefined image direction")
    return x / norm


def flat(x):
    return x.reshape(-1, x.shape[-1])


def standardized(x):
    x = center(x)
    sd = flat(x).std(0).clip(1e-6)
    return x / sd, sd


def ridge(gram, cross):
    d = gram.shape[0]
    return cho_solve(
        cho_factor(gram + 2.0 * np.eye(d), check_finite=False), cross, check_finite=False
    )


def reduced(gram, chol, cross_teacher, cross_visual):
    # Whiten the regularized source metric before selecting directions best
    # predicting the teacher. Image features enter only the final linear head.
    m = solve_triangular(chol, cross_teacher, lower=True, check_finite=False)
    k = m @ m.T
    v = eigh(k, subset_by_index=(len(k) - 64, len(k) - 1), check_finite=False)[1]
    p = solve_triangular(chol.T, v, lower=False, check_finite=False)
    pp = p.T @ gram @ p
    p /= np.sqrt(np.diag(pp)).clip(1e-6)
    return p @ ridge(p.T @ gram @ p, p.T @ cross_visual)


def fit_all(a, lock, image_bank):
    train = lock["train_classes"]
    physical = {s: standardized(load_photos(a, s, train)) for s in SUBJECTS}
    y = flat(
        center(unit(image_bank[10 * np.asarray(train)[:, None] + np.arange(10)].astype(float)))
    )
    n = len(y)
    rng = np.random.default_rng(260527)
    order = {
        s: (np.arange(10)[None, :] + rng.integers(1, 10, size=(len(train), 1))) % 10
        for s in SUBJECTS
    }
    for s in SUBJECTS:
        require(not np.any(order[s] == np.arange(10)[None, :]), "Null has unshuffled row")
        others = [t for t in SUBJECTS if t != s]
        x = flat(physical[s][0])
        gram = x.T @ x / n
        chol = np.linalg.cholesky(gram + 0.5 * np.eye(gram.shape[0]))
        cross = x.T @ y / n
        direct = ridge(gram, cross)
        heads = {"direct": direct}
        for method in ("shared", "sham", "visual_rank"):
            if method == "visual_rank":
                supervision = cross
            else:
                teachers = []
                for t in others:
                    xt = physical[t][0]
                    if method == "sham":
                        xt = xt[np.arange(len(train))[:, None], order[t]]
                    teachers.append(x.T @ flat(xt) / (n * np.sqrt(2.0)))
                supervision = np.concatenate(teachers, axis=1)
            heads[method] = 0.5 * direct + 0.5 * reduced(gram, chol, supervision, cross)
        np.savez(a.out / (s + ".npz"), scale=physical[s][1], **heads)
        print("Fitted", s, "on", len(train), "object classes", flush=True)


def score_person(a, s, held, actual):
    # Only this person's generic validation EEG enters their readout.
    with np.load(a.out / (s + ".npz"), allow_pickle=False) as model:
        x = load_photos(a, s, held) / model["scale"]
        result = {}
        # H14 directions are formed from each independent query + its own 8 supports.
        for method in ("direct", "shared", "sham", "visual_rank"):
            p = x @ model[method]
            margins = []
            for i, j in itertools.combinations(range(10), 2):
                use = np.delete(np.arange(10), [i, j])
                prior = actual[:, use].mean(1)
                offset = p[:, use].mean(1)
                first = unit(prior + p[:, i] - offset)
                second = unit(prior + p[:, j] - offset)
                margins.append(
                    0.5 * np.einsum("cd,cd->c", first - second, actual[:, i] - actual[:, j])
                )
            margin = np.stack(margins, axis=1)
            result[method] = margin
    np.savez(
        a.out / (s + "_generic_pairs.npz"),
        classes=np.asarray(held),
        offset_pairs=np.asarray(list(itertools.combinations(range(10), 2))),
        **result,
    )
    return result


def run(a, lock):
    bank = images(a.source)
    fit_all(a, lock, bank)
    fitted = {s: sha(a.out / (s + ".npz")) for s in SUBJECTS}
    write(
        a.out / "fit.json",
        dict(
            head_sha256=fitted,
            design_sha256=sha(a.out / "design.json"),
            script_sha256=sha(Path(__file__)),
        ),
    )
    held = lock["test_classes"]
    actual = unit(bank[10 * np.asarray(held)[:, None] + np.arange(10)].astype(float))
    scores = {s: score_person(a, s, held, actual) for s in SUBJECTS}
    methods = ("direct", "shared", "sham", "visual_rank")
    margins = {k: np.stack([scores[s][k] for s in SUBJECTS]) for k in methods}
    credit = {
        k: ((v > 1e-12).astype(float) + 0.5 * (np.abs(v) <= 1e-12)) for k, v in margins.items()
    }
    boot = np.random.default_rng(260528).integers(len(held), size=(10000, len(held)))

    def stat(a):
        require(a.shape == (len(held),), "Incorrect bootstrap unit")
        return dict(mean=float(a.mean()), ci95=np.percentile(a[boot].mean(1), [2.5, 97.5]).tolist())

    report = {}
    for k in methods:
        report[k] = dict(
            paired_assignment=stat(credit[k].mean((0, 2))),
            cosine_margin=stat(margins[k].mean((0, 2))),
            per_subject={
                s: dict(
                    correct_equivalent=float(credit[k][i].sum()),
                    ties=int((np.abs(margins[k][i]) <= 1e-12).sum()),
                    n=int(credit[k][i].size),
                )
                for i, s in enumerate(SUBJECTS)
            },
        )
    paired = {
        k: dict(
            assignment=stat((credit["shared"] - credit[k]).mean((0, 2))),
            margin=stat((margins["shared"] - margins[k]).mean((0, 2))),
            per_subject_assignment={
                s: stat((credit["shared"][i] - credit[k][i]).mean(1))
                for i, s in enumerate(SUBJECTS)
            },
        )
        for k in methods
        if k != "shared"
    }
    passed = (
        all(paired[k]["assignment"]["ci95"][0] > 0 for k in ("direct", "sham"))
        and paired["visual_rank"]["assignment"]["mean"] >= 0
        and all(d["mean"] > 0 for d in paired["direct"]["per_subject_assignment"].values())
    )
    write(
        a.out / "result.json",
        dict(
            protocol="Shared photograph-response supervision in generic calibration only",
            design_sha256=sha(a.out / "design.json"),
            script_sha256=sha(Path(__file__)),
            n_train_classes=416,
            n_test_classes=96,
            n_test_unique_photos=960,
            subjects=list(SUBJECTS),
            pairs_per_class=45,
            n_distinct_pairs=4320,
            n_participant_pair_assignments=12960,
            bootstrap_unit="96 shared object IDs, synchronously across the 3 fixed participants; 45 overlapping pairs per object",
            arms=report,
            shared_minus_control=paired,
            passed_generation_gate=bool(passed),
            head_sha256=fitted,
            pairwise_sha256={s: sha(a.out / (s + "_generic_pairs.npz")) for s in SUBJECTS},
        ),
    )
    print(
        "paired differences",
        json.dumps({k: v["assignment"] for k, v in paired.items()}),
        "advance",
        bool(passed),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "run"))
    for k, d in dict(
        out="outputs/shared-photo-eeg",
        expanded="outputs/photo-structure/expanded",
        source="outputs/official-object-readout/sub-03",
        public="research/eeg_stack_20261112/results/object_fidelity",
        exclusions="outputs/fresh-photo/registration/exclusion_sets.json",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(d))
    a = p.parse_args()
    lock = registration(a)
    path = a.out / "design.json"
    if a.stage == "register":
        write(path, lock)
        print("Locked 416 training, 96 held-out:", sha(path))
    else:
        require(read(path) == lock, "Data/model registration mismatch")
        run(a, lock)


if __name__ == "__main__":
    main()
