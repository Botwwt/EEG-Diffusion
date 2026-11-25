"""Test a support-image covariance adjustment to single-person EEG object decisions."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_shared_photo_eeg import require
from fit_support_covariance import eeg_features
from fit_support_structure import H14_SHA, unit
from fit_within_photo_embedding import images
from register_fresh_photo_panel import freeze, read, sha
from scipy.linalg import cho_factor, cho_solve, eigh
from test_cohort_session_matched import compact

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[2]
PUB = CODE.parent / "results/object_fidelity"
LOCK = PUB / "support_image_covariance_design.json"
DATA = REPO / "outputs"
A = SimpleNamespace(
    out=DATA / "session-matched-cohort",
    expanded=DATA / "photo-structure/expanded",
    double=DATA / "double-calibration",
)
PANELS = ("stress12", "new_three4")
SUBJECTS = ("sub-04", "sub-05", "sub-01")
METHODS = ("native", "image_covariance", "shuffled_covariance")
MENUS = PUB / "cohort_session_menus.json"
SPLIT = PUB / "visual_eeg_prototypes.json"
SEED = 261126


def lf(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_design():
    require(
        sha(LOCK) == "20b9d62a0a7ab2d65224c7a18190057f8b4e033a4eecdc65101057da89d63e52",
        "Registered design changed",
    )
    model = read(LOCK)
    expected = dict(
        menus_lf_sha256=lf(MENUS),
        split_lf_sha256=lf(SPLIT),
        registration_sha256=sha(A.out / "design.json"),
        generic_eeg_receipts={s: sha(A.expanded / "features" / (s + ".json")) for s in SUBJECTS},
        native_outputs={p: sha(A.out / p / "predictions.npz") for p in PANELS},
        visual_sha256=H14_SHA,
    )
    require(expected == model["sources"], "Registered source receipt mismatch")
    require(
        model["generic_test_objects"]
        == read(SPLIT)["generic"]["selection"]["generic_check_classes"],
        "Changed generic objects",
    )


def within(x):
    return (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])


def fit(classes, eeg, bank, test):
    nclass = len(classes)
    train = eeg.reshape(-1, eeg.shape[-1])
    proj = fisher(eeg, np.concatenate([train, test]))
    raw_train = proj[: len(train)].reshape(nclass, 10, 96)
    raw_query = proj[len(train) :]
    z = within(raw_train)
    cov = z.T @ z / (nclass * 9)
    ev, vec = eigh(cov)
    W = (vec / np.sqrt(0.9 * ev + 0.1 * ev.mean())) @ vec.T
    y = z @ W
    image_rows = classes[:, None] * 10 + np.arange(10)
    v = within(unit(bank[image_rows].astype(float)))
    sd = v.std(0).clip(1e-6)
    v /= sd
    gram = v.T @ v
    gram.flat[:: len(gram) + 1] += len(v)
    beta = cho_solve(cho_factor(gram, lower=True), v.T @ y)
    gain = np.sqrt(96 / np.mean(np.sum((v @ beta) ** 2, 1)))
    return raw_query, W, (beta / sd[:, None]) * gain


def classify(q, support_eeg, support_images, W, image_map):
    """All inputs are active-menu observations; the two query images are absent."""
    base = similarities(q, support_eeg)
    means = support_eeg.mean(1)
    x = unit(support_images.astype(float)) @ image_map
    x = (x - x.mean(1, keepdims=True)) * np.sqrt(0.10 / 7)
    # Woodbury uses one 8x8 solve per candidate. The isotropic term has already cancelled.
    r = (q[:, None] - means[None]) @ W
    corrections = []
    for shapes in (x, np.roll(x, 1, axis=0)):
        M = np.eye(8)[None] + shapes @ shapes.swapaxes(-1, -2)
        d = np.einsum("qkd,ksd->ksq", r, shapes)
        solved = np.linalg.solve(M, d)
        _, logdet = np.linalg.slogdet(M)
        correction = (np.einsum("ksq,ksq->qk", d, solved) - logdet[None]) / 2
        correction -= correction.mean(1, keepdims=True)
        correction /= correction.std(1, keepdims=True).clip(1e-8)
        corrections.append((base + 0.20 * correction).argmax(1))
    return np.stack((base.argmax(1), *corrections), axis=1)


def panel_predictions(bank, model, source_rows, reg, menus):
    raw, W, mapping = model
    rows, folds, cal = map(
        np.asarray, (reg["rows"], reg["folds"], reg["calibration_rows_by_fold_and_word"])
    )
    require(
        all(
            set(cal[f].ravel()) == set(rows[folds != f]) and np.unique(cal[f]).size == 96
            for f in range(5)
        ),
        "Support leaks active query rows",
    )
    alternatives = np.asarray(menus["alternatives"])
    assigned = np.asarray(menus["menu_at_ordinal"])
    out = np.full((120, 3), -1, dtype=int)
    for f in range(5):
        for g, choices in enumerate(alternatives):
            ix = np.flatnonzero((folds == f) & (assigned == g))
            allowed = cal[f, choices]
            require(
                len(ix) == 2 * len(choices)
                and len(set(allowed.ravel())) == 8 * len(choices)
                and not np.intersect1d(allowed, rows[folds == f]).size,
                "Active query photo in visual bank",
            )
            q = np.searchsorted(source_rows, rows[ix])
            s = np.searchsorted(source_rows, allowed)
            require(
                np.array_equal(source_rows[q], rows[ix])
                and np.array_equal(source_rows[s], allowed),
                "Misaligned support EEG",
            )
            out[ix] = choices[classify(raw[q], raw[s], bank[allowed], W, mapping)]
    require(np.all(out >= 0), "Missing query")
    return out


def generic_predictions(bank, classes, eeg):
    selected = read(SPLIT)["generic"]["selection"]
    ids = {c: i for i, c in enumerate(classes)}
    train_ids = np.array([v for fold in selected["heldout_folds"] for v in fold])
    test_ids = np.asarray(selected["generic_check_classes"])
    require(
        len(set(train_ids)) == 384
        and len(set(test_ids)) == 128
        and not set(train_ids) & set(test_ids),
        "Invalid generic split",
    )
    train = eeg[[ids[c] for c in train_ids]]
    query = eeg[[ids[c] for c in test_ids]].reshape(-1, 1003)
    q, W, mapping = fit(train_ids, train, bank, query)
    q = q.reshape(128, 10, 96)
    rng = np.random.default_rng(260601)
    counts = np.zeros((128, 3), int)
    truth = np.repeat(np.arange(12), 2)
    for _ in range(3):
        order = rng.permutation(128)
        for first in range(0, 128, 4):
            objects = np.roll(order, -first)[:12]
            schedule = np.stack([rng.permutation(10) for _ in objects])
            for fold in range(5):
                held = np.arange(10) // 2 == fold
                si = schedule[:, ~held]
                tgt = q[objects[:, None], schedule[:, held]].reshape(24, 96)
                supp = q[objects[:, None], si]
                support_rows = 10 * test_ids[objects, None] + si
                pred = classify(tgt, supp, bank[support_rows], W, mapping)
                counts[objects] += (pred == truth[:, None]).reshape(12, 2, 3).sum(1)
    return test_ids, counts


def stats(blocks, denoms):
    """Paired whole-object sampling, keeping every fixed subject and observation together."""
    total = sum(v.sum(0) for v in blocks)
    n = sum(np.sum(v) for v in denoms)
    rng = np.random.default_rng(SEED)
    bs = np.zeros((10000, 3))
    ns = np.zeros(10000)
    for b, ds in zip(blocks, denoms, strict=True):
        picks = rng.integers(len(b), size=(10000, len(b)))
        bs += b[picks].sum(1)
        ns += ds[picks].sum(1)
    bs /= ns[:, None]
    return dict(
        n=int(n),
        hits=dict(zip(METHODS, total.astype(int).tolist(), strict=True)),
        contrasts={
            name: dict(
                delta=float((total[1] - total[k]) / n),
                ci95=np.quantile(bs[:, 1] - bs[:, k], [0.025, 0.975]).tolist(),
            )
            for k, name in ((0, "candidate_minus_native"), (2, "candidate_minus_sham"))
        },
    )


def run(out):
    lock = LOCK
    verify_design()
    bank = images(DATA / "official-object-readout/sub-03")
    plan = read(A.out / "design.json")["panels"]
    menus = read(MENUS)["panels"]
    allpred = {}
    generic = []
    gclasses = None
    inputs = {}
    for sub in SUBJECTS:
        print("Fitting", sub, flush=True)
        classes, x = eeg_features(A.expanded, sub)
        gt, g = generic_predictions(bank, classes, x)
        require(gclasses is None or np.array_equal(gt, gclasses), "Generic check objects differ")
        gclasses = gt
        generic.append(g)
        qs = []
        rr = []
        inputs[sub] = {}
        for p in PANELS:
            specs = plan[p]
            candidates = np.asarray(specs["selected_concepts"])
            require(
                not np.intersect1d(candidates, classes).size,
                "Active test object appears in calibration",
            )
            xt, sig = compact(A, sub, p, candidates)
            require(
                read(A.out / p / "result.json")["target_feature_receipts"][sub] == sig,
                "Frozen target EEG differs",
            )
            raw = (10 * candidates[:, None] + np.arange(10)).ravel()
            order = np.argsort(raw)
            rr.append(raw[order])
            qs.append(xt[order])
            inputs[sub][p] = sig
        q, W, mapping = fit(classes, x, bank, np.concatenate(qs))
        for p, pq, rows in zip(PANELS, np.split(q, 2), rr, strict=True):
            reg = plan[p]["registration"]
            require(
                sha(A.out / "references" / (p + "_ablation.json"))
                == menus[p]["registration_sha256"],
                "Different frozen order or menu",
            )
            for name, menu in menus[p]["decisions"].items():
                pred = panel_predictions(bank, (pq, W, mapping), rows, reg, menu)
                with np.load(A.out / p / "predictions.npz", allow_pickle=False) as z:
                    require(
                        np.array_equal(pred[:, 0], z[name + "_" + sub])
                        and np.array_equal(reg["rows"], z["rows"]),
                        "Historical baseline mismatch",
                    )
                allpred[(sub, p, name)] = pred
    outpred = {k[0] + "/" + k[1] + "/" + k[2]: v.tolist() for k, v in allpred.items()}
    freeze(
        out / "predictions.json",
        dict(
            methods=list(METHODS),
            decisions=outpred,
            design_sha256=sha(lock),
            source_eeg_receipts=inputs,
        ),
    )
    # First use of panel truth for candidate accuracy, after every model's decisions are frozen.
    per = {}
    summary = {}
    blocks = {}
    for k, pred in allpred.items():
        sub, p, name = k
        t = np.asarray(plan[p]["registration"]["truth"])
        hit = pred == t[:, None]
        b = np.array([hit[t == i].sum(0) for i in range(12)])
        blocks[k] = b
        per["/".join(k)] = stats([b], [np.full(12, 10)])
    primary = [("stress12", "ordinary_12way"), ("new_three4", "within_matched_four")]
    for s in SUBJECTS:
        summary[s] = stats([blocks[(s, p, n)] for p, n in primary], [np.full(12, 10)] * 2)
    joint = [sum(blocks[(s, p, n)] for s in SUBJECTS) for p, n in primary]
    aggregate = stats(joint, [np.full(12, 30)] * 2)
    pooled = stats([sum(generic)], [np.full(128, 90 * 3)])
    check_generic = read(SPLIT)["generic"]["generic_check"]["hits"][0]
    require(
        int(generic[0][:, 0].sum()) == check_generic,
        "Original sub04 384/128 generic cosine did not replay",
    )
    advance = (
        all(k["ci95"][0] > 0 for k in aggregate["contrasts"].values())
        and sum(s["hits"]["image_covariance"] > s["hits"]["native"] for s in summary.values()) >= 2
    )
    freeze(
        out / "report.json",
        dict(
            design_sha256=sha(lock),
            predictions_sha256=sha(out / "predictions.json"),
            source_sha256=sha(Path(__file__)),
            scope="Three separately fitted people, fixed 24 previously studied target objects. Pooled target denominator counts 720 decisions on 240 unique original photos; clustered jointly by object and panel. Strict four-way receives a frozen candidate menu. Full twelve-way on those same images is separate.",
            per_case=per,
            primary_by_subject=summary,
            primary_pooled=aggregate,
            generic_by_subject={
                s: stats([g], [np.full(128, 90)]) for s, g in zip(SUBJECTS, generic, strict=True)
            },
            generic_pooled=pooled,
            generic_class_counts={s: g.tolist() for s, g in zip(SUBJECTS, generic, strict=True)},
            generic_classes=gclasses.tolist(),
            advance=bool(advance),
        ),
    )
    for k, v in {**per, **summary, "pooled": aggregate, "generic_pooled": pooled}.items():
        print(k, v)
    print("advance", advance)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DATA / "object-prior-replay/covariance")
    run(p.parse_args().out)
