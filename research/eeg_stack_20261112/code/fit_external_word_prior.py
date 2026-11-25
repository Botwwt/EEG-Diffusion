"""Fit and assess a full-vocabulary external photo prior using generic EEG only."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from calibrate_official_repeat_angles import expmap, tangent
from fit_official_repeat_objects import fit, read, require, sha, standard, training, unit
from fit_official_visual_anchors import prototype_fits
from fit_soft_class_residual import assess
from readout_external_images import __file__ as fingerprint_source
from readout_external_images import image_inputs, matches
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
LOCK = CODE.parent / "results/object_fidelity/external_word_prior_design.json"
DESIGN = "a06eb5dad878d076d1d59762dfbed05df96a1b5075e082390a42aba73fda21ab"
OUT = ROOT / "outputs"
BASE = OUT / "official-repeat-object"
BETAS = (0.25, 0.5, 1.0)


def write(out, name, obj):
    (out / name).write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )


def kernel_prototypes(ref, train, target):
    gram = ref @ ref.T
    _, values = next(prototype_fits((gram + gram**3) * 0.5, train, target, [0.5]))
    return values


def log_unit(v, axis):
    return tangent(np.asarray(v, dtype=np.float64), axis)


def compose(direction, theta, axis):
    n = np.linalg.norm(direction, axis=1)
    require(np.min(n) > 1e-10, "Indeterminate direction")
    return np.cos(theta)[:, None] * axis + np.sin(theta)[:, None] * direction / n[:, None]


def rank(v, labels):
    correct = v[np.arange(len(v)), labels]
    return (
        1
        + (v > correct[:, None]).sum(axis=1)
        + ((v == correct[:, None]) & (np.arange(v.shape[1]) < labels[:, None])).sum(axis=1)
    )


def metrics(estimated, real, text, ids):
    x = estimated.astype(np.float64)
    primary = assess(x, real.astype(np.float64))
    cosine = x @ text.T
    labels = np.repeat(ids, 10)
    full = rank(cosine, labels)
    restricted = rank(cosine[:, ids], np.repeat(np.arange(len(ids)), 10))
    return primary | dict(
        text1854_top1=full == 1, text1854_top5=full <= 5, text64_top1=restricted == 1
    )


def limits(x, boot):
    c = np.asarray(x).reshape(64, 10).mean(1)
    return dict(mean=float(c.mean()), ci95=np.quantile(c[boot].mean(1), [0.025, 0.975]).tolist())


def posterior_diagnostics(probabilities, ids, cal_conf):
    labels = np.repeat(ids, 10)
    full = rank(probabilities, labels)
    limit = rank(probabilities[:, ids], np.repeat(np.arange(64), 10))
    maxp = probabilities.max(1)
    boundaries = np.quantile(cal_conf, [0.2, 0.4, 0.6, 0.8])
    bins = np.searchsorted(boundaries, maxp, side="right")
    groups = []
    ece = 0
    for k in range(5):
        used = bins == k
        if used.any():
            p = float(maxp[used].mean())
            acc = float((full[used] == 1).mean())
            ece += used.mean() * abs(p - acc)
            groups.append(
                dict(
                    bin=k,
                    n=int(used.sum()),
                    average_confidence=p,
                    exact_hits=int((full[used] == 1).sum()),
                    top5_hits=int((full[used] <= 5).sum()),
                )
            )
    return dict(
        n=640,
        full1854_top1=int(sum(full == 1)),
        full1854_top5=int(sum(full <= 5)),
        names64_top1=int(sum(limit == 1)),
        nll=float(-np.log(probabilities[np.arange(640), labels]).mean()),
        uniform_nll=float(np.log(1854)),
        multiclass_brier=float(
            (1 + (probabilities**2).sum(1) - 2 * probabilities[np.arange(640), labels]).mean()
        ),
        mean_effective_labels=float(
            np.exp(-(probabilities * np.log(probabilities.clip(1e-300))).sum(1)).mean()
        ),
        confidence_bin_edges_from_calibration=boundaries.tolist(),
        mean_top1_confidence=float(maxp.mean()),
        ece_top1=float(ece),
        bins=groups,
    )


def inputs(out):
    require(sha(LOCK) == DESIGN, "Changed registered design")
    design = read(LOCK)
    old = read(BASE / "design.json")
    split = read(BASE / "angles/design.json")
    audit = read(out / "collision_audit.json")
    require(
        audit["design_sha256"] == DESIGN
        and audit["split_sha256"]
        == design["source_split_sha256"]
        == sha(BASE / "angles/design.json")
        and audit["comparisons"] == 1854 * 1280,
        "Collision audit mismatch",
    )
    ac, av = (np.array(split[k]) for k in ("angle_calibration_classes", "angle_validation_classes"))
    classes = np.setdiff1d(old["extended_classes"], np.r_[ac, av])
    require(
        len(classes) == 884 and len(set(ac) | set(av)) == 128 and len(ac) == len(av) == 64,
        "Incorrect generic split",
    )
    a = SimpleNamespace(
        source=OUT / "official-object-readout/sub-03",
        expanded=OUT / "photo-structure/expanded",
        extra=OUT / "double-calibration",
        external=OUT / "external-object-anchors/sub-03/external_anchors.npz",
    )
    data = training(a, old)
    blocks, visual, names, ids, _, text, external = data
    centers = unit(visual[classes].reshape(-1, 1024)).reshape(len(classes), 10, 1024).mean(axis=1)
    external = external.copy()
    flags = np.isin(ids, audit["flagged_anchor_ids"])
    if flags.any():
        external[flags] = unit(kernel_prototypes(text, names[classes], centers)[flags])
    data = (*data[:-1], external)
    bank = unit(0.8 * kernel_prototypes(external, names[classes], centers) + 0.2 * external)
    # Calibration/validation inputs share one frozen forward model. Official test inputs are never loaded.
    hc = np.r_[ac, av]
    x = np.concatenate([blocks[c] for c in hc])
    scores, _, native = fit(classes, data, x)
    logits = standard(scores["joint_32"]).astype(np.float64)
    native_axis = native["mean"].astype(np.float64)
    axis = native_axis / np.linalg.norm(native_axis)
    g = read(BASE / "angles/calibration.json")["parameters"]
    raw = native["mean"] + native["ridge"]
    global_x = expmap(log_unit(raw, axis) * g["global_gain"], axis, g["real_angle_cap_radians"])
    gt = log_unit(global_x, axis)
    return dict(
        ac=ac,
        av=av,
        train=classes,
        labels=names[hc],
        logits=logits,
        bank=bank,
        axis=axis,
        gt=gt,
        global_x=global_x,
        visual=visual,
        text=text,
        ids=ids,
    )


def calibrate(d, out):
    labels = np.repeat(d["labels"][:64], 10)
    logits = d["logits"][:640]

    def objective(t):
        return -log_softmax(logits / t, axis=1)[np.arange(640), labels].mean()

    result = minimize_scalar(
        objective, bounds=(0.15, 4.0), method="bounded", options={"xatol": 1e-9}
    )
    require(result.success, "Temperature calibration failed")
    temp = float(result.x)
    probs = np.exp(log_softmax(logits / temp, axis=1))
    p = log_unit(probs @ d["bank"], d["axis"])
    avg = p.mean(0)
    gt = d["gt"][:640]
    theta = np.linalg.norm(gt, axis=1)
    scale = float(np.median(theta) / np.median(np.linalg.norm(p - avg, axis=1)))
    direction = (p - avg) * scale
    real = unit(d["visual"][d["ac"]].reshape(-1, 1024))
    grid = {}
    for beta in BETAS:
        v = compose(gt + beta * direction, theta, d["axis"])
        rows = assess(v, real)
        grid[beta] = [int(rows[k].sum()) for k in ("object_top1", "intra10_top1", "photo_top1")]
    chosen = max(BETAS, key=lambda k: (sum(grid[k][:2]), grid[k][2], -k))
    np.savez(
        out / "calibration.npz",
        mean_prior=avg,
        scale=scale,
        temp=temp,
        beta=chosen,
        calibration_confidence=probs.max(1),
    )
    write(
        out,
        "calibration.json",
        dict(
            design_sha256=DESIGN,
            audit_sha256=sha(out / "collision_audit.json"),
            condition_parameters_sha256=sha(out / "calibration.npz"),
            n=640,
            temperature=temp,
            nll=float(result.fun),
            prior_scale=scale,
            beta=chosen,
            calibration_counts_by_beta=grid,
            columns=["object64", "within10", "all640"],
        ),
    )
    print("Frozen calibration", temp, chosen, grid, flush=True)


def validate(d, out):
    calibration = read(out / "calibration.json")
    require(
        calibration["design_sha256"] == DESIGN
        and calibration["condition_parameters_sha256"] == sha(out / "calibration.npz")
        and calibration["audit_sha256"] == sha(out / "collision_audit.json"),
        "Calibration receipt changed",
    )
    with np.load(out / "calibration.npz", allow_pickle=False) as z:
        avg = z["mean_prior"]
        scale = float(z["scale"])
        temp = float(z["temp"])
        beta = float(z["beta"])
        cal_conf = z["calibration_confidence"]
    probs = np.exp(log_softmax(d["logits"][640:] / temp, axis=1))
    gt = d["gt"][640:]
    axis = d["axis"]
    theta = np.linalg.norm(gt, axis=1)
    prior = scale * (log_unit(probs @ d["bank"], axis) - avg)
    own = compose(gt + beta * prior, theta, axis)
    pred_names = d["ids"][probs.argmax(1)]
    # Freeze matched inputs and words before target photographs enter the evaluator.
    np.savez(
        out / "matched_conditions.npz",
        rows=(d["av"][:, None] * 10 + np.arange(10)).ravel(),
        own=own.astype(np.float32),
        global_angle=d["global_x"][640:],
        predicted_words=pred_names,
        probabilities=probs.astype(np.float32),
    )
    ids = np.arange(640).reshape(64, 10)
    wrong_prior = np.roll(ids, 17, axis=0).ravel()
    wrong_cont = np.roll(ids, -1, axis=1).ravel()
    arms = dict(
        global_angle=d["global_x"][640:],
        matched=own,
        prior_only=compose(prior, theta, axis),
        wrong_class_prior=compose(gt + beta * prior[wrong_prior], theta, axis),
        wrong_within_continuous=compose(
            gt[wrong_cont] / theta[wrong_cont, None] * theta[:, None] + beta * prior, theta, axis
        ),
    )
    real = unit(d["visual"][d["av"]].reshape(-1, 1024))
    val_ids = d["labels"][64:]
    stats = {k: metrics(v, real, d["text"], val_ids) for k, v in arms.items()}
    boot = np.random.default_rng(260550).integers(64, size=(10000, 64))
    summary = {
        k: {
            metric: dict(mean=float(np.mean(v)), hits=float(np.sum(v)))
            if metric.endswith(("top1", "top5"))
            else dict(mean=float(np.mean(v)))
            for metric, v in group.items()
        }
        for k, group in stats.items()
    }
    differences = {
        k: {
            metric: limits(stats["matched"][metric].astype(float) - v.astype(float), boot)
            for metric, v in stats[k].items()
        }
        for k in stats
        if k != "matched"
    }
    dglob = differences["global_angle"]
    gate = dict(
        object64_lcb_positive=dglob["object_top1"]["ci95"][0] > 0,
        within10_nondecrease=dglob["intra10_top1"]["mean"] >= 0,
        all640_nondecrease=dglob["photo_top1"]["mean"] >= 0,
        beats_wrong_prior_object64=differences["wrong_class_prior"]["object_top1"]["ci95"][0] > 0,
        beats_wrong_eeg_within10=differences["wrong_within_continuous"]["intra10_top1"]["ci95"][0]
        > 0,
    )
    write(
        out,
        "result.json",
        dict(
            design_sha256=DESIGN,
            calibration_sha256=sha(out / "calibration.json"),
            matched_conditions_sha256=sha(out / "matched_conditions.npz"),
            code_sha256=sha(Path(__file__)),
            photos=640,
            objects=64,
            results=summary,
            matched_minus=differences,
            interval="10000 paired whole-object bootstraps, fixed validation classes/participant; deltas in probability or cosine units.",
            full_eeg_posterior=posterior_diagnostics(probs, val_ids, cal_conf),
            gate=gate,
            advance=all(gate.values()),
            interpretation="Class prototype identification uses nine real photos per candidate at scoring only. H14 text naming uses external text vectors at scoring only. All inference ranks all 1854 words, uses no query support and no official test asset.",
        ),
    )
    np.savez(
        out / "evaluation_rows.npz",
        rows=(d["av"][:, None] * 10 + np.arange(10)).ravel(),
        **{k + "_" + metric: v for k, g in stats.items() for metric, v in g.items()},
    )
    print("Validation", summary, "gate", gate, flush=True)


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def audit(out):
    outputs = ROOT / "outputs"
    design = json.loads((LOCK).read_bytes())
    split = outputs / "official-repeat-object/angles/design.json"
    require(
        sha(LOCK) == DESIGN and sha(split) == design["source_split_sha256"],
        "Changed split or design",
    )
    reg = json.loads(split.read_bytes())
    concepts = sorted(reg["angle_calibration_classes"] + reg["angle_validation_classes"])
    ext_file = outputs / "external-object-anchors/sub-03/external_manifest.csv"
    manifest = outputs / "official-object-readout/sub-03/atm_train_rows.csv"
    archive = outputs / "photo-structure/training_images.zip"
    labels = load_csv(manifest)
    ext = load_csv(ext_file)
    signatures = [
        (
            v["sha256"],
            v["thumbnail_sha256"],
            (int(v["whole_phash"], 16), int(v["whole_dhash"], 16)),
            (int(v["crop_phash"], 16), int(v["crop_dhash"], 16)),
        )
        for v in ext
    ]
    found = []
    with ZipFile(archive) as z:
        files = {p.rsplit(".", 1)[0]: p for p in z.namelist() if "." in p}
        for c in concepts:
            for row in range(c * 10, c * 10 + 10):
                item = labels[row]
                require(int(item["train_row"]) == row, "Incorrect training image row")
                stem = "training_images/" + item["official_folder"] + "/" + item["stimulus_stem"]
                _, signature, _ = image_inputs(z.read(files[stem]))
                for i, ref in enumerate(signatures):
                    how = matches(signature, ref)
                    if how:
                        found.append(dict(row=row, anchor_id=ext[i]["unique_id"], reason=how))
    require(len(set(concepts)) == 128 and len(signatures) == 1854, "Incomplete audit galleries")
    result = dict(
        photos=1280,
        anchors=1854,
        comparisons=1280 * 1854,
        flagged=found,
        flagged_anchor_ids=sorted({v["anchor_id"] for v in found}),
        design_sha256=sha(LOCK),
        split_sha256=sha(split),
        zip_sha256=sha(archive),
        train_manifest_sha256=sha(manifest),
        anchor_manifest_sha256=sha(ext_file),
        fingerprint_source_sha256=sha(Path(fingerprint_source)),
    )
    write(out, "collision_audit.json", result)
    print(
        "Generic comparisons",
        result["comparisons"],
        "flagged pairs",
        len(found),
        "anchor IDs",
        len(result["flagged_anchor_ids"]),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["audit", "calibrate", "validate"])
    p.add_argument("--out", type=Path, default=OUT / "object-prior-replay/external")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        audit(args.out)
    else:
        (calibrate if args.stage == "calibrate" else validate)(inputs(args.out), args.out)
