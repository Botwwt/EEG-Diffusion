"""Blinded nine-person test of EEG-selected, whole-support contour templates."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from fit_fixed_word_photo_dose import PEOPLE
from fit_patch_transport import patch_encoder
from fit_personal_session_metric import load
from fit_repeat_photo_subspace import centered, ridge
from fit_spatial_photo import original
from fit_support_structure import population
from measure_foreground_geometry import MODEL_SHA, ZIP_SHA
from prepare_expanded_personal import extract
from render_within_photo_ip import check, read, sha
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from score_within_photo_dino import FILES

BASE = Path(__file__).resolve().parents[3]
PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
OUT = BASE / "outputs"
HERE = OUT / "eeg-shape-recovery"
PHOTO = OUT / "photo-structure"
GEN = OUT / "object-transfer/generic"
SOURCE = OUT / "official-object-readout/sub-03"
DESIGN = "eb3e802a7b98f4aa5a8b8504927c7f00bd1232c106894f3844e796a40d543733"


def save(name, obj):
    (HERE / name).write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def split():
    HERE.mkdir(parents=True, exist_ok=True)
    check(sha(PUBLIC / "support_contour_design.json") == DESIGN, "Changed study design")
    c = np.sort(np.array(read(GEN / "sub-04.json")["concepts"]))
    # The design specifies the first 32 entries of the common permutation.
    valid = np.sort(np.random.default_rng(260553).permutation(c)[:32])
    train = np.setdiff1d(c, valid)
    check(c.shape == (128,) and train.shape == (96,), "Wrong generic classes")
    return c, train, valid


def silhouette(session, im, with_frame=False):
    from PIL import Image

    px = np.array(im.resize((320, 320), Image.Resampling.LANCZOS), dtype=np.float32)
    tensor = (
        px / max(float(px.max()), 1) - np.array([0.485, 0.456, 0.406], np.float32)
    ) / np.array([0.229, 0.224, 0.225], np.float32)
    mask = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])},
    )[0][0, 0]
    m = mask > (mask.min() + 0.5 * (mask.max() - mask.min()))
    y, x = np.nonzero(m)
    check(len(x) > 4, "Missing foreground")
    cx, cy, sx, sy = x.mean(), y.mean(), max(x.std(), 0.5), max(y.std(), 0.5)
    x = (x - cx) / sx
    y = (y - cy) / sy
    a = np.mod(np.arctan2(y, x), 2 * np.pi)
    bins = np.floor(24 * a / (2 * np.pi)).astype(int).clip(0, 23)
    radius = np.hypot(x, y)
    available = np.unique(bins)
    r = np.array([np.quantile(radius[bins == j], 0.95) for j in available])
    r = np.interp(np.arange(24), available, r, period=24)
    v = gaussian_filter1d(np.log(np.maximum(r, 1e-4)), 1, mode="wrap")
    profile = (v - v.mean()).astype(np.float32)
    if with_frame:
        return profile, np.array([(cx + 0.5) / 320, (cy + 0.5) / 320, sx / 320, sy / 320])
    return profile


def segmentation():
    import sys

    if (PHOTO / "ort").is_dir():
        sys.path.insert(0, str(PHOTO / "ort"))
    import onnxruntime as ort

    model = PHOTO / "u2netp.onnx"
    check(sha(model) == MODEL_SHA, "Foreground network differs")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = [
        p
        for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        if p in ort.get_available_providers()
    ]
    ort_s = ort.InferenceSession(str(model), sess_options=so, providers=providers)
    print("Saliency providers", ort_s.get_providers(), flush=True)
    return ort_s


def visual(mode):
    import torch
    from torch.nn.functional import avg_pool2d

    classes, _, _ = split()
    rows = (classes[:, None] * 10 + np.arange(10)).ravel()
    archive = PHOTO / "training_images.zip"
    check(
        sha(archive) == read(PHOTO / "training_images.zip.receipt.json")["sha256"] == ZIP_SHA,
        "Training image archive changed",
    )
    tab = population(SimpleNamespace(source=SOURCE))
    if mode == "patches":
        enc = patch_encoder(
            SimpleNamespace(model=PHOTO / "dino/facebook-dinov2-small", device="cuda")
        )
    else:
        ort_s = segmentation()
    vals = []
    with ZipFile(archive) as z, torch.inference_mode():
        for start in range(0, len(rows), 16):
            ims = [original(z, tab, int(r)) for r in rows[start : start + 16]]
            if mode == "patches":
                vals.extend(avg_pool2d(enc(ims), 4).flatten(1).cpu().numpy())
            else:
                vals.extend([silhouette(ort_s, im) for im in ims])
            if (start + 16) % 256 == 0:
                print(mode, start + 16, flush=True)
    np.savez(HERE / (mode + ".npz"), rows=rows, values=np.stack(vals), design_sha256=DESIGN)


def generic_features(subject, classes):
    source = GEN / (subject + ".npz")
    meta = read(source.with_suffix(".json"))
    check(
        meta["subject"] == subject
        and sha(source) == meta["packet_sha256"]
        and np.array_equal(meta["concepts"], classes),
        "Physical subpopulation differs",
    )
    rr, _, _, wave = load(source)
    check(
        np.array_equal(rr, (classes[:, None] * 10 + np.arange(10)).ravel()),
        "EEG row order differs",
    )
    x = extract(wave).reshape(128, 10, -1).astype(float)
    return x, meta["packet_sha256"]


def train_and_select():
    classes, train, val = split()
    train_ix = np.searchsorted(classes, train)
    val_ix = np.searchsorted(classes, val)
    with np.load(HERE / "patches.npz", allow_pickle=False) as p:
        check(
            np.array_equal(p["rows"], (classes[:, None] * 10 + np.arange(10)).ravel())
            and str(p["design_sha256"]) == DESIGN,
            "Patch rows changed",
        )
        all_y = p["values"].reshape(128, 10, -1).astype(np.float64)
    residual = centered(all_y[train_ix])
    lam, u = eigh(
        residual @ residual.T,
        subset_by_index=[len(residual) - 16, len(residual) - 1],
        check_finite=False,
    )
    proj = residual.T @ (u / np.sqrt(lam))
    y = all_y @ proj
    truthrows = (val[:, None] * 10 + np.arange(10)).ravel()
    q = np.arange(len(truthrows))
    fold = (q % 10) // 2
    supports = np.array(
        [
            [10 * (i // 10) + k for k in range(10) if k // 2 != f]
            for i, f in zip(q, fold, strict=True)
        ]
    )
    actual = y[val_ix].reshape(-1, 16)
    sv = actual[supports]
    comp = unit(sv - sv.mean(1, keepdims=True))
    predictions = []
    eeg_sha = {}
    for subject in PEOPLE:
        x, eeg_sha[subject] = generic_features(subject, classes)
        continuous = ridge(x[train_ix], centered(y[train_ix]), x[val_ix].reshape(320, -1))
        predictions.append(unit(continuous - continuous[supports].mean(1)))
    cohort = unit(np.sum(predictions, axis=0))
    scores = (cohort[:, None, :] * comp).sum(2)
    own = np.argmax(scores, axis=1)
    # Freeze the EEG choices before any held-out contour is read.
    np.savez(
        HERE / "selection.npz",
        rows=truthrows,
        support_rows=truthrows[supports],
        selected=own,
        scores=scores,
        donor=q ^ 1,
        design_sha256=DESIGN,
    )
    save(
        "selection.json",
        dict(
            design_sha256=DESIGN,
            training_classes=train.tolist(),
            held_classes=val.tolist(),
            eeg_sources=eeg_sha,
            patches_sha256=sha(HERE / "patches.npz"),
            model=FILES,
            selection_sha256=sha(HERE / "selection.npz"),
        ),
    )
    # Pure scoring-only positive control accesses the held query's true DINO patches.
    oracle_patch = (unit(actual - sv.mean(1))[:, None, :] * comp).sum(2).argmax(1)
    np.save(HERE / "oracle_patch.npy", oracle_patch)
    print("Locked", len(own), "query choices", flush=True)


def evaluate(publish=False):
    classes, _, val = split()
    with (
        np.load(HERE / "contours.npz", allow_pickle=False) as w,
        np.load(HERE / "selection.npz", allow_pickle=False) as s,
    ):
        rows, supports = s["rows"], s["support_rows"]
        check(
            np.array_equal(w["rows"], (classes[:, None] * 10 + np.arange(10)).ravel())
            and np.array_equal(rows, (val[:, None] * 10 + np.arange(10)).ravel())
            and np.all(rows[:, None] // 10 == supports // 10)
            and not np.any(rows[:, None] // 2 == supports // 2),
            "Support leak",
        )
        sig = w["values"][np.searchsorted(w["rows"], rows)].astype(float)
        bank = w["values"][np.searchsorted(w["rows"], supports)].astype(float)
        own = s["selected"]
        donor = s["donor"]
        check(
            np.array_equal(supports, supports[donor]), "Wrong EEG lacks the same legal support bank"
        )
    target = np.abs(sig[:, None, :] - bank).mean(2)
    between = np.abs(bank[:, :, None, :] - bank[:, None, :, :]).mean(3)
    medoid = between.sum(2).argmin(1)
    oracle = target.argmin(1)
    qp = np.arange(320)
    choices = dict(
        matched=own,
        medoid=medoid,
        wrong_eeg=own[donor],
        best_legal_oracle=oracle,
        true_patch_oracle=np.load(HERE / "oracle_patch.npy", allow_pickle=False),
    )
    loss = {k: target[qp, v] for k, v in choices.items()} | {"random_mean": target.mean(1)}
    boot = np.random.default_rng(260554).integers(32, size=(10000, 32))

    def diff(a, b):
        d = (a - b).reshape(32, 10).mean(1)
        return dict(
            delta=float(d.mean()), ci95=np.quantile(d[boot].mean(1), [0.025, 0.975]).tolist()
        )

    contrast = {k: diff(v, loss["matched"]) for k, v in loss.items() if k != "matched"}
    gate = all(contrast[k]["ci95"][0] > 0 for k in ("medoid", "wrong_eeg"))
    summary = dict(
        design_sha256=DESIGN,
        selection_sha256=sha(HERE / "selection.npz"),
        shape_sha256=sha(HERE / "contours.npz"),
        observations=320,
        objects=32,
        supports_per_query=8,
        physical_trials_per_photo=36,
        mean_l1={k: float(v.mean()) for k, v in loss.items()},
        control_minus_matched=contrast,
        diagnostic_medoid_minus={k: diff(loss["medoid"], v) for k, v in loss.items()},
        counts_at_true_best={k: int(np.sum(v == oracle)) for k, v in choices.items()},
        exact_selected_true_patch_support=int(np.sum(own == choices["true_patch_oracle"])),
        same_selection_as_wrong=int(np.sum(own == own[donor])),
        pair_diagnostics={
            k: dict(
                smaller=int((loss["matched"] < v).sum()),
                ties=int((loss["matched"] == v).sum()),
                larger=int((loss["matched"] > v).sum()),
            )
            for k, v in loss.items()
        },
        advance=bool(gate),
        interpretation="Absolute log-radius error; lower is better. Intervals resample 32 whole held-out objects, 10000 draws. Oracle scores use the identical eight distinct permitted support photographs and target shape only for evaluation. No formal target set opens unless both gate conditions pass.",
    )
    save("result.json", summary)
    if publish:
        (PUBLIC / "support_contour.json").write_bytes((HERE / "result.json").read_bytes())
    np.savez(
        HERE / "evaluation_rows.npz",
        rows=rows,
        **{"select_" + k: v for k, v in choices.items()},
        **{"loss_" + k: v for k, v in loss.items()},
    )
    print("Generic test", summary["mean_l1"], "controls", contrast, "advance", gate, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["patches", "contours", "select", "score"])
    p.add_argument(
        "--publish",
        action="store_true",
        help="Write the aggregate generic result on the score stage",
    )
    a = p.parse_args()
    if a.stage in ("patches", "contours"):
        visual(a.stage)
    elif a.stage == "select":
        train_and_select()
    else:
        evaluate(a.publish)
