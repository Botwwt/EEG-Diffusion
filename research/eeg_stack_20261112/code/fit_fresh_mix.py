"""Cross-fit one mixture coefficient on external EEG objects, before reading registered query EEG."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import torch
from ablate_trial_object_encoder import fit
from fit_double_calibration import training
from fit_fresh_photo_panel import fisher
from fit_hard_object_pairs import episodes
from fit_personal_session_metric import similarities
from fit_scale_trial import rows
from fit_trial_object_encoder import DESIGN as ORIGINAL
from prepare_scale_trial import metadata
from prepare_trial_object_inputs import check
from register_fresh_mix import LOCK, SEED, SUB
from register_fresh_photo_panel import freeze, read, sha
from safetensors.torch import save_file
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def inputs(a):
    check(sha(LOCK) == sha(a.out / "design.json"), "Study registration changed")
    historical, added, native = metadata(a.data)
    transform_path = a.data / "double-calibration"
    combine = SimpleNamespace(expanded=a.data / "photo-structure/expanded", out=transform_path)
    _, features = training(combine, SUB)
    orig = read(a.data / "trial-object-inputs/sub-04/generic512.json")["blocks"]
    extra = read(a.scale / "extra/receipt.json")
    ids = np.sort(np.r_[historical["train_classes"], historical["test_classes"], added]).astype(int)
    check(
        len(set(ids)) == 1024
        and features.shape == (10240, 1003)
        and all(
            v["source_raw_sha256"] == native["raw_sha256"][k] for k, v in extra["blocks"].items()
        )
        and set(extra["blocks"]) == set(map(str, added)),
        "Incomplete calibration EEG",
    )
    sess = np.array(
        read(a.data / "object-transfer/sessions.json")["subjects"][SUB]["session_pairs"]
    )
    _, codes = np.unique(sess, axis=0, return_inverse=True)
    return ids, features.reshape(1024, 10, 1003), codes, orig, extra["blocks"]


def folds(ids, sessions):
    omitted = read(LOCK)["original_validation_ids_excluded_from_weight_loss"]
    pool = np.setdiff1d(ids, omitted)
    check(
        len(pool) == 928 and len(omitted) == 96,
        "Weight calibration includes previously scored classes",
    )
    rng = np.random.default_rng(SEED + 2)
    groups = [[] for _ in range(4)]
    for ss in np.unique(sessions[pool]):
        order = rng.permutation(pool[sessions[pool] == ss])
        for j, v in enumerate(groups):
            v.extend(order[j::4])
    return [np.sort(g) for g in groups]


def load_trials(a, ids, old, ext):
    return rows(SimpleNamespace(data=a.data, out=a.scale), ids, old, ext, torch.device(a.device))


def train_model(a, ids, sessions, old, ext):
    x = load_trials(a, ids, old, ext)
    sd, mu = torch.std_mean(x, dim=(0, 1, 2, 4), unbiased=False)
    mu, sd = mu[:, None], sd.clamp_min(1e-5)[:, None]
    x.sub_(mu).div_(sd).clamp_(-12, 12)
    model, trace = fit(
        x, sessions[ids], True, 0.2, read(ORIGINAL) | dict(updates=1500), torch.device(a.device)
    )
    return model, mu, sd, trace


def neural(model, raw, mu, sd):
    with torch.inference_mode():
        y = torch.cat(
            [model(((b - mu) / sd).clamp(-12, 12)) for b in raw.reshape(-1, 4, 63, 125).split(128)]
        )
    return y.cpu().numpy().astype(float)


def calibration(a):
    ids, features, sessions, old, ext = inputs(a)
    groups = folds(ids, sessions)
    scores = []
    receipts = []
    for j, held in enumerate(groups):
        train = np.setdiff1d(ids, held)
        folder = a.out / "fit" / f"fold{j}"
        folder.mkdir(parents=True, exist_ok=True)
        rec = folder / "receipt.json"
        outputs = folder / "scores.npz"
        header = dict(
            design_sha256=sha(LOCK),
            driver_sha256=sha(Path(__file__)),
            train_ids=train.tolist(),
            held_ids=held.tolist(),
        )
        if not rec.exists():
            model, mu, sd, trace = train_model(a, train, sessions, old, ext)
            x = load_trials(a, held, old, ext)
            learned = neural(model, x, mu, sd)
            del x, model
            physical = fisher(
                features[np.searchsorted(ids, train)], features[np.searchsorted(ids, held)]
            )
            phys = physical.reshape(-1, 96)
            raw = held[:, None] * 10 + np.arange(10)
            packs = []
            for _, _, _, label, qr, _, sr in episodes(
                raw, held, sessions[held], 12, SEED + 100 * j + 12
            ):
                qi, si = np.searchsorted(raw.ravel(), qr), np.searchsorted(raw.ravel(), sr)
                check(
                    np.array_equal(raw.ravel()[si], sr) and not np.isin(qr, sr).any(),
                    "OOF support overlap",
                )
                packs.append(
                    (
                        qr,
                        np.full(2, label),
                        similarities(phys[qi], phys[si]),
                        similarities(learned[qi], learned[si]),
                    )
                )
            v = dict(
                zip(
                    ("rows", "slots", "fisher", "neural"),
                    (np.concatenate(a) for a in zip(*packs, strict=True)),
                    strict=True,
                )
            )
            np.savez(outputs, **v)
            freeze(rec, header | dict(score_sha256=sha(outputs), training_trace=trace))
        r = read(rec)
        check(
            all(r[k] == v for k, v in header.items()) and r["score_sha256"] == sha(outputs),
            "OOF fit changed",
        )
        with np.load(outputs) as z:
            scores.append(dict(z))
        receipts.append(sha(rec))
    allrows = np.concatenate([z["rows"] for z in scores])
    authorized = np.concatenate(groups)
    check(
        len(allrows) == len(set(allrows)) == 9280 and set(allrows // 10) == set(authorized),
        "OOF coverage",
    )
    f, n, y = (np.concatenate([z[k] for z in scores]) for k in ("fisher", "neural", "slots"))

    def criterion(w):
        logits = 2 * ((1 - w) * f + w * n)
        return float(
            (logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]).mean() + 0.1 * (w - 0.5) ** 2
        )

    solution = minimize_scalar(criterion, bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
    check(solution.success, "Scalar fit failed")
    w = min([0.0, 1.0, float(solution.x)], key=criterion)
    result = dict(
        design_sha256=sha(LOCK),
        training_driver_sha256=sha(Path(__file__)),
        training_receipts=receipts,
        n=len(y),
        objects=928,
        neural_weight=w,
        penalized_nll=criterion(w),
        equal_weight_penalized_nll=criterion(0.5),
        oof_hits={
            str(k): int((((1 - weight) * f + weight * n).argmax(1) == y).sum())
            for k, weight in (("fisher", 0.0), ("neural", 1.0), ("equal", 0.5), ("learned", w))
        },
    )
    freeze(a.out / "fit/mixture.json", result)
    print("OOF weight locked", w, result["oof_hits"], flush=True)


def final(a):
    ids, _, sessions, old, ext = inputs(a)
    weight = a.out / "fit/mixture.json"
    check(
        read(weight)["design_sha256"] == sha(LOCK)
        and read(weight)["training_driver_sha256"] == sha(Path(__file__)),
        "Weight not frozen",
    )
    folder = a.out / "fit/final1024"
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "model.safetensors"
    r = folder / "receipt.json"
    header = dict(
        design_sha256=sha(LOCK),
        training_driver_sha256=sha(Path(__file__)),
        train_classes=ids.tolist(),
        mixture_sha256=sha(weight),
        updates=1500,
    )
    if r.exists():
        record = read(r)
        check(
            all(record[k] == v for k, v in header.items()) and record["weights_sha256"] == sha(p),
            "Final fit changed",
        )
        return
    model, mu, sd, trace = train_model(a, ids, sessions, old, ext)
    state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
    state.update(input_mean=mu.cpu(), input_scale=sd.cpu())
    save_file(state, p)
    freeze(r, header | dict(weights_sha256=sha(p), training_trace=trace))
    print("Final fit", sha(p), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("calibrate", "final", "all"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-mix"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--scale", type=Path, default=Path("outputs/scale-trial"))
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if a.stage in ("all", "calibrate"):
        calibration(a)
    if a.stage in ("all", "final"):
        final(a)
