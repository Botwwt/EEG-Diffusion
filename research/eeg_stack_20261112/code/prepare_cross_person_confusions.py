"""Fit photo-confusion teachers on disjoint external objects and freeze class-held-out features."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import torch
from ablate_trial_object_encoder import fit
from fit_fresh_mix import neural
from fit_fresh_photo_panel import fisher
from fit_trial_object_encoder import DESIGN as ORIGINAL
from prepare_scale_trial import PUBLIC
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz

DESIGN = PUBLIC / "cross_person_confusions_design.json"
LOCK = "cd2f96fb948c27785185c69155718f141697e80b2ad91da547a7b018f464a409"
PEOPLE = ("sub-04", "sub-05", "sub-01")


def frame(a):
    check(sha(DESIGN) == LOCK, "Study protocol changed")
    spec = read(DESIGN)
    double = PUBLIC / "double_calibration_design.json"
    check(
        sha(double) == spec["class_universe_sha256"]
        and sha(PUBLIC / "shared_trial_encoder.json") == spec["old_result_sha256"],
        "Earlier study changed",
    )
    historic = a.data / "trial-object-encoder/generic/receipt.json"
    check(
        sha(historic)
        == read(PUBLIC / "trial_object_encoder.json")["generic"]["training_receipt_sha256"],
        "Wrong held object split",
    )
    r = read(historic)
    train = np.sort(np.r_[r["train_classes"], read(double)["extra_objects"]]).astype(int)
    held = np.array(r["test_classes"], int)
    check(
        len(set(train)) == 928 and len(set(held)) == 96 and not set(train) & set(held),
        "Generic split",
    )
    path = a.data / "object-transfer/sessions.json"
    check(sha(path) == r["session_manifest_sha256"], "Session plan changed")
    session = read(path)["subjects"]
    schedules = np.stack([np.array(session[s]["session_pairs"]) for s in PEOPLE])
    joint = np.unique(schedules.transpose(1, 0, 2).reshape(-1, 6), axis=0, return_inverse=True)[1]
    cells = [np.flatnonzero(joint[train] == j) for j in np.unique(joint[train])]
    eligible = [c for c in cells if len(c) >= 12]
    check(
        len(cells) == 64 and len(eligible) == 52 and sum(map(len, eligible)) == 816,
        "Wrong joint session coverage",
    )
    perm = np.random.default_rng(spec["seed"]).permutation(len(cells))
    fold = np.empty(len(train), int)
    sizes = np.zeros(4, int)
    for j in sorted(perm, key=lambda j: -len(cells[j])):
        k = int(sizes.argmin())
        fold[cells[j]] = k
        sizes[k] += len(cells[j])
    return train, held, schedules, fold, eligible


def participant(a, sub, requested):
    """Read only the requested physical classes, verifying trial/feature source digests agree."""
    base = a.data / "photo-structure/expanded/features"
    more = a.data / "double-calibration/features"
    old = (
        (a.data / "trial-object-inputs" / sub)
        if sub == "sub-04"
        else (a.data / "trial-object-replication" / sub / "trials")
    )
    extra = (
        a.data / "scale-trial/extra" if sub == "sub-04" else a.data / "fresh-people" / sub / "extra"
    )
    oldfile = old / ("generic512.json" if sub == "sub-04" else "receipt.json")
    raw0 = read(oldfile)["blocks"]
    raw1 = read(extra / "receipt.json")["blocks"]
    oldref = read(base / (sub + ".json"))
    newref = read(more / (sub + ".json"))
    check(
        oldref["subject"] == newref["subject"] == sub
        and oldref["source"] == newref["source"]
        and set(raw0) == set(oldref["raw_block_sha256"])
        and set(raw1) == set(newref["raw_sha256"]),
        "Wrong participant/physical registry",
    )
    raw = np.empty((len(requested), 10, 4, 63, 125), np.float32)
    physical = np.empty((len(requested), 10, 1003), float)
    for i, c in enumerate(requested):
        key = str(c)
        if key in raw0:
            directory, info, phys, h = old, raw0[key], base / sub, oldref["raw_block_sha256"][key]
        else:
            directory, info, phys, h = extra, raw1[key], more / sub, newref["raw_sha256"][key]
        f = directory / f"{c:04d}.npy"
        check(
            sha(f) == info["output_sha256"] and h == info["source_raw_sha256"],
            "Physical crop differs",
        )
        v = np.load(f, allow_pickle=False)
        check(
            v.shape == (10, 4, 63, 125) and v.dtype == np.float32 and np.isfinite(v).all(),
            "Wrong trial block",
        )
        with np.load(phys / f"{c:04d}.npz", allow_pickle=False) as z:
            p = z["features"]
            from hashlib import sha256

            check(
                str(z["raw_sha"]) == h
                and int(z["start"]) == 10 * c
                and p.shape == (10, 1003)
                and sha256(p.tobytes()).hexdigest() == str(z["feature_sha"]),
                "Physical feature mismatch",
            )
        raw[i] = v
        physical[i] = p
    return torch.from_numpy(raw).to(a.device), physical


def normfit(raw):
    sd, mu = torch.std_mean(raw, dim=(0, 1, 2, 4), unbiased=False)
    mu, sd = mu[:, None], sd.clamp_min(1e-5)[:, None]
    return raw.sub_(mu).div_(sd).clamp_(-12, 12), mu, sd


def teachers(a):
    ids, held, schedules, fold, _ = frame(a)
    freeze(a.out / "design.json", read(DESIGN))
    for i, sub in enumerate(PEOPLE):
        folder = a.out / "teachers" / sub
        folder.mkdir(exist_ok=True, parents=True)
        x = phys = None
        for k in range(4):
            ft = np.flatnonzero(fold != k)
            ev = np.flatnonzero(fold == k)
            p = folder / f"fold{k}.npz"
            receipt = p.with_suffix(".json")
            signature = dict(
                design_sha256=LOCK,
                source_sha256=sha(Path(__file__)),
                subject=sub,
                train_ids=ids[ft].tolist(),
                encoded_ids=ids[ev].tolist(),
            )
            if receipt.exists():
                before = read(receipt)
                check(
                    all(before[j] == v for j, v in signature.items())
                    and before["features_sha256"] == sha(p),
                    "Teacher changed",
                )
                continue
            if x is None:
                x, phys = participant(a, sub, ids)
            data, mu, sd = normfit(x[ft])
            _, sessions = np.unique(schedules[i, ids[ft]], axis=0, return_inverse=True)
            net, trace = fit(
                data,
                sessions,
                True,
                0.2,
                read(ORIGINAL) | dict(updates=1500),
                torch.device(a.device),
            )
            del data
            deep = neural(net, x[ev], mu, sd).reshape(len(ev), 10, 96)
            f = fisher(phys[ft], phys[ev])
            check(
                not np.intersect1d(np.r_[ids[ft], ids[ev]], held).size
                and not np.intersect1d(ids[ft], ids[ev]).size,
                "A teacher used its own auxiliary objects or held test classes",
            )
            frozen_npz(
                p, dict(ids=ids[ev], neural=deep.astype(np.float32), fisher=f.astype(np.float32))
            )
            freeze(receipt, signature | dict(features_sha256=sha(p), training_trace=trace))
            print("OOF", sub, k, len(ft), len(ev), flush=True)
        del x, phys


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/cross-person-confusions"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--device", default="cuda")
    return p


def setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    setup()
    teachers(cli().parse_args())
