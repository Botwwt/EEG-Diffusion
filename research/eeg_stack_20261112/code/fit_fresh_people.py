"""Transfer the discovery mixture coefficient and test two personal EEG encoders on new menus."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ablate_trial_object_encoder import fit
from fit_double_calibration import training
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_trial_object_encoder import DESIGN as TRIAL
from prepare_fresh_people import DESIGN, PUBLIC, SUBJECTS
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz
from safetensors.torch import save_file
from score_fresh_mix import encode


def materials(a):
    spec = read(DESIGN)
    sub = a.subject
    root = a.out / sub
    historic = a.data / "trial-object-replication" / sub
    old = historic / "trials/receipt.json"
    new = root / "extra/receipt.json"
    checkpoints = historic / "fit/receipt.json"
    source_old = a.data / "photo-structure/expanded/features" / (sub + ".json")
    source_new = a.data / "double-calibration/features" / (sub + ".json")
    check(
        sha(DESIGN) == sha(a.out / "design.json")
        and sha(old) == spec["sources"][sub]["old_trials"]
        and sha(source_old) == spec["sources"][sub]["old_physical"]
        and sha(source_new) == spec["sources"][sub]["extra_physical"]
        and sha(checkpoints) == spec["sources"][sub]["old_fit"]
        and read(new)["design_sha256"] == sha(DESIGN)
        and read(new)["subject"] == sub
        and sha(historic / "fit/model.safetensors") == read(checkpoints)["weights_sha256"],
        "Source trial manifest or personal checkpoint changed",
    )
    batches = {}
    for record, expected in (
        (old, read(source_old)["raw_block_sha256"]),
        (new, read(source_new)["raw_sha256"]),
    ):
        check(set(read(record)["blocks"]) == set(expected), "Missing or extra physical object")
        for key, item in read(record)["blocks"].items():
            c = int(key)
            check(
                c not in batches and item["source_raw_sha256"] == expected[key],
                "Overlapping object or wrong participant's physical EEG",
            )
            batches[c] = (record.parent / f"{c:04d}.npy", item["output_sha256"])
    classes = np.array(sorted(batches))
    registered = read(PUBLIC / "double_calibration_design.json")
    check(
        len(classes) == 1024
        and set(classes) == set(registered["old_objects"] + registered["extra_objects"]),
        "Missing training class",
    )
    schedule = a.data / "object-transfer/sessions.json"
    check(sha(schedule) == spec["session_sha256"], "Changed acquisition sessions")
    sessions = np.array(read(schedule)["subjects"][sub]["session_pairs"])
    _, codes = np.unique(sessions, axis=0, return_inverse=True)
    return root, historic, classes, codes, batches


def train(a):
    root, _, ids, sessions, batches = materials(a)
    folder = root / "fit"
    folder.mkdir(exist_ok=True, parents=True)
    checkpoint = folder / "model.safetensors"
    report = folder / "receipt.json"
    info = dict(
        subject=a.subject,
        design_sha256=sha(DESIGN),
        train_ids=ids.tolist(),
        updates=1500,
        trainer_sha256=sha(Path(__file__).with_name("ablate_trial_object_encoder.py")),
        implementation_sha256=sha(Path(__file__)),
        extra_manifest_sha256=sha(root / "extra/receipt.json"),
    )
    if report.exists():
        prior = read(report)
        check(
            all(prior[k] == v for k, v in info.items())
            and sha(checkpoint) == prior["weights_sha256"],
            "Conflicting fitted model",
        )
        return
    samples = np.empty((len(ids), 10, 4, 63, 125), np.float32)
    for j, c in enumerate(ids):
        src, digest = batches[c]
        check(sha(src) == digest, f"EEG crop changed for {c}")
        v = np.load(src, allow_pickle=False)
        check(
            v.shape == samples.shape[1:] and v.dtype == np.float32 and np.isfinite(v).all(),
            "Unexpected physical data",
        )
        samples[j] = v
    device = torch.device(a.device)
    x = torch.from_numpy(samples).to(device)
    sd, mu = torch.std_mean(x, dim=(0, 1, 2, 4), unbiased=False)
    mu, sd = mu[:, None], sd.clamp_min(1e-5)[:, None]
    x.sub_(mu).div_(sd).clamp_(-12, 12)
    model, trace = fit(x, sessions[ids], True, 0.2, read(TRIAL) | dict(updates=1500), device)
    state = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
    state.update(input_mean=mu.cpu(), input_scale=sd.cpu())
    save_file(state, checkpoint)
    freeze(report, info | dict(weights_sha256=sha(checkpoint), training_trace=trace))


def score(a):
    root, historic, ids, sessions, _ = materials(a)
    spec = read(DESIGN)
    fn = root / "eeg" / (a.subject + ".npz")
    check(
        sha(fn) == read(fn.with_suffix(".json"))["packet_sha256"]
        and read(fn.with_suffix(".json"))["design_sha256"] == sha(DESIGN),
        "Query EEG source changed",
    )
    with np.load(fn, allow_pickle=False) as e:
        q = e["features"].astype(float)
        rr, trials = e["rows"], e["trials"]
    check(
        len(rr) == 480
        and not np.intersect1d(ids, rr // 10).size
        and np.array_equal(rr, np.unique(rr)),
        "Query-training overlap",
    )
    oldmodel, newmodel = (p / "model.safetensors" for p in (historic / "fit", root / "fit"))
    check(
        sha(newmodel) == read(newmodel.with_name("receipt.json"))["weights_sha256"],
        "Fitted model changed",
    )
    _, large = training(
        SimpleNamespace(
            out=a.data / "double-calibration", expanded=a.data / "photo-structure/expanded"
        ),
        a.subject,
    )
    large = large.reshape(-1, 10, 1003)
    oldids = np.array(read(historic / "fit/receipt.json")["train_ids"])
    check(
        len(np.unique(oldids)) == 512 and np.array_equal(ids[np.searchsorted(ids, oldids)], oldids),
        "Legacy Fisher lookup differs from the personal training IDs",
    )
    vec = [
        fisher(large[np.searchsorted(ids, oldids)], q),
        encode(oldmodel, trials, a.device),
        fisher(large, q),
        encode(newmodel, trials, a.device),
    ]
    for name, panel in spec["panels"][a.subject].items():
        file = root / "references" / (name + "_ablation.json")
        check(sha(file) == panel["registration_sha256"], "Changed panel")
        meta = read(file)
        rows, folds, cal, words = (
            np.array(meta[k])
            for k in ("rows", "folds", "calibration_rows_by_fold_and_word", "words")
        )
        donors = np.roll(rows.reshape(5, 12, 2), -1, axis=1).ravel()
        check(
            len(np.unique(sessions[rows // 10])) == 1 and np.all(rows // 10 != donors // 10),
            "Session or control mismatch",
        )
        scores = np.empty((2, 4, 120, 12))
        for fold in range(5):
            ix = folds == fold
            si = np.searchsorted(rr, cal[fold])
            check(
                set(cal[fold].ravel()) == set(rows[~ix]) and np.array_equal(rr[si], cal[fold]),
                "Support leakage",
            )
            for j, source in enumerate((rows, donors)):
                qi = np.searchsorted(rr, source[ix])
                check(np.array_equal(rr[qi], source[ix]), "Query lookup mismatch")
                for k, v in enumerate(vec):
                    scores[j, k, ix] = similarities(v[qi], v[si])
        of, on, nf, nn = scores.transpose(1, 0, 2, 3)
        w = spec["neural_weight"]
        mixed = dict(
            legacy=(of + on) / 2,
            learned=(1 - w) * nf + w * nn,
            equal=(nf + nn) / 2,
            neural=nn,
            fisher=nf,
        )
        choices = {k: v[0].argmax(1) for k, v in mixed.items()}
        choices.update({k + "_wrong": v[1].argmax(1) for k, v in mixed.items()})
        dest = root / "decisions" / (name + ".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        frozen_npz(
            dest,
            dict(
                rows=rows,
                words=words,
                folds=folds,
                donor_rows=donors,
                scores=scores,
                registration_sha256=sha(file),
                **choices,
            ),
        )


def summarize(a):
    spec = read(DESIGN)
    reports = {}
    records = {}
    arms = ("legacy", "learned", "learned_wrong", "neural", "fisher", "equal")
    for sub in SUBJECTS:
        records[sub] = {}
        for name, meta in spec["panels"][sub].items():
            root = a.out / sub
            file = root / "decisions" / (name + ".npz")
            reg = root / "references" / (name + "_ablation.json")
            with np.load(file, allow_pickle=False) as z:
                truth = np.array(read(reg)["truth"])
                rows = z["rows"]
                check(
                    str(z["registration_sha256"]) == sha(reg) == meta["registration_sha256"]
                    and np.array_equal(rows // 10, np.array(meta["object_ids"])[truth]),
                    "Changed labels at evaluation",
                )
                for c in meta["object_ids"]:
                    records[sub][c] = (
                        c in meta["primary_object_ids"],
                        np.array(
                            [
                                [int(x) for x in z[k][rows // 10 == c] == truth[rows // 10 == c]]
                                for k in arms
                            ]
                        ),
                    )
    for label in ("primary", "all"):
        bysub = {
            s: {c: v[1] for c, v in cx.items() if label == "all" or v[0]}
            for s, cx in records.items()
        }
        keys = sorted(set.union(*(set(cx) for cx in bysub.values())))
        totals = np.stack(
            [
                sum(
                    (cx[c].sum(1) for cx in bysub.values() if c in cx),
                    start=np.zeros(len(arms), int),
                )
                for c in keys
            ]
        )
        weights = np.array([10 * sum(c in cx for cx in bysub.values()) for c in keys])
        check(weights.sum() == (700 if label == "primary" else 960), "Incorrect denominator")
        draw = np.random.default_rng(260631).integers(len(keys), size=(10000, len(keys)))
        bs = totals[draw].sum(1) / weights[draw].sum(1)[:, None]
        rows = np.concatenate([v for sub in bysub.values() for v in sub.values()], axis=1)
        reports[label] = dict(
            n=int(weights.sum()),
            objects=len(keys),
            hits=dict(zip(arms, totals.sum(0).tolist(), strict=True)),
            by_subject={
                s: dict(zip(arms, np.stack(list(cx.values())).sum((0, 2)).tolist(), strict=True))
                for s, cx in bysub.items()
            },
            paired={
                k: dict(
                    gained=int(((rows[1] == 1) & (rows[i] == 0)).sum()),
                    lost=int(((rows[1] == 0) & (rows[i] == 1)).sum()),
                    delta_pp=float(100 * (rows[1] - rows[i]).mean()),
                    ci95_pp=(100 * np.quantile(bs[:, 1] - bs[:, i], [0.025, 0.975])).tolist(),
                )
                for i, k in enumerate(arms[:3])
                if i != 1
            },
        )
    primary = reports["primary"]
    advance = all(v["ci95_pp"][0] > 0 for v in primary["paired"].values()) and all(
        v["learned"] >= v["legacy"] for v in primary["by_subject"].values()
    )
    result = dict(
        design_sha256=sha(DESIGN),
        driver_sha256=sha(Path(__file__)),
        trained={s: sha(a.out / s / "fit/receipt.json") for s in SUBJECTS},
        decisions={
            s: {k: sha(a.out / s / "decisions" / (k + ".npz")) for k in spec["panels"][s]}
            for s in SUBJECTS
        },
        weight=spec["neural_weight"],
        populations=reports,
        advance=advance,
    )
    freeze(PUBLIC / "fresh_people_words.json", result)
    print(reports["primary"], "advance", advance)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "score", "summarize"))
    p.add_argument("--subject", choices=SUBJECTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-people"))
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if a.stage == "summarize":
        summarize(a)
    elif a.subject:
        (train if a.stage == "train" else score)(a)
    else:
        p.error("--subject required for this stage")
