"""Compare the locked calibrated and legacy object classifiers on four new, session-matched menus."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import torch
from fit_fresh_mix import inputs, neural
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_trial_object_encoder import TrialObjectEncoder
from prepare_scale_trial import PUBLIC
from prepare_trial_object_inputs import check
from register_fresh_mix import LOCK, SEED, SUB
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz
from safetensors.torch import load_file


def encode(path, x, device):
    state = load_file(path, device=device)
    mu, sd = state.pop("input_mean"), state.pop("input_scale")
    model = TrialObjectEncoder().to(device)
    model.load_state_dict(state)
    return neural(model.eval(), torch.from_numpy(x).to(device), mu, sd)


def decisions(a):
    design = read(LOCK)
    weight_file = a.out / "fit/mixture.json"
    weight = read(weight_file)
    fit = a.out / "fit/final1024"
    fitted = read(fit / "receipt.json")
    check(
        weight["design_sha256"] == sha(LOCK)
        and weight["training_driver_sha256"] == sha(Path(__file__).with_name("fit_fresh_mix.py"))
        and fitted["mixture_sha256"] == sha(weight_file)
        and fitted["weights_sha256"] == sha(fit / "model.safetensors")
        and fitted["updates"] == 1500
        and len(set(fitted["train_classes"])) == 1024,
        "Final model registration differs",
    )
    ids, physical, sessions, _, _ = inputs(a)
    check(np.array_equal(fitted["train_classes"], ids), "Final training object order changed")
    legacy = a.data / "trial-object-encoder/target"
    old_spec = read(PUBLIC / "trial_object_personal12_design.json")["inputs"]
    check(
        sha(legacy / "model.safetensors") == old_spec["model"]
        and sha(legacy / "receipt.json") == old_spec["fit_receipt"],
        "Legacy word encoder changed",
    )
    old_ids = np.array(read(legacy / "receipt.json")["train_classes"])
    source = a.out / "eeg/sub-04.npz"
    eeg_receipt = read(source.with_suffix(".json"))
    check(
        eeg_receipt["design_sha256"] == sha(LOCK) and eeg_receipt["packet_sha256"] == sha(source),
        "Target EEG changed",
    )
    with np.load(source, allow_pickle=False) as eeg:
        source_rows = eeg["rows"]
        target = eeg["features"].astype(float)
        trials = eeg["trials"]
    check(
        trials.shape == (480, 4, 63, 125)
        and target.shape == (480, 1003)
        and len(np.unique(source_rows)) == 480
        and not np.intersect1d(source_rows // 10, ids).size,
        "Target and calibration overlap",
    )
    embeddings = [
        fisher(physical[np.searchsorted(ids, old_ids)], target),
        encode(legacy / "model.safetensors", trials, a.device),
        fisher(physical, target),
        encode(fit / "model.safetensors", trials, a.device),
    ]
    w = weight["neural_weight"]
    result = {}
    for panel, meta in design["panels"].items():
        ref = a.out / "references" / (panel + "_ablation.json")
        check(sha(ref) == meta["registration_sha256"], "Target menu changed")
        reg = read(ref)
        rows, folds, cal, words = (
            np.array(reg[k])
            for k in ("rows", "folds", "calibration_rows_by_fold_and_word", "words")
        )
        check(
            np.array_equal(words, meta["words"])
            and np.array_equal(np.unique(rows // 10), np.sort(meta["object_ids"]))
            and len(set(sessions[rows // 10])) == 1,
            "Candidate identities or sessions differ",
        )
        # A cyclic permutation of the 12 pairs at each fold maps both held photos to different
        # candidate positions. Its mapping is fixed before examining either query or label.
        donors = np.roll(rows.reshape(5, 12, 2), -1, axis=1).ravel()
        check(len(set(donors)) == 120 and np.all(rows // 10 != donors // 10), "Invalid permutation")
        score = np.empty((2, 4, 120, 12))
        for f in range(5):
            ix = np.flatnonzero(folds == f)
            si = np.searchsorted(source_rows, cal[f])
            check(
                np.array_equal(source_rows[si], cal[f])
                and len(set(cal[f].ravel())) == 96
                and set(cal[f].ravel()) == set(rows[folds != f]),
                "Support overlaps held fold",
            )
            for i, qr in enumerate((rows, donors)):
                qi = np.searchsorted(source_rows, qr[ix])
                check(np.array_equal(source_rows[qi], qr[ix]), "Missing EEG")
                score[i][:, ix, :] = np.stack([similarities(v[qi], v[si]) for v in embeddings])
        old_f, old_n, new_f, new_n = score.transpose(1, 0, 2, 3)
        arms = dict(
            legacy=(old_f + old_n) / 2,
            learned=(1 - w) * new_f + w * new_n,
            equal=(new_f + new_n) / 2,
            neural=new_n,
            fisher=new_f,
            legacy_fisher=old_f,
            legacy_neural=old_n,
        )
        picks = {k: v[0].argmax(1) for k, v in arms.items()}
        controls = {k + "_wrong": v[1].argmax(1) for k, v in arms.items()}
        path = a.out / "decisions" / (panel + ".npz")
        frozen_npz(
            path,
            dict(
                rows=rows,
                folds=folds,
                words=words,
                donor_rows=donors,
                scores=score,
                registration_sha256=sha(ref),
                target_eeg_sha256=sha(source),
                **picks,
                **controls,
            ),
        )
        result[panel] = (path, picks | controls)
        for arm in ("legacy", "learned"):
            word = (
                a.out
                / "render_inputs"
                / arm
                / "evaluations"
                / SUB
                / panel
                / "word_predictions.json"
            )
            freeze(
                word,
                dict(
                    subject=SUB,
                    panel=panel,
                    physical_trials_per_photo=4,
                    rows=rows.tolist(),
                    folds=folds.tolist(),
                    words=words.tolist(),
                    predicted=picks[arm].tolist(),
                    registration_sha256=sha(ref),
                    protocol_sha256=sha(LOCK),
                    mixture_sha256=sha(weight_file),
                ),
            )
    return result


def summarize(a, data):
    design = read(LOCK)
    out = {}
    for population in ("strict", "all"):
        grouped = []
        counts = {}
        for name, (_, arm) in data.items():
            reg = read(a.out / "references" / (name + "_ablation.json"))
            meta = design["panels"][name]
            labels = np.asarray(reg["truth"])
            rows = np.asarray(reg["rows"])
            classes = np.asarray(
                meta["primary_object_ids"] if population == "strict" else meta["object_ids"]
            )
            check(
                np.array_equal(np.asarray(meta["object_ids"])[labels], rows // 10),
                "Scoring identity differs",
            )
            hits = {k: (v == labels) for k, v in arm.items()}
            counts[name] = {k: int(v[np.isin(rows // 10, classes)].sum()) for k, v in hits.items()}
            grouped.append(
                {k: np.stack([v[rows // 10 == c] for c in classes]) for k, v in hits.items()}
            )
        rng = np.random.default_rng(SEED)
        draws = [
            rng.integers(len(next(iter(g.values()))), size=(10000, len(next(iter(g.values())))))
            for g in grouped
        ]
        n = sum(len(g["legacy"]) * 10 for g in grouped)

        def diff(h1, h2, grouped=grouped, draws=draws, n=n):
            deltas = [g[h1].astype(int) - g[h2].astype(int) for g in grouped]
            delta = sum(v.sum() for v in deltas)
            boot = sum(v.sum(1)[ix].sum(1) for ix, v in zip(draws, deltas, strict=True)) * 100 / n
            return dict(
                net=int(delta),
                gains=int(sum((v > 0).sum() for v in deltas)),
                losses=int(sum((v < 0).sum() for v in deltas)),
                delta_pp=float(delta * 100 / n),
                ci95_pp=np.quantile(boot, [0.025, 0.975]).tolist(),
            )

        comparators = ("legacy", "equal", "neural", "fisher", "learned_wrong")
        out[population] = dict(
            n=n,
            objects=n // 10,
            hits={k: sum(v[k] for v in counts.values()) for k in grouped[0]},
            per_menu_hits=counts,
            learned_minus={k: diff("learned", k) for k in comparators},
        )
    strict = out["strict"]["learned_minus"]
    check(out["strict"]["n"] == 350 and out["all"]["n"] == 480, "Incorrect denominator")
    advance = all(
        strict[k]["net"] > 0 and strict[k]["ci95_pp"][0] > 0 for k in ("legacy", "learned_wrong")
    )
    result = dict(
        design_sha256=sha(LOCK),
        scorer_sha256=sha(Path(__file__)),
        participant=SUB,
        weight=read(a.out / "fit/mixture.json")["neural_weight"],
        weight_sha256=sha(a.out / "fit/mixture.json"),
        fit_sha256=sha(a.out / "fit/final1024/receipt.json"),
        target_sha256=sha(a.out / "eeg/sub-04.npz"),
        decision_sha256={k: sha(v[0]) for k, v in data.items()},
        populations=out,
        images_authorized_by_registered_gate=bool(advance),
    )
    freeze(a.out / "summary.json", result)
    freeze(PUBLIC / "fresh_mix.json", result)
    print(
        {
            "weight": result["weight"],
            "strict": out["strict"]["hits"],
            "primary": strict["legacy"],
            "advance": advance,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--scale", type=Path, default=Path("outputs/scale-trial"))
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-mix"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    summarize(a, decisions(a))
