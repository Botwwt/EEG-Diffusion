"""Apply the frozen trial/Fisher classifier on sub-04 and export native visual conditions."""

import argparse
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_trial_object_encoder import TrialObjectEncoder, get_records, source_data
from fit_within_photo_embedding import H14_SHA, bases, images, predict, training
from package_full_head_512 import directions
from prepare_expanded_personal import extract
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE, SIZE, axes, digest
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import packet
from run_prospective_readout import frozen_npz

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity"
PROTOCOL = PUBLIC / "trial_object_personal12_design.json"
LOCK = "6c2f49313a12bc9e6c5b2952749c3e55766fa3c54063b6586e27d415b9d8e537"
SUB = "sub-04"
PANEL = "personal12"


def check(v, msg):
    if not v:
        raise ValueError(msg)


def read_trials(base, data, ids, eeg):
    receipt = read(base / "eeg/sub-04.json")
    order = receipt["stored_slots"]
    legacy = data / "object-transfer/ranges" / REV / SUB
    tail = OFFSET + SHAPE[0] * ROW_BYTES
    channels, t = axes(
        (legacy / f"0-{OFFSET - 1}.bin").read_bytes(),
        (legacy / f"{tail}-{SIZE - 1}.bin").read_bytes(),
    )
    pick = [channels.tolist().index(n) for n in eeg["channels"].tolist()]
    check(
        np.array_equal(t[50:], eeg["times"])
        and np.array_equal(eeg["rows"] // 10, np.repeat(ids, 10)),
        "EEG axes or stimulus order differs",
    )
    rec = {v["concept_id"]: v for v in receipt["blocks"]}
    halves = eeg["half_eeg"]
    values = []
    for i, c in enumerate(ids):
        lo = OFFSET + int(c) * 10 * ROW_BYTES
        hi = lo + 10 * ROW_BYTES - 1
        b = (base / "ranges" / REV / SUB / f"{lo}-{hi}.bin").read_bytes()
        check(
            len(b) == 10 * ROW_BYTES and digest(b) == rec[int(c)]["sha256"],
            "Target physical range mismatch",
        )
        x = np.frombuffer(b, "<f8").reshape(10, *SHAPE[1:])
        post = np.take(x, pick, axis=2).astype(np.float32)
        check(
            np.array_equal(
                np.stack([post[:, slots].mean(1) for slots in order]),
                halves[:, 10 * i : 10 * i + 10],
            ),
            "The two decoders did not receive the same four physical presentations",
        )
        values.append(np.ascontiguousarray(x[:, :, :, :125], dtype=np.float32))
    result = np.concatenate(values)
    check(np.isfinite(result).all(), "Nonfinite physical inputs")
    return result


def deep_directions(data, fit, source, ids, eeg, device):
    import torch
    from safetensors.torch import load_file

    prior = read(fit / "receipt.json")
    check(
        len(set(prior["train_classes"])) == 512
        and not set(ids).intersection(prior["train_classes"] + prior["test_classes"]),
        "Trial encoder fit or known targets overlap the new menu",
    )
    weights = load_file(fit / "model.safetensors", device=device)
    mu, sd = weights.pop("input_mean"), weights.pop("input_scale")
    model = TrialObjectEncoder().to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()

    def forward(x):
        with torch.inference_mode():
            return (
                torch.cat([model(((block - mu) / sd).clamp(-12, 12)) for block in x.split(128)])
                .cpu()
                .numpy()
            )

    with np.load(fit / "embeddings.npz", allow_pickle=False) as z:
        old_ids = z["ids"]
        ref = z["features"].reshape(-1, 96)
    records = get_records(data / "trial-object-inputs" / SUB, ("stress12", "new_three4"))
    check(
        all(v["output_sha256"] == prior["sources"][k] for k, v in records.items()),
        "Known inputs changed",
    )
    x = source_data(
        data / "trial-object-inputs" / SUB, old_ids, records, torch.device(device)
    ).flatten(0, 1)
    err = float(np.max(np.abs(forward(x) - ref)))
    check(err < 2e-5, "Failed to reproduce the old 24-class readout with the fixed checkpoint")
    trial = read_trials(source, data, ids, eeg)
    new = forward(torch.from_numpy(trial).to(device))
    check(
        new.shape == (120, 96) and np.allclose(np.linalg.norm(new, axis=1), 1, atol=1e-5),
        "Bad embeddings",
    )
    return new.astype(float), err


def run(a):
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    base = a.data / "object-head-confirmation"
    fit = a.data / "trial-object-encoder/target"
    ref = base / "references/personal12_sub-04_ablation.json"
    old_root = base / "render_inputs"
    oldfile = old_root / "conditions" / SUB / PANEL / "conditions_512.npz"
    inputs = dict(
        trial_design=PUBLIC / "trial_object_encoder_design.json",
        training_driver=CODE / "fit_trial_object_encoder.py",
        model=fit / "model.safetensors",
        historical_embeddings=fit / "embeddings.npz",
        fit_receipt=fit / "receipt.json",
        development_readouts=fit / "score.json",
        decision_driver=CODE / "score_trial_object_encoder.py",
        panel_lock=base / "candidate_panel_lock.json",
        single_person_protocol=PUBLIC / "single_person_confirmation_image_design.json",
        baseline_inputs=PUBLIC / "single_person_confirmation_render_inputs.json",
        baseline_conditions=oldfile,
        registration=ref,
        sealed_eeg_receipt=base / "eeg/sub-04.json",
        intervention_renderer=CODE / "render_within_photo_ip.py",
    )
    check(
        sha(PROTOCOL) == LOCK
        and {k: sha(p) for k, p in inputs.items()} == read(PROTOCOL)["inputs"],
        "Training, stimulus order or generator registration changed",
    )
    reg = read(ref)
    rows, folds, vocab, cal = [
        np.array(reg[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
    ]
    source_rows = np.sort(rows)
    ids = np.unique(source_rows // 10)
    check(
        len(ids) == 12 and np.array_equal(source_rows, (ids[:, None] * 10 + np.arange(10)).ravel()),
        "Missing photos",
    )
    eeg_path = base / "eeg/sub-04.npz"
    check(
        sha(eeg_path) == read(inputs["sealed_eeg_receipt"])["packet_sha256"],
        "Changed physical data",
    )
    with np.load(eeg_path, allow_pickle=False) as eeg:
        deep, replay_error = deep_directions(a.data, fit, base, ids, eeg, a.device)
        physical = extract(eeg["half_eeg"].mean(0)).astype(float)
    expanded = a.data / "photo-structure/expanded"
    within = a.data / "photo-structure/within_photo"
    angle = a.data / "photo-structure/residual_angle"
    bank = images(a.data / "official-object-readout/sub-03")
    classes, x, y = training(SimpleNamespace(expanded=expanded), SUB, 512, bank)
    check(
        set(classes) == set(read(fit / "receipt.json")["train_classes"]),
        "Fisher and trial training objects differ",
    )
    fx = fisher(x, physical)
    scores = np.empty((2, 120, 12))
    for f in range(5):
        ix = np.flatnonzero(folds == f)
        qi = np.searchsorted(source_rows, rows[ix])
        si = np.searchsorted(source_rows, cal[f])
        check(
            set(cal[f].ravel()) == set(rows[folds != f])
            and np.array_equal(source_rows[si], cal[f]),
            "Leaking supports",
        )
        scores[:, ix, :] = [similarities(v[qi], v[si]) for v in (fx, deep)]
    native, deep_words = scores.argmax(-1)
    mixed = scores.mean(0).argmax(-1)
    b = bases(x, y, physical)
    res = predict(b, read(within / "selection.json")["selected"]) @ b[2].T
    scale = read(angle / "scale.json")
    dirs = directions(rows, folds, vocab, cal, mixed, res, source_rows, bank, scale)
    replay = directions(rows, folds, vocab, cal, native, res, source_rows, bank, scale)
    with np.load(oldfile, allow_pickle=False) as old:
        for key in ("rows", "folds", "support_rows", "donor_rows", "predicted_words"):
            check(np.array_equal(replay[key], old[key]), f"Fisher parity failed for {key}")
        visual_error = max(
            float(np.max(np.abs(replay[k] - old[k]))) for k in ("own", "uniform", "wrong")
        )
        same = native == mixed
        check(
            visual_error < 2e-6
            and np.array_equal(dirs["support_rows"][same], old["support_rows"][same])
            and np.allclose(dirs["own"][same], old["own"][same], rtol=0, atol=2e-6),
            "Changed historical visual decoder",
        )
    dest = a.out / "render_inputs"
    word = dest / "evaluations" / SUB / PANEL / "word_predictions.json"
    freeze(
        word,
        dict(
            subject=SUB,
            panel=PANEL,
            physical_trials_per_photo=4,
            rows=rows.tolist(),
            folds=folds.tolist(),
            words=vocab.tolist(),
            predicted=mixed.tolist(),
            registration_sha256=sha(ref),
            protocol_sha256=LOCK,
        ),
    )
    out = dest / "conditions" / SUB / PANEL / "conditions_512.npz"
    frozen_npz(
        out,
        dirs
        | dict(
            subject=SUB,
            physical_trials_per_photo=4,
            training_classes=classes,
            visual_sha256=H14_SHA,
            target_eeg_sha256=sha(eeg_path),
            word_prediction_sha256=sha(word),
            registration_sha256=sha(ref),
            selection_sha256=sha(within / "selection.json"),
            angular_scale_sha256=sha(angle / "scale.json"),
            trial_readout_design_sha256=LOCK,
        ),
    )
    config = read(old_root / "design.json")
    freeze(
        dest / "design.json",
        config
        | dict(
            cohort=[SUB],
            protocol_sha256=LOCK,
            packets_sha256={f"{SUB}/{PANEL}": sha(out)},
            registration_sha256={f"{SUB}/{PANEL}": sha(ref)},
        ),
    )
    shutil.copyfile(old_root / "radius.json", dest / "radius.json")
    packet(dest, SUB, PANEL)
    raw = a.out / "decisions.npz"
    frozen_npz(
        raw,
        dict(
            rows=rows,
            folds=folds,
            words=vocab,
            fisher=native,
            encoder=deep_words,
            mixed=mixed,
            deep_embeddings=deep.astype(np.float32),
            embedding_rows=source_rows,
            standardized_scores=scores,
        ),
    )
    # New choices and visual conditions are already frozen at this point.
    labels = np.asarray(reg["truth"])
    hits = {
        k: v == labels for k, v in dict(fisher=native, trial_only=deep_words, mixed=mixed).items()
    }
    diff = hits["mixed"].astype(float) - hits["fisher"]
    clustered = np.array([diff[labels == i].mean() for i in range(12)])
    draws = np.random.default_rng(260552).integers(12, size=(10000, 12))
    rec = dict(
        protocol_sha256=LOCK,
        registration_sha256=sha(ref),
        historical_embedding_max_abs_replay=replay_error,
        reconstructed_fisher_max_abs_visual_error=visual_error,
        fisher_word_replay_count=120,
        same_word_rows=int(same.sum()),
        effective_wrong_donors=int(sum(rows != dirs["donor_rows"])),
        n=120,
        n_objects=12,
        word_hits={k: int(v.sum()) for k, v in hits.items()},
        mixed_minus_fisher=dict(
            gains=int((diff > 0).sum()),
            losses=int((diff < 0).sum()),
            delta=float(diff.mean()),
            ci95=np.quantile(clustered[draws].mean(1), [0.025, 0.975]).tolist(),
        ),
        mixed_packet_sha256=sha(out),
        render_design_sha256=sha(dest / "design.json"),
        radius_sha256=sha(dest / "radius.json"),
        original_fisher_packet_sha256=sha(oldfile),
        raw_results_sha256=sha(raw),
        word_predictions_sha256=sha(word),
    )
    freeze(a.out / "receipt.json", rec)
    print(rec, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/trial-object-encoder/personal12"))
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
