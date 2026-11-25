"""Train and package the preregistered sub-05 and sub-01 single-person replications."""

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE, SIZE, axes
from prepare_trial_object_inputs import check, convert, write_json
from prepare_trial_object_inputs import sha as bytes_sha
from register_fresh_photo_panel import freeze, read, sha

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity"
PROTOCOL = PUBLIC / "trial_object_personal12_replication_design.json"
LOCK = "ae40d1d6ed8fedb55f22442097e0d850b47c6680561d1e98ce7727635110d148"
PANEL = "personal12"


def inputs(a):
    base = a.data / "object-head-confirmation"
    original = base / "render_inputs"
    p = dict(
        expanded=a.data / "photo-structure/expanded/features" / (a.subject + ".json"),
        model_training_order=a.data / "episodic-object" / a.subject / "model.npz",
        physical_receipt=base / "eeg" / (a.subject + ".json"),
        physical_posterior=base / "eeg" / (a.subject + ".npz"),
        reference=base / "references" / f"personal12_{a.subject}_ablation.json",
        original_packet=original / "conditions" / a.subject / PANEL / "conditions_512.npz",
    )
    shared = dict(
        sub04_protocol=PUBLIC / "trial_object_personal12_design.json",
        sub04_word_result=PUBLIC / "trial_object_personal12.json",
        source_training_driver=CODE / "fit_trial_object_encoder.py",
        source_protocol=PUBLIC / "trial_object_encoder_design.json",
        sub04_packager=CODE / "package_trial_object_confirmation.py",
        registration_receipt=PUBLIC / "single_person_confirmation_render_inputs.json",
        physical_sessions=a.data / "object-transfer/sessions.json",
        renderer=CODE / "render_within_photo_ip.py",
        radius=original / "radius.json",
    )
    check(sha(PROTOCOL) == LOCK, "Different replication registration")
    protocol = read(PROTOCOL)
    check(
        {k: sha(f) for k, f in shared.items()} == protocol["inputs"]
        and {k: sha(f) for k, f in p.items()} == protocol["by_subject"][a.subject]["inputs"],
        "Frozen sources changed",
    )
    return p | shared


def prepare(a, p):
    source = read(p["expanded"])
    ids = source["training_concepts"]
    with np.load(p["model_training_order"], allow_pickle=False) as z:
        check(set(z["train_ids"]) == set(ids) and len(set(ids)) == 512, "Wrong training classes")
    cache = a.data / "object-transfer/ranges" / REV / a.subject
    tail = OFFSET + SHAPE[0] * ROW_BYTES
    channels, times = axes(
        (cache / f"0-{OFFSET - 1}.bin").read_bytes(),
        (cache / f"{tail}-{SIZE - 1}.bin").read_bytes(),
    )
    check(np.allclose(times[50:175], np.arange(125) / 250, rtol=0, atol=1e-7), "Bad time axis")
    out = a.root / a.subject / "trials"
    out.mkdir(exist_ok=True, parents=True)
    opts = SimpleNamespace(out=out, raw_cache=None, offline=a.offline)

    def one(c):
        f = out / f"{c:04d}.npy"
        rec = out / f"{c:04d}.json"
        expected = source["raw_block_sha256"][str(c)]
        if rec.exists():
            prev = read(rec)
            check(
                prev["source_raw_sha256"] == expected and prev["output_sha256"] == sha(f),
                f"Cached EEG no longer matches receipt for {c}",
            )
            return prev
        r = convert(c, expected, cache, source["source"], opts)
        write_json(rec, r)
        return r

    records = {}
    with ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(one, c): c for c in ids}
        for i, f in enumerate(as_completed(futures), 1):
            records[str(futures[f])] = f.result()
            if i % 32 == 0:
                print(a.subject, i, "/512", flush=True)
    write_json(
        out / "receipt.json",
        dict(
            subject=a.subject,
            shape=[10, 4, 63, 125],
            dtype="<f4",
            full_source=source["source"],
            full_lfs_sha256=source["source_full_lfs_sha256"],
            source_receipt_sha256=sha(p["expanded"]),
            protocol_sha256=LOCK,
            channels=channels.tolist(),
            times_seconds=times[50:175].tolist(),
            blocks={str(c): records[str(c)] for c in ids},
        ),
    )


def torch_setup(a):
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch.device(a.device)


def train(a, p):
    from tempfile import TemporaryDirectory

    import torch
    from fit_trial_object_encoder import fit, source_data
    from safetensors.torch import save_file

    device = torch_setup(a)
    folder = a.root / a.subject
    with np.load(p["model_training_order"], allow_pickle=False) as z:
        ids = z["train_ids"]
    original = a.data / "trial-object-encoder/target/receipt.json"
    check(
        sha(original) == read(p["sub04_protocol"])["inputs"]["fit_receipt"]
        and ids.tolist() == read(original)["train_classes"],
        "Replication must reuse the exact ordered original 512 training objects",
    )
    source = read(p["expanded"])
    src = folder / "trials/receipt.json"
    meta = read(src)
    blocks = meta["blocks"]
    check(
        meta["protocol_sha256"] == LOCK
        and meta["subject"] == a.subject
        and meta["source_receipt_sha256"] == sha(p["expanded"])
        and meta["full_lfs_sha256"] == source["source_full_lfs_sha256"]
        and set(blocks) == set(map(str, ids))
        and all(
            blocks[str(c)]["source_raw_sha256"] == source["raw_block_sha256"][str(c)] for c in ids
        ),
        "Incorrect prepared subject or physical trial source",
    )
    lock = a.data / "object-head-confirmation/candidate_panel_lock.json"
    check(sha(lock) == read(p["registration_receipt"])["study_sha256"], "Wrong panel registry")
    excluded = {c for v in read(lock)["panels"].values() for c in v["selected_concepts"]}
    check(len(excluded) == 29 and not excluded.intersection(ids), "Confirmation object in fit")
    spec = read(p["source_protocol"])
    manifest = dict(
        subject=a.subject,
        replication_design_sha256=LOCK,
        seed=spec["seed"],
        updates=750,
        trial_receipt_sha256=sha(src),
        training_function_sha256=sha(p["source_training_driver"]),
        session_manifest_sha256=sha(p["physical_sessions"]),
        train_ids=ids.tolist(),
    )
    target = folder / "fit"
    if target.exists():
        rec = read(target / "receipt.json")
        check(
            all(rec[k] == v for k, v in manifest.items())
            and rec["weights_sha256"] == sha(target / "model.safetensors"),
            "Existing fit differs",
        )
        return
    dataset = source_data(folder / "trials", ids, blocks, device)
    sd, mu = torch.std_mean(dataset, dim=(0, 1, 2, 4), unbiased=False)
    sd = sd.clamp_min(1e-5)[:, None]
    mu = mu[:, None]
    dataset.sub_(mu).div_(sd).clamp_(-12, 12)
    pairs = np.array(read(p["physical_sessions"])["subjects"][a.subject]["session_pairs"])
    _, cell = np.unique(pairs[ids], axis=0, return_inverse=True)
    model, losses = fit(dataset, cell, spec, device)
    with TemporaryDirectory(dir=folder) as tmp:
        stage = Path(tmp) / "fit"
        stage.mkdir()
        weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        weights.update(input_mean=mu.cpu(), input_scale=sd.cpu())
        save_file(weights, stage / "model.safetensors")
        freeze(
            stage / "receipt.json",
            manifest
            | dict(
                weights_sha256=sha(stage / "model.safetensors"),
                torch_version=str(torch.__version__),
                first_150_training_loss=float(np.mean(losses[:150])),
                last_150_training_loss=float(np.mean(losses[-150:])),
            ),
        )
        stage.rename(target)
    print(a.subject, "trained", sha(target / "model.safetensors"), flush=True)


def trial_features(a, p, eeg, fit_dir):
    import torch
    from fit_trial_object_encoder import TrialObjectEncoder
    from safetensors.torch import load_file

    device = torch_setup(a)
    rec = read(p["physical_receipt"])
    fit = read(fit_dir / "receipt.json")
    check(
        rec["subject"] == a.subject == fit["subject"]
        and rec["four_physical_repeats"]
        and rec["packet_sha256"] == sha(p["physical_posterior"])
        and fit["replication_design_sha256"] == LOCK
        and fit["weights_sha256"] == sha(fit_dir / "model.safetensors")
        and fit["trial_receipt_sha256"] == sha(a.root / a.subject / "trials/receipt.json"),
        "Wrong trained model or target waveforms",
    )
    metadata = read(a.root / a.subject / "trials/receipt.json")
    pick = [metadata["channels"].index(n) for n in eeg["channels"].tolist()]
    check(np.array_equal(eeg["times"][:125], metadata["times_seconds"]), "Wrong waveform times")
    target = a.data / "object-head-confirmation/ranges" / REV / a.subject
    blocks = {b["concept_id"]: b["sha256"] for b in rec["blocks"]}
    ids = np.unique(eeg["rows"] // 10)
    check(
        len(ids) == 12
        and set(ids) == set(blocks)
        and not set(ids).intersection(fit["train_ids"])
        and np.array_equal(eeg["rows"], (ids[:, None] * 10 + np.arange(10)).ravel()),
        "Wrong query rows",
    )
    data = []
    halves = eeg["half_eeg"]
    for i, c in enumerate(ids):
        lo = OFFSET + int(c) * 10 * ROW_BYTES
        raw = (target / f"{lo}-{lo + 10 * ROW_BYTES - 1}.bin").read_bytes()
        check(bytes_sha(raw) == blocks[c], "Changed four-trial query block")
        x = np.frombuffer(raw, "<f8").reshape(10, *SHAPE[1:])
        post = np.take(x, pick, axis=2).astype(np.float32)
        rebuilt = np.stack([post[:, slots].mean(1) for slots in rec["stored_slots"]])
        check(
            np.array_equal(rebuilt, halves[:, 10 * i : 10 * i + 10]),
            "Target R4 mismatch with Fisher",
        )
        data.append(np.ascontiguousarray(x[:, :, :, :125], dtype=np.float32))
    tensor = torch.from_numpy(np.concatenate(data)).to(device)
    weights = load_file(fit_dir / "model.safetensors", device=str(device))
    mu, sd = weights.pop("input_mean"), weights.pop("input_scale")
    model = TrialObjectEncoder().to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.inference_mode():
        deep = model(((tensor - mu) / sd).clamp(-12, 12)).cpu().numpy()
    check(
        deep.shape == (120, 96) and np.allclose(np.linalg.norm(deep, axis=1), 1, atol=1e-5),
        "Bad trial embeddings",
    )
    return deep


def package(a, p):
    from fit_fresh_photo_panel import fisher
    from fit_personal_session_metric import similarities
    from fit_within_photo_embedding import H14_SHA, bases, images, predict, training
    from package_full_head_512 import directions
    from prepare_expanded_personal import extract
    from render_within_photo_ip import packet
    from run_prospective_readout import frozen_npz

    sub = a.subject
    folder = a.root / sub
    dest = folder / "render_inputs"
    fit = folder / "fit"
    reg = read(p["reference"])
    rows, folds, vocab, cal = [
        np.array(reg[k]) for k in ("rows", "folds", "words", "calibration_rows_by_fold_and_word")
    ]
    with np.load(p["physical_posterior"], allow_pickle=False) as eeg:
        sr = eeg["rows"]
        check(np.array_equal(sr, np.sort(rows)), "Reference rows differ")
        deep = trial_features(a, p, eeg, fit)
        physical = extract(eeg["half_eeg"].mean(0)).astype(float)
    base = a.data / "photo-structure"
    selection = base / "within_photo/selection.json"
    angle = base / "residual_angle/scale.json"
    bank = images(a.data / "official-object-readout/sub-03")
    classes, x, y = training(SimpleNamespace(expanded=base / "expanded"), sub, 512, bank)
    check(
        set(classes) == set(read(fit / "receipt.json")["train_ids"]),
        "Inconsistent training objects",
    )
    fx = fisher(x, physical)
    scores = np.empty((2, 120, 12))
    for f in range(5):
        ix = np.flatnonzero(folds == f)
        q = np.searchsorted(sr, rows[ix])
        s = np.searchsorted(sr, cal[f])
        check(
            cal[f].shape == (12, 8)
            and set(cal[f].ravel()) == set(rows[folds != f])
            and np.array_equal(sr[s], cal[f]),
            "Leaking EEG or visual supports",
        )
        scores[:, ix, :] = [similarities(v[q], v[s]) for v in (fx, deep.astype(float))]
    native, neural = scores.argmax(-1)
    mixed = scores.mean(0).argmax(-1)
    b = bases(x, y, physical)
    res = predict(b, read(selection)["selected"]) @ b[2].T

    def condition(word):
        return directions(rows, folds, vocab, cal, word, res, sr, bank, read(angle))

    old = condition(native)
    new = condition(mixed)
    same = native == mixed
    with np.load(p["original_packet"], allow_pickle=False) as baseline:
        for k in ("rows", "folds", "support_rows", "predicted_words", "donor_rows"):
            check(np.array_equal(old[k], baseline[k]), f"Fisher replay failed for {k}")
        err = max(float(np.max(np.abs(old[k] - baseline[k]))) for k in ("own", "uniform", "wrong"))
        check(
            err < 2e-6
            and np.array_equal(new["support_rows"][same], baseline["support_rows"][same])
            and all(
                np.allclose(new[k][same], baseline[k][same], rtol=0, atol=2e-6)
                for k in ("own", "uniform")
            ),
            "Same-word visual directions changed",
        )
    word = dest / "evaluations" / sub / PANEL / "word_predictions.json"
    freeze(
        word,
        dict(
            subject=sub,
            panel=PANEL,
            physical_trials_per_photo=4,
            rows=rows.tolist(),
            folds=folds.tolist(),
            words=vocab.tolist(),
            predicted=mixed.tolist(),
            registration_sha256=sha(p["reference"]),
            protocol_sha256=LOCK,
        ),
    )
    out = dest / "conditions" / sub / PANEL / "conditions_512.npz"
    frozen_npz(
        out,
        new
        | dict(
            subject=sub,
            physical_trials_per_photo=4,
            training_classes=classes,
            visual_sha256=H14_SHA,
            target_eeg_sha256=sha(p["physical_posterior"]),
            word_prediction_sha256=sha(word),
            registration_sha256=sha(p["reference"]),
            selection_sha256=sha(selection),
            angular_scale_sha256=sha(angle),
            trial_readout_design_sha256=LOCK,
        ),
    )
    cfg = read(a.data / "object-head-confirmation/render_inputs/design.json")
    check(
        sha(a.data / "object-head-confirmation/render_inputs/design.json")
        == read(p["registration_receipt"])["render_design_sha256"],
        "Different renderer config",
    )
    freeze(
        dest / "design.json",
        cfg
        | dict(
            cohort=[sub],
            protocol_sha256=LOCK,
            packets_sha256={f"{sub}/{PANEL}": sha(out)},
            registration_sha256={f"{sub}/{PANEL}": sha(p["reference"])},
        ),
    )
    shutil.copyfile(p["radius"], dest / "radius.json")
    packet(dest, sub, PANEL)
    raw = folder / "decisions.npz"
    frozen_npz(
        raw,
        dict(
            rows=rows,
            folds=folds,
            words=vocab,
            fisher=native,
            mixed=mixed,
            encoder=neural,
            standardized_scores=scores,
            embedding_rows=sr,
            deep_embeddings=deep,
        ),
    )
    labels = np.asarray(reg["truth"])
    hits = {k: v == labels for k, v in dict(fisher=native, trial_only=neural, mixed=mixed).items()}
    check(
        int(hits["fisher"].sum()) == read(PROTOCOL)["by_subject"][sub]["known_fisher_hits"],
        "Baseline hit count mismatch",
    )
    delta = hits["mixed"].astype(float) - hits["fisher"]
    check(np.all(np.bincount(labels, minlength=12) == 10), "Missing observations")
    grouped = np.array([delta[labels == i].mean() for i in range(12)])
    draws = np.random.default_rng(260552).integers(12, size=(10000, 12))
    receipt = dict(
        subject=sub,
        protocol_sha256=LOCK,
        registration_sha256=sha(p["reference"]),
        fit_sha256=sha(fit / "receipt.json"),
        weights_sha256=sha(fit / "model.safetensors"),
        inputs_sha256=sha(folder / "trials/receipt.json"),
        fisher_word_replay_count=120,
        reconstructed_fisher_max_abs_visual_error=err,
        same_word_rows=int(same.sum()),
        effective_wrong_donors=int(sum(rows != new["donor_rows"])),
        n=120,
        n_objects=12,
        word_hits={k: int(v.sum()) for k, v in hits.items()},
        mixed_minus_fisher=dict(
            gains=int((delta > 0).sum()),
            losses=int((delta < 0).sum()),
            delta=float(delta.mean()),
            ci95=np.quantile(grouped[draws].mean(1), [0.025, 0.975]).tolist(),
        ),
        mixed_packet_sha256=sha(out),
        render_design_sha256=sha(dest / "design.json"),
        radius_sha256=sha(dest / "radius.json"),
        original_fisher_packet_sha256=sha(p["original_packet"]),
        raw_results_sha256=sha(raw),
        word_predictions_sha256=sha(word),
    )
    freeze(folder / "receipt.json", receipt)
    print(receipt, flush=True)


def summarize(a):
    """Group shared objects across people when bootstrapping the pooled hit difference."""
    counts = {}
    reports = {}
    for subject in read(PROTOCOL)["participants"]:
        opts = SimpleNamespace(**(vars(a) | {"subject": subject}))
        p = inputs(opts)
        folder = a.root / subject
        rec = read(folder / "receipt.json")
        raw = folder / "decisions.npz"
        reg = read(p["reference"])
        check(
            rec["subject"] == subject
            and rec["protocol_sha256"] == LOCK
            and rec["raw_results_sha256"] == sha(raw)
            and rec["registration_sha256"] == sha(p["reference"]),
            "Unknown prediction source",
        )
        with np.load(raw, allow_pickle=False) as d:
            check(np.array_equal(d["rows"], reg["rows"]), "Private evaluation rows changed")
            truth = np.asarray(reg["truth"])
            hits = (d["mixed"] == truth).astype(int)
            base = d["fisher"] == truth
            check(
                rec["word_hits"]["mixed"] == hits.sum()
                and rec["word_hits"]["fisher"] == base.sum(),
                "Public hits disagree with the frozen decisions",
            )
            obj = d["rows"] // 10
            diff = hits - base
            for c in np.unique(obj):
                ix = obj == c
                check(ix.sum() == 10, "Incomplete ten-photo object")
                n, delta = counts.get(int(c), (0, 0))
                counts[int(c)] = (n + 10, delta + int(diff[ix].sum()))
        reports[subject] = rec
    grouped = np.array([counts[k] for k in sorted(counts)])
    rng = np.random.default_rng(260552)
    ix = rng.integers(len(grouped), size=(10000, len(grouped)))
    draws = grouped[ix].sum(1)
    n, delta = grouped.sum(0)
    pooled = dict(
        n_person_photo_queries=int(n),
        distinct_objects=len(grouped),
        distinct_photographs=len(grouped) * 10,
        mixed_minus_fisher=float(delta / n),
        ci95=np.quantile(draws[:, 1] / draws[:, 0], [0.025, 0.975]).tolist(),
        sampling_unit="Resample distinct objects, retaining both people's rows when the object is shared. Participant population uncertainty is not estimated.",
    )
    freeze(a.summary, dict(protocol_sha256=LOCK, participant_results=reports, pooled=pooled))
    print(pooled, flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "train", "package", "summarize"])
    p.add_argument("--subject", choices=["sub-05", "sub-01"])
    p.add_argument(
        "--summary", type=Path, default=PUBLIC / "trial_object_personal12_replication.json"
    )
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--root", type=Path, default=Path("outputs/trial-object-replication"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, choices=range(1, 6), default=4)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    if args.stage != "summarize" and args.subject is None:
        p.error("--subject is required for preparation, training and packaging")
    return args


if __name__ == "__main__":
    a = arguments()
    if a.stage == "summarize":
        summarize(a)
        raise SystemExit
    p = inputs(a)
    if a.stage == "prepare":
        prepare(a, p)
    elif a.stage == "train":
        train(a, p)
    else:
        package(a, p)
