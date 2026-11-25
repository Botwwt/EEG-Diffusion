"""Prepare strictly object-disjoint EEG anchors and four-view temporal inputs for support-conditioned matching."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_shared_photo_eeg import load_photos
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity"
DESIGN = PUBLIC / "support_conditioned_object_design.json"
LOCK = "f346cd9c02ab6097066806d019f8378d378fed3247b94c3fc869946e712aaebc"


def context(a):
    d = read(DESIGN)
    check(sha(DESIGN) == LOCK, "Changed registration")
    o = a.data
    p = dict(
        prior_split=o / "episodic-object/sub-04/model.npz",
        prior_generic_receipt=o / "trial-object-encoder/generic/receipt.json",
        prior_generic_scores=o / "trial-object-encoder/generic/score.json",
        prior_classifier=o / "trial-object-encoder/generic/model.safetensors",
        physical=o / "trial-object-inputs/sub-04/generic512.json",
        expanded=o / "photo-structure/expanded/features/sub-04.json",
        session=o / "object-transfer/sessions.json",
        original_training=CODE / "fit_trial_object_encoder.py",
        menus=CODE / "fit_hard_object_pairs.py",
        decoder=CODE / "fit_personal_session_metric.py",
    )
    check({k: sha(f) for k, f in p.items()} == d["sources"], "Different training bank or baseline")
    rec = read(p["prior_generic_receipt"])
    train = np.asarray(rec["train_classes"])
    held = np.asarray(rec["test_classes"])
    with np.load(p["prior_split"]) as z:
        check(
            np.array_equal(held, z["generic_heldout_ids"])
            and np.array_equal(train, z["train_ids"][~np.isin(z["train_ids"], held)]),
            "Different historical split",
        )
    check(
        len(set(train)) == 416 and len(set(held)) == 96 and not set(train) & set(held),
        "Invalid split",
    )
    sched = np.array(read(p["session"])["subjects"]["sub-04"]["session_pairs"])
    _, cell = np.unique(sched, axis=0, return_inverse=True)
    return p, train, held, cell


def incl(a):
    from fit_shared_trial_encoder import torch_setup

    t = torch_setup()
    device = t.device(a.device)
    if device.type == "cuda":
        t.cuda.set_per_process_memory_fraction(0.24, device.index)
    return t, device


def prepare(a):
    from fit_episodic_object import apply_projection, projection
    from fit_trial_object_encoder import TrialObjectEncoder, fit, source_data
    from safetensors.torch import load_file

    t, dev = incl(a)
    p, train, held, cell = context(a)
    stage = a.stage
    rawdir = p["physical"].parent
    records = read(p["physical"])["blocks"]
    if stage == "held":
        m = read(a.root / "models/receipt.json")
        check(m["design_sha256"] == LOCK, "Evaluation requires both frozen heads")
    ids = train if stage == "train" else held
    phys = source_data(rawdir, ids, records, dev)
    ew = load_file(p["prior_classifier"], device=str(dev))
    mu, sd = ew.pop("input_mean"), ew.pop("input_scale")
    # Original measured trial order, cropped at 0.5 seconds. Five successive windows, each 63x5.
    norm = (phys - mu) / sd
    tok = (
        norm.reshape(len(ids), 10, 4, 63, 5, 5, 5)
        .mean(-1)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(len(ids), 10, 4, 5, 315)
        .cpu()
        .numpy()
    )
    del norm
    args = SimpleNamespace(expanded=a.data / "photo-structure/expanded")
    posterior = load_photos(args, "sub-04", train)
    fold = np.full(416, -1, dtype=int)

    def apply(net, raw, mean, scale):
        with t.inference_mode():
            return (
                t.cat(
                    [
                        net(((batch - mean) / scale).clamp(-12, 12))
                        for batch in raw.flatten(0, 1).split(128)
                    ]
                )
                .cpu()
                .numpy()
                .reshape(len(raw), 10, 96)
            )

    if stage == "train":
        f = np.empty((416, 10, 96))
        deep = np.empty((416, 10, 96), np.float32)
        for j, val in enumerate(np.array_split(np.random.default_rng(260564).permutation(416), 4)):
            fold[val] = j
            tr = np.setdiff1d(np.arange(416), val)
            data = phys[tr]
            sd2, mu2 = t.std_mean(data, dim=(0, 1, 2, 4), unbiased=False)
            mu2 = mu2[:, None]
            sd2 = sd2.clamp_min(1e-5)[:, None]
            data.sub_(mu2).div_(sd2).clamp_(-12, 12)
            model, _ = fit(data, cell[train[tr]], {"seed": 260553}, dev)
            deep[val] = apply(model, phys[val], mu2, sd2)
            f[val] = apply_projection(posterior[val], projection(posterior[tr], cell[train[tr]]))[
                ..., :96
            ]
            del model, data
            print("Completed generic anchor fold", j + 1, flush=True)
    else:
        model = TrialObjectEncoder().to(dev)
        model.load_state_dict(ew, strict=True)
        model.eval()
        deep = apply(model, phys, mu, sd)
        old_file = a.data / "trial-object-encoder/generic/embeddings.npz"
        check(
            sha(old_file) == read(p["prior_generic_receipt"])["packet_sha256"],
            "Wrong original embeddings",
        )
        with np.load(old_file) as z:
            check(
                np.array_equal(z["ids"], held) and np.array_equal(z["features"], deep),
                "Baseline trial embeddings differ",
            )
        raw = load_photos(args, "sub-04", held)
        f = apply_projection(raw, projection(posterior, cell[train]))[..., :96]
        fold = np.full(96, -1, dtype=int)
    dest = a.root / "features" / (stage + ".npz")
    frozen_npz(
        dest,
        dict(
            ids=ids, fold=fold, cell=cell[ids], tokens=tok, deep=deep, fisher=f, design_sha256=LOCK
        ),
    )
    freeze(
        dest.with_suffix(".json"),
        dict(
            stage=stage,
            design_sha256=LOCK,
            driver_sha256=sha(Path(__file__)),
            features_sha256=sha(dest),
            n_objects=len(ids),
            posterior_source_sha256=sha(p["expanded"]),
            waveforms_sha256=sha(p["physical"]),
            recording_schedule_sha256=sha(p["session"]),
        ),
    )
    print("Prepared", stage, len(ids), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["train", "held"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--root", type=Path, default=Path("outputs/support-conditioned"))
    p.add_argument("--device", default="cuda")
    prepare(p.parse_args())
