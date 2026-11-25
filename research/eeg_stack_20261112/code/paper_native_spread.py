"""Calibrate a frozen personal Brain-HiVE encoder in native F space using only training concepts.

Run with the same RAD_ROOT and personal EEG_RUNS_ROOT as the restricted native fit:
    python paper_native_spread.py --subject 1
Each output is an ordinary BrainEncoder state_dict usable by paper_object_infer.py.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from filelock import FileLock

from paper_instance_residual import SEED, digest_array, inner_partition, make_encoder, require, sha, shadow_features
from paper_render_cache import atomic_file, save_json


MODES = ("token-center-only", "token-center-spread")


def moments(visual, native, shadow):
    """Per-coordinate means, and population RMS across photos and coordinates after centering."""
    require(visual.shape == native.shape == shadow.shape and visual.ndim == 2 and len(visual) > 1,
            "The three F arrays must cover the same photographs and coordinates")
    result = {}
    for name, matrix in (("visual", visual), ("native", native), ("shadow", shadow)):
        x = matrix.detach().cpu().double()
        require(torch.isfinite(x).all(), f"Nonfinite {name} F")
        center = x.mean(0)
        scale = float((x-center).square().mean().sqrt())
        require(math.isfinite(scale) and scale > 0., f"Degenerate centered {name} F spread")
        result[f"mean_{name}"] = center.tolist()
        result[f"rms_{name}"] = scale
    result["gain"] = result["rms_visual"] / result["rms_shadow"]
    require(math.isfinite(result["gain"]) and result["gain"] > 0., "Invalid OOF scale")
    return result


def audit_oof(rows, labels, folds, parts, checks):
    require(len(checks) == 3 and len(rows) == len(labels) == len(folds) == len(parts), "Incomplete shadow table")
    require(len(np.unique(rows)) == len(rows) and np.all(np.isin(folds, [0, 1, 2])), "Invalid OOF row assignments")
    require(len(set(zip(labels.tolist(), folds.tolist()))) == len(np.unique(labels)), "A concept spans OOF folds")
    coverage = np.zeros(len(rows), dtype=np.int8)
    audited = []
    for k, check in enumerate(checks):
        train = np.flatnonzero((parts == "fit") & (folds != k))
        pred = np.flatnonzero(folds == k)
        require(len(train) and len(pred) and not np.intersect1d(labels[train], labels[pred]).size,
                "OOF training and predicted concepts overlap or are empty")
        expected = dict(fold=k, train_rows=len(train), predict_rows=len(pred),
                        train_rows_sha256=digest_array(rows[train]), predicted_rows_sha256=digest_array(rows[pred]))
        require(check == expected, f"OOF receipt disagrees with the registered rows in fold {k}")
        coverage[pred] += 1
        audited.append(dict(**expected, train_concepts=len(np.unique(labels[train])),
                            predicted_concepts=len(np.unique(labels[pred])), seed=2025+k,
                            optimizer_steps=25*math.ceil(len(train)/1024)))
    require(np.all(coverage == 1), "OOF predictions do not cover every permitted row exactly once")
    return dict(predicted_rows=len(rows), concepts=len(np.unique(labels)), concept_disjoint=True,
                each_row_once=True, folds=audited, optimizer_steps=sum(q["optimizer_steps"] for q in audited))


def terminal_norm(encoder):
    layers = encoder.backbone.block.layers
    layer = layers[-1].norm
    name = f"backbone.block.layers.{len(layers)-1}.norm"
    require(isinstance(layer, torch.nn.LayerNorm) and layer.elementwise_affine and layer.bias is not None
            and encoder.get_submodule(name) is layer, "Expected the native terminal affine LayerNorm")
    return name, layer


def folded_state(encoder, mean_visual, mean_native, gain):
    name, norm = terminal_norm(encoder)
    require(mean_visual.shape == mean_native.shape == norm.weight.shape and math.isfinite(gain) and gain > 0,
            "Affine calibration and native output dimensions disagree")
    require(torch.isfinite(mean_visual).all() and torch.isfinite(mean_native).all(), "Nonfinite translation")
    state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    gamma, beta = state[name+".weight"], state[name+".bias"]
    state[name+".weight"] = (gain*gamma.double()).to(gamma.dtype)
    state[name+".bias"] = (mean_visual.cpu().double() + gain*(beta.double()-mean_native.cpu().double())).to(beta.dtype)
    return state


@torch.inference_mode()
def native_predictions(model, means):
    # Prove that the discovered LayerNorm output is exactly the value used by the native renderer.
    _, norm = terminal_norm(model)
    observed = []
    hook = norm.register_forward_hook(lambda _module, _args, value: observed.append(value))
    try:
        first = model.backbone(means[:1])
        require(len(observed) == 1 and torch.equal(first, observed[0]), "Further operations follow the LayerNorm")
    finally:
        hook.remove()
    return torch.cat([model.backbone(batch).float().cpu() for batch in means.split(128)])


def registered_inputs(subject):
    from common import HIVE, PREP, PRIOR, RUNS, concept_folds

    require(RUNS.name == f"personal-{subject:02d}", "Use the restricted personal native fitting root")
    native = RUNS / f"enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
    fit_receipt = native.with_name("fit.json")
    fit = json.loads(fit_receipt.read_bytes())
    fixed = dict(sub=subject, seed=2025, fold=0, nfold=5, mode="r4", inorm=0, t0=0, t1=250, epochs=25,
                 chan="correct", emb="", chkeep=0, target="fused", tag="native", paper_fit_only=True)
    require(fit["n_train"] == 13230 and all(fit["args"].get(k) == v for k, v in fixed.items()),
            "The calibration requires the original restricted R4 native fit")
    plan = PREP / "paper_split.json"
    rows = np.array(json.loads(plan.read_bytes())["train_rows"])
    labels = np.load(PREP / "train_concept.npy", allow_pickle=False)
    ok = np.load(PREP / "lat512_train_ok.npy", allow_pickle=False)
    require(labels.shape == ok.shape == (16540,) and labels.dtype.kind in "iu" and labels.min() == 0
            and labels.max() == 1653 and ok.dtype.kind == "b" and len(rows) == 13230 and rows.dtype.kind in "iu"
            and np.array_equal(rows, np.flatnonzero((concept_folds(1654)[labels] != 0) & ok)),
            "Unexpected restricted training rows")
    classes, counts = np.unique(labels[rows], return_counts=True)
    require(len(classes) == 1323 and (counts == 10).all(), "Expected all ten permitted photos per training concept")
    eeg_ids, photo_ids = PREP / "eeg_train_ids.json", PREP / "train_ids.json"
    ids = json.loads(eeg_ids.read_bytes())["ids"]
    require(len(ids) == len(set(ids)) == len(labels) and ids == json.loads(photo_ids.read_bytes())["ids"],
            "Native EEG and visual training photo IDs differ")
    brain = PREP / f"sub-{subject:02d}_train.npy"
    meta = json.loads(brain.with_suffix(".json").read_bytes())
    sources = dict(code=sha(Path(__file__)), shadow_code=sha(Path(__file__).with_name("paper_instance_residual.py")),
                   brain_source=sha(HIVE / "main/models_brain.py"), layers_source=sha(HIVE / "main/layers.py"),
                   native=sha(native), native_receipt=sha(fit_receipt), train_rows=digest_array(rows), split=sha(plan),
                   concept=sha(PREP / "train_concept.npy"), eeg=sha(brain), eeg_ids=sha(eeg_ids), photo_ids=sha(photo_ids),
                   F=sha(PREP / "F_train.npy"), embeddings=sha(PREP / "emb_train.pt"),
                   teacher=sha(PRIOR / "fusion_encoder/diffusion_pytorch_model.safetensors"))
    require(sources["eeg"] == meta["output_sha256"] and meta.get("split", "train") == "train"
            and int(str(meta.get("subject", subject)).removeprefix("sub-")) == subject, "Native EEG receipt mismatch")
    return rows, labels[rows], native, brain, sources


def shadow_statistics(subject, rows, labels, eeg, means, base, device, sources):
    from common import EMB_KEYS, HIVE, PREP, PRIOR
    if str(HIVE) not in sys.path:
        sys.path.insert(0, str(HIVE))
    from main.models_adapter import FusionEncoderModel

    fold, parts = inner_partition(labels)
    packed = torch.load(PREP / "emb_train.pt", weights_only=True, map_location="cpu", mmap=True)
    require(set(packed) == set(EMB_KEYS), "Expected the three native training visual targets")
    emb = {k: v[rows].to(device=device, dtype=torch.float32) for k, v in packed.items()}
    teacher = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(device)
    teacher.train().requires_grad_(False)
    for module in teacher.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = .1
    semantic, photo, checks = shadow_features(eeg, rows, fold, parts, means, emb, teacher, device)
    shadow = semantic[:, 0].contiguous()  # Four-trial mean EEG, excluding both two-trial diagnostics.
    del packed, emb, teacher, semantic, photo
    proof = audit_oof(rows, labels, fold, parts, checks)
    real = torch.from_numpy(np.load(PREP / "F_train.npy", mmap_mode="r")[rows].copy()).float()
    require(real.shape == shadow.shape == base.shape == (13230, 1024), "Unexpected F axes or sample coverage")
    stats = moments(real, base, shadow)
    return dict(subject=subject, sources=sources, seed=SEED, native_seed=2025, photos=len(rows),
                concepts=proof["concepts"], oof=proof, oof_features_sha256=digest_array(shadow.numpy()),
                native_features_sha256=digest_array(base.numpy()), visual_features_sha256=digest_array(real.numpy()),
                estimator="Euclidean RMS of all deviations from each distribution's own per-coordinate mean (population denominator)",
                deployment_center="Full restricted native encoder evaluated on every permitted mean-of-four training EEG",
                oof_scope="Three independent shadows predict every permissible mean-of-four row once across held concepts. Each shadow fits a subset of the original encoder's training concepts.",
                scale_rule="RMS(visual)/RMS(OOF shadow), fixed with no gain search or clipping", **stats)


def export(subject, device):
    from common import RUNS

    rows, labels, native, brain, sources = registered_inputs(subject)
    dest = RUNS / f"enc/object_tokens/sub-{subject:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    calibration = dest / "native_spread_calibration.json"
    with FileLock(str(dest / ".native-spread.lock")):
        if calibration.exists():
            cal = json.loads(calibration.read_bytes())
            require(cal["sources"] == sources and cal["subject"] == subject and cal["oof"]["each_row_once"],
                    "Existing calibration belongs to different inputs")
        else:
            cal = None
        pending = []
        for mode in MODES:
            path = dest / f"{mode}.pt"
            receipt = path.with_suffix(".json")
            if receipt.is_file():
                rec = json.loads(receipt.read_bytes())
                require(rec["mode"] == mode and rec["subject"] == subject and rec["sources"] == sources,
                        "Existing fit belongs to different training inputs")
                if path.is_file() and cal is not None:
                    require(rec["checkpoint_sha256"] == sha(path) and rec["calibration_sha256"] == sha(calibration),
                            "Existing fitted checkpoint or calibration differs")
                    continue
            pending.append(mode)
        if not pending:
            return
        raw = np.load(brain, mmap_mode="r")
        require(raw.shape == (16540, 4, 17, 250), "Unexpected physical training EEG axes")
        means = torch.cat([torch.from_numpy(raw[r].copy()).to(device=device, dtype=torch.float32).mean(1)
                           for r in np.array_split(rows, math.ceil(len(rows)/128))])
        enc = make_encoder(device).float().eval().requires_grad_(False)
        enc.load_state_dict(torch.load(native, map_location="cpu", weights_only=True), strict=True)
        base = native_predictions(enc, means)
        if cal is None:
            cal = shadow_statistics(subject, rows, labels, raw, means, base, device, sources)
            save_json(calibration, cal)
        else:
            # Identical FP32 weights can round differently across GPU backends. The frozen calibration
            # and original SHA stay in the receipt. Verify this execution's mean before folding.
            torch.testing.assert_close(base.double().mean(0), torch.tensor(cal["mean_native"], dtype=torch.float64),
                                       rtol=2e-5, atol=2e-5)
        cal_sha = sha(calibration)
        vis = torch.tensor(cal["mean_visual"], dtype=torch.float64)
        pred = torch.tensor(cal["mean_native"], dtype=torch.float64)
        verify = make_encoder(device).float().eval().requires_grad_(False)
        for mode in pending:
            gain = 1. if mode == "token-center-only" else cal["gain"]
            state = folded_state(enc, vis, pred, gain)
            verify.load_state_dict(state, strict=True)
            actual = native_predictions(verify, means)
            expected = (vis + gain*(base.double()-pred)).float()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            path = dest / f"{mode}.pt"
            atomic_file(path, lambda f: torch.save(state, f))
            rec = dict(mode=mode, subject=subject, checkpoint_sha256=sha(path), sources=sources,
                       photos=len(rows), concepts=cal["concepts"], seed=SEED, native_seed=2025, steps=0,
                       calibration_sha256=cal_sha, calibration_file=calibration.name, gain=gain,
                       oof_verified=True, oof=cal["oof"], oof_features_sha256=cal["oof_features_sha256"],
                       shadow_optimizer_steps=cal["oof"]["optimizer_steps"],
                       affine=terminal_norm(enc)[0], checked_actual_eeg=len(rows),
                       max_abs_folding_error=float((actual-expected).abs().max()),
                       frozen_generator=True, extra_inference_operations=0,
                       inference="Mean of the participant's EEG, calibrated native BrainMLP, unchanged adapter, projection and sampler")
            save_json(path.with_suffix(".json"), rec)
            print(mode, subject, "gain", gain, "max folded error", rec["max_abs_folding_error"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True, choices=range(1, 11))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    export(args.subject, torch.device(args.device))
