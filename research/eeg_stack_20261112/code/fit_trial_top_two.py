"""Extract class-held-out single and complementary trial views for the top-two object test."""

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import torch
from fit_shared_photo_eeg import require
from fit_trial_object_encoder import DESIGN as ORIGINAL_DESIGN
from fit_trial_object_encoder import PUBLIC, TrialObjectEncoder, fit, get_records, source_data
from register_fresh_photo_panel import freeze, read, sha
from safetensors.torch import load_file

DESIGN = PUBLIC / "trial_top_two_design.json"
PAIRS = tuple(itertools.combinations(range(4), 2))
VIEWS = [(0, 1, 2, 3), *((j,) for j in range(4)), *PAIRS]


def splits(ids, sessions, seed):
    rng = np.random.default_rng(seed)
    parts = [[], []]
    for session in np.unique(sessions):
        shuffled = rng.permutation(np.flatnonzero(sessions == session))
        for j, part in enumerate(parts):
            part.extend(shuffled[j::2])
    return [ids[np.sort(part)] for part in parts]


def context(a):
    spec = read(DESIGN)
    base = a.base / "receipt.json"
    require(sha(base) == spec["base_receipt_sha256"], "Changed original encoder training")
    meta = read(base)
    require(sha(a.raw / "generic512.json") == spec["raw_manifest_sha256"], "Changed raw receipts")
    require(sha(a.sessions) == spec["session_manifest_sha256"], "Changed recording sessions")
    require(meta["design_sha256"] == sha(ORIGINAL_DESIGN), "Changed original protocol")
    ids, held = (np.array(meta[k]) for k in ("train_classes", "test_classes"))
    require(len(set(ids)) == 416 and len(set(held)) == 96 and not set(ids) & set(held), "Split")
    sessions = np.array(read(a.sessions)["subjects"]["sub-04"]["session_pairs"])
    _, sessions = np.unique(sessions, return_inverse=True, axis=0)
    return spec, ids, held, sessions


def extract(model, raw, mu, sd):
    result = []
    with torch.inference_mode():
        for block in raw.flatten(0, 1).split(128):
            signal = model.trials(((block - mu) / sd).clamp(-12, 12))
            result.append(torch.stack([model.aggregate(signal[:, list(v)]) for v in VIEWS], 1).cpu())
    return torch.cat(result).numpy().reshape(len(raw), 10, len(VIEWS), 96)


def run(a):
    spec, ids, held, sessions = context(a)
    records = get_records(a.raw, ("generic512",))
    device = torch.device(a.device)
    groups = splits(ids, sessions[ids], spec["seed"])
    tasks = [(f"fold{j}", np.setdiff1d(ids, h), h) for j, h in enumerate(groups)]
    if a.stage == "held":
        require((a.out / "locked_rule.json").is_file(), "Calibrate and freeze the rule first")
        tasks = [("held", ids, held)]
    for name, train, test in tasks:
        folder = a.out / name
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / "views.npz"
        receipt = folder / "receipt.json"
        signature = dict(design_sha256=sha(DESIGN), training_source_sha256=sha(Path(__file__)),
                         base_trainer_sha256=sha(Path(__file__).with_name("fit_trial_object_encoder.py")),
                         train_classes=train.tolist(), test_classes=test.tolist())
        require(signature["base_trainer_sha256"] == read(a.base / "receipt.json")["source_sha256"],
                "Original encoder source differs")
        if receipt.exists():
            prior = read(receipt)
            require(prior == dict(signature, packet_sha256=sha(dest)), "Changed cached embeddings")
            continue
        if name == "held":
            path = a.base / "model.safetensors"
            require(sha(path) == read(a.base / "receipt.json")["weights_sha256"], "Changed full fit")
            state = load_file(path)
            mu, sd = state.pop("input_mean").to(device), state.pop("input_scale").to(device)
            model = TrialObjectEncoder().to(device)
            model.load_state_dict(state)
            model.eval()
        else:
            data = source_data(a.raw, train, records, device)
            sd, mu = torch.std_mean(data, dim=(0, 1, 2, 4), unbiased=False)
            mu, sd = mu[:, None], sd.clamp_min(1e-5)[:, None]
            data.sub_(mu).div_(sd).clamp_(-12, 12)
            model, _ = fit(data, sessions[train], read(ORIGINAL_DESIGN), device)
            del data
        raw = source_data(a.raw, test, records, device)
        values = extract(model, raw, mu, sd)
        require(np.allclose(np.linalg.norm(values, axis=-1), 1, atol=2e-5), "Nonunit embedding")
        np.savez(dest, ids=test, values=values)
        freeze(receipt, dict(signature, packet_sha256=sha(dest)))
        print(name, values.shape, flush=True)
        del raw, model


def paths(p):
    for key, value in dict(raw="outputs/trial-object-inputs/sub-04",
                           base="outputs/trial-object-encoder/generic",
                           sessions="outputs/object-transfer/sessions.json",
                           out="outputs/trial-top-two").items():
        p.add_argument("--" + key, type=Path, default=Path(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("folds", "held"))
    parser.add_argument("--device", default="cuda")
    paths(parser)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    run(parser.parse_args())
