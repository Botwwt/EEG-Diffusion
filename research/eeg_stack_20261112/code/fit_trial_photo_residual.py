"""Test whether the frozen trial encoder preserves photograph identity within an object."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from fit_repeat_photo_subspace import centered, pair_hits, ridge
from fit_support_structure import unit
from fit_trial_object_encoder import TrialObjectEncoder, get_records, source_data
from fit_within_photo_embedding import images, training
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check
from safetensors.torch import load_file

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "trial_photo_residual_design.json"


def encoded(fit, source, ids, device):
    state = load_file(str(fit / "model.safetensors"), device=device)
    mean, scale = state.pop("input_mean"), state.pop("input_scale")
    model = TrialObjectEncoder().to(device).eval()
    model.load_state_dict(state)
    records = get_records(source, ("generic512",))
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(ids), 12):
            raw = source_data(source, ids[start : start + 12], records, device)
            norm = ((raw.flatten(0, 1) - mean) / scale).clamp(-12, 12)
            vectors.append(model(norm).cpu().numpy())
    return np.concatenate(vectors).reshape(len(ids), 10, 96).astype(float)


def ten_way(y, pred):
    hits = np.zeros(len(y))
    for i in range(0, 10, 2):
        keep = np.ones(10, dtype=bool)
        keep[i : i + 2] = False
        prior = y[:, keep].mean(1) - pred[:, keep].mean(1)
        reconstructed = unit(prior[:, None] + pred[:, i : i + 2])
        scores = reconstructed @ y.transpose(0, 2, 1)
        hits += (scores.argmax(-1) == np.arange(i, i + 2)).sum(1)
    return hits


def run(a):
    cfg, receipt = read(DESIGN), read(a.fit / "receipt.json")
    frozen = read(PUBLIC / "trial_object_encoder.json")
    check(receipt["weights_sha256"] == sha(a.fit / "model.safetensors"), "Wrong encoder weights")
    check(sha(a.fit / "receipt.json") == frozen["generic"]["training_receipt_sha256"], "Wrong fit")
    fit_ids, test_ids = (np.array(receipt[k]) for k in ("train_classes", "test_classes"))
    check(len(fit_ids) == 416 and len(test_ids) == 96, "Expected the generic class partition")
    ids = np.r_[fit_ids, test_ids]
    check(len(np.unique(ids)) == 512, "Overlapping object classes")
    deep = encoded(a.fit, a.trials, ids, a.device)
    with np.load(a.fit / "embeddings.npz", allow_pickle=False) as z:
        check(np.array_equal(z["ids"], test_ids), "Wrong evaluation order")
        check(np.allclose(deep[416:], z["features"], atol=2e-5, rtol=0), "Encoder replay failed")
    cids, posterior, y = training(
        SimpleNamespace(expanded=a.expanded), "sub-04", 512, images(a.bank)
    )
    lookup = {c: i for i, c in enumerate(cids)}
    order = [lookup[c] for c in ids]
    posterior, y = posterior[order], y[order]
    target = centered(y[:416])
    estimates = {
        k: ridge(x[:416], target, x[416:]) for k, x in (("posterior", posterior), ("deep", deep))
    }
    estimates["mixed"] = (estimates["posterior"] + estimates["deep"]) / 2
    estimates["exchanged"] = np.roll(estimates["mixed"], 1, axis=1)
    draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(
        len(test_ids), size=(cfg["bootstrap_draws"], len(test_ids))
    )
    report = {}
    for name, method, denominator in (
        ("pair_assignment", pair_hits, 45),
        ("within_ten", ten_way, 10),
    ):
        per = {k: method(y[416:], x) for k, x in estimates.items()}
        delta = (per["mixed"] - per["posterior"]) / denominator
        report[name] = dict(
            n=len(test_ids) * denominator,
            hits={k: float(v.sum()) for k, v in per.items()},
            mixed_minus_posterior=float(delta.mean()),
            ci95=np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist(),
        )
    freeze(
        a.out,
        dict(
            design_sha256=sha(DESIGN),
            driver_sha256=sha(Path(__file__)),
            fit_receipt_sha256=sha(a.fit / "receipt.json"),
            feature_receipt_sha256=sha(a.expanded / "features/sub-04.json"),
            evaluation=report,
            advance=report["pair_assignment"]["ci95"][0] > 0
            and report["within_ten"]["mixed_minus_posterior"] >= 0,
        ),
    )
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, default=Path("outputs/trial-object-encoder/generic"))
    p.add_argument("--trials", type=Path, default=Path("outputs/trial-object-inputs/sub-04"))
    p.add_argument("--expanded", type=Path, default=Path("outputs/photo-structure/expanded"))
    p.add_argument("--bank", type=Path, default=Path("outputs/official-object-readout/sub-03"))
    p.add_argument("--out", type=Path, default=PUBLIC / "trial_photo_residual.json")
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
