"""Compare two equally sized EEG negative sets for one fixed repeat-pair auxiliary loss."""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import ablate_trial_object_encoder as harness
import numpy as np
import torch
from fit_shared_photo_eeg import require
from fit_trial_object_encoder import get_records, source_data
from register_fresh_photo_panel import freeze, read, sha
from safetensors.torch import save_file
from torch.nn import functional as F

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "trial_object_negative_set_design.json"
EXPECTED = "5136188bb02ca0e99e3fa0fefa80c126d6fb5654698717ac51395ffa5bae8495"


def cross_object_logits(left, right):
    classes, photos, _ = left.shape
    slots = torch.arange(photos, device=left.device)
    offsets = (slots[None, :] - slots[:, None]) % photos
    candidates = (
        torch.arange(classes, device=left.device)[:, None, None] + offsets[None]
    ) % classes
    bank = right[candidates, slots[None, :, None]]
    return torch.einsum("cpd,cpjd->cpj", left, bank)


def loss(model, samples, permutations, full_query, coefficient):
    features = model.trials(samples.flatten(0, 1))
    index = torch.arange(len(features), device=features.device)[:, None]
    left = model.aggregate(features[index, permutations[:, :2]]).reshape(12, 10, 96)
    right = model.aggregate(features[index, permutations[:, 2:]]).reshape(12, 10, 96)
    query = model.aggregate(features).reshape(12, 10, 96) if full_query else right
    supports = left.reshape(12, 5, 2, 96)
    centers = F.normalize(left.sum(1)[:, None] - supports.sum(2), dim=-1)
    scores = torch.einsum("cfpd,kfd->cfpk", query.reshape(12, 5, 2, 96), centers)
    objects = torch.arange(12, device=scores.device).repeat_interleave(10)
    photo_scores = cross_object_logits(left, right)
    photos = torch.arange(10, device=scores.device).repeat(12)
    return F.cross_entropy(scores.reshape(-1, 12) * 12, objects) + coefficient * F.cross_entropy(
        photo_scores.reshape(-1, 10) * 12, photos
    )


def run(a):
    spec = read(DESIGN)
    hspec = read(harness.DESIGN)
    full = a.ablation / "weighted_pair"
    full_run = read(full / "receipt.json")
    require(
        sha(DESIGN) == EXPECTED
        and sha(Path(harness.__file__)) == spec["original_ablation_driver_sha256"]
        and sha(harness.DESIGN) == spec["original_ablation_design_sha256"]
        and sha(full / "receipt.json") == spec["full_arm_receipt_sha256"]
        and sha(PUBLIC / "trial_object_ablation.json") == spec["original_ablation_result_sha256"]
        and sha(a.original / "receipt.json") == hspec["original_full_generic_training_receipt"]
        and all(sha(full / k) == v for k, v in full_run["files"].items()),
        "Prior experimental inputs changed",
    )
    baseline = read(a.original / "receipt.json")
    train, test = (np.asarray(baseline[k]) for k in ("train_classes", "test_classes"))
    require(
        len(train) == 416 and len(test) == 96 and not np.intersect1d(train, test).size,
        "Wrong classes",
    )
    records = get_records(a.source, ("generic512",))
    require(
        baseline["sources"]
        == {str(i): records[str(i)]["output_sha256"] for i in np.r_[train, test]}
        and sha(a.sessions) == baseline["session_manifest_sha256"],
        "Registration differs",
    )
    device = torch.device(a.device)
    x = source_data(a.source, train, records, device)
    scale, mean = torch.std_mean(x, dim=(0, 1, 2, 4), unbiased=False)
    scale = scale.clamp_min(1e-5)[:, None]
    mean = mean[:, None]
    x.sub_(mean).div_(scale).clamp_(-12, 12)
    dates = np.array(read(a.sessions)["subjects"]["sub-04"]["session_pairs"])
    _, sessions = np.unique(dates[train], axis=0, return_inverse=True)
    # The existing trainer keeps all episode/RNG instrumentation and optimization settings.
    with patch.object(harness, "loss_for", loss):
        net, trace = harness.fit(x, sessions, True, 0.2, spec, device)
    require(trace == full_run["trace_sha256"], "Initialization or random streams differ")
    del x
    x = source_data(a.source, test, records, device)
    with torch.inference_mode():
        y = torch.cat(
            [net(((batch - mean) / scale).clamp(-12, 12)) for batch in x.flatten(0, 1).split(128)]
        )
    y = y.reshape(len(test), 10, 96).cpu().numpy()
    require(
        np.isfinite(y).all() and np.allclose(np.linalg.norm(y, axis=-1), 1, atol=1e-5),
        "Nonunit output",
    )
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
    save_file(
        dict(state, input_mean=mean.cpu(), input_scale=scale.cpu()), out / "model.safetensors"
    )
    np.savez(out / "embeddings.npz", ids=test, features=y)
    analysis = {}
    for case, v in harness.score_embeddings(a.original, test, y).items():
        np.savez(out / (case + ".npz"), **v)
        with np.load(full / (case + ".npz"), allow_pickle=False) as z:
            require(
                np.array_equal(z["rows"], v["rows"]) and np.array_equal(z["truth"], v["truth"]),
                "Menus differ",
            )
            truth = v["truth"]
            ids = np.unique(truth)
            draws = np.random.default_rng(spec["bootstrap_seed"]).integers(96, size=(10000, 96))
            entry = {}
            for mode in ("mixed", "deep"):
                a_hit = z[mode] == truth
                b_hit = v[mode] == truth
                differences = a_hit.astype(int) - b_hit
                cluster = np.array([differences[truth == c].sum() for c in ids])
                entry[mode] = dict(
                    n=len(truth),
                    within_object=int(a_hit.sum()),
                    cross_object=int(b_hit.sum()),
                    gained=int((differences == 1).sum()),
                    lost=int((differences == -1).sum()),
                    within_minus_cross=float(differences.mean()),
                    ci95=np.quantile(cluster[draws].sum(1) / len(truth), [0.025, 0.975]).tolist(),
                )
        analysis[case] = entry
    result = dict(
        design_sha256=sha(DESIGN),
        source_sha256=sha(Path(__file__)),
        shared_trainer_sha256=sha(Path(harness.__file__)),
        within_object_receipt_sha256=sha(full / "receipt.json"),
        same_initialization_and_random_streams=trace,
        files_sha256={
            k: sha(out / k)
            for k in ("model.safetensors", "embeddings.npz", "generic_12.npz", "generic_4.npz")
        },
        denominator="One sub-04; 416 training objects, 96 held objects, 10 photographs per object; each observed in both menu widths.",
        cases=analysis,
        positive_primary_lower_bound=analysis["generic_12"]["mixed"]["ci95"][0] > 0,
    )
    freeze(out / "result.json", result)
    freeze(PUBLIC / "trial_object_negative_set.json", result)
    print(analysis, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in dict(
        out="outputs/trial-object-negative-set",
        source="outputs/trial-object-inputs/sub-04",
        original="outputs/trial-object-encoder/generic",
        sessions="outputs/object-transfer/sessions.json",
        ablation="outputs/trial-object-ablation",
    ).items():
        p.add_argument("--" + name, type=Path, default=Path(default))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    run(p.parse_args())
