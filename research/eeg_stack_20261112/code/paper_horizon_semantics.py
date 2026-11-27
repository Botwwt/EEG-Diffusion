"""Read subjects from frozen D2 horizons using train-only, matched temporal features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, require, units
from paper_horizon_readout import VIEWS, registration
from paper_raw_semantic_readout import leave_photo_out, normalize, read_json, top_credit
from paper_render_cache import PinnedFiles, atomic_file, save_json
from paper_semantic_contrast import METRICS, nearest_other
from paper_temporal_readout import BATCH, SEED, evaluate_arm, train
from paper_token_evidence import ridge_weights


ARMS = ("native-hidden", "separate-horizons", "temporal-residuals")
RIDGE = .08
EPOCHS = 24


class HorizonReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256), nn.GELU())
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 1024))
                                         for _ in range(2))

    def forward(self, eeg):
        feature = self.channels(eeg).flatten(1)
        return tuple(F.normalize(project(feature).float(), dim=-1) for project in self.projections)


def residual_weights(training):
    flat = training.flatten(0, 1).float()
    return (ridge_weights(flat[:, 0], flat[:, 1], RIDGE),
            ridge_weights(flat[:, :2].flatten(1), flat[:, 2], RIDGE))


def temporal_residuals(values, weights):
    w1, w2 = weights
    values = values.float()
    return torch.stack((values[:, :, 0], values[:, :, 1] - values[:, :, 0] @ w1,
                        values[:, :, 2] - values[:, :, :2].flatten(2) @ w2), dim=2)


def read_exports(folder, registration_proof, held, n_train, pins):
    stored, native, proof = [], [], {}
    common_model = None
    for subject in range(1, 11):
        features = folder / f"sub-{subject:02d}.npy"
        manifest = features.with_suffix(".json")
        head = features.with_suffix(".h14.npz")
        entry = read_json(manifest)
        details = entry["proof"]
        require(details["registration"] == registration_proof and details["subject"] == subject and details["views"] == list(VIEWS)
                and details["repeats"] == list(range(4)) and details["weights"] == [.72, .28]
                and details["train_size"] == n_train and entry["features_sha256"] == pins.pin(features)
                and entry["h14_sha256"] == pins.pin(head), "An export differs from the registered model or EEG queries")
        if common_model is None:
            common_model = details["source"]
        require(details["source"] == common_model, "All viewers must use the same D2 model")
        proof[str(subject)] = pins.pin(manifest)
        x = np.load(features, mmap_mode="r", allow_pickle=False)
        require(x.shape == (n_train + len(held), 4, 1024) and x.dtype == np.float16, "Wrong horizon shape")
        stored.append(torch.from_numpy(x.copy()).cuda())
        with np.load(head, allow_pickle=False) as values:
            h = values["embedding"]
            require(np.array_equal(held, values["row"]) and h.shape == (len(held), 4, 1024), "Native H/14 query order differs")
            native.append(h)
    return torch.stack(stored), normalize(np.stack(native)), proof


def inputs(prep, bank, n_labels, pins):
    info = bank.with_suffix(".json")
    receipt = read_json(info)
    hashes = {"bank": pins.pin(bank), "receipt": pins.pin(info), "visual": pins.pin(prep / "emb_train.pt")}
    require(hashes["bank"] == receipt["bank_sha256"] and hashes["visual"] == receipt["visual_embedding_sha256"],
            "The training image or text teacher changed")
    with np.load(bank, allow_pickle=False) as payload:
        category, vocab, ids, text = (payload[k] for k in ("category", "evaluation_text", "evaluation_id", "training_text"))
        order = json.loads((prep / "train_ids.json").read_bytes())["ids"]
        require(np.array_equal(n_labels, payload["labels"]) and np.array_equal(order, payload["photos"])
                and category.shape == (1654,) and text.shape == (1654, 1024)
                and vocab.shape == (1854, 1024) and len(set(ids)) == 1854,
                "Training and held-out photo identities must match")
    emb = torch.load(prep / "emb_train.pt", map_location="cpu", weights_only=True)
    key = "CLIP-ViT-H-14-laion2B-s32B-b79K"
    image = normalize(emb[key].float().numpy())
    layout = F.avg_pool2d(emb["vae"].float().reshape(16540, 4, 16, 16), 2).flatten(1).numpy()
    return image, normalize(text), category, normalize(vocab), ids, layout, hashes


def summarize(output, results, parts, labels, category, evaluation, ids, image, layout, proof):
    vocabulary = {c: i for i, c in enumerate(ids)}
    data = {}
    for panel, rows, where in (("dev", parts[1], slice(0, 200)), ("fresh", parts[2], slice(200, 700))):
        concepts = category[labels[rows]]
        names = np.asarray([vocabulary[c] for c in concepts])
        donor = nearest_other(layout[rows], labels[rows])
        x = {k: evaluate_arm(v[:, where], evaluation, names, image[rows], concepts, donor) for k, v in results.items()}
        comparisons = (
            ("separate-horizons.semantic", "native-hidden.semantic"),
            ("temporal-residuals.semantic", "native-hidden.semantic"),
            ("temporal-residuals.semantic", "separate-horizons.semantic"),
            *[(f"{arm}.semantic", "archived-native") for arm in ARMS],
        )
        paired = {}
        for a, b in comparisons:
            delta = x[a] - x[b]
            lo, hi = bootstrap_ci(units(delta, concepts, panel))
            paired[f"{a} minus {b}"] = {k: [float(delta[:, :, i].mean()), float(lo[i]), float(hi[i])]
                                        for i, k in enumerate(METRICS)}
        data[panel] = dict(queries=10 * len(rows), concepts=len(set(concepts)), truth=float(top_credit(image[rows] @ evaluation.T, names, 1).sum()),
                           hits={a: dict(zip(METRICS, v.sum((0, 1)).tolist(), strict=True)) for a, v in x.items()}, paired=paired)
        atomic_file(output / f"{panel}-scores.npz", lambda f: np.savez_compressed(f, row=rows, concept=concepts, **x))
    save_json(output / "horizon-readout-verified.json", dict(proof=proof, **data))
    print(json.dumps({k: v["hits"] for k, v in data.items()}, indent=2), flush=True)


def fit(folder, bank, output):
    from common import PREP

    pins = PinnedFiles()
    parts, labels, split_proof = registration(PREP, pins)
    held = np.concatenate(parts[1:])
    n = len(parts[0])
    image, text, category, evaluation, ids, layout, visual_proof = inputs(PREP, bank, labels, pins)
    layout = (layout - layout[parts[0]].mean(0)) / layout[parts[0]].std(0).clip(1e-5)
    eeg, native, feature_sources = read_exports(folder, split_proof, held, n, pins)
    center = eeg[:, :n].float().mean(1, keepdim=True)
    scale = eeg[:, :n].float().std(1, keepdim=True).clamp_min(1e-5)
    standardized = ((eeg.float() - center) / scale).half()
    del eeg
    named = {f"archived-{view}": native[:, :, i] for i, view in enumerate(VIEWS)}
    groups = np.unique(labels[parts[0]], return_inverse=True)[1]
    grouped = np.argsort(groups, kind="stable").reshape(1323, 10)
    visual = image[parts[0]]
    prototypes = normalize(leave_photo_out(visual, groups))
    target = [torch.from_numpy(a).float().cuda() for a in (visual, prototypes, text[labels[parts[0]]])]
    output.mkdir(parents=True, exist_ok=True)
    normalization = output / "normalization.npz"
    atomic_file(normalization, lambda f: np.savez_compressed(f, center=center.cpu().numpy(), scale=scale.cpu().numpy()))
    original = standardized[:, :, 1:]
    weights = residual_weights(original[:, :n])
    ridge_file = output / "residual-weights.pt"
    atomic_file(ridge_file, lambda f: torch.save([p.cpu() for p in weights], f))
    proof = dict(split=split_proof, teacher=visual_proof, features=feature_sources, code=pins.pin(Path(__file__)),
                 training=pins.pin(Path(__file__).with_name("paper_temporal_readout.py")), normalization=pins.pin(normalization),
                 residual=pins.pin(ridge_file), ridge=RIDGE, epochs=EPOCHS, seed=SEED, batch_size=BATCH)
    receipts, schedules = {}, set()
    for arm in ARMS:
        if arm == "native-hidden":
            values = standardized[:, :, :1].expand(-1, -1, 3, -1)
        elif arm == "separate-horizons":
            values = original
        else:
            values = temporal_residuals(original, weights).half()
        print(f"Fit {arm} with {values.shape[0]} participants and {n} training photos", flush=True)
        weight_file = output / f"{arm}.pt"
        q, z, log = train("visual-targets", values[:, :n], values[:, n:], *target, grouped, EPOCHS, weight_file,
                          model_factory=HorizonReadout)
        del values
        schedules.add(log["schedule_sha256"])
        prediction = output / f"{arm}.npz"
        atomic_file(prediction, lambda f: np.savez_compressed(f, embedding=z, photo_embedding=q, row=held,
                                                             concept=category[labels[held]]))
        receipts[arm] = dict(predictions=pins.pin(prediction), checkpoint=pins.pin(weight_file), **log)
        named[f"{arm}.photo"] = normalize(q)
        named[f"{arm}.semantic"] = normalize(z)
    require(len(schedules) == 1, "All temporal arms must use the same training order")
    verified = dict(**proof, schedule_sha256=schedules.pop(), arms=receipts)
    pins.verify()
    summarize(output, named, parts, labels, category, evaluation, ids, image, layout, verified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    fit(args.features, args.bank, args.output)
