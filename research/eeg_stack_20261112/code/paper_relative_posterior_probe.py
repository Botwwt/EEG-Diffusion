"""Screen mean versus mode on held concepts with true tokens and physical R4 EEG.

The sensitivity operator in this probe is a training-photo DINOv2 readout of
tokens. The final-RGB Jacobian and image reconstruction are separate tests.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from paper_relative_posterior import conditional_mixture_estimates


SUBJECTS = (1, 3)
SEED = 20261126
RANK = 32
COMPONENTS = 16


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gram(features, ridge):
    result = features.mT @ features / len(features)
    result.diagonal().add_(ridge * result.diagonal().mean())
    return result


def ridge_fit(features, targets, penalty):
    return torch.linalg.solve(gram(features, penalty), features.mT @ targets / len(features))


def group_mean(values, index):
    count = torch.bincount(index)
    sums = values.new_zeros(len(count), values.shape[-1])
    return sums.index_add_(0, index, values) / count[:, None]


def true_relative_tokens(photos, fusion_path, projection_path, device):
    adapter = load_file(fusion_path)
    projection = load_file(projection_path)
    w1 = adapter["adapter.ff.w1.weight"].to(device, torch.float16)
    w2 = adapter["adapter.ff.w2.weight"].to(device, torch.float16)
    w = projection["proj.weight"].to(device, torch.float16)
    b = projection["proj.bias"].to(device, torch.float16)
    norm_w = projection["norm.weight"].to(device, torch.float16)
    norm_b = projection["norm.bias"].to(device, torch.float16)
    output = torch.empty(len(photos), 8192, device=device)
    for start in range(0, len(photos), 256):
        batch = photos[start:start + 256].half()
        adapted = batch + F.linear(F.gelu(F.linear(batch, w1)), w2)
        projected = F.linear(adapted, w, b).reshape(-1, 4, 2048)
        tokens = F.layer_norm(projected, (2048,), norm_w, norm_b).float()
        output[start:start + len(batch)] = (tokens - tokens.mean(1, keepdim=True)).flatten(1)
    require(bool(torch.isfinite(output).all()), "Non-finite reference tokens")
    return output


def actual_eeg(path, rows, device):
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    require(source.shape == (16540, 4, 17, 250), "Unexpected physical EEG layout")
    out = np.empty((len(rows), 680), dtype=np.float32)
    for start in range(0, len(rows), 256):
        batch = source[rows[start:start + 256], :, :, :200]
        time = batch.reshape(len(batch), 4, 17, 40, 5).mean((1, 4), dtype=np.float32)
        out[start:start + len(batch)] = time.reshape(len(batch), -1)
    require(bool(np.isfinite(out).all()), "Non-finite physical EEG")
    return torch.from_numpy(out).to(device)


def crossfit_signal(eeg, target, group_folds):
    predicted = torch.empty_like(target)
    for fold in group_folds.unique().tolist():
        hold = group_folds == fold
        h, z = eeg[~hold], target[~hold]
        hmean, zmean = h.mean(0), z.mean(0)
        predictor = ridge_fit(h - hmean, z - zmean, 0.8)
        predicted[hold] = (eeg[hold] - hmean) @ predictor + zmean
    cross = (predicted - predicted.mean(0)).mT @ (target - target.mean(0)) / len(target)
    return (cross + cross.mT) / 2


def subspaces(signal, sensitivity):
    values, vectors = torch.linalg.eigh(signal)
    require(int((values > 0).sum()) >= RANK, "Insufficient positive EEG rank")
    root = (vectors * values.clamp_min(0).sqrt()) @ vectors.mT
    _, directions = torch.linalg.eigh(root @ sensitivity @ root)
    dual = torch.linalg.qr(root @ directions[:, -RANK:]).Q
    return {"eeg-only": vectors[:, -RANK:], "eeg-dino": dual}


def fit_visual_mixture(q, generator):
    draw = torch.randperm(len(q), generator=generator, device=q.device)
    centers = q[draw[:COMPONENTS]].clone()
    for _ in range(20):
        index = torch.cdist(q, centers).argmin(dim=1)
        counts = torch.bincount(index, minlength=COMPONENTS)
        sums = torch.zeros_like(centers).index_add_(0, index, q)
        centers = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers)
    index = torch.cdist(q, centers).argmin(dim=1)
    counts = torch.bincount(index, minlength=COMPONENTS)
    cov = gram(q - centers[index], 0.1)
    return (counts + 1) / (len(q) + COMPONENTS), centers, cov


def infer_variants(train_eeg, test_eeg, train_code, true_test, base, axes, generator):
    q_train = train_code @ axes
    q_base = base @ axes
    sensing = ridge_fit(q_train, train_eeg, 0.05)
    residual = train_eeg - q_train @ sensing
    inverse_noise_sensing = torch.linalg.solve(gram(residual, 0.1), sensing.mT)
    precision = sensing @ inverse_noise_sensing
    precision = (precision + precision.mT) / 2
    information = test_eeg @ inverse_noise_sensing
    visual = fit_visual_mixture(q_train, generator)
    output = {}
    for start in range(0, len(test_eeg), 384):
        target = slice(start, start + 384)
        mean, mode = conditional_mixture_estimates(*visual, precision, information[target])
        delta_mean, delta_mode = mean - q_base[target], mode - q_base[target]
        match = delta_mode.norm(dim=-1, keepdim=True) / delta_mean.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        permutation = (torch.arange(start, min(start + 384, len(base)), device=base.device) + len(base) // 2) % len(base)
        _, other_mode = conditional_mixture_estimates(*visual, precision, information[permutation])
        methods = {"mean": delta_mean, "mode": delta_mode, "mean-norm-matched": delta_mean * match,
                   "wrong-eeg-mode": other_mode - q_base[target], "true-projection": true_test[target] @ axes - q_base[target]}
        for method, diff in methods.items():
            output.setdefault(method, []).append(base[target] + diff @ axes.mT)
    return {method: torch.cat(chunks) for method, chunks in output.items()}


def score(predicted, truth, reference, groups, readout, vision_mean):
    v = F.normalize(predicted @ readout + vision_mean, dim=-1)
    counts = torch.bincount(groups)
    means = group_mean(reference, groups)
    loo = F.normalize(means[groups] * counts[groups, None] - reference, dim=-1)
    cosine = v @ F.normalize(means, dim=-1).mT
    each = torch.arange(len(v), device=v.device)
    cosine[each, groups] = (v * loo).sum(dim=-1)
    object_hit = cosine.argmax(dim=-1) == groups

    order = groups.argsort()
    require(bool((counts == 10).all()), "Expected ten physical images per training concept")
    p = F.normalize(predicted[order].reshape(-1, 10, predicted.shape[-1]), dim=-1)
    t = F.normalize(truth[order].reshape_as(p), dim=-1)
    hit = (p @ t.mT).argmax(dim=-1) == torch.arange(10, device=v.device)
    within_hit = torch.zeros_like(object_hit)
    within_hit[order] = hit.flatten()
    return torch.stack([object_hit, within_hit], dim=-1).cpu().numpy()


@torch.inference_mode()
def run(benchmark, study, output):
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    generator = torch.Generator(device=device).manual_seed(SEED)
    fixed = benchmark / "verified" / "data" / "prep"
    split = fixed / "paper_split.json"
    rows = np.asarray(json.loads(split.read_bytes())["train_rows"])
    bank = torch.load(study / "training-photo-dino.pt", map_location="cpu", weights_only=True)
    require(bank["proof"]["split"] == digest(split) and np.array_equal(bank["rows"].numpy(), rows), "Vision bank split mismatch")
    reference = F.normalize(bank["image_features"].to(device).float(), dim=-1)
    labels = bank["labels"].to(device)
    _, group = labels.unique(return_inverse=True)
    group_folds = torch.randperm(int(group.max()) + 1, generator=generator, device=device) % 4
    fold_id = group_folds[group]

    source = benchmark / "inputs" / "fusion-prior-source"
    fusion = source / "fusion_encoder" / "diffusion_pytorch_model.safetensors"
    projection = source / "proj" / "diffusion_pytorch_model.safetensors"
    native = study / "F_train.npy"
    render = benchmark / "photo_top1_20261126" / "images" / "shared" / "o13-late" / "official-r80" / "sub-01" / "render.json"
    pinned = json.loads(render.read_bytes())["generator_files_sha256"]
    sources = {"split": digest(split), "dino": digest(study / "training-photo-dino.pt"), "fusion_features": digest(native),
               "adapter": digest(fusion), "projection": digest(projection), "reference_render": digest(render)}
    require(sources["adapter"] == pinned["prior/fusion_encoder/diffusion_pytorch_model.safetensors"] and
            sources["projection"] == pinned["prior/proj/diffusion_pytorch_model.safetensors"], "Relative token projector differs from archived renders")
    expected = json.loads((benchmark / "verified" / "expected-inputs.json").read_bytes())
    eeg = {}
    for subject in SUBJECTS:
        name = f"sub-{subject:02d}_train"
        waveforms = benchmark / "inputs" / "prep17" / (name + ".npy")
        sources[name] = digest(waveforms)
        require(sources[name] == expected[name + ".json"]["output_sha256"], "Physical EEG SHA mismatch")
        eeg[subject] = actual_eeg(waveforms, rows, device)
    relative = true_relative_tokens(torch.from_numpy(np.load(native, mmap_mode="r")[rows]).to(device), fusion, projection, device)

    scores = {s: {} for s in SUBJECTS}
    for fold in range(4):
        hold = fold_id == fold
        r = relative[~hold]
        center = r.mean(dim=0)
        _, _, pca = torch.pca_lowrank(r, q=96, center=True, niter=3)
        scale = ((r - center) @ pca).std(dim=0).clamp_min(1e-5)
        xyz = ((relative - center) @ pca) / scale
        train_code, test_code = xyz[~hold], xyz[hold]
        _, tr_group = group[~hold].unique(return_inverse=True)
        _, va_group = group[hold].unique(return_inverse=True)
        cc = group_mean(train_code, tr_group)
        ref_centers = group_mean(reference[~hold], tr_group)
        vision_mean = ref_centers.mean(0)
        visual_readout = ridge_fit(cc - cc.mean(0), ref_centers - vision_mean, 0.1)
        sensitivity = visual_readout @ visual_readout.mT
        fitted_groups = group[~hold].unique()
        for person, physical in eeg.items():
            train, query = physical[~hold], physical[hold]
            mean = train.mean(dim=0)
            sd = train.std(dim=0).clamp_min(1e-5)
            train, query = (train - mean) / sd, (query - mean) / sd
            base = query @ ridge_fit(train, train_code, 0.2)
            signal = crossfit_signal(group_mean(train, tr_group), cc, group_folds[fitted_groups])
            predictions = {"base-ridge": base, "true-code": test_code}
            for name, axes in subspaces(signal, sensitivity).items():
                fits = infer_variants(train, query, train_code, test_code, base, axes, generator)
                predictions.update({f"{name}/{method}": vals for method, vals in fits.items()})
            for name, prediction in predictions.items():
                if name not in scores[person]:
                    scores[person][name] = np.zeros((len(rows), 2), bool)
                scores[person][name][hold.cpu().numpy()] = score(prediction, test_code, reference[hold], va_group, visual_readout, vision_mean)
        print("Completed fold", fold + 1, "with", int(hold.sum()), "held training photos", flush=True)

    rows_report = {str(s): {name: vals.sum(0).tolist() for name, vals in v.items()} for s, v in scores.items()}
    totals = {name: np.sum([scores[s][name].sum(0) for s in SUBJECTS], axis=0).tolist() for name in scores[SUBJECTS[0]]}
    report = dict(phase="Pre-generation proxy diagnostic only", comparator="posterior mean and mode of the same fitted mixture",
                  metrics=["held-concept DINOv2 identification, own photo excluded from positive prototype", "within-concept ten-photo token identification"],
                  n_per_subject=len(rows), subjects=SUBJECTS, total=len(rows) * len(SUBJECTS), rank=RANK,
                  held_concept_folds=4, mixture_components=COMPONENTS, sources=sources, totals=totals, by_subject=rows_report)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "observations.npz", row=rows, concept=labels.cpu().numpy(), fold=fold_id.cpu().numpy(),
                        **{f"sub-{s:02}/{name}": vals for s, methods in scores.items() for name, vals in methods.items()})
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.benchmark, args.study, args.output)
