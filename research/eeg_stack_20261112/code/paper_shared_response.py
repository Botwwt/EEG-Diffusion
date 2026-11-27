"""Decode an unseen object with canonical directions learned from other viewers' training EEG."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_oof_conditions import SEED, native_features, registered_rows, require, study_root


SUBJECTS = tuple(range(1, 11))
ARMS = ("personal", "shared", "shuffled")
RANK = 128
SAMPLES, BIN = 150, 5  # 0–600 ms at 250 Hz, averaged in 20 ms bins


def group_mean(values, index):
    counts = torch.bincount(index)
    sums = values.new_zeros(len(counts), values.shape[1]).index_add_(0, index, values)
    return sums / counts.clamp_min(1)[:, None]


def noise_regularized_views(x, penalty):
    x = x - x.mean(dim=0)
    covariance = x.T @ x / len(x)
    values, basis = torch.linalg.eigh(covariance)
    eigenvalue = (values.clamp_min(0) + penalty * covariance.diagonal().mean()).clamp_min(1e-6)
    return (x @ basis) * eigenvalue.rsqrt()


def shared_directions(views, rank):
    matrix = torch.cat(views, dim=1)
    require(0 < rank <= min(matrix.shape), "Rank exceeds the common response dimensions")
    u, _, _ = torch.pca_lowrank(matrix, min(rank + 24, *matrix.shape), center=False, niter=4)
    return u[:, :rank] * len(matrix) ** .5


def ridge(x, y, penalty):
    gram = x.T @ x / len(x)
    gram.diagonal().add_(penalty)
    return torch.linalg.solve(gram, x.T @ y / len(x))


def binned(raw, rows, repeats, device):
    result = []
    for start in range(0, len(rows), 64):
        array = raw[rows[start:start + 64]][:, repeats, :, :SAMPLES].copy()
        x = torch.from_numpy(array).to(device=device, dtype=torch.float32).mean(1)
        result.append(x.reshape(-1, 17, SAMPLES // BIN, BIN).mean(-1).flatten(1))
    return torch.cat(result)


def recovered(x, state, mode):
    v = (x - state["mean"]) / state["scale"]
    return state["center"] + (v @ state["heads"][mode]["map"]) * state["heads"][mode]["gain"]


def accuracy(features, true):
    origin = true.mean(0)
    f, y = F.normalize(features - origin, dim=-1), F.normalize(true - origin, dim=-1)
    s = f @ y.T
    best = s.diagonal() >= s.max(1).values
    return dict(n=len(true), image_top1=int(best.sum()), mean_target_cos=float(s.diagonal().mean()))


def fit():
    from common import PREP, ROOT
    from paper_instance_residual import make_encoder
    from paper_render import physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, save_json

    root = study_root() / "shared-response"
    root.mkdir(parents=True, exist_ok=True)
    files = PinnedFiles()
    split = PREP / "paper_split.json"
    split_plan = json.loads(split.read_bytes())
    rows, labels, _ = registered_rows(PREP)
    _, names = np.unique(labels, return_inverse=True)
    device = torch.device("cuda")
    index = torch.from_numpy(names).to(device)
    full_path = PREP / "F_train.npy"
    official_path = PREP / "F_test.npy"
    visual = np.load(full_path, mmap_mode="r", allow_pickle=False)
    y = torch.from_numpy(visual[rows].copy()).to(device)
    y_center = y.mean(0)
    y_class = group_mean(y, index)
    y_residual = y - y_class[index]
    bases, view, eeg = [], [], []
    for s in SUBJECTS:
        trials, proof = physical_eeg(s, "train", files)
        x = binned(trials, rows, [0, 1, 2, 3], device)
        mean, scale = x.mean(0), x.std(0).clamp_min(1e-5)
        standard = (x - mean) / scale
        means = group_mean(standard, index)
        bases.append((mean, scale, standard, means, proof))
        view.append(noise_regularized_views(means, .5))
        eeg.append(trials)
    rng = np.random.default_rng(SEED)
    permutations = [rng.permutation(len(y_class)) for _ in SUBJECTS]
    shuffled = [v[ix] for v, ix in zip(view, permutations, strict=True)]
    full_digest = files.pin(full_path)
    split_digest = files.pin(split)
    training_eeg = {str(s): pack[-1] for s, pack in zip(SUBJECTS, bases, strict=True)}
    scores = {}
    for s, (mean, scale, standard, class_mean, _) in zip(SUBJECTS, bases, strict=True):
        i = s - 1
        detail = ridge(standard - class_mean[index], y_residual, 1.)
        all_heads = {}
        for mode, members in [
            ("personal", [view[i]]),
            ("shared", [v for j, v in enumerate(view) if j != i]),
            ("shuffled", [v for j, v in enumerate(shuffled) if j != i]),
        ]:
            torch.manual_seed(SEED + s)
            canonical = shared_directions(members, RANK)
            semantic = ridge(standard, canonical[index], 1.) @ ridge(canonical, y_class - y_center, .1)
            combined = semantic + detail
            radius = (standard @ combined).square().sum(1).mean().sqrt()
            target = (y - y_center).square().sum(1).mean().sqrt()
            gain = float((target / radius.clamp_min(1e-6)).clamp(.5, 4.))
            all_heads[mode] = {"map": combined, "gain": gain}
        state = dict(mean=mean, scale=scale, center=y_center, heads=all_heads, subject=s,
                     sources=dict(physical=training_eeg, images=full_digest, split=split_digest, code=files.pin(Path(__file__))))
        # Only the scorer reads the held-out visual feature files.
        observed, _ = physical_eeg(s, "test", files)
        official = torch.from_numpy(np.load(official_path, allow_pickle=False)).to(device)
        fresh_rows = split_plan["fresh_rows"]
        fresh = torch.from_numpy(visual[fresh_rows].copy()).to(device)
        local = ROOT / f"fits/personal-{s:02d}/enc/native/sub-{s:02d}/fold0_seed2025/brain_model.pt"
        native = make_encoder(device).eval()
        native.load_state_dict(torch.load(local, weights_only=True, map_location="cpu"))
        for panel, photo_rows, raw, repeats, target in [
            ("official-r4", split_plan["official_rows"], observed, list(range(4)), official),
            ("official-r80", split_plan["official_rows"], observed, list(range(80)), official),
            ("fresh-r4", fresh_rows, eeg[i], list(range(4)), fresh),
        ]:
            x = binned(raw, photo_rows, repeats, device)
            raw_native = [torch.from_numpy(raw[photo_rows[k:k+64]][:, repeats].copy()).to(device).float().mean(1)
                          for k in range(0, len(photo_rows), 64)]
            features = native_features(native, torch.cat(raw_native)).to(device)
            scores.setdefault(panel, {})[str(s)] = {"native": accuracy(features, target),
                                                    **{mode: accuracy(recovered(x, state, mode), target) for mode in ARMS}}
            print(panel, s, scores[panel][str(s)], flush=True)
        state["heads"] = {k: {"map": v["map"].cpu(), "gain": v["gain"]} for k, v in all_heads.items()}
        del native
        for key in ("mean", "scale", "center"):
            state[key] = state[key].cpu()
        atomic_file(root / f"sub-{s:02d}.pt", lambda stream: torch.save(state, stream))
    files.verify()
    save_json(root / "feature-results.json", {"scores": scores, "rank": RANK, "source_conditions": list(SUBJECTS),
                                             "official_images_sha256": files.pin(official_path),
                                             "train_images_sha256": full_digest, "split_sha256": split_digest})


@torch.inference_mode()
def render(subject, mode, budget):
    from common import PREP, TURBO
    from diffusers import AutoencoderKL
    from paper_render import draw_images, generator_inputs, physical_eeg, query_seed, runtime
    from paper_render_cache import PinnedFiles, render_transaction
    from paper_score import score
    from paper_training import generator

    root = study_root() / "shared-response"
    files = PinnedFiles()
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    panel = budget.split("-")[0]
    rows = json.loads(split.read_bytes())[f"{panel}_rows"]
    phase = "test" if panel == "official" else "train"
    raw, physical = physical_eeg(subject, phase, files)
    repeats = list(range(80 if budget.endswith("r80") else 4))
    pack = root / f"sub-{subject:02d}.pt"
    checkpoint_hash = files.pin(pack)
    state = torch.load(pack, map_location="cuda", weights_only=True)
    require(state["subject"] == subject and state["sources"]["split"] == split_sha, "Wrong readout or data split")
    predicted = recovered(binned(raw, rows, repeats, "cuda"), state, mode)
    base_assets = generator_inputs(files)
    pipe, adapter, projection, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    condition = torch.cat([projection(adapter(x.half())).cpu() for x in predicted.split(20)])
    seeds = [query_seed(panel, i) for i in range(len(rows))]
    dest = root / "images" / mode / budget / f"sub-{subject:02d}"
    request = dict(render_cache_version=2, subject=subject, arm=mode, training="shared-response",
                   budget=budget, n=len(rows), row=rows, repeats=[repeats], seeds=seeds, draws=1, prompt="",
                   unet_calls_per_draw=4, physical_eeg=physical, split_sha256=split_sha,
                   checkpoint_sha256=checkpoint_hash, runtime=runtime(), eeg_donor=list(range(len(rows))),
                   generator_files_sha256=base_assets, render_sha256=files.pin(Path(__file__)))

    def generate(stage, starts, complete):
        draw_images(stage, pipe, vae, (prompt, pooled, dims), condition, seeds,
                    batch_starts=starts, batch_complete=complete)

    render_transaction(dest, request, generate, files.verify)
    del pipe, adapter, projection, vae
    torch.cuda.empty_cache()
    if not (dest / "scores.json").is_file():
        score(dest, panel, Path(os.environ["ARCHIVE"]), dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("fit")
    image = sub.add_parser("render")
    image.add_argument("--subject", type=int, choices=SUBJECTS, required=True)
    image.add_argument("--mode", choices=ARMS, required=True)
    image.add_argument("--budget", choices=("official-r4", "official-r80", "fresh-r4"), default="official-r4")
    args = p.parse_args()
    fit() if args.action == "fit" else render(args.subject, args.mode, args.budget)
