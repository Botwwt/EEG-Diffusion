"""Diagnose the timing and repeat stability of a four-trial spectral residual."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_object_margin_encoder import visual_transfer
from paper_oof_conditions import SOURCES
from paper_render_cache import PinnedFiles, require, save_json
from paper_spectral_conditions import bounded, fit_readouts, readouts, standardize
from paper_trial_spectra import FFT_SIZE, REPEATS, SAMPLE_RATE, SAMPLES, WINDOWS, read_spectra, wavelet_bank


PAIRINGS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))


def paired_residual_power(coeff):
    require(coeff.is_complex() and coeff.shape[1] == 4, "Require four measured trials")
    powers = [(.5 * (coeff[:, a] - coeff[:, b]).abs()) * (coeff[:, c] - coeff[:, d]).abs()
              for a, b, c, d in PAIRINGS]
    return torch.stack(powers).mean(0)


def read_pair_residual(array, rows, batch=48):
    bank = wavelet_bank("cuda")
    offset = (FFT_SIZE - SAMPLES) // 2
    blocks = []
    for start in range(0, len(rows), batch):
        actual = array[rows[start:start + batch]][:, list(REPEATS)].copy()
        x = torch.from_numpy(actual).cuda().float()
        spectrum = torch.fft.fft(F.pad(x, (offset, FFT_SIZE - SAMPLES - offset))).unsqueeze(-2)
        coeffs = torch.fft.ifft(spectrum * bank)[..., offset:offset + SAMPLES]
        power = paired_residual_power(coeffs)
        windows = [power[..., a * SAMPLE_RATE // 1000:b * SAMPLE_RATE // 1000].mean(-1) for a, b in WINDOWS]
        blocks.append(torch.stack(windows, dim=-1).flatten(1))
    return torch.cat(blocks)


def timed_residual(features, snapshot, start, stop):
    wave = standardize(features["mean"], snapshot["wave_norm"])
    params = snapshot["moments"]["residual"]
    power = standardize((features["residual"] / params["input_scale"]).asinh(), params["norm"])
    residual = (power - wave @ params["expected"]).reshape(len(wave), -1, len(WINDOWS))
    selected = torch.zeros_like(residual)
    selected[..., start:stop] = residual[..., start:stop]
    return selected.flatten(1) @ params["matrix"]


def paired_interval(values, order, first, second):
    people = len(values)
    concepts = len(order)
    raw = np.stack([row[first].astype(np.float32) - row[second] for row in values.values()])
    score = raw[:, order].mean(2)
    rng = np.random.default_rng(20261126)
    blocks = []
    for _ in range(64):
        sampled_people = rng.integers(people, size=(64, people, 1))
        sampled_concepts = rng.integers(concepts, size=(64, 1, concepts))
        blocks.append(score[sampled_people, sampled_concepts].mean((1, 2)))
    return dict(delta=float(score.mean()), ci95=np.quantile(np.concatenate(blocks), (.025, .975)).tolist())


@torch.inference_mode()
def audit(prep, eeg_root, photos_path, source, calibration_path, output):
    files = PinnedFiles()
    photo_sha = files.pin(photos_path)
    split = prep / "paper_split.json"
    split_sha = files.pin(split)
    locked = json.loads(calibration_path.read_bytes())
    beta, strength = locked["selected_wave"], locked["selected_moments"]["residual"]
    screen_sha = files.pin(calibration_path)
    require(locked["split_sha256"] == split_sha and locked["photo_sha256"] == photo_sha
            and locked["people"] == list(SOURCES), "Training calibration does not match the audited rows")
    target_path = source / "object-identity/training-photo-dino.pt"
    dino_sha = files.pin(target_path)
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    rows = target["rows"].numpy()
    require(len(rows) == 13230 and target["proof"]["split"] == split_sha
            and rows.tolist() == json.loads(split.read_bytes())["train_rows"], "DINO training rows differ")
    dino, ids, sums = target["image_features"].cuda(), target["labels"].cuda(), target["sums"].cuda()
    photos = torch.from_numpy(np.load(photos_path, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda()
    arms = ("wave", "full-residual", "wrong-residual", "early-residual", "late-residual", "paired-residual", "wrong-pair")
    result, evidence, observations = {}, {}, {}
    for person in SOURCES:
        path = eeg_root / f"sub-{person:02d}_train.npy"
        cache_path = source / f"conditions-sub-{person:02d}.pt"
        evidence[person] = dict(physical_eeg=files.pin(path), native=files.pin(cache_path))
        model = torch.load(cache_path, map_location="cpu", weights_only=True)
        proof = model["sources"]
        require(proof["eeg"]["array_sha256"] == evidence[person]["physical_eeg"] and proof["subject"] == person
                and proof["images"] == photo_sha and proof["split"] == split_sha
                and np.array_equal(model["rows"].numpy(), rows), "Unexpected source EEG, photo or split")
        eeg = np.load(path, mmap_mode="r", allow_pickle=False)
        spectra = read_spectra(eeg, rows)
        pairs = read_pair_residual(eeg, rows)
        native = model["out_of_fold"].cuda()
        folds = model["folds"].numpy()
        require(set(folds) == {1, 2, 3, 4}, "Use four whole-concept folds")
        counts = dict.fromkeys(arms, 0)
        observations[person] = {arm: np.zeros(len(rows), dtype=bool) for arm in arms}
        fold_rms = []
        for held in (1, 2, 3, 4):
            train, val = folds != held, folds == held
            require(not np.intersect1d(ids[train].cpu().numpy(), ids[val].cpu().numpy()).size,
                    "A test concept entered the training fold")
            fit_features, val_features = ({k: v[d] for k, v in spectra.items()} for d in (train, val))
            state = fit_readouts(fit_features, photos[train])
            values = readouts(val_features, state)
            paired_state = fit_readouts(dict(fit_features, residual=pairs[train]), photos[train])
            paired_out = readouts(dict(val_features, residual=pairs[val]), paired_state)["residual"]
            early = timed_residual(val_features, state, 1, 3)
            late = timed_residual(val_features, state, 3, 5)
            sizes = torch.stack([v.norm(dim=1, keepdim=True) for v in (early, late)])
            shared = sizes.amin(0)
            early, late = (v * (shared / size.clamp_min(1e-8)) for v, size in zip((early, late), sizes, strict=True))
            query_ids = ids[val]
            order = torch.argsort(query_ids, stable=True)
            donors = torch.empty_like(order)
            donors[order] = order.roll(len(order) // 2)
            require(bool((query_ids != query_ids[donors]).all()), "Wrong residual must come from another concept")
            original = native[val]
            wave = original.lerp(values["wave"], beta)
            zero = torch.zeros_like(wave)
            additions = (zero, values["residual"], values["residual"][donors], early, late, paired_out, paired_out[donors])
            center, image_center, lift = visual_transfer(photos[train], dino[train])
            concepts, inverse = torch.unique(query_ids, return_inverse=True)
            gallery = F.normalize(F.normalize(sums[concepts]) - image_center)
            positive = F.normalize(F.normalize(sums[query_ids] - dino[val]) - image_center)
            for arm, extra in zip(arms, additions, strict=True):
                estimate = bounded(original, wave + strength * extra, center)
                vec = F.normalize((estimate - center) @ lift)
                other = (vec @ gallery.T).scatter_(1, inverse[:, None], float("-inf"))
                hit = (vec * positive).sum(1) > other.max(1).values
                observations[person][arm][val] = hit.cpu().numpy()
                counts[arm] += int(hit.sum())
            fold_rms.append(dict(full=float(values["residual"].square().mean().sqrt()),
                                 paired=float(paired_out.square().mean().sqrt()),
                                 equal_time=float(early.square().mean().sqrt())))
        result[person] = dict(hits=counts, fold_rms=fold_rms)
        print("same EEG time and repeat audit", person, counts, flush=True)
    totals = {k: sum(x["hits"][k] for x in result.values()) for k in arms}
    require(totals["wave"] == locked["hits"][f"wave/{beta}"]
            and totals["full-residual"] == locked["hits"][f"residual/{beta}/{strength}"],
            "Unmodified arms must reproduce the locked calibration")
    groups = ids.cpu().numpy()
    _, counts = np.unique(groups, return_counts=True)
    require((counts == 10).all(), "The paired resampling unit is a complete ten-photo object")
    order = np.argsort(groups, kind="stable").reshape(-1, 10)
    intervals = {f"{first}_vs_{second}": paired_interval(observations, order, first, second)
                 for first, second in (("full-residual", "wave"), ("full-residual", "wrong-residual"),
                                       ("early-residual", "late-residual"), ("paired-residual", "wrong-pair"))}
    files.verify()
    save_json(output, dict(n=len(rows) * len(SOURCES), hit=totals, per_subject=result, wave_weight=beta,
                           residual_weight=strength, screen_sha256=screen_sha, dino_sha256=dino_sha,
                           sources=evidence, code_sha256=files.pin(Path(__file__)),
                           nonoverlapping_pairings=[list(p) for p in PAIRINGS], early_windows_ms=[80, 260],
                           late_windows_ms=[260, 640], conditional_paired_intervals=intervals))
    print("same EEG time and repeat audit", totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prep", "eeg-root", "photos", "source-root", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.prep, args.eeg_root, args.photos, args.source_root, args.calibration, args.output)
