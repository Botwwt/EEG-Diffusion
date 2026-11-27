"""Measure what trial-varying band power adds beyond the repeat-averaged EEG."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_aggregate import require
from paper_raw_semantic_readout import leave_photo_out, normalize, read_json, sha
from paper_render_cache import atomic_file, save_json
from paper_semantic_contrast import data_registry
from paper_temporal_readout import BATCH, EPOCHS, SEED, held_concepts, score, train


ARMS = ("erp", "coherent-power", "induced-power")
HZ, SAMPLES, FFT, BIN = 250, 200, 512, 10
FREQUENCIES = (6., 10., 18., 26.)
INPUT = 17 * (1 + len(FREQUENCIES)) * (SAMPLES // BIN)


def power_components(filtered):
    mean = filtered.mean(1)
    return mean.square(), filtered.var(1, unbiased=True)


def feature_blocks(trials, response):
    n = len(trials)
    reshaped = (n, 17, SAMPLES // BIN, BIN)
    voltage = trials[:, :, :, :SAMPLES].mean(1).reshape(reshaped).mean(-1).flatten(1)
    bands = torch.fft.rfft(trials, n=FFT).unsqueeze(-2) * response
    filtered = torch.fft.irfft(bands, n=FFT)[..., :SAMPLES]
    power = [g.reshape(n, 17, len(FREQUENCIES), SAMPLES // BIN, BIN).mean(-1).clamp_min(1e-12).log().flatten(1)
             for g in power_components(filtered)]
    return voltage, *power


class SpectralReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(INPUT, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(.10))
        self.heads = nn.ModuleList(nn.Linear(512, 1024) for _ in range(2))

    def forward(self, x):
        hidden = self.stem(x)
        return tuple(F.normalize(head(hidden).float(), dim=-1) for head in self.heads)


def physical_features(source, rows, n_train, identities, device):
    freq = torch.fft.rfftfreq(FFT, 1. / HZ, device=device)
    centers = torch.tensor(FREQUENCIES, device=device)[:, None]
    response = (-.5 * ((freq - centers) / (centers / 4)) ** 2).exp()
    widths = [17 * (SAMPLES // BIN), *[17 * len(FREQUENCIES) * (SAMPLES // BIN)] * 2]
    data = [np.empty((10, len(rows), width), dtype=np.float16) for width in widths]
    hashes, stats = {}, []
    for subject in range(1, 11):
        path = source / f"sub-{subject:02d}_train.npy"
        metadata = read_json(path.with_suffix(".json"))
        hashes[str(subject)] = sha(path)
        require(hashes[str(subject)] == metadata["output_sha256"] and metadata["ids_sha256"] == identities
                and metadata["checked_physical_photo_ids"] == 66160 and metadata["sample_rate_hz"] == HZ,
                "Changed physical waveforms")
        eeg = np.load(path, mmap_mode="r", allow_pickle=False)
        require(eeg.shape == (16540, 4, 17, 250) and eeg.dtype == np.float16, "Expected all four physical measurements")
        features = [np.empty((len(rows), width), dtype=np.float32) for width in widths]
        for start in range(0, len(rows), 128):
            block = slice(start, start + 128)
            trials = torch.from_numpy(eeg[rows[block]].copy()).to(device=device, dtype=torch.float32)
            for result, samples in zip(features, feature_blocks(trials, response), strict=True):
                result[block] = samples.cpu().numpy()
        stats.append([])
        for result, feature in zip(data, features, strict=True):
            fit = feature[:n_train]
            center, scale = fit.mean(0), fit.std(0).clip(1e-5)
            require(np.isfinite(feature).all(), "Nonfinite physical voltage or power")
            stats[-1].append(np.stack((center, scale)))
            result[subject - 1] = (feature - center) / scale
        print(f"Verified physical and power features for subject {subject:02d}", flush=True)
    return data, np.concatenate([np.stack(item) for item in zip(*stats, strict=True)], axis=-1), hashes


def run(benchmark, output, device):
    raw, training, dev, labels, category, image, text, layout, data_proof = data_registry(benchmark)
    hold = held_concepts(benchmark, training, dev, labels)
    rows = np.concatenate((training, hold))
    n = len(training)
    features, stats, eeg_proof = physical_features(raw, rows, n, data_proof["physical_ids"], device)
    group = np.unique(labels[training], return_inverse=True)[1]
    groups = np.argsort(group, kind="stable").reshape(1323, 10)
    visual = image[training]
    targets = [torch.from_numpy(a).float().to(device)
               for a in (visual, normalize(leave_photo_out(visual, group)), text[labels[training]])]
    output.mkdir(parents=True, exist_ok=True)
    norm = output / "normalization.npy"
    atomic_file(norm, lambda f: np.save(f, stats, allow_pickle=False))
    proof = dict(data=data_proof, eeg=eeg_proof, code=sha(Path(__file__)),
                 seeds=SEED, epochs=EPOCHS, batch_size=BATCH, repeats=4, frequencies=list(FREQUENCIES),
                 time_bins=SAMPLES // BIN, bin_samples=BIN, fft_size=FFT, normalization=sha(norm))
    for mode, power in zip(ARMS, (np.zeros_like(features[1]), features[1], features[2]), strict=True):
        print("Fit feature comparison", mode, flush=True)
        x = torch.from_numpy(np.concatenate((features[0], power), axis=-1)).to(device)
        weight_file = output / f"{mode}.pt"
        photo, semantic, run = train("visual-targets", x[:, :n], x[:, n:], *targets, groups, EPOCHS, weight_file,
                                     model_factory=SpectralReadout)
        file = output / f"{mode}.npz"
        atomic_file(file, lambda f: np.savez_compressed(f, embedding=semantic, photo_embedding=photo, row=hold,
                                                        concept=category[labels[hold]]))
        save_json(file.with_suffix(".json"), dict(arm=mode, sources=proof, predictions_sha256=sha(file),
                                                 weights_sha256=sha(weight_file), **run))
        del x
    comparisons = (("induced-power", "coherent-power"), ("induced-power", "erp"), ("coherent-power", "erp"))
    score(output, benchmark, hold, labels, category, image, layout, proof, ARMS,
          comparisons=comparisons, report_name="spectral-readout-verified.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(6)
    run(args.benchmark, args.output, torch.device(args.device))
