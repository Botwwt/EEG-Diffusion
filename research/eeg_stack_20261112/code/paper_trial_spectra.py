"""Separate local wavelet power of the repeat mean and repeat deviations."""
import numpy as np
import torch

from paper_render_cache import require


FREQUENCIES_HZ = (6., 9., 13., 19., 27., 38., 54., 76.)
SAMPLE_RATE = 250
WINDOWS = ((0, 80), (80, 160), (160, 260), (260, 400), (400, 640))
FFT_SIZE = 512
SAMPLES = 250
CHANNELS = 17
REPEATS = (0, 1, 2, 3)
FEATURE_DIM = CHANNELS * len(FREQUENCIES_HZ) * len(WINDOWS)
MOMENTS = ("locked", "cross", "residual")


def wavelet_bank(device="cpu"):
    centers = torch.tensor(FREQUENCIES_HZ, device=device).view(-1, 1)
    frequency = torch.fft.fftfreq(FFT_SIZE, 1 / SAMPLE_RATE, device=device)
    positive = torch.exp(-.5 * ((frequency - centers) * 3 / centers).square()) * (frequency > 0)
    return positive / positive.square().mean(1, keepdim=True).sqrt()


def repeat_moments(coeff):
    """Compute mean power, signed cross-repeat power and within-repeat variance."""
    count = coeff.shape[1]
    require(count >= 2 and coeff.is_complex(), "Expected multiple complex EEG coefficients")
    mean = coeff.mean(1, keepdim=True)
    locked = mean[:, 0].abs().square()
    residual = (coeff - mean).abs().square().sum(1) / (count - 1)
    return locked, locked - residual / count, residual


def extract_spectra(trials, bank):
    require(trials.ndim == 4 and trials.shape[1:] == (len(REPEATS), CHANNELS, SAMPLES),
            "Expected four posterior-electrode EEG presentations at 250 Hz")
    x = trials.float()
    start = (FFT_SIZE - SAMPLES) // 2
    padded = torch.nn.functional.pad(x, (start, FFT_SIZE - SAMPLES - start))
    spectrum = torch.fft.fft(padded).unsqueeze(-2)
    coeff = torch.fft.ifft(spectrum * bank)[..., start:start + SAMPLES]
    output = {"mean": x[:, :, :, :200].mean(1).reshape(len(x), CHANNELS, 40, 5).mean(-1).flatten(1)}
    for key, stat in zip(MOMENTS, repeat_moments(coeff), strict=True):
        pool = [stat[..., int(a * SAMPLE_RATE / 1000):int(b * SAMPLE_RATE / 1000)].mean(-1) for a, b in WINDOWS]
        output[key] = torch.stack(pool, dim=-1).flatten(1)
    return output


@torch.inference_mode()
def read_spectra(raw, rows, device="cuda", batch=48):
    bank = wavelet_bank(device)
    output = {key: [] for key in ("mean", *MOMENTS)}
    for start in range(0, len(rows), batch):
        selected = np.asarray(rows[start:start + batch])
        signal = torch.from_numpy(raw[selected][:, list(REPEATS)].copy()).to(device=device, dtype=torch.float32)
        for k, features in extract_spectra(signal, bank).items():
            output[k].append(features)
    full = {k: torch.cat(v) for k, v in output.items()}
    require(all(value.shape == (len(rows), FEATURE_DIM) and torch.isfinite(value).all() for value in full.values()),
            "Unexpected local spectrum shape")
    return full
