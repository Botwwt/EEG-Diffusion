"""Synthetic counterexample, not a replay of the unknown released preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import scipy
from scipy.signal import resample


def run():
    start = time.monotonic()
    rng = np.random.default_rng(1729)
    signal = rng.normal(size=(17, 4000))
    spectrum = np.fft.rfft(signal, axis=-1)
    frequency = np.fft.rfftfreq(4000, d=.001)
    spectrum[:, (frequency < 1) | (frequency > 95)] = 0
    signal = np.fft.irfft(spectrum, n=4000, axis=-1)
    rotation = rng.normal(size=(17, 17))
    rotation, _ = np.linalg.qr(rotation)
    signal = rotation @ signal
    records = []

    def metric(a, b):
        a, b = a[:, 84:149], b[:, 84:149]
        da, db = np.diff(a, axis=-1), np.diff(b, axis=-1)
        denominator = np.sqrt((np.square(da).sum() + np.square(db).sum()) / 2)
        return dict(relative_difference_RMS=float(np.linalg.norm(da - db) / denominator),
                    difference_cosine=float(np.vdot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db))))

    for length in (1200, 1201):
        # Both epochs are cuts from exactly the SAME continuously sampled data,
        # exactly200 original samples apart, with shared spatial processing.
        x = signal[:, 800:800 + length].copy()
        y = signal[:, 1000:1000 + length].copy()
        x -= x[:, :200].mean(-1, keepdims=True)
        y -= y[:, :200].mean(-1, keepdims=True)
        a = resample(x, 300, axis=-1)[:, 50:]
        b = resample(y, 300, axis=-1)[:, 50:]
        result = metric(a[:, 50:], b)
        assert result['relative_difference_RMS'] > 1e-4
        records.append(dict(branch='Independent_epoch_FFT_resampling', original_epoch_samples=length,
                            resampled_epoch_samples=300, nominal_sampling_rate=250,
                            implied_Scipy_sample_interval_seconds=.001 * length / 300, **result))
    continuous = resample(signal, 1000, axis=-1)
    a = continuous[:, 250:500]
    b = continuous[:, 300:550]
    result = metric(a[:, 50:], b)
    assert result['relative_difference_RMS'] < 1e-12
    records.append(dict(branch='Common_continuous_FFT_resampling_then_epoch', **result))
    a = signal[:, 1000:2000:4]
    b = signal[:, 1200:2200:4]
    result = metric(a[:, 50:], b)
    assert result['relative_difference_RMS'] < 1e-12
    records.append(dict(branch='Uniform_decimation_on_common_grid_then_epoch', **result))
    return dict(status='FINITE_EPOCH_RESAMPLING_NEED_NOT_COMMUTE_WITH_SHARED_SEGMENT_SHIFT_COUNTEREXAMPLE_PASSED',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=1729,
                NumPy=np.__version__, SciPy=scipy.__version__, results=records, seconds=time.monotonic() - start,
                actual_EEG_read=False, actual_cache_preprocessing_reproduced=False,
                MNE_padding_or_historical_version_reproduced=False,
                D33_result_or_threshold_changed=False, new_model_fits=0, new_GPU_jobs=0, new_images=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))
