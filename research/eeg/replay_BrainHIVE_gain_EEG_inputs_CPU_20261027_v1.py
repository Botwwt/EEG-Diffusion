"""Independent CPU replay from bound physical mean4 EEG to D09 cached c.

Checks the released preprocessing-cache boundary, not raw event acquisition.
No fitting, generator, target features, CUDA, or revised policy is used.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
from brainhive_native_models_v1 import classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert not args.output.exists() and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    inputs = U.read(args.input_completion); panel = U.read(U.checked(inputs['panel']))
    bind_path = Path('/root/brainhive_reproduction_20261020_v1/data/actual_input_bindings.json')
    binding = U.read(bind_path); subject = binding['subjects']['sub-01']
    assert subject['train_repeats'] == 4 and len(binding['selected_indices']) == 17
    assert binding['image_manifest_sha256'] == inputs['image_manifest']['sha256']
    raw_path = Path(subject['train_path']); assert U.sha(raw_path) == subject['train_sha256']
    raw = torch.load(raw_path, map_location='cpu', mmap=True, weights_only=True)
    assert raw.shape == (16540, 63, 250)
    cache = torch.load(U.checked(inputs['native_cache']), map_location='cpu', mmap=True, weights_only=True)
    saved = torch.load(U.checked(inputs['native_checkpoint']), map_location='cpu', weights_only=True)
    assert cache['model_sha256'] == inputs['native_checkpoint']['sha256']
    model = classes()['BrainMLP'](4250, 1024, num_layers=1, dropout=.1).eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    queries = panel['Dfit'] + panel['Dcheck']; assert len(queries) == 600
    ids = [query['array_row'] for query in queries]
    eeg = raw[ids][:, binding['selected_indices'], :].contiguous()
    assert eeg.shape == (600, 17, 250) and torch.isfinite(eeg).all()
    lookup = {int(row): i for i, row in enumerate(cache['array_rows'])}
    positions = [lookup[row] for row in ids]
    assert cache['array_rows'][positions].tolist() == ids
    cached = cache['condition'][positions].clone()
    with torch.inference_mode(): replay = torch.cat([model(batch) for batch in eeg.split(128)])
    difference = replay - cached
    relative = difference.norm(dim=1) / cached.norm(dim=1).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(replay, cached, dim=1)
    assert torch.allclose(replay, cached, atol=1e-4, rtol=1e-5) and relative.max() < 5e-5
    source = U.checked(inputs['panel']).parent.parent / 'stage80'
    for partition, expected in [('Dfit', cached[:200]), ('Dcheck', cached[200:])]:
        bank = np.load(source/f'conditions_{partition}.npy', allow_pickle=False)
        assert np.array_equal(expected.numpy(), bank[:, 0])
    result = dict(status='ALL600_BOUND_MEAN4_EEG_CPU_REPLAYS_MATCH_NATIVE_CACHE_WITH_FP32_TOLERANCE',
        source=U.ref(__file__), inputs=U.ref(args.input_completion), raw_EEG=U.ref(raw_path), binding=U.ref(bind_path),
        physical_repeats=4, selected_channels=binding['selected_channels'], query_count=600,
        partitions=dict(Dfit=200, Dcheck=400), cache_array_row_identity_exact=True,
        rendered_parent_condition_bank_matches_cache_bitwise=True,
        CPU_vs_original_GPU_float32_max_absolute_error=float(difference.abs().max()),
        CPU_vs_original_GPU_float32_RMS_error=float(difference.square().mean().sqrt()),
        maximum_relative_L2_error=float(relative.max()), minimum_cosine=float(cosine.min()),
        equality_tolerance=dict(absolute=1e-4, relative=1e-5, maximum_relative_L2=5e-5),
        GPU_or_original_training_cache_modified=False, original_raw_event_pairing_independently_rebuilt=False,
        CUDA_initialized=torch.cuda.is_initialized(), model_or_policy_fitted=False, seconds=time.monotonic()-started)
    assert result['CUDA_initialized'] is False
    U.atomic(args.output, result); print(result['status'], result['CPU_vs_original_GPU_float32_max_absolute_error'])


if __name__ == '__main__': main()
