"""Strict-load the published priors and their EEG features; no image generation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not a.output.exists()
    import torch
    torch.set_num_threads(2)
    start = time.monotonic()
    proof = json.loads((a.assets/'completion.json').read_text())
    assert proof['status'] == 'COMPLETED_VERIFIED_PUBLIC_AUTHOR_ASSET_TRANSFER'
    expected = {x['path']: x for x in proof['files']}
    assert len(expected) == 20
    manifest = json.loads((a.source_root/'source_manifest.json').read_text())
    src = a.source_root/'Generation/diffusion_prior.py'
    assert sha(src) == next(x['export_sha256'] for x in manifest['files'] if x['author_path'] == 'Generation/diffusion_prior.py')
    spec = importlib.util.spec_from_file_location('pinned_author_diffusion_prior', src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows, failures = [], []
    for i in range(1, 11):
        subject = f'sub-{i:02d}'
        try:
            checkpoint = a.assets/f'fintune_ckpts/{subject}/diffusion_prior.pt'
            feature = a.assets/f'emb_eeg/ATM_S_eeg_features_{subject}_test.pt'
            for path in [checkpoint, feature]:
                assert sha(path) == expected[str(path)]['sha256']
            state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            ce = torch.load(feature, map_location='cpu', weights_only=True, mmap=True)
            assert isinstance(ce, torch.Tensor) and list(ce.shape) == [200, 1024]
            assert torch.isfinite(ce).all()
            model = mod.DiffusionPriorUNet(cond_dim=1024, dropout=0.1).eval()
            incompatible = model.load_state_dict(state, strict=True)
            assert not incompatible.missing_keys and not incompatible.unexpected_keys
            torch.manual_seed(1729)
            with torch.inference_mode():
                out = model(torch.randn_like(ce[:4]), torch.arange(4), ce[:4])
            assert out.shape == (4, 1024) and torch.isfinite(out).all()
            rows.append(dict(subject=subject, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                             feature=str(feature), feature_sha256=sha(feature), feature_shape=list(ce.shape),
                             feature_dtype=str(ce.dtype), feature_mean_norm=float(ce.norm(dim=1).mean()),
                             class_name='DiffusionPriorUNet', parameters=sum(x.numel() for x in model.parameters()),
                             load_missing_keys=0, load_unexpected_keys=0, finite_forward=True,
                             training_updates='UNKNOWN: published file is a parameter state dict',
                             encoder_checkpoint='NOT INCLUDED in the 71-file pinned author dataset inventory'))
        except Exception:
            failures.append(dict(subject=subject, error=traceback.format_exc()))
    scheduler = mod.Pipe(mod.DiffusionPriorUNet(cond_dim=1024), device='cpu').scheduler
    receipt = dict(status='PASSED_ALL10_AUTHOR_PRIOR_STRICT_LOAD_AND_FEATURE_FORWARD' if not failures else 'FAILED_AUTHOR_CHECKPOINT_PREFLIGHT',
                   rows=rows, failures=failures, author_source_commit=manifest['commit'],
                   source_sha256=sha(src), source_default_scheduler=dict(scheduler.config),
                   training_started=False, images_generated=0, gpu_initialized=torch.cuda.is_initialized(),
                   image_reproduction_complete=False, EEG_feature_row_identity_independently_reconstructed=False,
                   seconds=time.monotonic()-start, script_sha256=sha(Path(__file__)))
    assert not receipt['gpu_initialized']
    a.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'], subjects=len(rows), failures=failures,
                          parameters=[x['parameters'] for x in rows], seconds=receipt['seconds'])))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
