"""Generate one photograph from one participant's EEG, a BrainMLP checkpoint, and a fixed seed."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


def load_trial_mean(path):
    """Read [repetitions, 17 posterior channels, 250 samples] of already-preprocessed EEG."""
    x = np.load(path, allow_pickle=False)
    if (x.ndim != 3 or not 1 <= len(x) <= 80 or x.shape[1:] != (17, 250)
            or x.dtype.kind != "f" or not np.isfinite(x).all()):
        raise ValueError("Expected 1–80 finite EEG presentations with shape (repeats, 17, 250)")
    return x.astype(np.float32, copy=False)


def reconstruct(trials, checkpoint, seed, output):
    import torch
    from common import HIVE, TURBO
    if str(HIVE) not in sys.path:
        sys.path.insert(0, str(HIVE))
    from diffusers import AutoencoderKL
    from paper_instance_residual import make_encoder
    from paper_render import draw_images
    from paper_training import generator

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the SDXL-Turbo reconstruction")
    if not 0 <= seed < 2**63 or output.suffix.lower() != ".png":
        raise ValueError("Use a nonnegative 63-bit seed and a .png output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with torch.inference_mode(), tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        brain = make_encoder("cuda").float().eval()
        brain.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        eeg = torch.from_numpy(trials).to("cuda").mean(0, keepdim=True)
        embedding = brain.backbone(eeg)
        pipe, adapter, projection, text, pooled, dims = generator(torch.float16)
        tokens = projection(adapter(embedding.half()))
        pipe.vae = None
        vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
        vae.enable_slicing()
        draw_images(Path(tmp), pipe, vae, (text, pooled, dims), tokens, [seed])
        # Publish only a completed PNG. An exclusive hard link preserves another caller's result.
        os.link(Path(tmp) / "00000.png", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eeg", type=Path, required=True, help="One preprocessed float .npy, shape repeats x 17 x 250")
    p.add_argument("--checkpoint", type=Path, required=True, help="The fitted native or object-token EEG encoder state_dict")
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reconstruct(load_trial_mean(args.eeg), args.checkpoint, args.seed, args.output)
