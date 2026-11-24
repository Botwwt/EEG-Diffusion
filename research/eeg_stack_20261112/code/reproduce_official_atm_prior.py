"""CPU reproduction of the released subject-03 ATM feature diffusion prior.

The default generates the 10-step / CFG 2 demo and 50-step / CFG 5 metric
recipes, with one prespecified sample per official R80 EEG and recipe.
--evaluate opens the text gallery and official reference embeddings only after
both feature arrays have been written. Runtime uses diffusers 0.30.0.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from official_object_readout import ATM, acquire, evaluate_readout, predict, sha, unit

COMMIT = "49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d"
UPSTREAM = f"https://raw.githubusercontent.com/dongyangli-del/EEG_Image_decode/{COMMIT}/Generation/"
SOURCE = (
    UPSTREAM + "diffusion_prior.py",
    "821765335dc965ecc7511add10c9262bc379a11504b6e9ae039fcbee01d5950c",
)
CHECKPOINT = (
    ATM + "fintune_ckpts/sub-03/diffusion_prior.pt",
    "27c0454e8fbef5d28302d28a5a02f47dd00995a3897b52e355ff2cac5329aeae",
)
RECIPES = (
    ("demo_10_cfg2", 10, 2.0, "1x1024_reconstruct_sdxl.ipynb"),
    ("metrics_50_cfg5", 50, 5.0, "Generation_metrics_sub8.ipynb"),
)
SEED = 1729


def sample(model, eeg, rows, steps, cfg):
    import torch
    from diffusers.schedulers import DDPMScheduler

    scheduler = DDPMScheduler()
    scheduler.set_timesteps(steps, device="cpu")
    rngs = [torch.Generator(device="cpu").manual_seed(SEED + i) for i in rows]
    h = torch.cat([torch.randn(1, 1024, generator=g) for g in rngs])
    with torch.inference_mode():
        for t in scheduler.timesteps:
            t_batch = torch.full((len(h),), t.item(), dtype=torch.float32)
            con = model(h, t_batch, eeg)
            unc = model(h, t_batch)
            eps = unc + cfg * (con - unc)
            h = scheduler.step(eps, t, h, generator=rngs).prev_sample
    return h


def reference_metrics(features, images):
    score = unit(features) @ images.T
    ranks = (score > score.diagonal()[:, None]).sum(axis=1) + 1
    return {
        "top1": int((ranks == 1).sum()),
        "top5": int((ranks <= 5).sum()),
        "median_rank": float(np.median(ranks)),
        "2afc": float(np.mean((len(ranks) - ranks) / (len(ranks) - 1))),
        "ranks": ranks.tolist(),
    }


def main():
    import diffusers
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--threads", default=6, type=int)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if diffusers.__version__ != "0.30.0" or args.batch < 1 or args.threads < 1:
        raise ValueError("Use diffusers==0.30.0 and positive batch/thread counts")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    source = acquire(args.base / "assets/atm_author_diffusion_prior.py", SOURCE)
    ckpt = acquire(args.base / "assets/atm_diffusion_prior.pt", CHECKPOINT)
    spec = importlib.util.spec_from_file_location("pinned_atm_original", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    model = original.DiffusionPriorUNet(cond_dim=1024, dropout=0.1).cpu().float().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    eeg = torch.from_numpy(np.load(args.base / "eeg_r80.npy")).cpu().float()
    if eeg.shape != (200, 1024) or not torch.isfinite(eeg).all():
        raise ValueError("Expected the released raw sub-03 R80 feature tensor")
    metadata = {
        "sub": "sub-03",
        "trials": 80,
        "input_transform": "none; published raw ATM EEG embedding, no norm or multiplier",
        "output": "raw 1024d output of author DDPMScheduler sample",
        "seed_policy": "one independent CPU generator per row, seed 1729 + original EEG row",
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "source": SOURCE,
        "checkpoint": CHECKPOINT,
        "eeg_npy_sha256": sha(args.base / "eeg_r80.npy"),
        "runs": {},
    }
    for name, steps, cfg, notebook in RECIPES:
        output = []
        for i in range(0, len(eeg), args.batch):
            rows = range(i, min(len(eeg), i + args.batch))
            output.append(sample(model, eeg[i : i + args.batch], rows, steps, cfg))
        values = torch.cat(output)
        # Author Pipe.generate calls .item() on the timestep, so we verify
        # vectorization against its original single-image path on the same RNG.
        with torch.inference_mode():
            reference = original.Pipe(model, device="cpu").generate(
                eeg[:1],
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=torch.Generator(device="cpu").manual_seed(SEED),
            )
        error = (reference[0] - values[0]).abs().max().item()
        if not torch.allclose(reference[0], values[0], rtol=7e-4, atol=1e-4):
            raise ValueError(f"Batched prior disagrees with author single-row path: {error}")
        dest = args.out / name
        dest.mkdir(exist_ok=True)
        path = dest / "h14_prediction.npy"
        np.save(path, values.numpy())
        metadata["runs"][name] = {
            "steps": steps,
            "guidance": cfg,
            "notebook": UPSTREAM + notebook,
            "sha256": sha(path),
            "reference_row_max_abs_error": error,
            "norm_quantiles": torch.quantile(
                values.norm(dim=1), torch.tensor([0.0, 0.5, 1.0])
            ).tolist(),
        }
        print(
            f"{name}: wrote {len(values)} predictions; reference max error {error:.3g}", flush=True
        )
    (args.out / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf8"
    )
    if args.evaluate:
        from fit_eeg_image_likelihood import image_features

        images = image_features(args.base, "test")
        evaluation = {"direct": reference_metrics(eeg.numpy(), images)}
        for name, _, _, _ in RECIPES:
            dest = args.out / name
            path = dest / "h14_prediction.npy"
            predict(dest, dictionary_path=args.base / "dictionary.npz", eeg_path=path)
            evaluate_readout(dest, False, reference=args.base)
            evaluation[name] = reference_metrics(np.load(path), images)
        (args.out / "evaluation_only_image_gallery.json").write_text(
            json.dumps(evaluation, indent=2) + "\n", encoding="utf8"
        )
        print(
            json.dumps(
                {k: {a: b for a, b in v.items() if a != "ranks"} for k, v in evaluation.items()},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
