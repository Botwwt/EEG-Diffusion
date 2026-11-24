"""Replay the selected M1a + D2_Compete4 reconstruction model on R4 fold 0."""

import argparse
import io
import json
import os
import sys
from importlib.machinery import PathFinder
from pathlib import Path
from tempfile import TemporaryDirectory


class ReleaseScriptFinder:
    """Keep the archived runners' sys.path edits from selecting stale project scripts."""

    def __init__(self):
        self.directory = Path(__file__).resolve().parent
        self.names = {script.stem for script in self.directory.glob("*.py")}

    def find_spec(self, fullname, path=None, target=None):
        if path is None and fullname in self.names:
            return PathFinder.find_spec(fullname, [str(self.directory)], target)
        return None


def selected_checkpoint(d2_path: Path, backbone_path: Path) -> io.BytesIO:
    """Pass both states to the archived evaluator without following a saved server path."""
    import torch

    d2 = torch.load(d2_path, map_location="cpu", weights_only=True)
    if d2["mode"] != "d2_compete4_cond" or d2["steps"] != 120:
        raise ValueError(f"Expected the 120-step D2_Compete4 conditioning layer in {d2_path}")
    backbone = torch.load(backbone_path, map_location="cpu", weights_only=True)
    if backbone["arm"] != "arm3b_rep_rawswap_condlyr" or backbone["steps"] != 600:
        raise ValueError(f"Expected the 600-step SwapDiff Arm3b backbone in {backbone_path}")
    bundle = io.BytesIO()
    torch.save(
        {"state_dict": backbone["state_dict"], "cond_layer": d2["cond_layer"], "cone": d2["cone"]},
        bundle,
    )
    bundle.seek(0)
    return bundle


def evaluate(subjects, checkpoint):
    import numpy as np
    import torch
    from common import PREP, PRIOR, TURBO
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from exp0_intra10_swap_gate import get_concept_splits
    from exp5_z0_2x2_diag import extract_eval_feats
    from exp8_trial_precision_swapdiff import fit_tr14_ridge_z0_prec
    from exp9b_m2_and_m1m2_confirm import run_s375_eval
    from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel
    from metrics import _nets

    device = "cuda:0"
    dtype = torch.float16
    concept, _, train_indices, _, _, concept_ids, photo_indices = get_concept_splits()
    if len(concept_ids) != 50 or not np.array_equal(
        concept[photo_indices], np.repeat(concept_ids, 10)
    ):
        raise ValueError("Expected 50 fold-0 concepts with ten prepared photos each")
    starts = {
        sub: fit_tr14_ridge_z0_prec(sub, train_indices, photo_indices, dev=device, p=0.5)
        for sub in subjects
    }
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dtype, variant="fp16")
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to(device, dtype)
    pipe.unet.eval().requires_grad_(False)
    pipe.scheduler.set_timesteps(4, device=device)
    with torch.inference_mode():
        prompt, _, pooled, _ = pipe.encode_prompt(
            "", device=device, do_classifier_free_guidance=False, num_images_per_prompt=1
        )
        time_ids = pipe._get_add_time_ids(
            (512, 512),
            (0, 0),
            (512, 512),
            dtype=dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(device)
    del pipe.text_encoder, pipe.text_encoder_2, pipe.vae
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter
    adapter = adapter.to(device, dtype).eval().requires_grad_(False)
    projection = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj")
    projection = projection.to(device, dtype).eval().requires_grad_(False)
    vae = (
        AutoencoderKL.from_pretrained(
            str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    vae.enable_slicing()
    latents = np.load(PREP / "lat512_train.npy", mmap_mode="r")[photo_indices]
    with torch.inference_mode():
        decoded = []
        for batch in torch.from_numpy(latents).split(20):
            reference = vae.decode(
                batch.to(device, torch.float32) / vae.config.scaling_factor
            ).sample
            decoded.append((reference / 2.0 + 0.5).clamp(0.0, 1.0).cpu())
        references = torch.cat(decoded)
        del decoded
        nets = _nets()
        features = extract_eval_feats(references, nets, dev=device)
    result = run_s375_eval(
        "M1a_D2_Compete4",
        checkpoint,
        True,
        subjects,
        photo_indices,
        starts,
        pipe.unet,
        pipe.scheduler,
        vae,
        adapter,
        projection,
        prompt,
        pooled,
        time_ids,
        references,
        features,
        nets,
        dev=device,
    )
    return {
        "model": "M1a_D2_Compete4",
        "regime": "s375",
        "repeats": 4,
        "nfe": 4,
        "candidates": 1,
        "subjects": subjects,
        "fold": 0,
        "concept_ids": concept_ids.tolist(),
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        choices=range(1, 11),
        default=list(range(1, 11)),
        metavar="ID",
        help="Participant IDs 1 through 10 (default: all ten).",
    )
    parser.add_argument(
        "--d2-checkpoint", type=Path, help="Path to the trained D2 conditioning layer."
    )
    parser.add_argument(
        "--backbone-checkpoint", type=Path, help="Path to the trained Arm3b backbone."
    )
    parser.add_argument(
        "--output", type=Path, help="Result JSON (default: under RAD_ROOT/runs/swap_diff)."
    )
    args = parser.parse_args()
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("Each participant may be selected only once.")
    if not os.environ.get("RAD_ROOT"):
        parser.error("Set RAD_ROOT to the prepared Linux GPU workspace.")
    runs = Path(os.environ["RAD_ROOT"]) / "runs" / "swap_diff"
    d2 = args.d2_checkpoint or runs / "draft_intra4/d2_compete4_cond_st120_s2025.pt"
    backbone = args.backbone_checkpoint or runs / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    for path in (d2, backbone):
        if not path.is_file():
            parser.error(f"Missing trained checkpoint: {path}")
    group = "-".join(f"{s:02d}" for s in sorted(args.subjects))
    output = args.output or runs / f"m1a_d2_s375_sub-{group}.json"
    if output.resolve() in (d2.resolve(), backbone.resolve()):
        parser.error("Choose a result path distinct from the checkpoints.")

    with selected_checkpoint(d2, backbone) as checkpoint:
        finder = ReleaseScriptFinder()
        sys.meta_path.insert(0, finder)
        try:
            result = evaluate(args.subjects, checkpoint)
        finally:
            sys.meta_path.remove(finder)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp_dir:
        temp = Path(temp_dir) / "result.json"
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
