"""Measure the frozen generator from known visual conditions in a diagnostic-only directory."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from paper_oof_conditions import study_root
from paper_render_cache import PinnedFiles, render_transaction, require, save_json, sha


def official_features(prep, pins):
    target = prep / "F_test.npy"
    receipt = prep / "paper_test_F_receipt.json"
    index = prep / "test_ids.json"
    split = prep / "paper_split.json"
    proof = {p.name: pins.pin(p) for p in (target, receipt, index, split)}
    registration = json.loads(receipt.read_bytes())
    rows = json.loads(split.read_bytes())["official_rows"]
    ids = json.loads(index.read_bytes())["ids"]
    require(rows == list(range(200)) and len(set(ids)) == len(ids) == len(rows)
            and ids == registration["ordered_ids"] and proof[target.name] == registration["F_test_sha256"],
            "The visual diagnostic inputs disagree about the official photograph order")
    features = np.load(target, allow_pickle=False)
    require(features.shape == (len(rows), 1024) and np.isfinite(features).all(), "Invalid registered photograph features")
    return torch.from_numpy(features).float(), rows, proof


@torch.inference_mode()
def run():
    from common import PREP, TURBO
    from diffusers import AutoencoderKL
    from paper_render import draw_images, generator_inputs, query_seed
    from paper_score import score
    from paper_training import generator

    pins = PinnedFiles()
    features, rows, proof = official_features(PREP, pins)
    generators = generator_inputs(pins)
    seeds = [query_seed("official", row) for row in rows]
    code = pins.pin(Path(__file__))
    pipe, adapter, project, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16",
                                        torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    tokens = torch.cat([project(adapter(value.cuda().half())).cpu() for value in features.split(20)])
    root = study_root() / "visual-ceiling"
    controls = {"original-visual": rows, "different-object-visual": np.roll(rows, len(rows) // 2).tolist()}
    for arm, donors in controls.items():
        out = root / arm
        request = dict(render_cache_version=2, arm=arm, diagnostic_only=True, n=len(rows), row=rows,
                       visual_feature_donor=donors, prompt="", draws=1, unet_calls_per_draw=4, seeds=seeds,
                       source_sha256=proof, generator_files_sha256=generators, code_sha256=code)

        def generate(stage, starts, complete):
            draw_images(stage, pipe, vae, (prompt, pooled, dims), tokens[donors], seeds,
                        batch_starts=starts, batch_complete=complete)

        render_transaction(out, request, generate, pins.verify)
    del pipe, adapter, project, vae
    torch.cuda.empty_cache()
    archive = Path(os.environ["ARCHIVE"])
    results = {}
    for arm in controls:
        out = root / arm
        if not (out / "scores.json").exists():
            score(out, "official", archive, out)
        result = json.loads((out / "scores.json").read_bytes())
        require(result["render_sha256"] == sha(out / "render.json")
                and result["observations_sha256"] == sha(out / "observations.npz"), "Diagnostic evaluation receipt differs")
        results[arm] = result
    pins.verify()
    save_json(root / "summary.json", dict(diagnostic_only=True, rows=len(rows), scores=results))


if __name__ == "__main__":
    run()
