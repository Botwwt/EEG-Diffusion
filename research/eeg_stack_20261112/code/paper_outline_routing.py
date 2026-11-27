"""Route only the token-relative image attention with a foreground estimate from EEG."""
import argparse
from contextlib import contextmanager, nullcontext
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, render_transaction, require, save_json

STRENGTH = .9
ROUTED_STEPS = 2
MASK_SIZE = 20
ARMS = ("mean-outline", "eeg-outline", "wrong-outline", "true-outline")
AMPLIFIED_ARMS = ("amplified-outline", "amplified-wrong")
MASK_RECEIPTS = {
    "train": "844a13b7a65e0698cfbc5a2e414fb833e4cc8ad3737b2ab5dbf8bf91634edc18",
    "dev": "70cfa2da4483c05480236f99f978585b4d396cf78f0cceae276e1aba2a52a5db",
}


def spatial_delta(field, queries, strength):
    side = math.isqrt(queries)
    require(side * side == queries, "The 512px denoiser needs a square spatial attention lattice")
    # Subtract a common value before resizing so constant fields stay exactly zero.
    centered = field.float() - field[:, :, :1, :1]
    grid = F.interpolate(centered, size=(side, side), mode="bilinear", align_corners=False)
    rows = grid.flatten(2).transpose(1, 2)
    return strength * (rows - rows.mean(1, keepdim=True))


class OutlineRouter:
    """Reuse the native text and image attention, adding one local relative-value term."""

    def __init__(self, unet):
        self.unet = unet
        self.handles = []
        self.field = None
        self.active = False
        self.calls = 0
        self.maps = {}
        self.captures = []
        self.layers = 0

    def __enter__(self):
        require(not self.handles, "Use a single router on each pipeline")
        self.layers = 0
        self.handles.append(self.unet.register_forward_pre_hook(self._begin_step))
        for attention in self.unet.modules():
            processor = getattr(attention, "processor", None)
            if hasattr(processor, "to_k_ip") and hasattr(processor, "to_v_ip"):
                self._attach(attention, processor)
                self.layers += 1
        if not self.layers:
            self.__exit__(None, None, None)
            raise ValueError("The native pipeline has no image-token attention")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captures.clear()

    @contextmanager
    def batch(self, field):
        require(self.handles and self.field is None, "Each outline must address a single render batch")
        require(field.ndim == 4 and field.shape[1] == 1, "Expected a batch of one-channel outlines")
        self.field = field
        self.calls = 0
        try:
            yield
            require(self.calls == 4, "The outline must cover one four-step draw")
        finally:
            self.field = None
            self.active = False
            self.maps.clear()
            for capture in self.captures:
                capture.clear()

    def _begin_step(self, module, args):
        if self.field is not None:
            self.calls += 1
            self.active = self.calls <= ROUTED_STEPS

    def _attach(self, attention, processor):
        saved = {}
        self.captures.append(saved)

        def capture(name):
            def forward(module, inputs, output):
                if self.active:
                    saved[name] = output
            return forward

        def before_output(module, inputs):
            if not self.active:
                return None
            require(len(saved) == 3, "The native image attention did not expose its query, key and value")
            query, key, value = (saved[name] for name in ("query", "key", "value"))
            saved.clear()
            batch, positions, width = query.shape
            require(batch == len(self.field) and value.shape[-1] == width,
                    "Outline batch or native image value axes changed")
            heads = attention.heads

            def separate_heads(x):
                return x.reshape(batch, -1, heads, width // heads).transpose(1, 2)

            q = separate_heads(query)
            if attention.norm_q is not None:
                q = attention.norm_q(q)
            contribution = F.scaled_dot_product_attention(
                q, separate_heads(key), separate_heads(value), dropout_p=0., is_causal=False
            )
            contribution = contribution.transpose(1, 2).reshape_as(inputs[0])
            local_value = contribution - value.mean(dim=1, keepdim=True)
            if positions not in self.maps:
                self.maps[positions] = spatial_delta(self.field, positions, STRENGTH).to(query.dtype)
            shift = processor.scale * self.maps[positions] * local_value
            return (inputs[0] + shift, *inputs[1:])

        hooks = (
            attention.to_q.register_forward_hook(capture("query")),
            processor.to_k_ip.register_forward_hook(capture("key")),
            processor.to_v_ip.register_forward_hook(capture("value")),
            attention.to_out[0].register_forward_pre_hook(before_output),
        )
        self.handles.extend(hooks)


def outline_latents(field, vae):
    encoded = []
    for block in field.split(4):
        mask = F.interpolate(block.float(), size=(512, 512), mode="bilinear", align_corners=False)
        grayscale = (1.2 * mask - .6).expand(-1, 3, -1, -1).contiguous()
        z = vae.encode(grayscale).latent_dist.mode().float() * vae.config.scaling_factor
        encoded.append((z - z.mean(dim=(2, 3), keepdim=True)).cpu())
    return torch.cat(encoded)


def load_masks(path, rows, phase, files):
    require(files.pin(path) == MASK_RECEIPTS[phase], "Foreground mask archive differs from the audited source")
    with np.load(path, allow_pickle=False) as panel:
        mask = panel["masks"]
        require(np.array_equal(panel["rows"], rows) and panel["valid"].all()
                and mask.shape == (len(rows), 160, 160) and mask.dtype == np.uint8,
                "The foreground masks must follow the registered photographs")
    blocks = []
    for source in np.array_split(mask, 16):
        x = torch.from_numpy(source).cuda().float().unsqueeze(1) / 255.
        blocks.append(F.avg_pool2d(x, kernel_size=160 // MASK_SIZE))
    return torch.cat(blocks)


@torch.inference_mode()
def fit(subject, masks_path):
    from common import PREP
    from paper_oof_conditions import registered_rows, study_root
    from paper_paired_encoder import prepare_eeg
    from paper_render import physical_eeg
    from paper_token_evidence import ridge_weights

    files = PinnedFiles()
    rows, _, _ = registered_rows(PREP)
    masks = load_masks(masks_path, rows, "train", files)
    eeg, source = physical_eeg(subject, "train", files)
    proof = dict(physical_eeg=source, split=files.pin(PREP / "paper_split.json"), masks=files.pin(masks_path),
                 code=files.pin(Path(__file__)), eeg_reader=files.pin(Path(inspect.getfile(prepare_eeg))),
                 regression=files.pin(Path(inspect.getfile(ridge_weights))))
    folder = study_root()
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"outline-sub-{subject:02d}.pt"
    if output.exists():
        old = torch.load(output, map_location="cpu", weights_only=True)
        require(old["proof"] == proof, "Existing outline fit used other input bytes")
        return
    features = prepare_eeg(eeg, rows).flatten(1)
    center, scale = features.mean(0), features.std(0).clamp_min(1e-5)
    mask_center = masks.mean(0)
    matrix = ridge_weights((features - center) / scale, masks - mask_center, penalty=1.)
    require(torch.isfinite(matrix).all(), "Nonfinite EEG outline fit")
    state = dict(center=center.cpu(), scale=scale.cpu(), mask_center=mask_center.cpu(), matrix=matrix.cpu())
    resume = dict(subject=subject, proof=proof, training_rows=len(rows), penalty=1., repeats=[[0, 1, 2, 3]])
    files.verify()
    atomic_file(output, lambda stream: torch.save(dict(resume, state=state), stream))
    save_json(output.with_suffix(".json"), resume)
    print("outline fit", subject, len(rows), flush=True)


@torch.inference_mode()
def render(subject, panel, arms, true_masks, calibration, use_layout):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_paired_encoder import prepare_eeg
    from paper_render import condition, donor_rows, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_score import score
    from paper_token_evidence import development

    root = study_root()
    files = PinnedFiles()
    split = files.pin(PREP / "paper_split.json")
    registration = json.loads((PREP / "paper_split.json").read_bytes())
    dev = None
    if panel == "dev":
        plan, rows = development(PREP, files)
        require(subject in plan["subjects"], "Participant does not belong to the development panel")
        dev = files.pin(PREP / "paper_development.json")
    else:
        rows = registration[f"{panel}_rows"]
        require(len(rows) == len(set(rows)) == (200 if panel == "official" else 500),
                "Unregistered shape panel")
        require(not np.intersect1d(rows, registration["train_rows"]).size or panel == "official",
                "Fresh shape queries entered the fitting set")
    require("true-outline" not in arms or (panel == "dev" and true_masks is not None),
            "Reference outlines are a development-only diagnostic")
    eeg, physical = physical_eeg(subject, "test" if panel == "official" else "train", files)
    checkpoint = root / f"outline-sub-{subject:02d}.pt"
    regression_sha = files.pin(checkpoint)
    fit = torch.load(checkpoint, map_location="cuda", weights_only=True)
    require(fit["subject"] == subject and fit["proof"]["split"] == split, "Outline fit registration changed")
    if panel != "official":
        require(fit["proof"]["physical_eeg"] == physical, "Train-side query bytes differ from the outline fit")
    state = fit["state"]
    observed = prepare_eeg(eeg, rows).flatten(1)
    n = len(rows)
    guess = ((observed - state["center"]) / state["scale"]) @ state["matrix"]
    average = state["mask_center"].expand(n, -1, -1, -1)
    outline = (guess.reshape_as(average) + average).clamp(0, 1)
    fields = {"mean-outline": average, "eeg-outline": outline}
    spread_sha, multiplier = None, None
    if any(arm in AMPLIFIED_ARMS for arm in arms):
        require(calibration is not None, "An amplified outline needs training-fold spread calibration")
        spread_sha = files.pin(calibration)
        spread = json.loads(calibration.read_bytes())
        source = spread["out_of_fold_gain"][str(subject)]
        require(spread["split_sha256"] == split and spread["masks_train_sha256"] == fit["proof"]["masks"]
                and source["physical_sha256"] == fit["proof"]["physical_eeg"]["array_sha256"],
                "Outline spread was fitted on different training inputs")
        multiplier = source["gain"]
        require(math.isfinite(multiplier) and multiplier > 0, "Invalid training-fold outline spread")
        fields["amplified-outline"] = (multiplier * guess.reshape_as(average) + average).clamp(0, 1)
    donors = np.arange(n)
    if any(arm in ("wrong-outline", "amplified-wrong") for arm in arms):
        if panel == "official":
            donors = np.roll(donors, n // 2)
        else:
            labels_path = PREP / "train_concept.npy"
            files.pin(labels_path)
            labels = np.load(labels_path, allow_pickle=False)[rows]
            donors = donor_rows("wrong", panel, labels)
            require(np.all(donors != np.arange(n)) and np.all(labels == labels[donors]),
                    "The outline exchange must stay within object and change photos")
        fields["wrong-outline"] = outline[donors]
        if "amplified-wrong" in arms:
            fields["amplified-wrong"] = fields["amplified-outline"][donors]
    true_digest = None
    if "true-outline" in arms:
        fields["true-outline"] = load_masks(true_masks, rows, "dev", files)
        true_digest = files.pin(true_masks)
    assets = generator_inputs(files)
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    repeats = groups(panel + "-r4")
    tokens, native_sha = condition("native-paper", subject, eeg[rows], repeats, adapter, projection, files=files)
    seeds = [query_seed(panel, i) for i in range(n)]
    output = []
    with (nullcontext() if use_layout else OutlineRouter(pipe.unet)) as router:
        shared = dict(render_cache_version=2, subject=subject, n=n, row=rows, repeats=repeats, seeds=seeds,
                      budget=panel + "-r4", prompt="", draws=1, unet_calls_per_draw=4, split_sha256=split,
                      physical_eeg=physical, development_split_sha256=dev,
                      native_checkpoint_sha256=native_sha, outline_checkpoint_sha256=regression_sha,
                      generator_files_sha256=assets, routed_ip_layers=router.layers if router else 0,
                      routed_steps=ROUTED_STEPS if router else 0,
                      spatial_strength=STRENGTH if router else 0.,
                      layout_mode="grayscale-outline" if use_layout else "noise",
                      code_sha256=files.pin(Path(__file__)),
                      renderer_sha256=files.pin(Path(inspect.getfile(draw_images))))
        for arm in arms:
            shape = fields[arm]
            location = root / ("layout" if use_layout else "images") / panel / f"sub-{subject:02d}" / arm
            request = dict(shared, arm=arm, mask_grid=[MASK_SIZE, MASK_SIZE],
                           semantic_eeg_donor=list(range(n)),
                           outline_eeg_donor=(donors.tolist() if arm in ("wrong-outline", "amplified-wrong")
                                              else list(range(n)) if arm in ("eeg-outline", "amplified-outline") else None),
                           reference_mask_sha256=true_digest if arm == "true-outline" else None,
                           diagnostic_only=arm == "true-outline")
            if arm in AMPLIFIED_ARMS:
                request.update(outline_rms_calibration_sha256=spread_sha, outline_multiplier=multiplier)

            def generate(folder, starts, complete):
                if use_layout:
                    start_latents = outline_latents(shape, vae)
                    draw_images(folder, pipe, vae, prompts, tokens, seeds, z0s=start_latents,
                                batch_starts=starts, batch_complete=complete)
                else:
                    for start in starts:
                        with router.batch(shape[start:start + 8]):
                            draw_images(folder, pipe, vae, prompts, tokens, seeds,
                                        batch_starts=[start], batch_complete=complete)

            render_transaction(location, request, generate, files.verify)
            output.append(location)
            print("outline render", subject, panel, arm, n, flush=True)
    del router, pipe, adapter, projection, vae
    torch.cuda.empty_cache()
    for location in output:
        if not (location / "scores.json").is_file():
            score(location, panel, Path(os.environ["ARCHIVE"]), location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    training = commands.add_parser("fit")
    training.add_argument("subject", type=int, choices=range(1, 11))
    training.add_argument("--masks", type=Path, required=True)
    generation = commands.add_parser("render")
    generation.add_argument("subject", type=int, choices=range(1, 11))
    generation.add_argument("--panel", choices=("dev", "fresh", "official"), default="dev")
    generation.add_argument("--arms", nargs="+", choices=(*ARMS, *AMPLIFIED_ARMS), default=ARMS[:-1])
    generation.add_argument("--true-masks", type=Path)
    generation.add_argument("--spread-calibration", type=Path)
    generation.add_argument("--layout", action="store_true")
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.subject, args.masks)
    else:
        render(args.subject, args.panel, args.arms, args.true_masks, args.spread_calibration, args.layout)
