"""Decode four local visual conditions from EEG and route them into native image attention."""
import argparse
from contextlib import contextmanager
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, render_transaction, require, save_json
from paper_spatial_photo_targets import read_targets


STEPS, BLEND, PENALTY = 2, .65, .5
EEG_ARMS = ("native-paper", "spatial", "uniform", "rotated", "wrong-eeg", "mean-crops")
ORACLE_ARMS = ("true-spatial", "true-uniform", "true-rotated", "true-wrong")


def visual_scale(code):
    return F.normalize(code.float(), dim=-1) * 32.


def compose_crops(global_code, residual):
    return visual_scale(visual_scale(global_code)[:, None] + residual)


def position_weights(queries):
    width = math.isqrt(queries)
    require(width * width == queries and width > 0, "Expected a square attention grid")
    axis = (torch.arange(width, dtype=torch.float32) + .5) / width
    high = ((axis - .35) / .30).clamp(0., 1.)
    y, x = torch.meshgrid(high, high, indexing="ij")
    return torch.stack(((1.-y)*(1.-x), (1.-y)*x, y*(1.-x), y*x), dim=-1).reshape(queries, 4)


def routed_attention(query, base_key, base_value, crop_key, crop_value, position):
    global_value = F.scaled_dot_product_attention(query, base_key, base_value, dropout_p=0.)
    update = torch.zeros_like(global_value)
    for slot, weight in enumerate(position.unbind(-1)):
        local = F.scaled_dot_product_attention(query, crop_key[:, slot], crop_value[:, slot], dropout_p=0.)
        update.add_((local - global_value) * weight[:, None])
    return update


class SpatialImageRouter:
    """Change only the image contribution at the original output projection."""

    def __init__(self, unet):
        self.unet = unet
        self.hooks = []
        self.state = None
        self.layers = 0

    def __enter__(self):
        require(not self.hooks, "Use one spatial router per pipeline")
        self.hooks.append(self.unet.register_forward_pre_hook(self._step))
        self.layers = 0
        for attention in self.unet.modules():
            processor = getattr(attention, "processor", None)
            if hasattr(processor, "to_k_ip") and hasattr(processor, "to_v_ip"):
                self._connect(attention, processor)
                self.layers += 1
        if not self.layers:
            self.__exit__(None, None, None)
            raise ValueError("The generator has no native image attention")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @contextmanager
    def batch(self, tokens, uniform):
        require(self.hooks and self.state is None and tokens.ndim == 4 and tokens.shape[1:3] == (4, 4),
                "Expected one batch of four native token banks")
        self.state = dict(tokens=tokens, uniform=uniform, step=0, grids={}, captures={}, local={})
        try:
            yield
            require(self.state["step"] == 4 and not self.state["captures"],
                    "Spatial routing must accompany one complete four-step draw")
        finally:
            self.state = None

    def _step(self, module, inputs):
        if self.state is not None:
            self.state["step"] += 1

    def _connect(self, attention, processor):
        def capture(name):
            def forward(module, inputs, output):
                state = self.state
                if state is not None and state["step"] <= STEPS:
                    state["captures"][attention, name] = output
            return forward

        def before_output(module, inputs):
            state = self.state
            if state is None or state["step"] > STEPS:
                return
            captures = state["captures"]
            q, k, v = (captures.pop((attention, name)) for name in ("query", "key", "value"))
            batch, positions, width = q.shape
            heads, native_banks = attention.heads, state["tokens"]
            require(len(native_banks) == batch and k.shape == v.shape and k.shape[1] == 4,
                    "Native image attention axes differ")
            q, k, v = (value.reshape(batch, -1, heads, width // heads).transpose(1, 2) for value in (q, k, v))
            if attention.norm_q is not None:
                q = attention.norm_q(q)
            if attention not in state["local"]:
                # Prevent the hooks from recording this explicit projection as native K/V.
                step = state["step"]
                state["step"] = STEPS + 1
                try:
                    lk = processor.to_k_ip(native_banks)
                    lv = processor.to_v_ip(native_banks)
                finally:
                    state["step"] = step
                state["local"][attention] = tuple(
                    bank.reshape(batch, 4, 4, heads, width // heads).transpose(2, 3)
                    for bank in (lk, lv)
                )
            lk, lv = state["local"][attention]
            if positions not in state["grids"]:
                window = torch.full((positions, 4), .25) if state["uniform"] else position_weights(positions)
                state["grids"][positions] = window.to(device=q.device, dtype=q.dtype)
            shift = routed_attention(q, k, v, lk, lv, state["grids"][positions])
            shift = shift.transpose(1, 2).reshape_as(inputs[0])
            return (inputs[0] + processor.scale * BLEND * shift, *inputs[1:])

        self.hooks.extend((
            attention.to_q.register_forward_hook(capture("query")),
            processor.to_k_ip.register_forward_hook(capture("key")),
            processor.to_v_ip.register_forward_hook(capture("value")),
            attention.to_out[0].register_forward_pre_hook(before_output),
        ))


def readout_inputs(eeg, rows, native):
    from paper_paired_encoder import prepare_eeg

    averaged = prepare_eeg(eeg, rows).flatten(1)
    return torch.cat((native.to(averaged.device), averaged), dim=1)


def fit_readout(inputs, delta):
    from paper_token_evidence import ridge_weights

    center, scale = inputs.mean(0), inputs.std(0).clamp_min(1e-5)
    crop_mean = delta.mean(0)
    regression = ridge_weights((inputs - center) / scale, delta - crop_mean, penalty=PENALTY)
    require(torch.isfinite(regression).all(), "Nonfinite EEG-to-crop fit")
    return dict(center=center.cpu(), scale=scale.cpu(), crop_mean=crop_mean.cpu(), regression=regression.cpu())


def predict(inputs, state):
    center, scale, regression = (state[name].to(inputs.device) for name in ("center", "scale", "regression"))
    return (((inputs - center) / scale) @ regression).reshape(len(inputs), 4, 1024).cpu()


@torch.inference_mode()
def fit(subject):
    from common import PREP
    from paper_oof_conditions import registered_rows, study_root
    from paper_paired_encoder import prepare_eeg
    from paper_render import condition, groups, physical_eeg
    from paper_token_evidence import ridge_weights

    root = study_root()
    files = PinnedFiles()
    rows, _, _ = registered_rows(PREP)
    split = files.pin(PREP / "paper_split.json")
    target_path = root / "visual-train.pt"
    target_sha = files.pin(target_path)
    target = read_targets(target_path, rows, "train")
    photo_path = PREP / "F_train.npy"
    photo_sha = files.pin(photo_path)
    require(target["metadata"]["sources"]["split_sha256"] == split
            and target["metadata"]["sources"]["full_image_features_sha256"] == photo_sha,
            "Spatial targets were prepared against another training registration")
    native_eeg, physical = physical_eeg(subject, "train", files)
    native, checkpoint = condition("native-paper", subject, native_eeg[rows], groups("official-r4"),
                                   features_only=True, files=files)
    source = dict(subject=subject, split_sha256=split, training_physical=physical, training_rows=len(rows),
                  target_sha256=target_sha, full_target_sha256=photo_sha, native_checkpoint_sha256=checkpoint,
                  code_sha256=files.pin(Path(__file__)), eeg_bins_sha256=files.pin(Path(inspect.getfile(prepare_eeg))),
                  eeg_native_sha256=files.pin(Path(inspect.getfile(condition))),
                  ridge_sha256=files.pin(Path(inspect.getfile(ridge_weights))), repeats=[0, 1, 2, 3], penalty=PENALTY)
    destination = root / f"spatial-sub-{subject:02d}.pt"
    if destination.is_file():
        old = torch.load(destination, map_location="cpu", weights_only=True)
        require(old["source"] == source, "Existing EEG spatial fit used other sources")
        return
    full = torch.from_numpy(np.load(photo_path, mmap_mode="r", allow_pickle=False)[rows].copy())
    delta = visual_scale(target["crops"]) - visual_scale(full)[:, None]
    state = fit_readout(readout_inputs(native_eeg, rows, native), delta.cuda())
    source_rms = float((delta - state["crop_mean"]).square().mean().sqrt())
    files.verify()
    atomic_file(destination, lambda stream: torch.save(dict(source=source, state=state, target_rms=source_rms), stream))
    save_json(destination.with_suffix(".json"), dict(source=source, target_rms=source_rms))
    print("EEG crop fit", subject, len(rows), source_rms, flush=True)


def probe_crops(predicted, calibration, target, wrong):
    """Compare EEG crop positions and whole exemplars before running the generator."""
    relative = lambda code: code - code.mean(1, keepdim=True)
    p, t = relative(predicted - calibration), relative(target - calibration)
    compare = lambda a, b: (a - b).square().mean((1, 2))
    correct = compare(p, t)
    rotated = compare(p.flip(1), t)
    mismatched = compare(p[wrong], t)
    wmean = lambda control: float(((correct < control).float() + .5 * (correct == control)).mean())
    return dict(relative_mse=float(correct.mean()), relative_mean_mse=float(t.square().mean()),
                beats_rotated=wmean(rotated), beats_example_exchange=wmean(mismatched))


@torch.inference_mode()
def render(subject, phase, arms):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import condition, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_token_evidence import project_tokens, score_outputs

    root, files = study_root(), PinnedFiles()
    registration = PREP / "paper_split.json"
    split_sha = files.pin(registration)
    rows = json.loads(registration.read_bytes())["official_rows"]
    require(rows == list(range(200)), "Expected the registered 200 official queries")
    eeg, physical = physical_eeg(subject, "test", files)
    repeats = groups("official-r4")
    own = np.arange(len(rows))
    wrong = np.roll(own, len(rows) // 2)
    assets = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    pipe, adapter, projection, prompts, vae = pipeline
    native, checkpoint = condition("native-paper", subject, eeg[rows], repeats, features_only=True, files=files)
    base_tokens = project_tokens(native, adapter, projection).cpu().half()
    diagnostic = phase == "oracle"
    if diagnostic:
        visual_path = root / "visual-oracle.pt"
        source_sha = files.pin(visual_path)
        target = read_targets(visual_path, np.asarray(rows), "oracle")
        require(target["metadata"]["sources"]["split_sha256"] == split_sha
                and target["metadata"]["sources"]["row_files"]["eeg_test_ids.json"] == physical["eeg_ids_sha256"],
                "Oracle photos are not registered to this EEG acquisition")
        visual = visual_scale(target["crops"])
        crops = {"true-spatial": visual, "true-uniform": visual, "true-rotated": visual.flip(1),
                 "true-wrong": visual[wrong]}
        readout_rms = None
    else:
        fit_path = root / f"spatial-sub-{subject:02d}.pt"
        source_sha = files.pin(fit_path)
        trained = torch.load(fit_path, map_location="cpu", weights_only=True)
        proof = trained["source"]
        require((proof["subject"], proof["split_sha256"], proof["native_checkpoint_sha256"]) ==
                (subject, split_sha, checkpoint), "EEG crop fit is incompatible with this render")
        delta = predict(readout_inputs(eeg, rows, native), trained["state"])
        require(torch.isfinite(delta).all(), "Nonfinite EEG crop prediction")
        mean = trained["state"]["crop_mean"]
        local = compose_crops(native, mean + delta)
        crops = {"spatial": local, "uniform": local, "rotated": local.flip(1),
                 "wrong-eeg": compose_crops(native, mean + delta[wrong]), "mean-crops": compose_crops(native, mean)}
        readout_rms = float(delta.square().mean().sqrt())
        # Write predictions without touching reference features. The optional probe reads those after inference.
        probe_path = root / f"prediction-sub-{subject:02d}.pt"
        payload = dict(rows=torch.tensor(rows), fit_sha256=source_sha, delta=delta, mean=mean, wrong=torch.from_numpy(wrong))
        atomic_file(probe_path, lambda stream: torch.save(payload, stream))
    seeds = [query_seed("official", i) for i in own]
    folders = []
    with SpatialImageRouter(pipe.unet) as router:
        shared = dict(render_cache_version=2, subject=subject, n=len(rows), row=rows, repeats=repeats, seeds=seeds,
                      budget="official-r4", prompt="", draws=1, unet_calls_per_draw=4, split_sha256=split_sha,
                      physical_eeg=physical, native_checkpoint_sha256=checkpoint, generator_files_sha256=assets,
                      code_sha256=files.pin(Path(__file__)), renderer_sha256=files.pin(Path(inspect.getfile(draw_images))),
                      diagnostic_only=diagnostic, crop_condition_source_sha256=source_sha, eeg_crop_rms=readout_rms)
        banks = {}
        for arm in arms:
            has_crops = arm in crops
            policy = arm.removeprefix("true-")
            if has_crops:
                view = crops[arm]
                identity = id(view)
                if identity not in banks:
                    banks[identity] = project_tokens(view.reshape(-1, 1024), adapter, projection).cpu().half().reshape(
                        len(rows), 4, *base_tokens.shape[1:]
                    )
                tokens = banks[identity]
            folder = root / ("oracle" if diagnostic else "images") / f"sub-{subject:02d}" / arm
            request = dict(shared, arm=arm, spatial_steps=STEPS if has_crops else 0,
                           spatial_blend=BLEND if has_crops else 0., ip_layers=router.layers,
                           crop_layout="uniform" if policy == "uniform" else "spatial" if has_crops else None,
                           crop_slots=[3, 2, 1, 0] if policy == "rotated" else [0, 1, 2, 3] if has_crops else None,
                           crop_donors=wrong.tolist() if policy in ("wrong", "wrong-eeg") else own.tolist() if has_crops else None)

            def generate(stage, starts, complete):
                if not has_crops:
                    draw_images(stage, pipe, vae, prompts, base_tokens, seeds, batch_starts=starts, batch_complete=complete)
                    return
                for start in starts:
                    with router.batch(tokens[start:start + 8].cuda(), uniform=policy == "uniform"):
                        draw_images(stage, pipe, vae, prompts, base_tokens, seeds,
                                    batch_starts=[start], batch_complete=complete)

            render_transaction(folder, request, generate, files.verify)
            print("spatial photo render", subject, arm, len(rows), flush=True)
            folders.append(folder)
    del pipeline, pipe, adapter, projection, vae, router
    torch.cuda.empty_cache()
    score_outputs(folders, "official")


def probe(subject):
    from common import PREP
    from paper_oof_conditions import study_root
    from paper_visual_ceiling import official_features

    root, files = study_root(), PinnedFiles()
    full, rows, proof = official_features(PREP, files)
    target_path = root / "visual-oracle.pt"
    prediction_path = root / f"prediction-sub-{subject:02d}.pt"
    proof.update(target=files.pin(target_path), prediction=files.pin(prediction_path), code=files.pin(Path(__file__)))
    prediction = torch.load(prediction_path, map_location="cpu", weights_only=True)
    target = read_targets(target_path, np.asarray(rows), "oracle")
    require(prediction["rows"].tolist() == rows, "Predicted and visual crop rows disagree")
    delta = visual_scale(target["crops"]) - visual_scale(full)[:, None]
    values = probe_crops(prediction["delta"] + prediction["mean"], prediction["mean"], delta, prediction["wrong"])
    files.verify()
    save_json(root / f"probe-sub-{subject:02d}.json", dict(subject=subject, n=len(rows), scores=values, source=proof))
    print("crop readout probe", subject, values, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("fit", "eeg", "oracle", "probe"):
        command = commands.add_parser(action)
        command.add_argument("subject", type=int, choices=range(1, 11))
        if action in ("eeg", "oracle"):
            allowed = ORACLE_ARMS if action == "oracle" else EEG_ARMS
            command.add_argument("--arms", nargs="+", choices=allowed, default=allowed)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.subject)
    elif args.action == "probe":
        probe(args.subject)
    else:
        render(args.subject, args.action, args.arms)
