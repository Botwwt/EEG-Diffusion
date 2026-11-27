"""Render matched clean-prior controls with the frozen D2 acquisition and layout."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from paper_render_cache import PinnedFiles, render_transaction, require, save_json
from paper_token_evidence import project_tokens, score_outputs, token_parts


CONDITIONS = ("direct", "gaussian", "flow", "flow-wrong")
EDIT_CONDITIONS = ("direct", "edit", "edit-wrong", "edit-posterior")


def with_learned_residual(raw, repeats, subject, variants, adapter, projector, pins):
    from paper_d2_object_swap import d2_models
    from paper_photo_top1 import ordinary_readout
    from paper_render import per_sample_cone

    encoder, residual, cone, fit = d2_models(pins)
    hidden = []
    features = []
    for start in range(0, len(raw), 20):
        trials = torch.from_numpy(np.array(raw[start:start + 20], copy=True)).cuda().float()
        f, h = ordinary_readout(encoder, trials, subject, repeats)
        features.append(f)
        hidden.append(h)
    h = torch.cat(hidden)
    original = per_sample_cone(torch.cat(features), cone["R"], cone["alpha"], cone["beta"], cone["u_tr"])
    require(torch.allclose(original.cpu(), variants["direct"], rtol=.002, atol=.001),
            "The frozen encoder did not reproduce the D2 instance target")
    prepared = {}
    for key, value in dict(variants, direct=original).items():
        tokens = project_tokens(value, adapter, projector).half()
        prepared[key] = residual(h, tokens).cpu()
    return prepared, fit


def with_current_mean(native, changed):
    mean, _ = token_parts(native.float())
    _, relative = token_parts(changed.float())
    return (mean + relative).to(native.dtype)


@torch.inference_mode()
def run(subject, conditions, receipt_path, output, keep_residual=False, edit=False, values_only=False):
    from common import PREP, RUNS
    from paper_identity_render import load_generator
    from paper_photo_top1 import DEFAULT_LAYOUT, render_inputs
    from paper_render import condition, draw_images, generator_inputs, layout, query_seed, runtime, tensor_fingerprint
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == "shared", "Expected the registered frozen shared D2 condition")
    pins = PinnedFiles()
    edit = edit or values_only
    output.mkdir(parents=True, exist_ok=True)
    inputs = pins.pin(conditions)
    receipt = json.loads(receipt_path.read_bytes())
    source_receipt = pins.pin(receipt_path)
    require(inputs == receipt["source_conditions_sha256"] and receipt["queries"] == 2000,
            "Unverified D2 source optimization")
    if edit:
        require(receipt["aux_updates_only"] is True and receipt["strength"] == .25
                and set(receipt["variants"]) == set(EDIT_CONDITIONS[1:]), "Unregistered semantic update")
    raw, rows, train, budget, repeats, _, split_sha, dev_sha, eeg = render_inputs(subject, "dev", pins)
    require((split_sha, dev_sha) == (receipt["proof"]["split"], receipt["proof"]["development"]),
            "The optimized queries use a different stimulus registration")
    labels_path = PREP / "train_concept.npy"
    require(pins.pin(labels_path) == receipt["proof"]["labels"], "Concept registration changed")
    labels = np.load(labels_path, allow_pickle=False)[rows]
    with np.load(conditions, allow_pickle=False) as archive:
        require(np.array_equal(rows, archive["rows"]), "Optimized source rows changed")
        donors = archive["donors"]
        require(len(np.unique(donors)) == len(rows) and np.all(labels != labels[donors]),
                "Incorrect wrong-condition donor rows")
        keys = EDIT_CONDITIONS if edit else CONDITIONS
        variants = {key: torch.from_numpy(archive[key][subject - 1]) for key in keys}
        if not edit:
            require(torch.equal(variants["flow-wrong"], variants["flow"][donors]),
                    "Wrong EEG should only reassign the optimized condition")
    for value in variants.values():
        require(value.shape == (len(rows), 1024) and bool(torch.isfinite(value).all()),
                "Invalid fused feature shape")
    generator_hashes = generator_inputs(pins)
    pipe, adapter, projector, prompts, vae = load_generator(torch.float16, False)
    if keep_residual or edit:
        prepared, fit = with_learned_residual(raw, repeats, subject, variants, adapter, projector, pins)
        checkpoint = fit["condition_checkpoint"]["sha256"]
        if edit:
            d2, moved, wrong, posterior = (prepared[k] for k in EDIT_CONDITIONS)
            arms = {"d2": d2, "semantic-edit": moved, "semantic-edit-early": (moved, moved, d2, d2),
                    "semantic-edit-early-wrong": (wrong, wrong, d2, d2),
                    "semantic-edit-posterior": posterior}
        else:
            d2, gf, fl, wrong = (prepared[k] for k in CONDITIONS)
            arms = {"d2": d2, "gaussian-resid": gf, "flow-resid": fl,
                    "flow-resid-late": (d2, d2, fl, fl), "flow-resid-late-wrong": (d2, d2, wrong, wrong)}
    else:
        args = ("d2", subject, raw, repeats)
        reference, checkpoint, fit = condition(*args, features_only=True, files=pins, return_source=True)
        require(torch.allclose(reference, variants["direct"], rtol=.002, atol=.001),
                "Current D2 fused output differs from the frozen instance scorer")
        tokens, checked = condition(*args, adapter, projector, files=pins)
        require(checked == checkpoint, "D2 checkpoint changed across controls")
        arms = {"d2": tokens.cpu(), **{key: project_tokens(value, adapter, projector).half().cpu()
                                      for key, value in variants.items()}}
    if values_only:
        from paper_eeg_attention import image_layers
        from paper_joint_condition_diagnostic import ReplacedImageProjection, RoutedUNet

        moved = with_current_mean(d2, moved)
        wrong = with_current_mean(d2, wrong)
        steps = pipe.scheduler.timesteps.tolist()
        require(steps == [999, 749, 499, 249], "Unexpected four-step generator")
        value_routes = {"semantic-values": (moved, steps), "semantic-values-early": (moved, steps[:2]),
                        "semantic-values-wrong": (wrong, steps)}
        arms = {"d2": d2, "semantic-values": moved, "semantic-values-early": (moved, moved, d2, d2),
                "semantic-values-wrong": wrong}
        value_projections = []
        for processor, attr in image_layers(pipe.unet).values():
            if attr == "to_v_ip":
                projection = ReplacedImageProjection(getattr(processor, attr))
                setattr(processor, attr, projection)
                value_projections.append(projection)
        require(value_projections, "The frozen image attention does not expose its Value projections")
    image_path = PREP / "F_train.npy"
    require(pins.pin(image_path) == receipt["proof"]["truth"], "Native image token reference changed")
    visual = np.load(image_path, mmap_mode="r", allow_pickle=False)
    train_tokens = project_tokens(torch.from_numpy(visual[train].copy()), adapter, projector)
    _, relative_train = token_parts(train_tokens)
    baseline = relative_train.mean(0).cpu()
    train_rms = float((relative_train - baseline.cuda()).square().sum((1, 2)).mean().sqrt())
    del train_tokens, relative_train, adapter, projector
    torch.cuda.empty_cache()

    strength, low_keep = DEFAULT_LAYOUT
    ridge, layout_source = reference_layout(subject, split_sha, pins)
    starts = layout(ridge, raw, repeats)
    seeds = [query_seed("dev", i) for i in range(len(rows))]
    track = "semantic-values" if values_only else (
        "semantic-edit" if edit else ("clean-flow-residual" if keep_residual else "clean-flow")
    )
    common = dict(render_cache_version=2, n=len(rows), row=rows, subject=subject, training=track,
                  budget=budget, repeats=repeats, prompt="", draws=1, unet_calls_per_draw=4, seeds=seeds,
                  development_split_sha256=dev_sha, split_sha256=split_sha, physical_eeg=eeg,
                  checkpoint_sha256=checkpoint, condition_fit=fit, source_conditions_sha256=inputs,
                  source_receipt_sha256=source_receipt, layout_strength=strength, layout_low_keep=list(low_keep),
                  layout_source=layout_source, layout_sha256=tensor_fingerprint(starts),
                  generator_files_sha256=generator_hashes, code_sha256=pins.pin(Path(__file__)), runtime=runtime())
    observations = {}
    paths = []
    for arm, tokens in arms.items():
        path = output / "images" / track / arm / budget / f"sub-{subject:02d}"
        target = donors.tolist() if arm.endswith("-wrong") else list(range(len(rows)))
        per_step = tokens if isinstance(tokens, tuple) else (tokens,)
        rms, digests = [], []
        for step_tokens in per_step:
            _, relative = token_parts(step_tokens.float())
            rms.append(float((relative - baseline).square().sum((1, 2)).mean().sqrt()))
            digests.append(tensor_fingerprint(step_tokens))
        if len(per_step) == 1:
            rms, digests = rms[0], digests[0]
        manifest = dict(common, arm=arm, eeg_donor=target, layout_donor=list(range(len(rows))),
                        token_sha256=digests, relative_token_rms=rms, train_reference_rms=train_rms,
                        residual_eeg_donor=list(range(len(rows))))
        if values_only:
            route = value_routes.get(arm)
            manifest.update(key_token_sha256=tensor_fingerprint(d2), value_timesteps=[] if route is None else route[1])

        def render(stage, batches, done):
            if not values_only or route is None:
                draw_images(stage, pipe, vae, prompts, tokens, seeds, starts, batch_starts=batches,
                            batch_complete=done, layout_strength=strength, layout_low_keep=low_keep)
                return
            reference, timesteps = route
            for first in batches:
                unet = RoutedUNet(pipe.unet, value_projections, reference[first:first+8].cuda(), timesteps)
                runner = SimpleNamespace(unet=unet, scheduler=pipe.scheduler)
                draw_images(stage, runner, vae, prompts, d2, seeds, starts, batch_starts=[first],
                            batch_complete=done, layout_strength=strength, layout_low_keep=low_keep)

        render_transaction(path, manifest, render, pins.verify)
        paths.append(path)
        observations[arm] = {"relative_token_rms": rms}
        print("Matched prior render", subject, arm, len(rows), flush=True)
    del pipe, vae
    torch.cuda.empty_cache()
    score_outputs(paths, "dev")
    for arm, path in zip(arms, paths, strict=True):
        observations[arm]["scores"] = json.loads((path / "scores.json").read_bytes())["mean"]
    save_json(output / "clean-prior-render.json", dict(design=common, reference_rms=train_rms, arms=observations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    trial = parser.add_mutually_exclusive_group()
    trial.add_argument("--keep-d2-residual", action="store_true")
    trial.add_argument("--semantic-edit", action="store_true")
    trial.add_argument("--semantic-values", action="store_true")
    args = parser.parse_args()
    run(args.subject, args.conditions, args.receipt, args.output, args.keep_d2_residual, args.semantic_edit, args.semantic_values)
