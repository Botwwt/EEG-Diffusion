"""Reconstruct registered four-trial queries with matched visual-target encoders."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from paper_blur_condition import ARMS, EPOCHS, MODES, BlurCondition, fit, measured_mean
from paper_d2_object_swap import d2_models, image_tokens, repeat_hidden
from paper_o13_composition import MIX, mix_directions, pooled_head
from paper_render_cache import PinnedFiles, render_transaction, require


R4 = [[0, 1, 2, 3]]


def load_fits(output, files, d2, split_sha, training):
    models, evidence, expected = {}, {}, None
    trainer = Path(inspect.getfile(fit))
    for mode in MODES:
        path = output / "fits" / f"{mode}.pt"
        checkpoint_sha = files.pin(path)
        fit_state = torch.load(path, map_location="cpu", weights_only=True)
        report_path = path.with_suffix(".json")
        report = json.loads(report_path.read_bytes())
        proof = fit_state["proof"]
        require(proof["mode"] == mode and proof["epochs"] == len(fit_state["history"]) == EPOCHS
                and proof["split"] == split_sha and proof["d2"] == d2 and training in proof["training_eeg"]
                and proof["sources"][trainer.name] == files.pin(trainer)
                and report == dict(fit_state["report"], checkpoint_sha256=checkpoint_sha),
                f"Checkpoint is not the registered matched fit: {path}")
        invariant = {key: value for key, value in proof.items() if key != "mode"}
        require(expected is None or expected == invariant, "The two fits use different training budgets or inputs")
        expected = invariant
        state = fit_state["model"]
        model = BlurCondition(state["input_mean"], state["input_scale"], state["axis"])
        model.load_state_dict(state, strict=True)
        models[mode] = model.cuda().eval()
        evidence[mode] = dict(checkpoint_sha256=checkpoint_sha, report_sha256=files.pin(report_path),
                              per_scale_mean_weight=report["per_scale_mean_weight"])
    return models, evidence


@torch.inference_mode()
def render(subject, output, panel):
    from common import PREP, RUNS
    from paper_identity_render import load_generator
    from paper_render import (donor_rows, draw_images, fit_layout, generator_inputs, layout, per_sample_cone,
                              physical_eeg, query_seed, render_code, runtime, tensor_fingerprint)
    from paper_token_evidence import development

    require(RUNS.name == "shared", "Select the registered shared D2 fit")
    files = PinnedFiles()
    split_path = PREP / "paper_split.json"
    split_sha = files.pin(split_path)
    split = json.loads(split_path.read_bytes())
    train, train_receipt = physical_eeg(subject, "train", files)
    dev_sha = None
    if panel == "dev":
        dev, rows = development(PREP, files)
        require(subject in dev["subjects"], "Select only the registered development participants")
        label_path = PREP / "train_concept.npy"
        files.pin(label_path)
        donors = donor_rows("token-reliable-wrong", "dev", np.load(label_path, allow_pickle=False)[rows])
        eeg, eeg_receipt = train, train_receipt
        dev_sha = files.pin(PREP / "paper_development.json")
    else:
        rows = split["official_rows"]
        require(rows == list(range(200)), "Unexpected official EEG order")
        donors = np.roll(np.arange(len(rows)), len(rows) // 2)
        eeg, eeg_receipt = physical_eeg(subject, "test", files)

    encoder, residual, cone, d2 = d2_models(files)
    models, fit_sources = load_fits(output, files, d2, split_sha, train_receipt)
    source = render_code(files, "d2")
    for function in (render, load_generator, image_tokens, mix_directions, measured_mean):
        path = Path(inspect.getfile(function))
        source[path.name] = files.pin(path)
    generators = generator_inputs(files)
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    ridge, layout_source = fit_layout(subject, split, train, files=files, train_proof=train_receipt, with_source=True)
    index = np.asarray(rows)
    starts = layout(ridge, eeg[index, :4], R4)
    measured = measured_mean(eeg, index)
    people = torch.full((len(rows),), subject - 1, device="cuda", dtype=torch.long)
    new_direction = {mode: model(measured, people) for mode, model in models.items()}
    baseline_tokens, base_fused, base_hidden = [], [], []
    for block in np.array_split(index, (len(index) + 19) // 20):
        raw = torch.from_numpy(eeg[block, :4].copy()).cuda().float()
        hidden = repeat_hidden(encoder, raw, subject)
        baseline_tokens.append(image_tokens(encoder.head_f, residual, hidden, cone, adapter, projection).cpu())
        base_fused.append(pooled_head(encoder.head_f, hidden))
        grouped, individual = hidden
        base_hidden.append(.72 * grouped + .28 * individual.reshape(len(raw), 4, -1).mean(1))
    baseline_tokens = torch.cat(baseline_tokens)
    baseline_fused = torch.cat(base_fused)
    hidden = torch.cat(base_hidden)
    seeds = [query_seed(panel, j) for j in range(len(rows))]
    original = dict(render_cache_version=2, subject=subject, row=rows, n=len(rows), budget=panel + "-r4",
                    repeats=R4, physical_eeg=eeg_receipt, split_sha256=split_sha, development_split_sha256=dev_sha,
                    generator_files_sha256=generators, d2_sources=d2, layout_source=layout_source, render_sources=source,
                    runtime=runtime(), seeds=seeds, timesteps=pipe.scheduler.timesteps.tolist(), draws=1,
                    unet_calls_per_draw=4, prompt="", base_donor=rows, layout_donor=rows,
                    base_tokens=tensor_fingerprint(baseline_tokens))
    require(torch.isfinite(baseline_tokens).all() and all(torch.isfinite(v).all() for v in new_direction.values()),
            "Nonfinite conditioning")
    for arm in ARMS:
        mode = arm.removesuffix("-wrong")
        swapped = arm.endswith("-wrong")
        donor = donors if swapped else np.arange(len(rows))
        tokens = baseline_tokens
        if mode in MODES:
            direction = mix_directions(baseline_fused, new_direction[mode][donor], cone["u_tr"])
            fused = per_sample_cone(direction, cone["R"], cone["alpha"], cone["beta"], cone["u_tr"])
            projected = torch.cat([projection(adapter(v.half())) for v in fused.split(20)])
            tokens = residual(hidden, projected).cpu()
        request = dict(original, arm=arm, diagnostic_only=swapped, fit=fit_sources.get(mode),
                       semantic_donor=index[donor].tolist(), semantic_mix=MIX if mode in MODES else 0.,
                       condition=tensor_fingerprint(tokens))
        destination = output / "images" / panel / f"sub-{subject:02d}" / arm

        def generate(folder, batches, complete):
            draw_images(folder, pipe, vae, prompts, tokens, seeds, starts,
                        batch_starts=batches, batch_complete=complete)

        render_transaction(destination, request, generate, files.verify)
        print("Blur condition render", subject, arm, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int, choices=range(1, 11))
    parser.add_argument("--panel", choices=("dev", "official"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.subject, args.output, args.panel)
