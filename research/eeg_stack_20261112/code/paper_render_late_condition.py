"""Compare late condition fits with one reconstruction per registered four-trial EEG query."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_d2_object_swap import d2_models, image_tokens, repeat_hidden
from paper_late_condition import ARMS, MODES, PAIR_WEIGHT, R4, STEPS, TIMES, LateCondition, make_step_tokens, pooled_hidden, sources
from paper_render_cache import render_transaction, require


def load_decoders(output, files, code, d2, generator, split, training):
    models, receipts, common_proofs = {}, {}, []
    for arm in MODES:
        path = output / "fits" / f"{arm}.pt"
        digest = files.pin(path)
        report_path = path.with_suffix(".json")
        report_digest = files.pin(report_path)
        report = json.loads(report_path.read_bytes())
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        proof = checkpoint["proof"]
        require(proof["arm"] == arm and proof["steps"] == len(checkpoint["history"]) == STEPS
                and proof["code"] == code and proof["d2"] == d2 and proof["generators"] == generator
                and proof["inputs"]["paper_split.json"] == split and training in proof["eeg"]
                and proof["pair_weight"] == (PAIR_WEIGHT if arm == "late-paired" else 0.)
                and checkpoint["report"] == {k: v for k, v in report.items() if k != "checkpoint_sha256"}
                and report["proof"] == proof and report["checkpoint_sha256"] == digest,
                f"Wrong registered late condition checkpoint: {path}")
        model = LateCondition()
        model.load_state_dict(checkpoint["model"], strict=True)
        models[arm] = model.cuda().eval()
        receipts[arm] = dict(checkpoint_sha256=digest, report_sha256=report_digest, steps=STEPS,
                             training_order_sha256=report["training_order_sha256"])
        common_proofs.append({k: v for k, v in proof.items() if k not in ("arm", "pair_weight")})
    require(common_proofs[0] == common_proofs[1]
            and receipts[MODES[0]]["training_order_sha256"] == receipts[MODES[1]]["training_order_sha256"],
            "Both fits must use the same training data, initialization and episodes")
    return models, receipts


@torch.inference_mode()
def render(subject, panel, output):
    from common import PREP, RUNS
    from paper_identity_render import load_generator
    from paper_render import (donor_rows, draw_images, fit_layout, generator_inputs, layout, physical_eeg,
                              query_seed, runtime, tensor_fingerprint)
    from paper_token_evidence import development

    require(RUNS.name == "shared", "Use the registered shared D2 fit")
    files, code = sources()
    split_path = PREP / "paper_split.json"
    split_digest = files.pin(split_path)
    split = json.loads(split_path.read_bytes())
    train, training = physical_eeg(subject, "train", files)
    dev_digest = None
    if panel == "dev":
        dev, rows = development(PREP, files)
        require(subject in dev["subjects"], "Use the registered development participants")
        dev_digest = files.pin(PREP / "paper_development.json")
        labels = PREP / "train_concept.npy"
        files.pin(labels)
        donor = donor_rows("token-reliable-wrong", "dev", np.load(labels, allow_pickle=False)[rows])
        eeg, physical = train, training
    else:
        rows = split["official_rows"]
        require(rows == list(range(200)), "Use the full official test set in its original order")
        donor = np.roll(np.arange(len(rows)), len(rows) // 2)
        eeg, physical = physical_eeg(subject, "test", files)

    encoder, original, cone, d2 = d2_models(files)
    generators = generator_inputs(files)
    models, fits = load_decoders(output, files, code, d2, generators, split_digest, training)
    pipe, adapter, projection, prompts, vae = load_generator(torch.float16, False)
    require(tuple(pipe.scheduler.timesteps.tolist()) == TIMES, "The late decoder expects the registered four steps")
    ridge, layout_source = fit_layout(subject, split, train, files=files, train_proof=training, with_source=True)
    index = np.asarray(rows)
    starts = layout(ridge, eeg[index, :4], R4)
    base_blocks = []
    delta_blocks = {arm: [] for arm in MODES}
    for block in np.array_split(index, (len(index) + 19) // 20):
        raw = torch.from_numpy(eeg[block, :4].copy()).cuda().float()
        encoded = repeat_hidden(encoder, raw, subject)
        base = image_tokens(encoder.head_f, original, encoded, cone, adapter, projection)
        hidden = pooled_hidden(encoded)
        base_blocks.append(base.cpu())
        for arm, model in models.items():
            delta_blocks[arm].append(model(hidden, base).cpu())
    base = torch.cat(base_blocks)
    residuals = {arm: torch.cat(blocks, dim=1) for arm, blocks in delta_blocks.items()}
    require(base.shape == (len(rows), 4, 2048) and torch.isfinite(base).all()
            and all(torch.isfinite(value).all() for value in residuals.values()), "Invalid inference conditions")
    seeds = [query_seed(panel, index) for index in range(len(rows))]
    source = dict(code, **{Path(__file__).name: files.pin(Path(__file__))})
    shared = dict(render_cache_version=2, subject=subject, row=rows, n=len(rows), budget=panel + "-r4", repeats=R4,
                  physical_eeg=physical, split_sha256=split_digest, development_split_sha256=dev_digest,
                  generator_files_sha256=generators, layout_source=layout_source, d2_sources=d2,
                  render_sources=source, runtime=runtime(), seeds=seeds, timesteps=list(TIMES),
                  draws=1, unet_calls_per_draw=4, prompt="", base_donor=rows, layout_donor=rows,
                  base_tokens=tensor_fingerprint(base), diagnostic_only=False)
    for arm in ARMS:
        fit = arm.removesuffix("-wrong")
        swapped = arm.endswith("-wrong")
        delta = residuals.get(fit)
        if delta is not None and swapped:
            delta = delta[:, donor]
        tokens = (base,) * 4 if delta is None else make_step_tokens(base, delta)
        destination = output / "images" / panel / f"sub-{subject:02d}" / arm
        request = dict(shared, arm=arm, fit=fits.get(fit), diagnostic_only=swapped,
                       late_donor=(index[donor] if swapped else index).tolist() if delta is not None else None,
                       condition=[tensor_fingerprint(t) for t in tokens])

        def generate(folder, batches, complete):
            draw_images(folder, pipe, vae, prompts, tokens, seeds, starts,
                        batch_starts=batches, batch_complete=complete)

        render_transaction(destination, request, generate, files.verify)
        print("Late condition render", subject, arm, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int, choices=range(1, 11))
    parser.add_argument("--panel", choices=("dev", "official"), default="dev")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.subject, args.panel, args.output)
