"""Test earlier semantic influence with EEG and a separate real-photo positive control."""
import argparse
import inspect

import numpy as np
import torch


EARLY_GAIN = .25


def image_tokens_by_step(tokens, early_gain):
    early = tokens * early_gain
    return (early, early, tokens, tokens)


@torch.inference_mode()
def run(subject, diagnostic):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_object_foreground import load_encoder, predict_at_budget, root, text_targets
    from paper_object_foreground_render import split_input
    from paper_oof_conditions import fitted_native
    from paper_render import draw_images, generator_inputs, groups, query_seed, tensor_fingerprint
    from paper_render_cache import PinnedFiles, render_transaction
    from paper_repeat_semantic_bridge import fit_head, predict, text_condition
    from paper_soft_text import wrong_donors
    from paper_token_evidence import project_tokens, score_outputs

    pins = PinnedFiles()
    budget = "dev-r4"
    data, physical, rows = split_input(subject, budget, pins)
    repeats = groups(budget)
    native, checkpoint = fitted_native(subject, "cuda")
    native_hash = pins.pin(checkpoint)
    features = predict_at_budget(native.backbone, data, repeats)
    train_rows, labels, _, targets, states, basis = text_targets(pins)
    if diagnostic:
        source = PREP / "F_train.npy"
        proof = dict(held_photograph_features=pins.pin(source))
        array = np.load(source, mmap_mode="r", allow_pickle=False)
        train_photo = torch.from_numpy(array[train_rows].copy()).float().cuda()
        head = fit_head(train_photo, targets.cuda(), labels.cuda())
        held_photo = torch.from_numpy(array[rows].copy()).float().cuda()
        coords = predict(head, held_photo)
        settings = [
            ("visual-correct", 1., 1., False),
            ("visual-wrong", 1., 1., True),
            ("visual-correct-early", EARLY_GAIN, 1., False),
            ("visual-wrong-early", EARLY_GAIN, 1., True),
        ]
        dest = root() / "visual_timing_diagnostic"
    else:
        model, scale, digest = load_encoder(subject, "background", basis, native_hash, pins)
        coords = predict_at_budget(model, data, repeats) * scale
        proof = dict(trained_object_encoder=digest)
        settings = [
            ("background-early", EARLY_GAIN, 1., False),
            ("background-wrong-early", EARLY_GAIN, 1., True),
            ("background-boosted-early", EARLY_GAIN, 2., False),
            ("background-wrong-boosted-early", EARLY_GAIN, 2., True),
        ]
        dest = root() / "eeg_semantic_timing"
    cap = targets.cuda().norm(dim=1).quantile(.95)
    pipe, adapter, projection, empty, vae = load_generator(torch.float16, False)
    tokens = project_tokens(features, adapter, projection).half().cpu()
    original = list(range(len(tokens)))
    swapped = wrong_donors(len(tokens)).tolist()
    seeds = [query_seed("dev", i) for i in original]
    manifest = dict(render_cache_version=2, subject=subject, budget=budget, row=rows, n=len(rows), seeds=seeds,
                    draws=1, unet_calls_per_draw=4, repeats=repeats, physical_eeg=physical, diagnostic_only=diagnostic,
                    native_checkpoint_sha256=native_hash, semantic_basis_sha256=basis, **proof,
                    native_tokens=tensor_fingerprint(tokens), generator_inputs=generator_inputs(pins),
                    code=pins.pin(__file__), renderer=pins.pin(inspect.getfile(draw_images)), image_eeg_donor=original,
                    split_sha256=pins.pin(PREP / "paper_split.json"))
    locations = []
    for arm, gain, semantic_gain, wrong in settings:
        donors = swapped if wrong else original
        scaled = coords * semantic_gain
        scaled /= (scaled.norm(dim=1, keepdim=True) / cap).clamp_min(1.)
        prompt = (*text_condition(states, scaled[donors].cpu()), empty[2])
        step = image_tokens_by_step(tokens, gain)
        out = dest / f"sub-{subject:02d}" / arm

        def sample(path, starts, complete):
            draw_images(path, pipe, vae, prompt, step, seeds, batch_starts=starts, batch_complete=complete)

        render_transaction(out, dict(manifest, arm=arm, early_gain=gain, semantic_gain=semantic_gain,
                                     semantic_donor=donors), sample, pins.verify)
        locations.append(out)
        print("Object condition timing", subject, arm, flush=True)
    del pipe, vae, projection, adapter
    torch.cuda.empty_cache()
    score_outputs(locations, "dev")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("subject", type=int, choices=(1, 3))
    cli.add_argument("--real-photo-diagnostic", action="store_true")
    config = cli.parse_args()
    run(config.subject, config.real_photo_diagnostic)
