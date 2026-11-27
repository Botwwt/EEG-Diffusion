"""Estimate the token flow's conditional mean with paired, cancelling noise."""
import argparse
from pathlib import Path

import numpy as np
import torch

from paper_paired_token_flow import BUDGETS, FIT_ARMS, FLOW_STEPS, TokenFlow, output_root, seeded_initial, solve_flow


PAIRS = FLOW_STEPS
ARMS = ("native-paper", "flow-mean", "paired-mean", "paired-mean-wrong")


def expected_endpoint(field, condition, noise):
    """At time zero the expected velocity equals the expected target."""
    whole = condition.repeat(2, 1)
    time = noise.new_zeros(len(whole))
    total = torch.zeros_like(noise[0])
    for positive in noise:
        first, second = field(torch.cat((positive, -positive)), time, whole).chunk(2)
        total += first + second
    return total / (2 * len(noise))


def distance(first, second):
    return float((first - second).square().mean().sqrt())


def destination():
    return output_root() / "conditional-mean"


@torch.inference_mode()
def infer(subject, budgets, render_budgets):
    from common import PRIOR, ROOT
    from main.models_adapter import FusionEncoderModel, IPProjectionModel
    from paper_direct_tokens import restore_relative
    from paper_object_foreground import predict_at_budget, text_targets
    from paper_object_foreground_render import split_input
    from paper_oof_conditions import fitted_native
    from paper_render import generator_inputs, groups, query_seed, tensor_fingerprint
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_repeat_semantic_bridge import fit_head, predict
    from paper_soft_text import wrong_donors
    from paper_token_evidence import mix_tokens, project_tokens, render_arms, token_parts

    files = PinnedFiles()
    training, labels, _, text, states, basis_hash = text_targets(files)
    target_path = ROOT / "direct_token_encoder/training-targets.pt"
    target_hash = files.pin(target_path)
    target = torch.load(target_path, weights_only=True, map_location="cuda")
    require(np.array_equal(training, target["rows"].cpu()), "Token and semantic training orders differ")
    basis = target["basis"]
    width = target["coordinates"].shape[1]
    mapper = fit_head(target["coordinates"], text.cuda(), labels.cuda())
    native, checkpoint = fitted_native(subject, "cuda")
    checkpoints, models = {}, {}
    for arm in FIT_ARMS:
        path = output_root() / f"{arm}.pt"
        checkpoints[arm] = files.pin(path)
        packet = torch.load(path, weights_only=True, map_location="cuda")
        require(packet["arm"] == arm and packet["proof"]["target"] == target_hash, "Flow checkpoint does not fit these tokens")
        model = TokenFlow(width, len(packet["scale"])).cuda().eval()
        model.load_state_dict(packet["weights"])
        models[arm] = (model, packet["center"], packet["scale"], packet["batch_sha256"])
    require(models["flow"][3] == models["paired-flow"][3], "The two models used different training episodes")
    assets = generator_inputs(files)
    pipeline = None
    if render_budgets:
        from paper_identity_render import load_generator

        pipeline = load_generator(torch.float16, False)
        _, adapter, projection, _, _ = pipeline
    else:
        adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.cuda().half().eval()
        projection = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").cuda().half().eval()
    code_hash = files.pin(__file__)
    flow_code = files.pin(Path(__file__).with_name("paper_paired_token_flow.py"))
    native_hash = files.pin(checkpoint)
    for budget in budgets:
        raw, receipt, rows = split_input(subject, budget, files)
        repeats = groups(budget)
        features = predict_at_budget(native.backbone, raw, repeats)
        tokens = project_tokens(features, adapter, projection)
        mean, relative = token_parts(tokens)
        panel = budget.split("-")[0]
        seeds = [query_seed(panel, i) for i in range(len(rows))]
        noise = torch.stack([seeded_initial([seed ^ (i * 0x357AC259) for seed in seeds], width, features.device)
                             for i in range(PAIRS)])
        donors = wrong_donors(len(rows)).tolist()
        plain, center, scale, _ = models["flow"]
        paired, center_paired, scale_paired, _ = models["paired-flow"]
        context = ((features - center) / scale).clamp(-6, 6)
        context_paired = ((features - center_paired) / scale_paired).clamp(-6, 6)
        coords = {"flow-mean": expected_endpoint(plain, context, noise),
                  "paired-mean": expected_endpoint(paired, context_paired, noise),
                  "paired-mean-wrong": expected_endpoint(paired, context_paired[donors], noise)}
        deltas = {"native-paper": relative}
        deltas.update({arm: restore_relative(value, basis, calibrate=True) for arm, value in coords.items()})
        codes = {arm: ((delta - basis["center"]).flatten(1) @ basis["axes"]) / basis["scale"]
                 for arm, delta in deltas.items()}
        pooled = {arm: states["mean_pooled"] + predict(mapper, code).cpu() @ states["pooled_basis"]
                  for arm, code in codes.items()}
        out = destination() / "images" / budget / f"sub-{subject:02d}"
        out.mkdir(parents=True, exist_ok=True)
        proof = dict(subject=subject, budget=budget, repeats=repeats, physical_eeg=receipt,
                     native_checkpoint_sha256=native_hash, flow_checkpoints=checkpoints, training_target_sha256=target_hash,
                     semantic_basis_sha256=basis_hash, code_sha256=code_hash, flow_code_sha256=flow_code,
                     generator_inputs=assets, latent_solver_evaluations=2 * PAIRS,
                     latent_initial_sha256=tensor_fingerprint(noise.cpu()))
        files.verify()
        atomic_file(out / "semantic-readout.pt", lambda stream:
                    torch.save(dict(proof=dict(proof, row=rows), pool=pooled, donors=donors), stream))
        if budget not in render_budgets:
            path = solve_flow(paired, noise[0], context_paired)
            alternate = solve_flow(paired, -noise[0], context_paired)
            swapped = solve_flow(paired, noise[0], context_paired[donors])
            stats = dict(sample_noise_rms=distance(path, alternate), sample_eeg_rms=distance(path, swapped),
                         mean_eeg_rms=distance(coords["paired-mean"], coords["paired-mean-wrong"]))
            save_json(out / "noise-diagnostic.json", dict(stats, proof=dict(proof, row=rows)))
            print("Token conditional mean", subject, budget, stats, flush=True)
        else:
            own = list(range(len(rows)))
            variants = []
            for arm, delta in deltas.items():
                actual = tokens.half() if arm == "native-paper" else mix_tokens(mean, delta, torch.float16)
                donor = donors if arm == "paired-mean-wrong" else own
                variants.append((arm, actual.cpu(), dict(image_token_sha256=tensor_fingerprint(actual.cpu()),
                                 mean_eeg_donor=own, relative_eeg_donor=donor)))
            render_arms(out, variants, rows, seeds, proof, files, pipeline)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("infer", "score"))
    cli.add_argument("subject", type=int, choices=range(1, 11))
    cli.add_argument("--budgets", nargs="+", choices=BUDGETS, default=BUDGETS)
    cli.add_argument("--render-budgets", nargs="+", choices=BUDGETS, default=[])
    args = cli.parse_args()
    if not set(args.render_budgets).issubset(args.budgets):
        cli.error("Actual rendering also needs its input budget selected")
    if args.phase == "infer":
        infer(args.subject, args.budgets, args.render_budgets)
    else:
        import os

        from paper_object_foreground_render import score

        os.environ["EEG_CROSSFIT_ROOT"] = str(destination())
        score(args.subject, args.budgets, args.render_budgets, ARMS)
