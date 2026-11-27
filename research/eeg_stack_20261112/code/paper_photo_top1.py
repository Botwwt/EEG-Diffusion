"""Evaluate single-query photo conditioning with the same four-step renderer and scorer."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import EMB_KEYS, PREP, PRIOR, ROOT, RUNS, TURBO
from exp0_intra10_swap_gate import OUT_DIR
from exp6_draft_intra4 import DRAFT_DIR
from paper_d2_object_swap import d2_models
from paper_o13_composition import binned_layout, match_training_layout
from paper_render import (
    CrossSubjectHorizonEncoder, condition, draw_images, forward_branches_and_trunk, generator_inputs,
    groups, layout, per_sample_cone, physical_eeg, prec_erp_analytic, query_seed, render_code, runtime,
    semantic_branch, tensor_fingerprint,
)
from paper_render_cache import PinnedFiles, render_transaction, require
from paper_training import generator
from paper_uniform_baselines import reference_layout

ARMS = ("d2", "d1", "consensus", "strong-layout", "native-late", "o13-late", "dual-late", "native-layout")
FULL_ARMS = ("native-full-layout", "native-full-late")
O13_WRONG_ARMS = ("o13-late-wrong", "o13-relative-late-wrong")
O13_RELATIVE_ARMS = ("o13-relative-late", "o13-relative-late-wrong")
O13_ARMS = ("o13-late", "o13-after-first", "dual-late", "o13-relative-late", *O13_WRONG_ARMS)
TANGENT_ARMS = ("center-late", "photo-align-late", "photo-inverse-late",
                "concept-align-late", "concept-inverse-late")
DEFAULT_LAYOUT = (.375, (8, .11, .02))
STRONG_LAYOUT = (.625, (8, .22, .04))


def balanced_groups(repetitions):
    """Ten disjoint eight-trial groups with two observations from each recording block."""
    require(repetitions == 80, "The balanced groups are defined for the complete physical R80 condition")
    return [[20*block + 2*i + j for block in range(4) for j in range(2)] for i in range(10)]


def direction_blend(cone, vectors, weights):
    axis = F.normalize(cone["u_tr"].to(vectors[0]).reshape(1, -1))
    projected = [F.normalize(v - (v @ axis.T) * axis, dim=1) for v in vectors]
    return sum(w * v for w, v in zip(weights, projected, strict=True))


def keep_token_mean(base, candidate):
    candidate_fp32 = candidate.float()
    relative = candidate_fp32 - candidate_fp32.mean(dim=1, keepdim=True)
    return (base.float().mean(dim=1, keepdim=True) + relative).to(base.dtype)


def ordinary_readout(encoder, trials, subject, repeat_groups):
    features, hidden = zip(*(semantic_branch(encoder, trials[:, group], subject, "d2")
                             for group in repeat_groups), strict=True)
    return torch.stack(features).mean(0), torch.stack(hidden).mean(0)


def consensus_hidden(encoder, trials, subject):
    n = trials.shape[1]
    whole = forward_branches_and_trunk(encoder, prec_erp_analytic(trials), subject - 1, 72. if n == 80 else 4.)[1]
    if n == 4:
        views = [forward_branches_and_trunk(encoder, trials[:, i], subject - 1, 1.)[1] for i in range(4)]
    else:
        views = [forward_branches_and_trunk(encoder, prec_erp_analytic(trials[:, bag]), subject - 1, 8.)[1]
                 for bag in balanced_groups(n)]
    return .85 * whole + .15 * torch.stack(views).mean(0)


def project_condition(feature, hidden, residual, cone, adapter, projection):
    fused = per_sample_cone(feature, cone["R"], cone["alpha"], cone["beta"], cone["u_tr"])
    encoded = torch.cat([projection(adapter(x.half())) for x in fused.split(20)])
    return residual(hidden, encoded).cpu()


def load_composition(subject, train_rows, starts, files):
    from main.models_adapter import FusionEncoderModel

    base_path = OUT_DIR / "arm0_tr14_xbrain_s2025.pt"
    layout_path = DRAFT_DIR / "z0_oof_tr14_256.pt"
    visual_path = PREP / "emb_train.pt"
    sources = {p.name: files.pin(p) for p in (base_path, layout_path, visual_path)}
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    require(base["seed"] == 2025 and base["epochs"] == 150, "Unexpected shared composition fit")
    brain = CrossSubjectHorizonEncoder().cuda().float().eval().requires_grad_(False)
    brain.load_state_dict(base["state_dict"], strict=True)
    image = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model
    image = image.cuda().float().eval().requires_grad_(False)

    fold = torch.load(layout_path, map_location="cpu", weights_only=True)
    require(fold["Z_oof"].shape == (10, len(train_rows), 256), "Training layout split changed")
    oof = binned_layout(fold["Z_oof"][subject - 1].cuda().float(),
                        fold["V_pca"].cuda().float(), fold["my"].cuda().float())
    visual = torch.load(visual_path, map_location="cpu", weights_only=True)[EMB_KEYS[2]][train_rows].float().flatten(1)
    visual = visual.cuda()
    require(visual.shape == oof.shape, "Visual and EEG layout targets differ")
    decoded = match_training_layout(F.avg_pool2d(starts.cuda().float(), 4).flatten(1), oof, visual)
    require(bool(torch.isfinite(decoded).all()), "Nonfinite composition layout")
    return brain, image, decoded, float(base["h14_norm"]), float(base["b32_norm"]), sources


def render_inputs(subject, panel, files):
    split_path = PREP / "paper_split.json"
    split_hash = files.pin(split_path)
    split = json.loads(split_path.read_bytes())
    train = split["train_rows"]
    dev_hash = None
    if panel == "dev":
        registration = PREP / "paper_development.json"
        dev_hash = files.pin(registration)
        rows = json.loads(registration.read_bytes())["dev_rows"]
        require(not set(rows) & set(train), "Development rows overlap model training")
        source, eeg_proof = physical_eeg(subject, "train", files)
        raw = source[rows]
        budget = "dev-r4"
    elif panel == "fresh":
        rows = split["fresh_rows"]
        registration = PREP / "paper_development.json"
        files.pin(registration)
        dev_rows = json.loads(registration.read_bytes())["dev_rows"]
        require(set(rows).isdisjoint(train) and set(rows).isdisjoint(dev_rows),
                "Fresh rows overlap training or the development panel")
        source, eeg_proof = physical_eeg(subject, "train", files)
        raw = source[rows]
        budget = "fresh-r4"
    else:
        rows = split["official_rows"]
        raw, eeg_proof = physical_eeg(subject, "test", files)
        require(rows == list(range(len(raw))), "Official EEG order changed")
        budget = "official-r80"
    repeat_groups = groups(budget)
    repeats = np.asarray(repeat_groups).ravel().tolist()
    require(len(rows) == len(set(rows)) == (500 if panel == "fresh" else 200),
            "Run the complete registered cohort for each subject")
    return raw, rows, train, budget, repeat_groups, repeats, split_hash, dev_hash, eeg_proof


def full_native_fit(subject, files):
    name = f"sub-{subject:02d}"
    folder = ROOT / f"fits/native-full-{subject:02d}/enc/native" / name / "fold-1_seed2025"
    cp, meta = folder / "brain_model.pt", folder / "fit.json"
    meta_hash = files.pin(meta)
    fit = json.loads(meta.read_bytes())
    require(fit["n_train"] == 16540 and fit["args"]["sub"] == subject and fit["args"]["fold"] == -1,
            "Expected the original full-training Brain-HiVE encoder")
    proof = {"condition_checkpoint": {"path": str(cp.relative_to(ROOT)), "sha256": files.pin(cp)},
             "fit_sha256": meta_hash, "train_concepts": 1654}
    return cp, None, proof


def full_shared_fit(files):
    run = RUNS.parent / "shared-full"
    release = Path(os.environ.get("EEG_FULL_SHARED_RELEASE", ROOT / "shared-full-run/fitted-weights.sha256"))
    attestation = {}
    for stage in ("backbone", "rep", "swap", "oof", "d2"):
        info = run / f"{stage}-training.json"
        event = json.loads(info.read_bytes())
        require(event["stage"] == stage and event["scope"] == "official-training" and event["photos"] == 16540
                and event["concepts"] == 1654 and event["people"] == list(range(1, 11)), "Incomplete full-training fit")
        attestation[stage] = files.pin(info)
    pins = {Path(name.strip()): digest for digest, name in (line.split(maxsplit=1) for line in release.read_text().splitlines())}
    cp = run / "swap_diff/draft_intra4/d2_compete4_cond_st120_s2025.pt"
    backbone = run / "swap_diff/arm3b_rep_rawswap_condlyr_st600_s2025.pt"
    require(len(pins) == 5 and all(path.is_relative_to(run) and path.is_file() and files.pin(path) == digest
                                   for path, digest in pins.items()) and cp in pins and backbone in pins,
            "Released full-population weights differ from the completed training job")
    saved = torch.load(cp, weights_only=True, map_location="meta")
    require(Path(saved["backbone_ckpt"]) == backbone and saved["mode"] == "d2_compete4_cond",
            "The D2 residual does not reference the full-training backbone")
    proof = {"semantic_arm": "d2", "condition_checkpoint": {"path": str(cp.relative_to(ROOT)), "sha256": pins[cp]},
             "nested_backbone": {"path": str(backbone.relative_to(ROOT)), "sha256": pins[backbone]},
             "full_training": {"release": files.pin(release), "stages": attestation, "train_concepts": 1654}}
    return cp, backbone, proof


def load_tangent_calibration(folder, checkpoint, cone, registrations, files):
    import paper_tangent_calibration as fit_code

    metadata = folder / "tangent-calibration.json"
    state = folder / "tangent-calibration.pt"
    receipt = json.loads(metadata.read_bytes())
    digest = files.pin(state)
    require(receipt["weights"] == digest and receipt["model"] == checkpoint
            and receipt["angle_mix"] == fit_code.ANGLE_MIX and receipt["code_sha256"] == files.pin(Path(fit_code.__file__))
            and receipt["training"]["split"] == registrations["split"]
            and (registrations["development"] is None or receipt["training"]["development"] == registrations["development"])
            and receipt["training"]["labels"] == files.pin(PREP / "train_concept.npy")
            and receipt["training"]["truth"] == files.pin(PREP / "F_train.npy"),
            "Tangent calibration was fitted to a different D2 model or training split")
    fitted = torch.load(state, map_location="cuda", weights_only=True)
    axis, centers, rotations = (fitted[k] for k in ("axis", "centers", "rotations"))
    require(axis.shape == (1024,) and centers.shape == (10, 1024)
            and set(rotations) == {"photo", "concept"}
            and all(x.shape == (1024, 1024) and torch.isfinite(x).all() for x in rotations.values())
            and torch.isfinite(centers).all()
            and torch.allclose(axis, cone["u_tr"].reshape(-1).to(axis), rtol=0., atol=2e-6)
            and abs(fitted["alpha"] - cone["alpha"]) < 2e-5, "Tangent calibration changed the registered cone")
    return axis, centers, rotations, {"fit": files.pin(metadata), "weights": digest, "mix": fit_code.ANGLE_MIX}


@torch.inference_mode()
def run(subject, panel, arms, output, tangent_calibration=None, full_d2=False):
    require(RUNS.name == "shared", "Use the registered shared D2 fit root")
    require(not full_d2 or panel == "official", "The full fit uses the development concepts")
    from paper_render import condition_files

    files = PinnedFiles()
    x, rows, train, budget, repeat_groups, repeats, split_hash, dev_hash, physical = render_inputs(subject, panel, files)
    ridge, layout_source = reference_layout(subject, split_hash, files)
    starts = layout(ridge, x, repeat_groups)
    generator_files = generator_inputs(files)
    from diffusers import AutoencoderKL

    pipe, adapter, projection, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16",
                                       torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    prompts = prompt, pooled, dims
    d2, residual, cone, d2_sources = d2_models(files, full_shared_fit(files) if full_d2 else None)
    if any(a in TANGENT_ARMS for a in arms):
        from paper_tangent_calibration import aligned_tangent

        require(tangent_calibration is not None, "Select the frozen training-only tangent calibration")
        axis, centers, rotations, calibration = load_tangent_calibration(
            tangent_calibration, d2_sources, cone, dict(split=split_hash, development=dev_hash), files)

    needs_native = any(a in ("native-late", "dual-late", "native-layout") for a in arms)
    needs_full = any(a in FULL_ARMS for a in arms)
    needs_o13 = any(a in O13_ARMS for a in arms)
    needs_o13_wrong = any(a in O13_WRONG_ARMS for a in arms)
    native_sources, composition_sources = None, None
    if needs_native:
        prepared = condition_files("native-paper", subject, files)
        native, _, native_sources = condition("native-paper", subject, x, repeat_groups, features_only=True,
                                               files=files, prepared=prepared, return_source=True)
        native = native.cuda()
    full_sources = None
    if needs_full:
        require(panel == "official", "Full-training encoders overlap the restricted development gallery")
        prepared = full_native_fit(subject, files)
        native_full, _, full_sources = condition("native-paper", subject, x, repeat_groups, features_only=True,
                                                   files=files, prepared=prepared, return_source=True)
        native_full = native_full.cuda()
    if needs_o13:
        brain, visual, layout_features, h_scale, b_scale, composition_sources = load_composition(
            subject, train, starts, files)

    f_base, h_base, f_consensus, h_consensus, f_composition = [], [], [], [], []
    h14_composition, b32_composition = [], []
    need_consensus = any(a in ("consensus", "dual-late") for a in arms)
    for lo in range(0, len(x), 20):
        hi = lo + 20
        xb = torch.from_numpy(np.array(x[lo:hi], copy=True)).cuda().float()
        f, h = ordinary_readout(d2, xb, subject, repeat_groups)
        f_base.append(f)
        h_base.append(h)
        if need_consensus or needs_o13:
            selected = xb[:, repeats]
        if need_consensus:
            pooled = consensus_hidden(d2, selected, subject)
            f_consensus.append(d2.head_f(pooled))
            h_consensus.append(pooled)
        if needs_o13:
            pooled = consensus_hidden(brain, selected, subject)
            clip_h = F.normalize(brain.head_h14(pooled)) * h_scale
            clip_b = F.normalize(brain.head_b32(pooled)) * b_scale
            fv = visual(**dict(zip(EMB_KEYS, (clip_h, clip_b, layout_features[lo:hi]), strict=True)))
            f_composition.append(fv.float())
            if needs_o13_wrong:
                h14_composition.append(clip_h)
                b32_composition.append(clip_b)

    f0, h0 = map(torch.cat, (f_base, h_base))
    t0 = project_condition(f0, h0, residual, cone, adapter, projection)
    if "d1" in arms:
        require(not full_d2, "D1 is matched to the registered restricted-training D2 backbone")
        t_d1, d1_digest, d1_sources = condition("d1", subject, x, repeat_groups, adapter, projection,
                                               files=files, return_source=True)
        require(d1_sources["nested_backbone"] == d2_sources["nested_backbone"],
                "The D1 comparison requires the same EEG backbone")
    if need_consensus:
        fc, hc = map(torch.cat, (f_consensus, h_consensus))
    if needs_o13:
        fv = torch.cat(f_composition)
    if needs_o13_wrong:
        donors = np.roll(np.arange(len(rows)), len(rows) // 2)
        swapped_h14 = torch.cat(h14_composition)[donors]
        swapped_b32 = torch.cat(b32_composition)[donors]
        swapped = []
        for lo in range(0, len(rows), 20):
            batch = slice(lo, lo + 20)
            parts = swapped_h14[batch], swapped_b32[batch], layout_features[batch]
            swapped.append(visual(**dict(zip(EMB_KEYS, parts, strict=True))).float())
        wrong_composition = torch.cat(swapped)

    source_hashes = render_code(files, "d2")
    for path in (Path(__file__), Path(__import__("paper_o13_composition").__file__),
                 Path(__import__("paper_d2_object_swap").__file__)):
        source_hashes[path.name] = files.pin(path)
    seeds = [query_seed(panel, i) for i in range(len(rows))]
    track = "shared-full" if full_d2 else "shared"
    receipts = dict(render_cache_version=2, training=track, subject=subject, budget=budget,
                    n=len(rows), prompt="", draws=1, unet_calls_per_draw=4, repeats=repeat_groups,
                    row=rows, eeg_donor=list(range(len(rows))), seeds=seeds, split_sha256=split_hash,
                    development_split_sha256=dev_hash, checkpoint_sha256=d2_sources["condition_checkpoint"]["sha256"],
                    d2_sources=d2_sources, physical_eeg=physical, layout_source=layout_source,
                    layout_mode="ridge-shell", generator_files_sha256=generator_files, render_code_sha256=source_hashes,
                    runtime=runtime())
    for arm in arms:
        anchoring, low = STRONG_LAYOUT if arm == "strong-layout" else DEFAULT_LAYOUT
        tokens = t0
        if arm == "d1":
            tokens = t_d1
        elif arm == "consensus":
            tokens = project_condition(fc, hc, residual, cone, adapter, projection)
        elif arm in ("native-layout", "native-full-layout"):
            source = native_full if arm == "native-full-layout" else native
            tokens = torch.cat([projection(adapter(z.half())).cpu() for z in source.split(20)])
        elif arm in (*O13_ARMS, "native-late", "native-full-late"):
            if arm == "dual-late":
                mix = direction_blend(cone, (fc, native, fv), (.5, .25, .25))
                hidden = hc
            else:
                if arm == "native-late":
                    other = native
                elif arm == "native-full-late":
                    other = native_full
                elif arm in O13_WRONG_ARMS:
                    other = wrong_composition
                else:
                    other = fv
                mix = direction_blend(cone, (f0, other), (.5, .5))
                hidden = h0
            detail = project_condition(mix, hidden, residual, cone, adapter, projection)
            if arm in O13_RELATIVE_ARMS:
                detail = keep_token_mean(t0, detail)
            tokens = (t0, detail if arm == "o13-after-first" else t0, detail, detail)
        elif arm in TANGENT_ARMS:
            edit = None if arm == "center-late" else rotations[arm.split("-", 1)[0]]
            if "-inverse-" in arm:
                edit = edit.T
            direction = aligned_tangent(f0, axis, centers[subject - 1], edit)
            detail = project_condition(direction, h0, residual, cone, adapter, projection)
            tokens = t0, t0, detail, detail

        token_list = tokens if isinstance(tokens, tuple) else (tokens,) * 4
        require(all(bool(torch.isfinite(t).all()) and len(t) == len(rows) for t in token_list), "Invalid query tokens")
        out = output / "images" / track / arm / budget / f"sub-{subject:02d}"
        request = dict(receipts, arm=arm, semantic_condition=arm, layout_strength=anchoring,
                       layout_low_keep=list(low), token_sha256=[tensor_fingerprint(t) for t in token_list])
        if arm == "d1":
            request.update(checkpoint_sha256=d1_digest, d1_sources=d1_sources)
        if arm in ("native-late", "native-layout", "dual-late"):
            request["native_sources"] = native_sources
        if arm in FULL_ARMS:
            request["native_sources"] = full_sources
        if arm in O13_ARMS:
            request["composition_sources"] = composition_sources
        if arm in O13_WRONG_ARMS:
            request["composition_semantic_donor"] = donors.tolist()
            request["recipient_composition_layout_preserved"] = True
        if arm in O13_RELATIVE_ARMS:
            request["token_mean"] = "shared-d2"
        if arm in (*O13_ARMS, "consensus"):
            request["consensus"] = {"whole_weight": .85, "bag_weight": .15,
                                    "balanced_r80": balanced_groups(80) if panel == "official" else None}
        if arm in TANGENT_ARMS:
            request["tangent_calibration"] = calibration

        def generate(stage, batches, done):
            draw_images(stage, pipe, vae, prompts, tokens, seeds, starts, batch_starts=batches,
                        batch_complete=done, layout_strength=anchoring, layout_low_keep=low)

        render_transaction(out, request, generate, files.verify)
        print("rendered", out, len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("dev", "fresh", "official"), required=True)
    parser.add_argument("--subject", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(dict.fromkeys(ARMS + FULL_ARMS + TANGENT_ARMS + O13_ARMS)), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tangent-calibration", type=Path)
    parser.add_argument("--full-d2", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "Use a Slurm compute node for EEG and model workloads")
    require(len(args.arms) == len(set(args.arms)), "Duplicate method in run")
    run(args.subject, args.panel, args.arms, args.output, args.tangent_calibration, args.full_d2)


if __name__ == "__main__":
    main()
