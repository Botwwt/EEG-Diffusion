"""Measure the spread and bias of EEG conditions at the actual SDXL image-token interface."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, require, save_json


def geometry(prediction, visual_center, visual_radius):
    values = prediction.float().flatten(1)
    center = values.mean(0)
    flat_center = visual_center.flatten()
    offset = values - center
    power = offset.square().mean()
    rng = torch.Generator(device=values.device).manual_seed(20261127)
    directions = torch.randn(values.shape[1], 32, device=values.device, generator=rng)
    require(power > 0 and visual_radius > 0 and torch.isfinite(values).all(), "Invalid condition geometry")
    normalized = F.normalize(offset)
    probe = normalized @ directions
    # Use projected directions only as a compact diversity diagnostic, not a calibration statistic.
    concentration = (probe.T @ probe / len(values)).square().sum().sqrt() / probe.square().mean(0).sum()
    bias = (center - flat_center).square().mean().sqrt()
    return dict(radius_relative_to_training_photos=float(power.sqrt() / visual_radius),
                bias_relative_to_training_photos=float(bias / visual_radius),
                bias_over_radius=float(bias / power.sqrt()), random_projection_concentration=float(concentration),
                mean_norm=float(center.norm()), centered_row_norm_quantiles=offset.norm(dim=1).quantile(
                    torch.tensor([.05, .5, .95], device=values.device)).tolist())


def reference(train):
    values = train.float()
    center = values.mean(0)
    return center, (values - center).square().mean().sqrt()


@torch.inference_mode()
def cbramod_features(subject, arm, previous, raw, rows, train_receipt, files):
    import paper_cbramod_repeats as cbramod

    path = cbramod.model_dir(previous, subject, arm) / "model.pt"
    pack_path = path.with_name("fit.json")
    pack_sha, fit_sha = files.pin(path), files.pin(pack_path)
    fit_receipt = json.loads(pack_path.read_bytes())
    pack = torch.load(path, map_location="cpu", weights_only=True)
    require(fit_receipt["checkpoint_sha256"] == pack_sha and pack["subject"] == subject and pack["arm"] == arm
            and pack["proof"]["physical_eeg"] == train_receipt and pack["steps"] == 880
            and pack["proof"]["code_sha256"] == files.pin(Path(cbramod.__file__)), "Wrong CBraMod input or fit provenance")
    state = pack["state"]
    args = [state[k] for k in ("input_center", "input_scale", "photo_center", "photo_scale")]
    model = cbramod.PhotoCondition(cbramod.upstream_model(previous / "upstream", False, files), arm, *args)
    model.load_state_dict(state, strict=True)
    model.cuda().eval().requires_grad_(False)
    values = []
    for recordings, ids in zip(raw, rows, strict=True):
        values.append(torch.cat([model(cbramod.measured_trials(recordings, group)).cpu()
                                 for group in np.array_split(ids, (len(ids) + 31) // 32)]))
    return values, dict(model_sha256=pack_sha, fit_sha256=fit_sha)


def load_projection():
    from common import PRIOR
    from main.models_adapter import FusionEncoderModel, IPProjectionModel

    return (FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.cuda().half().eval(),
            IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").cuda().half().eval())


@torch.inference_mode()
def run(subject, previous, output):
    from common import PREP
    from paper_oof_conditions import registered_rows
    from paper_render import condition, generator_inputs, groups, physical_eeg
    from paper_token_evidence import project_tokens

    files = PinnedFiles()
    train_rows, _, _ = registered_rows(PREP)
    train, train_proof = physical_eeg(subject, "train", files)
    test, test_proof = physical_eeg(subject, "test", files)
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    test_rows = np.asarray(json.loads(split.read_bytes())["official_rows"])
    visual_path = PREP / "F_train.npy"
    visual_sha = files.pin(visual_path)
    visual = torch.from_numpy(np.load(visual_path, mmap_mode="r", allow_pickle=False)[train_rows].copy()).cuda().float()
    repeat = groups("official-r4")
    require(repeat == [[0, 1, 2, 3]] and np.array_equal(test_rows, np.arange(200)), "Wrong registered observations")
    adapter, project = load_projection()
    photo_token = project_tokens(visual, adapter, project)
    centers = dict(fused=reference(visual), ip=reference(photo_token))
    probes, sources, outputs = {}, {}, {}
    for arm in ("native-paper", "joint", "separate", "d2"):
        if arm in ("joint", "separate"):
            (fit, query), sources[arm] = cbramod_features(subject, arm, previous, (train, test), (train_rows, test_rows), train_proof, files)
        else:
            fit, checkpoint1 = condition(arm, subject, train[train_rows], repeat, adapter, project,
                                        features_only=arm == "native-paper", files=files)
            query, checkpoint2 = condition(arm, subject, test[test_rows], repeat, adapter, project,
                                          features_only=arm == "native-paper", files=files)
            require(checkpoint1 == checkpoint2, "Training and official query used different encoders")
            sources[arm] = dict(model_sha256=checkpoint1)
        stats = {}
        if arm == "d2":
            fit_token, query_token = fit.cuda(), query.cuda()
        else:
            fit, query = fit.cuda().float(), query.cuda().float()
            fit_token, query_token = (project_tokens(x, adapter, project) for x in (fit, query))
            stats["fused"] = {phase: geometry(value, *centers["fused"])
                              for phase, value in (("train", fit), ("official", query))}
        stats["ip"] = {phase: geometry(value, *centers["ip"])
                       for phase, value in (("train", fit_token), ("official", query_token))}
        outputs[arm] = dict(fit_fused=fit.cpu().float() if arm != "d2" else None,
                            query_fused=query.cpu().float() if arm != "d2" else None,
                            fit_ip=fit_token.cpu().half() if arm == "d2" else None, query_ip=query_token.cpu().half())
        probes[arm] = stats
        print("Condition spread", subject, arm, json.dumps(stats), flush=True)
        del fit, query, fit_token, query_token
    proof = dict(subject=subject, rows_sha256=files.pin(PREP / "train_ids.json"), training=train_proof, official=test_proof,
                 split_sha256=split_sha, visual_sha256=visual_sha, repeats=repeat, sources=sources,
                 generators=generator_inputs(files), code=files.pin(Path(__file__)))
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"conditions-sub-{subject:02d}.pt"
    files.verify()
    atomic_file(path, lambda stream: torch.save(dict(proof=proof, statistics=probes, rows=torch.from_numpy(train_rows),
                visual=visual.cpu(), visual_ip_center=centers["ip"][0].cpu(), visual_ip_radius=centers["ip"][1].cpu(),
                models=outputs), stream))
    save_json(path.with_suffix(".json"), dict(proof=proof, statistics=probes, output_sha256=files.pin(path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int, choices=range(1, 11))
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.subject, args.previous, args.output)
