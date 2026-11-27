"""Train matched EEG encoders with I-JEPA supervision on relative generator tokens."""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from paper_ijepa_context import MODES, RANK, UPSTREAM
from paper_middle_teacher import losses
from paper_object_tokens import BATCH, EPOCHS, SEED, epoch_batches, group_index
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


ARMS = ("native-paper", *MODES, "photo-map-wrong")
AUXILIARY_WEIGHT = .5


def open_targets(root, rows, prep, files):
    path = root / "targets.pt"
    digest = files.pin(path)
    report_path = path.with_suffix(".json")
    files.pin(report_path)
    proof = json.loads(report_path.read_bytes())
    values = torch.load(path, weights_only=True, map_location="cpu")
    info = values["info"]
    require(proof == dict(info, packet_sha256=digest), "I-JEPA target receipt does not match the tensor packet")
    registration = info["proof"]
    ids = json.loads((prep / "eeg_train_ids.json").read_bytes())["ids"]
    identity = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    require(np.array_equal(values["rows"].numpy(), rows) and info["n"] == len(rows)
            and registration["split"] == files.pin(prep / "paper_split.json")
            and registration["concepts"] == files.pin(prep / "train_concept.npy")
            and registration["eeg_identity_list"] == identity
            and registration["source_revision"] == UPSTREAM
            and set(values["targets"]) == set(MODES), "Targets must align to the registered training EEG")
    for arm in MODES:
        target = values["targets"][arm]
        require(target.shape == (len(rows), RANK) and torch.isfinite(target).all(), "Invalid auxiliary target")
    return values["targets"], digest


def fit(subject, root):
    from common import PREP, PRIOR, ROOT
    from main.models_adapter import FusionEncoderModel, IPProjectionModel
    from paper_instance_residual import make_encoder
    from paper_oof_conditions import fitted_native, registered_rows
    from paper_render import physical_eeg
    from paper_token_evidence import token_parts

    files = PinnedFiles()
    rows, labels, _ = registered_rows(PREP)
    cached, target_digest = open_targets(root, rows, PREP, files)
    raw, eeg = physical_eeg(subject, "train", files)
    original, native = fitted_native(subject, "cpu")
    initial = original.state_dict()
    eeg_batches = [torch.from_numpy(raw[idx].copy()).cuda().float().mean(1)
                   for idx in np.array_split(rows, math.ceil(len(rows)/128))]
    waveforms = torch.cat(eeg_batches)
    del eeg_batches
    photo_features = PREP / "F_train.npy"
    split_hash = files.pin(PREP / "paper_split.json")
    photo_hash = files.pin(photo_features)
    source_f = torch.from_numpy(np.load(photo_features, allow_pickle=False, mmap_mode="r")[rows].copy()).float().cuda()
    target_path = ROOT / "direct_token_encoder/training-targets.pt"
    basis_source = files.pin(target_path)
    relative = torch.load(target_path, weights_only=True, map_location="cuda")
    require(relative["sources"]["split"] == split_hash and relative["sources"]["photos"] == photo_hash
            and np.array_equal(relative["rows"].cpu().numpy(), rows), "Generator basis used other photographs")
    basis = relative["basis"]
    offset = basis["center"].float().cuda()
    axes = basis["axes"].float().cuda()
    scale = basis["scale"].float().cuda().clamp_min(.01)
    require(axes.shape == (8192, RANK) and torch.isfinite(axes).all(), "Unexpected token representation")
    del relative

    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.cuda().float().eval()
    adapter.requires_grad_(False)
    project = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").cuda().float().eval()
    project.requires_grad_(False)
    with torch.no_grad():
        target_tokens = torch.cat([project(adapter(block)).float() for block in source_f.split(128)])
        center = target_tokens.mean(0)
        variance = (target_tokens-center).square().mean((0, 2)).clamp_min(1e-6)
    _, members = group_index(torch.from_numpy(labels).cuda())
    steps = EPOCHS * members.shape[1] * math.ceil(len(members) / BATCH)
    out = root / "fits" / f"sub-{subject:02d}"
    out.mkdir(parents=True, exist_ok=True)
    proof = dict(subject=subject, split=split_hash, eeg=eeg, native=files.pin(native), visual=photo_hash,
                 teacher=target_digest, token_basis=basis_source, code=files.pin(__file__),
                 loss=files.pin(Path(inspect.getfile(losses))),
                 adapter=files.pin(PRIOR / "fusion_encoder/diffusion_pytorch_model.safetensors"),
                 projection=files.pin(PRIOR / "proj/diffusion_pytorch_model.safetensors"))
    histories = {}
    for arm in MODES:
        torch.manual_seed(SEED)
        ordering = torch.Generator(device="cuda").manual_seed(SEED)
        model = make_encoder("cuda").train()
        model.load_state_dict(initial)
        aux = nn.Sequential(nn.Linear(RANK, RANK), nn.LayerNorm(RANK)).cuda().train()
        params = list(model.parameters()) + list(aux.parameters())
        optimizer = torch.optim.AdamW(params, lr=8e-5, betas=(.9,.995), weight_decay=.05)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=5e-6)
        goals = cached[arm].cuda()
        batch_digest = hashlib.sha256()
        history = []
        for epoch in range(EPOCHS):
            total = source_f.new_zeros(2)
            for index in epoch_batches(members, ordering):
                batch_digest.update(index.cpu().numpy().astype("<i8").tobytes())
                optimizer.zero_grad(set_to_none=True)
                pred = project(adapter(model.backbone(waveforms[index]).float()))
                _, relative_tokens = token_parts(pred)
                centered = (relative_tokens - offset).flatten(1) @ axes / scale
                token_loss, auxiliary_loss = losses(pred, target_tokens[index], center, variance,
                                                     aux(centered), goals[index])
                (token_loss + AUXILIARY_WEIGHT * auxiliary_loss).backward()
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                total += torch.stack((token_loss.detach(), auxiliary_loss.detach())) * len(index)
            history.append((total / len(rows)).tolist())
            print("I-JEPA supervision", subject, arm, epoch + 1, history[-1], flush=True)
        path = out / f"{arm}.pt"
        files.verify()
        atomic_file(path, lambda stream: torch.save(model.state_dict(), stream))
        histories[arm] = dict(
            arm=arm, digest=files.pin(path), proof=proof, updates=steps, seed=SEED, photos=len(rows),
            batch_sha256=batch_digest.hexdigest(), auxiliary_weight=AUXILIARY_WEIGHT, losses=history,
            auxiliary_parameters=sum(x.numel() for x in aux.parameters()),
        )
    require(len({s["batch_sha256"] for s in histories.values()}) == 1, "The three training sequences must match")
    save_json(out / "training.json", histories)


@torch.inference_mode()
def infer(subject, root):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_object_foreground import predict_at_budget
    from paper_object_foreground_render import split_input
    from paper_oof_conditions import fitted_native
    from paper_render import generator_inputs, query_seed, tensor_fingerprint
    from paper_soft_text import wrong_donors
    from paper_token_evidence import project_tokens, render_arms

    budget = "dev-r4"
    files = PinnedFiles()
    raw, eeg, rows = split_input(subject, budget, files)
    model, native = fitted_native(subject, "cuda")
    fit_root = root / "fits" / f"sub-{subject:02d}"
    path = fit_root / "training.json"
    training_digest = files.pin(path)
    proof = json.loads(path.read_bytes())
    seed = [query_seed("dev", k) for k in range(len(rows))]
    repeats = [[0, 1, 2, 3]]
    native_digest = files.pin(native)
    target_hash = files.pin(root / "targets.pt")
    code_digest = files.pin(__file__)
    features = {"native-paper": predict_at_budget(model.backbone, raw, repeats)}
    for arm in MODES:
        checkpoint = fit_root / f"{arm}.pt"
        run = proof[arm]
        fit_proof = run["proof"]
        require(run["digest"] == files.pin(checkpoint) and run["arm"] == arm
                and fit_proof["subject"] == subject and fit_proof["native"] == native_digest
                and fit_proof["eeg"] == eeg and fit_proof["teacher"] == target_hash
                and fit_proof["split"] == files.pin(PREP / "paper_split.json") and fit_proof["code"] == code_digest,
                "Wrong EEG encoder source or training teacher")
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cuda"))
        features[arm] = predict_at_budget(model.backbone, raw, repeats)
    donors = wrong_donors(len(rows)).tolist()
    features["photo-map-wrong"] = features["photo-map"][donors]
    pipeline = load_generator(torch.float16, False)
    _, adapter, project, _, _ = pipeline
    native_order = list(range(len(rows)))
    variants = []
    for arm, source in features.items():
        tokens = project_tokens(source, adapter, project).half().cpu()
        variants.append((arm, tokens, dict(image_token_sha256=tensor_fingerprint(tokens),
                                          image_eeg_donor=donors if arm.endswith("-wrong") else native_order)))
    shared = dict(subject=subject, budget=budget, repeats=repeats, physical_eeg=eeg,
                  training_sha256=training_digest, generator_inputs=generator_inputs(files),
                  native_checkpoint_sha256=native_digest, teacher_sha256=target_hash, code_sha256=code_digest)
    out = root / "images" / budget / f"sub-{subject:02d}"
    render_arms(out, variants, rows, seed, shared, files, pipeline)


def score(subject, root):
    from paper_token_evidence import score_outputs

    path = root / "images" / "dev-r4" / f"sub-{subject:02d}"
    score_outputs([path / arm for arm in ARMS], "dev")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("fit", "infer", "score"))
    cli.add_argument("subject", choices=(1, 3), type=int)
    args = cli.parse_args()
    root = Path(os.environ["EEG_CROSSFIT_ROOT"])
    {"fit": fit, "infer": infer, "score": score}[args.phase](args.subject, root)
