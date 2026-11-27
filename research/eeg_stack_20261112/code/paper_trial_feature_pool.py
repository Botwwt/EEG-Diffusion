"""Compare pooling EEG repeats before and after a learned temporal convolution."""

import argparse
import hashlib
import inspect
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_paired_encoder import BINS, SAMPLES_PER_BIN, SpatiotemporalEncoder
from paper_render_cache import PinnedFiles, atomic_file, require, save_json


FIT_ARMS = ("before", "after")
IMAGE_ARMS = ("before", "after", "after-wrong")
SEED, EPOCHS, BATCH = 20261127, 30, 256


def repeat_weights(trials):
    precision = trials.square().mean(-1, keepdim=True).clamp_min(1e-6).rsqrt()
    return precision / precision.sum(1, keepdim=True)


def pool_features(encode, trials, where, weights=None):
    """Give the two orders the same weighted input in the limit of a linear encoder."""
    n, repeats, _, _ = trials.shape
    require(where in FIT_ARMS, "Choose pooling before or after temporal encoding")
    if weights is None:
        weights = repeat_weights(trials)
    weighted = (repeats * weights) * trials
    if where == "before":
        return encode(weighted.mean(1))
    encoded = encode(weighted.flatten(0, 1))
    return encoded.unflatten(0, (n, repeats)).mean(1)


class TrialFeatureEncoder(SpatiotemporalEncoder):
    def __init__(self, mean, scale, visual_center, visual_scale):
        super().__init__(mean, scale)
        self.register_buffer("visual_center", visual_center)
        self.register_buffer("visual_scale", visual_scale)
        self.head = nn.Linear(self.hidden, len(visual_center))

    def local_features(self, eeg):
        local = self.temporal(self.montage[0] @ eeg)
        return local + self.context(local)

    def forward(self, trials, weights, where):
        standardized = (trials - self.input_mean[0]) / self.input_scale[0]
        local = pool_features(self.local_features, standardized, where, weights)
        hidden = self.pool(local)
        return self.visual_center + self.visual_scale * self.head(hidden + self.mix(hidden)).float()


def batches(raw, rows):
    for selected in np.array_split(rows, math.ceil(len(rows) / 128)):
        x = torch.from_numpy(np.array(raw[selected, :4], copy=True)).cuda().float()
        weights = repeat_weights(x)
        durations = x[..., :BINS * SAMPLES_PER_BIN].unflatten(-1, (BINS, SAMPLES_PER_BIN)).mean(-1)
        yield durations, weights


def fit(subject, root):
    from common import PREP
    from paper_oof_conditions import registered_rows
    from paper_render import physical_eeg
    from paper_repeat_semantics import photo_loss

    files = PinnedFiles()
    rows, _, _ = registered_rows(PREP)
    measurements, source = physical_eeg(subject, "train", files)
    features = PREP / "F_train.npy"
    visual = torch.from_numpy(np.load(features, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    require(visual.shape == (len(rows), 1024) and torch.isfinite(visual).all(), "Invalid frozen photo targets")
    proof = dict(subject=subject, eeg=source, split=files.pin(PREP / "paper_split.json"),
                 photos=files.pin(features), code=files.pin(__file__),
                 encoder=files.pin(Path(inspect.getfile(SpatiotemporalEncoder))))
    blocks = list(batches(measurements, rows))
    eeg = torch.cat([e for e, _ in blocks])
    weights = torch.cat([w for _, w in blocks])
    del blocks
    precision = (eeg * weights).sum(1)
    center, scale = precision.mean(0, keepdim=True), precision.std(0, keepdim=True).clamp_min(.02)
    visual_center = visual.mean(0)
    variance = (visual - visual_center).square().mean()
    require(variance > 0, "Photo targets have no variation")
    visual_scale = variance.sqrt()
    out = root / "fits" / f"sub-{subject:02d}"
    out.mkdir(parents=True, exist_ok=True)
    total = EPOCHS * math.ceil(len(eeg) / BATCH)
    receipts = {}
    for arm in FIT_ARMS:
        torch.manual_seed(SEED + subject)
        rng = np.random.default_rng(SEED + subject)
        model = TrialFeatureEncoder(center, scale, visual_center, visual_scale).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0004, betas=(.9, .995), weight_decay=.05)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total, eta_min=.00002)
        order_hash = hashlib.sha256()
        history = []
        for epoch in range(EPOCHS):
            order = rng.permutation(len(eeg)).astype("<i8")
            order_hash.update(order.tobytes())
            cumulative = visual.new_zeros(())
            for block in torch.from_numpy(order).cuda().split(BATCH):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(eeg[block], weights[block], arm)
                    loss = photo_loss(prediction, visual[block], visual_center, variance)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                cumulative += loss.detach() * len(block)
            history.append(float(cumulative / len(eeg)))
            if (epoch + 1) % 5 == 0:
                print("Trial pooling", subject, arm, epoch + 1, history[-1], flush=True)
        path = out / f"{arm}.pt"
        metadata = dict(arm=arm, proof=proof, batch_order_sha256=order_hash.hexdigest(), updates=total, history=history,
                        parameters=sum(x.numel() for x in model.parameters()))
        files.verify()
        atomic_file(path, lambda f: torch.save(dict(metadata=metadata, state=model.state_dict()), f))
        receipts[arm] = dict(metadata, checkpoint=files.pin(path))
    require(receipts["before"]["batch_order_sha256"] == receipts["after"]["batch_order_sha256"],
            "Training queries differ between arms")
    save_json(out / "training.json", receipts)


@torch.inference_mode()
def infer(subject, root):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_object_foreground_render import split_input
    from paper_render import generator_inputs, query_seed, tensor_fingerprint
    from paper_soft_text import wrong_donors
    from paper_token_evidence import project_tokens, render_arms

    files = PinnedFiles()
    raw, receipt, rows = split_input(subject, "dev-r4", files)
    # Split input already returns exactly the 200 registered rows in stimulus order.
    own = list(range(len(rows)))
    donors = wrong_donors(len(rows)).tolist()
    readings = list(batches(raw, own))
    features, checkpoints = {}, {}
    participant = root / "fits" / f"sub-{subject:02d}"
    sources = {}
    for arm in FIT_ARMS:
        path = participant / f"{arm}.pt"
        checkpoints[arm] = files.pin(path)
        pack = torch.load(path, weights_only=True, map_location="cuda")
        state, metadata = pack["state"], pack["metadata"]
        require(metadata["arm"] == arm and metadata["proof"]["subject"] == subject
                and metadata["proof"]["eeg"] == receipt
                and metadata["proof"]["split"] == files.pin(PREP / "paper_split.json")
                and metadata["proof"]["photos"] == files.pin(PREP / "F_train.npy")
                and metadata["proof"]["code"] == files.pin(__file__)
                and metadata["proof"]["encoder"] == files.pin(Path(inspect.getfile(SpatiotemporalEncoder))),
                "Unexpected experimental encoder checkpoint")
        sources[arm] = metadata
        model = TrialFeatureEncoder(state["input_mean"], state["input_scale"], state["visual_center"],
                                    state["visual_scale"]).cuda().eval()
        model.load_state_dict(state)
        features[arm] = torch.cat([model(eeg, weights, arm) for eeg, weights in readings])
    order = sources["before"]["batch_order_sha256"]
    require(order == sources["after"]["batch_order_sha256"], "Training cohort must be matched")
    features["after-wrong"] = features["after"][donors]
    assets = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    variants = []
    for arm, features_for_arm in features.items():
        tokens = project_tokens(features_for_arm, adapter, projection).half().cpu()
        variants.append((arm, tokens, dict(image_token_sha256=tensor_fingerprint(tokens),
                                          image_eeg_donor=donors if arm == "after-wrong" else own)))
    proof = dict(subject=subject, budget="dev-r4", repeats=[[0, 1, 2, 3]], physical_eeg=receipt,
                 generator_inputs=assets, train_checkpoints=checkpoints, train_order=order,
                 code_sha256=files.pin(__file__))
    out = root / "images" / "dev-r4" / f"sub-{subject:02d}"
    render_arms(out, variants, rows, [query_seed("dev", i) for i in own], proof, files, pipeline)


def score(subject, root):
    from paper_token_evidence import score_outputs
    out = root / "images" / "dev-r4" / f"sub-{subject:02d}"
    score_outputs([out / arm for arm in IMAGE_ARMS], "dev")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("fit", "infer", "score"))
    cli.add_argument("subject", type=int, choices=(1, 3))
    args = cli.parse_args()
    root = Path(os.environ["EEG_CROSSFIT_ROOT"])
    {"fit": fit, "infer": infer, "score": score}[args.phase](args.subject, root)
