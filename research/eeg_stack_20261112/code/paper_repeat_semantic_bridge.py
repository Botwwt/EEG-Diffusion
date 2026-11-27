"""Fit continuous SDXL states from concept-held neural features and render blind EEG queries."""
import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch


RIDGE = .20
WITHIN_WEIGHT = .65
GAIN_CAP = 3.
ARMS = ("native-paper", "soft-center", "photo-map", "crossfit-map",
        "repeat-agree", "crossfit-wrong", "agree-wrong")
WRONG_CONDITION = {"crossfit-wrong": "crossfit-map", "agree-wrong": "repeat-agree", "photo-wrong": "photo-map"}


def fit_head(features, targets, labels, prior=None):
    """Predict a concept's state while penalizing the changes across its training photos."""
    count = torch.bincount(labels, minlength=len(targets)).to(features)
    if len(count) != len(targets) or (count <= 1).any():
        raise ValueError("Every training concept needs repeated photographs")
    means = features.new_zeros(len(targets), features.shape[1])
    means.index_add_(0, labels, features)
    means /= count[:, None]
    center = means.mean(0)
    between = means - center
    within = features - means[labels]
    weight = (count[labels] * len(targets)).reciprocal().sqrt()[:, None]
    within = within * weight
    scale = (between.square().mean(0) + WITHIN_WEIGHT * within.square().sum(0)).sqrt().clamp_min(1e-5)
    classes = between / scale
    residual = within / scale
    system = classes.T @ classes / len(targets) + WITHIN_WEIGHT * (residual.T @ residual)
    system.diagonal().add_(RIDGE)
    rhs = classes.T @ targets / len(targets)
    if prior is not None:
        rhs += RIDGE * prior
    weights = torch.linalg.solve(system, rhs)
    signal = (classes @ weights).square().mean().item()
    gain = min(GAIN_CAP, math.sqrt(targets.square().mean().item() / max(signal, 1e-12)))
    return dict(center=center, scale=scale, weights=weights, gain=gain)


def predict(head, features):
    return ((features - head["center"]) / head["scale"]) @ head["weights"] * head["gain"]


def repeat_views(indices):
    if len(indices) not in (4, 80) or len(set(indices)) != len(indices):
        raise ValueError("Expected four or eighty distinct physical trials")
    count = 4 if len(indices) == 80 else 2
    return [indices[i::count] for i in range(count)]


def stable_coordinates(pooled, views, target_variance):
    noise = views.var(dim=0, unbiased=True) / len(views)
    signal = (views.mean(dim=0).square() - noise).clamp_min(0.)
    floor = (1e-6 * target_variance).clamp_min(1e-12)
    reliability = signal / torch.maximum(signal + noise, floor)
    return pooled * reliability


def text_condition(states, coordinates):
    hidden = coordinates @ states["hidden_basis"].flatten(1)
    pooled = coordinates @ states["pooled_basis"]
    return hidden.reshape(len(coordinates), *states["mean_hidden"].shape) + states["mean_hidden"], pooled + states["mean_pooled"]


def condition_coordinates(arm, predictions, donors):
    if arm == "soft-center":
        return torch.zeros_like(predictions["photo-map"])
    if arm in WRONG_CONDITION:
        return predictions[WRONG_CONDITION[arm]][donors]
    return predictions[arm]


def fit(subject, pins, states_path=None):
    from common import PREP
    from paper_oof_conditions import registered_rows
    from paper_soft_text import locations, registry

    rows, labels, names, source = registry()
    canonical, _, folds = registered_rows(PREP)
    if not np.array_equal(rows, canonical):
        raise ValueError("The continuous text and OOF feature splits differ")
    base = Path(os.environ["RAD_ROOT"]) / "crossfit_generator"
    features_path = base / f"conditions-sub-{subject:02d}.pt"
    states_path = locations() if states_path is None else states_path
    photo_path = PREP / "F_train.npy"
    proof = {key: pins.pin(p) for key, p in (("shadow", features_path), ("text", states_path), ("photo", photo_path))}
    oof = torch.load(features_path, map_location="cpu", weights_only=True)
    texts = torch.load(states_path, map_location="cpu", weights_only=True)
    if (oof["sources"]["subject"] != subject or oof["sources"]["images"] != proof["photo"]
            or not np.array_equal(oof["rows"].numpy(), rows) or not np.array_equal(oof["folds"].numpy(), folds)
            or texts["meta"]["inputs"] != source):
        raise ValueError("Stored neural or text supervision does not match this training split")
    target = texts["final_codes"].float().cuda()
    labels = labels.cuda()
    photos = torch.from_numpy(np.load(photo_path, mmap_mode="r", allow_pickle=False)[rows].copy()).float().cuda()
    shadow = oof["out_of_fold"].float().cuda()
    if photos.shape != shadow.shape or photos.shape != (len(rows), 1024) or len(target) != len(names):
        raise ValueError("Unexpected neural or visual feature shape")
    photo = fit_head(photos, target, labels)
    neural = fit_head(shadow, target, labels, prior=photo["weights"])
    proof.update(training_rows=len(rows), training_concepts=len(target),
                 native_training=oof["sources"]["native"], photo_gain=photo["gain"], crossfit_gain=neural["gain"],
                 ridge=RIDGE, within_class_weight=WITHIN_WEIGHT, gain_cap=GAIN_CAP)
    return photo, neural, texts["states"], target.square().mean(0), target.norm(dim=1).quantile(.95), proof


def pooled_features(encoder, raw, indices):
    views = repeat_views(indices)
    pooled, splits = [], []
    for start in range(0, len(raw), 20):
        batch = torch.from_numpy(np.array(raw[start:start + 20], copy=True)).cuda().float()
        pooled.append(encoder.backbone(batch[:, indices].mean(1)))
        splits.append(torch.stack([encoder.backbone(batch[:, selector].mean(1)) for selector in views]))
    return torch.cat(pooled), torch.cat(splits, dim=1)


def semantic_coordinates(features, views, heads, target_variance, maximum):
    photo, neural = heads
    crossfit = predict(neural, features)
    agreement = stable_coordinates(crossfit, predict(neural, views), target_variance)
    return {name: z / (z.norm(dim=1, keepdim=True) / maximum).clamp_min(1.) for name, z in
            (("photo-map", predict(photo, features)), ("crossfit-map", crossfit), ("repeat-agree", agreement))}


@torch.inference_mode()
def render(subject, budget, readout_only, states_path=None, arms=ARMS):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import fitted_native
    from paper_render import draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_render_cache import PinnedFiles, atomic_file, render_transaction
    from paper_soft_text import wrong_donors

    root = Path(os.environ["EEG_CROSSFIT_ROOT"])
    output = root / "images" / budget / f"sub-{subject:02d}"
    output.mkdir(parents=True, exist_ok=True)
    pins = PinnedFiles()
    photo, neural, states, variance, maximum, training = fit(subject, pins, states_path)
    eeg, receipt = physical_eeg(subject, "test", pins)
    encoder, path = fitted_native(subject, "cuda")
    native = pins.pin(path)
    if native != training["native_training"]:
        raise ValueError("Native model differs from the model in the neural feature cache")
    budgets = ("official-r4", "official-r80") if budget == "official-r80" else (budget,)
    packets, projections = {}, {}
    for count in budgets:
        repeats = groups(count)
        indices = sum(repeats, [])
        features, views = pooled_features(encoder, eeg, indices)
        codes = semantic_coordinates(features, views, (photo, neural), variance, maximum)
        packets[count] = {name: z.cpu() for name, z in codes.items()}
        projections[count] = {name: (states["mean_pooled"] + z @ states["pooled_basis"]).cpu()
                              for name, z in packets[count].items()}
    pins.verify()
    atomic_file(output / "prompt-readout.pt", lambda f: torch.save(dict(
        pools=projections, states=states["mean_pooled"], subject=subject, rows=list(range(len(eeg))),
        training=training, eeg=receipt, native=native), f))
    del encoder, views, photo, neural
    if readout_only:
        return
    generator_files = generator_inputs(pins)
    code = pins.pin(Path(__file__))
    pipe, adapter, projection, blank, vae = load_generator(torch.float16, False)
    tokens = torch.cat([projection(adapter(f.half())).cpu() for f in features.split(20)])
    rows = list(range(len(eeg)))
    seeds = [query_seed("official", i) for i in rows]
    donors = wrong_donors(len(eeg)).tolist()
    own = packets[budget]
    groups_seen = groups(budget)
    for arm in arms:
        prompts = blank if arm == "native-paper" else (*text_condition(states, condition_coordinates(arm, own, donors)), blank[2])
        manifest = dict(render_cache_version=2, arm=arm, subject=subject, budget=budget,
                        n=len(eeg), row=rows, seeds=seeds, repeats=groups_seen, draws=1,
                        unet_calls_per_draw=4, eeg=receipt, native_checkpoint_sha256=native,
                        image_tokens_donor=rows, text_donor=donors if arm in WRONG_CONDITION else (
                            [] if arm in ("native-paper", "soft-center") else rows),
                        text="empty" if arm == "native-paper" else "continuous SDXL states",
                        training=training, generator_inputs=generator_files, code=code,
                        official_split=pins.pin(PREP / "paper_split.json"))

        def generate(destination, starts, complete):
            draw_images(destination, pipe, vae, prompts, tokens, seeds, batch_starts=starts, batch_complete=complete)

        render_transaction(output / arm, manifest, generate, pins.verify)
        print("repeat semantic bridge render", subject, budget, arm, flush=True)


@torch.inference_mode()
def score_readout(output, official):
    import torch.nn.functional as F
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    from common import TURBO
    from paper_render_cache import PinnedFiles, save_json
    from paper_soft_text import TEMPLATE, wrong_donors
    from score_official_objects import load_dictionary, topk_credit

    packet = torch.load(output / "prompt-readout.pt", map_location="cpu", weights_only=True)
    dictionary = official / "dictionary.csv"
    ids, words, inverse, _, _ = load_dictionary(dictionary)
    manifest = official / "originals_manifest.csv"
    with manifest.open() as stream:
        references = list(csv.DictReader(stream))
    if packet["rows"] != [int(row["eeg_row"]) for row in references]:
        raise ValueError("Reference identities do not align with the frozen EEG predictions")
    lookup = {name: i for i, name in enumerate(ids)}
    targets = np.array([lookup[row["unique_id"]] for row in references])
    text_dir = TURBO / "text_encoder_2"
    tokenizer = CLIPTokenizer.from_pretrained(str(TURBO / "tokenizer_2"), local_files_only=True)
    variant = "fp16" if any("fp16" in p.name for p in text_dir.iterdir()) else None
    model = CLIPTextModelWithProjection.from_pretrained(str(text_dir), local_files_only=True, variant=variant,
                                                         torch_dtype=torch.float16).cuda().eval()
    bank = []
    for start in range(0, len(words), 64):
        prompts = [TEMPLATE.format(w) for w in words[start:start + 64]]
        batch = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        bank.append(model(batch.input_ids.cuda()).text_embeds.float())
    bank = F.normalize(torch.cat(bank))
    del model
    pins = PinnedFiles()
    proof = dict(prediction=pins.pin(output / "prompt-readout.pt"), dictionary=pins.pin(dictionary),
                 evaluation_ids=pins.pin(manifest), text_model={p.name: pins.pin(p) for p in text_dir.iterdir()
                                                                 if p.is_file() and ("fp16" in p.name or p.name == "config.json")})
    reports = {}
    for budget, original in packet["pools"].items():
        fixed = packet["states"].expand(len(references), -1)
        pools = dict(original, **{"soft-center": fixed, "crossfit-wrong": original["crossfit-map"][wrong_donors(len(fixed))],
                                  "agree-wrong": original["repeat-agree"][wrong_donors(len(fixed))]})
        reports[budget] = {}
        for arm, vector in pools.items():
            logits = (F.normalize(vector.to(bank.device)) @ bank.T).cpu().numpy()[:, inverse]
            actual = logits[np.arange(len(targets)), targets]
            ranks = 1 + (logits > actual[:, None]).sum(1)
            reports[budget][arm] = dict(top1=float(topk_credit(logits, 1)[np.arange(len(targets)), targets].sum()),
                                        top5=float(topk_credit(logits, 5)[np.arange(len(targets)), targets].sum()),
                                        median_rank=float(np.median(ranks)))
    pins.verify()
    save_json(output / "prompt-readout-score.json", dict(analysis=reports, proof=proof, n=len(references)))
    print("repeat semantic bridge readout", packet["subject"], reports, flush=True)


def score(subject, budget, readout_only, arms=ARMS):
    from paper_score import score as score_images

    output = Path(os.environ["EEG_CROSSFIT_ROOT"]) / "images" / budget / f"sub-{subject:02d}"
    archive = Path(os.environ["ARCHIVE"])
    official = archive / "runs/visual_fidelity_20261125_000818/official"
    score_readout(output, official)
    if not readout_only:
        for arm in arms:
            if not (output / arm / "scores.json").is_file():
                score_images(output / arm, "official", archive, output / arm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "score"))
    parser.add_argument("subject", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("budget", choices=("official-r4", "official-r80"))
    parser.add_argument("--readout-only", action="store_true")
    parser.add_argument("--space", type=Path, help="Alternative train-only continuous text basis")
    parser.add_argument("--arms", nargs="+", choices=(*ARMS, "photo-wrong"), default=ARMS)
    args = parser.parse_args()
    if args.action == "render":
        render(args.subject, args.budget, args.readout_only, args.space, args.arms)
    else:
        score(args.subject, args.budget, args.readout_only, args.arms)
