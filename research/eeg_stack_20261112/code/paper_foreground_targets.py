"""Compare EEG readouts supervised by aligned and displaced visual foreground features."""
import argparse
from inspect import getfile
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F


VIEWS = ("uniform", "foreground", "shifted")
ARMS = ("native-paper", "native-map", *VIEWS, "foreground-wrong")
BUDGETS = ("dev-r4", "official-r4", "official-r80")
RIDGE = .20


def patch_representations(patches, mask):
    batch, height, width = mask.shape
    if patches.shape[:2] != (batch, height * width):
        raise ValueError("The foreground weights must align to the visual patch grid")
    tokens = F.normalize(patches.float(), dim=-1)
    result = {"uniform": F.normalize(tokens.mean(1))}
    mask = mask.clamp(0., 1.)
    for name, region in (("foreground", mask), ("shifted", mask.roll((height // 2, width // 2), (1, 2)))):
        weighted = (tokens * region.flatten(1).unsqueeze(-1)).sum(1)
        present = region.flatten(1).sum(1, keepdim=True) > 1e-6
        result[name] = torch.where(present, F.normalize(weighted), result["uniform"])
    return result


def matched_conditions(native, views, donors, maximum):
    from paper_repeat_noise_readout import bound

    own = bound(.5 * (native + views["foreground"]), maximum)
    norm = own.norm(dim=1, keepdim=True)
    result = {"native-map": bound(native, maximum), "foreground": own}
    directions = {"uniform": views["uniform"], "shifted": views["shifted"],
                  "foreground-wrong": views["foreground"][donors]}
    for name, direction in directions.items():
        result[name] = F.normalize(native + direction) * norm
    return result


def output_root():
    return Path(os.environ["EEG_CROSSFIT_ROOT"])


@torch.inference_mode()
def prepare(mask_path):
    import json

    from common import PREP, ROOT
    from paper_dino import MODEL_FILES, TRAIN_ZIPS
    from paper_identity_bank import class_sums, object_logits, visual_model
    from paper_oof_conditions import registered_rows
    from paper_outline_routing import MASK_RECEIPTS
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_rgb_object import dino_pixels, read_photo, registered_photos

    files = PinnedFiles()
    rows, labels, members = registered_photos(PREP)
    require(np.array_equal(rows, registered_rows(PREP)[0]), "Different photograph support for EEG and masks")
    photo_index = PREP / "train_ids.json"
    eeg_index = PREP / "eeg_train_ids.json"
    photo_zip = Path(os.environ["ARCHIVE"]) / "assets/things_images_v254/training_images.zip"
    visual_path = ROOT / "models/paper-dinov2-small"
    proof = dict(archive=files.pin(photo_zip), split=files.pin(PREP / "paper_split.json"),
                 photo_index=files.pin(photo_index), eeg_index=files.pin(eeg_index),
                 masks=files.pin(mask_path), dino={key: files.pin(visual_path / key) for key in MODEL_FILES},
                 code=files.pin(__file__), preprocess=files.pin(getfile(dino_pixels)))
    require(proof["archive"] in TRAIN_ZIPS and proof["dino"] == MODEL_FILES
            and proof["masks"] == MASK_RECEIPTS["train"], "Unexpected RGB, mask or DINO release")
    require(json.loads(photo_index.read_bytes())["ids"] == json.loads(eeg_index.read_bytes())["ids"],
            "Training photos and EEG do not have the same order")
    with np.load(mask_path, allow_pickle=False) as panel:
        masks = panel["masks"]
        require(np.array_equal(panel["rows"], rows) and panel["valid"].all()
                and masks.shape == (len(rows), 160, 160) and masks.dtype == np.uint8,
                "The exact registered photos must supply the foreground masks")
    path = output_root() / "visual-targets.pt"
    if path.is_file():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        require(existing["proof"] == proof and np.array_equal(existing["rows"], rows), "Visual target inputs changed")
        return
    visual = visual_model(visual_path)
    blocks = {name: [] for name in (*VIEWS, "whole-cls")}
    cover = []
    with ZipFile(photo_zip) as archive:
        for start in range(0, len(rows), 32):
            selection = rows[start:start + 32]
            rgb = torch.from_numpy(np.stack([read_photo(archive, members[int(i)], 256) for i in selection]))
            rgb = rgb.permute(0, 3, 1, 2).cuda().float() / 255.
            features = visual(pixel_values=dino_pixels(rgb)).last_hidden_state.float()
            mask = torch.from_numpy(masks[start:start + len(selection)]).cuda().float().unsqueeze(1) / 255.
            mask = F.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False)
            mask = F.avg_pool2d(mask[:, :, 16:240, 16:240], 14).squeeze(1)
            values = patch_representations(features[:, 1:], mask)
            values["whole-cls"] = F.normalize(features[:, 0])
            for name, value in values.items():
                blocks[name].append(value.cpu())
            cover.append(mask.mean((1, 2)).cpu())
            if start % 1280 == 0:
                print("Visual targets", min(start + 32, len(rows)), len(rows), flush=True)
    views = {name: torch.cat(parts) for name, parts in blocks.items()}
    _, label = np.unique(labels, return_inverse=True)
    labels = torch.from_numpy(label).cuda()
    identity = {}
    for name, features in views.items():
        features = features.cuda()
        logits = object_logits(features, features, labels, class_sums(features, labels), "identity")
        identity[name] = int((logits.argmax(1) == labels).sum())
    cover = torch.cat(cover)
    record = dict(photos=len(rows), concepts=int(labels.unique().numel()), photos_recognized=identity,
                  foreground_mean_mass=float(cover.mean()), foreground_nonempty=int((cover > 0).sum()))
    files.verify()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_file(path, lambda stream: torch.save(dict(rows=torch.from_numpy(rows), views=views, proof=proof), stream))
    save_json(path.with_suffix(".json"), dict(record, proof=proof, digest=files.pin(path)))
    print("Visual target recognition", record, flush=True)


def eeg_bins(voltage):
    return voltage[..., :200].reshape(-1, 17, 40, 5).mean(-1).flatten(1)


def eeg_matrix(epochs, indices):
    parts = []
    for start in range(0, len(epochs), 128):
        raw = np.array(epochs[start:start + 128, indices], copy=True)
        parts.append(eeg_bins(torch.from_numpy(raw).cuda().float().mean(1)))
    return torch.cat(parts)


def regression(inputs, outputs):
    from paper_token_evidence import ridge_weights

    center = inputs.mean(0)
    scale = inputs.std(0, unbiased=False).clamp_min(1e-5)
    mean = outputs.mean(0)
    coefficients = ridge_weights((inputs - center) / scale, outputs - mean, penalty=RIDGE)
    return dict(center=center, scale=scale, mean=mean, coefficients=coefficients)


def estimate(model, signals):
    standardized = (signals - model["center"]) / model["scale"]
    return standardized @ model["coefficients"] + model["mean"]


@torch.inference_mode()
def fit(subject):
    from common import PREP
    from paper_identity_bank import class_sums
    from paper_object_foreground import text_targets
    from paper_oof_conditions import registered_rows
    from paper_render import physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_repeat_semantic_bridge import fit_head

    files = PinnedFiles()
    rows, labels, _, target, _, basis = text_targets(files)
    registered, _, folds = registered_rows(PREP)
    source = output_root() / "visual-targets.pt"
    raw, physical = physical_eeg(subject, "train", files)
    proof = dict(physical_eeg=physical, visual=files.pin(source), split=files.pin(PREP / "paper_split.json"),
                 semantic_basis=basis, code=files.pin(__file__))
    packet = torch.load(source, map_location="cpu", weights_only=True)
    require(np.array_equal(rows, registered) and np.array_equal(rows, packet["rows"])
            and packet["proof"]["split"] == proof["split"], "EEG, semantic targets and foreground photos differ")
    signals = eeg_matrix(raw[rows], [0, 1, 2, 3])
    labels = labels.cuda()
    target = target.cuda()
    visual = {name: features.cuda() for name, features in packet["views"].items()}
    reference = visual.pop("whole-cls")
    sums = class_sums(reference, labels)
    bank = F.normalize(sums)
    held = {}
    for fold in sorted(set(folds)):
        take = folds != fold
        labels_held = labels[~take]
        true = F.normalize(sums[labels_held] - reference[~take])
        result = {}
        for name, features in visual.items():
            head = regression(signals[take], features[take])
            bridge = regression(features[take], reference[take])
            query = F.normalize(estimate(bridge, estimate(head, signals[~take])))
            control = F.normalize(estimate(bridge, features[~take]))
            result[name] = {}
            for kind, predicted in (("eeg", query), ("photo", control)):
                own = (predicted * true).sum(1, keepdim=True)
                score = (predicted @ bank.T).scatter(1, labels_held[:, None], own)
                rank = 1 + (score > own).sum(1)
                result[name][kind] = dict(top1=int((rank == 1).sum()), top5=int((rank <= 5).sum()),
                                           own_cosine=float(own.mean()))
        held[str(fold)] = dict(n=int((~take).sum()), scores=result)
    models = {name: dict(eeg=regression(signals, features), language=fit_head(features, target, labels))
              for name, features in visual.items()}
    models = {name: {part: {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in state.items()}
                     for part, state in mode.items()} for name, mode in models.items()}
    dest = output_root() / "fits" / f"sub-{subject:02d}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    files.verify()
    atomic_file(dest, lambda stream: torch.save(dict(subject=subject, models=models, proof=proof), stream))
    save_json(dest.with_suffix(".json"), dict(subject=subject, held=held, proof=proof, digest=files.pin(dest)))
    print("Whole-concept EEG naming probe", subject,
          {view: {key: sum(x["scores"][view]["eeg"][key] for x in held.values()) for key in ("top1", "top5")}
           for view in VIEWS}, flush=True)


@torch.inference_mode()
def infer(subject, budgets, render_budgets):
    from common import PREP
    from paper_object_foreground import predict_at_budget, text_targets
    from paper_object_foreground_render import generate, split_input
    from paper_oof_conditions import fitted_native
    from paper_render import groups
    from paper_render_cache import PinnedFiles, atomic_file, require
    from paper_repeat_semantic_bridge import fit_head, predict
    from paper_soft_text import wrong_donors

    files = PinnedFiles()
    registered, labels, _, target, state, basis = text_targets(files)
    native, checkpoint = fitted_native(subject, "cuda")
    fit_path = output_root() / "fits" / f"sub-{subject:02d}.pt"
    fitted = files.pin(fit_path)
    packet = torch.load(fit_path, map_location="cuda", weights_only=True)
    split = files.pin(PREP / "paper_split.json")
    require(packet["subject"] == subject and packet["proof"]["split"] == split
            and packet["proof"]["semantic_basis"] == basis and set(packet["models"]) == set(VIEWS),
            "The foreground readout uses a different participant or training registration")
    photo_file = PREP / "F_train.npy"
    photo = torch.from_numpy(np.load(photo_file, mmap_mode="r", allow_pickle=False)[registered].copy())
    maximum = target.norm(dim=1).quantile(.95).item()
    prior = fit_head(photo.cuda().float(), target.cuda(), labels.cuda())
    checkpoint_hash, photo_hash, code_hash = files.pin(checkpoint), files.pin(photo_file), files.pin(__file__)
    pipeline = None
    for budget in budgets:
        raw, proof, query_rows = split_input(subject, budget, files)
        repeats = groups(budget)
        native_features = predict_at_budget(native.backbone, raw, repeats)
        signals = eeg_matrix(raw, sum(repeats, []))
        views = {name: predict(model["language"], estimate(model["eeg"], signals))
                 for name, model in packet["models"].items()}
        donors = wrong_donors(len(raw)).tolist()
        codes = {name: value.cpu() for name, value in
                 matched_conditions(predict(prior, native_features), views, donors, maximum).items()}
        vectors = {name: state["mean_pooled"] + code @ state["pooled_basis"] for name, code in codes.items()}
        shared = dict(subject=subject, budget=budget, row=query_rows, repeats=repeats, physical_eeg=proof,
                      native_checkpoint_sha256=checkpoint_hash, semantic_basis_sha256=basis, photo_features=photo_hash,
                      foreground_fit_sha256=fitted, split_sha256=split, code_sha256=code_hash)
        out = output_root() / "images" / budget / f"sub-{subject:02d}"
        out.mkdir(parents=True, exist_ok=True)
        files.verify()
        atomic_file(out / "semantic-readout.pt", lambda stream: torch.save(dict(proof=shared, pool=vectors, donors=donors), stream))
        print("Foreground EEG readout", subject, budget, len(codes["foreground"]), flush=True)
        if budget in render_budgets:
            if pipeline is None:
                from paper_identity_render import load_generator

                pipeline = load_generator(torch.float16, False)
            generate(out, codes, state, native_features, pipeline, shared, files, donors, ARMS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("prepare").add_argument("masks", type=Path)
    for action in ("fit", "infer", "score"):
        stage = actions.add_parser(action)
        stage.add_argument("subject", type=int, choices=range(1, 11))
        if action != "fit":
            stage.add_argument("--budgets", choices=BUDGETS, nargs="+", default=BUDGETS)
            stage.add_argument("--render-budgets", choices=BUDGETS, nargs="+", default=[])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.masks)
    elif args.action == "fit":
        fit(args.subject)
    else:
        if not set(args.render_budgets).issubset(args.budgets):
            parser.error("Render budgets must also be EEG readout budgets")
        if args.action == "infer":
            infer(args.subject, args.budgets, args.render_budgets)
        else:
            from paper_object_foreground_render import score

            score(args.subject, args.budgets, args.render_budgets, ARMS)
