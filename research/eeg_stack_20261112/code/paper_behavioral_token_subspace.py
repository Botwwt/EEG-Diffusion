"""Test whether behavior-derived image attributes locate more useful EEG token directions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ARMS = ("native-paper", "visual-66", "behavior-66", "behavior-66-wrong")
BUDGETS = ("dev-r4", "official-r4", "official-r80")


def attribute_axes(tokens, attributes):
    z = tokens - tokens.mean(0)
    values = (attributes - attributes.mean(0)) / attributes.std(0, unbiased=False).clamp_min(1e-5)
    return torch.linalg.qr(z.T @ values / len(z), mode="reduced").Q


def replace_coordinates(original, estimate, axes):
    return original + ((estimate - original) @ axes) @ axes.T


def root():
    return Path(os.environ["EEG_CROSSFIT_ROOT"])


def projection():
    from common import PRIOR
    from main.models_adapter import FusionEncoderModel, IPProjectionModel

    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.cuda().half().eval()
    project = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").cuda().half().eval()
    return adapter, project


def coordinates(relative, basis):
    return ((relative - basis["center"]).flatten(1) @ basis["axes"]) / basis["scale"]


def spaces(target, behavior, visual):
    centered = visual - visual.mean(0)
    _, axes = torch.linalg.eigh(centered.T @ centered / len(centered))
    return {"behavior-66": attribute_axes(target, behavior),
            "visual-66": attribute_axes(target, centered @ axes[:, -behavior.shape[1]:])}


@torch.inference_mode()
def fit(subject, attribute_file):
    from common import PREP, ROOT
    from paper_foreground_targets import eeg_matrix, estimate, regression
    from paper_identity_bank import class_sums
    from paper_oof_conditions import registered_rows
    from paper_render import generator_inputs, physical_eeg
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json
    from paper_token_evidence import project_tokens, token_parts

    files = PinnedFiles()
    rows, label_ids, folds = registered_rows(PREP)
    split = files.pin(PREP / "paper_split.json")
    raw, eeg_proof = physical_eeg(subject, "train", files)
    target_path = ROOT / "direct_token_encoder/training-targets.pt"
    condition_path = ROOT / "crossfit_generator" / f"conditions-sub-{subject:02d}.pt"
    image_path = ROOT / "crossfit_generator/object-identity/training-photo-dino.pt"
    attribute_receipt = attribute_file.with_suffix(".json")
    proof = dict(split=split, eeg=eeg_proof, token_target=files.pin(target_path), oof=files.pin(condition_path),
                 dino=files.pin(image_path), attributes=files.pin(attribute_file),
                 attribute_receipt=files.pin(attribute_receipt), code=files.pin(__file__),
                 projection=generator_inputs(files))
    source = json.loads(attribute_receipt.read_bytes())
    require(source["pack_sha256"] == proof["attributes"] and source["restricted_split_sha256"] == split
            and source["training_photo_index"] == files.pin(PREP / "train_ids.json"),
            "The attribute rows must match the registered training photos")
    with np.load(attribute_file, allow_pickle=False) as package:
        require(np.array_equal(package["rows"], rows), "Incorrect attribute order")
        attributes = torch.from_numpy(package["attributes"].copy()).cuda()
    teacher = torch.load(image_path, weights_only=True, map_location="cuda")
    target = torch.load(target_path, weights_only=True, map_location="cuda")
    eeg = torch.load(condition_path, weights_only=True, map_location="cpu")
    require(all(np.array_equal(item["rows"].cpu(), rows) for item in (teacher, target, eeg))
            and np.array_equal(teacher["concepts"][teacher["labels"]].cpu(), label_ids)
            and np.array_equal(eeg["folds"].numpy(), folds) and eeg["sources"]["eeg"] == eeg_proof
            and eeg["sources"]["split"] == target["sources"]["split"] == teacher["proof"]["split"] == split,
            "Behavior, DINO and real EEG registration changed")
    target_codes, basis = target["coordinates"], target["basis"]
    labels, dino = teacher["labels"], teacher["image_features"]
    require(attributes.shape == (len(rows), 66) and torch.isfinite(attributes).all(), "Expected 66 finite visual attributes")
    adapter, project = projection()
    encoded = eeg["out_of_fold"].cuda()
    native = []
    for block in encoded.split(240):
        native.append(coordinates(token_parts(project_tokens(block, adapter, project))[1], basis))
    native = torch.cat(native)
    input_features = torch.cat((encoded, eeg_matrix(raw[rows], [0, 1, 2, 3])), dim=1)
    sums = class_sums(dino, labels)
    gallery = F.normalize(sums)
    held_records = {}
    for fold in np.unique(folds):
        learn = folds != fold
        held = ~learn
        visual_head = regression(target_codes[learn], dino[learn])
        trans = spaces(target_codes[learn], attributes[learn], dino[learn])
        brain_head = regression(input_features[learn], target_codes[learn])
        predicted = estimate(brain_head, input_features[held])
        reference = F.normalize(sums[labels[held]] - dino[held])
        signals = {"native-paper": native[held]}
        signals.update({arm: replace_coordinates(native[held], predicted, axes) for arm, axes in trans.items()})
        outcomes = {}
        for arm, signal in signals.items():
            features = F.normalize(estimate(visual_head, signal))
            own = (features * reference).sum(1, keepdim=True)
            score = (features @ gallery.T).scatter(1, labels[held][:, None], own)
            rank = 1 + (score > own).sum(1)
            outcomes[arm] = dict(top1=int((rank == 1).sum()), top5=int((rank <= 5).sum()),
                                 own_cosine=float(own.mean()))
        held_records[str(fold)] = dict(n=int(held.sum()), scores=outcomes)
    axes = {name: value.cpu() for name, value in spaces(target_codes, attributes, dino).items()}
    predictor = {key: value.cpu() for key, value in regression(input_features, target_codes).items()}
    dest = root() / "fits" / f"sub-{subject:02d}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    files.verify()
    atomic_file(dest, lambda f: torch.save(dict(subject=subject, proof=proof, axes=axes, predictor=predictor), f))
    save_json(dest.with_suffix(".json"), dict(subject=subject, proof=proof, held=held_records, checkpoint=files.pin(dest)))
    print("Attribute subspaces, object folds", subject,
          {arm: sum(fold["scores"][arm]["top1"] for fold in held_records.values()) for arm in ARMS[:3]}, flush=True)


@torch.inference_mode()
def infer(subject, budgets, render_budgets):
    from common import ROOT
    from paper_foreground_targets import eeg_matrix, estimate
    from paper_object_foreground import predict_at_budget, text_targets
    from paper_object_foreground_render import split_input
    from paper_oof_conditions import fitted_native
    from paper_render import generator_inputs, groups, query_seed, tensor_fingerprint
    from paper_render_cache import PinnedFiles, atomic_file, require
    from paper_repeat_semantic_bridge import fit_head, predict
    from paper_soft_text import wrong_donors
    from paper_token_evidence import project_tokens, render_arms, token_parts

    files = PinnedFiles()
    training, labels, _, target_text, text_states, semantic_hash = text_targets(files)
    source = ROOT / "direct_token_encoder/training-targets.pt"
    source_hash = files.pin(source)
    target = torch.load(source, weights_only=True, map_location="cuda")
    require(np.array_equal(target["rows"].cpu(), training), "Target and text registration differ")
    mapper = fit_head(target["coordinates"], target_text.cuda(), labels.cuda())
    basis = target["basis"]
    native, checkpoint = fitted_native(subject, "cuda")
    fit_file = root() / "fits" / f"sub-{subject:02d}.pt"
    fit_hash = files.pin(fit_file)
    packet = torch.load(fit_file, weights_only=True, map_location="cuda")
    require(packet["subject"] == subject and packet["proof"]["token_target"] == source_hash, "Unexpected attribute fit")
    assets = generator_inputs(files)
    require(packet["proof"]["projection"] == assets, "Unexpected frozen visual generator")
    pipe = None
    if render_budgets:
        from paper_identity_render import load_generator

        pipe = load_generator(torch.float16, False)
        _, adapter, project, _, _ = pipe
    else:
        adapter, project = projection()
    code = files.pin(__file__)
    native_hash = files.pin(checkpoint)
    for budget in budgets:
        raw, eeg_proof, rows = split_input(subject, budget, files)
        repeats = groups(budget)
        visual = predict_at_budget(native.backbone, raw, repeats)
        tokens = project_tokens(visual, adapter, project)
        _, delta = token_parts(tokens)
        before = coordinates(delta, basis)
        inputs = torch.cat((visual, eeg_matrix(raw, sum(repeats, []))), dim=1)
        predicted = estimate(packet["predictor"], inputs)
        donors = wrong_donors(len(rows)).tolist()
        readouts = {"native-paper": before}
        readouts.update({arm: replace_coordinates(before, predicted, axes) for arm, axes in packet["axes"].items()})
        readouts["behavior-66-wrong"] = replace_coordinates(before, predicted[donors], packet["axes"]["behavior-66"])
        pooled = {arm: text_states["mean_pooled"] + predict(mapper, codes).cpu() @ text_states["pooled_basis"]
                  for arm, codes in readouts.items()}
        proof = dict(subject=subject, budget=budget, physical_eeg=eeg_proof, repeats=repeats, generator_inputs=assets,
                     native_checkpoint_sha256=native_hash, attribute_fit_sha256=fit_hash, token_target_sha256=source_hash,
                     semantic_basis_sha256=semantic_hash, code_sha256=code)
        out = root() / "images" / budget / f"sub-{subject:02d}"
        out.mkdir(parents=True, exist_ok=True)
        files.verify()
        atomic_file(out / "semantic-readout.pt", lambda f: torch.save(dict(proof=dict(proof, row=rows), pool=pooled, donors=donors), f))
        print("Attribute subspace readout", subject, budget, len(rows), flush=True)
        if budget in render_budgets:
            panel = budget.split("-")[0]
            index = list(range(len(rows)))
            seeds = [query_seed(panel, i) for i in index]
            variants = []
            for arm, codes in readouts.items():
                update = (((codes - before) * basis["scale"]) @ basis["axes"].T).reshape_as(tokens)
                _, exact = token_parts(update)
                condition = (tokens + exact).half().cpu()
                metadata = dict(image_token_sha256=tensor_fingerprint(condition), mean_eeg_donor=index,
                                attributes_eeg_donor=donors if arm == "behavior-66-wrong" else index)
                variants.append((arm, condition, metadata))
            render_arms(out, variants, rows, seeds, proof, files, pipe)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    phases = cli.add_subparsers(dest="phase", required=True)
    for name in ("fit", "infer", "score"):
        stage = phases.add_parser(name)
        stage.add_argument("subject", choices=(1, 2, 3, 4), type=int)
        if name == "fit":
            stage.add_argument("--attributes", type=Path, required=True)
        else:
            stage.add_argument("--budgets", nargs="+", choices=BUDGETS, default=BUDGETS)
            stage.add_argument("--render-budgets", nargs="+", choices=BUDGETS, default=[])
    args = cli.parse_args()
    if args.phase == "fit":
        fit(args.subject, args.attributes)
    else:
        if not set(args.render_budgets).issubset(args.budgets):
            cli.error("Render budgets must also be included in the EEG readout")
        if args.phase == "infer":
            infer(args.subject, args.budgets, args.render_budgets)
        else:
            from paper_object_foreground_render import score

            score(args.subject, args.budgets, args.render_budgets, ARMS)
