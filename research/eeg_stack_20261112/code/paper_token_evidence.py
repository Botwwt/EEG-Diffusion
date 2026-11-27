"""Isolate the global value and the relative content of Brain-HiVE's image tokens."""
import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, render_transaction, require, save_json

ORACLE_ARMS = ("own-mean-other-relative", "other-mean-own-relative", "other-visual")
ARMS = ("centered-native", "photo-target", "object-target", "object-target-wrong",
        "spread-control", "photo-spread", "photo-spread-wrong")
COMPONENTS = ("mean", "relative")
PENALTY = .1


def token_parts(tokens):
    values = tokens.float()
    center = values.mean(dim=1, keepdim=True)
    return center, values - center


def mix_tokens(mean, relative, dtype):
    return (mean + relative).to(dtype=dtype)


def adjust_part(native, input_center, visual_center, predicted):
    native_offset = (native - input_center).flatten(1)
    direction = F.normalize(predicted.flatten(1), dim=1)
    radius = native_offset.norm(dim=1, keepdim=True)
    offset = .5 * (native_offset + direction * radius)
    return visual_center + offset.reshape_as(native)


def ridge_weights(inputs, targets, penalty=PENALTY):
    response = targets.flatten(1)
    require(inputs.ndim == 2 or inputs.ndim == 3, "Expected native features and optional viewer axis")
    flat = inputs.reshape(-1, inputs.shape[-1])
    corresponding = inputs.mean(0) if inputs.ndim == 3 else inputs
    require(len(corresponding) == len(response), "Native and visual photographs have different rows")
    cov = (flat.T @ flat) / len(flat)
    cov.diagonal().add_(penalty * cov.diagonal().mean())
    return torch.linalg.solve(cov, corresponding.T @ response / len(response))


def photo_targets(tokens, labels):
    unique, reverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    require(len(unique) == 1323 and np.all(sizes == 10), "Training needs ten photos of each registered object")
    ordered = torch.from_numpy(np.argsort(reverse, kind="stable").reshape(-1, 10)).to(tokens.device)
    own = tokens.mean(0, keepdim=True)
    targets = tokens - own
    object_sum = targets[ordered].sum(1)
    mean_excluding_photo = (object_sum[reverse] - targets) / 9.
    return own, targets, mean_excluding_photo


def project_tokens(x, adapter, projection):
    return torch.cat([projection(adapter(block.to(device="cuda", dtype=torch.float16))).float()
                      for block in x.split(20)])


def development(prep, files):
    path = prep / "paper_development.json"
    split = prep / "paper_split.json"
    files.pin(path)
    files.pin(split)
    query = json.loads(path.read_bytes())
    row = query["dev_rows"]
    training = json.loads(split.read_bytes())["train_rows"]
    require(len(row) == len(set(row)) == 200 and not np.intersect1d(row, training).size,
            "The entire development panel must be excluded from fitting")
    return query, row


def render_arms(root, variants, rows, seeds, shared, files, pipeline):
    from paper_render import draw_images

    pipe, _, _, prompts, vae = pipeline
    saved = []
    for arm, tokens, per_arm in variants:
        location = root / arm
        manifest = dict(render_cache_version=2, arm=arm, n=len(rows), row=rows, seeds=seeds, draws=1,
                        unet_calls_per_draw=4, prompt="", **shared, **per_arm)
        def generate(out, starts, complete):
            draw_images(out, pipe, vae, prompts, tokens, seeds, batch_starts=starts, batch_complete=complete)
        render_transaction(location, manifest, generate, files.verify)
        saved.append(location)
        print("token evidence render", location.name, len(rows), flush=True)
    # The SDXL pipe still belongs to the caller. Return all output directories for later scoring.
    return saved


def score_outputs(destinations, panel):
    from paper_score import score
    for folder in destinations:
        if not (folder / "scores.json").is_file():
            score(folder, panel, Path(os.environ["ARCHIVE"]), folder)


@torch.inference_mode()
def oracle():
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import donor_rows, draw_images, generator_inputs, query_seed

    files = PinnedFiles()
    _, rows = development(PREP, files)
    photo_file = PREP / "F_train.npy"
    label_file = PREP / "train_concept.npy"
    proof = {p.name:files.pin(p) for p in (photo_file, label_file, PREP / "paper_development.json", PREP / "paper_split.json")}
    labels = np.load(label_file, allow_pickle=False)[rows]
    donor = donor_rows("token-reliable-wrong", "dev", labels)
    own = np.arange(len(rows))
    gallery = np.load(photo_file, mmap_mode="r", allow_pickle=False)
    visual = torch.from_numpy(gallery[rows].copy())
    seeds = [query_seed("dev", k) for k in own]
    assets = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    tokens = project_tokens(visual, adapter, projection).cpu()
    center, relative = token_parts(tokens)
    sources = [(own, donor), (donor, own), (donor, donor)]
    variants = [(arm, tokens[means].half() if arm == "other-visual" else mix_tokens(center[means], relative[dims], torch.float16),
                 dict(mean_donor=means.tolist(), relative_donor=dims.tolist()))
                for arm, (means,dims) in zip(ORACLE_ARMS, sources, strict=True)]
    shared = dict(diagnostic_only=True, generator_files_sha256=assets, source_sha256=proof,
                  code_sha256=files.pin(Path(__file__)), render_source_sha256=files.pin(Path(inspect.getfile(draw_images))))
    output = render_arms(study_root() / "oracle", variants, rows, seeds, shared, files, pipeline)
    del pipeline, adapter, projection
    torch.cuda.empty_cache()
    score_outputs(output, "dev")


@torch.inference_mode()
def fit(component):
    from common import PREP, PRIOR, ROOT
    from main.models_adapter import FusionEncoderModel, IPProjectionModel
    from paper_oof_conditions import SOURCES, registered_rows, study_root

    files = PinnedFiles()
    train, labels, folds = registered_rows(PREP)
    split = files.pin(PREP / "paper_split.json")
    photos = PREP / "F_train.npy"
    asset_paths = [p for sub in ("fusion_encoder", "proj") for p in sorted((PRIOR/sub).rglob("*"))
                   if p.is_file() and not any(part.startswith(".") for part in p.relative_to(PRIOR).parts)]
    proof = dict(split=split, photos=files.pin(photos), labels=files.pin(PREP / "train_concept.npy"),
                 predictor={}, projection={str(p.relative_to(PRIOR)):files.pin(p) for p in asset_paths},
                 code=files.pin(Path(__file__)))
    fit_dir = study_root()
    fit_dir.mkdir(parents=True, exist_ok=True)
    dest = fit_dir / f"weights-{component}.pt"
    native = []
    for person in SOURCES:
        path = ROOT / "crossfit_generator" / f"conditions-sub-{person:02d}.pt"
        proof["predictor"][str(person)] = files.pin(path)
        source = torch.load(path,map_location="cpu",weights_only=True)
        require(source["sources"]["subject"] == person and source["sources"]["split"] == split
                and source["sources"]["images"] == proof["photos"]
                and np.array_equal(source["rows"].numpy(),train)
                and np.array_equal(source["folds"].numpy(),folds)
                and set(part["fold"] for part in source["split_audits"]) == set(int(k) for k in folds),
                "Out-of-concept fit registration differs")
        native.append(source["out_of_fold"])
    if dest.exists():
        saved = torch.load(dest,map_location="cpu",weights_only=True)
        require(saved["proof"] == proof and saved["component"] == component, "The prior checkpoint has different inputs")
        return
    visual = torch.from_numpy(np.load(photos, mmap_mode="r", allow_pickle=False)[train].copy())
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR),subfolder="fusion_encoder").adapter.cuda().half().eval()
    projection = IPProjectionModel.from_pretrained(str(PRIOR),subfolder="proj").cuda().half().eval()
    part = COMPONENTS.index(component)
    target_tokens = token_parts(project_tokens(visual, adapter, projection))[part]
    visual_center, photo, objects = photo_targets(target_tokens, labels)
    observed = torch.stack(native).cuda().float()
    require(observed.shape == (len(SOURCES),len(train),1024) and torch.isfinite(observed).all(),
            "Unexpected registered R4 prediction shape")
    raw_center = observed.mean(dim=(0, 1))
    native_parts = torch.stack([token_parts(project_tokens(view,adapter,projection))[part]
                                for view in observed])
    input_center = native_parts.mean(dim=(0,1),keepdim=False).unsqueeze(0)
    observed -= raw_center
    params = {
        "photo-target":ridge_weights(observed, photo),
        "object-target":ridge_weights(observed, objects),
    }
    offsets = native_parts.flatten(0,1)-input_center
    expected = photo.flatten(1).norm()/np.sqrt(len(photo))
    direction = (observed.flatten(0,1)@params["photo-target"]).reshape_as(offsets)
    calibrated = adjust_part(offsets,torch.zeros_like(input_center),torch.zeros_like(visual_center),direction)
    scalars = {
        "spread-control":float(expected/(offsets.flatten(1).norm()/np.sqrt(len(offsets)))),
        "photo-spread":float(expected/(calibrated.flatten(1).norm()/np.sqrt(len(calibrated)))),
    }
    require(all(np.isfinite(gain) and gain>0 for gain in scalars.values()), "Unusable fitted component spread")
    state = dict(raw_center=raw_center.cpu(), input_center=input_center.cpu(), visual_center=visual_center.cpu(),
                 weights={name:w.cpu() for name,w in params.items()}, gains=scalars)
    files.verify()
    summary = dict(component=component, proof=proof, penalty=PENALTY, observations=int(np.prod(observed.shape[:2])),
                   shape=list(target_tokens.shape), ratio=float(photo.norm()/objects.norm()), gains=scalars)
    atomic_file(dest, lambda stream:torch.save(dict(summary,state=state),stream))
    save_json(dest.with_suffix(".json"), summary)
    print("token evidence fit",component,summary["observations"],summary["shape"],flush=True)


@torch.inference_mode()
def render(subject, component, panel="dev", arms=ARMS):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import condition, donor_rows, draw_images, generator_inputs, groups, physical_eeg, query_seed

    files = PinnedFiles()
    split_file = PREP / "paper_split.json"
    split = files.pin(split_file)
    registration = json.loads(split_file.read_bytes())
    dev_digest = None
    if panel == "dev":
        plan, rows = development(PREP, files)
        require(subject in plan["subjects"], "Use the locked development participants")
        dev_digest = files.pin(PREP / "paper_development.json")
    else:
        rows = registration[f"{panel}_rows"]
        require(subject not in (1, 3), "Use an independent confirmation participant")
        require(len(rows) == len(set(rows)) == (500 if panel == "fresh" else 200), "Registration changed")
        require(panel == "official" or not np.intersect1d(rows, registration["train_rows"]).size,
                "Fresh images must be absent from fitting")
    root = study_root()
    model_file = root / f"weights-{component}.pt"
    model_sha = files.pin(model_file)
    snapshot = torch.load(model_file,map_location="cuda",weights_only=True)
    require(snapshot["proof"]["split"] == split and snapshot["component"] == component, "Mismatched token fit")
    state = snapshot["state"]
    exchange = np.arange(len(rows))
    controls = any(arm.endswith("-wrong") for arm in arms)
    if controls:
        if panel == "official":
            exchange = np.roll(exchange, len(rows)//2)
        else:
            label_file = PREP / "train_concept.npy"
            files.pin(label_file)
            exchange = donor_rows("token-reliable-wrong",panel,np.load(label_file,allow_pickle=False)[rows])
    repeats = groups(panel+"-r4")
    raw, physical = physical_eeg(subject,"test" if panel == "official" else "train",files)
    native, checkpoint = condition("native-paper",subject,raw[rows],repeats,features_only=True,files=files)
    assets = generator_inputs(files)
    for name, digest in snapshot["proof"]["projection"].items():
        require(assets[f"prior/{name}"] == digest, "Frozen projection changed after fitting")
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    native = native.cuda()
    tokens = project_tokens(native,adapter,projection)
    pieces = token_parts(tokens)
    index = COMPONENTS.index(component)
    old = pieces[index]
    centered = native - state["raw_center"]
    shared = dict(budget=panel+"-r4", subject=subject, repeats=repeats, component=component,
                  split_sha256=split, development_split_sha256=dev_digest, physical_eeg=physical,
                  native_checkpoint_sha256=checkpoint, checkpoint_sha256=model_sha, generator_files_sha256=assets,
                  code_sha256=files.pin(Path(__file__)), render_source_sha256=files.pin(Path(inspect.getfile(draw_images))))
    variants = []
    for arm in arms:
        donor = exchange if arm.endswith("-wrong") else np.arange(len(rows))
        name = arm.removesuffix("-wrong")
        gain = state["gains"].get(name,1.)
        if name in ("centered-native","spread-control"):
            offset = old-state["input_center"]
        else:
            key = "photo-target" if name=="photo-spread" else name
            prediction = (centered @ state["weights"][key]).reshape_as(old)
            offset = adjust_part(old,state["input_center"],torch.zeros_like(state["visual_center"]),prediction)
        revised = state["visual_center"] + gain*offset
        combined = list(pieces)
        combined[index] = revised[donor] if component == "mean" else token_parts(revised[donor])[1]
        variants.append((arm, mix_tokens(*combined, torch.float16).cpu(),
                         dict(edited_component_donor=donor.tolist(), untouched_component_donor=list(range(len(rows))), component_gain=gain)))
    seeds = [query_seed(panel,i) for i in range(len(rows))]
    output = render_arms(root / "images" / component / panel / f"sub-{subject:02d}", variants, rows, seeds, shared, files, pipeline)
    del pipeline, adapter, projection
    torch.cuda.empty_cache()
    score_outputs(output,panel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    tasks.add_parser("oracle")
    task = tasks.add_parser("fit")
    task.add_argument("component", choices=COMPONENTS)
    task = tasks.add_parser("render")
    task.add_argument("component", choices=COMPONENTS)
    task.add_argument("subject", type=int, choices=range(1,11))
    task.add_argument("--panel",choices=("dev","fresh","official"),default="dev")
    task.add_argument("--arms",nargs="+",choices=ARMS,default=ARMS)
    opts = parser.parse_args()
    if opts.task == "oracle":
        oracle()
    elif opts.task == "fit":
        fit(opts.component)
    else:
        render(opts.subject, opts.component, opts.panel, opts.arms)
