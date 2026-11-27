"""Measure held-out object information and render qualifying personal EEG encoders."""

import argparse
import inspect
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_repeat_semantics import ARMS, select_candidate
from paper_token_evidence import development, project_tokens, ridge_weights, token_parts


def object_scores(vectors, bank, leave_out, own_labels, photos):
    query = F.normalize(vectors)
    own = (query * leave_out).sum(1)
    object_sim = (query @ bank.T).scatter(1, own_labels[:, None], -torch.inf)
    rank = 1 + (object_sim >= own[:, None]).sum(1)
    same_concept = own_labels[:, None] == own_labels[None, :]
    images = query @ photos.T
    correct = torch.arange(len(query), device=query.device)
    return dict(full_object=rank == 1, full_object_top5=rank <= 5, object_rank=rank,
                other_photo_similarity=own,
                exact_photo=images.argmax(1) == correct,
                same_object_photo=images.masked_fill(~same_concept, -torch.inf).argmax(1) == correct)


def read_checkpoint(root, person, arm, files, native_hash):
    directory = root / f"fits/sub-{person:02d}"
    weight = directory / f"{arm}.pt"
    receipt_path = weight.with_suffix(".json")
    receipt_sha = files.pin(receipt_path)
    receipt = json.loads(receipt_path.read_bytes())
    digest = files.pin(weight)
    require(receipt["arm"] == arm and receipt["participant"] == person and receipt["checkpoint_sha256"] == digest
            and receipt["inputs"]["base"] == native_hash, "The encoder is not the matched fit of this participant")
    return weight, dict(weights=digest, receipt=receipt_sha, schedule=receipt["training_schedule_sha256"],
                        training_inputs=receipt["inputs"])


def decode(arm, root, person, eeg, files, native_hash):
    from paper_instance_residual import make_encoder

    path, source = read_checkpoint(root, person, arm, files, native_hash)
    model = make_encoder(torch.device("cuda")).float().eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    f = torch.cat([model.backbone(batch).float() for batch in eeg.split(20)])
    require(f.shape == (len(eeg), 1024) and torch.isfinite(f).all(), "Nonfinite fitted model output")
    return f, source


@torch.inference_mode()
def run_screen():
    from common import PREP, PRIOR, ROOT
    from main.models_adapter import FusionEncoderModel, IPProjectionModel
    from paper_dino import MODEL_FILES, TRAIN_ZIPS
    from paper_identity_bank import visual_model
    from paper_oof_conditions import registered_rows, study_root
    import paper_render
    from paper_rgb_object import dino_cls, read_photo

    root, files = study_root(), PinnedFiles()
    root.mkdir(parents=True, exist_ok=True)
    plan, rows = development(PREP, files)
    train, concepts, _ = registered_rows(PREP)
    label_file = PREP / "train_concept.npy"
    source_pixels = Path(os.environ["ARCHIVE"]) / "assets/things_images_v254/training_images.zip"
    dino_source = ROOT / "crossfit_generator/object-identity/training-photo-dino.pt"
    relative_basis = ROOT / "direct_token_encoder/training-targets.pt"
    photo_path, photo_index = PREP / "F_train.npy", PREP / "train_ids.json"
    model_dir = ROOT / "models/paper-dinov2-small"
    proof = dict(split=files.pin(PREP / "paper_split.json"), development=files.pin(PREP / "paper_development.json"),
                 labels=files.pin(label_file), reference_index=files.pin(photo_index), source_photos=files.pin(source_pixels),
                 visual_targets=files.pin(photo_path), dino_bank=files.pin(dino_source), relative_basis=files.pin(relative_basis),
                 code=files.pin(Path(__file__)), scorer={name: files.pin(model_dir / name) for name in MODEL_FILES},
                 subjects={})
    require(proof["source_photos"] in TRAIN_ZIPS and proof["scorer"] == MODEL_FILES, "Visual scorer or archive differs")
    bank = torch.load(dino_source, map_location="cuda", weights_only=True)
    basis_file = torch.load(relative_basis, map_location="cuda", weights_only=True)
    require(bank["proof"]["split"] == basis_file["sources"]["split"] == proof["split"]
            and bank["proof"]["dino"] == proof["scorer"]
            and basis_file["sources"]["photos"] == proof["visual_targets"]
            and torch.equal(bank["rows"], basis_file["rows"])
            and np.array_equal(bank["rows"].cpu().numpy(), train)
            and np.array_equal(bank["concepts"][bank["labels"]].cpu().numpy(), concepts),
            "The frozen evaluation maps must fit exactly the registered training photos")
    assets = paper_render.generator_inputs(files)
    require(all(assets[f"prior/{k}"] == v for k, v in basis_file["sources"]["projection"].items()),
            "The basis used a different feature projection")
    proof["generator_assets"] = assets
    indices = json.loads(photo_index.read_bytes())["paths"]
    target = np.load(label_file, allow_pickle=False)[rows]
    require(len(np.unique(target)) == 20 and not np.intersect1d(target, concepts).size,
            "Held objects and the fit may not overlap")
    # These references belong to the evaluator. The encoders only receive EEG.
    _, held_inverse, held_count = np.unique(target, return_inverse=True, return_counts=True)
    require(np.all(held_count == 10), "Each held object needs nine other visible photographs")
    registered = torch.from_numpy(np.argsort(held_inverse, kind="stable").reshape(-1, 10)).cuda()
    class_index = torch.from_numpy(held_inverse).cuda()
    dino = visual_model(model_dir)
    test_features = []
    with zipfile.ZipFile(source_pixels) as z:
        for i in range(0, len(rows), 20):
            pixels = np.stack([read_photo(z, indices[k], 512) for k in rows[i:i+20]])
            rgb = torch.from_numpy(pixels).permute(0, 3, 1, 2).cuda().float().div_(255)
            test_features.append(dino_cls(dino, rgb))
    truth = torch.cat(test_features)
    del dino
    visual_center = bank["image_features"].mean(0)
    sums = truth[registered].sum(1)
    positive = F.normalize(F.normalize(sums[class_index] - truth) - visual_center)
    gallery = F.normalize(torch.cat((F.normalize(sums), F.normalize(bank["sums"]))) - visual_center)
    true_photos = F.normalize(truth - visual_center)
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.cuda().half().eval()
    projector = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").cuda().half().eval()
    photo_bank = np.load(photo_path, mmap_mode="r", allow_pickle=False)
    training = torch.from_numpy(photo_bank[train].copy()).cuda().float()
    basis = basis_file["basis"]

    def relative(shown):
        value = token_parts(project_tokens(shown, adapter, projector))[1]
        return (value - basis["center"]).flatten(1) @ basis["axes"]

    image_center = training.mean(0)
    relative_visual = relative(training)
    relative_center = relative_visual.mean(0)
    dino_target = bank["image_features"] - visual_center
    map_fused = ridge_weights(training - image_center, dino_target, .02)
    map_relative = ridge_weights(relative_visual - relative_center, dino_target, .02)
    stats = {arm: {"relative": {}, "fused": {}} for arm in ("native-paper", *ARMS)}
    observations = {}

    def evaluate(arm, person, value):
        decoded = {
            "fused": (value - image_center) @ map_fused,
            "relative": (relative(value) - relative_center) @ map_relative,
        }
        for key, direction in decoded.items():
            measures = object_scores(direction, gallery, positive, class_index, true_photos)
            stats[arm][key][person] = {k: int(x.sum()) if x.dtype == torch.bool else float(x.mean(dtype=torch.float32))
                                     for k, x in measures.items()}
            for name, column in measures.items():
                observations[f"{person:02d}_{arm}_{key}_{name}"] = column.cpu().numpy()

    for person in plan["subjects"]:
        paper_render.RUNS = ROOT / "fits" / f"personal-{person:02d}"
        raw, eeg_proof = paper_render.physical_eeg(person, "train", files)
        selected = raw[rows]
        native, native_hash = paper_render.condition("native-paper", person, selected, [[0, 1, 2, 3]],
                                                     features_only=True, files=files)
        proof["subjects"][str(person)] = dict(physical=eeg_proof, native=native_hash, fits={})
        evaluate("native-paper", person, native.cuda().float())
        eeg = torch.from_numpy(selected.copy()).cuda().float().mean(1)
        for arm in ARMS:
            condition, meta = decode(arm, root, person, eeg, files, native_hash)
            require(meta["training_inputs"]["eeg"] == eeg_proof and meta["training_inputs"]["split"] == proof["split"]
                    and meta["training_inputs"]["photos"] == proof["visual_targets"],
                    "Matched model did not use the registered training data")
            proof["subjects"][str(person)]["fits"][arm] = meta
            evaluate(arm, person, condition)
        fits = list(proof["subjects"][str(person)]["fits"].values())
        require(len({f["schedule"] for f in fits}) == 1
                and all(f["training_inputs"] == fits[0]["training_inputs"] for f in fits),
                "Training intervention changed the schedule or registered inputs")
    selected = select_candidate({arm: value["relative"] for arm, value in stats.items()})
    files.verify()
    atomic_file(root / "development-screen.npz", lambda out: np.savez_compressed(
        out, rows=np.asarray(rows), object_ids=target, **observations))
    result = dict(selected=selected, stats=stats, proof=proof, photos_per_subject=len(rows),
                  held_object_count=20, comparison_object_count=gallery.shape[0],
                  observations_sha256=files.pin(root / "development-screen.npz"))
    save_json(root / "development-screen.json", result)
    print(json.dumps({"selected": selected, "relative": {k: v["relative"] for k, v in stats.items()}}), flush=True)


@torch.inference_mode()
def render(person, panel, diagnostic_arm=None):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import condition, donor_rows, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_token_evidence import render_arms, score_outputs

    root, files = study_root(), PinnedFiles()
    screen_file = root / "development-screen.json"
    score_digest = files.pin(screen_file)
    screen = json.loads(screen_file.read_bytes())
    winner = screen["selected"]
    if diagnostic_arm is None:
        stats = {k: v["relative"] for k, v in screen["stats"].items()}
        require(winner is not None and winner == select_candidate(stats), "The registered pre-render gate has not passed")
    else:
        require(panel == "dev" and diagnostic_arm in ARMS and diagnostic_arm != "photo",
                "An exploratory image check is only available on the development panel")
        winner = diagnostic_arm
    split_file = PREP / "paper_split.json"
    split = files.pin(split_file)
    require(screen["proof"]["split"] == split, "Selection used a different training split")
    dev_digest = None
    if panel == "dev":
        plan, rows = development(PREP, files)
        require(person in plan["subjects"], "Invalid development participant")
        dev_digest = files.pin(PREP / "paper_development.json")
    else:
        rows = json.loads(split_file.read_bytes())["official_rows"]
        require(rows == list(range(200)), "Official physical identities differ")
    repeats = groups(f"{panel}-r4")
    raw, physical = physical_eeg(person, "train" if panel == "dev" else "test", files)
    query = raw[rows]
    native, native_hash = condition("native-paper", person, query, repeats, features_only=True, files=files)
    assets = generator_inputs(files)
    require(assets == screen["proof"]["generator_assets"], "Frozen generator differs from the pre-render screen")
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    recipients = list(range(len(rows)))
    if panel == "dev":
        label_file = PREP / "train_concept.npy"
        files.pin(label_file)
        donor = donor_rows("token-reliable-wrong", panel, np.load(label_file, allow_pickle=False)[rows]).tolist()
    else:
        donor = np.roll(recipients, len(rows) // 2).tolist()
    variants = [("native-paper", project_tokens(native, adapter, projection).half().cpu(), dict(eeg_donor=recipients))]
    eeg = torch.from_numpy(query.copy()).cuda().float()[:, repeats[0]].mean(1)
    checkpoint_source = {}
    for arm in ("photo", winner):
        features, meta = decode(arm, root, person, eeg, files, native_hash)
        require(meta["training_inputs"]["split"] == split, "Ranked checkpoint is from another split")
        if panel == "dev":
            require(meta == screen["proof"]["subjects"][str(person)]["fits"][arm], "Ranked checkpoint changed after selection")
        checkpoint_source[arm] = meta
        tokens = project_tokens(features, adapter, projection).half().cpu()
        variants.append((arm, tokens, dict(eeg_donor=recipients, fitted_checkpoint=meta["weights"])))
        if arm == winner:
            variants.append((arm + "-wrong", tokens[donor], dict(eeg_donor=donor, fitted_checkpoint=meta["weights"])))
    matched = list(checkpoint_source.values())
    require(matched[0]["schedule"] == matched[1]["schedule"] and matched[0]["training_inputs"] == matched[1]["training_inputs"],
            "Fitted comparison used different training data or batch order")
    shared = dict(budget=f"{panel}-r4", subject=person, repeats=repeats, split_sha256=split,
                  development_split_sha256=dev_digest, physical_eeg=physical, native_checkpoint_sha256=native_hash,
                  generator_files_sha256=assets, selected_screen_sha256=score_digest, code_sha256=files.pin(Path(__file__)),
                  render_source_sha256=files.pin(Path(inspect.getfile(draw_images))),
                  exploratory=diagnostic_arm is not None)
    location = root / "images"
    if diagnostic_arm is not None:
        location = location / "diagnostic" / diagnostic_arm
    outputs = render_arms(location / panel / f"sub-{person:02d}", variants, rows,
                          [query_seed(panel, i) for i in recipients], shared, files, pipeline)
    del pipeline, adapter, projection
    torch.cuda.empty_cache()
    score_outputs(outputs, panel)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    tasks = cli.add_subparsers(dest="task", required=True)
    tasks.add_parser("screen")
    generate = tasks.add_parser("render")
    generate.add_argument("subject", type=int, choices=range(1, 11))
    generate.add_argument("--panel", choices=("dev", "official"), default="dev")
    generate.add_argument("--diagnostic-arm", choices=tuple(arm for arm in ARMS if arm != "photo"))
    args = cli.parse_args()
    if args.task == "screen":
        run_screen()
    else:
        render(args.subject, args.panel, args.diagnostic_arm)
