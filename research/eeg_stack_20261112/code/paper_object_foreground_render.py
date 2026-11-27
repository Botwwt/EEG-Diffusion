"""Replay object-specific EEG encoders with the same Brain-HiVE image tokens and noise."""
import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from paper_object_foreground import MODES, ObjectEncoder, load_encoder, predict_at_budget, root, text_targets


BUDGETS = ("dev-r4", "official-r4", "official-r80")
ARMS = ("native-paper", "soft-center", "uniform", "background", "background-wrong")


def split_input(subject, budget, files):
    from common import PREP
    from paper_render import physical_eeg
    from paper_token_evidence import development

    if budget == "dev-r4":
        _, rows = development(PREP, files)
        source, proof = physical_eeg(subject, "train", files)
        return source[rows], proof, rows

    source, proof = physical_eeg(subject, "test", files)
    return source, proof, list(range(len(source)))


@torch.inference_mode()
def infer(subject, budgets, render_budgets, arms):
    from common import PREP
    from paper_oof_conditions import fitted_native
    from paper_render import groups
    from paper_render_cache import PinnedFiles, atomic_file
    from paper_repeat_semantic_bridge import fit_head
    from paper_soft_text import wrong_donors

    files = PinnedFiles()
    rows, labels, _, target, states, basis = text_targets(files)
    native, native_path = fitted_native(subject, "cuda")
    native_digest = files.pin(native_path)
    models = {mode: load_encoder(subject, mode, basis, native_digest, files) for mode in MODES}
    photo_path = PREP / "F_train.npy"
    photo_source = files.pin(photo_path)
    target = target.cuda()
    training_photos = torch.from_numpy(np.load(photo_path, mmap_mode="r", allow_pickle=False)[rows].copy()).float().cuda()
    photo_head = fit_head(training_photos, target, labels.cuda())
    del training_photos
    prior = ObjectEncoder(native.backbone, 160).cuda().eval()
    prior.initialize_from_photo(photo_head, 1.)
    maximum = target.norm(dim=1).quantile(.95)
    pipeline = None
    for budget in budgets:
        raw, receipt, query_rows = split_input(subject, budget, files)
        repeats = groups(budget)
        z = {"photo-map": predict_at_budget(prior, raw, repeats)}
        for name, (model, scale, _) in models.items():
            z[name] = predict_at_budget(model, raw, repeats) * scale
        z = {name: (value / (value.norm(dim=1, keepdim=True) / maximum).clamp_min(1.)).cpu()
             for name, value in z.items()}
        donors = wrong_donors(len(raw)).tolist()
        z["background-wrong"] = z["background"][donors]
        z["soft-center"] = torch.zeros_like(z["background"])
        pooled = {name: states["mean_pooled"] + delta @ states["pooled_basis"] for name, delta in z.items()}
        out = root() / "images" / budget / f"sub-{subject:02d}"
        out.mkdir(parents=True, exist_ok=True)
        shared = dict(subject=subject, budget=budget, row=query_rows, repeats=repeats, physical_eeg=receipt,
                      native_checkpoint_sha256=native_digest, semantic_basis_sha256=basis, photo_features=photo_source,
                      encoder_sha256={name: value[2] for name, value in models.items()},
                      split_sha256=files.pin(PREP / "paper_split.json"), code_sha256=files.pin(Path(__file__)))
        files.verify()
        atomic_file(out / "semantic-readout.pt", lambda f: torch.save(dict(proof=shared, pool=pooled, donors=donors), f))
        print("Object EEG readout", subject, budget, len(query_rows), flush=True)
        if budget in render_budgets:
            if pipeline is None:
                from paper_identity_render import load_generator

                pipeline = load_generator(torch.float16, False)
            features = predict_at_budget(native.backbone, raw, repeats)
            generate(out, z, states, features, pipeline, shared, files, donors, arms)


def generate(out, codes, states, features, pipeline, shared, files, donors, arms):
    from paper_render import draw_images, generator_inputs, query_seed, tensor_fingerprint
    from paper_render_cache import render_transaction
    from paper_repeat_semantic_bridge import text_condition
    from paper_token_evidence import project_tokens

    pipe, adapter, project, empty, vae = pipeline
    tokens = project_tokens(features, adapter, project).half().cpu()
    assets = generator_inputs(files)
    panel = shared["budget"].split("-")[0]
    positions = list(range(len(tokens)))
    seeds = [query_seed(panel, i) for i in positions]
    token_hash = tensor_fingerprint(tokens)
    for name in arms:
        prompt = empty if name == "native-paper" else (*text_condition(states, codes[name]), empty[2])
        meta = dict(render_cache_version=2, arm=name, n=len(tokens), seeds=seeds, draws=1, unet_calls_per_draw=4,
                    image_token_sha256=token_hash, generator_inputs=assets, image_eeg_donor=positions,
                    text_eeg_donor=donors if name.endswith("-wrong") else
                    [] if name in ("native-paper", "soft-center") else positions, **shared)

        def sample(path, starts, complete):
            draw_images(path, pipe, vae, prompt, tokens, seeds, batch_starts=starts, batch_complete=complete)

        render_transaction(out / name, meta, sample, files.verify)
        print("Object EEG render", shared["subject"], shared["budget"], name, flush=True)


@torch.inference_mode()
def language_scorer(archive, files):
    from common import TURBO
    from paper_soft_text import TEMPLATE
    from score_official_objects import load_dictionary
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    path = archive / "runs/visual_fidelity_20261125_000818/official"
    dictionary = path / "dictionary.csv"
    ids, words, inverse, _, _ = load_dictionary(dictionary)
    folder = TURBO / "text_encoder_2"
    tokenizer = CLIPTokenizer.from_pretrained(str(TURBO / "tokenizer_2"), local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(str(folder), local_files_only=True, variant="fp16",
                                                         torch_dtype=torch.float16).cuda().eval()
    blocks = []
    for start in range(0, len(words), 64):
        prompts = [TEMPLATE.format(word) for word in words[start:start + 64]]
        batch = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        blocks.append(model(batch.input_ids.cuda()).text_embeds.float())
    source = dict(dictionary=files.pin(dictionary), text_model=files.pin(folder / "model.fp16.safetensors"))
    return SimpleNamespace(ids=ids, reverse={name: i for i, name in enumerate(ids)}, inverse=inverse,
                           anchors=F.normalize(torch.cat(blocks)), source=source, official=path)


def targets_for_budget(budget, rows, evaluator, files):
    from common import PREP
    from paper_render_cache import require

    if budget == "dev-r4":
        path = PREP / "train_ids.json"
        registration = json.loads((PREP / "paper_development.json").read_bytes())
        require(rows == registration["dev_rows"], "Wrong development query order")
        names = json.loads(path.read_bytes())["paths"]
        objects = [Path(names[row]).parent.name.split("_", 1)[1] for row in rows]
    else:
        path = evaluator.official / "originals_manifest.csv"
        with path.open() as stream:
            records = list(csv.DictReader(stream))
        require(rows == [int(record["eeg_row"]) for record in records], "Wrong official query order")
        objects = [record["unique_id"] for record in records]
    return objects, files.pin(path)


@torch.inference_mode()
def score(subject, budgets, render_budgets, arms):
    from paper_render_cache import PinnedFiles, require, save_json
    from paper_score import score as score_images
    from score_official_objects import topk_credit

    files = PinnedFiles()
    archive = Path(os.environ["ARCHIVE"])
    evaluate = language_scorer(archive, files)
    for budget in budgets:
        out = root() / "images" / budget / f"sub-{subject:02d}"
        path = out / "semantic-readout.pt"
        digest = files.pin(path)
        packet = torch.load(path, map_location="cpu", weights_only=True)
        require(packet["proof"]["subject"] == subject and packet["proof"]["budget"] == budget,
                "Readout packet has a different subject or repeat budget")
        label, reference = targets_for_budget(budget, packet["proof"]["row"], evaluate, files)
        actual = np.array([evaluate.reverse[name] for name in label])
        donor = np.asarray(packet["donors"])
        require(np.array_equal(np.sort(donor), np.arange(len(label))), "Wrong EEG control is not a permutation")
        changed = np.asarray(label)[donor] != label
        metrics = {}
        for arm, pooled in packet["pool"].items():
            scores = (F.normalize(pooled.cuda()) @ evaluate.anchors.T).cpu().numpy()[:, evaluate.inverse]
            correct = scores[np.arange(len(actual)), actual]
            ranks = 1 + (scores > correct[:, None]).sum(1)
            metrics[arm] = {f"top{k}": float(topk_credit(scores, k)[np.arange(len(actual)), actual].sum())
                            for k in (1, 5)}
            metrics[arm]["median_rank"] = float(np.median(ranks))
        files.verify()
        save_json(out / "semantic-readout-score.json", dict(scores=metrics, n=len(label),
                  wrong_objects=int(changed.sum()), proof=dict(evaluate.source, input=digest, reference=reference)))
        print("Object EEG readout scored", subject, budget, metrics, flush=True)
    del evaluate
    torch.cuda.empty_cache()
    for budget in render_budgets:
        for arm in arms:
            dest = root() / "images" / budget / f"sub-{subject:02d}" / arm
            if not (dest / "scores.json").is_file():
                score_images(dest, budget.split("-")[0], archive, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("infer", "score"))
    parser.add_argument("subject", type=int, choices=range(1, 11))
    parser.add_argument("--budgets", nargs="+", choices=BUDGETS, required=True)
    parser.add_argument("--render-budgets", nargs="+", choices=BUDGETS, default=[])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    if not set(args.render_budgets).issubset(args.budgets):
        parser.error("Render budgets must also be listed as input budgets")
    (infer if args.action == "infer" else score)(args.subject, args.budgets, args.render_budgets, args.arms)
