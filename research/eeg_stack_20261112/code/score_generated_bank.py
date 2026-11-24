"""Encode the generated candidate bank independently and score the frozen per-query choices."""

import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from render_within_photo_ip import PANELS, check, read, save, sha
from score_object_panel import score_object_panel
from score_official_objects import load_dictionary, load_tokenizer, text_bank, topk_credit, unit
from score_personal_photo_layout import original_records
from score_reconstruction_jpegs import CLIP_SHA, originals, raw_features, registered
from score_within_photo_ip import estimate

ARMS = ("own", "uniform", "wrong", "fixed0")


def jpeg(path):
    with Image.open(path) as im:
        check(im.size == (512, 512), "Wrong candidate resolution")
        return im.convert("RGB")


def encode(args):
    paths = sorted(args.bank.glob("**/*.jpg"))
    check(
        len(paths) == 960, "Expected eight completed candidates for each of 120 object/fold cells"
    )
    hashes = np.asarray([sha(f) for f in paths])
    regs = {p: registered(args.references, p) for p in PANELS}
    extra = {}
    if args.backend == "clip":
        import torch
        from eval_instance_starts import load_visual_scorer

        torch.set_grad_enabled(False)
        model, ck = load_visual_scorer(args.archive.resolve())
        check(ck == CLIP_SHA, "Wrong L/14")
        ids, words, inverse, _, _ = load_dictionary(args.dictionary)
        tok = load_tokenizer(args.archive / "assets/native_generator_v243/turbo/tokenizer")
        extra.update(
            text=text_bank(model, tok, words, args.batch_size), labels=ids, inverse=inverse
        )
        bank = raw_features(model, lambda i: jpeg(paths[i]), len(paths), args.batch_size)
        rec, _ = original_records(
            args.archive,
            np.concatenate([v["rows"] for v in regs.values()]).tolist(),
            np.concatenate([v["labels"] for v in regs.values()]).tolist(),
        )
        for p, v in regs.items():
            targets = originals(args.archive, rec, v["rows"])
            extra[p] = raw_features(model, targets.__getitem__, len(targets), args.batch_size)
    else:
        from score_within_photo_dino import FILES, encoder, reference
        from score_within_photo_dino import originals as dino_originals

        ck = FILES["model.safetensors"]
        embed = encoder(args)
        chunks = [
            embed([jpeg(f) for f in paths[i : i + args.batch_size]])
            for i in range(0, len(paths), args.batch_size)
        ]
        bank = np.concatenate(chunks)
        table = reference(args)
        with zipfile.ZipFile(args.originals) as z:
            for p in PANELS:
                r, features, _ = dino_originals(args, embed, table, z, p)
                check(r["rows"] == regs[p]["rows"].tolist(), "Wrong original DINO gallery")
                extra[p] = features
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.features,
        jpeg_sha256=hashes,
        features=bank,
        model_sha=ck,
        backend=args.backend,
        **{p + "_rows": reg["rows"] for p, reg in regs.items()},
        **extra,
    )


def case(a, sub, p, z, ix):
    r = registered(a.references, p)
    rows, folds, labels = r["rows"], r["folds"], r["labels"]
    path = a.selection / p / sub / "settings.json"
    cfg = read(path)
    frozen = a.conditions / sub / p / "conditions_512.npz"
    with np.load(frozen, allow_pickle=False) as pred:
        words = np.array(cfg["predicted_words"])
        check(
            sha(frozen) == cfg["packet_sha256"]
            and np.array_equal(words, pred["predicted_words"])
            and all(np.array_equal(pred[k], cfg[k]) for k in ("rows", "folds", "donor_rows"))
            and np.array_equal(rows, cfg["rows"])
            and np.array_equal(rows, z[p + "_rows"])
            and np.array_equal(folds, cfg["folds"])
            and set(cfg["jpeg_sha256"]) == set(ARMS),
            "Selection, reference or prediction mismatch",
        )
    pairs = []
    for w in np.unique(labels):
        for f in range(5):
            positions = np.flatnonzero((labels == w) & (folds == f) & (words == w))
            if len(positions) == 2:
                pairs.append(positions)
    i, j = np.asarray(pairs).T
    measures = {}
    y = z[p]
    stored = z["features"]
    for arm in ARMS:
        chosen = cfg["jpeg_sha256"][arm]
        check(
            len(chosen) == 120 and set(chosen) <= ix.keys(), "Missing or altered selected candidate"
        )
        v = stored[[ix[h] for h in chosen]]
        sim = unit(v.astype(float)) @ unit(y.astype(float)).T
        gallery_sim = (unit(v - v.mean(0)) @ unit(y - y.mean(0)).T) if a.backend == "clip" else sim
        gallery = score_object_panel(gallery_sim, rows, labels)
        left, right = sim[i, i] - sim[i, j], sim[j, j] - sim[j, i]

        def credit(d):
            return (d > 1e-12).astype(float) + 0.5 * (abs(d) <= 1e-12)

        ms = {
            "within_class_photo": np.array([v["intra10_top1"] for v in gallery["per_photo"]]),
            "paired_assignment": credit(left + right),
            "single_two_choice": (credit(left) + credit(right)) / 2,
            "cosine_to_original": sim.diagonal(),
        }
        if a.backend == "clip":
            ids = z["labels"].tolist()
            scores = (unit(v) @ z["text"].T)[:, z["inverse"]]
            ms["full1854"] = topk_credit(scores, 1)[
                np.arange(len(v)), [ids.index(k) for k in labels]
            ]
        for metric, vec in ms.items():
            measures.setdefault(metric, {})[arm] = (vec, labels if len(vec) == 120 else labels[i])
    return measures, {"settings_sha256": sha(path), "packet_sha256": sha(frozen)}


def summarize(all_cases):
    out = {}
    for metric in all_cases[0]:
        by = {}
        for arm in ARMS:
            vals, groups = zip(*(v[metric][arm] for v in all_cases), strict=True)
            by[arm] = np.concatenate(vals)
        labels = np.concatenate(groups)
        out[metric] = dict(
            arms={k: estimate(v, labels) for k, v in by.items()},
            own_minus={k: estimate(by["own"] - by[k], labels) for k in ARMS if k != "own"},
        )
    return out


def score(a):
    subjects = {}
    with np.load(a.features, allow_pickle=False) as z:
        check(str(z["backend"]) == a.backend, "Incorrect features")
        ix = {v: i for i, v in enumerate(z["jpeg_sha256"])}
        for s in ("sub-04", "sub-05", "sub-01"):
            blocks = []
            panels = {}
            for p in PANELS:
                data, provenance = case(a, s, p, z, ix)
                panels[p] = dict(**provenance, metrics=summarize([data]))
                blocks.append(data)
            subjects[s] = {"panels": panels, "both_panels": summarize(blocks)}
        save(
            a.output,
            dict(
                backend=a.backend,
                model_sha256=str(z["model_sha"]),
                feature_file_sha256=sha(a.features),
                generator_budget=8,
                unique_generated_candidates=960,
                protocol="Four selectors share eight fixed candidate JPEGs for each object/fold. "
                "The full-vocabulary and ten-photo metrics include all 120 queries per panel. "
                "Assignment and independent two-choice retain pairs with both words correct. "
                "Independent two-choice credits are averaged within the pair; multiply the "
                "pair credit and denominator by two to recover query counts. Confidence intervals "
                "resample represented object classes within participant.",
                subjects=subjects,
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("encode", "score"))
    p.add_argument("--backend", choices=("dino", "clip"), required=True)
    p.add_argument("--features", type=Path, required=True)
    for name in ("archive", "dictionary", "bank", "output", "selection"):
        p.add_argument("--" + name, type=Path)
    defaults = {
        "references": "outputs/object-fidelity",
        "conditions": "outputs/photo-structure/within_photo",
        "model": "outputs/photo-structure/dino/facebook-dinov2-small",
        "originals": "outputs/photo-structure/training_images.zip",
        "manifest": "outputs/official-object-readout/sub-03/atm_train_rows.csv",
        "out": "outputs/photo-structure/dino/scoring",
    }
    for name, path in defaults.items():
        p.add_argument("--" + name, type=Path, default=Path(path))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    (encode if a.stage == "encode" else score)(a)
