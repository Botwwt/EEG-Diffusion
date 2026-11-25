"""Score six frozen official EEG image arms against all THINGS labels and 200 original photographs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from eval_instance_starts import load_visual_scorer, sha
from PIL import Image
from render_official_words import crop512, ordered, read_csv, reference
from score_official_objects import (
    N_RESAMPLES,
    SEED,
    TEMPLATES,
    image_features,
    load_dictionary,
    load_tokenizer,
    paired_arms,
    text_bank,
    topk_credit,
)

ARMS = ("r4", "r80", "wrong", "r4_visual", "r80_visual", "wrong_visual")
CLIP_SHA = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
DICT_SHA = "1aebc7a2c35edcfcfffc63934754440c90534ccc46df0537db41dc6197450132"
MANIFEST_SHA = "74773a1ef288b8d1c974a256a6b1c39940949c36842fca1302dc76e641943089"
ZIP_SHA = "57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a"
PACKET_SHA = "214c21ee3c1c53e7d40927a536d9684b164e11fe9a94ec1c9e8d6ed1257a1983"


def check(ok, why):
    if not ok:
        raise ValueError(why)


def checked_image(path, digest):
    raw = path.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == digest, f"Changed generated JPEG at {path}")
    with Image.open(io.BytesIO(raw)) as image:
        check(
            image.format == "JPEG" and image.mode == "RGB" and image.size == (512, 512),
            f"Invalid generated image at {path}",
        )
        return image.copy()


def inputs(a):
    manifest = a.references / "originals_manifest.csv"
    photos = a.references / "test_images.zip"
    check(
        sha(manifest) == MANIFEST_SHA and sha(photos) == ZIP_SHA and sha(a.dictionary) == DICT_SHA,
        "Expected the locked official original-photo manifest, archive and 1854-word dictionary",
    )
    rec = ordered(read_csv(manifest))
    check(
        len({r["unique_id"] for r in rec}) == 200
        and len({r["archive_member"] for r in rec}) == 200,
        "Repeated original class or member",
    )
    ids, common, inverse, _, _ = load_dictionary(a.dictionary)
    by_id = {v: i for i, v in enumerate(ids)}
    truth = np.array([by_id[r["unique_id"]] for r in rec])
    marker = a.render / "settings.json"
    cfg = json.loads(marker.read_bytes())
    rows = list(range(200))
    arms = ARMS
    packet_sha = PACKET_SHA
    if getattr(a, "design", None):
        design = json.loads(a.design.read_bytes())
        arms = tuple(design["arms"])
        packet_sha = design["packet_sha256"]
        check(
            arms in (ARMS, ARMS[3:], ARMS[:3]) and cfg["design_sha256"] == sha(a.design),
            "Different render design",
        )
    check(cfg["packet_sha256"] == packet_sha, "Render used different frozen EEG conditions")
    check(
        cfg["rows"] == rows
        and cfg["seeds"] == [923761 + i for i in rows]
        and set(cfg["jpeg_sha256"]) == set(arms),
        "Expected 200 original query rows, their shared seeds and all registered arms",
    )
    for k in ("prompts", "eeg_rows"):
        check(
            set(cfg[k]) == set(arms) and all(len(cfg[k][arm]) == 200 for arm in arms),
            f"Wrong {k} metadata",
        )
    for arm in arms:
        check(
            all(isinstance(x, str) for x in cfg["prompts"][arm]), f"Wrong prompt records in {arm}"
        )
        eeg = cfg["eeg_rows"][arm]
        check(sorted(eeg) == rows, f"Missing or repeated conditioning EEG row in {arm}")
        if arm.startswith("wrong"):
            check(
                eeg == np.roll(np.arange(200), 67).tolist(),
                "Wrong-EEG control has an aligned row or differs across render modes",
            )
        else:
            check(eeg == rows, f"Unexpected EEG permutation in {arm}")
        if arm.endswith("_visual"):
            check(
                cfg["prompts"][arm] == ["A photograph."] * 200,
                "Visual-only arm used a named object prompt",
            )
    if "wrong" in arms:
        check(
            cfg["prompts"]["wrong"] == [cfg["prompts"]["r80"][i] for i in cfg["eeg_rows"]["wrong"]],
            "Wrong-EEG word prompts differ from the frozen R80 donors",
        )
    folders = {}
    for arm in arms:
        found = [p for p in (a.render / arm, a.render / "images" / arm) if p.is_dir()]
        check(len(found) == 1, f"Missing or ambiguous JPEG folder for {arm}")
        folders[arm] = found[0]
        hashes = cfg["jpeg_sha256"][arm]
        check(
            len(hashes) == 200
            and {p.name for p in found[0].glob("*.jpg")} == {f"{i:03}.jpg" for i in rows},
            f"Incomplete generated JPEG registry for {arm}",
        )
        for i, h in enumerate(hashes):
            checked_image(found[0] / f"{i:03}.jpg", h)
    with ZipFile(photos) as z:
        for row in rec:
            reference(z, row)
    return cfg, folders, rec, ids, common, inverse, truth


def ranking(scores, truth, labels):
    scores = np.asarray(scores)
    check(
        scores.shape == (len(truth), len(labels)) and np.isfinite(scores).all(),
        "Invalid score matrix",
    )
    idx = np.arange(len(truth))
    best = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    target = scores[idx, truth, None]
    first = 1 + (scores > target).sum(1)
    last = (scores >= target).sum(1)
    credits = {f"top{k}": topk_credit(scores, k)[idx, truth] for k in (1, 5)}
    means = {f"{k}_credit": float(v.sum()) for k, v in credits.items()}
    means.update(
        n=len(truth),
        top1_display_hits=int((best[:, 0] == truth).sum()),
        top5_display_hits=int((best == truth[:, None]).any(1).sum()),
        median_target_midrank=float(np.median((first + last) / 2)),
    )
    rows = [
        dict(
            top1=str(labels[b[0]]),
            top5=labels[b].tolist(),
            target_score=float(target[i, 0]),
            target_rank_first=int(first[i]),
            target_rank_last=int(last[i]),
            top1_credit=float(credits["top1"][i]),
            top5_credit=float(credits["top5"][i]),
        )
        for i, b in enumerate(best)
    ]
    return means, rows, credits


def run(a):
    cfg, folders, rec, ids, common, inverse, truth = inputs(a)
    if a.check_only:
        print(
            f"Checked {200 * len(folders)} generated JPEGs, 200 originals, dictionary, rows and seeds.",
            flush=True,
        )
        return
    import torch

    tokenizer_path = a.archive / "assets/native_generator_v243/turbo/tokenizer"
    tok = load_tokenizer(tokenizer_path)
    model, weights = load_visual_scorer(a.archive.resolve())
    check(weights == CLIP_SHA, "Expected frozen independent OpenAI CLIP-L/14")
    source_names = ("original", *folders)
    expected = np.arange(200)
    subset = np.sort(truth)
    localtruth = np.searchsorted(subset, truth)
    models = {}
    decisions = []
    scores = {}
    matched = {}
    variants = {}
    with torch.inference_mode(), ZipFile(a.references / "test_images.zip") as z:
        bank = text_bank(model, tok, common, a.batch_size).astype(np.float64)

        def get_original(i):
            return crop512(reference(z, rec[i]))

        ref = image_features(model, get_original, 200, a.batch_size).astype(np.float64)
        for name in source_names:
            if name == "original":
                features = ref
            else:

                def get(i, arm=name):
                    return checked_image(folders[arm] / f"{i:03}.jpg", cfg["jpeg_sha256"][arm][i])

                features = image_features(model, get, 200, a.batch_size).astype(np.float64)
            full = (features @ bank.T)[:, inverse]
            photo = features @ ref.T
            scores[name] = full
            matched[name] = photo
            layers = (
                ("full1854", full, truth, ids),
                ("restricted200", full[:, subset], localtruth, ids[subset]),
                ("photo200", photo, expected, ids[truth]),
            )
            summary = {}
            values = {}
            records = [
                dict(source=name, eeg_row=i, truth_id=rec[i]["unique_id"])
                for i in expected.tolist()
            ]
            for key, mat, targets, labels in layers:
                summary[key], detail, credits = ranking(mat, targets, labels)
                for record, row in zip(records, detail, strict=True):
                    record[key] = row
                values.update({key + "_" + k: v for k, v in credits.items()})
            diag = np.diag(photo)
            off = (photo.sum(1) - diag) / 199
            summary["photo_cosine"] = dict(
                mean_paired=float(diag.mean()),
                mean_wrong=float(off.mean()),
                paired_minus_wrong=float((diag - off).mean()),
            )
            for i, r in enumerate(records):
                r["generation_seed"] = None if name == "original" else cfg["seeds"][i]
                r["conditioning_eeg_row"] = None if name == "original" else cfg["eeg_rows"][name][i]
                r["mean_other_photo_cosine"] = float(off[i])
            values.update(paired_photo_cosine=diag, paired_minus_wrong_photo_cosine=diag - off)
            models[name] = summary
            variants[name] = values
            decisions.extend(records)
            print(
                name,
                "full label top1/top5",
                summary["full1854"]["top1_credit"],
                summary["full1854"]["top5_credit"],
                "image retrieval",
                summary["photo200"]["top1_credit"],
                flush=True,
            )
    comparisons = [
        ("r80", "r4"),
        ("r80", "wrong"),
        ("r4", "wrong"),
        ("r80_visual", "r4_visual"),
        ("r80_visual", "wrong_visual"),
        ("r4_visual", "wrong_visual"),
        ("r4_visual", "r4"),
        ("r80_visual", "r80"),
    ]
    paired = {
        f"{left}_minus_{right}": paired_arms(variants[left], variants[right])
        for left, right in comparisons
        if left in variants and right in variants
    }
    out = a.out if a.out.suffix.lower() == ".json" else a.out / "summary.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    rows_file = out.with_name(out.stem + "_rows.jsonl")
    raw_file = out.with_name(out.stem + "_scores.npz")
    rows_file.write_bytes(
        b"".join(
            (json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n").encode() for v in decisions
        )
    )
    np.savez_compressed(
        raw_file,
        sources=np.asarray(source_names),
        dictionary_ids=ids,
        official_ids=ids[truth],
        rows=expected,
        text_scores=np.stack([scores[n] for n in source_names]),
        photo_scores=np.stack([matched[n] for n in source_names]),
    )
    provenance = dict(
        render_settings_sha256=sha(a.render / "settings.json"),
        original_zip_sha256=ZIP_SHA,
        manifest_sha256=MANIFEST_SHA,
        dictionary_sha256=DICT_SHA,
        model_sha256=weights,
        vocab_sha256=sha(tokenizer_path / "vocab.json"),
        merges_sha256=sha(tokenizer_path / "merges.txt"),
        evaluation_code_sha256=sha(Path(__file__)),
        archive_clip_implementation_sha256=sha(
            a.archive / "code/metric_sources_v272/openai_clip_model.py"
        ),
    )
    report = dict(
        n_official_photos=200,
        n_generated_jpegs=200 * len(folders),
        independent_clip="OpenAI CLIP ViT-L/14",
        sources=list(source_names),
        templates=list(TEMPLATES),
        distinct_readable_names=len(common),
        provenance=provenance,
        ranking="Full naming ranks all 1854 THINGS IDs, including IDs sharing a text feature. The 200-label diagnostic uses the known official test-class dictionary. Display ties break in original dictionary order; boundary credits divide equally among tied IDs. Retrieval ranks the 200 original-photo embeddings by raw cosine in official row order. One photo per object means the retrieval task can use object identity. Original-on-original photo retrieval is a self-match integrity check.",
        uncertainty=dict(
            n=200,
            unit="paired official stimulus ID (one per class, fixed person)",
            seed=SEED,
            draws=N_RESAMPLES,
            method="Per-contrast paired percentile bootstrap and two-sided paired swap; unadjusted.",
        ),
        summary=models,
        paired=paired,
        per_row=dict(file=rows_file.name, sha256=sha(rows_file)),
        complete_scores=dict(
            file=raw_file.name,
            sha256=sha(raw_file),
            text_shape=[len(source_names), 200, 1854],
            photo_shape=[len(source_names), 200, 200],
        ),
    )
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    tmp.replace(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("render", "references", "archive", "dictionary", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--design", type=Path, help="Match the declared design and its packet digest")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all source JPEGs and inputs on CPU without loading model weights",
    )
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch-size must be positive")
    run(a)
