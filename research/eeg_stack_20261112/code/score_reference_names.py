"""Measure the same frozen CLIP naming rule on the photographs used as references."""

import argparse
from pathlib import Path

import numpy as np
from eval_instance_starts import load_visual_scorer
from render_within_photo_ip import check, save, sha
from score_official_objects import load_dictionary, load_tokenizer, text_bank, topk_credit, unit
from score_personal_photo_layout import original_records
from score_reconstruction_jpegs import CLIP_SHA, originals, raw_features, registered


def run(a):
    import torch

    reg = registered(a.references, a.panel)
    records, manifest_sha = original_records(
        a.archive, reg["rows"].tolist(), reg["labels"].tolist()
    )
    model, model_sha = load_visual_scorer(a.archive.resolve())
    check(model_sha == CLIP_SHA, "Wrong frozen naming evaluator")
    ids, common, inverse, _, _ = load_dictionary(a.dictionary)
    tok = load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer")
    refs = originals(a.archive, records, reg["rows"])
    with torch.inference_mode():
        bank = text_bank(model, tok, common, a.batch_size)
        features = raw_features(model, refs.__getitem__, len(refs), a.batch_size)
    logits = (unit(features) @ bank.T)[:, inverse]
    lookup = {word: i for i, word in enumerate(ids)}
    truth = np.array([lookup[word] for word in reg["labels"]])
    candidates = np.unique(truth)
    credit = topk_credit(logits, 1)[np.arange(len(truth)), truth]
    closed = topk_credit(logits[:, candidates], 1)[
        np.arange(len(truth)), np.searchsorted(candidates, truth)
    ]
    save(
        a.out,
        dict(
            panel=a.panel,
            registration_sha256=reg["sha"],
            clip_sha256=model_sha,
            dictionary_sha256=sha(a.dictionary),
            image_manifest_sha256=manifest_sha,
            targets=len(truth),
            full1854_credit=float(credit.sum()),
            closed12_credit=float(closed.sum()),
            class_full1854_credit={
                word: float(credit[reg["labels"] == word].sum()) for word in reg["vocab"]
            },
            scorer_sha256=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "references", "dictionary", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--panel", default="fresh")
    p.add_argument("--batch-size", type=int, default=32)
    run(p.parse_args())
