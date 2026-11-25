"""Independently aggregate the registered generic and original-panel trial-encoder scores."""

import argparse
from pathlib import Path

import numpy as np
from fit_shared_photo_eeg import require
from register_fresh_photo_panel import freeze, read, sha
from score_trial_object_encoder import ARMS, BOOT_SEED, PUBLIC, SUB, metrics, primary_gate


def summarize(root, out):
    study = PUBLIC / "trial_object_encoder_design.json"
    trainer = Path(__file__).with_name("fit_trial_object_encoder.py")
    scorer = Path(__file__).with_name("score_trial_object_encoder.py")
    audit = dict(
        subject=SUB,
        design_sha256=sha(study),
        trainer_sha256=sha(trainer),
        scorer_sha256=sha(scorer),
        summarizer_sha256=sha(Path(__file__)),
        run="Single sub-04; four real presentations per photograph, 63-channel learned encoder. Fixed half-weight fusion with the 17-channel Fisher. No image features. Generic models fit on 416 objects; target models refit on all 512. Four-way and twelve-way generic decisions concern the same 960 photos.",
        intervals="Percentile paired intervals resample whole objects with 10000 draws, seed 260552. The two disjoint original target panels pool one 12-class and one provided four-class menu by independently bootstrapping their twelve objects. These are historical development panels.",
    )
    for phase, cases in (
        ("generic", ("generic_4", "generic_12")),
        ("target", ("stress12", "new_three4")),
    ):
        folder = root / phase
        receipt = read(folder / "receipt.json")
        score = read(folder / "score.json")
        require(
            receipt["source_sha256"] == audit["trainer_sha256"]
            and score["scorer_sha256"] == audit["scorer_sha256"]
            and receipt["design_sha256"] == score["design_sha256"] == audit["design_sha256"]
            and sha(folder / "model.safetensors") == receipt["weights_sha256"]
            and sha(folder / "embeddings.npz") == score["packet_sha256"] == receipt["packet_sha256"]
            and sha(folder / "receipt.json") == score["encoder_receipt_sha256"]
            and receipt["updates"] == 750
            and receipt["subject"] == score["subject"] == SUB
            and score["bootstrap_seed"] == BOOT_SEED,
            "Different training run or score",
        )
        evidence = {}
        raw = []
        for name in cases:
            path = folder / (name + ".npz")
            require(sha(path) == score["predictions_sha256"][name], "Changed prediction packet")
            with np.load(path, allow_pickle=False) as z:
                d = dict(z)
            require(
                np.array_equal(d["mixed_scores"], (d["encoder_scores"] + d["fisher_scores"]) / 2),
                "Wrong fusion",
            )
            for arm in ARMS:
                observed = d["candidate_ids"][
                    np.arange(len(d["rows"])), d[arm + "_scores"].argmax(1)
                ]
                require(np.array_equal(observed, d[arm]), "Wrong exported " + arm + " label")
            summary = metrics([d])
            require(summary == score["cases"][name], "Recomputed paired case differs")
            evidence[name] = summary
            raw.append(d)
        info = dict(
            training_objects=len(receipt["train_classes"]),
            evaluation_objects=len(receipt["test_classes"]),
            updates=receipt["updates"],
            weights_sha256=receipt["weights_sha256"],
            embedding_packet_sha256=receipt["packet_sha256"],
            training_receipt_sha256=sha(folder / "receipt.json"),
            score_sha256=sha(folder / "score.json"),
            row_packet_sha256=score["predictions_sha256"],
            results=evidence,
        )
        if phase == "generic":
            require(
                primary_gate(evidence["generic_12"]) == score["advance"], "Generic decision differs"
            )
            info["advance"] = score["advance"]
        else:
            require(
                score["generic_score_sha256"] == audit["generic"]["score_sha256"]
                and audit["generic"]["advance"],
                "Target unbound from the generic decision",
            )
            pooled = metrics(raw)
            require(
                score["panel_stratified"] == pooled, "Recomputed original-panel interval differs"
            )
            info["panel_stratified"] = pooled
        audit[phase] = info
    freeze(out, audit)
    print(out, sha(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/trial-object-encoder"))
    p.add_argument("--out", type=Path, default=PUBLIC / "trial_object_encoder.json")
    args = p.parse_args()
    summarize(args.root, args.out)
