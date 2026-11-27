"""Audit same-query image generation and summarize the cross-fitted condition study."""

import argparse
import json
import os

import numpy as np

from paper_eeg_attention import MATCHED_MODES
from paper_oof_conditions import HELD_OUT, require, study_root


PRIMARY = ("generated_name1854", "clip_image_top1", "ssim", "pixcorr", "clip_2way")
DINO = ("dino_top1", "dino_correct_cosine", "dino_2way")


def audit_query(native, candidate, mode):
    for key in ("subject", "budget", "n", "row", "seeds", "repeats", "draws", "unet_calls_per_draw", "prompt", "split_sha256"):
        require(candidate[key] == native[key], f"Comparison does not match the native {key}")
    require(candidate["native_checkpoint_sha256"] == native["native_checkpoint_sha256"], "A different brain encoder supplied EEG")
    require(candidate["generator_files_sha256"] == native["generator_files_sha256"],
            "Generator assets differ from Brain-HiVE")
    for key in ("array_sha256", "eeg_ids_sha256", "photo_ids_sha256"):
        require(candidate["physical_eeg"][key] == native["physical_eeg"][key], f"EEG {key} differs")
    perm = np.asarray(candidate["eeg_donor"])
    self_rows = np.arange(native["n"])
    expected = np.roll(self_rows, len(perm) // 2) if mode.endswith("-wrong") else self_rows
    require(candidate["arm"] == mode and np.array_equal(perm, expected) and candidate["updates_merged"] is True,
            "The image-attention arm or EEG donor assignment differs")


def run(panel, include_rollout=False):
    import torch

    from common import PREP, ROOT
    from paper_aggregate import bootstrap_ci, units
    from paper_dino import build_encoder, gallery, metrics, query_input
    from paper_render_cache import atomic_file, save_json, sha
    from pathlib import Path

    root = study_root()
    prep = PREP
    archive = Path(os.environ["ARCHIVE"])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    encoder = build_encoder(ROOT / "models/paper-dinov2-small", torch.device("cuda"), 24)
    ref = gallery(panel, archive, prep, encoder)
    fields = PRIMARY + DINO + (("clip_within10", "dino_within10") if panel == "fresh" else ())
    modes = ("seen", "out-of-fold") + (("out-of-fold-rollout",) if include_rollout else ())
    models = {
        mode: torch.load(root / f"attention-{mode}.pt", map_location="cpu", weights_only=True)
        for mode in modes
    }
    require(models["seen"]["proof"] == models["out-of-fold"]["proof"], "Attention comparisons use different training schedules")
    sparse_keys = ("training_subjects", "training_samples", "distinct_photos", "rank", "steps", "seed", "timesteps")
    require(all(models["seen"][key] == models["out-of-fold"][key] for key in sparse_keys), "Attention settings differ")
    snapshots = {}
    for mode, model in models.items():
        snapshots[mode] = dict(
            last128_loss=float(np.mean(model["training_loss"][-128:])),
            first128_loss=float(np.mean(model["training_loss"][:128])),
            checkpoint=sha(root / f"attention-{mode}.pt"),
            state_source=model.get("state_source", "forward-noised original-photo latent"),
        )
    settings = {key: models["seen"][key] for key in sparse_keys}
    if include_rollout:
        a, b = (models["out-of-fold"]["proof"], models["out-of-fold-rollout"]["proof"])
        require({k: v for k, v in a.items() if k != "code"} == {k: v for k, v in b.items() if k != "code"},
                "Rollout study uses different training pairs")
        require(all(models["out-of-fold-rollout"][key] == settings[key] for key in sparse_keys),
                "Rollout training settings differ")
    del models
    arms = ("native", *MATCHED_MODES) + (("out-of-fold-rollout", "out-of-fold-rollout-wrong") if include_rollout else ())
    results, receipts = {arm: [] for arm in arms}, {arm: [] for arm in arms}
    baseline_replay = {}
    for sub in HELD_OUT:
        baseline = root / f"images/native/{panel}-r4/sub-{sub:02d}"
        native = json.loads((baseline / "render.json").read_bytes())
        historical_file = ROOT / f"images/personal/native-paper/{panel}-r4/sub-{sub:02d}/render.json"
        historical = json.loads(historical_file.read_bytes())
        require(native["row"] == historical["row"] and native["seeds"] == historical["seeds"]
                and native["native_checkpoint_sha256"] == historical["checkpoint_sha256"],
                "Native replay differs from the archived query contract")
        baseline_replay[str(sub)] = sum(x == y for x, y in zip(native["pixels_sha256"], historical["pixels_sha256"], strict=True))
        for arm in arms:
            path = baseline if arm == "native" else root / f"images/{arm}/{panel}-r4/sub-{sub:02d}"
            record = json.loads((path / "render.json").read_bytes())
            if arm != "native":
                audit_query(native, record, arm)
                mode = arm.removesuffix("-wrong")
                require(record["attention_checkpoint_sha256"] == snapshots[mode]["checkpoint"], "Unexpected trained attention weights")
            primary_bytes = (path / "scores.json").read_bytes()
            tiles, _, _, _, ordered = query_input(path, panel, ref, primary_bytes, json.loads(primary_bytes))
            with np.load(path / "observations.npz", allow_pickle=False) as obs:
                values = {key: obs[key].copy() for key in fields if key in obs}
            _, dino = metrics(tiles, ref, encoder, panel)
            values.update(dino)
            summary = np.stack([values[key] for key in fields], axis=-1)
            require(summary.shape == (len(ref.rows), len(fields)) and np.isfinite(summary).all(), "Missing per-query score")
            results[arm].append(summary)
            receipts[arm].append({"subject": sub, "pixel_sequence_sha256": ordered,
                                  "score_sha256": sha(path / "scores.json"), "render_sha256": sha(path / "render.json")})
            print(panel, sub, arm, {key: float(summary[:, fields.index(key)].sum())
                                    for key in ("generated_name1854", "clip_image_top1", "dino_top1")}, flush=True)
    arrays = {arm: np.stack(tables) for arm, tables in results.items()}
    contrasts = {}
    comparisons = [("out-of-fold", "native"), ("out-of-fold", "seen"), ("out-of-fold", "out-of-fold-wrong"), ("seen", "native")]
    if include_rollout:
        comparisons += [("out-of-fold-rollout", "native"), ("out-of-fold-rollout", "out-of-fold"),
                        ("out-of-fold-rollout", "out-of-fold-rollout-wrong")]
    for a, b in comparisons:
        diff = arrays[a] - arrays[b]
        interval = bootstrap_ci(units(diff, ref.concepts, panel))
        contrasts[f"{a} minus {b}"] = {key: {"mean": float(diff[:, :, i].mean()), "ci95": interval[:, i].tolist()}
                                         for i, key in enumerate(fields)}
    suffix = f"{panel}-rollout" if include_rollout else panel
    destination = root / f"summary-{suffix}.npz"
    atomic_file(destination, lambda stream: np.savez_compressed(
        stream, arms=np.asarray(arms), people=np.asarray(HELD_OUT), metrics=np.asarray(fields),
        rows=ref.rows, concepts=ref.concepts, scores=np.stack([arrays[arm] for arm in arms])))
    save_json(root / f"summary-{suffix}.json", {
        "panel": panel, "people": list(HELD_OUT), "rows_per_person": len(ref.rows), "settings": settings,
        "per_person_means": {arm: {str(sub): dict(zip(fields, arrays[arm][i].mean(0).tolist(), strict=True))
                                    for i, sub in enumerate(HELD_OUT)} for arm in arms},
        "aggregate_means": {arm: dict(zip(fields, x.mean((0, 1)).tolist(), strict=True)) for arm, x in arrays.items()},
        "contrasts": contrasts, "fit_summaries": snapshots, "image_receipts": receipts, "gallery": ref.source,
        "native_identical_pngs_vs_archive": baseline_replay,
        "observations_sha256": sha(destination),
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("panel", choices=("official", "fresh"))
    p.add_argument("--include-rollout", action="store_true")
    args = p.parse_args()
    run(args.panel, args.include_rollout)
