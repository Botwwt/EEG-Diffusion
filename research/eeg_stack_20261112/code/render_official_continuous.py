"""Render official R80 reconstructions from continuous EEG predictions and score all 200."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from eval_instance_starts import relocated, rollout, write_json
from render_official_words import (
    file_sha,
    fixed_plan,
    gallery,
    ordered,
    output_image,
    read_csv,
    score_generated,
)
from torchvision.transforms import ToPILImage


@torch.inference_mode()
def run(args):
    root = args.archive.resolve()
    sys.path.insert(0, str(root / "code"))
    import PKU_HIVE_native_classes_20261109_v239 as classes
    import PKU_native_generator_20261109_v253 as native

    args.out.mkdir(exist_ok=True, parents=True)
    with np.load(args.conditions, allow_pickle=False) as b:
        rows, c, start, cm, sm = (
            b[k] for k in ("rows", "cone", "layout", "mean_condition", "mean_layout")
        )
    if (
        not np.array_equal(rows, np.arange(200))
        or c.shape != (200, 1024)
        or start.shape != (200, 4, 8, 8)
    ):
        raise ValueError("Continuous predictions do not cover official 200 rows")
    arms = ["predicted"] if args.candidates_only else ["predicted", "shuffled", "empty"]
    plan = fixed_plan(
        [dict(predicted_word="continuous condition", predicted_id="continuous") for _ in rows]
    )
    order = np.array([v["shuffled_row"] for v in plan])
    names = None
    if args.words:
        names = [r["predicted_word"] for r in ordered(read_csv(args.words))]
        arms.append("word_fused")
    settings = dict(
        arms=arms,
        condition_sha=file_sha(args.conditions),
        rows=rows.tolist(),
        words_sha=file_sha(args.words) if args.words else None,
        seed=args.seed,
        source_sha=file_sha(Path(__file__)),
        batch_size=args.batch_size,
    )
    path = args.out / "settings.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Different configuration in output directory")
    write_json(path, settings)
    classes.ROOT = native.ROOT = root
    native.Path = lambda path: relocated(path, root)
    pipe, adapter, projector, empty, _ = native.load_native()
    pipe.scheduler.set_timesteps(4, device="cuda")
    pipe.vae.float().enable_slicing()
    encoded = {}
    if names:
        for noun in sorted(set(names)):
            p, _, pooled, _ = pipe.encode_prompt(
                "A photo of " + noun + ".", device="cuda", do_classifier_free_guidance=False
            )
            encoded[noun] = (p.cpu(), pooled.cpu())
    pipe.text_encoder = pipe.text_encoder_2 = None
    torch.cuda.empty_cache()
    noise = torch.stack(
        [
            torch.randn((5, 4, 64, 64), generator=torch.Generator().manual_seed(args.seed + int(i)))
            for i in rows
        ]
    )
    for arm in arms:
        ix = order if arm == "shuffled" else rows
        condition = np.repeat(cm[None], len(c), axis=0) if arm == "empty" else c[ix]
        anchor = np.repeat(sm[None], len(c), axis=0) if arm == "empty" else start[ix]
        tokens = projector(adapter(torch.from_numpy(condition).half().cuda()))
        for lo in range(0, len(rows), args.batch_size):
            hi = min(lo + args.batch_size, len(rows))
            files = [output_image(args.out, arm, i) for i in range(lo, hi)]
            if all(p.is_file() for p in files):
                continue
            z0 = F.interpolate(
                torch.from_numpy(anchor[lo:hi]).cuda(),
                (64, 64),
                mode="bilinear",
                align_corners=False,
            )
            prompt = empty
            if arm == "word_fused":
                prompt = tuple(
                    torch.cat([encoded[w][j] for w in names[lo:hi]]).cuda() for j in (0, 1)
                ) + (empty[2],)
            z = rollout(pipe, prompt, tokens[lo:hi], z0, noise[lo:hi].cuda())
            rgb = (pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample * 0.5 + 0.5).clamp(
                0, 1
            )
            for img, path in zip(rgb, files, strict=True):
                path.parent.mkdir(exist_ok=True, parents=True)
                tmp = path.with_suffix(".tmp")
                ToPILImage()(img.cpu()).save(tmp, format="JPEG", quality=95)
                tmp.replace(path)
        print(f"Rendered {arm} for 200 EEG rows", flush=True)
    del pipe, tokens, adapter, projector
    torch.cuda.empty_cache()
    if args.candidates_only:
        write_json(
            args.out / "completion.json",
            dict(arms=arms, photos=200, seed=args.seed, source="EEG-only candidate generation"),
        )
        return
    manifest = ordered(read_csv(args.manifest))
    with zipfile.ZipFile(args.images) as images:
        gallery(args.out, plan, arms, [r["unique_id"] for r in manifest], manifest, images)
        metrics = score_generated(args, arms, manifest, images)
    write_json(
        args.out / "completion.json",
        dict(
            means=metrics,
            arms=arms,
            photos=200,
            controls={
                "empty": "T-mean condition and layout",
                "shuffled": "both EEG predictions from different row",
            },
        ),
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "conditions", "out", "images", "manifest"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--words", type=Path)
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help="Render EEG-derived candidates only, without opening the evaluation gallery",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=531466)
    run(parser.parse_args())
