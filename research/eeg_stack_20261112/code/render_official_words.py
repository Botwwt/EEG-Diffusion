"""Render official sub-03 R80 EEG-predicted object words with the unmodified SDXL-Turbo.

The predicted, shuffled and empty arms use only the frozen prediction CSV and
query IDs. Ground-truth objects are read only for the optional oracle, contact
sheet and image scoring. No exemplar enters the generator.
"""

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import random
import re
import time
import zipfile
from collections import Counter
from pathlib import Path

NUM_QUERIES = 200
SEED = 20261125


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ordered(records):
    records = sorted(records, key=lambda r: int(r["eeg_row"]))
    if [int(r["eeg_row"]) for r in records] != list(range(NUM_QUERIES)):
        raise ValueError("Expected all 200 unique official EEG rows 0..199")
    return records


def clean_word(value):
    word = re.sub(r"^\d+[_\s-]+", "", value.strip()).replace("_", " ")
    word = " ".join(word.split())
    if not word or len(word) > 100 or not re.fullmatch(r"[A-Za-z0-9 '\-.,()/&]+", word):
        raise ValueError(f"Invalid English object label {value!r}")
    return word


def fixed_plan(records, seed=SEED):
    words = [clean_word(r["predicted_word"]) for r in records]
    # Rotate a class-sorted permutation. This minimizes duplicate-word matches
    # while preserving the exact multiset of predictions, without consulting truth.
    shuffled = list(range(len(words)))
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=lambda i: words[i].casefold())
    count = max(Counter(w.casefold() for w in words).values())
    shift = count if count < len(words) else 1
    target = shuffled[shift:] + shuffled[:shift]
    partner = dict(zip(shuffled, target, strict=True))
    plan = []
    for i, word in enumerate(words):
        source = partner[i]
        if source == i:
            raise ValueError("Shuffled-word source must be a different EEG row")
        tag = f"official-word|sub-03|{seed}|{i}".encode()
        image_seed = int(hashlib.sha256(tag).hexdigest()[:16], 16) % (2**63 - 1)
        plan.append(
            dict(
                row=i,
                predicted_id=records[i]["predicted_id"],
                word=word,
                shuffled_row=source,
                shuffled_word=words[source],
                seed=image_seed,
            )
        )
    return plan


def prompt(word):
    return "A photo of " + word + "."


def output_image(out, arm, row):
    where = "diagnostics" if arm == "oracle" else "images"
    return out / where / arm / f"{row:03d}.jpg"


def arm_prompts(plan, arm, truth_words=None):
    if arm == "oracle":
        return [prompt(w) for w in truth_words]
    if arm == "empty":
        return [""] * len(plan)
    key = "word" if arm == "predicted" else "shuffled_word"
    return [prompt(r[key]) for r in plan]


def generator(args, plan, arms, truth_words):
    import torch
    from diffusers import StableDiffusionXLPipeline

    turbo = args.archive / "assets/native_generator_v243/turbo"
    weight = turbo / "sd_xl_turbo_1.0_fp16.safetensors"
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(weight),
        config=str(turbo),
        torch_dtype=torch.float16,
        local_files_only=True,
        add_watermarker=False,
    )
    if (
        type(pipe.scheduler).__name__ != "EulerAncestralDiscreteScheduler"
        or pipe.scheduler.config.timestep_spacing != "trailing"
        or pipe.scheduler.config.prediction_type != "epsilon"
    ):
        raise ValueError("Unexpected native SDXL-Turbo four-step scheduler")
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_slicing()
    pipe.scheduler.set_timesteps(4, device="cuda")
    if pipe.scheduler.timesteps.tolist() != [999, 749, 499, 249]:
        raise ValueError("Unexpected scheduler timesteps")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    metadata = dict(
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        model_sha256=file_sha(weight),
        timesteps=pipe.scheduler.timesteps.tolist(),
        guidance_scale=0,
        nfe=4,
        nsel=1,
        ip_adapter=False,
        model_revision="71153311d3dbb46851df1931d3ca6e939de83304",
    )
    runtime = args.out / "runtime.json"
    if runtime.exists() and json.loads(runtime.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Generator model or runtime differs from the existing render")
    write_json(runtime, metadata)
    with torch.inference_mode():
        for arm in arms:
            prompts = arm_prompts(plan, arm, truth_words)
            for lo in range(0, len(plan), args.batch_size):
                indices = list(range(lo, min(lo + args.batch_size, len(plan))))
                files = [output_image(args.out, arm, i) for i in indices]
                if not all(p.exists() for p in files):
                    # Always regenerate the complete registered batch if partially interrupted.
                    seeds = [
                        torch.Generator(device="cuda").manual_seed(plan[i]["seed"]) for i in indices
                    ]
                    images = pipe(
                        prompt=[prompts[i] for i in indices],
                        width=512,
                        height=512,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        generator=seeds,
                        num_images_per_prompt=1,
                        output_type="pil",
                    ).images
                    if len(images) != len(files):
                        raise ValueError("Incomplete generator batch")
                    for im, path in zip(images, files, strict=True):
                        if im.size != (512, 512):
                            raise ValueError("Unexpected image size")
                        path.parent.mkdir(exist_ok=True, parents=True)
                        tmp = path.with_suffix(".tmp")
                        im.convert("RGB").save(tmp, "JPEG", quality=95, subsampling=0)
                        os.replace(tmp, path)
                status = dict(stage="render", arm=arm, completed=indices[-1] + 1, total=len(plan))
                write_json(args.out / "progress.json", status)
                print(json.dumps(status), flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def reference(zipped, record):
    from PIL import Image

    raw = zipped.read(record["archive_member"])
    expected = record.get("image_sha256") or record.get("sha256")
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Original photo hash mismatch at EEG row {record['eeg_row']}")
    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB")


def crop512(im):
    from PIL import Image

    w, h = im.size
    aspect = (512, int(512 * h / w)) if w <= h else (int(512 * w / h), 512)
    im = im.resize(aspect, getattr(Image, "Resampling", Image).BILINEAR)
    x, y = round((aspect[0] - 512) / 2), round((aspect[1] - 512) / 2)
    return im.crop((x, y, x + 512, y + 512))


def gallery(out, plan, arms, truth_words, manifest, zipped):
    from PIL import Image, ImageDraw

    chosen = [i * (len(plan) - 1) // 19 for i in range(20)]
    names = (["original"] if zipped is not None else []) + list(arms)
    tile, gap, line = 125, 5, 44
    groupw = len(names) * (tile + gap) + 12
    canvas = Image.new("RGB", (2 * groupw, 10 * (tile + line + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for pos, i in enumerate(chosen):
        x, y = (pos % 2) * groupw, (pos // 2) * (tile + line + gap)
        label = f"{i:03d} EEG: {plan[i]['word']}   shuffled: {plan[i]['shuffled_word']}"
        if truth_words is not None:
            label += f"   original: {truth_words[i]}"
        draw.text((x + 2, y + 2), label, fill="black")
        for j, arm in enumerate(names):
            if arm == "original":
                im = crop512(reference(zipped, manifest[i]))
            else:
                with Image.open(output_image(out, arm, i)) as image:
                    im = image.convert("RGB")
            px = x + j * (tile + gap)
            canvas.paste(
                im.resize((tile, tile), getattr(Image, "Resampling", Image).LANCZOS), (px, y + line)
            )
            draw.text(
                (px + 2, y + 23), "oracle (diagnostic)" if arm == "oracle" else arm, fill="black"
            )
    tmp = out / "gallery.tmp"
    canvas.save(tmp, "JPEG", quality=94)
    os.replace(tmp, out / "gallery.jpg")
    return chosen


def score_generated(args, arms, manifest, zipped):
    import numpy as np
    import torch
    from eval_instance_starts import load_visual_scorer
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from torchvision import transforms as T

    model, evaluator_sha = load_visual_scorer(args.archive.resolve())
    pre = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ]
    )

    def features(arm):
        all_f, small = [], []
        for lo in range(0, len(manifest), args.batch_size):
            imgs = []
            for i in range(lo, min(lo + args.batch_size, len(manifest))):
                if arm is None:
                    im = crop512(reference(zipped, manifest[i]))
                else:
                    with Image.open(output_image(args.out, arm, i)) as jpg:
                        im = jpg.convert("RGB")
                small.append(
                    np.asarray(
                        im.resize((128, 128), getattr(Image, "Resampling", Image).BILINEAR),
                        dtype=np.float32,
                    )
                    / 255
                )
                imgs.append(T.ToTensor()(im))
            f = model.encode_image(pre(torch.stack(imgs).cuda())).float()
            if not torch.isfinite(f).all():
                raise ValueError("Nonfinite image embedding")
            all_f.append(f.cpu().numpy())
        return np.concatenate(all_f), np.stack(small)

    def unit(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)

    def gray_ssim(a, b):
        def smooth(x):
            return gaussian_filter(x, sigma=1.5, truncate=3.5, mode="reflect")

        ma, mb = smooth(a), smooth(b)
        va, vb = smooth(a * a) - ma * ma, smooth(b * b) - mb * mb
        cov = smooth(a * b) - ma * mb
        value = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
            (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
        )
        return value[5:-5, 5:-5].mean()

    idx = np.arange(len(manifest))
    metrics = {}
    with torch.inference_mode():
        gt, gt_rgb = features(None)
        gc = unit(gt - gt.mean(0))
        for arm in arms:
            f, rec = features(arm)
            s = unit(f - f.mean(0)) @ gc.T
            diag = s.diagonal()[:, None]
            ties = (s == diag).sum(1) - 1
            top = (diag[:, 0] == s.max(1)) / (s == s.max(1, keepdims=True)).sum(1)
            afc = ((s < diag).sum(1) + 0.5 * ties) / (len(idx) - 1)
            u = rec.reshape(len(idx), -1)
            v = gt_rgb.reshape(len(idx), -1)
            corr = (unit(u - u.mean(1, keepdims=True)) * unit(v - v.mean(1, keepdims=True))).sum(1)
            luma = np.array([0.299, 0.587, 0.114])
            ssim = np.array(
                [gray_ssim(r @ luma, t @ luma) for r, t in zip(rec, gt_rgb, strict=True)]
            )
            metrics[arm] = dict(
                clip_target_cos=(unit(f) * unit(gt)).sum(1),
                clip_top1_200=top,
                clip_2afc_200=afc,
                rgb_corr128=corr,
                gray_ssim128=ssim,
            )
            np.save(args.out / f"{arm}_scores.npy", s, allow_pickle=False)
    boot = np.random.default_rng(SEED).integers(len(idx), size=(10000, len(idx)))
    paired = {}
    for method, control in (
        ("predicted", "shuffled"),
        ("predicted", "empty"),
        ("selected", "first_draw"),
        ("selected", "wrong_eeg"),
        ("gaussian_image", "gaussian_global"),
    ):
        if method not in metrics or control not in metrics:
            continue
        rows = {}
        for k in metrics[method]:
            d = metrics[method][k] - metrics[control][k]
            rows[k] = dict(
                delta=float(d.mean()), ci95=np.quantile(d[boot].mean(1), [0.025, 0.975]).tolist()
            )
        paired[method + " minus " + control] = rows
    report = dict(
        clip_sha256=evaluator_sha,
        query_rows=idx.tolist(),
        means={arm: {k: float(v.mean()) for k, v in m.items()} for arm, m in metrics.items()},
        paired=paired,
        per_query={arm: {k: v.tolist() for k, v in m.items()} for arm, m in metrics.items()},
        uncertainty="200 paired official rows, fixed reference gallery, percentile bootstrap 10000 draws",
        target="Original RGB, Resize512 bilinear/CenterCrop512; one photo per official concept.",
    )
    write_json(args.out / "scores.json", report)
    return report["means"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predicted-only",
        action="store_true",
        help="Render only the new word readout when shared controls have already been measured",
    )
    parser.add_argument("--archive", type=Path, required=True, help="Legacy PKU EEG archive root")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Official originals_manifest.csv")
    parser.add_argument("--images", type=Path, help="Official test_images.zip")
    parser.add_argument("--dictionary", type=Path, help="Dictionary CSV for readable truth labels")
    parser.add_argument(
        "--oracle", action="store_true", help="Also render a separate truth-word diagnostic"
    )
    parser.add_argument(
        "--score-clip", action="store_true", help="Score saved JPGs versus the official originals"
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Build gallery and optional scores from saved JPGs",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate inputs and print the fixed design on a CPU",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 16:
        parser.error("--batch-size must be 1..16")
    if args.images and not args.manifest or args.oracle and not args.manifest:
        parser.error("--images and --oracle each require --manifest")
    if args.score_clip and not (args.images and args.manifest):
        parser.error("--score-clip requires --images and --manifest")
    plan = fixed_plan(ordered(read_csv(args.predictions)), args.seed)
    manifest = ordered(read_csv(args.manifest)) if args.manifest else None
    truth_words = None
    if manifest:
        d = (
            {r["unique_id"]: r["word"] for r in read_csv(args.dictionary)}
            if args.dictionary
            else {}
        )
        truth_words = [clean_word(d.get(r["unique_id"], r["unique_id"])) for r in manifest]
        if len({r["archive_member"] for r in manifest}) != NUM_QUERIES:
            raise ValueError("Duplicate or missing reference identity")
    arms = (["predicted"] if args.predicted_only else ["predicted", "shuffled", "empty"]) + (
        ["oracle"] if args.oracle else []
    )
    identical = sum(r["word"].casefold() == r["shuffled_word"].casefold() for r in plan)
    desc = dict(
        queries=len(plan),
        arms=arms,
        identical_shuffled_prompt_count=identical,
        seed=args.seed,
        batch_size=args.batch_size,
        predictions_sha256=file_sha(args.predictions),
        script_sha256=file_sha(Path(__file__)),
        prompt_template="A photo of {predicted_word}.",
        oracle_words=truth_words if args.oracle else None,
        subject="sub-03",
        repeat_count=80,
    )
    if args.plan_only:
        print(json.dumps(desc, indent=2), flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = args.out / "settings.json"
    if cfg.exists() and json.loads(cfg.read_text(encoding="utf-8")) != desc:
        raise ValueError("Existing output has a different render registration")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    write_json(cfg, desc)
    write_json(args.out / "query_plan.json", plan)
    start = time.monotonic()
    if not args.finalize_only:
        generator(args, plan, arms, truth_words)
    for arm in arms:
        for r in plan:
            if not output_image(args.out, arm, r["row"]).is_file():
                raise ValueError(f"Missing saved JPG for {arm}, row {r['row']}")
    zipped = zipfile.ZipFile(args.images) if args.images else None
    try:
        chosen = gallery(args.out, plan, arms, truth_words, manifest, zipped)
        scores = score_generated(args, arms, manifest, zipped) if args.score_clip else None
    finally:
        if zipped is not None:
            zipped.close()
    status = dict(
        stage="complete",
        rows=len(plan),
        images_per_arm=len(plan),
        gallery_rows=chosen,
        seconds=round(time.monotonic() - start, 2),
        mean_scores=scores,
    )
    write_json(args.out / "completion.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
