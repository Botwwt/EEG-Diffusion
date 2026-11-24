"""Render native and train-only EEG conditions against two fixed sub-03 R4 panels."""

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decode_eeg_layout import registration
from eval_instance_starts import (
    clip_features,
    load_visual_scorer,
    make_conditions,
    read_json,
    relocated,
    rollout,
    sha,
    summarize,
    write_json,
)
from PIL import Image, ImageDraw
from score_object_panel import score_object_panel


def load_conditions(root, rows, classes, bank):
    from PKU_ordinary_component_source_v253 import BoundedConditionAdapter

    parent = root / "runs/HIVE_native_sub03_5096542"
    native = make_conditions(root, rows, classes)
    path = root / "runs/HIVE_matching_ordinary_calibration_v264/component_final.pt"
    if sha(path) != "f6998db9a49d566547bfbca045d47d471c771d4d0c6880322fbd9067d7024f99":
        raise ValueError("Unexpected ordinary calibration checkpoint")
    head = BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    head.radius = 0.25
    state = torch.load(parent / "full/brain_final.pt", map_location="cpu", weights_only=True)[
        "model"
    ]
    norm = [
        state["block.layers.0.norm." + key].cuda().expand_as(native) for key in ("weight", "bias")
    ]
    ordinary = head(native, *norm, gate_mode="fixed")[0]
    candidate = torch.from_numpy(bank["selected"]).float().cuda()
    return {"native": native, "ordinary": ordinary, "selected": candidate}


def save_montage(out, examples, rows, records, names):
    tile, gap, label = 129, 4, 125
    # Each sheet shows the first photo of ten consecutive concepts in manifest order.
    for page, start in enumerate(range(0, len(examples), 10)):
        subset = examples[start : start + 10]
        canvas = Image.new(
            "RGB", (label + (tile + gap) * len(names), 25 + (tile + gap) * len(subset)), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for col, name in enumerate(names):
            x = label + col * (tile + gap)
            draw.text((x, 4), name, fill="black")
            for rank, i in enumerate(subset):
                y = 25 + rank * (tile + gap)
                if col == 0:
                    draw.text((3, y + tile // 2), records[rows[i]]["concept_id"], fill="black")
                with Image.open(out / "examples" / f"{name}_{i:03d}.jpg") as thumb:
                    canvas.paste(thumb.resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
        canvas.save(out / f"gallery_{page + 1}.jpg", quality=94)


@torch.inference_mode()
def run(args):
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Resize,
        ToPILImage,
        ToTensor,
    )

    root = args.archive.resolve()
    sys.path.insert(0, str(root / "code"))
    import PKU_HIVE_native_classes_20261109_v239 as classes
    import PKU_native_generator_20261109_v253 as native

    classes.ROOT = native.ROOT = root
    native.Path = lambda path: relocated(path, root)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, concept, split, _ = registration(root)
    pipe, adapter, projector, empty_prompt, _ = native.load_native()
    pipe.scheduler.set_timesteps(4, device="cuda")
    if pipe.scheduler.timesteps.tolist() != [999, 749, 499, 249]:
        raise ValueError("Expected native four-step Turbo schedule")
    pipe.vae.float().enable_slicing()
    evaluator, evaluator_sha = load_visual_scorer(root)
    pre = Compose([Resize(512, InterpolationMode.BILINEAR), CenterCrop(512), ToTensor()])
    for panel in args.panels:
        layout_path = args.layout / f"{panel}_layout.npz"
        conditions_path = args.conditions / f"{panel}_conditions.npz"
        with (
            np.load(layout_path, allow_pickle=False) as layout,
            np.load(conditions_path, allow_pickle=False) as bank,
        ):
            rows = layout["rows"].astype(int)
            if not np.array_equal(rows, bank["rows"]) or set(rows) & set(split["T"]):
                raise ValueError("Misaligned or training rows in evaluation")
            labels = concept[rows]
            groups = {int(c): np.flatnonzero(labels == c).tolist() for c in np.unique(labels)}
            if len(groups) != 20 or any(len(g) != 10 for g in groups.values()):
                raise ValueError("Expected complete 20 by 10 photo panels")
            rng = np.random.default_rng(531453)
            order = rng.permutation(list(groups))
            perm = np.empty(len(rows), dtype=int)
            for group, other in zip(order, np.roll(order, 1), strict=True):
                perm[groups[group]] = rng.permutation(groups[other])
            if not np.array_equal(np.sort(perm), np.arange(len(rows))) or np.any(
                labels == labels[perm]
            ):
                raise ValueError("EEG shuffle must always change object class")
            examples = [g[0] for g in groups.values()]
            base = load_conditions(root, rows, classes, bank)
            arms = ["native", "ordinary", "selected", "shuffled", "image_oracle"]
            if args.free_arms:
                arms = args.free_arms
            if args.words:
                arms = ["selected", "eeg_words", "words_shuffled"]
            out = args.out / panel
            out.mkdir(exist_ok=True, parents=True)
            (out / "examples").mkdir(exist_ok=True)
            meta = dict(
                panel=panel,
                rows=rows.tolist(),
                arms=arms,
                seed=args.seed,
                selected=str(bank["selected_name"]),
                layout_sha=sha(layout_path),
                condition_sha=sha(conditions_path),
                scorer_sha=evaluator_sha,
                source_sha=sha(Path(__file__)),
            )
            word_prompts = None
            if args.words:
                path = args.words / f"{panel}_words.json"
                meta["words_sha"] = sha(path)
                words = read_json(path)
                if not np.array_equal(words["rows"], rows):
                    raise ValueError("Word predictions use a different row order")
                names = np.asarray(words["predicted_words"], dtype=str)
                word_prompts = {}
                for noun in np.unique(names):
                    text = "A photograph of a " + noun.replace("_", " ") + "."
                    p, _, pooled, _ = pipe.encode_prompt(
                        text, device="cuda", do_classifier_free_guidance=False
                    )
                    word_prompts[noun] = (p.cpu(), pooled.cpu())
            if (out / "settings.json").exists() and read_json(out / "settings.json") != meta:
                raise ValueError("Existing output was produced by different inputs")
            write_json(out / "settings.json", meta)
            rgb32 = torch.from_numpy(layout["layout_rgb"]).float()
            starts = []
            for batch in rgb32.split(args.batch_size):
                rgb = F.interpolate(batch.cuda(), (512, 512), mode="bilinear", align_corners=False)
                latent = pipe.vae.encode(2 * rgb - 1).latent_dist.mode()
                starts.append((latent * pipe.vae.config.scaling_factor).cpu())
            starts = torch.cat(starts)
            # Every arm uses the same row-specific noise, independent of its target category.
            noise = torch.stack(
                [
                    torch.randn(
                        (5, 4, 64, 64), generator=torch.Generator().manual_seed(args.seed + int(r))
                    )
                    for r in rows
                ]
            )
            outcomes = {}
            features = {}
            for arm in arms:
                idx = perm if arm.endswith("shuffled") else np.arange(len(rows))
                if arm == "image_oracle":
                    target_features = torch.load(
                        root / "runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt",
                        weights_only=True,
                        map_location="cpu",
                        mmap=True,
                    )
                    condition = target_features[rows].float().cuda()
                else:
                    condition = base.get(arm, base["selected"])[idx]
                tokens = projector(adapter(condition.half()))
                arm_features = []
                for lo in range(0, len(rows), args.batch_size):
                    hi = min(lo + args.batch_size, len(rows))
                    x = idx[lo:hi]
                    prompts = empty_prompt
                    if arm in ("eeg_words", "words_shuffled"):
                        prompts = tuple(
                            torch.cat([word_prompts[w][j] for w in names[x]]).cuda() for j in (0, 1)
                        ) + (empty_prompt[2],)
                    free = arm in (
                        "free",
                        "free_stronger",
                        "free_shuffled",
                        "free_stronger_shuffled",
                    )
                    stronger = arm in ("free_stronger", "free_stronger_shuffled")
                    if free:
                        for proc in pipe.unet.attn_processors.values():
                            if hasattr(proc, "to_v_ip"):
                                proc.scale = 1.5 if stronger else 1.0
                    z = rollout(
                        pipe,
                        prompts,
                        tokens[lo:hi],
                        starts[x].cuda(),
                        noise[lo:hi].cuda(),
                        first_hold=0.0 if free else 0.75,
                        tether=not free,
                    )
                    rgb = (
                        pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample * 0.5 + 0.5
                    ).clamp(0, 1)
                    arm_features.append(clip_features(evaluator, rgb))
                    for i in examples:
                        if lo <= i < hi:
                            ToPILImage()(rgb[i - lo].cpu()).resize((244, 244)).save(
                                out / "examples" / f"{arm}_{i:03d}.jpg", quality=94
                            )
                features[arm] = np.concatenate(arm_features)
                print(f"{panel} rendered {arm}: {len(rows)}", flush=True)

            gt = []
            with zipfile.ZipFile(root / "assets/things_images_v254/training_images.zip") as archive:
                for lo in range(0, len(rows), args.batch_size):
                    batch = []
                    for i in range(lo, min(lo + args.batch_size, len(rows))):
                        rec = records[rows[i]]
                        raw = archive.read(rec["image_member"])
                        if hashlib.sha256(raw).hexdigest() != rec["image_sha256"]:
                            raise ValueError("Reference archive differs from manifest")
                        with Image.open(io.BytesIO(raw)) as im:
                            batch.append(pre(im.convert("RGB")))
                        if i in examples:
                            ToPILImage()(batch[-1]).resize((244, 244)).save(
                                out / "examples" / f"original_{i:03d}.jpg", quality=94
                            )
                    gt.append(clip_features(evaluator, torch.stack(batch)))
            gt = np.concatenate(gt)
            for arm in arms:
                _, scores = summarize(gt, features[arm], groups, rows)
                np.save(out / f"{arm}_scores.npy", scores)
                outcomes[arm] = score_object_panel(scores, rows, labels)
            write_json(out / "object_results.json", outcomes)
            save_montage(out, examples, rows, records, ["original", *arms])
            print(f"{panel} scored, galleries saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "layout", "conditions", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--words", type=Path)
    parser.add_argument(
        "--free-arms",
        nargs="+",
        choices=["free", "free_stronger", "free_shuffled", "free_stronger_shuffled"],
    )
    parser.add_argument(
        "--panels", nargs="+", choices=("dev", "confirm"), default=["dev", "confirm"]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=531451)
    run(parser.parse_args())
