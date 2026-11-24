"""Generate and score all unseen photographs in the calibrated twelve-object R4 task."""

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
    relocated,
    rollout,
    sha,
    write_json,
)
from PIL import Image, ImageDraw, ImageOps
from render_official_words import crop512, read_csv
from score_object_panel import score_object_panel
from score_official_objects import load_dictionary, load_tokenizer, text_bank, topk_credit, unit
from torchvision.transforms import ToPILImage, ToTensor


@torch.inference_mode()
def run(a):
    root = a.archive.resolve()
    sys.path.insert(0, str(root / "code"))
    import PKU_HIVE_native_classes_20261109_v239 as cls
    import PKU_native_generator_20261109_v253 as native

    cls.ROOT = native.ROOT = root
    native.Path = lambda path: relocated(path, root)
    records, _, _, _ = registration(root)
    with np.load(a.conditions, allow_pickle=False) as b:
        folds = b["folds"]
        rows, words, conditional, category, starts = (
            b[k] for k in ("rows", "words", "condition", "class_only", "start")
        )
    if (
        len(np.unique(rows)) != 120
        or starts.shape != (120, 4, 8, 8)
        or conditional.shape != (120, 1024)
    ):
        raise ValueError("Expected one registered reconstruction per held-out photo")
    arms = ("same_word_swap",) if a.swap_check else ("category", "aligned", "shuffled")
    a.out.mkdir(exist_ok=True, parents=True)
    pipe, adapter, proj, empty, _ = native.load_native()
    pipe.scheduler.set_timesteps(4, device="cuda")
    pipe.vae.float().enable_slicing()
    prompts = {}
    readable = {r["unique_id"]: r["word"] for r in read_csv(a.dictionary)}
    for noun in sorted(set(words)):
        p, _, pooled, _ = pipe.encode_prompt(
            "A photo of " + readable[noun] + ".", device="cuda", do_classifier_free_guidance=False
        )
        prompts[noun] = (p.cpu(), pooled.cpu())
    pipe.text_encoder = pipe.text_encoder_2 = None
    torch.cuda.empty_cache()
    noise = torch.stack(
        [
            torch.randn(5, 4, 64, 64, generator=torch.Generator().manual_seed(923761 + int(i)))
            for i in rows
        ]
    )
    shuffle = np.random.default_rng(923762).permutation(len(rows))
    perm = np.empty_like(shuffle)
    perm[shuffle] = np.roll(shuffle, 1)
    if a.swap_check:
        perm = np.arange(len(rows))
        for fold in np.unique(folds):
            for word in np.unique(words):
                group = np.flatnonzero((folds == fold) & (words == word))
                if len(group) > 1:
                    perm[group] = np.roll(group, 1)
    features = {}
    model, clip_sha = load_visual_scorer(root)
    for arm in arms:
        ix = perm if arm in ("shuffled", "same_word_swap") else np.arange(len(rows))
        c = (category if arm == "category" else conditional)[ix]
        goal = np.zeros_like(starts) if arm == "category" else starts[ix]
        tokens = proj(adapter(torch.from_numpy(c).cuda().half()))
        blocks = []
        folder = a.out / "images" / arm
        folder.mkdir(exist_ok=True, parents=True)
        for lo in range(0, len(rows), a.batch_size):
            hi = min(lo + a.batch_size, len(rows))
            prompt = tuple(
                torch.cat([prompts[w][j] for w in words[ix[lo:hi]]]).cuda() for j in (0, 1)
            ) + (empty[2],)
            z0 = F.interpolate(
                torch.from_numpy(goal[lo:hi]).cuda(), (64, 64), mode="bilinear", align_corners=False
            )
            z = rollout(
                pipe,
                prompt,
                tokens[lo:hi],
                z0,
                noise[lo:hi].cuda(),
                first_hold=0 if arm == "category" else 0.25,
                tether=arm != "category",
            )
            rgb = (0.5 * pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample + 0.5).clamp(
                0, 1
            )
            blocks.append(clip_features(model, rgb))
            for i, im in enumerate(rgb, lo):
                ToPILImage()(im.cpu()).save(folder / f"{i:03d}.jpg", quality=95)
        features[arm] = np.concatenate(blocks)
        print(f"Rendered {arm}: {len(rows)}", flush=True)
    del pipe, adapter, proj
    torch.cuda.empty_cache()
    # Reference filenames and class labels are accessed only after all outputs are fixed.
    gt = []
    refs = []
    with zipfile.ZipFile(root / "assets/things_images_v254/training_images.zip") as z:
        for row in rows:
            record = records[row]
            data = z.read(record["image_member"])
            if hashlib.sha256(data).hexdigest() != record["image_sha256"]:
                raise ValueError("Original photograph hash differs from manifest")
            with Image.open(io.BytesIO(data)) as im:
                refs.append(crop512(im.convert("RGB")))
    for lo in range(0, len(refs), a.batch_size):
        gt.append(
            clip_features(
                model, torch.stack([ToTensor()(im) for im in refs[lo : lo + a.batch_size]])
            )
        )
    features["original"] = np.concatenate(gt)
    if a.swap_check:
        eligible = perm != np.arange(len(rows))
        labels = np.array([records[r]["concept_id"] for r in rows])
        ground = unit(features["original"])
        cos = {}
        for arm in ("aligned", "same_word_swap"):
            blocks = []
            for lo in range(0, len(rows), a.batch_size):
                imgs = []
                for i in range(lo, min(lo + a.batch_size, len(rows))):
                    with Image.open(a.out / "images" / arm / f"{i:03d}.jpg") as im:
                        imgs.append(ToTensor()(im.convert("RGB")))
                blocks.append(clip_features(model, torch.stack(imgs)))
            cos[arm] = np.sum(unit(np.concatenate(blocks)) * ground, axis=1)
        delta = cos["aligned"] - cos["same_word_swap"]
        by = np.array([delta[eligible & (labels == k)].sum() for k in np.unique(labels)])
        n = np.array([(eligible & (labels == k)).sum() for k in np.unique(labels)])
        boot = np.random.default_rng(923764).integers(len(n), size=(10000, len(n)))
        ci = np.quantile(by[boot].sum(1) / n[boot].sum(1), [0.025, 0.975])
        write_json(
            a.out / "same_word_swap.json",
            dict(
                eligible=int(eligible.sum()),
                paired_target_cosine_delta=float(delta[eligible].mean()),
                ci95=ci.tolist(),
                own_closer=int((delta[eligible] > 0).sum()),
                ties=int((delta[eligible] == 0).sum()),
                donor_rows=rows[perm].tolist(),
                rows=rows.tolist(),
                folds=folds.tolist(),
            ),
        )
        return
    ids, unique, inverse, _, _ = load_dictionary(a.dictionary)
    B = text_bank(
        model, load_tokenizer(root / "assets/native_generator_v243/turbo/tokenizer"), unique, 32
    )
    labels = [records[r]["concept_id"].split("_", 1)[1] for r in rows]
    truth = np.array([np.flatnonzero(ids == v).item() for v in labels])
    y = np.arange(len(rows))
    report = {}
    for arm, v in features.items():
        scores = (unit(v) @ B.T)[:, inverse]
        subset = np.unique(truth)
        sem = topk_credit(scores, 1)[y, truth]
        closed = topk_credit(scores[:, subset], 1)[y, np.searchsorted(subset, truth)]
        orig = features["original"]
        vis = unit(v - v.mean(0)) @ unit(orig - orig.mean(0)).T
        report[arm] = dict(
            full_top1=float(sem.sum()),
            closed_top1=float(closed.sum()),
            n=len(rows),
            per_photo_full=sem.tolist(),
            per_photo_closed=closed.tolist(),
            original_photo=score_object_panel(vis, rows, np.asarray(labels))["hits"],
        )
    write_json(
        a.out / "results.json",
        dict(
            results=report,
            rows=rows.tolist(),
            labels=labels,
            predicted_words=words.tolist(),
            condition_sha=sha(a.conditions),
            clip_sha=clip_sha,
        ),
    )
    # First and last stimulus by original row for each registered object, regardless of prediction.
    ix = np.concatenate(
        [
            np.flatnonzero(np.asarray(labels) == w)[
                np.argsort(rows[np.asarray(labels) == w])[[0, -1]]
            ]
            for w in sorted(set(labels))
        ]
    )
    for page, sub in enumerate(np.array_split(ix, 2), 1):
        tile = 130
        cw = 3 * (tile + 3) + 102
        canvas = Image.new("RGB", (2 * cw, 6 * (tile + 28)), "white")
        draw = ImageDraw.Draw(canvas)
        for j, i in enumerate(sub):
            x = j % 2 * cw
            y0 = j // 2 * (tile + 28)
            draw.text((x + 2, y0 + 20), labels[i], fill="black")
            draw.text((x + 2, y0 + 42), "EEG: " + words[i], fill="black")
            for col, (name, img) in enumerate(
                [("Original", refs[i]), ("Class only", None), ("EEG + class", None)]
            ):
                xx = x + 100 + col * (tile + 3)
                draw.text((xx, y0 + 3), name, fill="black")
                if img is None:
                    with Image.open(
                        a.out / "images" / ("category" if col == 1 else "aligned") / f"{i:03d}.jpg"
                    ) as im:
                        img = im.copy()
                canvas.paste(ImageOps.fit(img, (tile, tile)), (xx, y0 + 19))
        canvas.save(a.out / f"gallery_{page}.jpg", quality=94)
    print({k: (v["full_top1"], v["closed_top1"]) for k, v in report.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("archive", "conditions", "dictionary", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--swap-check",
        action="store_true",
        help="Swap only among queries with the same fold and predicted word; requires original aligned images",
    )
    run(p.parse_args())
