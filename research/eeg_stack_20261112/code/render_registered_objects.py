"""Generate and score all unseen photographs in the calibrated twelve-object R4 task."""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def load_packet(path):
    with np.load(path, allow_pickle=False) as b:
        data = SimpleNamespace(
            **{k: b[k] for k in ("folds", "rows", "words", "condition", "class_only", "start")}
        )
    if (
        data.rows.dtype.kind not in "iu"
        or data.rows.shape != (120,)
        or len(np.unique(data.rows)) != 120
        or np.any((data.rows < 0) | (data.rows >= 16540))
        or data.folds.shape != (120,)
        or data.words.shape != (120,)
        or data.start.shape != (120, 4, 8, 8)
        or data.condition.shape != (120, 1024)
        or data.class_only.shape != data.condition.shape
        or not all(np.isfinite(x).all() for x in (data.start, data.condition, data.class_only))
    ):
        raise ValueError(f"Expected one registered reconstruction per held-out photo in {path}")
    return data


def load_generator(root, dictionary, words):
    import torch

    sys.path.insert(0, str(root / "code"))
    import PKU_HIVE_native_classes_20261109_v239 as cls
    import PKU_native_generator_20261109_v253 as native

    cls.ROOT = native.ROOT = root
    native.Path = lambda path: relocated(path, root)
    pipe, adapter, proj, empty, _ = native.load_native()
    pipe.scheduler.set_timesteps(4, device="cuda")
    pipe.vae.float().enable_slicing()
    prompts = {}
    readable = {r["unique_id"]: r["word"] for r in read_csv(dictionary)}
    for noun in sorted(set(words)):
        p, _, pooled, _ = pipe.encode_prompt(
            "A photo of " + readable[noun] + ".", device="cuda", do_classifier_free_guidance=False
        )
        prompts[noun] = (p.cpu(), pooled.cpu())
    pipe.text_encoder = pipe.text_encoder_2 = None
    torch.cuda.empty_cache()
    return pipe, adapter, proj, empty, prompts


def generate(a, packet, modules, model):
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import ToPILImage

    pipe, adapter, proj, empty, prompts = modules
    rows, folds, words = packet.rows, packet.folds, packet.words
    conditional, category, starts = packet.condition, packet.class_only, packet.start
    if a.swap_check:
        arms = ("same_word_swap",)
    elif a.aligned_only:
        arms = ("aligned",)
    else:
        arms = ("category", "aligned", "shuffled")
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
    return features, perm


def original_features(root, records, rows, model, batch_size):
    import torch
    from torchvision.transforms import ToTensor

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
    for lo in range(0, len(refs), batch_size):
        gt.append(
            clip_features(model, torch.stack([ToTensor()(im) for im in refs[lo : lo + batch_size]]))
        )
    return refs, np.concatenate(gt)


def report_case(a, packet, generated, records, originals, model, clip_sha, vocabulary):
    rows, words, folds = packet.rows, packet.words, packet.folds
    features, perm = generated
    refs, features["original"] = originals
    if a.swap_check:
        import torch
        from torchvision.transforms import ToTensor

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
    ids, inverse, B = vocabulary
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
        retrieval = score_object_panel(vis, rows, np.asarray(labels))
        report[arm] = dict(
            full_top1=float(sem.sum()),
            closed_top1=float(closed.sum()),
            n=len(rows),
            per_photo_full=sem.tolist(),
            per_photo_closed=closed.tolist(),
            original_photo=retrieval["hits"],
        )
        if a.aligned_only:
            report[arm]["per_photo_intra10"] = [r["intra10_top1"] for r in retrieval["per_photo"]]
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
    if a.aligned_only:
        print(a.out, {k: (v["full_top1"], v["closed_top1"]) for k, v in report.items()}, flush=True)
        return
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


def verify_replay(out, reference):
    prior = json.loads((reference / "results.json").read_text(encoding="utf-8"))
    replay = json.loads((out / "results.json").read_text(encoding="utf-8"))
    if not {"aligned", "category", "shuffled"} <= set(prior["results"]):
        raise ValueError(f"Expected an original three-arm comparison in {reference}")
    checks = {
        k: replay[k] == prior[k]
        for k in ("rows", "labels", "predicted_words", "condition_sha", "clip_sha")
    }
    for arm in ("aligned", "original"):
        for k in (
            "n",
            "full_top1",
            "closed_top1",
            "per_photo_full",
            "per_photo_closed",
            "original_photo",
        ):
            checks[f"{arm}.{k}"] = replay["results"][arm][k] == prior["results"][arm][k]
    mismatches = []
    for i, row in enumerate(prior["rows"]):
        rel = Path("images/aligned") / f"{i:03d}.jpg"
        if sha(out / rel) != sha(reference / rel):
            mismatches.append(row)
    result = dict(
        passed=all(checks.values()) and not mismatches,
        identical_fields=checks,
        n=len(prior["rows"]),
        identical_jpegs=len(prior["rows"]) - len(mismatches),
        differing_jpeg_rows=mismatches,
        reference_results_sha256=sha(reference / "results.json"),
    )
    write_json(out / "replay_check.json", result)
    return result["passed"]


def run(a):
    pairs = a.case or [(a.conditions, a.out)]
    references = a.verify_against or []
    destinations = {out.resolve() for _, out in pairs}
    if len(destinations) != len(pairs) or destinations.intersection(
        p.resolve() for p in references
    ):
        raise ValueError(
            "Each case requires a separate output folder from all cases and references"
        )
    for reference in references:
        if not (reference / "results.json").is_file() or any(
            not (reference / "images/aligned" / f"{i:03d}.jpg").is_file() for i in range(120)
        ):
            raise ValueError(f"Incomplete prior three-arm render at {reference}")
    cases = []
    for conditions, out in pairs:
        if a.aligned_only and any(
            (out / "images" / arm).exists() for arm in ("category", "shuffled")
        ):
            raise ValueError(
                f"Choose a separate aligned-only output to preserve the prior render at {out}"
            )
        cases.append(
            (
                SimpleNamespace(**(vars(a) | {"conditions": conditions, "out": out})),
                load_packet(conditions),
            )
        )
    import torch

    root = a.archive.resolve()
    records, _, _, _ = registration(root)
    with torch.inference_mode():
        modules = load_generator(
            root, a.dictionary, np.concatenate([packet.words for _, packet in cases])
        )
        model, clip_sha = load_visual_scorer(root)
        generated = [generate(case, packet, modules, model) for case, packet in cases]
        del modules
        torch.cuda.empty_cache()
        # All query images in this invocation are fixed before opening a reference photograph.
        vocabulary = None
        if not a.swap_check:
            ids, unique, inverse, _, _ = load_dictionary(a.dictionary)
            bank = text_bank(
                model,
                load_tokenizer(root / "assets/native_generator_v243/turbo/tokenizer"),
                unique,
                32,
            )
            vocabulary = ids, inverse, bank
        galleries = {}
        for (case, packet), output in zip(cases, generated, strict=True):
            key = tuple(packet.rows)
            if key not in galleries:
                galleries[key] = original_features(root, records, packet.rows, model, a.batch_size)
            report_case(case, packet, output, records, galleries[key], model, clip_sha, vocabulary)
    if references:
        failed = [
            case.out
            for (case, _), reference in zip(cases, references, strict=True)
            if not verify_replay(case.out, reference)
        ]
        if failed:
            raise ValueError(
                f"Replay differs from the three-arm run; see replay_check.json in {failed}"
            )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("archive", "dictionary"):
        p.add_argument("--" + k, type=Path, required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--conditions", type=Path)
    inputs.add_argument(
        "--case",
        type=Path,
        nargs=2,
        action="append",
        metavar=("CONDITIONS", "OUT"),
        help="Repeat to render multiple condition packets with one generator and scorer",
    )
    p.add_argument("--out", type=Path, help="Output directory for --conditions")
    p.add_argument("--batch-size", type=int, default=8)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--swap-check",
        action="store_true",
        help="Swap only among queries with the same fold and predicted word; requires original aligned images",
    )
    mode.add_argument(
        "--aligned-only", action="store_true", help="Generate and score one matched JPEG per row"
    )
    p.add_argument(
        "--verify-against",
        type=Path,
        nargs="+",
        metavar="THREE_ARM_RENDER",
        help="Compare every JPEG and shared score to the previous full render for each case, in case order",
    )
    a = p.parse_args(argv)
    if (a.conditions is not None) != (a.out is not None):
        p.error("Use --conditions with --out, or supply repeating --case pairs")
    if a.batch_size < 1:
        p.error("Positive --batch-size required")
    if a.verify_against and (
        not a.aligned_only or len(a.verify_against) != len(a.case or [a.conditions])
    ):
        p.error("--verify-against requires --aligned-only and one prior render per case")
    return a


if __name__ == "__main__":
    run(parse_args())
