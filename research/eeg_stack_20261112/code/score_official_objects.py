"""Offline CLIP-L/14 full-vocabulary object evaluation of official R80 renders."""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from eval_instance_starts import clip_features, load_visual_scorer, write_json
from PIL import Image
from render_official_words import crop512, file_sha, ordered, read_csv, reference, score_generated

TEMPLATES = ("a photo of the {}.", "a close-up photo of the {}.", "an image of the {}.")
SEED = 20261125
N_RESAMPLES = 10000


def load_dictionary(path):
    rows = read_csv(path)
    ids = [r["unique_id"] for r in rows]
    words = [r["word"].strip() for r in rows]
    manual = [{s.strip() for s in r["manual_categories"].split(",") if s.strip()} for r in rows]
    categories = sorted(set().union(*manual))
    if (
        len(rows) != 1854
        or len(set(ids)) != 1854
        or any(not word for word in words)
        or len(categories) != 27
        or {int(r["things_id"]) for r in rows} != set(range(1, 1855))
    ):
        raise ValueError("Expected the full THINGS vocabulary and its 27 manual categories")
    membership = np.array([[c in group for c in categories] for group in manual])
    unique_words, inverse = np.unique(words, return_inverse=True)
    return np.array(ids), unique_words, inverse, categories, membership


def load_tokenizer(path):
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(str(path), local_files_only=True)
    known = tok("a photo of a cat", add_special_tokens=True)["input_ids"]
    if (tok.vocab_size, tok.bos_token_id, tok.eos_token_id, tok.model_max_length) != (
        49408,
        49406,
        49407,
        77,
    ):
        raise ValueError("SDXL tokenizer does not match the original OpenAI CLIP text vocabulary")
    if known != [49406, 320, 1125, 539, 320, 2368, 49407]:
        raise ValueError(f"CLIP tokenizer failed its reference phrase: {known}")
    return tok


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(n).all() or np.any(n == 0):
        raise ValueError("Nonfinite or zero CLIP feature")
    return x / n


def text_bank(model, tok, unique_words, size):
    import torch

    if model.context_length != 77 or model.token_embedding.num_embeddings != 49408:
        raise ValueError("Incompatible OpenAI CLIP text encoder")
    encoded = []
    for template in TEMPLATES:
        features = []
        for lo in range(0, len(unique_words), size):
            texts = [template.format(x) for x in unique_words[lo : lo + size]]
            sequences = tok(texts, add_special_tokens=True, padding=False, truncation=False)[
                "input_ids"
            ]
            # Match OpenAI CLIP's zero padding and first-EOS argmax pooling.
            tokens = np.zeros((len(texts), 77), dtype=np.int64)
            for j, seq in enumerate(sequences):
                if len(seq) > 77 or seq[0] != 49406 or seq[-1] != 49407 or max(seq[1:-1]) >= 49406:
                    raise ValueError(f"Invalid CLIP sequence for {texts[j]}")
                tokens[j, : len(seq)] = seq
            features.append(
                model.encode_text(torch.from_numpy(tokens).cuda()).float().cpu().numpy()
            )
        encoded.append(unit(np.concatenate(features)))
    return unit(np.mean(encoded, axis=0))


def image_features(model, getter, n, size):
    import torch

    all_features = []
    for lo in range(0, n, size):
        pixels = np.stack(
            [np.asarray(getter(i), dtype=np.uint8) for i in range(lo, min(lo + size, n))]
        )
        if pixels.shape[1:] != (512, 512, 3):
            raise ValueError("Expected a 512-square RGB image after preprocessing")
        x = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().div_(255)
        all_features.append(clip_features(model, x))
    return unit(np.concatenate(all_features))


def fixed_gallery_rows(count=200):
    """Use the same twenty evenly spaced rows as the registered R80 render gallery."""
    if count < 20:
        raise ValueError("The blind gallery needs at least 20 rows")
    return np.array([j * (count - 1) // 19 for j in range(20)])


def render_arms(render):
    available = [(p.name, p) for p in sorted((render / "images").iterdir()) if p.is_dir()]
    if not available:
        raise ValueError(f"No EEG render arms under {render}/images")
    oracle = render / "diagnostics/oracle"
    if oracle.is_dir():
        available.append(("diagnostic-oracle", oracle))
    expected = {f"{i:03d}.jpg" for i in range(200)}
    for label, path in available:
        actual = {p.name for p in path.glob("*.jpg")}
        if actual != expected:
            raise ValueError(
                f"{render.name}/{label} differs from 200 registered rows: missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
            )
    return available


def load_jpg(folder, row):
    with Image.open(folder / f"{row:03d}.jpg") as im:
        if im.size != (512, 512):
            raise ValueError(f"Unexpected image size at {folder}/{row:03d}.jpg")
        return im.convert("RGB")


def blind_gallery(out, folders, original, include_oracle):
    out.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(SEED)
    arms = [k for k in folders if include_oracle or not k.endswith("/diagnostic-oracle")]
    choices = fixed_gallery_rows()
    thumb, gap, gutter = 176, 4, 16
    pair_w = 2 * thumb + gap
    records = []
    for number, name in enumerate(rng.permutation(arms), 1):
        # Keep filenames, two-sided positioning and the pixels on the sheet blind.
        sheet = Image.new("RGB", (2 * pair_w + gutter, 10 * thumb + 9 * gutter), "white")
        sides = rng.permutation([0] * 10 + [1] * 10)
        for j, row in enumerate(choices):
            orig = original(row)
            recon = load_jpg(folders[name], row)
            pair = (orig, recon) if sides[j] == 0 else (recon, orig)
            x, y = (j % 2) * (pair_w + gutter), (j // 2) * (thumb + gutter)
            for col, photo in enumerate(pair):
                sheet.paste(
                    photo.resize((thumb, thumb), getattr(Image, "Resampling", Image).LANCZOS),
                    (x + col * (thumb + gap), y),
                )
            records.append(
                [
                    f"sheet-{number:02d}.jpg",
                    j + 1,
                    int(row),
                    name,
                    "left" if sides[j] == 0 else "right",
                ]
            )
        sheet.save(out / f"sheet-{number:02d}.jpg", quality=95, subsampling=0)
    save_csv(
        out / "key.csv",
        ["sheet", "position_row_major", "eeg_row", "render_arm", "original_side"],
        records,
    )


def topk_credit(scores, k):
    cutoff = np.partition(scores, -k, axis=1)[:, -k, None]
    above, equal = scores > cutoff, scores == cutoff
    fraction = (k - above.sum(1)) / equal.sum(1)
    return above + equal * fraction[:, None]


def pairing(matrix):
    """Same-row mean vs all wrong rows; unrestricted row permutations give the null."""
    n = len(matrix)
    diag = np.diag(matrix)
    obs = diag.mean()
    wrong = (matrix.sum() - diag.sum()) / (n * (n - 1))
    rng = np.random.default_rng(SEED)
    ids = np.arange(n)
    exceed = 0
    for _ in range(N_RESAMPLES // 250):
        perm = rng.permuted(np.broadcast_to(ids, (250, n)), axis=1)
        exceed += int(np.sum(matrix[ids, perm].mean(1) >= obs - 1e-12))
    return dict(
        n=n,
        correct_equivalent=float(diag.sum()),
        matched=float(obs),
        all_wrong_row_mean=float(wrong),
        matched_minus_wrong=float(obs - wrong),
        permutation_one_sided_p=(exceed + 1) / (N_RESAMPLES + 1),
    )


def measurements(features, bank, inverse, membership, truth, broad_bank):
    name_scores = (features @ bank.T)[:, inverse]
    order = np.argsort(-name_scores, axis=1, kind="stable")[:, :5]
    broad_scores = features @ broad_bank.T
    broad_ix = broad_scores.argmax(1)
    eligible = membership[truth].any(1)
    matrices = {f"object_top{k}": topk_credit(name_scores, k)[:, truth] for k in (1, 5)}
    broad = topk_credit(broad_scores, 1) @ membership[truth].T
    matrices["manual_category"] = broad[np.ix_(eligible, eligible)]
    stats = {k: pairing(v) for k, v in matrices.items()}
    correct = {k: v.diagonal().copy() for k, v in matrices.items()}
    full_broad = np.full(len(truth), np.nan)
    full_broad[eligible] = correct["manual_category"]
    return stats, correct, order, broad_ix, full_broad


def paired_arms(a, b):
    result = {}
    for name, v in a.items():
        delta = v - b[name]
        rng = np.random.default_rng(SEED)
        n = len(delta)
        draws = rng.integers(0, n, (N_RESAMPLES, n))
        ci = np.quantile(delta[draws].mean(1), [0.025, 0.975])
        flips = rng.integers(0, 2, (N_RESAMPLES, n), dtype=np.int8) * 2 - 1
        p = (1 + np.sum(abs((delta * flips).mean(1)) >= abs(delta.mean()) - 1e-12)) / (
            N_RESAMPLES + 1
        )
        result[name] = dict(
            n=n, delta=float(delta.mean()), ci95=ci.tolist(), paired_swap_two_sided_p=float(p)
        )
    return result


def save_csv(path, headers, rows):
    with path.open("w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def score_photos(paths, archive, size, manifest, zipped):
    for render in paths:
        arms = [
            "oracle" if name == "diagnostic-oracle" else name for name, _ in render_arms(render)
        ]
        args = SimpleNamespace(archive=archive, out=render, batch_size=size)
        print(f"Scoring original-photo agreement for {render}", flush=True)
        score_generated(args, arms, manifest, zipped)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "dictionary", "images", "manifest", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--render", type=Path, action="append", nargs="+", required=True)
    p.add_argument(
        "--tokenizer", type=Path, help="Override the archive SDXL-Turbo tokenizer/ directory"
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--include-oracle-gallery", action="store_true")
    p.add_argument(
        "--score-photos",
        action="store_true",
        help="Also write scores.json for original-photo CLIP, RGB and SSIM inside each eligible render",
    )
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("Positive --batch-size required")
    import torch

    paths = [v.resolve() for group in args.render for v in group]
    if len(set(paths)) != len(paths):
        p.error("Duplicate render directory")
    folders = {}
    for i, path in enumerate(paths, 1):
        for arm, folder in render_arms(path):
            folders[f"{i:02d}-{path.name}/{arm}"] = folder
    ids, unique_words, inverse, categories, membership = load_dictionary(args.dictionary)
    manifest = ordered(read_csv(args.manifest))
    lookup = {v: i for i, v in enumerate(ids)}
    truth = np.array([lookup[r["unique_id"]] for r in manifest])
    if len(set(truth)) != 200:
        raise ValueError("Expected 200 distinct official concept IDs")
    tokenizer_path = args.tokenizer or args.archive / "assets/native_generator_v243/turbo/tokenizer"
    tok = load_tokenizer(tokenizer_path)
    model, weights_sha = load_visual_scorer(args.archive.resolve())
    args.out.mkdir(exist_ok=True, parents=True)
    rows, summaries, per_query = [], {}, {}
    with torch.inference_mode(), ZipFile(args.images) as zipped:

        def original(i):
            return crop512(reference(zipped, manifest[i]))

        bank = text_bank(model, tok, unique_words, args.batch_size)
        broad_bank = unit(
            np.stack([bank[inverse][membership[:, j]].mean(0) for j in range(len(categories))])
        )
        for name in ["real-originals", *folders]:
            getter = (
                original
                if name == "real-originals"
                else lambda i, folder=folders[name]: load_jpg(folder, i)
            )
            features = image_features(model, getter, len(truth), args.batch_size)
            stats, correct, top, broad_ix, broad = measurements(
                features, bank, inverse, membership, truth, broad_bank
            )
            summaries[name], per_query[name] = stats, correct
            for i, ranked in enumerate(top):
                rows.append(
                    [
                        name,
                        i,
                        ids[truth[i]],
                        ids[ranked[0]],
                        "|".join(ids[ranked]),
                        categories[broad_ix[i]],
                        correct["object_top1"][i],
                        correct["object_top5"][i],
                        "" if np.isnan(broad[i]) else broad[i],
                    ]
                )
            print(name, {k: v["matched"] for k, v in stats.items()}, flush=True)
        blind_gallery(args.out / "blind", folders, original, args.include_oracle_gallery)
    comparisons = {}
    for i, path in enumerate(paths, 1):
        tag = f"{i:02d}-{path.name}/"
        for a, b in (
            ("predicted", "shuffled"),
            ("predicted", "empty"),
            ("word_fused", "predicted"),
            ("word_fused", "shuffled"),
            ("word_fused", "empty"),
            ("selected", "first_draw"),
            ("selected", "wrong_eeg"),
            ("gaussian_image", "gaussian_global"),
        ):
            if tag + a in per_query and tag + b in per_query:
                comparisons[f"{tag}{a} minus {tag}{b}"] = paired_arms(
                    per_query[tag + a], per_query[tag + b]
                )
    primary = [k for k in folders if k.endswith("/predicted")]
    for i, a in enumerate(primary):
        for b in primary[:i]:
            comparisons[f"{a} minus {b}"] = paired_arms(per_query[a], per_query[b])
    save_csv(
        args.out / "per_row.csv",
        [
            "source",
            "eeg_row",
            "truth_id",
            "first_display_id",
            "top5_display_ids",
            "category",
            "object_top1_credit",
            "object_top5_credit",
            "category_credit",
        ],
        rows,
    )
    write_json(
        args.out / "summary.json",
        dict(
            clip="OpenAI CLIP ViT-L/14",
            clip_sha256=weights_sha,
            dictionary_sha256=file_sha(args.dictionary),
            images_zip_sha256=file_sha(args.images),
            manifest_sha256=file_sha(args.manifest),
            vocab_sha256=file_sha(tokenizer_path / "vocab.json"),
            merges_sha256=file_sha(tokenizer_path / "merges.txt"),
            scorer_sha256=file_sha(Path(__file__)),
            render_directories=[str(x) for x in paths],
            vocabulary_size=len(ids),
            distinct_name_prompts=len(unique_words),
            templates=TEMPLATES,
            categories=categories,
            object_metric="Cosine to normalized average of three unit OpenAI CLIP text features per human-readable THINGS name. Expected top-k credit splits ties at the boundary. Display IDs break ties by dictionary order.",
            broad_metric="Nearest centroid of text features of all member concepts per THINGS manual category. Any annotated category is accepted; empty annotations excluded.",
            matching_test="All wrong-row average plus one-sided Monte Carlo target-label permutation over eligible stimuli (allows fixed points); assumes exchangeability across rows.",
            arm_test="Two-sided paired swap and paired percentile bootstrap per stimulus. Unadjusted, fixed dictionary and model.",
            resamples=N_RESAMPLES,
            seed=SEED,
            fixed_gallery_rows=fixed_gallery_rows().tolist(),
            original_preprocess="Short edge to 512 bilinear; center crop 512; shared CLIP-L/14 bilinear 224.",
            oracle_role="Truth-word diagnostic only",
            means=summaries,
            paired=comparisons,
        ),
    )
    if args.score_photos:
        del model
        torch.cuda.empty_cache()
        with ZipFile(args.images) as zipped:
            score_photos(paths, args.archive.resolve(), args.batch_size, manifest, zipped)


if __name__ == "__main__":
    main()
