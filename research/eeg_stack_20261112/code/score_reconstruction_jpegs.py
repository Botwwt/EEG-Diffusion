"""Score saved registered support/geometry JPEGs with the frozen OpenAI CLIP-L/14."""

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from decode_eeg_layout import digest, load_json, save_json
from PIL import Image
from render_official_words import crop512
from score_object_panel import score_object_panel
from score_official_objects import load_dictionary, load_tokenizer, text_bank, topk_credit, unit

CLIP_SHA = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
METRICS = (
    "full1854",
    "closed12",
    "intra10",
    "two_photo_assignment",
    "two_photo_margin",
    "two_choice_credit",
)
SEED = 260503


def check(value, message):
    if not value:
        raise ValueError(message)


def only_file(options):
    found = [p for p in options if p.is_file()]
    if not found:
        raise FileNotFoundError("Missing any of " + ", ".join(str(p) for p in options))
    if len({digest(p) for p in found}) != 1:
        raise ValueError(f"Ambiguous files for {options[0]}")
    return found[0]


def registered(reference, panel):
    path = only_file([reference / panel / "ablation.json", reference / (panel + "_ablation.json")])
    reg = load_json(path)
    rows, folds, truth, words = (np.asarray(reg[k]) for k in ("rows", "folds", "truth", "words"))
    check(
        rows.shape == folds.shape == truth.shape == (120,)
        and len(set(rows)) == 120
        and all(x.dtype.kind in "iu" for x in (rows, folds, truth))
        and len(set(words)) == len(words) == 12
        and set(truth) == set(range(12))
        and set(folds) == set(range(5)),
        "Wrong registered 12-way panel",
    )
    calibration = np.asarray(reg["calibration_rows_by_fold_and_word"])
    check(calibration.shape == (5, 12, 8) and calibration.dtype.kind in "iu", "Wrong support axes")
    for fold, groups in enumerate(calibration):
        for c, g in enumerate(groups):
            check(
                len(set(g)) == 8
                and set(g) == set(rows[(truth == c) & (folds != fold)])
                and np.sum((truth == c) & (folds == fold)) == 2,
                "Fold needs eight distinct same-class supports and two disjoint held-out photos",
            )
    return dict(
        rows=rows,
        folds=folds,
        truth=truth,
        vocab=words,
        labels=words[truth],
        calibration=calibration,
        sha=digest(path),
    )


def geometry_prior(args, cfg, panel, subject, reg, support, records):
    check(
        args.geometry_estimates and args.geometry_features,
        "Geometry renders require --geometry-estimates and --geometry-features to verify their frozen eight-photo priors",
    )
    path = only_file(
        [
            args.geometry_estimates / subject / panel / "geometry.npz",
            args.geometry_estimates / panel / subject / "geometry.npz",
        ]
    )
    check(digest(path) == cfg["geometry_sha256"], "Wrong frozen geometry packet")
    with (
        np.load(args.geometry_features, allow_pickle=False) as bank,
        np.load(path, allow_pickle=False) as fit,
    ):
        lookup = {int(r): i for i, r in enumerate(bank["rows"])}
        check(len(lookup) == len(bank["rows"]), "Duplicate geometry bank rows")
        pos = np.array([[lookup[int(r)] for r in g] for g in support])
        check(
            np.array_equal(fit["rows"], reg["rows"])
            and np.array_equal(fit["donor_rows"], cfg["donor_rows"])
            and all(
                np.array_equal(cfg["targets"][k], fit[k]) for k in ("prior", "query", "swapped")
            )
            and np.allclose(fit["prior"], bank["geometry"][pos].mean(1), atol=1e-7, rtol=0),
            "Frozen geometry priors differ from the registered eight support photographs",
        )
        hashes = bank["jpeg_sha256"]
        check(
            all(str(hashes[lookup[int(r)]]) == records[r]["image_sha256"] for r in reg["rows"]),
            "Geometry image bytes differ from PKU originals",
        )
        check(
            str(bank["model_sha256"]) == cfg["model_sha256"],
            "Foreground measurement model hash differs",
        )
    return dict(
        geometry_sha256=digest(path), geometry_features_sha256=digest(args.geometry_features)
    )


def registry(args, root, panel, subject, reg, records, manifest_sha):
    directory = root / panel / subject
    path = directory / "settings.json"
    cfg = load_json(path)
    rows, folds, labels = reg["rows"], reg["folds"], reg["labels"]
    words = np.array(cfg["predicted_words"])
    check(
        np.array_equal(cfg["rows"], rows)
        and np.array_equal(cfg["folds"], folds)
        and words.shape == rows.shape
        and np.isin(words, reg["vocab"]).all(),
        "Render rows, words or folds disagree",
    )
    check(
        cfg.get("subject", subject) == subject and cfg.get("panel", panel) == panel,
        "Render subject/panel mismatch",
    )
    check(
        cfg.get("true_labels", labels.tolist()) == labels.tolist(),
        "Render truth labels disagree with registration",
    )
    check(
        cfg.get("manifest_sha256", manifest_sha) == manifest_sha
        and cfg.get("ablation_sha256", reg["sha"]) == reg["sha"],
        "Original manifest or ablation fingerprint differs",
    )
    lookup = {w: i for i, w in enumerate(reg["vocab"])}
    support = reg["calibration"][folds, [lookup[w] for w in words]]
    hashes = cfg["jpeg_sha256"]
    arm_names = list(hashes)
    if "support_rows" in cfg:
        primary = "matched"
        check(
            {"baseline", "matched", "mean", "swapped", "random"} <= set(arm_names),
            "Missing support render arms",
        )
        found = np.asarray(cfg["support_rows"])
        check(
            found.shape == support.shape
            and found.dtype.kind in "iu"
            and np.array_equal(np.sort(found, axis=1), np.sort(support, axis=1)),
            "Rendered supports include a query or differ from the eight registered rows",
        )
        evidence = dict(selection_sha256=cfg["selection_sha256"])
    else:
        primary = "query"
        check(
            {"baseline", "neutral", "prior", "query", "swapped"} <= set(arm_names),
            "Missing geometry render arms",
        )
        evidence = geometry_prior(args, cfg, panel, subject, reg, support, records)
    evidence["conditions_sha256"] = cfg["conditions_sha256"]
    evidence["donor_rows"] = cfg["donor_rows"]
    evidence["swap_eligible"] = cfg["swap_eligible"]
    donor = np.asarray(cfg["donor_rows"])
    loc = {int(r): i for i, r in enumerate(rows)}
    check(
        donor.shape == rows.shape and set(donor) == set(rows), "Donor rows must permute the panel"
    )
    donor_ix = np.array([loc[r] for r in donor])
    valid = donor != rows
    check(
        np.array_equal(folds[donor_ix], folds)
        and np.array_equal(words[donor_ix], words)
        and np.array_equal(cfg["swap_eligible"], valid),
        "Swap donor changes predicted word or fold",
    )
    pairs = []
    for cls in range(12):
        for fold in range(5):
            left, right = np.flatnonzero((reg["truth"] == cls) & (folds == fold))
            if words[left] == words[right]:
                pairs.append([left, right])
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    expected = {f"{i:03d}.jpg" for i in range(120)}
    for arm in arm_names:
        check(
            Path(arm).name == arm
            and arm not in {".", ".."}
            and len(hashes[arm]) == 120
            and {p.name for p in (directory / "images" / arm).glob("*.jpg")} == expected,
            "Unexpected JPEG inventory",
        )
    check(
        "source_jpeg_sha256" not in cfg or cfg["source_jpeg_sha256"] == hashes["baseline"],
        "Frozen baseline JPEG hashes differ from its sources",
    )
    return dict(
        path=directory,
        settings_sha=digest(path),
        hashes=hashes,
        primary=primary,
        words=words,
        valid=valid,
        pairs=pairs,
        support=support,
        evidence=evidence,
    )


def load_jpeg(case, arm, i):
    path = case["path"] / "images" / arm / f"{i:03d}.jpg"
    raw = path.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == case["hashes"][arm][i], f"Wrong JPEG SHA256: {path}")
    with Image.open(io.BytesIO(raw)) as im:
        check(im.format == "JPEG" and im.size == (512, 512), f"Expected 512px JPEG at {path}")
        return im.convert("RGB")


def originals(archive, records, rows):
    imgs = []
    with ZipFile(archive / "assets/things_images_v254/training_images.zip") as z:
        for r in rows:
            rec = records[r]
            data = z.read(rec["image_member"])
            check(
                len(data) == int(rec["image_bytes"])
                and hashlib.sha256(data).hexdigest() == rec["image_sha256"],
                f"Original photo bytes differ for row {r}",
            )
            with Image.open(io.BytesIO(data)) as im:
                imgs.append(crop512(im.convert("RGB")))
    return imgs


def raw_features(model, getter, n, batch):
    import torch
    from eval_instance_starts import clip_features

    values = []
    for lo in range(0, n, batch):
        arr = np.stack(
            [np.asarray(getter(i), dtype=np.uint8) for i in range(lo, min(n, lo + batch))]
        )
        pixels = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255
        values.append(clip_features(model, pixels))
    return np.concatenate(values)


def measure(x, y, reg, case, bank, ids, inverse):
    lookup = {w: i for i, w in enumerate(ids)}
    truth = np.array([lookup[w] for w in reg["labels"]])
    subset = np.unique(truth)
    n = len(truth)
    i = np.arange(n)
    scores = (unit(x) @ bank.T)[:, inverse]
    # Exact protocol of render_registered_objects: center raw outputs separately then normalize.
    centered = unit(x - x.mean(0)) @ unit(y - y.mean(0)).T
    retrieval = score_object_panel(centered, reg["rows"], reg["labels"])
    out = dict(
        full1854=topk_credit(scores, 1)[i, truth],
        closed12=topk_credit(scores[:, subset], 1)[i, np.searchsorted(subset, truth)],
        intra10=np.array([r["intra10_top1"] for r in retrieval["per_photo"]]),
    )
    # Conditional two-photo identification uses raw cosine. Query candidates share truth, prompt and fold.
    a, b = case["pairs"].T
    similarity = unit(x.astype(np.float64)) @ unit(y.astype(np.float64)).T
    first = similarity[a, a] - similarity[a, b]
    second = similarity[b, b] - similarity[b, a]
    margin = first + second

    def credit(v):
        return (v > 0).astype(float) + 0.5 * (v == 0)

    out.update(
        two_photo_assignment=credit(margin),
        two_photo_margin=margin,
        two_choice_credit=(credit(first) + credit(second)) / 2,
    )
    return out


def cluster_table(values, classes):
    return np.bincount(classes, weights=values, minlength=12), np.bincount(classes, minlength=12)


def interval(sums, counts):
    # Rows are subjects; columns are the fixed panel's true object classes.
    sums, counts = np.atleast_2d(sums), np.atleast_2d(counts)
    n = int(counts.sum())
    if not n:
        return dict(n=0, mean=None, ci95=None)
    rng = np.random.default_rng(SEED)
    persons = rng.integers(len(sums), size=(10000, len(sums)))
    active = np.flatnonzero(counts.sum(0))
    classes = rng.choice(active, size=(10000, len(active)))
    den = counts[persons[:, :, None], classes[:, None, :]].sum((1, 2))
    num = sums[persons[:, :, None], classes[:, None, :]].sum((1, 2))
    ci = (
        np.quantile(num[den > 0] / den[den > 0], [0.025, 0.975]).tolist()
        if len(active) > 1
        else None
    )
    return dict(n=n, mean=float(sums.sum() / n), ci95=ci)


def run(args):
    from score_personal_photo_layout import original_records

    args.out.parent.mkdir(parents=True, exist_ok=True)
    roots = [p.resolve() for p in args.renders]
    check(
        len(set(roots)) == len(roots) and len(set(p.name for p in roots)) == len(roots),
        "Each render root needs a distinct name",
    )
    check(
        len(set(args.panels)) == len(args.panels)
        and len(set(args.subjects)) == len(args.subjects)
        and args.batch_size > 0,
        "Duplicated case or invalid batch size",
    )
    ids, common, inverse, _, _ = load_dictionary(args.dictionary)
    steps = {p: registered(args.reference_dir, p) for p in args.panels}
    all_rows = np.concatenate([r["rows"] for r in steps.values()])
    check(len(np.unique(all_rows)) == len(all_rows), "The two panels overlap")
    labels = np.concatenate([r["labels"] for r in steps.values()]).tolist()
    records, manifest_hash = original_records(args.archive, all_rows.tolist(), labels)
    output = dict(
        mode="registry" if args.check_only else "scores",
        manifest_sha256=manifest_hash,
        dictionary_sha256=digest(args.dictionary),
        script_sha256=digest(Path(__file__)),
        cases={},
        pooled={},
        bootstrap="10,000 paired draws, seed 260503; true object clusters per case, crossed subjects and true objects for panel pooling",
        pair_rule="Both held-out rows of one true class and fold qualify only when predicted words match. Assignment compares sums of the two original-image cosine similarities for the correct versus exchanged correspondence; ties get half credit. Two-choice averages the two independent row decisions.",
    )
    model = bank = None
    if not args.check_only:
        import torch
        from eval_instance_starts import load_visual_scorer

        torch.set_grad_enabled(False)
        model, ch = load_visual_scorer(args.archive.resolve())
        check(ch == CLIP_SHA, "Unexpected OpenAI CLIP-L/14 checkpoint")
        output["clip_sha256"] = ch
        tok = load_tokenizer(
            args.tokenizer or args.archive / "assets/native_generator_v243/turbo/tokenizer"
        )
        bank = text_bank(model, tok, common, args.batch_size)
    for panel, reg in steps.items():
        references = originals(args.archive, records, reg["rows"])
        original = (
            None
            if args.check_only
            else raw_features(model, references.__getitem__, len(references), args.batch_size)
        )
        word_reference = {}
        for root in roots:
            pool = {}
            for sub in args.subjects:
                case = registry(args, root, panel, sub, reg, records, manifest_hash)
                words = case["words"].tolist()
                identity = (
                    words,
                    case["hashes"]["baseline"],
                    case["evidence"]["conditions_sha256"],
                )
                check(
                    word_reference.setdefault(sub, identity) == identity,
                    "Pipelines used different frozen words, baseline JPEGs or condition packets",
                )
                pair = case["pairs"]
                summary = dict(
                    settings_sha256=case["settings_sha"],
                    registration_sha256=reg["sha"],
                    original_sha256=[records[r]["image_sha256"] for r in reg["rows"]],
                    **case["evidence"],
                    rows=reg["rows"].tolist(),
                    true_labels=reg["labels"].tolist(),
                    folds=reg["folds"].tolist(),
                    predicted_words=words,
                    eligible_pair_rows=reg["rows"][pair].tolist(),
                    eligible_pair_count=len(pair),
                    arms={},
                    paired={},
                )
                values = {}
                for arm, hashes in case["hashes"].items():

                    def get(i, c=case, a=arm):
                        return load_jpeg(c, a, i)

                    if args.check_only:
                        for i in range(120):
                            get(i).close()
                    else:
                        x = raw_features(model, get, 120, args.batch_size)
                        values[arm] = measure(x, original, reg, case, bank, ids, inverse)
                    summary["arms"][arm] = dict(
                        jpeg_hash_list_sha256=hashlib.sha256("".join(hashes).encode()).hexdigest()
                    )
                    if not args.check_only:
                        summary["arms"][arm].update(
                            {
                                m: dict(
                                    n=len(v), total=float(v.sum()), per_query_or_pair=v.tolist()
                                )
                                for m, v in values[arm].items()
                            }
                        )
                for ctrl in values.keys() - {case["primary"]}:
                    key = case["primary"] + "_minus_" + ctrl
                    summary["paired"][key] = {}
                    for m in METRICS:
                        is_pair = m not in METRICS[:3]
                        take = (
                            np.ones(len(pair), bool)
                            if is_pair
                            else case["valid"]
                            if ctrl == "swapped"
                            else np.ones(120, bool)
                        )
                        truth = reg["truth"][pair[:, 0]] if is_pair else reg["truth"]
                        delta = (values[case["primary"]][m] - values[ctrl][m])[take]
                        s, n = cluster_table(delta, truth[take])
                        summary["paired"][key][m] = interval(s, n)
                        pool.setdefault((key, m), []).append((s, n))
                output["cases"].setdefault(root.name, {}).setdefault(panel, {})[sub] = summary
                print(
                    root.name,
                    panel,
                    sub,
                    "verified",
                    len(case["hashes"]),
                    "arms and",
                    len(pair),
                    "eligible pairs",
                    flush=True,
                )
            pooled = {}
            for (key, m), data in pool.items():
                sums, counts = zip(*data, strict=True)
                pooled.setdefault(key, {})[m] = interval(sums, counts)
            output["pooled"].setdefault(root.name, {})[panel] = pooled
    save_json(args.out, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "dictionary", "reference-dir", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--renders",
        type=Path,
        action="append",
        required=True,
        help="Repeat for multiple render roots",
    )
    p.add_argument("--subjects", nargs="+", default=["sub-04", "sub-05"])
    p.add_argument(
        "--panels",
        nargs="+",
        default=["registered", "replicate"],
        choices=["registered", "replicate"],
    )
    p.add_argument("--geometry-estimates", type=Path)
    p.add_argument("--geometry-features", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Hash JPEGs and originals and validate fold registration on CPU",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
