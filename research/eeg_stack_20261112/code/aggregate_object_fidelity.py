"""Publish aggregate object-fidelity evidence from completed local score exports.

Run from the repository root. Additional official CLIP object / original-photo
score exports can be added with --objects NAME=FILE or --photos NAME=FILE.
Paths in those options are relative to --inputs (default outputs/).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from score_object_panel import paired_object_panel


def keep(data, *keys):
    return {k: data[k] for k in keys}


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    if isinstance(x, float):
        return round(x, 7)
    return x


def concept_ci(values, labels):
    by = np.array([values[labels == label].mean() for label in np.unique(labels)])
    boot = np.random.default_rng(20261125).integers(len(by), size=(10000, len(by)))
    return dict(
        mean=float(values.mean()), ci95=np.quantile(by[boot].mean(1), [0.025, 0.975]).tolist()
    )


def registered(eeg, images):
    truth = np.array(eeg["truth"])
    predicted = np.array(eeg["predicted"])
    labels = np.array(images["labels"])
    stages = images["results"]
    n = eeg["total"]
    if (
        eeg["rows"] != images["rows"]
        or not np.array_equal(labels, np.array(eeg["words"])[truth])
        or int((truth == predicted).sum()) != eeg["exact_hits"]
        or n != len(truth)
        or len(set(eeg["rows"])) != n
    ):
        raise ValueError("Registered EEG and render exports do not share the same scored rows")
    summary = {}
    for k, arm in stages.items():
        if arm["n"] != n or not np.isclose(sum(arm["per_photo_full"]), arm["full_top1"]):
            raise ValueError("Registered image count mismatch")
        summary[k] = dict(
            full1854_top1_credits=arm["full_top1"],
            closed12_top1_credits=arm["closed_top1"],
            original_gallery_credits=arm["original_photo"],
        )
    paired = {}
    for a, b in (("aligned", "shuffled"), ("aligned", "category")):
        paired[a + "_minus_" + b] = {
            k: concept_ci(np.array(stages[a][v]) - np.array(stages[b][v]), labels)
            for k, v in (("full1854_top1", "per_photo_full"), ("closed12_top1", "per_photo_closed"))
        }
    category = np.asarray(stages["category"]["per_photo_closed"])
    aligned = np.asarray(stages["aligned"]["per_photo_closed"])
    shared = dict(
        rows_equal=int(np.sum(category == aligned)),
        both_correct_credits=float(np.minimum(category, aligned).sum()),
        aligned_credit_equals_eeg_correct_rows=int(np.sum(aligned == (predicted == truth))),
    )
    return dict(
        n_photos=n,
        concepts=len(set(labels)),
        folds=len(set(eeg["folds"])),
        repeats=eeg["trial_count"],
        calibration_photos_per_object=eeg["n_calibration_photos_per_object"],
        generic_train_photos=eeg["generic_train_photos"],
        closed12_eeg_hits=eeg["exact_hits"],
        closed12_eeg_rate=concept_ci((truth == predicted).astype(float), labels),
        mix=eeg["mix"],
        renders=summary,
        closed12_category_vs_aligned=shared,
        paired=paired,
    )


def panel(arms, aligned, shuffled):
    ref = arms[aligned]
    data = dict(
        n_photos=ref["queries"],
        concepts=ref["n_concepts"],
        photos_per_concept=ref["examples_per_concept"],
        chance=ref["chance"],
        credits={k: v["hits"] for k, v in arms.items()},
    )
    for name, arm in arms.items():
        if arm["queries"] != ref["queries"] or any(
            not np.isclose(arm["hits"][k], arm["mean"][k] * arm["queries"]) for k in arm["hits"]
        ):
            raise ValueError(f"Panel hit counts disagree for {name}")
    if shuffled:
        pairs = paired_object_panel(arms[shuffled], ref)
        data["matched_arm"] = aligned
        data["control_arm"] = shuffled
        data["paired_matched_minus_shuffled"] = {
            k: keep(v, "delta", "ci95") for k, v in pairs.items()
        }
    return data


def official_objects(src):
    stats = {}
    for name, arm in src["means"].items():
        stats[name] = {
            m: keep(v, "n", "correct_equivalent", "all_wrong_row_mean", "permutation_one_sided_p")
            for m, v in arm.items()
        }
    controls = {"shuffled", "wrong_eeg", "first_draw", "gaussian_global", "raw", "predicted"}
    pairs = {
        name: metrics
        for name, metrics in src["paired"].items()
        if " minus " in name and name.split(" minus ", 1)[1].rsplit("/", 1)[-1] in controls
    }
    return dict(
        vocabulary=src["vocabulary_size"],
        unique_prompt_names=src["distinct_name_prompts"],
        manual_categories=len(src["categories"]),
        credit="Expected correct count with split boundary ties.",
        means=stats,
        paired=pairs,
    )


def official_photos(src):
    n = len(src["query_rows"])
    if n != 200 or set(src["query_rows"]) != set(range(n)):
        raise ValueError("Expected every official test stimulus once")
    means = {
        arm: {**v, "clip_top1_200_credits": float(sum(src["per_query"][arm]["clip_top1_200"]))}
        for arm, v in src["means"].items()
    }
    return dict(n=n, means=means, paired=src["paired"])


def registered_followups(eeg, ablation, color, objects):
    if any(src["rows"] != eeg["rows"] for src in (ablation, color, objects)):
        raise ValueError("A registered follow-up scored a different row list")
    ab = dict(
        closed12_method_hits={k: v["hits"] for k, v in ablation["results"].items()},
        n=eeg["total"],
        fused_minus={
            k: keep(v, "delta_fraction", "ci95") for k, v in ablation["fused_minus"].items()
        },
    )
    if ab["closed12_method_hits"]["fused"] != eeg["exact_hits"]:
        raise ValueError("Fused ablation does not reproduce the registered EEG result")
    controls = (
        "aligned minus category",
        "color minus aligned_jpeg",
        "color minus train_rgb",
        "color minus train_layout",
        "color minus word_swap",
        "color minus word_swap among same true category donors",
    )
    paired = {}
    for name in controls:
        n = color["n"]
        if name.endswith("same true category donors"):
            n = color["n_swappable_same_true_class"]
        elif name.endswith("word_swap"):
            n = color["n_swappable"]
        paired[name] = dict(n=n, metrics=color["paired"][name])
    color_summary = dict(
        units="RGB in [0,1]; mean-channel MAE and 32x32 RGB RMSE lower is better; luminance correlation higher is better.",
        means=color["metrics"],
        metric_n=color["metric_rows"],
        paired=paired,
    )
    corrected = objects["results"]["color"]
    semantic = dict(
        n=corrected["n"],
        reference=objects["retention_reference"],
        **{
            space: keep(
                corrected[space],
                "top1_credit",
                "aligned_top1_on_same_rows",
                "retained_aligned_credit",
                "lost_credit",
                "gained_credit",
            )
            for space in ("full1854", "closed12")
        },
    )
    return dict(eeg_ablation=ab, color=color_summary, jpeg_rescored_color_object_identity=semantic)


def generate(root, extra_objects, extra_photos):
    receipts = {}

    def read(rel):
        raw = (root / rel).read_bytes()
        receipts[rel] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def local(name):
        return read("object-fidelity/" + name + ".json")

    r4 = {}
    closed_words = []
    for group in ("registered", "replicate"):
        eeg = local(group + "_eeg")
        closed_words.append(set(eeg["words"]))
        r4[group] = registered(eeg, local(group + "_images"))
        r4[group].update(
            registered_followups(
                eeg,
                local(group + "_ablation"),
                local(group + "_color"),
                local(group + "_color_objects"),
            )
        )
        if eeg.get("replication_seed") is not None:
            r4[group]["concept_selection_seed"] = eeg["replication_seed"]
    if closed_words[0] & closed_words[1]:
        raise ValueError("Replication reuses registered target concepts")
    swap = local("registered_swap")
    first = local("registered_eeg")
    by_row = {
        row: (fold, pred, truth)
        for row, fold, pred, truth in zip(
            first["rows"], first["folds"], first["predicted"], first["truth"], strict=True
        )
    }
    swaps = [
        (by_row[i], by_row[j])
        for i, j in zip(swap["rows"], swap["donor_rows"], strict=True)
        if i != j
    ]
    if len(swaps) != swap["eligible"] or any(a[:2] != b[:2] for a, b in swaps):
        raise ValueError("Swap must preserve both fold and predicted object")
    r4["within_predicted_word_and_fold_swap"] = dict(
        **keep(swap, "eligible", "paired_target_cosine_delta", "ci95", "own_closer", "ties"),
        same_true_class_in_eligible=sum(a[2] == b[2] for a, b in swaps),
        unit="Original-photo CLIP cosine for own continuous features minus exchanged features. Eligible rows share predicted word and fold; bootstrap samples concepts and weights by eligible rows.",
    )
    held = {}
    identities = {}
    for name, source, aligned, control in (
        ("dev", "dev_object_results", "selected", "shuffled"),
        ("confirm", "confirm_object_results", "selected", "shuffled"),
        ("free_dev", "free_dev_results", "free_stronger", None),
        ("free_confirm", "free_confirm_results", "free_stronger", "free_stronger_shuffled"),
    ):
        arms = local(source)
        held[name] = panel(arms, aligned, control)
        identities[name] = {(p["row"], p["concept"]) for p in arms[aligned]["per_photo"]}
    if (
        identities["free_dev"] != identities["dev"]
        or identities["free_confirm"] != identities["confirm"]
        or {c for _, c in identities["dev"]} & {c for _, c in identities["confirm"]}
    ):
        raise ValueError("Development and confirmation panels disagree with the registered split")
    word = local("r4_words")
    held["full_dictionary_eeg"] = dict(
        vocabulary=word["candidates"], selected=word["selected"], panels={}
    )
    for name, p in word["panels"].items():
        ranks = np.array(p["true_class_ranks"])
        held["full_dictionary_eeg"]["panels"][name] = dict(
            n=len(ranks),
            top1_hits=int((ranks == 1).sum()),
            top5_hits=int((ranks <= 5).sum()),
            median_rank=float(np.median(ranks)),
        )
    objects = {
        "word_and_continuous": official_objects(local("official_objects")),
        "reranked_and_external": official_objects(local("official_final_objects")),
        "ip_adapter": official_objects(local("official_ip_objects")),
    }
    photos = {
        k: official_photos(local("official_" + k + "_scores"))
        for k in ("word", "continuous", "rerank", "external", "ip")
    }
    rerank = read("object-fidelity/reranked/report.json")
    rerank_selection = dict(
        n_candidates=len(rerank["ordered_candidate_sources"]),
        **keep(rerank, "selected_gamma", "selected_mix", "selection"),
    )
    for options, target, fn in (
        (extra_objects, objects, official_objects),
        (extra_photos, photos, official_photos),
    ):
        for spec in options:
            label, rel = spec.split("=", 1)
            if label in target or Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError("Use a unique condition name and an input-relative path")
            target[label] = fn(read(rel))
    official = "official-object-readout/sub-03/"
    word_arms = {}
    for k, rel in {
        "text": official + "evaluation/summary.json",
        "train_visual_anchors": "official-visual-anchors/sub-03/summary.json",
        "train_eeg_anchors": official + "eeg-anchors/evaluation/summary.json",
    }.items():
        src = read(rel)
        word_arms[k] = dict(
            n=200,
            broad_n=src.get("broad_eligible_n", src.get("broad_eligible")),
            **keep(src, "top1_hits", "top5_hits", "median_rank", "broad_correct"),
        )
    cohort = read("official-visual-anchors/cohort10/summary.json")
    fused = dict(
        subjects=10,
        same_stimulus_average=True,
        **keep(
            cohort, "n", "top1_hits", "top5_hits", "median_rank", "broad_correct", "broad_eligible"
        ),
    )
    external = read("external-object-anchors/sub-03/summary.json")
    word_arms["external_reference_assisted"] = keep(
        external,
        "n",
        "vocabulary",
        "eligible_anchors",
        "masked_truth_rows",
        "top1",
        "top5",
        "median_rank",
        "broad_correct",
        "broad_evaluable",
        "broad_minus_text",
        "paired_with_text",
    )
    collision = read("external-object-anchors/sub-03/collision_audit.json")
    word_arms["external_reference_assisted"]["collision_check"] = dict(
        **keep(collision, "comparisons", "eligible", "thresholds"),
        n_globally_excluded=len(collision["global_excluded_indices"]),
    )
    gallery = read(official + "atm-prior/evaluation_only_image_gallery.json")
    likelihood = read(official + "image-likelihood/evaluation_only_official_gallery.json")
    return dict(
        subject="sub-03",
        ci_conventions={
            "R4": "95% percentile bootstrap, 10,000 resamples of whole concepts; single subject and fixed image galleries. Each arm contrast keeps rows paired. Deltas are fractions unless a metric names another unit.",
            "R80": "95% paired percentile bootstrap over official stimulus rows. Deltas are fractions for classification, native units for cosine, correlation and SSIM; tests are unadjusted. Category scoring excludes six unlabeled stimuli.",
        },
        registered_R4=dict(
            protocol="Closed 12-way EEG calibration. Five photo-disjoint folds; eight other images of each target class calibrate each fold. Replication uses a disjoint seeded list of 12 classes. Rendered objects also scored against the full dictionary.",
            **r4,
        ),
        held_concept_R4=dict(
            protocol="Twenty development and twenty separate confirmation concepts, ten photos each. Confirmation comes from historical V excluding development. Object/photo/within-concept retrieval use 20/200/10 candidates respectively. Image oracle uses target image information.",
            **held,
        ),
        official_R80=dict(
            protocol="Official 200 test objects, one photo per object, author-released 80-repeat ATM embeddings. Gallery retrieval, full-vocabulary word prediction and pixel-based reconstruction scores are distinct tasks.",
            encoder_provenance="The ATM authors' public retrieval code monitors official test performance over epochs. The exact epoch of the released feature export is unknown.",
            full1854_word_readout=word_arms,
            official_image_gallery=dict(
                n=200,
                candidates=200,
                frozen_prior={
                    k: keep(v, "top1", "top5", "median_rank", "2afc") for k, v in gallery.items()
                },
                image_likelihood=likelihood,
            ),
            independent_clip_l14_object_scores=objects,
            original_photo_scores=photos,
            candidate_reranking=rerank_selection,
        ),
        cohort10_R80_train_visual_anchors=fused,
        source_sha256=receipts,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=Path("outputs"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/object_fidelity/summary.json",
    )
    p.add_argument("--objects", action="append", default=[], metavar="NAME=FILE")
    p.add_argument("--photos", action="append", default=[], metavar="NAME=FILE")
    args = p.parse_args()
    result = generate(args.inputs, args.objects, args.photos)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(
        json.dumps(rounded(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
        newline="\n",
    )
