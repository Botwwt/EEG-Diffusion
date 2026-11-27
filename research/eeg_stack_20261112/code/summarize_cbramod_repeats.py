"""Verify a CBraMod repeat experiment and compare its renders with historical D2."""

import argparse
import json
from pathlib import Path

import numpy as np

import paper_aggregate as agg


PEOPLE = (6, 9)
NEW = ("native-paper", "joint", "separate", "random-joint", "joint-wrong")
ARMS = (*NEW, "personal/d2", "shared/d2")
METRICS = ("generated_name1854", "siglip_name1854", "clip_image_top1", "clip_2way", "ssim", "pixcorr")
SEED = 20261127


def read(path):
    return json.loads(path.read_bytes())


def case_dir(root, subject, arm):
    name = f"sub-{subject:02d}"
    track, separator, saved_arm = arm.partition("/")
    if separator:
        return root / "historical" / track / saved_arm / "official-r4" / name, saved_arm
    return root / "images" / name / arm, arm


def scored_images(folder, arm, subject, independent, *, diagnostic_only=False):
    render_bytes = (folder / "render.json").read_bytes()
    render = json.loads(render_bytes)
    metrics, rows, labels, words, values, score, proof = agg.read_scores(folder, "official")
    proof["render_sha256"] = agg.sha(render_bytes)
    hashes = render["pixels_sha256"]
    agg.require(render["arm"] == arm and render["subject"] == subject and render["row"] == rows.tolist()
                and render["budget"] == "official-r4" and render["repeats"] == [[0, 1, 2, 3]]
                and render["seeds"] == agg.panel_seeds("official", len(rows)).tolist()
                and render["draws"] == 1 and render["unet_calls_per_draw"] == 4 and render["prompt"] == ""
                and len(hashes) == 200 and all(agg.valid_sha(item) for item in hashes)
                and bool(render.get("diagnostic_only", False)) == diagnostic_only, "Wrong rendering inputs or generation budget")
    agg.require(score["render_sha256"] == independent["render_sha256"] == proof["render_sha256"],
                "Naming and photo scores refer to different renders")
    agg.check_scored_images(score, hashes, proof)
    agg.verify_pixels(folder, [f"{i:05d}.png" for i in range(200)], hashes)
    columns = dict(zip(metrics, values.T, strict=True))
    siglip = np.asarray(independent["per_row"], dtype=np.float64)
    agg.require(siglip.shape == (200,) and np.all((0 <= siglip) & (siglip <= 1))
                and siglip.sum() == independent["credits"], "Independent naming credit totals disagree")
    columns["siglip_name1854"] = siglip
    with np.load(folder / "observations.npz", allow_pickle=False) as observations:
        predictions = observations["generated_top_word"][agg.row_order(observations["row"], 200)]
    positive = dict(clip={str(i): predictions[i] for i in np.flatnonzero(columns["generated_name1854"])},
                    siglip={str(i): independent["top"][i] for i in np.flatnonzero(siglip)})
    return render, labels, columns, positive, proof


def trained_states(root, subject):
    directory = root / "fit" / f"sub-{subject:02d}"
    fits = {arm: read(directory / arm / "fit.json") for arm in ("joint", "separate", "random-joint")}
    first = fits["joint"]
    for arm, fit in fits.items():
        agg.require(fit["subject"] == subject and fit["arm"] == arm and fit["proof"] == first["proof"]
                    and fit["parameters"] == first["parameters"] and fit["steps"] == first["steps"] == 880
                    and fit["batch_order_sha256"] == first["batch_order_sha256"] and len(fit["history"]) == 8,
                    "Training controls did not share the measured signal or training budget")
    return fits


def run(root, independent_path, output):
    independent = read(independent_path)
    agg.require(independent["people"] == list(PEOPLE) and set(independent["totals"]) == set(ARMS)
                and independent["n_per_arm"] == 400, "Independent scores cover a different experiment")
    fits, evidence, data, positive, replay, pre_gen = {}, {}, {}, {}, {}, {}
    order = np.arange(200)
    keys = ("physical_eeg", "native_checkpoint_sha256", "generator_files_sha256", "code_sha256", "split_sha256")
    gallery = reference = None
    for subject in PEOPLE:
        fits[subject] = trained_states(root, subject)
        evidence[subject], data[subject], positive[subject] = {}, {}, {}
        pre_gen[subject] = read(root / f"pre-gen-sub-{subject:02d}.json")
        common = None
        for arm in ARMS:
            folder, saved_arm = case_dir(root, subject, arm)
            score = independent["subjects"][str(subject)][arm]
            render, labels, columns, positive[subject][arm], evidence[subject][arm] = scored_images(folder, saved_arm, subject, score)
            if gallery is None:
                gallery, reference = labels, columns["original_name1854"]
            agg.require(np.array_equal(labels, gallery) and np.array_equal(columns["original_name1854"], reference),
                        "Stimulus identities or reference naming differ across methods")
            data[subject][arm] = np.column_stack([columns[m] for m in METRICS])
            if arm == "native-paper":
                common = {key: render[key] for key in keys}
                historic, _ = case_dir(root, subject, "personal/native-paper")
                old = read(historic / "render.json")
                agg.require(old["row"] == render["row"] and old["seeds"] == render["seeds"]
                            and old["pixels_sha256"] == render["pixels_sha256"], "Native baseline replay differs")
                replay[subject] = len(render["pixels_sha256"])
            if arm in NEW:
                agg.require({key: render[key] for key in keys} == common, "New generation control used different inputs")
                fit_arm = arm.removesuffix("-wrong")
                if fit_arm in fits[subject]:
                    agg.require(render["fit"]["model_sha256"] == fits[subject][fit_arm]["checkpoint_sha256"],
                                "Fit and render receipts disagree")
            else:
                agg.require(render["training"] == arm.partition("/")[0] and render["split_sha256"] == common["split_sha256"],
                            "Historical D2 render uses a different training track or stimulus split")
            donor = np.roll(order, 100) if arm == "joint-wrong" else order
            agg.require(render["eeg_donor"] == donor.tolist(), "Wrong EEG donor for the arm")

    stacked = {arm: np.stack([data[p][arm] for p in PEOPLE]) for arm in ARMS}
    comparisons = [("joint", a) for a in ARMS if a != "joint"]
    comparisons.extend((a, "native-paper") for a in ("personal/d2", "shared/d2"))
    paired = {}
    for a, b in comparisons:
        overall, per_person = agg.describe(stacked[a], stacked[b], PEOPLE, gallery, "official", METRICS, 10000, SEED)
        paired[f"{a}_vs_{b}"] = dict(overall=overall, per_subject=per_person)
    scores = {a: dict(mean=dict(zip(METRICS, v.mean((0, 1)).tolist(), strict=True)),
                      clip_name_by_subject=v[:, :, 0].sum(1).tolist(), siglip_name_by_subject=v[:, :, 1].sum(1).tolist())
              for a, v in stacked.items()}
    for arm in ARMS:
        agg.require(sum(scores[arm]["siglip_name_by_subject"]) == independent["totals"][arm], "SigLIP2 aggregate mismatch")
    result = dict(people=PEOPLE, n_per_arm=400, new_rgb=2000, historical_d2_rgb=800, arms=ARMS,
                  scores=scores, paired=paired, reference_names=dict(clip=float(reference.sum()), siglip=independent["reference"]["credits"]),
                  reproducibility=dict(native_same_png=replay, fits=fits, render_scores=evidence,
                                       independent_score_sha256=agg.sha(independent_path.read_bytes()),
                                       independent_model=independent["model"], independent_model_files=independent["model_files_sha256"],
                                       archive_sha256=independent["images_sha256"], independent_scorer_sha256=independent["code_sha256"]),
                  pre_gen=pre_gen, positive_rows=positive)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    for arm, summary in scores.items():
        print(arm, summary["clip_name_by_subject"], summary["siglip_name_by_subject"], summary["mean"]["clip_image_top1"],
              summary["mean"]["ssim"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("receipts", "independent", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    run(args.receipts, args.independent, args.output)
