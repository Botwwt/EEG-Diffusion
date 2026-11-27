"""Audit paired token-calibration renders and report the object and photo endpoints."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import paper_aggregate as agg
from paper_condition_calibration import ARMS
from paper_render_cache import save_json
from summarize_cbramod_repeats import METRICS, PEOPLE, case_dir, read, scored_images


HISTORICAL = ("native-paper", "joint", "personal/d2", "shared/d2")
ESTIMATES = (
    ("center", "joint"), ("spread", "center"), ("spread", "joint"), ("spread", "spread-wrong"),
    ("d2-relative", "shared-d2"), ("d2-relative", "d2-relative-wrong"), ("d2-relative", "spread"),
)


def run(root, independent, prior_root, prior_names, output):
    scorer, prior = read(independent), read(prior_names)
    batch_audit = read(root / "probe/prior-batches.json")
    for key in ("people", "model", "model_files_sha256", "sources", "code_sha256", "reference"):
        agg.require(scorer[key] == prior[key], f"Original naming replay differs in {key}")
    agg.require(set(scorer["totals"]) == set(ARMS) and scorer["people"] == list(PEOPLE), "Wrong official score export")
    statistics, cal, source, images, positives, replay, breadth, data = {}, {}, {}, {}, {}, {}, {}, {}
    gallery = references = None
    order = list(range(200))
    exchanged = np.roll(order, 100).tolist()
    for participant in PEOPLE:
        key = str(participant)
        who = f"sub-{participant:02d}"
        part = root / "images" / who
        images[participant], positives[participant], breadth[participant], data[participant] = {}, {}, {}, {}
        measured_path = root / "probe" / f"conditions-{who}.json"
        measured = read(measured_path)
        calibration_path = part / "calibration.json"
        cal[participant], statistics[participant] = read(calibration_path), measured["statistics"]
        proof = measured["proof"]
        agg.require(batch_audit[key]["probe_sha256"] == measured["output_sha256"]
                    and batch_audit[key]["checkpoint_sha256"] == proof["sources"]["joint"]["model_sha256"],
                    "Historical joint precision check used different inputs")
        source[participant] = dict(probe_receipt_sha256=agg.sha(measured_path.read_bytes()),
                                   calibration_sha256=agg.sha(calibration_path.read_bytes()),
                                   condition_source=proof["sources"], probe_code_sha256=proof["code"])
        agg.require(cal[participant]["input_sha256"] == measured["output_sha256"]
                    and cal[participant]["input_receipt_sha256"] == source[participant]["probe_receipt_sha256"],
                    "Calibration was prepared from different training moments")
        layouts = {}
        renders = {}
        for arm in (*HISTORICAL, *ARMS):
            folder, saved_arm = case_dir(prior_root, participant, arm) if arm in HISTORICAL else (part / arm, arm)
            naming = (prior if arm in HISTORICAL else scorer)["subjects"][key][arm]
            render, labels, columns, positives[participant][arm], images[participant][arm] = scored_images(
                folder, saved_arm, participant, naming)
            renders[arm] = render
            if gallery is None:
                gallery, references = labels, columns["original_name1854"]
            agg.require(np.array_equal(labels, gallery) and np.array_equal(references, columns["original_name1854"]),
                        "Compared arms scored a different official photograph bank")
            data[participant][arm] = np.column_stack([columns[m] for m in METRICS])
            counter = Counter(naming["top"])
            breadth[participant][arm] = dict(unique=len(counter), most_frequent=counter.most_common(6))
            if arm in ARMS:
                agg.require(render["physical_eeg"] == proof["official"] and render["source_models"] == proof["sources"]
                            and render["split_sha256"] == proof["split_sha256"] and render["generators"] == proof["generators"]
                            and render["probe_sha256"] == measured["output_sha256"]
                            and render["eeg_donor"] == (exchanged if arm.endswith("-wrong") else order),
                            f"Control evidence changed in {folder}")
                expected = {k: v for k, v in cal[participant].items() if not k.startswith("input_")}
                agg.require(render["calibration"] == expected, f"Different moment calibration in {folder}")
                if arm == "shared-d2" or arm.startswith("d2-relative"):
                    agg.require(render["d2_eeg_donor"] == order and render["layout"] is not None, "D2 layout switched")
                    layouts[arm] = render["layout"]
                else:
                    agg.require(render["d2_eeg_donor"] is None and render["layout"] is None, "Non-D2 control used EEG layout")
        agg.require(layouts["shared-d2"] == layouts["d2-relative"] == layouts["d2-relative-wrong"],
                    "D2 layouts differ across its controls")
        agg.require(renders["shared/d2"]["checkpoint_sha256"] == proof["sources"]["d2"]["model_sha256"]
                    and renders["joint"]["fit"]["model_sha256"] == proof["sources"]["joint"]["model_sha256"],
                    "Replayed encoders differ from the original render")
        replay[participant] = sum(a == b for a, b in zip(renders["shared-d2"]["pixels_sha256"],
                                                       renders["shared/d2"]["pixels_sha256"], strict=True))
    scores = {arm: np.stack([data[p][arm] for p in PEOPLE]) for arm in (*HISTORICAL, *ARMS)}
    paired = {}
    for a, b in ESTIMATES:
        overall, subjects = agg.describe(scores[a], scores[b], PEOPLE, gallery, "official", METRICS, 10000, 20261127)
        paired[f"{a}_vs_{b}"] = dict(overall=overall, subjects=subjects)
    summary = {a: dict(mean=dict(zip(METRICS, v.mean((0, 1)).tolist(), strict=True)),
                       clip_name_by_subject=v[:, :, 0].sum(1).tolist(), siglip_name_by_subject=v[:, :, 1].sum(1).tolist())
               for a, v in scores.items()}
    save_json(output, dict(people=PEOPLE, n_per_arm=400, new_rgb=400*len(ARMS), scores=summary, paired=paired,
                           pre_generation_geometry=statistics, calibration=cal, predicted_word_breadth=breadth,
                           positive_rows=positives, shared_d2_exact_png_replay=replay, sources=source, image_receipts=images,
                           historical_joint_batch_precision=batch_audit,
                           independent_naming=dict(model=scorer["model"], raw_reference_credit=scorer["reference"]["credits"],
                                                   prior_archive=prior["images_sha256"], current_archive=scorer["images_sha256"],
                                                   current_scores=agg.sha(independent.read_bytes()), prior_scores=agg.sha(prior_names.read_bytes()))))
    for arm, stats in summary.items():
        print(arm, stats["clip_name_by_subject"], stats["siglip_name_by_subject"], stats["mean"]["clip_image_top1"], stats["mean"]["ssim"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipts", "independent", "prior-receipts", "prior-names", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.receipts, args.independent, args.prior_receipts, args.prior_names, args.output)
