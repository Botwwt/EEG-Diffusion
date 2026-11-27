"""Verify the shared-D2 object swap experiment and report paired single-image scores."""

import argparse
from pathlib import Path

import numpy as np

import paper_aggregate as agg
from paper_d2_object_swap import ARMS, MODES, PANEL, SEED, STEPS, SWAP_WEIGHT
from paper_render_cache import save_json
from summarize_cbramod_repeats import METRICS, case_dir, read, scored_images


NATIVE = "personal/native-paper"
SCORE_ARMS = (NATIVE, *ARMS)
PAIRS = (("object-swap", "shared-d2"), ("object-swap", "object-anchor"),
         ("object-swap", "object-swap-wrong"), ("shared-d2", "shared-d2-wrong"),
         ("object-anchor", "shared-d2"), ("object-swap", NATIVE), ("shared-d2", NATIVE))


def fits(root):
    result = {mode: read(root / "fits" / f"{mode}.json") for mode in MODES}
    first, second = (result[mode] for mode in MODES)
    agg.require(first["parameters"] == second["parameters"]
                and first["training_order_sha256"] == second["training_order_sha256"]
                and first["proof"]["swap_weight"] == 0 and second["proof"]["swap_weight"] == SWAP_WEIGHT
                and {k: v for k, v in first["proof"].items() if k != "swap_weight"}
                == {k: v for k, v in second["proof"].items() if k != "swap_weight"},
                "The fitting arms must share all sources, model size and training orders")
    for mode, report in result.items():
        hist = report["history"]
        agg.require(report["mode"] == mode and len(hist) == report["steps"] == STEPS
                    and [p["subject"] for p in report["proof"]["eeg"]] == list(range(1, 11))
                    and all(np.isfinite(list(record.values())).all() and record["gradient"] > 0 for record in hist),
                    "The two shared fits must use the registered ten participants and finish all updates")
    return result


def run(root, naming, output, exchange=None):
    independent = read(naming)
    agg.require(independent["people"] == list(PANEL) and set(independent["totals"]) == set(SCORE_ARMS)
                and independent["n_per_arm"] == len(PANEL)*200, "Incorrect scored image coverage")
    weights = fits(root)
    data, receipts, positives, native_replay = {}, {}, {}, {}
    gallery = references = None
    same = list(range(200))
    wrong = np.roll(same, 100).tolist()
    common = ("physical_eeg", "generator_files_sha256", "layout_source", "code_sha256", "split_sha256", "d2_sources")
    for subject in PANEL:
        evidence = independent["subjects"][str(subject)]
        data[subject], receipts[subject], positives[subject] = {}, {}, {}
        baseline = native_split = None
        for arm in SCORE_ARMS:
            folder, saved_arm = case_dir(root, subject, arm)
            render, labels, columns, positives[subject][arm], receipts[subject][arm] = scored_images(
                folder, saved_arm, subject, evidence[arm])
            if gallery is None:
                gallery, references = labels, columns["original_name1854"]
            agg.require(np.array_equal(labels, gallery) and np.array_equal(columns["original_name1854"], references),
                        "Scores used different official stimulus registries")
            data[subject][arm] = np.column_stack([columns[m] for m in METRICS])
            if arm == NATIVE:
                agg.require(render["training"] == "personal" and render["eeg_donor"] == same,
                            "Archived Brain-HiVE is not a matched personal run")
                native_split = render["split_sha256"]
                continue
            agg.require(render["split_sha256"] == native_split,
                        "Archived Brain-HiVE uses a different stimulus split")
            environment = {k: render[k] for k in common}
            if arm == "shared-d2":
                baseline = environment
                historic_dir, _ = case_dir(root, subject, "shared/d2")
                historic = read(historic_dir / "render.json")
                agg.require(historic["seeds"] == render["seeds"] and historic["row"] == render["row"]
                            and historic["split_sha256"] == render["split_sha256"]
                            and historic["checkpoint_sha256"] == render["d2_sources"]["condition_checkpoint"]["sha256"],
                            "Archived D2 uses a different shared fit or stimulus schedule")
                native_replay[subject] = sum(a == b for a, b in zip(historic["pixels_sha256"], render["pixels_sha256"], strict=True))
            agg.require(environment == baseline and render["layout_donor"] == same
                        and render["eeg_donor"] == (wrong if arm.endswith("-wrong") else same),
                        "A generator or donor control varies more than its intended EEG condition")
            mode = arm.removesuffix("-wrong")
            if mode in MODES:
                report = weights[mode]
                agg.require(render["fit"] == dict(checkpoint_sha256=report["checkpoint_sha256"],
                            report_sha256=agg.sha((root / "fits" / f"{mode}.json").read_bytes()),
                            training_order_sha256=report["training_order_sha256"], steps=STEPS),
                            "Render names an unrelated trained checkpoint")
            else:
                agg.require(render["fit"] is None, "Shared D2 control must use the original weights")
    scores = {a: np.stack([data[p][a] for p in PANEL]) for a in SCORE_ARMS}
    summary = {a: dict(mean=dict(zip(METRICS, matrix.mean((0, 1)).tolist(), strict=True)),
                       clip_name_by_subject=matrix[:, :, 0].sum(1).tolist(), siglip_name_by_subject=matrix[:, :, 1].sum(1).tolist())
               for a, matrix in scores.items()}
    paired = {}
    for a, b in PAIRS:
        estimate, by_subject = agg.describe(scores[a], scores[b], PANEL, gallery, "official", METRICS, 10000, SEED)
        paired[f"{a}_vs_{b}"] = dict(overall=estimate, subjects=by_subject)
    exchange_summary = None
    if exchange is not None:
        measured = read(exchange)
        arms = ("shared-d2", "shared-d2-wrong", "object-swap", "object-swap-wrong")
        agg.require(measured["people"] == list(PANEL) and measured["model"] == independent["model"]
                    and measured["model_files_sha256"] == independent["model_files_sha256"]
                    and measured["donors"] == wrong and set(measured["render_sha256"]) == set(arms),
                    "EEG exchange scores used different models or donor ordering")
        for arm in arms:
            for person in PANEL:
                agg.require(measured["render_sha256"][arm][str(person)] == receipts[person][arm]["render_sha256"],
                            "EEG exchange scores were made from different generated images")
        exchange_summary = dict(artifact_sha256=agg.sha(exchange.read_bytes()), scores=measured["scores"],
                                differences=measured["differences"], code_sha256=measured["code_sha256"])
    save_json(output, dict(people=PANEL, n_per_arm=200*len(PANEL), new_rgb=len(ARMS)*200*len(PANEL),
                           scores=summary, paired=paired, fits=weights, receipts=receipts, positive_rows=positives,
                           shared_png_replay=native_replay, reference_credits=dict(clip=float(references.sum()),
                           siglip=independent["reference"]["credits"]), independent_model=independent["model"],
                           archive_sha256=independent["images_sha256"], independent_sha256=agg.sha(naming.read_bytes()),
                           exchange_effect=exchange_summary))
    for name, row in summary.items():
        print(name, row["clip_name_by_subject"], row["siglip_name_by_subject"], row["mean"]["clip_image_top1"], row["mean"]["ssim"])
    print("Shared D2 archived PNG matches", native_replay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("receipts", "independent", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--exchange", type=Path)
    args = parser.parse_args()
    run(args.receipts, args.independent, args.output, args.exchange)
