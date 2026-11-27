"""Verify and compare stage diagnostics and O13 composition on the official pictures."""

import argparse
from pathlib import Path

import numpy as np

import paper_aggregate as agg
import paper_o13_composition as o13
import paper_stage_reference as stage
from paper_render_cache import save_json
from summarize_cbramod_repeats import METRICS, read, scored_images


PEOPLE = (4, 7)
ROWS = list(range(200))
DONORS = np.roll(ROWS, 100).tolist()
BASE = stage.BASE


def check_stage(records):
    sources = {}
    for arm, render in records.items():
        plan = stage.condition_plan(arm)
        supplied = render["condition_sources"]
        agg.require(render["step_condition"] == list(plan) and set(supplied) == set(plan)
                    and render["timesteps"] == list(stage.TIMESTEPS),
                    "A substitution changed the wrong steps or used an extra condition")
        for name, metadata in supplied.items():
            agg.require(metadata == sources.setdefault(name, metadata), "A source differs across render arms")
    for name, groups in (("eeg4", stage.R4), ("eeg76", stage.R76)):
        provided = sources[name]
        agg.require(provided["physical_repeats"] == groups and provided["eeg_donor"] == ROWS
                    and provided["model"] == sources["eeg4"]["model"],
                    "The EEG reference must vary only the physical repeats")
    agg.require(sources["photo"]["visual_donor"] == ROWS and sources["other-photo"]["visual_donor"] == DONORS
                and sources["photo"]["sources"] == sources["other-photo"]["sources"],
                "The two photo controls must use the same reference features in different orders")


def check_o13(records):
    base = records[BASE]
    for arm, render in records.items():
        donor = None if arm in (BASE, "o13-direct") else DONORS if arm.endswith("-wrong") else ROWS
        agg.require(render["eeg_donor"] == ROWS and render["composition_donor"] == donor
                    and render["training_assets"] == base["training_assets"]
                    and render["d2_sources"] == base["d2_sources"]
                    and render["mix"] == o13.MIX and render["layout_gain"] == o13.LAYOUT_GAIN
                    and "condition_sources" not in render,
                    "The single-query composition comparison changed a shared input")


def summary(root, independent, historical, output, experiment):
    is_stage = experiment == "stage"
    arms = stage.ARMS if is_stage else o13.ARMS
    pairs = [(name, BASE) for name in arms if name != BASE]
    if is_stage:
        pairs += [("photo-all", "other-photo-all"), ("photo-999", "photo-749")]
    else:
        pairs += [("o13-compose", "o13-direct"), ("d2-o13-compose", "d2-o13-compose-wrong")]
    naming = read(independent)
    agg.require(naming["people"] == list(PEOPLE) and set(naming["totals"]) == set(arms)
                and naming["n_per_arm"] == len(PEOPLE) * len(ROWS), "Wrong independent coverage")
    common = ("physical_eeg", "split_sha256", "layout_source", "layout_donor", "seeds",
              "generator_files_sha256", "render_sources", "runtime")
    seq, saved, positives, replay = {}, {}, {}, {}
    labels = None
    for person in PEOPLE:
        seq[person], saved[person], positives[person] = {}, {}, {}
        records = {}
        for arm in arms:
            folder = root / "images" / f"sub-{person:02d}" / arm
            render, target, columns, positives[person][arm], saved[person][arm] = scored_images(
                folder, arm, person, naming["subjects"][str(person)][arm], diagnostic_only=is_stage and arm != BASE)
            if labels is None:
                labels = target
            agg.require(np.array_equal(target, labels) and render["layout_donor"] == ROWS
                        and all(render[key] == records.get(BASE, render)[key] for key in common),
                        "A comparator changes layout, stimulus order or generator")
            if is_stage:
                agg.require(render["layout_sha256"] == records.get(BASE, render)["layout_sha256"],
                            "A stage substitution altered the starting latent")
            records[arm] = render
            seq[person][arm] = np.column_stack([columns[field] for field in METRICS])
        (check_stage if is_stage else check_o13)(records)
        original = read(historical / f"sub-{person:02d}" / BASE / "render.json")
        rendered = records[BASE]
        for key in ("physical_eeg", "split_sha256", "layout_source", "layout_donor", "row", "seeds",
                    "repeats", "generator_files_sha256"):
            agg.require(rendered[key] == original[key], "The original shared D2 render was not matched")
        expected = original["pixels_sha256"]
        replay[person] = sum(a == b for a, b in zip(expected, rendered["pixels_sha256"], strict=True))
    scores = {arm: np.stack([seq[p][arm] for p in PEOPLE]) for arm in arms}
    means = {}
    for arm, value in scores.items():
        means[arm] = dict(scores=dict(zip(METRICS, value.mean((0, 1)).tolist(), strict=True)),
                          clip_name_by_subject=value[:, :, 0].sum(1).tolist(),
                          siglip_name_by_subject=value[:, :, 1].sum(1).tolist())
    results = {}
    for a, b in pairs:
        overall, by_subject = agg.describe(scores[a], scores[b], PEOPLE, labels, "official", METRICS, 10000, 20261128)
        results[f"{a}_vs_{b}"] = dict(overall=overall, by_subject=by_subject)
    save_json(output, dict(experiment=experiment, people=PEOPLE, n_per_arm=len(PEOPLE) * len(ROWS),
                           scores=means, paired=results, render_receipts=saved, positive_rows=positives,
                           replay_matches=replay, independent_sha256=agg.sha(independent.read_bytes()),
                           independent_model=naming["model"], archive_sha256=naming["images_sha256"]))
    for arm, row in means.items():
        print(arm, row["clip_name_by_subject"], row["siglip_name_by_subject"],
              row["scores"]["clip_image_top1"], row["scores"]["ssim"])
    print("Shared D2 PNG matches", replay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("stage", "o13"))
    for argument in ("receipts", "independent", "historical", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    summary(args.receipts, args.independent, args.historical, args.output, args.experiment)
