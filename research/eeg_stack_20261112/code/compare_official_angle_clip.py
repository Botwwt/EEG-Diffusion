"""Compare two registered render settings using the same 200 CLIP-scored rows."""

import argparse
import json
from pathlib import Path

import numpy as np
from score_official_objects import paired_arms
from score_official_repeat_jpegs import CLIP_SHA, check, ranking, sha

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
VISUAL = ("r4_visual", "r80_visual", "wrong_visual")
ROWS = np.arange(200)


def read_scores(case, public, arms=VISUAL):
    folder = case / "clip"
    summary = json.loads((folder / "summary.json").read_bytes())
    settings = case / "render/settings.json"
    rows_file = folder / summary["per_row"]["file"]
    scores_file = folder / summary["complete_scores"]["file"]
    check(
        summary == json.loads(public.read_bytes())
        and sha(rows_file) == summary["per_row"]["sha256"]
        and sha(scores_file) == summary["complete_scores"]["sha256"]
        and sha(settings) == summary["provenance"]["render_settings_sha256"]
        and summary["provenance"]["model_sha256"] == CLIP_SHA,
        "Scored artifact or settings differs",
    )
    with rows_file.open(encoding="utf8") as f:
        rows = [json.loads(line) for line in f]
    result = {}
    with np.load(scores_file, allow_pickle=False) as z:
        dictionary, ids = z["dictionary_ids"], z["official_ids"]
        check(
            np.array_equal(z["rows"], ROWS)
            and len(set(ids)) == 200
            and len(set(dictionary)) == 1854,
            "Score matrix order or lexicon differs",
        )
        text, photo = z["text_scores"], z["photo_scores"]
        ix = {w: j for j, w in enumerate(dictionary)}
        correct = np.array([ix[w] for w in ids])
        subset = np.sort(correct)
        names = z["sources"].tolist()
        orig = names.index("original")
        original = (text[orig], photo[orig], ids, dictionary)
        for arm in arms:
            j = names.index(arm)
            rec = [r for r in rows if r["source"] == arm]
            donors = np.roll(ROWS, 67) if arm.startswith("wrong") else ROWS
            check(
                [r["eeg_row"] for r in rec] == ROWS.tolist()
                and [r["truth_id"] for r in rec] == ids.tolist()
                and [r["generation_seed"] for r in rec] == (923761 + ROWS).tolist()
                and [r["conditioning_eeg_row"] for r in rec] == donors.tolist(),
                "Unpaired CLIP rows",
            )
            values = {}
            for label, mat, y, words in (
                ("full1854", text[j], correct, dictionary),
                (
                    "restricted200",
                    text[j][:, subset],
                    np.searchsorted(subset, correct),
                    dictionary[subset],
                ),
                ("photo200", photo[j], ROWS, ids),
            ):
                reported, details, credits = ranking(mat, y, words)
                check(
                    reported == summary["summary"][arm][label]
                    and details == [r[label] for r in rec],
                    "Raw CLIP matrix disagrees with score export",
                )
                values.update({label + "_" + k: v for k, v in credits.items()})
                values[label + "_midrank"] = np.array(
                    [(r["target_rank_first"] + r["target_rank_last"]) / 2 for r in details]
                )
            diag = np.diag(photo[j])
            off = (photo[j].sum(1) - diag) / 199
            values.update(paired_photo_cosine=diag, paired_minus_wrong_photo_cosine=diag - off)
            result[arm] = values
    return (
        result,
        original,
        dict(
            summary_sha256=sha(public),
            per_row_sha256=sha(rows_file),
            raw_matrices_sha256=sha(scores_file),
            render_settings_sha256=sha(settings),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/official-repeat-object/sub-04"))
    p.add_argument("--out", type=Path, default=PUBLIC / "official_global_angle_clip_paired.json")
    a = p.parse_args()
    native, old_ref, old_sha = read_scores(a.root, PUBLIC / "official_repeat_clip.json")
    scaled, new_ref, new_sha = read_scores(
        a.root / "global-angle", PUBLIC / "official_global_angle_clip.json"
    )
    check(
        all(np.array_equal(o, n) for o, n in zip(old_ref, new_ref, strict=True)),
        "The original-photo scores or text dictionary changed between render runs",
    )
    comparisons = {}
    for arm in VISUAL:
        comparisons[arm] = paired_arms(scaled[arm], native[arm])
        for k, e in comparisons[arm].items():
            if k.endswith(("_top1", "_top5")):
                e.update(
                    native_credit=float(native[arm][k].sum()),
                    calibrated_credit=float(scaled[arm][k].sum()),
                    rows_gaining_credit=int((scaled[arm][k] > native[arm][k]).sum()),
                    rows_losing_credit=int((scaled[arm][k] < native[arm][k]).sum()),
                )

    def diff(data, control):
        return {k: v - data[control][k] for k, v in data["r80_visual"].items()}

    for control in ("r4_visual", "wrong_visual"):
        comparisons["change_in_r80_visual_minus_" + control] = paired_arms(
            diff(scaled, control), diff(native, control)
        )
    report = dict(
        participant="sub-04",
        n_rows=200,
        unique_objects=200,
        population="Each arm pairs the same official stimulus, recipient seed, conditioning EEG row, IP-Adapter radius and fixed text across two generative settings. One photograph per object.",
        contrast="Calibrated angle minus native angle. Positive top-k, cosine and negative midrank differences favor calibrated angle. Counts use fractional top-k credit for tied text embeddings.",
        uncertainty="Existing CLIP comparison procedure: 10,000 paired row bootstrap draws and 10,000 two-sided paired sign flips, seed 20261125. Unadjusted, conditional on fixed person and gallery.",
        source=dict(
            native=old_sha,
            calibrated=new_sha,
            scorer_sha256=sha(Path(__file__)),
            matching_original_reference_scores_and_dictionaries=True,
        ),
        paired=comparisons,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    print("Paired CLIP results", sha(a.out), flush=True)


if __name__ == "__main__":
    main()
