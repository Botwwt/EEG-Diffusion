"""Compare same-stimulus single-person and cohort JPEGs on the two registered session-matched panels."""

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from render_official_words import crop512
from render_within_photo_ip import check, packet
from score_fresh_photo import run as score_pixels
from score_fresh_photo import validate_packet
from score_prospective_readouts import strict
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_within_photo_dino import ARMS, FILES, read, reference, sha

PANELS = {"stress12": ("ordinary_12way", 12), "new_three4": ("within_matched_four", 4)}
MODES = ("sub-04", "cohort-3")
SEED = 260532
PUBLIC = Path(__file__).resolve().parents[1] / "results" / "object_fidelity"
COUNTS = {
    "clip_full1854",
    "clip_closed12",
    "dino_intra10",
    "rgb_intra10",
    "luma_intra10",
    "clip_common_strict",
    "dino_common_strict",
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )


def case(a, name, mode, reg):
    folder = a.render / "renders" / name / mode
    marker = folder / "settings.json"
    s = read(marker)
    source = a.root / name
    actual = read(source / "result.json")
    primary, count = PANELS[name]
    receipt = read(PUBLIC / "cohort_session_render_inputs.json")
    word_file = a.render / "evaluations" / mode / name / "word_predictions.json"
    word_meta = read(word_file)
    check(
        receipt["design_sha256"] == sha(a.render / "design.json")
        and receipt["radius_sha256"] == sha(a.render / "radius.json")
        and receipt["reference_sha256"][name] == reg["sha"]
        and receipt["comparisons"][name]["source_result_sha256"] == sha(source / "result.json")
        and receipt["packets_sha256"][mode + "/" + name] == s["packet_sha256"]
        and word_meta["candidate_group_size"] == count
        and word_meta["physical_trials_per_photo"] == (4 if mode == "sub-04" else 12)
        and word_meta["original_word_choices_sha256"] == actual["predictions_sha256"],
        "Published render receipt or observation budget changed",
    )
    check(
        actual["primary"] == primary
        and actual["design_sha256"] == sha(PUBLIC / "cohort_session_matched_design.json")
        and actual["decisions"][primary]["candidates"] == count
        and actual["predictions_sha256"] == sha(source / "predictions.npz"),
        "Changed EEG decision experiment",
    )
    rows, folds, words, donors, support, _, _, digest = packet(a.render, mode, name)
    with np.load(source / "predictions.npz", allow_pickle=False) as z:
        idx = z[primary + "_" + ("cohort" if mode == "cohort-3" else mode)]
        check(
            np.array_equal(reg["rows"], rows)
            and np.array_equal(reg["folds"], folds)
            and np.array_equal(z["rows"], rows)
            and np.array_equal(z["folds"], folds)
            and np.array_equal(words, reg["vocab"][idx]),
            "Rendered words differ from locked decision",
        )
        menus = read(PUBLIC / "cohort_session_menus.json")["panels"][name]
        menu = menus["decisions"][primary]
        permitted = np.asarray(menu["alternatives"])[menu["menu_at_ordinal"]]
        check(
            menus["registration_sha256"] == reg["sha"]
            and permitted.shape == (120, count)
            and np.all(np.any(permitted == idx[:, None], axis=1)),
            "Word outside its registered menu",
        )
    check(
        s["design_sha256"] == sha(a.render / "design.json")
        and s["packet_sha256"] == digest
        and np.array_equal(s["seeds"], 923761 + rows),
        "Different generator or seeds",
    )
    for k, v in dict(
        rows=rows, folds=folds, predicted_words=words, donor_rows=donors, support_rows=support
    ).items():
        check(np.array_equal(s[k], v), "Different generation row field: " + k)
    validate_packet(
        SimpleNamespace(
            conditions=a.render / "conditions", evaluations=a.render / "evaluations", panel=name
        ),
        mode,
        reg,
        s,
    )
    return s


def bootstrap(x, categories, metric):
    """Every draw uses the same true object, its fixed photographs, and all six saved conditions."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return dict(n=0)
    check(x.shape == (len(categories), 2, 3), "Missing paired arm")
    groups = np.asarray(categories)
    n = np.bincount(groups, minlength=12)
    sums = np.zeros((12, 2, 3))
    np.add.at(sums, groups, x)
    draw = np.random.default_rng(SEED).integers(12, size=(10000, 12))
    den = n[draw].sum(1)
    means = sums[draw].sum(1)[den > 0] / den[den > 0, None, None]
    val = x.mean(0)

    def difference(v, sim):
        return dict(delta=float(v), ci95=np.quantile(sim, [0.025, 0.975]).tolist())

    values = x.sum(0) if metric in COUNTS else val
    return dict(
        n=len(x),
        **{
            "credits" if metric in COUNTS else "mean": {
                mode: dict(zip(ARMS, values[i].tolist(), strict=True))
                for i, mode in enumerate(MODES)
            }
        },
        cohort_minus_single_own=difference(val[1, 0] - val[0, 0], means[:, 1, 0] - means[:, 0, 0]),
        own_minus={
            mode: {
                ARMS[j]: difference(val[i, 0] - val[i, j], means[:, i, 0] - means[:, i, j])
                for j in (1, 2)
            }
            for i, mode in enumerate(MODES)
        },
    )


def image_rows(d, c):
    values = {
        "clip_full1854": np.column_stack([c["per_row"]["full1854"][arm] for arm in ARMS]),
        "clip_closed12": np.column_stack([c["per_row"]["closed12"][arm] for arm in ARMS]),
    }
    for field, metric in (
        ("intra10", "dino_intra10"),
        ("rgb_rmse32", "rgb_rmse32"),
        ("luma_pearson32", "luma_pearson32"),
        ("rgb_intra10", "rgb_intra10"),
        ("luma_intra10", "luma_intra10"),
    ):
        values[metric] = np.array([[r[field][arm] for arm in ARMS] for r in d["images"]])
    return values


def paired_gallery(a, panel, reg, cfgs, rows):
    """Show the same preassigned twelve rows, with their original, four-trial and twelve-trial JPEGs."""
    if not a.out.resolve().is_relative_to(Path("outputs").resolve()):
        raise ValueError("Original-photo contact sheets belong under outputs/")
    table = reference(a)
    cache = a.out / panel / MODES[0] / "original_cls" / (panel + ".npz")
    with np.load(cache, allow_pickle=False) as z:
        check(np.array_equal(z["rows"], reg["rows"]), "Different original gallery ordering")
        hashes = dict(zip(reg["rows"].tolist(), z["jpeg_sha256"].tolist(), strict=True))
    ordinal = {r: i for i, r in enumerate(reg["rows"])}
    size, gap, label = 148, 8, 34
    block = (size * 3 + gap * 4, size + label + gap * 2)
    sheet = Image.new("RGB", (block[0] * 3, block[1] * 4), "white")
    draw = ImageDraw.Draw(sheet)
    with ZipFile(a.originals) as bank:
        for j, row in enumerate(rows):
            i = ordinal[row]
            rec = table[row]
            raw = bank.read(f"training_images/{rec['official_folder']}/{rec['stimulus_stem']}.jpg")
            check(hashlib.sha256(raw).hexdigest() == hashes[row], "Gallery original changed")
            with Image.open(io.BytesIO(raw)) as im:
                original = crop512(im.convert("RGB"))
            pics = [original]
            words = [rec["unique_id"]]
            for mode in MODES:
                image = a.render / "renders" / panel / mode / "images" / "own" / f"{i:03d}.jpg"
                check(
                    sha(image) == cfgs[mode]["jpeg_sha256"]["own"][i],
                    "Gallery matched JPEG changed",
                )
                with Image.open(image) as im:
                    pics.append(im.convert("RGB"))
                words.append(cfgs[mode]["predicted_words"][i])
            for col, (pic, text, word) in enumerate(
                zip(
                    pics,
                    ("Original", "Sub-04, four trials", "Cohort, twelve trials"),
                    words,
                    strict=True,
                )
            ):
                x, y = (j % 3) * block[0] + gap + col * (size + gap), (j // 3) * block[1] + gap
                sheet.paste(pic.resize((size, size), Image.Resampling.LANCZOS), (x, y))
                draw.text((x, y + size + 2), text, fill="black")
                draw.text((x, y + size + 16), word.replace("_", " "), fill="black")
    dest = a.out / "galleries" / (panel + "_original_single_cohort.jpg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, "JPEG", quality=94)
    return sha(dest)


def score(a):
    design = read(PUBLIC / "cohort_session_matched_design.json")
    output = dict(
        study="Same-photo comparison with four physical observations from sub-04 or twelve from sub-04, sub-05 and sub-01. One output per photo and arm. Both panels contain 120 distinct original photographs.",
        metrics="EEG uses the registered 12- or 4-word decision set. Independent CLIP names each JPEG among all 12 panel labels and all 1854 labels. DINOv2-S/14 top-1 identifies one original among the ten of its true object, including every wrong word. Strict pairs require two correct words, and a cross-mode comparison takes the intersection.",
        intervals="Paired true-object percentile intervals conditional on these people and session cells. Each of 10000 draws resamples twelve shared true objects with all their photos, seeds, arms and acquisition modes. Rates and differences are fractions, seed 260532. RGB RMSE and luminance Pearson are means on 32 by 32 images.",
        panels={},
        design_sha256=sha(PUBLIC / "cohort_session_matched_design.json"),
        scorer_sha256=sha(Path(__file__)),
    )
    for name in a.panels:
        reg = registered(a.ref, name)
        original_reg = design["panels"][name]["registration"]
        check(
            all(
                np.array_equal(reg[k], original_reg[r])
                for k, r in (
                    ("rows", "rows"),
                    ("folds", "folds"),
                    ("truth", "truth"),
                    ("vocab", "words"),
                    ("calibration", "calibration_rows_by_fold_and_word"),
                )
            ),
            "Different prospective registration",
        )
        cfgs = {m: case(a, name, m, reg) for m in MODES}
        check(
            all(
                cfgs[MODES[0]][k] == cfgs[MODES[1]][k]
                for k in (
                    "seeds",
                    "radius",
                    "turbo_sha256",
                    "dictionary_sha256",
                    "batch_size",
                    "adapter_sha256",
                    "driver_sha256",
                )
            ),
            "Different generator or row seeds by acquisition budget",
        )
        for mode in MODES:
            dst = a.out / name / mode
            if a.stage in ("pixels", "all"):
                opts = SimpleNamespace(
                    **(
                        vars(a)
                        | dict(
                            panel=name,
                            subjects=[mode],
                            out=dst,
                            references=a.ref,
                            renders=a.render / "renders",
                            conditions=a.render / "conditions",
                            evaluations=a.render / "evaluations",
                            gallery_dir=a.out / "galleries",
                            panel_lock=a.ref / (name + "_" + mode + "_lock.json"),
                            private_gallery=True,
                            release=None,
                        )
                    )
                )
                score_pixels(opts)
        if a.stage == "pixels":
            continue
        path = a.root / (name + "-clip.json")
        clip = read(path)
        check(
            clip["clip_sha256"] == CLIP_SHA
            and clip["design_sha256"] == sha(a.render / "design.json"),
            "Different independent naming encoder or render design",
        )
        data, pairs, meta, diagnostics = {}, {}, {}, {}
        for mode, s in cfgs.items():
            dst = a.out / name / mode
            file = dst / name / mode / "per_query.json"
            d, c = read(file), clip["cases"][name + "/" + mode]
            summary = read(dst / "summary.json")
            gallery = dst / name / mode / "private_comparison.jpg"
            marker = a.render / "renders" / name / mode / "settings.json"
            words = np.asarray(s["predicted_words"])
            check(
                c["settings_sha256"] == d["settings_sha256"] == sha(marker)
                and d["condition_sha256"] == s["packet_sha256"]
                and summary["subjects"] == [mode]
                and summary["registration_sha256"] == reg["sha"]
                and summary["dino"]["sha256"] == FILES
                and all(
                    np.array_equal(c[k], v)
                    for k, v in dict(
                        rows=reg["rows"],
                        folds=reg["folds"],
                        truth=reg["labels"],
                        predicted_words=words,
                    ).items()
                )
                and all(
                    np.array_equal([r[k] for r in d["images"]], v)
                    for k, v in dict(
                        row=reg["rows"], fold=reg["folds"], object=reg["labels"], word=words
                    ).items()
                ),
                "DINO, CLIP, packet and registered photo identities disagree",
            )
            expected, dp, cp = strict(d, c, reg["labels"], reg["folds"], reg["rows"], words)
            pairs[mode] = (expected, dp, cp)
            data[mode] = image_rows(d, c)
            diagnostics[mode] = dict(
                eeg_correct=int((words == reg["labels"]).sum()),
                strict_eligible=len(expected),
                strict_credits={
                    enc: dict(zip(ARMS, np.sum(list(z.values()), axis=0).tolist(), strict=True))
                    if z
                    else {}
                    for enc, z in (("dino", dp), ("clip", cp))
                },
            )
            rows = summary["cases"][mode]["gallery"]["rows"]
            check(
                rows == [int(reg["rows"][reg["labels"] == w].min()) for w in sorted(reg["vocab"])],
                "Gallery did not show the first registered photo from every object",
            )
            meta[mode] = dict(
                physical_trials_per_photo=4 if mode == "sub-04" else 12,
                condition_sha256=s["packet_sha256"],
                settings_sha256=sha(marker),
                pixel_rows_sha256=sha(file),
                original_jpegs_sha256=d["original_jpeg_sha256_in_row_order"],
                private_gallery_sha256=sha(gallery),
                generated_gallery_sha256=summary["cases"][mode]["gallery"]["sha256"],
                gallery_rows=rows,
            )
            filename = f"photo_angle_{mode.replace('-', '')}_{name}.jpg"
            generated = a.out / "galleries" / filename
            check(
                sha(generated) == meta[mode]["generated_gallery_sha256"],
                "Generated-only gallery changed",
            )
            meta[mode]["generated_gallery_filename"] = filename
        check(
            meta["sub-04"]["original_jpegs_sha256"] == meta["cohort-3"]["original_jpegs_sha256"],
            "Original-photo galleries differ by readout",
        )
        result = {
            k: bootstrap(np.stack([data[m][k] for m in MODES], axis=1), reg["truth"], k)
            for k in data[MODES[0]]
        }
        lookup = {w: i for i, w in enumerate(reg["vocab"])}
        common = sorted(set(pairs[MODES[0]][0]) & set(pairs[MODES[1]][0]))
        for j, k in enumerate(("dino_common_strict", "clip_common_strict"), 1):
            result[k] = bootstrap(
                [[pairs[m][j][i] for m in MODES] for i in common],
                [lookup[pairs[MODES[0]][0][i]] for i in common],
                k,
            )
        out = dict(
            n_unique_photos=120,
            generated_jpegs=720,
            eeg_choices=PANELS[name][1],
            original_chance_dino=0.1,
            cases=diagnostics,
            paired_metrics=result,
            registration_sha256=reg["sha"],
            clip_report_sha256=sha(path),
            inputs=meta,
        )
        primary = PANELS[name][0]
        eeg = read(a.root / name / "result.json")["decisions"][primary]
        check(
            all(
                diagnostics[m]["eeg_correct"]
                == eeg["observed"]["cohort" if m == "cohort-3" else m]["hits"]
                for m in MODES
            ),
            "Published EEG hit counts changed",
        )
        out["paired_eeg_difference"] = eeg["cohort_minus"]["sub-04"]
        single, cohort = (np.asarray(cfgs[m]["predicted_words"]) for m in MODES)
        same, ca, cb = single == cohort, single == reg["labels"], cohort == reg["labels"]
        groups = {
            "same_correct": same & ca,
            "same_wrong": same & ~ca,
            "corrected": ~ca & cb,
            "lost": ca & ~cb,
            "changed_both_wrong": ~same & ~ca & ~cb,
        }
        check(np.all(np.sum(list(groups.values()), axis=0) == 1), "Overlapping word-change groups")
        out["word_change_groups"] = {
            k: dict(
                n=int(mask.sum()),
                **{
                    endpoint: {m: float(data[m][endpoint][mask, 0].sum()) for m in MODES}
                    for endpoint in ("clip_full1854", "clip_closed12", "dino_intra10")
                },
            )
            for k, mask in groups.items()
        }
        out["same_predicted_word_pixel_metrics"] = {
            endpoint: bootstrap(
                np.stack([data[m][endpoint][same] for m in MODES], axis=1),
                reg["truth"][same],
                endpoint,
            )
            for endpoint in ("rgb_rmse32", "luma_pearson32")
        }
        out["private_paired_gallery_sha256"] = paired_gallery(
            a, name, reg, cfgs, meta[MODES[0]]["gallery_rows"]
        )
        names = a.root / (name + "-refnames.json")
        check(
            names.is_file() or not a.release,
            "A public release requires the actual original-photo scores",
        )
        if names.is_file():
            ref = read(names)
            check(
                ref["targets"] == 120
                and ref["registration_sha256"] == reg["sha"]
                and ref["clip_sha256"] == CLIP_SHA
                and ref["dictionary_sha256"] == clip["dictionary_sha256"],
                "Original image naming source differs",
            )
            out["original_names"] = {
                k: ref[k] for k in ("targets", "full1854_credit", "closed12_credit")
            }
            out["original_names"]["report_sha256"] = sha(names)
        output["panels"][name] = out
        print(
            name,
            {
                k: result[k]["credits"]
                for k in ("clip_full1854", "clip_closed12", "dino_intra10", "dino_common_strict")
                if "credits" in result[k]
            },
            flush=True,
        )
    if a.stage != "pixels":
        if a.gallery_release:
            a.gallery_release.mkdir(parents=True, exist_ok=True)
            for panel in output["panels"].values():
                for receipt in panel["inputs"].values():
                    filename = receipt["generated_gallery_filename"]
                    src, dest = a.out / "galleries" / filename, a.gallery_release / filename
                    if not dest.exists():
                        shutil.copyfile(src, dest)
                    check(
                        sha(dest) == receipt["generated_gallery_sha256"],
                        "Published gallery checksum mismatch",
                    )
        save(a.out / "paired_summary.json", output)
        if a.release:
            check(set(a.panels) == set(PANELS), "Both panels are required in the release")
            save(a.release, output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("pixels", "join", "all"))
    p.add_argument("--panels", choices=PANELS, nargs="+", default=list(PANELS))
    p.add_argument("--root", type=Path, default=Path("outputs/session-matched-cohort"))
    for k in ("ref", "render", "out", "release"):
        p.add_argument("--" + k, type=Path)
    p.add_argument(
        "--gallery-release", type=Path, help="Copy the verified generated-only contact sheets here"
    )
    for k, v in dict(
        model="outputs/photo-structure/dino/facebook-dinov2-small",
        originals="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--batch-size", type=int, default=16)
    opts = p.parse_args()
    opts.ref = opts.ref or opts.root / "references"
    opts.render = opts.render or opts.root / "render_inputs"
    opts.out = opts.out or opts.root / "pixel-scores"
    check(len(set(opts.panels)) == len(opts.panels), "Repeated panel")
    score(opts)
