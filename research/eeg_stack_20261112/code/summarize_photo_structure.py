"""Verify and compact the independent JPEG CLIP and pixel reports for both photo structure methods."""

import argparse
from pathlib import Path

import numpy as np
from decode_eeg_layout import digest, load_json, save_json
from score_reconstruction_jpegs import CLIP_SHA, cluster_table, interval, registered

PANELS = ("registered", "replicate")
PEOPLE = ("sub-04", "sub-05")
METHODS = {"support": ("matched", "mean"), "geometry": ("query", "prior")}
CLIP_METRICS = ("full1854", "closed12", "intra10", "two_photo_assignment")
CPU_METRICS = ("luma_ssim", "contrast_structure", "luma_ssim_pair_assignment", "mask_mse")
STRICT = "verified_wrong_eeg_same_object"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def data_scopes(clip, registration):
    rows = np.asarray(clip["rows"])
    truth = np.array(clip["true_labels"])
    words = np.array(clip["predicted_words"])
    folds = np.array(clip["folds"])
    donor = np.array(clip["donor_rows"])
    loc = {r: i for i, r in enumerate(rows)}
    require(
        len(rows) == len(loc) == 120 and set(rows) == set(donor) and len(set(truth)) == 12,
        "Incorrect panel or donor registration",
    )
    ix = np.array([loc[r] for r in donor])
    valid = rows != donor
    require(
        np.array_equal(words[ix], words)
        and np.array_equal(folds[ix], folds)
        and np.array_equal(valid, clip["swap_eligible"]),
        "Frozen donors differ in word/fold",
    )
    strict = valid & (truth[ix] == truth)
    expected = set()
    for word in np.unique(truth):
        for fold in range(5):
            i, j = np.flatnonzero((truth == word) & (folds == fold))
            if words[i] == words[j]:
                expected.add(tuple(sorted((rows[i], rows[j]))))
    declared = [tuple(sorted(x)) for x in clip["eligible_pair_rows"]]
    require(
        set(declared) == expected and len(declared) == len(expected) == clip["eligible_pair_count"],
        "Strict true-class/word/fold two-photo pairs differ",
    )
    pairs = np.array([[loc[i], loc[j]] for i, j in declared], dtype=int).reshape(-1, 2)
    complete = strict[pairs].all(1)
    require(
        np.all(ix[pairs[complete, 0]] == pairs[complete, 1])
        and np.all(ix[pairs[complete, 1]] == pairs[complete, 0]),
        "Incorrect reciprocal donor pair",
    )
    require(
        registration["sha"] == clip["registration_sha256"]
        and np.array_equal(rows, registration["rows"])
        and np.array_equal(folds, registration["folds"])
        and np.array_equal(truth, registration["labels"]),
        "Summary input rows, folds or vocabulary differ from original registration",
    )
    classes = registration["truth"]
    return classes, pairs, strict, complete


def compact_effect(v):
    ci = v["ci95"]
    return dict(n=v["n"], delta=v["mean"], ci95=ci)


def same_values(left, right, tol=1e-12):
    if left is None or right is None:
        return left is None and right is None
    return np.allclose(left, right, atol=tol, rtol=0)


def counts(record, n):
    v = np.asarray(record["per_query_or_pair"])
    require(
        v.shape == (n,)
        and np.isfinite(v).all()
        and np.all((v >= 0) & (v <= 1))
        and n == record["n"]
        and np.isclose(v.sum(), record["total"], atol=1e-9, rtol=0),
        "Per-row CLIP credit differs from the count",
    )
    return v


def paired(clip, base, control, classes, pair, strict, complete):
    out = {}
    for metric in CLIP_METRICS:
        two = metric == "two_photo_assignment"
        labels = classes[pair[:, 0]] if two else classes
        take = (complete if two else strict) if control == "swapped" else np.ones(len(labels), bool)
        delta = (
            counts(clip["arms"][base][metric], len(labels))
            - counts(clip["arms"][control][metric], len(labels))
        )[take]
        sums, n = cluster_table(delta, labels[take])
        value = interval(sums, n)
        if control != "swapped":
            old = clip["paired"][f"{base}_minus_{control}"][metric]
            require(
                old["n"] == value["n"]
                and same_values(old["mean"], value["mean"])
                and same_values(old["ci95"], value["ci95"]),
                "Recomputed CLIP delta/CI differs from the saved GPU report",
            )
        out[metric] = compact_effect(value)
    return out


def case(clip, pixel, kind, registration):
    primary, prior = METHODS[kind]
    sources = pixel["sources_sha256"]
    require(
        sources["registration"] == clip["registration_sha256"]
        and sources.get("render_settings", sources.get("settings")) == clip["settings_sha256"],
        "Pixel and CLIP evaluations did not score the same settings or folds",
    )
    if kind == "support":
        require(sources["selection"] == clip["selection_sha256"], "Different support selector")
    else:
        require(
            sources["prediction"] == clip["geometry_sha256"], "Different foreground geometry packet"
        )
    hashes = pixel["ordered_jpeg_hash_digest"]
    require(
        set(hashes) == set(clip["arms"])
        and all(v == clip["arms"][arm]["jpeg_hash_list_sha256"] for arm, v in hashes.items()),
        "CLIP and pixel measurements used different JPEGs or arm inventories",
    )
    cls, pair, strict, complete = data_scopes(clip, registration)
    groups = pixel["groups"]
    require(
        groups["all"]["n_rows"] == 120
        and groups[STRICT]["n_rows"] == int(strict.sum())
        and groups["same_object_word_fold_peer"]["n_rows"] == 2 * len(pair),
        "Pixel and CLIP masks differ",
    )
    arms = {}
    for arm, rec in clip["arms"].items():
        # The CPU scorer publishes only verified wrong-donor rows for the swapped image arm.
        grp = groups[STRICT if arm == "swapped" else "all"]["arm_means"][arm]
        a = {}
        for k in CLIP_METRICS:
            n = len(pair) if k == "two_photo_assignment" else 120
            v = counts(rec[k], n)
            a[k] = {"credit": float(v.sum()), "n": n}
        cpu = {}
        for k in CPU_METRICS:
            if k not in grp:
                continue
            val = grp[k]
            if k.endswith("_pair_assignment"):
                npairs = int(complete.sum()) if arm == "swapped" else len(pair)
                require(
                    npairs == val["n_pairs"] and val["n"] == 2 * npairs, "CPU pair counts differ"
                )
                cpu[k] = dict(
                    credit=None if val["mean"] is None else float(val["mean"] * npairs), n=npairs
                )
            else:
                require(
                    val["n"] == (int(strict.sum()) if arm == "swapped" else 120),
                    "CPU arm denominator differs",
                )
                cpu[k] = dict(mean=val["mean"], n=val["n"])
        arms[arm] = dict(clip=a, pixel=cpu, jpeg_hash_list_sha256=hashes[arm])
    comparisons = {}
    for control in sorted(set(arms) - {primary}):
        scope = STRICT if control == "swapped" else "all"
        original = groups[scope]["primary_minus"][control]
        independent = {}
        for k in CPU_METRICS:
            if k not in original:
                continue
            v = original[k]
            diff = compact_effect(v)
            if k.endswith("_pair_assignment"):
                diff["n"] = v["n_pairs"]
            independent[k] = diff
        comparisons[f"{primary}_minus_{control}"] = dict(
            scope="distinct_donor_same_true_object_word_fold" if control == "swapped" else "all",
            clip=paired(clip, primary, control, cls, pair, strict, complete),
            pixel=independent,
        )
    return dict(
        n=120,
        primary=primary,
        category_prior=prior,
        eligible_pairs=len(pair),
        strict_swap_rows=int(strict.sum()),
        strict_swap_complete_pairs=int(complete.sum()),
        arms=arms,
        comparisons=comparisons,
        settings_sha256=clip["settings_sha256"],
        registration_sha256=clip["registration_sha256"],
    )


def run(args):
    clip = load_json(args.clip)
    require(
        clip["mode"] == "scores" and clip["clip_sha256"] == CLIP_SHA,
        "Wrong CLIP-L/14 checkpoint or mode",
    )
    geom = load_json(args.geometry)
    roles = {}
    for name, p in clip["cases"].items():
        require(
            set(p) == set(PANELS) and all(set(v) == set(PEOPLE) for v in p.values()),
            "Expected both participants and both complete panels",
        )
        available = set(p["registered"]["sub-04"]["arms"])
        k = "support" if "matched" in available else "geometry" if "query" in available else None
        require(k in METHODS and k not in roles, "Expected one support run and one geometry run")
        roles[k] = name
    require(set(roles) == set(METHODS), "Both image pipelines are required")
    hashes = {
        "clip_report": digest(args.clip),
        "geometry_pixel_report": digest(args.geometry),
        "support_pixel_reports": {},
    }
    panels = {}
    for panel in PANELS:
        panels[panel] = {}
        across_people = None
        registration = registered(args.reference_dir, panel)
        for subject in PEOPLE:
            path = args.support / panel / subject / "structure.json"
            raw = load_json(path)
            require(
                (raw["panel"], raw["subject"]) == (panel, subject), "Wrong support pixel report"
            )
            hashes["support_pixel_reports"][f"{panel}/{subject}"] = digest(path)
            methods = {k: clip["cases"][r][panel][subject] for k, r in roles.items()}
            s, g = methods["support"], methods["geometry"]
            key = (
                "rows",
                "folds",
                "true_labels",
                "predicted_words",
                "original_sha256",
                "eligible_pair_rows",
                "conditions_sha256",
                "registration_sha256",
                "donor_rows",
            )
            sf, gf = s["arms"]["baseline"], g["arms"]["baseline"]
            require(
                all(s[k] == g[k] for k in key)
                and sf["jpeg_hash_list_sha256"] == gf["jpeg_hash_list_sha256"]
                and all(sf[k] == gf[k] for k in CLIP_METRICS),
                "Pipelines disagree on query identity or shared baseline JPEGs/credits",
            )
            signature = tuple(s[k] for k in ("rows", "true_labels", "folds", "original_sha256"))
            require(
                across_people is None or signature == across_people,
                "Different stimuli between participants",
            )
            across_people = signature
            warp = geom["panels"][panel][subject]
            require(
                raw["sources_sha256"]["baseline_results"]
                == warp["sources_sha256"]["baseline_results"]
                and raw["sources_sha256"]["photo_report"]
                == warp["sources_sha256"]["original_photo_report"],
                "Independent scorers used different original or baseline records",
            )
            bs = raw["groups"]["all"]["arm_means"]["baseline"]["luma_ssim"]
            bg = warp["groups"]["all"]["arm_means"]["baseline"]["luma_ssim"]
            require(
                bs["n"] == bg["n"]
                and bs["n_objects"] == bg["n_objects"]
                and same_values(bs["mean"], bg["mean"])
                and same_values(bs["ci95"], bg["ci95"]),
                "Same baseline gave different independently calculated SSIM",
            )
            panels[panel][subject] = dict(
                support=case(s, raw, "support", registration),
                geometry=case(g, warp, "geometry", registration),
            )
    return dict(
        inputs_sha256=hashes,
        clip_weights_sha256=CLIP_SHA,
        dictionary_sha256=clip["dictionary_sha256"],
        query_scope="Two participants, two disjoint panels, 120 existing photographs and frozen words per participant/panel.",
        definitions=dict(
            clip="All semantic and photo credits score SHA-verified saved JPEGs. Top-one ties share credit.",
            two_photo_assignment="One denominator unit is one pair of held-out photos sharing truth, predicted word and fold. Correct assignment gets 1, ties 0.5.",
            swapped="The matched-versus-swapped deltas use distinct donors with the same truth, word and fold. Pair differences additionally require reciprocal swaps of both members. Absolute CLIP arm totals cover the full eligible registration; absolute swapped pixel scores use verified donor rows.",
            uncertainty="Paired 95% percentile intervals resample represented true objects 10,000 times separately for each participant/panel. Lower mask_mse is better; higher credit, SSIM and contrast structure are better.",
            geometry="Foreground mask MSE uses the same U2NetP as the geometry readout and warp. Luma SSIM and frozen CLIP are separately measured.",
        ),
        panels=panels,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path("outputs/photo-structure")
    p.add_argument("--clip", type=Path, default=root / "structure_clip_report.json")
    p.add_argument("--geometry", type=Path, default=root / "geometry_warp_evaluation.json")
    p.add_argument("--support", type=Path, default=root / "support_render")
    p.add_argument("--reference-dir", type=Path, default=Path("outputs/object-fidelity"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/object_fidelity/photo_structure.json",
    )
    args = p.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, result)
    print(args.out, digest(args.out), flush=True)


if __name__ == "__main__":
    main()
