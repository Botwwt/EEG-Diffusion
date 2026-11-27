"""Select on the registered development panel and compare a frozen R80 photo candidate."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper_aggregate import PANELS, SUBJECTS, bootstrap_ci, panel_seeds, read_scores, sha, units

BASE_METRICS = ("clip_image_top1", "generated_name1854", "clip_2way", "ssim", "pixcorr")
DEV_ARMS = ("d2", "consensus", "strong-layout", "native-late", "o13-late", "dual-late", "native-layout")
FIXED_ARMS = ("d2", "native-full-layout", "native-full-late")


def cohort(root, track, arm, budget, people, *, verify_render):
    panel = budget.split("-")[0]
    n = PANELS[panel][0]
    observations, sources = [], {}
    order = None
    metrics = BASE_METRICS + (("clip_within10",) if panel in ("dev", "fresh") else ())
    for person in people:
        folder = root / "images" / track / arm / budget / person
        keys, rows, concepts, _, values, receipt, source = read_scores(folder, panel)
        if verify_render:
            raw = (folder / "render.json").read_bytes()
            render = json.loads(raw)
            desired = {
                "subject": int(person[-2:]), "budget": budget, "arm": arm,
                "n": n, "row": rows.tolist(), "eeg_donor": list(range(n)),
                "seeds": panel_seeds(panel, n).tolist(), "draws": 1, "unet_calls_per_draw": 4, "prompt": "",
                "repeats": np.arange(80).reshape(20, 4).tolist() if budget == "official-r80" else [list(range(4))],
            }
            if receipt["render_sha256"] != sha(raw) or any(render.get(k) != v for k, v in desired.items()):
                raise ValueError(f"Changed render identity or sampling protocol in {folder}")
            hashes = render["pixels_sha256"]
            if len(hashes) != n or sha("\n".join(hashes).encode()) != receipt["ordered_image_hashes_sha256"]:
                raise ValueError(f"Differing RGB provenance in {folder}")
            source = {**source, "render_sha256": sha(raw), "image_sha256": receipt["ordered_image_hashes_sha256"]}
        if order is None:
            order = (rows, concepts)
        elif not np.array_equal(rows, order[0]) or not np.array_equal(concepts, order[1]):
            raise ValueError(f"Stimulus identities differ in {folder}")
        observations.append(values[:, [keys.index(k) for k in metrics]])
        sources[person] = source
    return np.stack(observations), order, metrics, sources


def collect(specs, people, budget):
    all_values, info = {}, {}
    order = None
    for name, root, track, arm, verify in specs:
        values, current, metrics, sources = cohort(root, track, arm, budget, people, verify_render=verify)
        if order is None:
            order = current
        elif any(not np.array_equal(a, b) for a, b in zip(current, order, strict=True)):
            raise ValueError(f"Different evaluation gallery for {name}")
        all_values[name] = values
        info[name] = {
            "n": values.shape[0] * values.shape[1], "mean": dict(zip(metrics, values.mean((0, 1)), strict=True)),
            "photo_hits": float(values[:, :, metrics.index("clip_image_top1")].sum()),
            "naming_hits": float(values[:, :, metrics.index("generated_name1854")].sum()),
            "by_subject": {p: dict(zip(metrics, a.mean(0), strict=True)) for p, a in zip(people, values, strict=True)},
            "sources": sources,
        }
    return all_values, info, order[1], metrics


def compare(all_values, first, second, labels, panel, metrics):
    diff = all_values[first] - all_values[second]
    lo, hi = bootstrap_ci(units(diff, labels, panel))
    return {k: [float(v), float(l), float(h)] for k, v, l, h in zip(metrics, diff.mean((0, 1)), lo, hi, strict=True)}


def develop(run):
    people = ("sub-01", "sub-03")
    specs = [(a, run, "shared", a, True) for a in DEV_ARMS]
    values, results, labels, keys = collect(specs, people, "dev-r4")
    ranking = sorted(DEV_ARMS[1:], key=lambda a: (results[a]["mean"]["clip_image_top1"],
                                                   results[a]["mean"]["clip_within10"],
                                                   results[a]["mean"]["generated_name1854"]), reverse=True)
    chosen = ranking[0]
    return {"panel": "dev-r4", "subjects": people, "selected": chosen, "ranking": ranking,
            "selected_better_than_incumbent": results[chosen]["photo_hits"] > results["d2"]["photo_hits"],
            "rows": results, "versus_d2": {a: compare(values, a, "d2", labels, "dev", keys) for a in ranking}}


def official(run, prior, selection):
    chosen = selection["selected"]
    if chosen not in DEV_ARMS[1:] or selection["panel"] != "dev-r4":
        raise ValueError("Expected a recorded development-only decision")
    paths = [
        ("d2-archive", prior, "shared", "d2", False),
        ("brain-hive-full", prior, "native-full", "native-paper", False),
        *[(a, run, "shared", a, True) for a in (*FIXED_ARMS, chosen)],
    ]
    values, results, labels, keys = collect(paths, SUBJECTS, "official-r80")
    pairs = {(a, b): compare(values, a, b, labels, "official", keys) for a in (*FIXED_ARMS, chosen)
             for b in ("d2", "brain-hive-full", "d2-archive") if a != b}
    return {"panel": "official-r80", "subjects": SUBJECTS, "selected_from_development": chosen,
            "rows": results, "paired": {f"{a} minus {b}": x for (a, b), x in pairs.items()}}


def followup(run, panel):
    people = ("sub-01", "sub-03", "sub-08") if panel == "fresh" else ("sub-01", "sub-03")
    arms = ("d2", "o13-late") if panel == "fresh" else ("d2", "o13-late", "o13-after-first")
    budget = f"{panel}-r4"
    specs = [(a, run, "shared", a, True) for a in arms]
    values, results, labels, keys = collect(specs, people, budget)
    for person in people:
        renders = {}
        for arm in arms:
            folder = run / "images" / "shared" / arm / budget / person
            renders[arm] = json.loads((folder / "render.json").read_bytes())
        base = renders["d2"]
        reference = renders["o13-late"]["token_sha256"]
        early, late = base["token_sha256"][0], reference[-1]
        template = {"d2": [early] * 4, "o13-late": [early, early, late, late],
                    "o13-after-first": [early, late, late, late]}
        for arm, render in renders.items():
            matched = ("layout_source", "layout_strength", "layout_low_keep", "split_sha256", "physical_eeg",
                       "d2_sources", "generator_files_sha256", "render_code_sha256")
            if render["token_sha256"] != template[arm] or any(render[k] != base[k] for k in matched):
                raise ValueError(f"The timing experiment changed more than the planned conditions for {person}/{arm}")
    paired = {f"{a} minus {b}": compare(values, a, b, labels, panel, keys)
              for i, a in enumerate(arms) for b in arms[:i]}
    return {"panel": budget, "subjects": people, "rows": results, "paired": paired}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--prior", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--followup", choices=("dev", "fresh"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.followup:
        result = followup(args.run, args.followup)
    elif args.selection is None:
        result = develop(args.run)
    else:
        result = official(args.run, args.prior, json.loads(args.selection.read_text("utf-8")))
        result["selection_sha256"] = sha(args.selection.read_bytes())
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for method, row in result["rows"].items():
        print(method, row["n"], "photo", row["photo_hits"], "name", row["naming_hits"],
              "SSIM", round(row["mean"]["ssim"], 5))
    selected = result.get("selected", result.get("selected_from_development"))
    if selected is not None:
        print("Selected on development", selected)


if __name__ == "__main__":
    main()
