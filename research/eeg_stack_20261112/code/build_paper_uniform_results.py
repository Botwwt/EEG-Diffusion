"""Build the registered common-generator table and the separate native comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_aggregate import COMMON_METRICS, SUBJECTS, bootstrap_ci, panel_seeds, read_scores, sha
from paper_inception_delta import METRIC, read_sidecar

BASELINES = ("brain-hive", "atm", "nice", "eegnet", "ridge")
NATIVE_BASELINES = ("atm-author", "cogcap-author", "brain-hive-native", "enigma-native")
LATE_ARMS = ("o13-late", "native-full-layout", "native-full-late")
FULL_SHARED_ARMS = {"m1a-d2-full-training": "d2",
                    "m1a-d2-full-training-hive-late": "native-full-late"}
METRICS = tuple(METRIC if name == "inception_2way" else name for name in COMMON_METRICS)


def paths(prior, run, enigma=None, late=None, full_shared=None):
    def shared(arm, budget, person):
        return prior / f"images/shared/{arm}/{budget}/{person}"

    native = {
        "atm-author": [prior / f"external/atm-sub{p[-2:]}" for p in SUBJECTS],
        "cogcap-author": [run / f"cogcap/cogcap-sub{p[-2:]}" for p in SUBJECTS],
        "brain-hive-native": [prior / f"images/native-full/native-paper/official-r80/{p}" for p in SUBJECTS],
        "m1a-d2": [shared("d2", "official-r80", p) for p in SUBJECTS],
    }
    if enigma is not None:
        native["enigma-native"] = [enigma / f"images/enigma/official-r80/{p}" for p in SUBJECTS]
    if late is not None:
        native.update({arm: [late / f"images/shared/{arm}/official-r80/{p}" for p in SUBJECTS] for arm in LATE_ARMS})
    if full_shared is not None:
        native.update({name: [full_shared / f"images/shared-full/{arm}/official-r80/{p}" for p in SUBJECTS]
                       for name, arm in FULL_SHARED_ARMS.items()})
    return {
        **{f"{budget}_common": {k: [run / f"images/{k}/official-{budget}/{p}" for p in SUBJECTS]
                               for k in (*BASELINES, "m1a-d2")} for budget in ("r4", "r80")},
        "r80_native": native,
        "r4_ablation": {
            **{k: [shared(k, "official-r4", p) for p in SUBJECTS]
               for k in ("d2-reference", "d2-positive-half", "d2-mean-erp", "swap", "rec", "wrong")},
            "d2-no-layout": [run / f"ablation/shared/d2-no-layout/official-r4/{p}" for p in SUBJECTS],
        },
    }


def read_arm(folders, reference, *, common_budget=None):
    real = reference.copy()
    coverage = {}
    sources, observations, metadata = {}, {}, {}
    for person, folder in zip(SUBJECTS, folders, strict=True):
        needed = ("scores.json", "observations.npz", "inception_official.json", "inception_official.npz")
        missing = [name for name in needed if not (folder / name).is_file()]
        if missing:
            coverage[person] = missing
            continue
        metrics, rows, concepts, words, values, score, provenance = read_scores(folder, "official")
        side, item = read_sidecar(folder)
        if not np.array_equal(rows, item["row"]) or not np.array_equal(concepts, item["concept"]):
            raise ValueError(f"Official Inception row order differs in {folder}")
        values[:, metrics.index("inception_2way")] = item[METRIC]
        observations[person] = values
        sources[person] = {**provenance, "inception_observations_sha256": side["observations_sha256"],
                           "ordered_image_hashes_sha256": score["ordered_image_hashes_sha256"]}
        if "concept" in real:
            if not np.array_equal(real["concept"], concepts) or not np.array_equal(real["original_word"], words):
                raise ValueError(f"Scored gallery differs in {folder}")
        else:
            real.update(concept=concepts, original_word=words)
        if common_budget is not None:
            raw = (folder / "render.json").read_bytes()
            info = json.loads(raw)
            if sha(raw) != score["render_sha256"]:
                raise ValueError(f"Render receipt differs from scored pixels in {folder}")
            repeats = np.arange(80 if common_budget == "r80" else 4).reshape(-1, 4).tolist()
            expected = {"row": list(range(200)), "eeg_donor": list(range(200)), "n": 200,
                        "budget": f"official-{common_budget}", "training": "shared", "prompt": "", "draws": 1,
                        "unet_calls_per_draw": 4, "repeats": repeats,
                        "seeds": panel_seeds("official", 200).tolist()}
            if any(info.get(k) != value for k, value in expected.items()) or info["subject"] != int(person[-2:]):
                raise ValueError(f"Render differs from the uniform protocol in {folder}")
            metadata[person] = {k: info[k] for k in ("split_sha256", "physical_eeg", "layout_source", "sampling")}
            sources[person].update(checkpoint_sha256=info["checkpoint_sha256"], render_sha256=sha(raw))
    reference.update(real)
    people = list(observations)
    mean = dict(zip(METRICS, np.stack(list(observations.values())).mean(axis=(0, 1)), strict=True)) if people else {}
    return {"subjects": people, "n": len(people)*200, "mean": mean, "sources": sources, "missing": coverage,
            "by_subject": {p: dict(zip(METRICS, values.mean(0), strict=True)) for p, values in observations.items()}}, observations, metadata


def compare(a, b):
    people = [p for p in SUBJECTS if p in a and p in b]
    if not people:
        return {}
    delta = np.stack([a[p] - b[p] for p in people])
    lo, hi = bootstrap_ci(delta)
    return {"subjects": people, "n": len(people)*200,
            "delta": {k: [float(x), float(l), float(h)] for k, x, l, h in zip(METRICS, delta.mean((0, 1)), lo, hi, strict=True)}}


def build(prior, run, enigma=None, late=None, full_shared=None):
    for source in (prior, run, enigma, late, full_shared):
        if source is not None and not source.is_dir():
            raise ValueError(f"Expected an experiment directory at {source}")
    records, reference = {}, {}
    for panel, arms in paths(prior, run, enigma, late, full_shared).items():
        common_budget = panel.removesuffix("_common") if panel.endswith("_common") else None
        reports, data, conditions = {}, {}, {}
        for arm, folders in arms.items():
            reports[arm], data[arm], shared_conditions = read_arm(folders, reference, common_budget=common_budget)
            for person, values in shared_conditions.items():
                if conditions.setdefault(person, values) != values:
                    raise ValueError(f"Shared generation settings differ for {arm}, {person}")
        if common_budget is not None:
            pairs = [("m1a-d2", a) for a in BASELINES]
        elif panel == "r80_native":
            pairs = [(a, b) for a in ("m1a-d2", *LATE_ARMS, *FULL_SHARED_ARMS) if a in arms
                     for b in NATIVE_BASELINES if b in arms]
            if full_shared is not None:
                pairs.append(("m1a-d2-full-training", "m1a-d2"))
                if late is not None:
                    pairs.append(("m1a-d2-full-training-hive-late", "native-full-late"))
        else:
            pairs = [("d2-reference", a) for a in ("d2-positive-half", "d2-mean-erp", "d2-no-layout", "wrong")]
            pairs.append(("swap", "rec"))
        records[panel] = {"rows": reports, "pairs": {f"{a} - {b}": compare(data[a], data[b]) for a, b in pairs}}
    records["all_common_complete"] = all(row["n"] == 2000 for row in records["r4_common"]["rows"].values())
    records["all_r80_common_complete"] = all(row["n"] == 2000 for row in records["r80_common"]["rows"].values())
    native = records["r80_native"]["rows"]
    records["all_r80_native_complete"] = set(NATIVE_BASELINES).issubset(native) and all(row["n"] == 2000 for row in native.values())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--enigma", type=Path)
    parser.add_argument("--late", type=Path, help="Experiment directory containing the registered late fusion renders")
    parser.add_argument("--full-shared", type=Path, help="R80 renders from the shared fit using all official training photos")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = build(args.prior, args.run, args.enigma, args.late, args.full_shared)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for panel in ("r4_common", "r80_common", "r80_native", "r4_ablation"):
        print(panel, {k: row["n"] for k, row in data[panel]["rows"].items()})


if __name__ == "__main__":
    main()
