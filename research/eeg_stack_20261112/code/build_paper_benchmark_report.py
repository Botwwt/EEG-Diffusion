"""Build the paper comparison from a verified score export. Runs on CPU with NumPy."""

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import paper_aggregate as agg
import paper_dino_report
import paper_error_readout
import paper_inception_report
import paper_object_report

REPO = Path(__file__).resolve().parents[3]
RESULT = REPO / "research/eeg_stack_20261112/results/paper_benchmark.json"
DOC = REPO / "docs/PAPER_BENCHMARK_RESULTS.md"
ROOT = REPO / "outputs/paper-benchmark/verified"
DRAWS = 10000
SEED = 20261125
BUDGETS = {"official-r4": "官方 R4，连续四次", "official-r4-balanced": "官方 R4，跨组四次",
           "official-r80": "官方 R80", "fresh-r4": "新概念 R4", "dev-r4": "开发集 R4"}
PAPER, CODE, D2, WRONG = ["personal:" + k for k in ("native-paper", "native-code", "d2", "wrong")]
SHARED, SHARED_WRONG = "shared:d2", "shared:wrong"
FULL, FULL_CODE, ATM = "native-full:native-paper", "native-full:native-code", "atm"
NATIVE_LAYOUT, NO_LAYOUT = "personal:native-layout", "personal:d2-no-layout"
RESPONSE, POSITIVE = "personal:response", "personal:response-positive"
INSTANCE, DIRECT, RESIDUAL = ("personal:" + k for k in agg.INSTANCE_ARMS)
MAIN = ("generated_name1854", "clip_image_top1", "clip_within10", "ssim")
SECONDARY = ("pixcorr", "ssim", "alex2_2way", "alex5_2way", "inception_2way", "clip_2way",
             "swav_distance", "effnet_distance")
SECONDARY_HEADERS = ["PixCorr", "SSIM", "Alex2", "Alex5", "Inception", "CLIP", "SwAV ↓", "EffNet ↓"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plans():
    result = {}
    for budget in BUDGETS:
        if budget == "dev-r4":
            arms = [PAPER, NATIVE_LAYOUT, NO_LAYOUT, D2, POSITIVE, RESPONSE, INSTANCE, DIRECT, RESIDUAL]
            pairs = [(D2, PAPER), (D2, NATIVE_LAYOUT), (NATIVE_LAYOUT, PAPER), (D2, NO_LAYOUT),
                     (NO_LAYOUT, PAPER), (RESPONSE, D2), (POSITIVE, D2), (RESPONSE, POSITIVE),
                     (INSTANCE, PAPER), (DIRECT, INSTANCE), (RESIDUAL, INSTANCE), (RESIDUAL, DIRECT)]
        else:
            arms = [PAPER, CODE, D2, WRONG, SHARED, SHARED_WRONG]
            pairs = [(D2, PAPER), (D2, CODE), (D2, WRONG), (PAPER, CODE), (SHARED, PAPER),
                     (SHARED, CODE), (SHARED, SHARED_WRONG), (SHARED, D2)]
            if budget in ("official-r4", "fresh-r4"):
                controls = ["shared:" + a for a in ("matched", "rec", "swap", "d1")]
                arms.extend(controls)
                pairs.extend((SHARED, method) for method in controls)
                pairs.extend([("shared:swap", "shared:rec"), ("shared:swap", "shared:matched"),
                              ("shared:d1", "shared:swap")])
            if budget == "official-r80":
                arms.extend([FULL, FULL_CODE, ATM])
                pairs.extend([(FULL, FULL_CODE), (FULL, PAPER), (FULL_CODE, CODE),
                              (D2, FULL), (D2, FULL_CODE), (SHARED, FULL), (SHARED, FULL_CODE),
                              (FULL, ATM), (FULL_CODE, ATM), (D2, ATM), (SHARED, ATM)])
        result[budget] = arms, pairs
    return result


def coverage_index(raw):
    manifest = json.loads(raw)
    sections = {}
    for k in ("rows", "features", "inception_official"):
        rows = manifest.get(k, [])
        sections[k] = {r["folder"]: r for r in rows}
        agg.require(len(sections[k]) == len(rows), f"Duplicate {k} entry in coverage.json")
    return sections


def check_export(index, folder, source, n):
    expected = index.get(folder)
    if source is None:
        agg.require(expected is None, f"Export lists a missing case at {folder}")
        return
    agg.require(expected is not None and expected["n"] == n and
                all(source[k] == expected[k] for k in ("observations_sha256", "scores_sha256")),
                f"Case disagrees with the committed export at {folder}. Finish synchronizing the snapshot.")


def feature_case(root, folder, track, arm, budget, sub, reg, image_source):
    n = agg.PANELS[budget.split("-")[0]][0]
    meta_raw = (folder / "features.json").read_bytes()
    obs_raw = (folder / "features.npz").read_bytes()
    meta = json.loads(meta_raw)
    proof = {"scores_sha256": digest(meta_raw), "observations_sha256": digest(obs_raw)}
    agg.require(meta["observations_sha256"] == proof["observations_sha256"] and meta["training"] == track
                and meta["arm"] == arm and meta["budget"] == budget and agg.subject_id(meta["subject"]) == sub
                and meta["n"] == meta["gallery"] == n, f"Inconsistent retrieval receipt at {folder}")
    repeats = np.arange(80).reshape(20, 4).tolist() if budget == "official-r80" else (
        [[0, 20, 40, 60]] if budget == "official-r4-balanced" else [[0, 1, 2, 3]])
    agg.require(meta["repeats"] == repeats and meta["stimulus_split_sha256"] == reg["hashes"]["paper_split.json"],
                f"Wrong physical trials or stimulus split at {folder}")
    for k in ("checkpoint_sha256", "target_sha256"):
        agg.require(agg.valid_sha(meta[k]), f"Invalid {k} at {folder}")
        proof[k] = meta[k]
    if image_source and "checkpoint_sha256" in image_source:
        agg.require(meta["checkpoint_sha256"] == image_source["checkpoint_sha256"], f"EEG encoder differs from render at {folder}")
    visual_receipt = root / "data/prep/paper_test_F_receipt.json"
    if budget.startswith("official-") and visual_receipt.is_file():
        raw = visual_receipt.read_bytes()
        ref = json.loads(raw)
        ids = json.loads((root / "data/prep/test_ids.json").read_bytes())["ids"]
        agg.require(ref["ordered_ids"] == ids and ref["F_test_sha256"] == meta["target_sha256"],
                    f"Incorrect official visual feature target at {folder}")
        proof["visual_reference_receipt_sha256"] = digest(raw)
    metrics = ["feature_top1", "feature_top5", "rank", "positive_cosine"]
    if budget == "fresh-r4":
        metrics.insert(2, "feature_within10")
    with np.load(io.BytesIO(obs_raw), allow_pickle=False) as obs:
        agg.require(set(obs.files) == {"row", *metrics} and set(meta["mean"]) == set(metrics),
                    f"Retrieval score schema differs at {folder}")
        order = agg.row_order(obs["row"], n)
        agg.require(np.array_equal(obs["row"][order], np.sort(reg["rows"])), f"Retrieval gallery identity differs at {folder}")
        cols = []
        for m in metrics:
            v = obs[m]
            agg.require(v.shape == (n,) and v.dtype.kind in "biuf" and np.isfinite(v).all()
                        and np.isclose(v.mean(dtype=np.float64), meta["mean"][m], rtol=1e-6, atol=1e-8),
                        f"Invalid retrieval scores or mean for {m} at {folder}")
            if m.startswith("feature_"):
                agg.require(np.all((v >= 0) & (v <= 1)), f"Invalid retrieval credit for {m} at {folder}")
            if m == "rank":
                agg.require(v.dtype.kind in "iu" and np.all((1 <= v) & (v <= n)), f"Invalid retrieval rank at {folder}")
            if m == "positive_cosine":
                agg.require(np.all(np.abs(v) <= 1 + 1e-6), f"Invalid cosine at {folder}")
            cols.append(v[order])
        agg.require(np.all(obs["feature_top1"] <= obs["feature_top5"]) and
                    all(np.all(obs[f"feature_top{k}"][obs["rank"] > k] == 0) for k in (1, 5)),
                    f"Retrieval credits disagree with ranks at {folder}")
    return metrics, np.column_stack(cols).astype(np.float64), proof


def feature_stats(left, right, people, labels, panel, metrics):
    a = np.stack([left[p] for p in people])
    b = np.stack([right[p] for p in people]) if right is not None else None
    x = a if b is None else a - b
    overall, conditional = agg.bootstrap(agg.units(x, labels, panel), DRAWS, SEED)
    output, per = {}, {p: {} for p in people}
    key = "mean" if b is None else "difference_a_minus_b"
    for j, m in enumerate(metrics):
        v = x[:, :, j]
        stats = {key: float(v.mean()), "crossed_ci95": overall[:, j].tolist() if len(people) >= 2 else None,
                 "better": "lower" if m == "rank" else "higher"}
        if b is not None:
            stats.update(mean_a=float(a[:, :, j].mean()), mean_b=float(b[:, :, j].mean()))
        if m.startswith("feature_"):
            stats["credit_denominator"] = int(v.size)
            if b is None:
                stats["credit"] = float(a[:, :, j].sum())
            else:
                stats.update(credit_a=float(a[:, :, j].sum()), credit_b=float(b[:, :, j].sum()))
        output[m] = stats
        for i, p in enumerate(people):
            per[p][m] = {key: float(v[i].mean()), "stimulus_ci95": conditional[:, i, j].tolist()}
    return output, per


def features(root, images, index):
    result = {}
    target_hash = {}
    for budget, matrix in images.items():
        if budget == "dev-r4":
            continue
        panel = matrix["panel"]
        reg = agg.registration(root, panel)
        arms = [PAPER, D2, SHARED] + ([FULL] if budget == "official-r80" else [])
        # Read additional measured native/D2 conditions if the export contains them.
        arms += sorted({":".join(f.split("/")[1:3]) for f in index
                        if len(f.split("/")) == 5 and f.split("/")[3] == budget} - set(arms))
        pairs = [(D2, PAPER), (SHARED, PAPER)] + ([(FULL, PAPER)] if FULL in arms else [])
        store, arms_out = {}, {}
        metrics = []
        labels = None
        for selector in arms:
            track, arm = selector.split(":")
            store[selector], sources = {}, {}
            for sub in agg.SUBJECTS:
                rel = f"features/{track}/{arm}/{budget}/{sub}"
                folder = root / rel
                if not (folder / "features.json").is_file() or not (folder / "features.npz").is_file():
                    check_export(index, rel, None, 0)
                    continue
                render = matrix["arms"].get(selector, {}).get("sources", {}).get(sub)
                metrics, values, source = feature_case(root, folder, track, arm, budget, sub, reg, render)
                check_export(index, rel, source, len(values))
                agg.require(target_hash.setdefault(panel, source["target_sha256"]) == source["target_sha256"],
                            f"Mixed visual feature targets in the {panel} panel")
                store[selector][sub], sources[sub] = values, source
            people = list(store[selector])
            n = matrix["expected_photographs_per_participant"]
            rec = {"participants": people, "missing": [p for p in agg.SUBJECTS if p not in people],
                   "scored_queries": len(people) * n, "sources": sources, "metrics": {}, "per_participant": {}}
            if people:
                rows = np.sort(reg["rows"])
                labels = rows if panel == "official" else reg["truth"][rows]
                rec["metrics"], rec["per_participant"] = feature_stats(store[selector], None, people, labels, panel, metrics)
            arms_out[selector] = rec
        pairs_out = []
        for a, b in pairs:
            people = [p for p in agg.SUBJECTS if p in store[a] and p in store[b]]
            rec = {"arm_a": a, "arm_b": b, "participants": people,
                   "scored_query_pairs": len(people) * n, "metrics": {}, "per_participant": {}}
            if people:
                rec["metrics"], rec["per_participant"] = feature_stats(store[a], store[b], people, labels, panel, metrics)
            pairs_out.append(rec)
        result[budget] = {"expected_participants": list(agg.SUBJECTS), "gallery": n,
                          "arms": arms_out, "comparisons": pairs_out}
    return result


def supplementary_case(root, rel, reg, index, arm, subject):
    folder = root / rel
    if not all((folder / f).is_file() for f in ("render.json", "scores.json", "observations.npz")):
        check_export(index, rel, None, 0)
        return None
    metrics, rows, labels, words, values, scores, proof = agg.read_scores(folder, "dev")
    agg.check_gallery(reg, rows, labels, "dev", {"track": "personal"})
    check_export(index, rel, proof, len(rows))
    raw = (folder / "render.json").read_bytes()
    meta = json.loads(raw)
    proof.update(folder=rel, render_sha256=digest(raw))
    agg.require(scores.get("render_sha256") == proof["render_sha256"] and "ordered_image_hashes_sha256" in scores,
                f"Scorer did not bind the supplementary render and image hashes at {rel}")
    image_hashes = meta["pixels_sha256"]
    agg.require(len(image_hashes) == len(rows) and all(agg.valid_sha(h) for h in image_hashes),
                f"Invalid supplementary image hashes at {rel}")
    agg.check_scored_images(scores, image_hashes, proof)
    n = len(rows)
    seeds = agg.panel_seeds("dev", n)
    agg.require(np.array_equal(meta["row"], reg["rows"]) and np.array_equal(meta["seeds"], seeds)
                and meta["draws"] == 1 and meta["unet_calls_per_draw"] == 4, f"Wrong gallery, seed or denoising budget at {rel}")
    dev_hash = reg["hashes"]["paper_development.json"]
    if subject is None:
        agg.require(meta["input"] == arm and meta["eeg"] is False and meta["panel"] == "dev"
                    and meta["prompt"] == "" and meta["split_sha256"] == dev_hash
                    and agg.valid_sha(meta["source_features_sha256"]), f"Invalid no-EEG calibration input at {rel}")
        proof["source_features_sha256"] = meta["source_features_sha256"]
    else:
        identity = np.arange(n)
        text_donor = [] if arm == "soft-mean" else (np.roll(identity, n // 2) if arm == "soft-wrong" else identity)
        agg.require(meta["arm"] == arm and agg.subject_id(meta["subject"]) == subject and meta["dev_sha256"] == dev_hash
                    and meta["repeats"] == [[0, 1, 2, 3]] and meta["text_supervision"] == "Training object names only",
                    f"Wrong soft-text arm, participant, training supervision, trials or split at {rel}")
        agg.require(np.array_equal(meta["ip_eeg_donor"], identity) and np.array_equal(meta["text_eeg_donor"], text_donor),
                    f"Incorrect image-token or text donor at {rel}")
        if arm == "soft-wrong":
            truth = reg["truth"][reg["rows"]]
            agg.require(np.all(truth != truth[text_donor]), f"Wrong-text shift kept the true concept at {rel}")
        for k in ("native_checkpoint_sha256", "soft_checkpoint_sha256"):
            agg.require(agg.valid_sha(meta[k]), f"Invalid {k} at {rel}")
            proof[k] = meta[k]
    proof["development_split_sha256"] = dev_hash
    return agg.Case(metrics, rows, labels, words, values, proof, seeds[np.argsort(reg["rows"])], None)


def supplementary(root, index, images, feature_matrices):
    calibration_arms = ("original-photo", "training-mean")
    text_arms = ("soft-mean", "soft-eeg", "soft-wrong")
    reg = agg.registration(root, "dev")
    subjects = [agg.subject_id(s) for s in (reg["development"] or {"subjects": [1, 3]})["subjects"]]
    calibration, text_cases = {}, {a: {} for a in text_arms}
    for arm in calibration_arms:
        case = supplementary_case(root, f"calibration/dev/{arm}", reg, index, arm, None)
        if case is not None:
            calibration[arm] = case
    for arm in text_arms:
        for sub in subjects:
            case = supplementary_case(root, f"exploratory/{arm}/dev-r4/{sub}", reg, index, arm, sub)
            if case is not None:
                text_cases[arm][sub] = case
    cases = [*calibration.values(), *(c for arms in text_cases.values() for c in arms.values())]
    if not cases:
        return None
    # Frozen original-name outputs should also agree with the formal development scorer.
    baseline = next((src for arm in images["dev-r4"]["arms"].values() for src in arm["sources"].values()), None)
    if baseline is not None:
        metrics, rows, labels, words, vals, _, proof = agg.read_scores(root / baseline["folder"], "dev")
        cases.append(agg.Case(metrics, rows, labels, words, vals, proof, None, None))
    ref = agg.compare_sources(cases)
    feature_hashes = {c.source["source_features_sha256"] for c in calibration.values()}
    for arm in feature_matrices.get("fresh-r4", {}).get("arms", {}).values():
        feature_hashes.update(s["target_sha256"] for s in arm["sources"].values())
    agg.require(len(feature_hashes) <= 1, "Calibration and fresh retrieval disagree on the source of frozen visual features")
    for sub in subjects:
        native = {c[sub].source["native_checkpoint_sha256"] for c in text_cases.values() if sub in c}
        for m in images.values():
            native.update(a["sources"][sub]["checkpoint_sha256"] for key, a in m["arms"].items()
                          if key == PAPER and sub in a["sources"] and "checkpoint_sha256" in a["sources"][sub])
        soft = {c[sub].source["soft_checkpoint_sha256"] for c in text_cases.values() if sub in c}
        agg.require(len(native) <= 1 and len(soft) <= 1, f"Native image condition or soft-text checkpoint differs for {sub}")
    output = {"panel": "dev-r4", "gallery": len(ref.rows), "concept_blocks": len(np.unique(ref.labels)),
              "registration_sha256": reg["hashes"], "included_in_main_coverage": False, "calibration": {},
              "class_name_supervision": {"expected_participants": subjects, "arms": {}, "comparisons": []}}
    for arm, c in calibration.items():
        stats, item_ci = agg.describe(c.values[None], None, ["gallery"], c.labels, "dev", c.metrics, DRAWS, SEED)
        for key, v in stats.items():
            v.pop("crossed_ci95")
            v["stimulus_ci95"] = item_ci["gallery"][key]["stimulus_ci95"]
        stats["original_name1854"]["scope"] = "Reference naming on 200 original photographs."
        output["calibration"][arm] = {"eeg": False, "participants": [], "scored_images": len(c.rows),
                                      "metrics": stats, "source": c.source}
    text = output["class_name_supervision"]
    for arm, by_sub in text_cases.items():
        people = list(by_sub)
        stats, per = {}, {}
        if people:
            a = np.stack([by_sub[p].values for p in people])
            stats, per = agg.describe(a, None, people, ref.labels, "dev", ref.metrics, DRAWS, SEED)
        text["arms"][arm] = {"participants": people, "missing": [p for p in subjects if p not in people],
                             "scored_images": len(people) * len(ref.rows), "metrics": stats, "per_participant": per,
                             "sources": {p: c.source for p, c in by_sub.items()}}
    for b in ("soft-mean", "soft-wrong"):
        a = "soft-eeg"
        people = [p for p in subjects if p in text_cases[a] and p in text_cases[b]]
        stats, per = {}, {}
        if people:
            left, right = (np.stack([text_cases[k][p].values for p in people]) for k in (a, b))
            stats, per = agg.describe(left, right, people, ref.labels, "dev", ref.metrics, DRAWS, SEED)
        text["comparisons"].append({"arm_a": a, "arm_b": b, "participants": people,
                                    "scored_image_pairs": len(people) * len(ref.rows), "metrics": stats, "per_participant": per})
    return output


def published(external_file):
    """Use the external audit's actual table and link targets, and the pinned Brain-HiVE table crops."""
    raw = external_file.read_bytes()
    text = raw.decode("utf-8-sig")
    table = []
    for name, cite in [("ATM，表 1", "ATM 表 1"), ("D²-FOSA，表 3", "CVF 主文表 3"),
                       ("Jo 等，表 1", "Jo 等表 1 与图 4"), ("ProtoEEG，表 3", "ProtoEEG 表 3"), ("SPSC，表 17", "SPSC 表 17")]:
        rows = [line for line in text.splitlines() if line.startswith("| " + name + " |")]
        links = re.findall(r"\[" + re.escape(cite) + r"\]\((https?://[^)]+)\)", text)
        agg.require(len(rows) == len(links) == 1, f"Missing or ambiguous literature row or citation for {name}")
        fields = [c.strip() for c in rows[0].strip("|").split("|")]
        agg.require(len(fields) == 9, f"Unexpected literature columns for {name}")
        values = {k: (None if v == "—" else float(v)) for k, v in zip(SECONDARY[:-1], fields[2:], strict=True)}
        table.append({"method": name, "participants": fields[1], "metrics": values, "url": links[0]})
    sources = json.loads((REPO / "docs/paper_sources.json").read_bytes())
    brain = next(s for s in sources if s["id"] == "brainhive")
    proofs = {}
    for item in brain["result_tables"]:
        path = REPO / "docs" / item["asset"]
        agg.require(digest(path.read_bytes()) == item["asset_sha256"], f"Published table crop changed at {path}")
        proofs[item["label"]] = item["asset_sha256"]
    # Transcriptions of the EEG rows in the two cited, hash-checked original table crops.
    brain_metrics = dict(zip(SECONDARY[:-1], [.195, .336, .843, .905, .756, .808, .554], strict=True))
    table.insert(0, {"method": "Brain-HiVE，表 2", "participants": "10 人均值", "metrics": brain_metrics,
                     "url": brain["url"]})
    return {"image_reconstruction_author_reported": table,
            "brainhive_eeg_retrieval_author_reported": {"intra_subject_gallery": 200, "top1": .757,
                "top5": .946, "visual_targets": ["RN50", "B32", "VAE"], "table": "1", "url": brain["url"]},
            "brainhive_table_images_sha256": proofs, "external_report_sha256": digest(raw)}


def build(root, external_file):
    coverage = root / "coverage.json"
    before = coverage.read_bytes()
    index = coverage_index(before)
    image_matrices = {}
    for budget, (arms, pairs) in plans().items():
        result = agg.aggregate(root, "personal", budget, arms, pairs=pairs, resamples=DRAWS, seed=SEED)
        for name, arm in result["arms"].items():
            for sub in result["requested_participants"]:
                rel = (f"external/atm-sub{sub[-2:]}" if name == ATM else
                       f"images/{arm['track']}/{arm['method']}/{budget}/{sub}")
                check_export(index["rows"], rel, arm["sources"].get(sub), result["expected_photographs_per_participant"])
        image_matrices[budget] = result
    feature_matrices = features(root, image_matrices, index["features"])
    extra = supplementary(root, index["rows"], image_matrices, feature_matrices)
    dino = paper_dino_report.build(root, before)
    object_studies = paper_object_report.build(root, before)
    transitions = paper_error_readout.analyze(root, image_matrices, feature_matrices)
    corrected = paper_inception_report.attach(root, index["inception_official"], index["rows"], image_matrices, extra,
                                              DRAWS, SEED, object_studies)
    literature = published(external_file)
    agg.require(coverage.read_bytes() == before, "The verified input snapshot changed during aggregation. Re-run after synchronization.")
    total = sum(len(m["arms"]) * len(m["requested_participants"]) for m in image_matrices.values())
    observed = sum(len(a["participants"]) for m in image_matrices.values() for a in m["arms"].values())
    q = sum(a["scored_images"] for m in image_matrices.values() for a in m["arms"].values())
    modules = (agg, paper_dino_report, paper_error_readout, paper_object_report, paper_inception_report, paper_inception_report.scorer)
    scripts = [Path(__file__), *(Path(module.__file__) for module in modules)]
    proofs = {f.name: digest(f.read_bytes()) for f in scripts}
    result = {"schema": 1, "coverage_sha256": digest(before), "report_code_sha256": proofs,
            "bootstrap": {"resamples": DRAWS, "seed": SEED, "interval": .95, "fixed_candidate_galleries": True},
            "coverage": {"planned_image_cases": total, "scored_image_cases": observed, "scored_reconstructions": q,
                         "image_complete": observed == total, "feature_cases": sum(len(a["participants"])
                             for m in feature_matrices.values() for a in m["arms"].values()),
                         "dino_cases": dino["scored_cases"], "planned_dino_cases": dino["planned_cases"]},
            "image_matrices": image_matrices, "eeg_feature_matrices": feature_matrices, "literature": literature,
            "dino_report": dino, "object_first_studies": object_studies, "feature_to_generated_object": transitions}
    result["inception_official"] = corrected
    result["coverage"]["inception_official_formal_cases"] = corrected["formal_scored_cases"]
    result["coverage"]["inception_official_auxiliary_cases"] = corrected["auxiliary_scored_cases"]
    if extra is not None:
        result["supplementary_development"] = extra
    return result


def arm_label(selector):
    if selector == ATM:
        return "ATM 发布图"
    track, method = selector.split(":")
    if method in ("token-peer-own", "token-peer-sibling"):
        return "单人输入、九人辅助训练 " + method
    return {"personal": "逐人", "shared": "共享", "native-full": "全 1654"}[track] + " " + method


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(str(c) for c in cells) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def credit_text(x):
    return f"{x:.3f}".rstrip("0").rstrip(".")


def abs_metric(metrics, key):
    if key not in metrics:
        return "—"
    m = metrics[key]
    if key == "ssim":
        return f"{m['mean']:.4f}"
    return f"{100*m['mean']:.2f}%（{credit_text(m['credit'])}/{m['credit_denominator']}）"


def delta_text(pair, metric):
    if metric not in pair["metrics"]:
        return "—"
    stat = pair["metrics"][metric]
    scale = 1 if metric in ("ssim", "pixcorr", "positive_cosine", "rank") or metric.endswith("_distance") else 100
    ci = stat["crossed_ci95"]
    if len(pair["participants"]) == 1:
        ci = pair["per_participant"][pair["participants"][0]][metric]["stimulus_ci95"]
    digits = 4 if scale == 1 else 2
    value = stat["difference_a_minus_b"] * scale
    return f"{value:+.{digits}f} [{ci[0]*scale:+.{digits}f}, {ci[1]*scale:+.{digits}f}]"


def contrast(m, a, b):
    return next(p for p in m["comparisons"] if p["arm_a"] == a and p["arm_b"] == b)


def interval_label(people):
    if len(people) == 1:
        return people[0] + "，刺激区间"
    return f"{len(people)} 人，交叉区间"


def reading(matrices):
    paragraphs = []
    # The direct comparisons are specified by arm and budget regardless of outcomes.
    for budget, endpoint, end_name in [("official-r4", "generated_name1854", "物体命名"),
                                       ("official-r80", "generated_name1854", "物体命名"),
                                       ("fresh-r4", "clip_within10", "同类十图识别")]:
        m = matrices[budget]
        p = contrast(m, D2, PAPER)
        if not p["participants"]:
            paragraphs.append(f"{BUDGETS[budget]} 的逐人 D2 与 native-paper 尚无共同完成的被试。")
            continue
        v = p["metrics"][endpoint]
        cohort = interval_label(p["participants"])
        other_key, other_name = ("clip_image_top1", "全图库 Top-1") if budget.startswith("official") else ("generated_name1854", "物体命名")
        paragraphs.append(f"**{BUDGETS[budget]}。** 逐人 D2 与 native-paper 在 {p['scored_image_pairs']:,} 个相同被试–刺激条件上配对（{cohort}）。"
            f"{end_name}得分分别为 {credit_text(v['credit_a'])}/{v['credit_denominator']:,} 和 "
            f"{credit_text(v['credit_b'])}/{v['credit_denominator']:,}，差 {delta_text(p, endpoint)} 个百分点。"
            f"{other_name}差值为 {delta_text(p, other_key)} 个百分点，SSIM 相差 {delta_text(p, 'ssim')}。")
    return paragraphs


def supplement_blocks(extra):
    if extra is None:
        return []
    blocks = ["## 原图视觉特征参照与训练类名监督的探索",
              "下列附项使用开发集的 20 个物体、每类十张照片，单列其计分张数。正式基准的覆盖分母保持 368。"]
    if extra["calibration"]:
        blocks += ["### 同一冻结生成器的视觉条件",
                   "两组均无 EEG 输入，每张照片只生成一次。original-photo 将目标原图的融合视觉特征交给生成器，衡量给定真实视觉条件时可达到的水平。"
                   "training-mean 对每个查询传入同一训练图库均值。二者使用相同空文本、四步采样和逐刺激随机种子。"]
        rows = [[arm, 0, s["scored_images"], *(abs_metric(s["metrics"], k) for k in MAIN)]
                for arm, s in extra["calibration"].items()]
        blocks.append(table(["输入", "EEG 人数", "生成图", "物体命名", "图片 Top-1", "同类十图", "SSIM"], rows))
    language = extra["class_name_supervision"]
    if any(v["participants"] for v in language["arms"].values()):
        blocks += ["### 使用训练物体名监督的连续文字条件",
                   "三组都使用同一被试、同一张照片的四次 EEG 与 native-paper 图像条件。训练额外使用训练物体名，推理时直接输入连续 SDXL 文字状态，查询时无候选名册。"
                   "soft-mean 使用固定训练均值文字条件，soft-eeg 使用本查询 EEG 预测的文字状态，soft-wrong 只把文字状态循环移至相隔 100 张的另一物体。"
                   "与正式纯视觉监督方法相比，这里额外获得了训练类名。"]
        rows = [[arm, f"{len(s['participants'])}/{len(language['expected_participants'])}", s["scored_images"],
                 *(abs_metric(s["metrics"], k) for k in MAIN)] for arm, s in language["arms"].items() if s["participants"]]
        blocks.append(table(["方法", "人数", "生成图", "物体命名", "图片 Top-1", "同类十图", "SSIM"], rows))
        pairs = [[p["arm_a"] + " − " + p["arm_b"], interval_label(p["participants"]), p["scored_image_pairs"],
                  *(delta_text(p, k) for k in MAIN)] for p in language["comparisons"] if p["participants"]]
        if pairs:
            blocks += ["以下差值只使用两组都已计分的被试与照片。95% 区间按同一批被试和 20 个完整物体重抽 10,000 次。"
                       "只有一名被试时，区间仅反映物体抽样。前三项单位为百分点。",
                       table(["A − B", "共同被试与区间", "成对查询", "Δ命名", "Δ图片 Top-1", "Δ同类十图", "ΔSSIM"], pairs)]
    return blocks


def render(data):
    images, feat = data["image_matrices"], data["eeg_feature_matrices"]
    cov = data["coverage"]
    links = ("[逐人结果、分母、来源散列与全部配对区间](../research/eeg_stack_20261112/results/paper_benchmark.json) · "
             "[汇总程序](../research/eeg_stack_20261112/code/build_paper_benchmark_report.py)")
    blocks = ["# EEG 重建原始照片的配对基准",
              f"这份快照覆盖预设的 **{cov['scored_image_cases']}/{cov['planned_image_cases']} 个被试–方法–预算组合**，"
              f"共 **{cov['scored_reconstructions']:,} 张计分生成图**。每张图都与对应的 THINGS 原始照片比较。{links}",
              *reading(images), "## 已完成的被试"]
    selectors = list(dict.fromkeys(a for m in images.values() for a in m["arms"]))
    cover_rows = []
    for a in selectors:
        cells = []
        for m in images.values():
            cells.append(f"{len(m['arms'][a]['participants'])}/{len(m['requested_participants'])}" if a in m["arms"] else "未安排")
        cover_rows.append([arm_label(a), *cells])
    blocks += [table(["方法", *BUDGETS.values()], cover_rows),
               "四个主面板各计划十名被试。开发集预先指定 sub-01、sub-03，用于方法选择。"
               "连续 R4 取呈现序号 0、1、2、3，跨组 R4 取 0、20、40、60，R80 使用同一位被试的全部 80 次。"
               "官方面板有 200 个物体、每个物体一张照片。新概念面板含 50 类 × 10 张，开发集含另 20 类 × 10 张。"
               "后两个面板排除了以往用于出图的 50 类。历史上选择 M1a 读出时，在同一批保留概念中测过 327/331 类 EEG 表征；"
               "这与新概念面板的 46–50 类、开发集的 16–20 类有交集。两批照片在本次训练中均整类留出。",
               "逐人及共享的受限训练均使用 1,323 个训练概念。共享 D2 的 EEG 语义编码器用十人的训练数据，每位被试另拟合自己的布局回归。"
               "全 1654 轨按被试用全部 1,654 个训练概念重训 Brain-HiVE。native-paper 对 EEG 表征施加融合适配器后再投影，native-code 直接投影。"
               "本库每个查询使用空文本、一个按刺激编号预设的随机种子和四次去噪。ATM 固定取作者发布的第零张候选，同一条件另有 50 步扩散先验，其随机种子由作者控制。",
               "wrong 将 D2 的 EEG 条件及布局移给另一查询。官方面板移到另一物体，新概念和开发集在同一物体的十张照片内循环移动。"
               "同类错配仍保留物体类别信息，因此这个控制检验同类照片的身份。native-layout 配合 native-paper 条件与逐人布局，d2-no-layout 从纯噪声生成。"
               "instance-native、instance-direct、instance-residual 均从纯噪声生成。第一支沿用 native-paper 编码器，后两支使用相同的逐人校准参数包。"
               "调参使用训练制度内其他物体的影子模型，开发集两名被试用于选型。",
               "## 从生成图读取目标",
               "物体命名使用冻结 CLIP 在 1,854 个名称中选择，并列名称分享命中分。图片 Top-1 用生成图检索该面板全部原图（特征维度中心化的 Pearson 相似度）。"
               "同类十图使用去除各自图库均值后的 CLIP 余弦相似度。SSIM 衡量亮度结构。各行按该方法实际完成人数计算，分数括号内为命中分与查询总数。"]
    ref_rows = []
    for budget in ("official-r4", "fresh-r4", "dev-r4"):
        ref = images[budget]["reference"]
        if ref:
            ref_rows.append([BUDGETS[budget], ref["scored_photographs"],
                             f"{100*ref['mean']:.3f}%（{credit_text(ref['original_name1854_credit'])}/{ref['scored_photographs']}）"])
    if ref_rows:
        blocks += ["同一命名器读取真实原图的结果如下。每张原图只计一次，重名的物体词平分命中分。",
                   table(["图库", "独立原图", "原图物体命名"], ref_rows)]
    for budget, m in images.items():
        rows = [[arm_label(a), f"{len(s['participants'])}/{len(m['requested_participants'])}", str(s["scored_images"]),
                 *(abs_metric(s["metrics"], k) for k in MAIN)] for a, s in m["arms"].items() if s["participants"]]
        blocks += [f"### {BUDGETS[budget]}",
                   table(["方法", "人数", "生成图", "物体命名", "图片 Top-1", "同类十图", "SSIM"], rows)
                   if rows else "这份快照尚无这个面板的完整评分。"]
    blocks += ["## 同一批被试、同一张照片的差值",
               "每行计算 A 减 B，只使用双方都完成的被试。命名、图片 Top-1 与同类十图的单位为百分点，SSIM 使用原始量纲。"
               "95% 百分位区间用 10,000 次有放回重抽样（种子 20261125）。两人以上独立重抽被试与共同刺激，每次对所有被试及两种方法使用同一组刺激。"
               "官方重抽照片，新概念与开发集重抽完整物体类别，保留十张照片和类内的 45 对无序配对。单人区间只重抽样刺激。"
               "各区间针对固定候选图库、模型和生成抽样，逐指标计算。"]
    for budget, m in images.items():
        rows = [[arm_label(p["arm_a"]) + " − " + arm_label(p["arm_b"]), interval_label(p["participants"]),
                 str(p["scored_image_pairs"]), *(delta_text(p, k) for k in MAIN)] for p in m["comparisons"] if p["participants"]]
        blocks += [f"### {BUDGETS[budget]}配对差",
                   table(["A − B", "共同被试与区间", "成对查询", "Δ命名", "Δ图片 Top-1", "Δ同类十图", "ΔSSIM"], rows)
                   if rows else "这份快照尚无该面板的完整配对。"]
    blocks += ["## 像素、结构与视觉特征",
               "各表均对原始 RGB 照片与生成图重新提取特征。Alex2、Alex5、Inception、CLIP 均为目标原图胜过同图库干扰照片的比例。"
               "PixCorr 是像素相关，EffNet 和 SwAV 为相关距离，越低越好。每个方法的逐人结果与全部特征差值区间均可从机器结果读取。",
               f"对照 Brain-HiVE 原表使用“官方 Inception”一列。该列已按同一批原始 RGB 补算正式组合中的 "
               f"{cov['inception_official_formal_cases']}/{cov['planned_image_cases']} 组。括号给出补算人数与本行已有 RGB 评分人数。"
               "照片先按短边缩放并中心裁切到 512，再对张量缩放到 342。ImageNet 归一化后，官方预训练模型执行内部通道变换。"
               "旧实现省略了该变换，历史列保留当时的分数。两列均用 avgpool 特征的 Pearson 相关作严格二选一辨认。"
               "使用官方权重和两张固定测试输入，修正模型的池化特征与官方模型直接执行、FX 特征提取的结果逐值相同。"
               "旧模型特征的最大绝对差为 2.9425，约为官方输出峰值的 62.9%。"
               "官方 Inception 的方法间区间只使用双方均已补算的同人同图。"
               " [原作者评估](https://github.com/ssshamiii/Brain-HIVE/blob/f06db69fcc4a029439e369c5c3dba6099e57560e/main/utils_eval.py) · "
               "[独立补算程序](../research/eeg_stack_20261112/code/paper_inception_delta.py)"]
    for budget, m in images.items():
        rows = []
        for a, s in m["arms"].items():
            if s["participants"]:
                cells = [paper_inception_report.table_cell(s) if k == "inception_2way" else f"{s['metrics'][k]['mean']:.4f}"
                         for k in SECONDARY]
                rows.append([arm_label(a), len(s["participants"]), *cells, f"{s['metrics']['inception_2way']['mean']:.4f}"])
        if rows:
            headers = ["官方 Inception（人数）" if h == "Inception" else h for h in SECONDARY_HEADERS]
            blocks += [f"### {BUDGETS[budget]}的图像特征", table(["方法", "人数", *headers, "历史 Inception（旧输入）"], rows)]
    blocks += ["### 多照片物体的配对与中心化敏感性",
               "每类十张照片形成 45 对。交换配对分数比较两张生成图与两张原图的正确对应和交叉对应，平局计半分。"
               "中心化全图库 Top-1 和未中心化的同类十图为补充指标。"]
    rows = []
    for budget in ("fresh-r4", "dev-r4"):
        m = images[budget]
        for name, s in m["arms"].items():
            if s["participants"]:
                n = m["expected_photographs_per_participant"]
                rows.append([BUDGETS[budget], arm_label(name), len(s["participants"]),
                             *(abs_metric(s["metrics"], k) for k in ("clip_within_swap", f"clip_global{n}_centered",
                                                                    "clip_within10_uncentered"))])
    blocks.append(table(["图库", "方法", "人数", "同类交换配对", "中心化全图库 Top-1", "未中心化同类十图"], rows)
                  if rows else "这份快照尚无多照片物体的完整评分。")
    blocks += ["## 独立 EEG 表征检索",
               "Brain-HiVE 论文在同被试、200 选一的专门检索配置中报告 **Top-1 75.7%、Top-5 94.6%**，"
               "视觉目标使用 RN50、B32、VAE。本次自检在扩散采样前，用重建模型的 EEG 输出检索 H14、B32、VAE 的冻结融合表征。"
               "这一检查衡量重建条件中的刺激信息，复核原论文的检索数值需要同一专用检索配置。生成图的识别率见前面的 RGB 表。"
               " [原论文检索表](assets/brainhive_table_1.png) · [Brain-HiVE 论文](https://arxiv.org/html/2602.07495) · "
               "[独立表征检索实现](../research/eeg_stack_20261112/code/paper_retrieval.py)"]
    feature_rows, feature_pairs = [], []
    for budget, m in feat.items():
        for a, record in m["arms"].items():
            if not record["participants"]:
                continue
            s = record["metrics"]
            feature_rows.append([BUDGETS[budget], arm_label(a), f"{len(record['participants'])}/{len(m['expected_participants'])}",
                                 record["scored_queries"], abs_metric(s, "feature_top1"), abs_metric(s, "feature_top5"),
                                 abs_metric(s, "feature_within10"), f"{s['rank']['mean']:.2f}"])
        for p in m["comparisons"]:
            if p["participants"]:
                feature_pairs.append([BUDGETS[budget], arm_label(p["arm_a"]) + " − " + arm_label(p["arm_b"]),
                                     interval_label(p["participants"]), p["scored_query_pairs"],
                                     *(delta_text(p, k) for k in ("feature_top1", "feature_top5", "feature_within10"))])
    if feature_rows:
        blocks.append(table(["预算", "EEG 表征", "人数", "查询", "Top-1", "Top-5", "同类十图", "平均名次 ↓"], feature_rows))
    else:
        blocks.append("本快照尚无完整的 EEG 表征检索评分文件。")
    if feature_pairs:
        blocks += ["表征检索的同人、同图配对差及 95% 区间，单位为百分点。",
                   table(["预算", "A − B", "共同被试与区间", "成对查询", "ΔTop-1", "ΔTop-5", "Δ同类十图"], feature_pairs)]
    blocks.extend(supplement_blocks(data.get("supplementary_development")))
    blocks.append(paper_error_readout.markdown(data["feature_to_generated_object"]))
    blocks.append(paper_object_report.markdown(data["object_first_studies"]))
    blocks += ["## 原论文与发布图像",
               "下表为作者主表的原报数值。Brain-HiVE 的重建表与专门检索表各有自己的评估。"
               "ATM 原表只列被试 08。D²-FOSA 的重建表未列被试范围，Jo 等报告十人均值并使用 50 次 Euler 函数评估。"
               "这些汇总数值缺少逐人逐图配对，本文只对有原始评分记录的方法计算差值区间。"]
    literature = data["literature"]["image_reconstruction_author_reported"]
    pubrows = []
    for r in literature:
        cells = [f"{r['metrics'][k]:.3f}" if r["metrics"].get(k) is not None else "—" for k in SECONDARY]
        pubrows.append([r["method"], r["participants"], *cells, f"[原表]({r['url']})"])
    blocks += [table(["作者主表", "人数范围", *SECONDARY_HEADERS, "出处"], pubrows)]
    native = images["official-r80"]["arms"][ATM]
    if native["participants"]:
        s = native["metrics"]
        blocks += [f"ATM 发布图第零张候选已用本库评分器评估 **{len(native['participants'])}/10 人、{native['scored_images']:,} 张图**。"
                   f"物体命名 {abs_metric(s, 'generated_name1854')}，图片 Top-1 {abs_metric(s, 'clip_image_top1')}，"
                   f"PixCorr {s['pixcorr']['mean']:.4f}，SSIM {s['ssim']['mean']:.4f}。上表的 ATM 原报列可另与被试 08 对照。"
                   " [作者发布图像归档](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/blob/dbe34bb2407164f70883c661c76664f0d596e522/generated_imgs.tar.gz)"]
        if "sub-08" in native["per_participant"]:
            p = native["per_participant"]["sub-08"]
            blocks += [f"同一名被试 08 的 200 张发布图重评分为 PixCorr {p['pixcorr']['mean']:.4f}、"
                       f"SSIM {p['ssim']['mean']:.4f}、CLIP 两选一 {p['clip_2way']['mean']:.4f}。"]
    blocks.append(paper_dino_report.markdown(data["dino_report"]))
    blocks += ["## 复算",
               "在仓库根目录先解包公开数值评分，再生成报告。数值包包含逐刺激观测、渲染或发布清单、测试与训练照片索引和划分注册。"
               "每个图像与表征分数文件均与快照清单核对，机器结果还记录报告代码和文献表格的 SHA-256。",
               "```sh\npython research/eeg_stack_20261112/code/package_paper_scores.py --unpack research/eeg_stack_20261112/results/paper_benchmark_scores.zip outputs/paper-benchmark/verified\n"
               "python research/eeg_stack_20261112/code/build_paper_benchmark_report.py --root outputs/paper-benchmark/verified\n```",
               f"本次快照清单 SHA-256 为 `{data['coverage_sha256']}`。{links}"]
    return "\n\n".join(blocks) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--external-report", type=Path, default=RESULT.parent / "paper_benchmark_external.md")
    parser.add_argument("--json", type=Path, default=RESULT)
    parser.add_argument("--markdown", type=Path, default=DOC)
    parser.add_argument("--check", action="store_true", help="Compare the recalculated tables with the released files")
    args = parser.parse_args()
    data = build(args.root.resolve(), args.external_report)
    md = render(data)
    content = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=1, separators=(",", ":")) + "\n"
    for path, text in ((args.json, content), (args.markdown, md)):
        if args.check:
            if not path.is_file() or path.read_bytes() != text.encode("utf-8"):
                raise SystemExit(f"Recomputed report differs from {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
        tmp.write_bytes(text.encode("utf-8"))
        tmp.replace(path)
    action = "Verified" if args.check else "Saved"
    print(f"{action} {args.markdown} and {args.json}; {data['coverage']}")


if __name__ == "__main__":
    main()
