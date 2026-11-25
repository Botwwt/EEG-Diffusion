"""Aggregate frozen DINO image scores and the public figure samples from a compact export."""

import csv
import io
import json
from pathlib import Path

import numpy as np

import paper_aggregate as agg


PAPER, D2, WRONG = "personal:native-paper", "personal:d2", "personal:wrong"
SHARED, SHARED_WRONG, FULL, ATM = "shared:d2", "shared:wrong", "native-full:native-paper", "atm"
R4_ARMS = (PAPER, D2, WRONG, SHARED, SHARED_WRONG)
PLAN = {"official-r4": R4_ARMS, "fresh-r4": R4_ARMS, "official-r80": (FULL, D2, SHARED, ATM)}
R4_PAIRS = ((D2, WRONG), (SHARED, SHARED_WRONG), (D2, PAPER), (SHARED, PAPER))
PAIRS = {"official-r4": R4_PAIRS, "fresh-r4": R4_PAIRS,
         "official-r80": ((D2, FULL), (SHARED, FULL), (D2, ATM), (SHARED, ATM), (FULL, ATM))}
ARM = {PAPER: "逐人 native-paper", D2: "逐人 D2", WRONG: "逐人 D2 错配", SHARED: "共享 D2",
       SHARED_WRONG: "共享 D2 错配", FULL: "全 1,654 类逐人 native-paper", ATM: "ATM 发布图"}
BUDGET = {"official-r4": "官方 R4", "fresh-r4": "新概念 R4", "official-r80": "官方 R80"}
GLOBAL = ("dino_top1", "dino_top5", "dino_correct_cosine", "dino_2way", "pixcorr256", "ssim256")
FRESH = ("dino_within10", "dino_within10_top5", "dino_within10_2way", "dino_within10_rank")
FIG_METRICS = ("pixcorr256", "ssim256", "dino_correct_cosine", "dino_top1", "dino_top5", "dino_2way")
FIGURES = Path(__file__).resolve().parents[1] / "results/paper_benchmark_figure_scores.csv"
FIG_SHA = "761b8fcc28bcc6389948411dda915f64fa44a8e834481fc17c27f81d2c8f2871"
MODEL = {"model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
         "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
         "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"}
REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
SCORER_SHA = "a2fc71d28f3ba5a8aa24f94e99673d8a386dd47834ca3125ec1b1a2fea255598"
GALLERY = {"official": "31d85c5dbd8e0f158d7546d141b55b89f07a027217cd1a3752bec0fcfd57c441",
           "fresh": "23ff3bf5edb4456d0962af940f2647f6350309cb840edc90be5d8f3b7b6be89b"}
N_DRAWS, SEED = 10000, 20261125


def read_case(root, entry, rgb_entry, reg, selector, budget, person):
    folder = root / entry["folder"]
    panel = budget.split("-")[0]
    n = agg.PANELS[panel][0]
    meta_bytes, obs_bytes = (folder / "dino.json").read_bytes(), (folder / "dino_rows.npz").read_bytes()
    meta = json.loads(meta_bytes)
    bind = meta["binding"]
    agg.require(agg.sha(meta_bytes) == entry["scores_sha256"] and agg.sha(obs_bytes) == entry["observations_sha256"]
                and agg.valid_sha(entry["complete_matrix_sha256"]) and meta["dino_npz_sha256"] == entry["complete_matrix_sha256"]
                and entry["n"] == meta["n"] == bind["n"] == n and meta["panel"] == bind["panel"] == panel,
                f"Compact DINO export differs from its coverage or complete matrix receipt at {folder}")
    agg.require(bind["model_revision"] == REV and bind["model_files_sha256"] == MODEL and bind["gallery_sha256"] == GALLERY[panel]
                and bind["paper_split_sha256"] == reg["hashes"]["paper_split.json"] and bind["scorer_sha256"] == SCORER_SHA,
                f"Unexpected encoder, registered split or original gallery at {folder}")
    spec = agg.arm_spec(selector, "personal", {}, budget)
    inputs = bind["input_sha256"]
    name = "manifest.csv" if selector == ATM else "render.json"
    agg.require(set(inputs) == {"scores.json", "observations.npz", name}
                and all(agg.sha((folder / file).read_bytes()) == value for file, value in inputs.items())
                and rgb_entry is not None and rgb_entry["n"] == n
                and inputs["scores.json"] == rgb_entry["scores_sha256"] and inputs["observations.npz"] == rgb_entry["observations_sha256"],
                f"DINO and primary scored stimuli are bound to different files at {folder}")
    _, rows, labels, _, _, primary, _ = agg.read_scores(folder, panel)
    agg.check_gallery(reg, rows, labels, panel, spec)
    identity = bind["identity"]
    agg.require(agg.subject_id(identity["subject"]) == person, f"Wrong DINO participant at {folder}")
    if selector == ATM:
        agg.require(identity["draw"] == "0", f"Unexpected ATM candidate at {folder}")
        agg.external_source(folder, folder / name, person, rows, labels, primary, False)
        source_hashes = [r["reconstruction_sha256"] for r in csv.DictReader(io.StringIO((folder / name).read_text(encoding="utf-8-sig")))]
    else:
        agg.require((identity["training"], identity["arm"], identity["budget"]) == (spec["track"], spec["method"], budget),
                    f"Wrong DINO training track, method or budget at {folder}")
        agg.internal_source(folder, spec, person, budget, rows, labels, primary, reg, False)
        source_hashes = json.loads((folder / name).read_bytes())["pixels_sha256"]
    agg.require(agg.sha("\n".join(source_hashes).encode()) == bind["ordered_image_hashes_sha256"],
                f"DINO query hashes differ from the verified render or manifest at {folder}")
    fields = (*GLOBAL, "dino_rank", *(FRESH if panel == "fresh" else ()))
    agg.require(set(meta["per_row_metrics"]) == set(meta["mean"]) == set(fields)
                and len(meta["per_row_metrics"]) == len(fields), f"Wrong DINO score schema at {folder}")
    with np.load(io.BytesIO(obs_bytes), allow_pickle=False) as obs:
        agg.require(set(obs.files) == {"row", "concept", *fields}, f"Incorrect DINO observation columns at {folder}")
        order = agg.row_order(obs["row"], n)
        agg.require(np.array_equal(obs["row"][order], rows) and np.array_equal(agg.strings(obs["concept"], n, "concept")[order], labels),
                    f"DINO and scored RGB row/concept pairs differ at {folder}")
        values = {}
        for k in fields:
            v = obs[k]
            agg.require(v.shape == (n,) and v.dtype.kind in "biuf" and np.isfinite(v).all()
                        and np.isclose(v.mean(dtype=np.float64), meta["mean"][k], rtol=1e-7, atol=1e-9),
                        f"{k} shape, finite values or registered mean differs at {folder}")
            v = v[order].astype(np.float64)
            if k.endswith("rank"):
                bounds = (1, 10 if "within10" in k else n)
            elif k in ("dino_correct_cosine", "pixcorr256", "ssim256"):
                bounds = (-1.000002, 1.000002)
            else:
                bounds = (0, 1)
            agg.require(np.all((v >= bounds[0]) & (v <= bounds[1])), f"{k} outside its range at {folder}")
            values[k] = v
    for size, rank, pair, one, five in [(n, "dino_rank", "dino_2way", "dino_top1", "dino_top5")] + (
        [(10, "dino_within10_rank", "dino_within10_2way", "dino_within10", "dino_within10_top5")] if panel == "fresh" else []
    ):
        agg.require(np.all(values[one] <= values[five]) and np.allclose(values[pair], (size - values[rank]) / (size - 1), rtol=0, atol=1e-10),
                    f"Inconsistent DINO retrieval ranks or credits at {folder}")
    proof = {**entry, "primary_scores_sha256": inputs["scores.json"], "primary_observations_sha256": inputs["observations.npz"],
             "render_or_manifest_sha256": inputs[name], "ordered_image_hashes_sha256": bind["ordered_image_hashes_sha256"],
             "scorer_sha256": bind["scorer_sha256"], "gallery_sha256": bind["gallery_sha256"]}
    return values, proof


def summarize(values, fields, selected=None):
    result = {}
    for k in fields:
        v = values[k] if selected is None else values[k][selected]
        result[k] = {"mean": float(np.mean(v))}
        if k in ("dino_top1", "dino_top5", "dino_within10", "dino_within10_top5"):
            result[k].update(credit=float(v.sum()), denominator=int(v.size))
    return result


def by_budget(budget, cases, sources, labels):
    panel = budget.split("-")[0]
    n = agg.PANELS[panel][0]
    fields = (*GLOBAL, *(("dino_within10",) if panel == "fresh" else ()))
    arms, comparisons = {}, []
    for selector in PLAN[budget]:
        people = [p for p in agg.SUBJECTS if p in cases[selector]]
        per = {p: summarize(cases[selector][p], fields) for p in people}
        pooled = {k: np.stack([cases[selector][p][k] for p in people]) for k in fields} if people else {}
        arms[selector] = {"participants": people, "missing": [p for p in agg.SUBJECTS if p not in people],
                          "scored_queries": n * len(people), "per_participant": per, "sources": sources[selector],
                          "metrics": summarize(pooled, fields) if people else {}}
    for a, b in PAIRS[budget]:
        people = [p for p in agg.SUBJECTS if p in cases[a] and p in cases[b]]
        case = {"arm_a": a, "arm_b": b, "participants": people, "scored_query_pairs": n * len(people), "metrics": {}, "per_participant": {}}
        if people:
            mats = [np.stack([np.column_stack([cases[arm][p][k] for k in fields]) for p in people]) for arm in (a, b)]
            case["metrics"], case["per_participant"] = agg.describe(*mats, people, labels, panel, fields, N_DRAWS, SEED)
            for k in ("dino_top1", "dino_top5", "dino_within10"):
                if k in fields:
                    left, right = (v[:, :, fields.index(k)] for v in mats)
                    case["metrics"][k].update(credit_a=float(left.sum()), credit_b=float(right.sum()), credit_denominator=int(left.size))
        comparisons.append(case)
    return {"gallery": n, "panel": panel, "arms": arms, "comparisons": comparisons}


def published_examples(official, registration):
    raw = FIGURES.read_bytes()
    agg.require(agg.sha(raw) == FIG_SHA, "Public figure-score CSV differs from its fixed, source-verified version")
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    agg.require(len(records) == 114, "Expected 57 displayed reconstructions and 57 matching ATM queries")
    selections = [
        ("Jo9", "Jo et al.", {"Figure 4"}, 9), ("D2_48", "D2-FOSA", {"Best", "Median", "Worst"}, 48),
        ("D2_best", "D2-FOSA", {"Best"}, 16), ("D2_median", "D2-FOSA", {"Median"}, 16), ("D2_worst", "D2-FOSA", {"Worst"}, 16),
    ]
    result = {}
    for tag, paper, groups, n in selections:
        author = [r for r in records if r["paper"] == paper and r["selection_group"] in groups and r["method"] == paper]
        atm = [r for r in records if r["paper"] == paper and r["selection_group"] in groups and r["method"] == "ATM"]
        ids = np.array([int(r["eeg_row"]) for r in author])
        labels = [r["concept"] for r in author]
        agg.require(len(set(ids)) == len(ids) == n and [(r["eeg_row"], r["concept"]) for r in author] ==
                    [(r["eeg_row"], r["concept"]) for r in atm], "Public study and ATM rows differ")
        if registration["truth"] is not None:
            agg.require(np.array_equal(registration["truth"][ids], labels), "Published stimulus identity differs from frozen gallery")
        values = {
            method: {k: np.asarray([float(r[k]) for r in rows]) for k in FIG_METRICS}
            for method, rows in [(paper, author), (ATM, atm)]
        }
        rec = {"paper": paper, "n": n, "author_subject": "sub-08" if paper == "Jo et al." else "unspecified",
               "eeg_rows": ids.tolist(), "concepts": labels, "metrics": {}, "local_comparison_subject": "sub-08"}
        for method, participant in official.items():
            if "sub-08" not in participant:
                continue
            new = {k: participant["sub-08"][k][ids] for k in FIG_METRICS}
            if method == ATM:
                rec["atm_export_vs_public_cpu"] = {
                    "max_abs_difference": {k: float(np.max(np.abs(new[k] - values[ATM][k]))) for k in FIG_METRICS},
                    "top1_decisions_changed": int(np.sum(new["dino_top1"] != values[ATM]["dino_top1"])),
                    "top5_decisions_changed": int(np.sum(new["dino_top5"] != values[ATM]["dino_top5"])),
                }
            else:
                values[method] = new
        rec["metrics"] = {m: summarize(v, FIG_METRICS) for m, v in values.items()}
        if paper == "Jo et al.":
            rec["paired_Jo_minus_comparator_on_nine"] = {
                m: {k: {"mean_difference": float(np.mean(values[paper][k] - other[k])),
                        "Jo_higher": int(np.sum(values[paper][k] > other[k])), "equal": int(np.sum(values[paper][k] == other[k]))}
                    for k in FIG_METRICS} for m, other in values.items() if m != paper
            }
        result[tag] = rec
    full = {m: summarize(v["sub-08"], FIG_METRICS) for m, v in official.items() if "sub-08" in v}
    return {"public_csv": FIGURES.name, "csv_sha256": FIG_SHA, "selections": result, "full_200_sub08": full,
            "Jo_source": "https://arxiv.org/html/2611.19232v1#S4.F4",
            "D2_source": "https://openaccess.thecvf.com/content/CVPR2026/supplemental/Yu_D2-FOSA_Dual-Diffusion_Guided_CVPR_2026_supplemental.pdf#page=15"}


def coverage_map(coverage, section):
    rows = coverage.get(section, [])
    if isinstance(rows, dict):
        rows = list(rows.values())
    by_folder = {r["folder"]: r for r in rows}
    agg.require(len(by_folder) == len(rows), f"Duplicate {section} folder in score export")
    return by_folder


def build(root, coverage):
    """Return JSON-serializable results, including incomplete exports; score only listed cases."""
    root = Path(root)
    if isinstance(coverage, (bytes, str)):
        coverage = json.loads(coverage)
    dino, rgb = coverage_map(coverage, "dino"), coverage_map(coverage, "rows")
    regs = {p: agg.registration(root, p) for p in GALLERY}
    all_cases, output = {}, {}
    for budget, selectors in PLAN.items():
        panel = budget.split("-")[0]
        reg = regs[panel]
        cases, sources = {s: {} for s in selectors}, {s: {} for s in selectors}
        for selector in selectors:
            for person in agg.SUBJECTS:
                rel = f"external/atm-sub{person[-2:]}" if selector == ATM else f"images/{selector.replace(':', '/')}/{budget}/{person}"
                if rel not in dino:
                    continue
                agg.require(reg["truth"] is not None and reg["rows"] is not None, f"Missing official photo registration for {panel}")
                try:
                    values, proof = read_case(root, dino[rel], rgb.get(rel), reg, selector, budget, person)
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    raise ValueError(f"{rel}: {exc}") from exc
                cases[selector][person], sources[selector][person] = values, proof
        all_cases[budget] = cases
        labels = reg["truth"][np.sort(reg["rows"])] if reg["rows"] is not None and reg["truth"] is not None else None
        output[budget] = by_budget(budget, cases, sources, labels)
    expected = sum(len(arms) for arms in PLAN.values()) * len(agg.SUBJECTS)
    count = sum(len(arm) for arms in all_cases.values() for arm in arms.values())
    # Optional future sub-08 R80 model results can join the same public figure rows.
    # Their presence does not change the fixed 14 x 10 full-cohort analysis.
    for_figures = dict(all_cases["official-r80"])
    additional = {}
    for rel, entry in sorted(dino.items()):
        path = rel.split("/")
        if len(path) != 5 or path[0] != "images" or path[3:] != ["official-r80", "sub-08"]:
            continue
        selector = ":".join(path[1:3])
        if selector in for_figures:
            continue
        values, proof = read_case(root, entry, rgb.get(rel), regs["official"], selector, "official-r80", "sub-08")
        for_figures[selector], additional[selector] = {"sub-08": values}, proof
    figures = published_examples(for_figures, regs["official"])
    figures["additional_sub08_R80_sources"] = additional
    return {"schema": 1, "planned_cases": expected, "scored_cases": count, "complete": count == expected,
            "scored_queries": sum(a["scored_queries"] for m in output.values() for a in m["arms"].values()),
            "model_revision": REV, "model_files_sha256": MODEL, "scorer_sha256": SCORER_SHA,
            "bootstrap": {"resamples": N_DRAWS, "seed": SEED, "percentile_interval": .95, "fixed_candidate_gallery": True,
                          "official_stimulus_unit": "photograph", "fresh_stimulus_unit": "whole concept with ten photographs"},
            "matrices": output, "public_figures": figures}


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(cells) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def markdown(data):
    """Return one Chinese Markdown section ready to include in the main paper report."""
    pieces = ["## DINOv2 独立图像检索",
              f"冻结的 DINOv2-S/14 已完成 {data['scored_cases']}/{data['planned_cases']} 个被试、方法与预算组合，"
              f"共 {data['scored_queries']:,} 个重建。每个重建先与原图统一缩放至 256 × 256，再经同一预处理取中央 224 × 224。"
              "检索用最后一层 CLS 向量的原始余弦，同类十图也使用这一数值。官方图库为 200 张，新概念图库为 500 张。"
              "两选一分数是正确照片胜过其余 199 或 499 张照片的比例，平局计半分。"]
    absolute, contrasts = [], []
    for budget, matrix in data["matrices"].items():
        for method, a in matrix["arms"].items():
            m = a["metrics"]
            rates = [f"{100*m[k]['mean']:.2f}%" if k in m else "—" for k in ("dino_top1", "dino_top5", "dino_within10")]
            absolute.append([BUDGET[budget], ARM[method], f"{len(a['participants'])}/10", *rates,
                             f"{m['dino_correct_cosine']['mean']:.4f}" if m else "—", f"{m['dino_2way']['mean']:.4f}" if m else "—"])
        for c in matrix["comparisons"]:
            if not c["participants"]:
                continue
            cells = []
            for k in ("dino_top1", "dino_top5", "dino_within10", "dino_correct_cosine"):
                m = c["metrics"].get(k)
                if m is None:
                    cells.append("—")
                    continue
                ci = m["crossed_ci95"]
                scale, digits = (1, 4) if k == "dino_correct_cosine" else (100, 2)
                value = f"{scale*m['difference_a_minus_b']:+.{digits}f}"
                cells.append(value if ci is None else f"{value} [{scale*ci[0]:+.{digits}f}, {scale*ci[1]:+.{digits}f}]")
            contrasts.append([BUDGET[budget], f"{ARM[c['arm_a']]} 减 {ARM[c['arm_b']]}", str(len(c["participants"])), *cells])
    pieces.append(table(["完整图库", "方法", "实到被试", "Top-1", "Top-5", "同类十图 Top-1", "真配余弦", "两选一"], absolute))
    if contrasts:
        pieces += ["区间来自同一被试和同一刺激的逐项差值。分别重抽被试与刺激 10,000 次，取 95% 百分位区间。"
                   "官方面板以照片为刺激单位，新概念面板一次抽取整个概念及其十张照片，所有方法共享抽样索引和固定图库。"
                   "前三个检索差以百分点计，真配余弦差以原单位计。仅一个共同被试时只列点估计。"
                   "官方 R80 的 native-paper 比较使用每人全部 1,654 个训练概念，D2 使用受限的 1,323 个训练概念。",
                   table(["面板", "配对方法", "共同被试", "Top-1 差 [95% CI]", "Top-5 差 [95% CI]", "同类十图差 [95% CI]", "余弦差 [95% CI]"], contrasts)]
    fig = data["public_figures"]
    pieces += ["### 作者展示图与同刺激 R80 分数",
               "Jo 图 4 的 Ours 列含被试 08 的 9 张重建。D²-FOSA 补充图 10–12 各取每行左起第一张候选，共 48 张，"
               "其被试身份未披露。同一批照片上的本项目结果均来自被试 08。D²-FOSA 的跨方法数值仅供描述。"
               f"[Jo 图 4]({fig['Jo_source']}) · [D²-FOSA 补充图]({fig['D2_source']}) · "
               "[公开图版逐刺激分数](../research/eeg_stack_20261112/results/paper_benchmark_figure_scores.csv)"]
    display = []

    def append_sample(label, method, m):
        hits = []
        for k in ("dino_top1", "dino_top5"):
            hits.append(f"{m[k]['credit']:g}/{m[k]['denominator']}")
        display.append([label, ARM.get(method, method), *(f"{m[k]['mean']:.4f}" for k in FIG_METRICS[:3]), *hits])

    for key, title in [("Jo9", "Jo 展示 9 张"), ("D2_48", "D²-FOSA 展示 48 张")]:
        for method, values in fig["selections"][key]["metrics"].items():
            append_sample(title, method, values)
    for key, title in [("D2_best", "Best 16 张"), ("D2_median", "Median 16 张"), ("D2_worst", "Worst 16 张")]:
        append_sample(title, "D2-FOSA", fig["selections"][key]["metrics"]["D2-FOSA"])
    for method, values in fig["full_200_sub08"].items():
        append_sample("被试 08 完整 200 张", method, values)
    pieces.append(table(["样本范围", "重建来源", "PixCorr 256", "SSIM 256", "真配余弦", "200 图 Top-1", "200 图 Top-5"], display))
    if ATM in fig["full_200_sub08"]:
        checks = []
        for key, name in [("Jo9", "Jo 九图"), ("D2_48", "D²-FOSA 四十八图")]:
            check = fig["selections"][key]["atm_export_vs_public_cpu"]
            checks.append(f"{name}的真配余弦最大绝对差为 {check['max_abs_difference']['dino_correct_cosine']:.2g}，"
                          f"Top-1、Top-5 命中状态变化分别为 {check['top1_decisions_changed']}、{check['top5_decisions_changed']} 张")
        pieces.append("将被试 08 的 ATM 全库导出与已发布的 CPU 图版对照逐行核验。" + "。".join(checks) + "。")
    return "\n\n".join(pieces)
