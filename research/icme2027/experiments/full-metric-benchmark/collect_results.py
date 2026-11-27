"""Rebuild the complete common-renderer tables from bound image scores."""
import argparse
from contextlib import nullcontext
from hashlib import sha256
from importlib import util
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np


BUDGETS = ("official-r4", "official-r80")
METHODS = {
    "brain-hive": "Brain-HiVE",
    "atm": "ATM",
    "nice": "NICE",
    "eegnet": "EEGNet",
    "ridge": "Ridge",
    "m1a-d2": "本文方法",
}
METRICS = (
    ("dino_top1", "DINOv2 原照 Top-1 ↑", True),
    ("dino_top5", "DINOv2 原照 Top-5 ↑", True),
    ("dino_2way", "DINOv2 两图辨别 ↑", False),
    ("dino_rank", "DINOv2 平均排名 ↓", False),
    ("dino_correct_cosine", "DINOv2 正配余弦 ↑", False),
    ("pixcorr256", "PixCorr 256 ↑", False),
    ("ssim256", "SSIM 256 ↑", False),
)
SIGLIP_NAME = ("siglip2_name1854", "SigLIP2 主体 ↑", True)
SIGLIP_MODEL = "google/siglip2-base-patch16-224@75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP_WEIGHTS = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
SIGLIP_REGISTRY = {
    "concepts": "8450a4e843040aad18972a5ced111e7a0da4c482e94081e336452f3aa3876514",
    "photographs": "e534427d01d13f4134cab32bf2364c0d1acf1845a46f12cec51281432eb4bd97",
    "photo_zip": "57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a",
}
MAIN = (
    ("generated_name1854", "CLIP 主体 ↑", True),
    ("clip_image_top1", "CLIP 原照 ↑", True),
    METRICS[0],
    ("pixcorr", "PixCorr ↑", False),
    ("ssim", "SSIM ↑", False),
    ("alex2_2way", "AlexNet(2) ↑", False),
    ("alex5_2way", "AlexNet(5) ↑", False),
    ("inception_2way", "Inception ↑", False),
    ("clip_2way", "CLIP 两图 ↑", False),
    ("effnet_distance", "EffNet ↓", False),
    ("swav_distance", "SwAV ↓", False),
)


def blob(archive, member):
    item = archive.getmember(member)
    if not item.isfile():
        raise ValueError(f"Expected a regular archive member: {member}")
    return archive.extractfile(item).read()


def read_result(archive, prefix, raw_score_sha):
    receipt = json.loads(blob(archive, prefix + "/dino.json"))
    payload = blob(archive, prefix + "/dino.npz")
    if (receipt["n"] != 200 or receipt["panel"] != "official"
            or sha256(payload).hexdigest() != receipt["dino_npz_sha256"]):
        raise ValueError(f"Invalid score binding at {prefix}")
    binding = receipt["binding"]
    if binding["input_sha256"]["scores.json"] != raw_score_sha:
        raise ValueError(f"Independent metric covers different RGB queries at {prefix}")
    if set(receipt["per_row_metrics"]) != {m[0] for m in METRICS}:
        raise ValueError(f"Unexpected metric list at {prefix}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        if not np.array_equal(data["row"], np.arange(200)):
            raise ValueError(f"Wrong official row order at {prefix}")
        score = {}
        for key in receipt["per_row_metrics"]:
            col = data[key].astype(np.float64)
            if col.shape != (200,) or not np.isfinite(col).all() or not np.isclose(
                col.mean(), receipt["mean"][key], rtol=1e-6, atol=1e-8,
            ):
                raise ValueError(f"Invalid {key} scores at {prefix}")
            score[key] = col
        png_order = "\n".join(data["query_sha256"]).encode()
        if sha256(png_order).hexdigest() != binding["ordered_image_hashes_sha256"]:
            raise ValueError(f"Independent scores do not bind their RGB files at {prefix}")
    keys = ("gallery_sha256", "model_revision", "model_files_sha256", "scorer_sha256")
    common = {key: binding[key] for key in keys}
    return score, common, receipt["dino_npz_sha256"], binding["input_sha256"]["render.json"]


def read_credits(entry, context):
    values = np.asarray(entry["per_row"], dtype=np.float64)
    if (values.shape != (200,) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()
            or not np.isclose(values.sum(), entry["credits"], rtol=0, atol=1e-9)
            or len(entry["top"]) != 200):
        raise ValueError(f"Invalid naming scores at {context}")
    return values


def read_naming_packet(archive, budget, subject, code_sha):
    prefix = f"siglip-results/{budget}-sub-{subject:02d}.json"
    payload = blob(archive, prefix)
    result = json.loads(payload)
    if (result["model"] != SIGLIP_MODEL
            or result["model_files_sha256"]["model.safetensors"] != SIGLIP_WEIGHTS
            or result["sources"] != SIGLIP_REGISTRY or result["code_sha256"] != code_sha
            or result["people"] != [subject] or result["n_per_arm"] != 200
            or set(result["subjects"]) != {str(subject)}):
        raise ValueError(f"Independent naming uses unexpected inputs at {prefix}")
    rows = result["subjects"][str(subject)]
    if set(rows) != set(METHODS) or set(result["totals"]) != set(METHODS):
        raise ValueError(f"The naming packet does not cover all methods at {prefix}")
    for method, row in rows.items():
        row["values"] = read_credits(row, f"{prefix}/{method}")
        if row["credits"] != result["totals"][method]:
            raise ValueError(f"Inconsistent naming totals at {prefix}/{method}")
    ref = result["reference"]
    read_credits(ref, prefix + "/reference")
    provenance = {key: result[key] for key in (
        "model", "model_files_sha256", "sources", "transformers_version", "code_sha256",
    )}
    return rows, ref, provenance, sha256(payload).hexdigest()


def pairwise_intervals(aggregate, arrays):
    pair = {}
    for method, values in arrays.items():
        if method == "m1a-d2":
            continue
        delta = arrays["m1a-d2"] - values
        ci = aggregate.bootstrap_ci(delta[:, :, None], 10000, 20261127).ravel().tolist()
        pair[method] = {"mean_delta": float(delta.mean()), "crossed_ci95": ci}
    return pair


def format_value(value, percent):
    return f"{100 * value:.2f}%" if percent else f"{value:.4f}"


def write_table(lines, metrics, methods):
    lines.append("| 方法 | " + " | ".join(label for _, label, _ in metrics) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in metrics) + " |")
    for method, row in methods.items():
        cells = [format_value(row[key], pct) for key, _, pct in metrics]
        lines.append("| " + METHODS[method] + " | " + " | ".join(cells) + " |")
    lines.append("")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-source", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--siglip-archive", type=Path)
    args = parser.parse_args()
    spec = util.spec_from_file_location("registered_paper_aggregate", args.aggregate_source)
    aggregate = util.module_from_spec(spec)
    sys.modules[spec.name] = aggregate
    spec.loader.exec_module(aggregate)

    root = Path(__file__).parent
    canonical = json.loads(args.canonical.read_text())
    report = {"n_per_row": 2000, "n_total": 24000, "job": 5395668, "budgets": {}}
    lines = [
        "# 统一配置的十人全量重建主表", "",
        "四次和八十次实测 EEG 各覆盖十名被试、每人 200 张正式测试照片。每个预算下六种方法采用相同生成器、布局和种子。"
        "DINOv2 重新评分已登记的 24,000 张原有生成图，并与此前逐人分数的散列逐一核对。",
        "",
    ]
    common = None
    siglip_common, siglip_reference = None, None
    main_metrics = MAIN
    names_context = nullcontext()
    if args.siglip_archive is not None:
        names_context = tarfile.open(args.siglip_archive)
        code_sha = sha256((root / "code/paper_independent_naming.py").read_bytes()).hexdigest()
        main_metrics = (MAIN[0], SIGLIP_NAME, *MAIN[1:])
        lines.extend([
            "SigLIP2 用同一批生成图在完整 1,854 个物体名中独立命名。图像与文本模型、词表及原照均固定。"
            "两种命名器的并列第一均分命中积分。", "",
        ])
    with tarfile.open(root / "common-dino.tar") as archive, names_context as siglip_archive:
        for budget in BUDGETS:
            means_by_method, top1_by_method, hashes = {}, {}, {}
            siglip_rows, siglip_by_method, siglip_hashes = {}, {}, {}
            if siglip_archive is not None:
                for subject in range(1, 11):
                    rows, ref, provenance, siglip_sha = read_naming_packet(
                        siglip_archive, budget, subject, code_sha,
                    )
                    if siglip_common is None:
                        siglip_common, siglip_reference = provenance, ref
                    if siglip_common != provenance or siglip_reference != ref:
                        raise ValueError("The naming models or reference scores differ across packets")
                    siglip_rows[subject] = rows
                    siglip_hashes[f"sub-{subject:02d}"] = siglip_sha
            for method in METHODS:
                meta = canonical[budget][method]
                if meta["n"] != 2000:
                    raise ValueError("The original table is not full coverage")
                people, names = [], []
                if sorted(src["subject"] for src in meta["subject_receipts"]) != list(range(1, 11)):
                    raise ValueError(f"Wrong participants for {budget}/{method}")
                for src in sorted(meta["subject_receipts"], key=lambda x: x["subject"]):
                    subject = f"sub-{src['subject']:02d}"
                    score, provenance, checksum, render_sha = read_result(
                        archive, f"results/{budget}/{method}/{subject}", src["score_sha256"],
                    )
                    if common is None:
                        common = provenance
                    if provenance != common:
                        raise ValueError("Different evaluation gallery or model across methods")
                    people.append(score)
                    hashes[f"{method}/{subject}"] = checksum
                    if siglip_archive is not None:
                        named = siglip_rows[src["subject"]][method]
                        if named["render_sha256"] != render_sha:
                            raise ValueError(f"Different RGB images for naming at {budget}/{method}/{subject}")
                        names.append(named["values"])
                if len(people) != 10:
                    raise ValueError("Missing participants")
                means = {name: float(np.mean([row[name] for row in people])) for name in people[0]}
                means_by_method[method] = dict(meta["mean"], **means)
                top1_by_method[method] = np.stack([row["dino_top1"] for row in people])
                if siglip_archive is not None:
                    siglip_by_method[method] = np.stack(names)
                    means_by_method[method][SIGLIP_NAME[0]] = float(siglip_by_method[method].mean())

            pair = pairwise_intervals(aggregate, top1_by_method)
            budget_report = {"means": means_by_method, "paired_dino_top1": pair, "npz_sha256": hashes}
            if siglip_archive is not None:
                budget_report["siglip_sha256"] = siglip_hashes
                budget_report["paired_siglip2_naming"] = pairwise_intervals(aggregate, siglip_by_method)
            report["budgets"][budget] = budget_report
            lines += [f"## {'R4' if budget.endswith('r4') else 'R80'} 全指标", ""]
            write_table(lines, main_metrics, means_by_method)
            lines += ["### DINOv2 附加指标", ""]
            write_table(lines, METRICS, means_by_method)
            best = max(pair, key=lambda m: means_by_method[m]["dino_top1"])
            lower, upper = pair[best]["crossed_ci95"]
            lines += [
                f"本文方法对该预算下 DINOv2 原照 Top-1 最高的基线 {METHODS[best]} 的差为 "
                f"{100*pair[best]['mean_delta']:+.2f} 个百分点，95% 配对区间 "
                f"[{100*lower:+.2f}, {100*upper:+.2f}]。", "",
            ]
            if siglip_archive is not None:
                siglip_best = max(pair, key=lambda m: means_by_method[m][SIGLIP_NAME[0]])
                stat = budget_report["paired_siglip2_naming"][siglip_best]
                lower, upper = stat["crossed_ci95"]
                lines += [
                    f"在 SigLIP2 主体命名上，本文方法相对该列最高基线 {METHODS[siglip_best]} 的差为 "
                    f"{100*stat['mean_delta']:+.2f} 个百分点，95% 配对区间 "
                    f"[{100*lower:+.2f}, {100*upper:+.2f}]。", "",
                ]
                print(budget, SIGLIP_NAME[0], {method: row[SIGLIP_NAME[0]]
                      for method, row in means_by_method.items()}, stat)
            print(budget, means_by_method["m1a-d2"]["dino_top1"], best,
                  means_by_method[best]["dino_top1"], pair[best]["crossed_ci95"])

    report["scoring"] = common
    report_name = "full-dino-results.json"
    if siglip_reference is not None:
        credits = siglip_reference["credits"]
        lines += [f"以同一套 SigLIP2 词表复评 200 张真实原照，得到 {credits:g}/200 命名积分。", ""]
        report["siglip2"] = dict(siglip_common, reference_credits=credits, reference_n=200)
        report_name = "full-benchmark-results.json"
    (root / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
