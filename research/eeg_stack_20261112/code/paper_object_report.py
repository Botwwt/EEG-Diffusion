"""Post-primary ablations for the object-first encoder and isolated components of M1a + D2."""

import hashlib
import json

import paper_aggregate as agg


SEED, DRAWS = 20261125, 10000
NATIVE, D2 = "personal:native-paper", "personal:d2"
PHOTO, OBJECT, RELIABLE, WRONG = ["personal:" + v for v in agg.TOKEN_ARMS]
FINAL_AUX, MIDDLE_AUX, RGB_PHOTO, RGB_SIBLING, CENTER, SPREAD, PEER_OWN, PEER_SIBLING = ["personal:" + v for v in agg.AUX_ARMS]
REFERENCE, HALF, MEAN = ["shared:" + v for v in ("d2-reference", "d2-positive-half", "d2-mean-erp")]
METRICS = ("generated_name1854", "clip_image_top1", "clip_within10", "ssim")


def registered(root, index, budget, arms, pairs, subjects):
    result = agg.aggregate(root, "personal", budget, arms, subjects=subjects, pairs=pairs, resamples=DRAWS, seed=SEED)
    for key, rec in result["arms"].items():
        for person in subjects:
            source = rec["sources"].get(person)
            rel = f"images/{key.replace(':','/')}/{budget}/{person}"
            if bool(source) != (rel in index):
                raise ValueError(f"Missing or unlisted supplemental score receipt {rel}")
            if source and (source["scores_sha256"] != index[rel]["scores_sha256"]
                           or source["observations_sha256"] != index[rel]["observations_sha256"]):
                raise ValueError(f"Supplementary source is not bound to the verified snapshot at {rel}")
    return result


def select_candidate(matrix, subjects, candidates=(PHOTO, OBJECT, RELIABLE)):
    """Apply the object-first development rule before considering the eight confirmation people."""
    baseline = matrix["arms"][NATIVE]
    rule = dict(criterion="Increase generated_name1854 hits for each development participant and preserve pooled clip_within10 hits.",
                tiebreak="More naming credit, then within-concept photo credit, then fixed target.")
    if any(set(matrix["arms"][a]["participants"]) != set(subjects) for a in (NATIVE, *candidates)):
        return dict(**rule, status="incomplete", selected=None)
    stats, eligible = {}, []
    for priority, arm in enumerate(candidates):
        item = matrix["arms"][arm]
        gains = {p: item["per_participant"][p]["generated_name1854"]["mean"]-baseline["per_participant"][p]["generated_name1854"]["mean"]
                 for p in subjects}
        within = item["metrics"]["clip_within10"]["credit"] - baseline["metrics"]["clip_within10"]["credit"]
        okay = min(gains.values()) > 0 and within >= 0
        stats[arm] = {"per_person_naming_differences": gains, "pooled_within10_credit_difference": within, "qualifies": okay}
        if okay:
            eligible.append((item["metrics"]["generated_name1854"]["credit"], item["metrics"]["clip_within10"]["credit"],
                             -priority, arm))
    selected = max(eligible)[-1] if eligible else None
    return dict(**rule, status="selected" if selected else "none_qualified", selected=selected, candidates=stats)


def verify_training_pair(root, matrix, control, intervention, fields):
    receipts = {}
    left, right = (matrix["arms"][a] for a in (control, intervention))
    for person in sorted(set(left["participants"]).intersection(right["participants"])):
        fits = []
        hashes = []
        for arm in (control, intervention):
            method = arm.split(":")[1]
            source = matrix["arms"][arm]["sources"][person]
            render = json.loads((root / source["folder"] / "render.json").read_bytes())
            ref = render["condition_sources"]["training_receipt"]
            expected = f"fits/personal-{person[-2:]}/enc/object_tokens/{person}/{method}.json"
            if ref["path"] != expected:
                raise ValueError(f"Unexpected training receipt path for {person} {method}")
            raw = (root/expected).read_bytes()
            fit = json.loads(raw)
            if (agg.sha(raw) != ref["sha256"] or fit["mode"] != method or fit["subject"] != int(person[-2:])
                    or fit["checkpoint_sha256"] != source["checkpoint_sha256"]):
                raise ValueError(f"Training evidence differs from the rendered weights for {person} {method}")
            fits.append(fit)
            hashes.append(ref["sha256"])
        for key in ("sources", "seed", "steps", "photos", "concepts", *fields):
            if fits[0][key] != fits[1][key]:
                raise ValueError(f"Unmatched {key} in the training comparison for {person} {intervention}")
        receipts[person] = {"control_training_sha256": hashes[0], "intervention_training_sha256": hashes[1],
                            "matched_training_steps": fits[0]["steps"], "allowed_photos": fits[0]["photos"]}
    return receipts


def audits(root, index):
    history, replays = [], []
    for path in sorted(root.glob("audit/archived/sub-*.json")):
        rec = json.loads(path.read_bytes())
        for row in rec["rows"]:
            folder = root / row["folder"]
            if row["folder"] not in index:
                # The independent filesystem audit can finish while a newer RGB score is still in flight.
                continue
            raw = (folder/"render.json").read_bytes()
            meta = json.loads(raw)
            ordered = hashlib.sha256("\n".join(meta["pixels_sha256"]).encode()).hexdigest()
            if (hashlib.sha256(raw).hexdigest()!=row["render_sha256"] or ordered!=row["ordered_png_list_sha256"]
                    or meta["n"]!=row["png_count"] or meta["checkpoint_sha256"]!=row["checkpoint_sha256"]):
                raise ValueError(f"Archived input/output verification disagrees with {row['folder']}")
            history.append(row)
    if len({r["folder"] for r in history}) != len(history):
        raise ValueError("Repeated audit of a single archived condition")
    for path in sorted(root.glob("audit/replay/*/*/sub-*.json")):
        record = json.loads(path.read_bytes())
        rel = "images/{training}/{arm}/{budget}/sub-{subject:02d}".format(**record)
        if rel not in index:
            continue
        original = (root/rel/"render.json").read_bytes()
        meta = json.loads(original)
        exact = sum(a == b for a,b in zip(meta["pixels_sha256"],record["replay_pixel_sha256"], strict=True))
        if (len(meta["row"])!=record["photos"] or hashlib.sha256(original).hexdigest()!=record["original_render_sha256"]
                or exact!=record["exact_pngs"] or record["checkpoint_sha256"] != meta["checkpoint_sha256"]):
            raise ValueError(f"Independent reconstruction disagrees with archived receipt at {rel}")
        replays.append({k:v for k,v in record.items() if k!="replay_pixel_sha256"})
    return dict(archived_internal_cases=len(history), archived_pngs=sum(r["png_count"] for r in history),
                later_modified_sources=[r["folder"] for r in history if r["artifacts_modified_after_first_png"]],
                replay_runs=replays, replayed_pngs=sum(r["photos"] for r in replays),
                replayed_exact_pngs=sum(r["exact_pngs"] for r in replays))


def build(root, raw):
    cov = json.loads(raw)
    by_folder = {r["folder"]:r for r in cov["rows"]}
    dev = ["sub-01", "sub-03"]
    object_arms = [PHOTO, OBJECT, RELIABLE, WRONG, NATIVE, D2]
    object_pairs = [(a,b) for a,b in ((RELIABLE, NATIVE), (OBJECT, NATIVE), (PHOTO, NATIVE),
                    (RELIABLE, PHOTO), (RELIABLE, OBJECT), (RELIABLE, WRONG), (RELIABLE, D2), (OBJECT, PHOTO))]
    results = {"development": registered(root, by_folder, "dev-r4", object_arms, object_pairs, dev),
               "components": {}, "confirmation": {}, "included_in_primary_coverage": False}
    results["selection"] = select_candidate(results["development"], dev)
    results["input_and_pixel_audit"] = audits(root, by_folder)
    results["historical_followups"] = {}
    for title, (control, intervention) in {
        "vision_teacher": (FINAL_AUX, MIDDLE_AUX),
        "rendered_object": (RGB_PHOTO, RGB_SIBLING),
        "condition_spread": (CENTER, SPREAD),
        "peer_teacher": (PEER_SIBLING, PEER_OWN),
    }.items():
        arms = [control, intervention, PHOTO, NATIVE]
        pairs = [(intervention, control), (control, PHOTO), (intervention, PHOTO), (control, NATIVE), (intervention, NATIVE)]
        matrix = registered(root, by_folder, "dev-r4", arms, pairs, dev)
        selection = select_candidate(matrix, dev, (control, intervention))
        check = {"vision_teacher": ("weight", "temperature", "auxiliary_parameters", "training_jpeg_sha256"),
                 "rendered_object": ("request", "episodes", "target_vectors_sha256", "trainable_parameters"),
                 "condition_spread": ("calibration_sha256",),
                 "peer_teacher": ("peer_cache_sha256",)}[title]
        results["historical_followups"][title] = {"development": matrix, "selection": selection,
            "matched_fits": verify_training_pair(root, matrix, control, intervention, check)}
    for budget in ("official-r4", "fresh-r4"):
        # Original D2 is included to test the numerical reproducibility of the new layout cache.
        arms = [REFERENCE, HALF, MEAN, "shared:d2"]
        group = registered(root, by_folder, budget, arms, [(REFERENCE, HALF), (REFERENCE, MEAN), (REFERENCE, "shared:d2")], agg.SUBJECTS)
        for p in group["comparisons"][:2]:
            for person in p["participants"]:
                l, r = (group["arms"][m]["sources"][person]["folder"] for m in (p["arm_a"], p["arm_b"]))
                left, right = [json.loads((root / folder / "render.json").read_bytes()) for folder in (l,r)]
                if left["layout_source"]["ridge_sha256"] != right["layout_source"]["ridge_sha256"]:
                    raise ValueError(f"Controlled ablation uses different fitted layouts for {person}")
        results["components"][budget] = group
        # Confirm on the participants excluded from the two-person development comparison.
        people = [p for p in agg.SUBJECTS if p not in dev]
        candidates = [a for a in (RELIABLE, OBJECT, PHOTO, FINAL_AUX, MIDDLE_AUX, RGB_PHOTO, RGB_SIBLING, CENTER, SPREAD, PEER_OWN, PEER_SIBLING) if any(
            f"images/{a.replace(':','/')}/{budget}/{p}" in by_folder for p in people)]
        if candidates:
            contrasts = [(a,b) for a in candidates for b in (NATIVE,D2)]
            contrasts.extend((a,b) for a,b in ((MIDDLE_AUX,FINAL_AUX),(RGB_SIBLING,RGB_PHOTO),(SPREAD,CENTER),
                              (PEER_OWN,PEER_SIBLING),(RELIABLE,PHOTO),(OBJECT,PHOTO)) if a in candidates and b in candidates)
            results["confirmation"][budget] = registered(root, by_folder, budget, [*candidates, NATIVE, D2],
                contrasts, people)
    return results


def markdown(data):
    from build_paper_benchmark_report import BUDGETS, abs_metric, arm_label, delta_text, table
    blocks = ["## 物体优先的单编码器与组件对照",
              "单编码器先平均同一人的 EEG，再输出同一个冻结投影器使用的视觉条件，SDXL-Turbo 用空文本生成一张图。"
              "三个训练组从相同的逐人 native 权重出发，使用相同的训练照片、训练类分组、优化步数和投影器。"
              "token-photo 的目标是当前照片的四个图像 token；token-object 使用该物体另外九张训练照片的 token 均值。"
              "token-reliable 从均值出发，只保留训练 EEG 跨物体预测得出的那部分单图偏差。"
              "三个模型推理时参数数目相同，每个查询只运行一次 EEG 编码器。"
              "token-reliable-wrong 保留收图者的种子，将 EEG 条件换成另一物体的 EEG。"]
    selections = [("两位既定开发被试", data["development"], [PHOTO, OBJECT, RELIABLE, WRONG, NATIVE, D2])]
    labels = {"vision_teacher": "训练时使用 H/14 末层或中间层",
              "rendered_object": "直接在生成 RGB 上使用本图或同物体另一张照片监督",
              "condition_spread": "固定个人编码器后，用训练物体校准融合特征的均值与离散程度",
              "peer_teacher": "训练时利用其余九人同图或同物体另一张照片的脑电，测试单人"}
    for key, study in data["historical_followups"].items():
        matrix = study["development"]
        arms = list(matrix["arms"])
        if any(matrix["arms"][a]["participants"] for a in arms if a not in (NATIVE, PHOTO)):
            selections.append((labels[key], matrix, arms))
    if data["confirmation"]:
        selections.extend((BUDGETS[b]+"，八位确认被试", m, list(m["arms"])) for b,m in data["confirmation"].items())
    for label, matrix, arms in selections:
        rows = [[arm_label(a), str(len(matrix["arms"][a]["participants"])), str(matrix["arms"][a]["scored_images"]),
                 *[abs_metric(matrix["arms"][a]["metrics"], k) for k in METRICS]] for a in arms if matrix["arms"][a]["participants"]]
        if rows:
            blocks.extend([f"### {label}", table(["配置", "人数", "图数", "物体命名", "图片 Top-1", "同类十图", "SSIM"], rows)])
    decision = data["selection"]
    if decision["status"] != "incomplete":
        blocks.append("选型规则要求两人的物体命中各自增加，同类十图的合计命中维持或提高。"
                      + ("达到条件并入选八人确认的配置为 " + arm_label(decision["selected"]) + "。"
                         if decision["selected"] else "三支单编码器均未达到这条开发集标准。"))
    for key, study in data["historical_followups"].items():
        decision = study["selection"]
        if decision["status"] != "incomplete":
            verdict = ("入选八人确认的是 " + arm_label(decision["selected"]) if decision["selected"]
                       else "两个配置均未达到预设的命名与同类照片标准")
            blocks.append(labels[key] + "。两人开发结果中，" + verdict + "。")
            pair = study["development"]["comparisons"][0]
            blocks.append(arm_label(pair["arm_a"]) + " 相对同预算的 " + arm_label(pair["arm_b"])
                          + "，配对命名差 " + delta_text(pair, "generated_name1854")
                          + " 个百分点，同类十图差 " + delta_text(pair, "clip_within10") + " 个百分点。")
    blocks.extend(["### 单独移除配对损失与重复 EEG 读出",
                   "历史 D1 用 12 倍正例相似度损失近似匹配初始梯度量级，D2 的正例权重为 0.5。"
                   "追加的 d2-positive-half 保持 D2 的 0.5 正例权重、基础权重、信赖项、学习率、随机数、优化步数和布局。"
                   "d2-mean-erp 保持 D2 训练权重和布局，将语义路径替换为每四次 EEG 的算术平均后单次编码。"
                   "d2-reference 使用相同版本的渲染器及同一份布局参数。以下差值单位为百分点，SSIM 使用原量纲。"])
    rows = []
    for budget, m in data["components"].items():
        for p in m["comparisons"]:
            if p["participants"]:
                rows.append([BUDGETS[budget], arm_label(p["arm_a"])+" − "+arm_label(p["arm_b"]), str(len(p["participants"])),
                             *[delta_text(p,k) for k in METRICS]])
    if rows:
        blocks.append(table(["图库", "对照", "被试", "Δ命名", "Δ图片 Top-1", "Δ同类十图", "ΔSSIM"], rows))
    audit = data["input_and_pixel_audit"]
    if audit["archived_internal_cases"] or audit["replay_runs"]:
        blocks += ["### 实际文件和独立重生成",
                   f"从原渲染目录重新读取了 {audit['archived_internal_cases']} 组、{audit['archived_pngs']:,} 张 PNG，"
                   f"复核了图像字节、原始试次、划分、权重及嵌套骨干的来源。布局文件或权重的修改日期晚于首张出图的组数为 "
                   f"{len(audit['later_modified_sources'])}。这次文件检查在首轮出图之后进行。",
                   f"独立从 EEG 与权重重画的 {audit['replayed_pngs']:,} 张官方 R4 图中，"
                   f"{audit['replayed_exact_pngs']:,} 张与首次评分的 PNG 字节完全一致。逐组原始与回放摘要保存在数值包中。"]
    return "\n\n".join(blocks)
