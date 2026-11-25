"""Follow each official EEG query from continuous feature retrieval to the saved image."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

import paper_aggregate as agg


def analyze(root, rgb, features):
    result = {}
    for budget in ("official-r4", "official-r80"):
        sides = {}
        for selector in ("personal:native-paper","personal:d2","shared:d2","native-full:native-paper"):
            image = rgb[budget]["arms"].get(selector)
            brain = features[budget]["arms"].get(selector)
            if image is None or brain is None:
                continue
            people = [p for p in agg.SUBJECTS if p in image["sources"] and p in brain["sources"]]
            parts = []
            mistakes = Counter()
            vocabulary = Counter()
            for person in people:
                entry = image["sources"][person]
                image_folder = root/entry["folder"]
                track, mode = selector.split(":")
                array = root/f"features/{track}/{mode}/{budget}/{person}/features.npz"
                eeg_meta = json.loads(array.with_suffix(".json").read_bytes())
                if (agg.sha(array.read_bytes()) != eeg_meta["observations_sha256"] or
                        eeg_meta["checkpoint_sha256"] != entry["checkpoint_sha256"] or
                        agg.sha((image_folder/"observations.npz").read_bytes()) != entry["observations_sha256"]):
                    raise ValueError(f"Feature and final image have inconsistent provenance for {selector} {person}")
                with np.load(array, allow_pickle=False) as f, np.load(image_folder/"observations.npz", allow_pickle=False) as o:
                    order_a, order_b = np.argsort(f["row"]), np.argsort(o["row"])
                    if not np.array_equal(f["row"][order_a],o["row"][order_b]):
                        raise ValueError("Feature and generated image have different photograph IDs")
                    top = f["feature_top1"][order_a]
                    if not np.all((top==0)|(top==1)):
                        raise ValueError("Conditional diagnostic requires uniquely ranked EEG features")
                    name = o["generated_name1854"][order_b]
                    photo = o["clip_image_top1"][order_b]
                    original = o["original_name1854"][order_b]
                    parts.append(np.column_stack((top, name, photo, original)))
                    words = o["generated_top_word"][order_b].tolist()
                    vocabulary.update(words)
                    mistakes.update(w for w,n in zip(words,name,strict=True) if n==0)
            if not parts:
                continue
            samples = np.concatenate(parts)
            def summary(mask):
                a = samples[mask]
                return dict(queries=len(a), naming_credit=float(a[:,1].sum()), photo_top1_credit=float(a[:,2].sum()),
                            names_on_recognizable_originals=float(a[a[:,3]==1,1].sum()), recognizable_originals=int(np.sum(a[:,3]==1)))
            sides[selector] = dict(participants=people, correct_continuous_retrieval=summary(samples[:,0]==1),
                    missed_continuous_retrieval=summary(samples[:,0]==0), distinct_generated_words=len(vocabulary),
                    top_five_uncredited_words=mistakes.most_common(5))
        result[budget] = sides
    return result


def markdown(data):
    from build_paper_benchmark_report import BUDGETS, arm_label, credit_text, table
    rows=[]
    for budget, arms in data.items():
        for arm,record in arms.items():
            def value(x,k):
                return f"{credit_text(x[k])}/{x['queries']}" if x["queries"] else "—"
            hit, miss = record["correct_continuous_retrieval"], record["missed_continuous_retrieval"]
            rows.append([BUDGETS[budget],arm_label(arm),len(record["participants"]),value(hit,"naming_credit"),
                         value(miss,"naming_credit"),value(hit,"photo_top1_credit"),value(miss,"photo_top1_credit")])
    if not rows:
        return ""
    return "\n\n".join(["## 同一条 EEG 的信息有没有传到最后一张图",
        "逐行核对 EEG 特征和生成图使用的被试、照片与检查点。先按扩散前能否从 200 个视觉向量中找回同一原图分组，再看生成图的得分。"
        "每格是命中分／该组查询数。连续特征检索第一只表明真实照片在这 200 项中排第一，其绝对语义精度和生成器条件质量仍有差异。",
        table(["预算","模型","人数","特征找对后物体命名","特征未找对时物体命名","特征找对后图片 Top-1","特征未找对时图片 Top-1"],rows)])
