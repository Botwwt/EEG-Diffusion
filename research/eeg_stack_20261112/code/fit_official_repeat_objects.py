"""Select a personal cross-modal readout on generic objects, then test five measured EEG budgets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_official_visual_anchors import EEG_HASHES, prototype_fits
from fit_support_covariance import eeg_features
from fit_visual_to_neural_objects import load_anchors
from fit_within_photo_embedding import images
from official_object_readout import ASSETS, sha, tensor_storage, unit
from prepare_official_repeat_eeg import BUDGETS, FULL_SHA
from prepare_registered_subjects import ROWS_SHA
from scipy.linalg import eigh, svd

LOCK = "3631b4228d6ca550f0fe7d8e787b9c1e54a71011533cc106c255fb856b6243e6"
S = "sub-04"
RANKS = (32, 64, 128)
FAMILIES = ("text", "external", "joint")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def dump(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def read(p):
    return json.loads(p.read_bytes())


def standard(s):
    return (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True).clip(1e-6)


def ranks(s, y):
    correct = s[np.arange(len(s)), y]
    return (
        1
        + (s > correct[:, None]).sum(1)
        + ((s == correct[:, None]) & (np.arange(s.shape[1]) < y[:, None])).sum(1)
    )


def metrics(r):
    return dict(
        n=len(r),
        top1=int(sum(r == 1)),
        top5=int(sum(r <= 5)),
        top10=int(sum(r <= 10)),
        median_rank=float(np.median(r)),
        mrr=float(np.mean(1 / r)),
    )


def register(a):
    public = Path(__file__).resolve().parents[1] / "results" / "object_fidelity"
    base = read(public / "double_calibration_design.json")
    other = read(public / "cohort_session_matched_design.json")
    old = np.array(base["old_objects"])
    held = np.sort(np.random.default_rng(260542).choice(old, 128, replace=False))
    excluded = sorted(
        set(
            base["fifth_objects"]
            + other["panels"]["stress12"]["selected_concepts"]
            + other["panels"]["new_three4"]["selected_concepts"]
        )
    )
    obj = dict(
        subject="sub-04",
        test="Official 200 unseen categories, one photograph each, full 1854 vocabulary; actual target photos and labels evaluation only.",
        budgets=[4, 8, 16, 32, 80],
        source_pinned_sha256="cba83bc49213f8783d29e1717e63c0b429967601ea417adf6671dbfd2336c03e",
        mean="EEG feature mean of disjoint original four-repeat blocks; same nested slots for all test stimuli, permutation seed 260541. Features of the raw voltage grand mean are a fixed averaging-stage control.",
        historically_excluded=excluded,
        old_classes=old.tolist(),
        extended_classes=sorted(set(base["old_objects"] + base["extra_objects"]) - set(excluded)),
        held_generic_classes=held.tolist(),
        held_generic_seed=260542,
        development="128 original generic classes held intact. Select modality mix and rank by full1854 top1 then mean reciprocal rank, at largest eligible training size only; primary and 512 use identical selection. Never select by official 200. Two scaling budgets describe sample-size sensitivity.",
        fixed_reg="1012 maximum eligible generic classes after keeping the 12 stress test classes out; 512 original as sensitivity; EEG CCA shrinkage 0.5*mean variance and H14 0.25*mean variance. Train image targets per photo. Rank 32,64,128; compare training-only text->H14 PRESS class prototype, independently collision-audited external single-photo H14, and 50/50 standardized-score combination. Plain EEG-to-H14 linear ridge to text-synthesized prototypes baseline.",
        shuffled="Same query budgets mapped by one common nonzero 67-position cyclic shift for every photographic outcome.",
        evaluation="Before scoring write predicted full vocabulary object/distributions for all budgets. Frozen metric top-1, top-5, median rank, comparative rank; secondary official 200-label discrimination and 200 target image gallery only for scoring, no within-class target-photo discrimination possible.",
    )
    path = a.out / "design.json"
    a.out.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == obj, "Refusing to replace a different registration")
    else:
        dump(path, obj)
    require(sha(path) == LOCK, "Official repeat-object protocol differs")
    print("Registered", path, LOCK)


def training(a, locked):
    source = a.source
    old, x = eeg_features(a.expanded, S)
    require(
        np.array_equal(old, locked["old_classes"]), "Changed original physical training classes"
    )
    blocks = dict(zip(old, x, strict=True))
    more = read(a.extra / "features" / (S + ".json"))
    for c in sorted(set(locked["extended_classes"]) - set(old)):
        with np.load(a.extra / "features" / S / f"{c:04d}.npz", allow_pickle=False) as z:
            v = z["features"]
            require(
                int(z["start"]) == 10 * c
                and v.shape == (10, 1003)
                and str(z["raw_sha"]) == more["raw_sha256"][str(c)]
                and hashlib.sha256(v.tobytes()).hexdigest() == more["feature_sha256"][str(c)],
                "Changed training EEG",
            )
            blocks[c] = v
    with np.load(source / "dictionary.npz", allow_pickle=False) as z:
        ids, words, text = (z[k] for k in ("unique_ids", "words", "text_embeddings"))
    require(sha(source / "atm_train_rows.csv") == ROWS_SHA, "Training row manifest changed")
    with (source / "atm_train_rows.csv").open(encoding="utf8", newline="") as f:
        table = list(csv.DictReader(f))
    require([int(r["train_row"]) for r in table] == list(range(16540)), "Unregistered training row")
    label = np.array([r["unique_id"] for r in table]).reshape(1654, 10)
    require(
        np.all(label == label[:, :1]) and len(set(label[:, 0])) == 1654, "Missing training concepts"
    )
    index = {w: i for i, w in enumerate(ids)}
    names = np.array([index[w] for w in label[:, 0]])
    visual = images(source).reshape(1654, 10, 1024)
    ext, provenance = load_anchors(a.external, ids)
    audit = read(a.external.parent / "collision_audit.json")
    require(
        audit["comparisons"] == 200 * 1854 and not audit["flagged"] and audit["eligible"] == 1854,
        "External anchor collision audit changed",
    )
    return blocks, visual, names, ids, words, unit(text), ext


def whitening(x, reg):
    val, vec = eigh((x.T @ x) / len(x), check_finite=False)
    return ((vec / (val.clip(0) + reg * val.mean()) ** 0.5) @ vec.T).astype(np.float32)


def fit(classes, data, q):
    blocks, visual, names, _, _, text, external = data
    x = np.concatenate([blocks[c] for c in classes])
    y = unit(np.concatenate([visual[c] for c in classes]))
    mu = x.mean(0)
    sd = x.std(0).clip(1e-4)
    x = (x - mu) / sd
    q = (q - mu) / sd
    centers = y.reshape(len(classes), 10, -1).mean(1)
    mean = y.mean(0)
    y = y - mean
    # Synthesize the same type of image centroid for every word. For fitted names,
    # PRESS excludes their own observed centroid from the prediction.
    k = text @ text.T
    _, txt = next(prototype_fits((k + k**3) * 0.5, names[classes], centers, [0.5]))
    wx, wy = whitening(x, 0.5), whitening(y, 0.25)
    cross = x.T @ y / len(x)
    u, vals, v = svd(wx @ cross @ wy, full_matrices=False, check_finite=False)
    # Both sides share the strongest canonical correlations. The baseline is
    # ordinary ridge with the identical input, visual targets and synthetic text anchors.
    baseline_prediction = q @ wx @ wx @ cross
    baseline = unit(baseline_prediction) @ unit(txt - mean).T
    weights = np.sqrt(vals[: max(RANKS)].clip(0))
    px = (wx @ u[:, : max(RANKS)]) * weights
    py = (wy @ v[: max(RANKS)].T) * weights
    q = q @ px
    anchors = [(a - mean) @ py for a in (txt, external)]
    scores = {}
    for rank in RANKS:
        query = unit(q[:, :rank])
        st, si = [query @ unit(a[:, :rank]).T for a in anchors]
        scores[f"text_{rank}"] = st
        scores[f"external_{rank}"] = si
        scores[f"joint_{rank}"] = (standard(st) + standard(si)) * 0.5
    return scores, baseline, dict(q=q, py=py, mean=mean, ridge=baseline_prediction)


def predict(a):
    require(sha(a.out / "design.json") == LOCK, "Unregistered experiment")
    lock = read(a.out / "design.json")
    data = training(a, lock)
    blocks, _, names, ids, words, _, _ = data
    held = np.array(lock["held_generic_classes"])
    full = np.array(lock["extended_classes"])
    require(
        not set(full) & set(lock["historically_excluded"]) and len(full) == 1012,
        "Protected test objects in fitting",
    )
    fitting = np.setdiff1d(full, held)
    scores, base, native_check = fit(fitting, data, np.concatenate([blocks[c] for c in held]))
    held_images = unit(np.concatenate([data[1][c] for c in held]))
    directions = unit(native_check["mean"][None] + native_check["ridge"])
    cos = directions @ held_images.T
    row = np.arange(len(cos))
    dump(
        a.out / "native_generic_validation.json",
        dict(
            n=len(row),
            training_classes=len(fitting),
            held_generic_classes=len(held),
            actual_photo_cosine=float(cos.diagonal().mean()),
            shuffled_photo_cosine=float(cos[row, np.roll(row, 333)].mean()),
            median_direction_cosine_to_train_mean=float(
                np.median(directions @ unit(native_check["mean"][None])[0])
            ),
            actual_photo_top1_within_10_per_held_generic_class=int(
                sum(
                    (cos[np.ix_(g, g)].argmax(1) == np.arange(10)).sum()
                    for g in row.reshape(-1, 10)
                )
            ),
            mean4_unit_to_h14="Frozen direct physical-to-native 1024-dimensional ridge companion, using 1280 held generic training photographs for verification. These generic objects have ten distinct pictures each.",
        ),
    )
    truth = names[held].repeat(10)
    valid = {key: metrics(ranks(score, truth)) for key, score in scores.items()}
    best = max(scores, key=lambda k: (valid[k]["top1"], valid[k]["mrr"]))
    select = dict(
        generic_validation_classes=len(held),
        generic_validation_photos=len(truth),
        full_vocabulary=1854,
        choice=best,
        validation=valid,
        baseline=metrics(ranks(base, truth)),
        selection_design_sha256=LOCK,
    )
    dump(a.out / "selection.json", select)
    print("Generic-only selection:", best, valid[best], select["baseline"], flush=True)
    with np.load(a.out / S / "test_budgets.npz", allow_pickle=False) as z:
        require(
            tuple(z["budgets"]) == BUDGETS and str(z["source_sha256"]) == FULL_SHA,
            "Different physical repeat packet",
        )
        query = np.stack([z["mean_four_features"], z["features"]])
        require(np.array_equal(query[0, 0], query[1, 0]), "Four-trial representation changed")
    primary_rank = int(best.split("_")[1])
    for classes in (np.array(lock["old_classes"]), full):
        size = len(classes)
        dest = a.out / S / str(size)
        dest.mkdir(exist_ok=True, parents=True)
        scores, base, model = fit(classes, data, query.reshape(-1, 1003))
        shape = (2, len(BUDGETS), 200, 1854)
        selected = scores[best].reshape(shape)
        np.savez(
            dest / "predictions.npz",
            ids=ids,
            chosen=selected,
            standard_ridge=base.reshape(shape),
            text_only=scores[f"text_{primary_rank}"].reshape(shape),
            external_only=scores[f"external_{primary_rank}"].reshape(shape),
            budgets=BUDGETS,
            method=best,
            classes=classes,
        )
        np.savez(
            dest / "visual_coordinates.npz",
            query=model["q"].reshape(2, len(BUDGETS), 200, -1)[:, :, :, :primary_rank],
            visual_projection=model["py"][:, :primary_rank],
            mean=model["mean"],
            ridge=model["ridge"].reshape(2, len(BUDGETS), 200, -1),
        )
        for j, budget in enumerate(BUDGETS):
            with (dest / f"predictions_R{budget}.csv").open("w", encoding="utf8", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["eeg_row", "predicted_id", "predicted_word", "top5_ids", "broad_category"]
                )
                for row, ind in enumerate(np.argsort(-selected[0, j], axis=1)[:, :5]):
                    w.writerow([row, ids[ind[0]], words[ind[0]], "|".join(ids[ind]), ""])
        print("Frozen", size, "class predictions", flush=True)
    dump(
        a.out / "inference_receipt.json",
        dict(
            design_sha256=LOCK,
            selection_sha256=sha(a.out / "selection.json"),
            physical_sha256=sha(a.out / S / "test_budgets.npz"),
            prediction_sha256={
                str(k): sha(a.out / S / str(k) / "predictions.npz") for k in (512, 1012)
            },
            coordinates_sha256={
                str(k): sha(a.out / S / str(k) / "visual_coordinates.npz") for k in (512, 1012)
            },
        ),
    )


def eval_images(src):
    p = src / "assets/text_test.pt"
    require(sha(p) == ASSETS["text_test.pt"][1], "Official test embedding bundle changed")
    with ZipFile(p) as z:
        (item,) = [v for v in z.infolist() if v.filename.endswith("/data/1")]
        require(item.file_size == 200 * 1024 * 4, "Unexpected official image gallery")
        return unit(np.frombuffer(z.read(item), dtype="<f4").reshape(200, 1024))


def paired(primary, other):
    x, y = np.asarray(primary, dtype=float), np.asarray(other, dtype=float)
    change = x - y
    resample = np.random.default_rng(260543).integers(len(x), size=(10000, len(x)))
    return dict(
        n=len(x),
        difference=float(change.mean()),
        ci95=np.quantile(change[resample].mean(1), [0.025, 0.975]).tolist(),
        gained=int((change > 0).sum()),
        lost=int((change < 0).sum()),
    )


def full_comparisons(primary, other, targets):
    own, control = ranks(primary, targets), ranks(other, targets)
    return {f"top{k}": paired(own <= k, control <= k) for k in (1, 5)}


def broad_foils(scores, annotations):
    """Evaluation only. Retain distractor objects sharing a manually assigned broad category."""
    hit, sizes = [], []
    for i, categories in enumerate(annotations):
        group = [j for j, other in enumerate(annotations) if categories & other]
        if len(group) < 2:
            continue
        values = scores[i, group]
        best = values == values.max()
        hit.append(float(best[group.index(i)]) / best.sum())
        sizes.append(len(group))
    return dict(
        n=len(hit),
        top1_credit=float(np.sum(hit)),
        expected_random_hits=float(np.sum(1 / np.array(sizes))),
        median_options=float(np.median(sizes)),
        min_options=min(sizes),
        max_options=max(sizes),
    )


def evaluate(a):
    frozen = read(a.out / "inference_receipt.json")
    require(
        frozen["design_sha256"] == LOCK
        and frozen["selection_sha256"] == sha(a.out / "selection.json"),
        "Selection changed",
    )
    with (a.source / "evaluation/originals_manifest.csv").open(encoding="utf8", newline="") as f:
        items = list(csv.DictReader(f))
    require(
        [int(v["eeg_row"]) for v in items] == list(range(200))
        and len({v["unique_id"] for v in items}) == 200,
        "Incorrect official test order",
    )
    original_archive = a.source / "evaluation/test_images.zip"
    require(
        sha(original_archive)
        == read(a.external.parent / "collision_audit.json")["official_archive"],
        "Unexpected official stimulus archive",
    )
    with ZipFile(original_archive) as z:
        require(
            all(
                hashlib.sha256(z.read(v["archive_member"])).hexdigest() == v["image_sha256"]
                for v in items
            ),
            "Stimulus identities differ from the 200 registered photos",
        )
    image = eval_images(a.source)
    out = dict(
        subject=S,
        unique_test_photographs=200,
        photos_per_test_object=1,
        word_candidates=1854,
        evaluation_rule="Official 200-name and 200-image restrictions used only for scoring frozen predictions. One test photo per object prevents within-object photo identification.",
        averaging=["mean_four_features", "features_of_mean_voltage"],
        seed=260541,
        by_calibration={},
    )
    shift = np.roll(np.arange(200), 67)
    for size in (512, 1012):
        dest = a.out / S / str(size)
        p = dest / "predictions.npz"
        v = dest / "visual_coordinates.npz"
        require(
            sha(p) == frozen["prediction_sha256"][str(size)]
            and sha(v) == frozen["coordinates_sha256"][str(size)],
            "Changed predictions",
        )
        with np.load(p, allow_pickle=False) as z, np.load(v, allow_pickle=False) as mod:
            ids = z["ids"]
            lookup = {w: i for i, w in enumerate(ids)}
            targets = np.array([lookup[i["unique_id"]] for i in items])
            if size == 1012:
                with (a.source / "atm_train_rows.csv").open(encoding="utf8", newline="") as f:
                    train_ids = {r["unique_id"] for r in csv.DictReader(f)}
                require(
                    not train_ids.intersection(ids[targets]),
                    "Official query categories overlap generic training",
                )
            vis = unit((image - mod["mean"]) @ mod["visual_projection"])
            # The only target-image use occurs here, after inference output hashes were frozen.
            out["by_calibration"][str(size)] = {}
            for method in ("chosen", "standard_ridge", "text_only", "external_only"):
                scores = z[method]
                out["by_calibration"][str(size)][method] = {}
                for j, b in enumerate(BUDGETS):
                    raw = scores[0, j]
                    entry = dict(
                        full1854=metrics(ranks(raw, targets)),
                        wrong_eeg_1854=metrics(ranks(raw[shift], targets)),
                        official200_names=metrics(ranks(raw[:, targets], np.arange(200))),
                    )
                    if method == "chosen":
                        entry["features_of_mean_voltage_1854"] = metrics(
                            ranks(scores[1, j], targets)
                        )
                        match = unit(mod["query"][0, j]) @ vis.T
                        entry["eval_only_actual_image_200"] = metrics(ranks(match, np.arange(200)))
                        entry["wrong_eeg_image_200"] = metrics(ranks(match[shift], np.arange(200)))
                    if method == "standard_ridge":
                        match = unit(mod["ridge"][0, j]) @ unit(image - mod["mean"]).T
                        entry["eval_only_actual_image_200"] = metrics(ranks(match, np.arange(200)))
                    out["by_calibration"][str(size)][method][str(b)] = entry
            if size == 1012:
                baseline = a.atm / "ATM_S_eeg_features_sub-04_test.pt"
                require(sha(baseline) == EEG_HASHES[3], "Released direct ATM checksum differs")
                atm = unit(tensor_storage(baseline, 200, text=False))
                with np.load(a.source / "dictionary.npz") as dic:
                    text = unit(dic["text_embeddings"])
                    annotations = [
                        {s.strip() for s in value.split(",") if s.strip()}
                        for value in dic["manual_categories"][targets]
                    ]
                require(
                    np.allclose(
                        text[targets],
                        unit(tensor_storage(a.source / "assets/text_test.pt", 200, True)),
                        rtol=1e-5,
                        atol=1e-6,
                    ),
                    "Official text row registration differs",
                )
                direct = atm @ text.T
                out["released_sub04_atm_r80"] = dict(
                    full1854_direct_text=metrics(ranks(direct, targets)),
                    official200_names=metrics(ranks(direct[:, targets], np.arange(200))),
                    eval_only_actual_image_200=metrics(ranks(atm @ image.T, np.arange(200))),
                )
                target = np.arange(200)
                primary = z["chosen"][0]
                control = z["standard_ridge"][0, -1]
                model_images = unit(mod["query"][0, -1]) @ vis.T
                out["exploratory_broadly_related_official_objects"] = dict(
                    definition="Scoring only, after the full-vocabulary inference freeze. For each official object keep other official objects sharing any manually assigned THINGS broad category. Exclude missing-category or singleton menus. These are different object identities, each with a single photograph.",
                    calibration_1012={
                        str(b): dict(
                            names=broad_foils(primary[j][:, targets], annotations),
                            eval_only_actual_images=broad_foils(
                                unit(mod["query"][0, j]) @ vis.T, annotations
                            ),
                        )
                        for j, b in enumerate(BUDGETS)
                    },
                    released_atm_r80=dict(
                        names=broad_foils(direct[:, targets], annotations),
                        eval_only_actual_images=broad_foils(atm @ image.T, annotations),
                    ),
                )

                out["paired_comparisons"] = dict(
                    intervals="10,000 percentile bootstrap draws of 200 paired object/stimulus identities, seed 260543. Conditional on this one person and the registered repeat permutation. Differences are fractions.",
                    full1854_r80_minus_atm_text=full_comparisons(primary[-1], direct, targets),
                    full1854_r80_minus_r4=full_comparisons(primary[-1], primary[0], targets),
                    full1854_r80_minus_ridge=full_comparisons(primary[-1], control, targets),
                    full1854_r80_minus_wrong=full_comparisons(
                        primary[-1], primary[-1][shift], targets
                    ),
                    names200_r80_minus_atm=paired(
                        ranks(primary[-1][:, targets], target) == 1,
                        ranks(direct[:, targets], target) == 1,
                    ),
                    names200_r80_minus_r4=paired(
                        ranks(primary[-1][:, targets], target) == 1,
                        ranks(primary[0][:, targets], target) == 1,
                    ),
                    actual_image200_r80_minus_atm=paired(
                        ranks(model_images, target) == 1, ranks(atm @ image.T, target) == 1
                    ),
                    actual_image200_r80_minus_r4=paired(
                        ranks(model_images, target) == 1,
                        ranks(unit(mod["query"][0, 0]) @ vis.T, target) == 1,
                    ),
                )
                native = unit(mod["mean"][None] + mod["ridge"][0, -1])
                cosine = native @ image.T
                out["render_direction_r80_diagnostic"] = dict(
                    note="Raw native 1024-dimensional ridge condition before IP-Adapter. This differs from the selected 32-dimensional CCA-space retrieval metric.",
                    raw_cosine_to_own_original=float(cosine.diagonal().mean()),
                    raw_cosine_to_shuffled_original=float(cosine[target, shift].mean()),
                    raw_cosine_eval_only_official200=metrics(ranks(cosine, target)),
                )
                with (a.out / "scored_queries.csv").open("w", encoding="utf8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "row",
                            "budget",
                            "true_id",
                            "predicted_id",
                            "raw_1854_rank",
                            "wrong_1854_rank",
                            "ridge_1854_rank",
                            "diagnostic_200_name_rank",
                            "diagnostic_200_image_rank",
                        ]
                    )
                    allridge = z["standard_ridge"][0]
                    for j, b in enumerate(BUDGETS):
                        x = primary[j]
                        byrow = zip(
                            ids[targets],
                            ids[x.argmax(1)],
                            ranks(x, targets),
                            ranks(x[shift], targets),
                            ranks(allridge[j], targets),
                            ranks(x[:, targets], target),
                            ranks(unit(mod["query"][0, j]) @ vis.T, target),
                            strict=True,
                        )
                        for i, item in enumerate(byrow):
                            writer.writerow([i, b, *item])
    out["inference_receipt_sha256"] = sha(a.out / "inference_receipt.json")
    dump(a.out / "evaluation.json", out)
    for size, results in out["by_calibration"].items():
        print(
            size,
            "full1854 top1/top5, closed200 names top1, actual-image200 top1",
            [
                (
                    b,
                    r["full1854"]["top1"],
                    r["full1854"]["top5"],
                    r["official200_names"]["top1"],
                    r["eval_only_actual_image_200"]["top1"],
                )
                for b, r in results["chosen"].items()
            ],
            flush=True,
        )
    print("Released direct ATM", out["released_sub04_atm_r80"], flush=True)


def package(a):
    frozen = read(a.out / "inference_receipt.json")
    dest = a.out / S / "1012"
    vectors = dest / "visual_coordinates.npz"
    prediction = dest / "predictions.npz"
    require(
        sha(vectors) == frozen["coordinates_sha256"]["1012"]
        and sha(prediction) == frozen["prediction_sha256"]["1012"],
        "Frozen inference changed",
    )
    with np.load(vectors, allow_pickle=False) as v, np.load(prediction, allow_pickle=False) as p:
        ids = p["ids"]
        require(
            len(set(ids)) == 1854 and tuple(p["budgets"]) == BUDGETS, "Different inference registry"
        )
        raw = p["chosen"]
        result = {}
        for slot, budget in ((0, 4), (4, 80)):
            tag = f"r{budget}"
            direction = unit(v["mean"][None] + v["ridge"][0, slot])
            require(
                direction.shape == (200, 1024)
                and np.allclose(np.linalg.norm(direction, axis=1), 1),
                "Expected 200 measured native H/14 directions",
            )
            result[tag] = direction.astype(np.float32)
            result[tag + "_words"] = ids[raw[0, slot].argmax(1)]
            with (dest / f"predictions_R{budget}.csv").open(encoding="utf8", newline="") as f:
                records = list(csv.DictReader(f))
            require(
                [int(r["eeg_row"]) for r in records] == list(range(200))
                and np.array_equal(result[tag + "_words"], [r["predicted_id"] for r in records]),
                "Different word packet",
            )
    r = read(a.radius)
    radius = float(np.median(r["norms"]))
    require(
        len(r["generic_rows"]) == len(r["norms"])
        and np.isclose(radius, r["median_norm"])
        and np.median(r["cosines_to_author"]) > 0.95,
        "Invalid independent visual radius",
    )
    out = a.out / S / "render_packet.npz"
    np.savez(out, rows=np.arange(200), donor=np.roll(np.arange(200), 67), radius=radius, **result)
    dump(
        out.with_suffix(".json"),
        dict(
            subject=S,
            rows=200,
            physical_repeats=[4, 80],
            words="Highest score among all 1854 unique IDs. The selected canonical similarity averages a train-synthesized text-to-image prototype score and an external reference photo score.",
            native_direction="Unit-normalized mean of unit training H14 image vectors plus the native 1024-coordinate image residual estimated by the fixed EEG-to-visual ridge. The 32 canonical coordinates are used only for semantic classification. Each EEG input averages features from disjoint four-repeat blocks.",
            radius=radius,
            radius_sha256=sha(a.radius),
            radius_source="Historical generic-only native ViT-H/14 projected-image radius.",
            donor="np.roll(arange(200),67); use the recipient noise seed after replacing the direction or word as specified.",
            condition_sha256=sha(out),
            inference_receipt_sha256=sha(a.out / "inference_receipt.json"),
            selected_word_prediction_sha256=sha(prediction),
            native_visual_prediction_sha256=sha(vectors),
        ),
    )
    print(out, sha(out), radius, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "predict", "evaluate", "package"))
    for k, val in dict(
        out="outputs/official-repeat-object",
        source="outputs/official-object-readout/sub-03",
        expanded="outputs/photo-structure/expanded",
        extra="outputs/double-calibration",
        external="outputs/external-object-anchors/sub-03/external_anchors.npz",
        atm="outputs/official-visual-anchors/assets",
        radius="outputs/photo-structure/residual_angle/radius.json",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(val))
    a = p.parse_args()
    {"register": register, "predict": predict, "evaluate": evaluate, "package": package}[a.stage](a)
