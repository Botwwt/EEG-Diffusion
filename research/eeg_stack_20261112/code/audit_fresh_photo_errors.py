"""Attribute same-object / ten-photo outcomes to the query, its held-out sibling, or eight other folds."""

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
from fit_shared_photo_eeg import require, unit
from fit_within_photo_embedding import images
from PIL import Image
from score_within_photo_dino import ARMS, FILES, encoder, read, save, sha


def hd(x):
    return hashlib.sha256(x).hexdigest()


def extract(a):
    scores = read(a.public)
    require(
        a.device == scores["dino"]["device"] and a.batch_size == scores["dino"]["batch_size"],
        "Use original encoding mode",
    )
    encode = encoder(a)
    for s in scores["subjects"]:
        source = a.root / "renders/fresh" / s
        settings = read(source / "settings.json")
        require(
            sha(source / "settings.json") == scores["cases"][s]["provenance"]["settings_sha256"],
            "Render changed",
        )
        arrays = {}
        for arm in ARMS:
            batch = []
            for i, expected in enumerate(settings["jpeg_sha256"][arm]):
                raw = (source / "images" / arm / f"{i:03d}.jpg").read_bytes()
                require(hd(raw) == expected, "Changed generated JPEG")
                with Image.open(io.BytesIO(raw)) as im:
                    require(
                        im.format == "JPEG" and im.size == (512, 512) and im.mode == "RGB",
                        "Malformed JPEG",
                    )
                    batch.append(im.copy())
            arrays[arm] = encode(batch)
            require(arrays[arm].shape == (120, 384), "Invalid generated features")
        a.out.mkdir(exist_ok=True, parents=True)
        np.savez(
            a.out / (s + ".npz"),
            **arrays,
            rows=np.array(settings["rows"]),
            settings_sha256=sha(source / "settings.json"),
            model_sha256=FILES["model.safetensors"],
            processor_sha256=FILES["preprocessor_config.json"],
            revision=scores["dino"]["revision"],
            feature_sha256=np.array([hd(arrays[k].tobytes()) for k in ARMS]),
        )
        print("Verified and encoded 360 JPEGs", s, flush=True)


def routes(mat, rows, folds, truth, words, support):
    result = []
    for i, row in enumerate(rows):
        obj = np.flatnonzero(truth == truth[i])
        sibling = np.flatnonzero((truth == truth[i]) & (folds == folds[i]) & (rows != row)).item()
        other = np.flatnonzero((truth == truth[i]) & (folds != folds[i]))
        correct = bool(words[i] == truth[i])
        require(
            len(obj) == 10
            and len(other) == 8
            and (
                (set(rows[other]) == set(support[i]))
                if correct
                else not set(rows[obj]).intersection(support[i])
            ),
            "Support attribution error",
        )
        scores = mat[i]
        top = obj[scores[obj] == scores[obj].max()]
        winner = [
            np.count_nonzero(top == i) / len(top),
            np.count_nonzero(top == sibling) / len(top),
            np.count_nonzero(np.isin(top, other)) / len(top),
        ]
        require(abs(sum(winner) - 1) < 1e-12, "Unallocated top-one credit")
        d = scores[i] - scores[sibling]
        result.append(
            [
                *winner,
                scores[i],
                scores[sibling],
                scores[other].mean(),
                float(d > 1e-12) + 0.5 * float(abs(d) <= 1e-12),
                d,
                scores[i] - scores[obj[obj != i]].max(),
            ]
        )
    return np.array(result)


def metrics(x):
    n = len(x)
    w = x[:, :3].sum(0)
    return dict(
        n=n,
        self_top1=float(w[0]),
        sibling_top1=float(w[1]),
        other_folds_top1=float(w[2]),
        wrong_top1_total=float(n - w[0]),
        top1_rate_per_candidate=dict(
            self=float(w[0] / n), sibling=float(w[1] / n), each_other=float(w[2] / (8 * n))
        ),
        conditional_on_error_per_candidate=dict(
            sibling=float(w[1] / (n - w[0])), each_other=float(w[2] / (8 * (n - w[0])))
        )
        if n > w[0]
        else None,
        mean_cosine=dict(
            self=float(x[:, 3].mean()),
            sibling=float(x[:, 4].mean()),
            each_other=float(x[:, 5].mean()),
        ),
        sibling_two_choice_correct=float(x[:, 6].sum()),
        sibling_two_choice_rate=float(x[:, 6].mean()),
        own_minus_sibling_cosine=float(x[:, 7].mean()),
        own_minus_best_rival_cosine=float(x[:, 8].mean()),
    )


def audit(a):
    previous = read(a.public)
    regpath = a.root / "references/fresh_ablation.json"
    reg = read(regpath)
    rows, folds = np.array(reg["rows"]), np.array(reg["folds"])
    truth = np.array(reg["words"])[reg["truth"]]
    original = a.root / "pixel-scores/original_cls/fresh.npz"
    with np.load(original, allow_pickle=False) as z:
        dino = z["cls"]
        raw_sha = hd("".join(z["jpeg_sha256"].tolist()).encode())
        require(
            np.array_equal(rows, z["rows"])
            and sha(regpath) == previous["registration_sha256"] == str(z["registration_sha"])
            and str(z["model_sha"]) == FILES["model.safetensors"]
            and str(z["processor_sha"]) == FILES["preprocessor_config.json"]
            and str(z["script_sha"]) == previous["inputs"]["dino_scorer_sha256"]
            and hd(dino.tobytes()) == str(z["feature_sha"]),
            "Unverified original DINO features",
        )
    h14 = unit(images(a.source)[rows].astype(float))
    pooled = {}
    participants = {}
    groups = []
    provenance = {}
    labels = []
    for s in previous["subjects"]:
        pf = a.root / "conditions" / s / "fresh/conditions_512.npz"
        report = read(a.root / "pixel-scores/fresh" / s / "per_query.json")
        source = a.out / (s + ".npz")
        out = a.root / "renders/fresh" / s / "settings.json"
        r = previous["cases"][s]["provenance"]
        require(
            r["original_jpeg_sha256_in_row_order"] == raw_sha
            and sha(out) == r["settings_sha256"]
            and sha(pf) == r["condition_sha256"]
            and all(r[k] == report[k] for k in r),
            "Original or render receipt mismatch",
        )
        provenance[s] = dict(
            conditions_sha256=sha(pf),
            render_settings_sha256=sha(out),
            dino_feature_cache_sha256=sha(source),
            per_query_sha256=sha(a.root / "pixel-scores/fresh" / s / "per_query.json"),
        )
        with np.load(source, allow_pickle=False) as z, np.load(pf, allow_pickle=False) as p:
            require(
                np.array_equal(rows, p["rows"])
                and np.array_equal(rows, z["rows"])
                and np.array_equal(folds, p["folds"])
                and str(z["settings_sha256"]) == sha(out)
                and str(z["model_sha256"]) == FILES["model.safetensors"]
                and str(z["processor_sha256"]) == FILES["preprocessor_config.json"]
                and all(hd(z[k].tobytes()) == z["feature_sha256"][i] for i, k in enumerate(ARMS)),
                "Incompatible generated features",
            )
            words = p["predicted_words"]
            support = p["support_rows"]
            correct = words == truth
            require(
                np.array_equal([v["row"] for v in report["images"]], rows)
                and np.array_equal([v["word"] for v in report["images"]], words),
                "Per-query provenance mismatch",
            )
            groups.extend(correct.tolist())
            labels.extend(truth.tolist())
            person = {}
            for space in ("dino_jpeg", "h14_condition"):
                for arm in ARMS:
                    sim = (
                        z[arm].astype(float) @ dino.astype(float).T
                        if space == "dino_jpeg"
                        else unit(p[arm].astype(float)) @ h14.T
                    )
                    x = routes(sim, rows, folds, truth, words, support)
                    if space == "dino_jpeg":
                        require(
                            np.array_equal([v["intra10"][arm] for v in report["images"]], x[:, 0]),
                            "DINO single-row published top1 changed",
                        )
                    key = space + "/" + arm
                    pooled.setdefault(key, []).append(x)
                    person[key] = {
                        g: metrics(x[m])
                        for g, m in (
                            ("correct_word", correct),
                            ("wrong_word", ~correct),
                            ("all", np.ones(120, bool)),
                        )
                    }
            participants[s] = person
    groups = np.array(groups)
    labels = np.asarray(labels)
    rng = np.random.default_rng(260529)
    vocab = np.unique(labels)
    boot = rng.integers(len(vocab), size=(10000, len(vocab)))
    out = {}
    per_class_two_choice = {}
    for k, records in pooled.items():
        x = np.concatenate(records)
        out[k] = {}
        per_class_two_choice[k] = {}
        for name, m in (
            ("all", np.ones(len(x), bool)),
            ("correct_word", groups),
            ("wrong_word", ~groups),
        ):
            d = metrics(x[m])
            cls = labels[m]
            per_class_two_choice[k][name] = {
                str(v): dict(
                    n=int((cls == v).sum()),
                    correct_credit=float(x[m][cls == v, 6].sum()),
                    mean_self_minus_sibling_cosine=float(x[m][cls == v, 7].mean())
                    if np.any(cls == v)
                    else None,
                )
                for v in vocab
            }
            # One competitor per alternative. CI for the eight-vs-one comparison
            # preserves repeated participants, arms and both photos in every fold.
            values = np.column_stack(
                (x[m, 2] / 8 - x[m, 1], x[m, 5] - x[m, 4], x[m, 6] - 0.5, x[m, 7])
            )
            den = np.array([(cls == v).sum() for v in vocab])[boot].sum(1)
            tot = np.stack([values[cls == v].sum(0) for v in vocab])[boot].sum(1)
            ci = np.percentile(tot[den > 0] / den[den > 0, None], [2.5, 97.5], axis=0)
            d["cluster_intervals"] = {
                key: dict(mean=float(values[:, i].mean()), ci95=ci[:, i].tolist())
                for i, key in enumerate(
                    (
                        "each_other_minus_sibling_top1",
                        "each_other_minus_sibling_cosine",
                        "two_choice_minus_chance",
                        "self_minus_sibling_cosine",
                    )
                )
            }
            out[k][name] = d
    paired = {}
    for space in ("dino_jpeg", "h14_condition"):
        own = np.concatenate(pooled[space + "/own"])
        paired[space] = {}
        for arm in ("uniform", "wrong"):
            ctrl = np.concatenate(pooled[space + "/" + arm])
            paired[space][arm] = {}
            for group, mask in (
                ("all", np.ones(len(own), bool)),
                ("correct_word", groups),
                ("wrong_word", ~groups),
            ):
                cls = labels[mask]
                delta = (own - ctrl)[mask][:, [0, 6, 7]]
                den = np.array([(cls == v).sum() for v in vocab])[boot].sum(1)
                sums = np.stack([delta[cls == v].sum(0) for v in vocab])[boot].sum(1)
                ci = np.percentile(sums[den > 0] / den[den > 0, None], [2.5, 97.5], axis=0)
                paired[space][arm][group] = {
                    k: dict(mean=float(delta[:, i].mean()), ci95=ci[:, i].tolist())
                    for i, k in enumerate(
                        ("self_top1", "sibling_two_choice", "self_minus_sibling_cosine")
                    )
                }
    save(
        a.release,
        dict(
            study="Post-hoc error localization, third (fresh) panel, fixed three participants and shared 12 objects. Original photograph identities are aligned between spaces; metrics use separate encoders.",
            unique_photos=120,
            participant_queries=360,
            generated_jpegs=1080,
            interpretation="Correct word: the eight other-fold photographs are exactly the allowed visual supports. Wrong word: the eight are other-fold photographs of the true object; the model's eight supports are outside this gallery. Per-candidate top1 rate divides their credit by 8*n; the sibling contributes one candidate. Two-choice compares each single query with its one true-object, same-fold sibling, irrespective of the sibling's predicted word.",
            uncertainty="Exploratory paired percentile intervals resample the 12 true classes with 10000 draws, seed 260529, retaining the same three participants. H14 condition scores reuse the training representation. The generated JPEGs use the previously registered DINOv2 CLS scorer.",
            registration_sha256=sha(regpath),
            source_score_sha256=sha(a.public),
            dino_model=previous["dino"],
            original_dino_cache_sha256=sha(original),
            participant_receipts=provenance,
            audit_driver_sha256=sha(Path(__file__)),
            pooled=out,
            per_true_object_two_choice=per_class_two_choice,
            paired_own_minus_comparator=paired,
            participants=participants,
        ),
    )
    print("Verified every published DINO top-one credit for 1080 JPEGs", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("extract", "audit"))
    for k, v in dict(
        root="outputs/fresh-photo",
        out="outputs/fresh-photo/error-routing",
        source="outputs/official-object-readout/sub-03",
        model="outputs/photo-structure/dino/facebook-dinov2-small",
        public="research/eeg_stack_20261112/results/object_fidelity/fresh_photo_dino_pixels.json",
        release="research/eeg_stack_20261112/results/object_fidelity/fresh_photo_error_routes.json",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    (extract if a.stage == "extract" else audit)(a)
