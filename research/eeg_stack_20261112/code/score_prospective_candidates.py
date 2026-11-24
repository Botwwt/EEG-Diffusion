"""Score two development-stage H/14 direction changes against the original posterior JPEGs."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from render_within_photo_ip import check, packet
from score_fresh_photo import validate_packet
from score_prospective_readouts import PANEL, SEED, SUBS, strict
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_within_photo_dino import (
    ARMS,
    FILES,
    encoder,
    originals,
    read,
    reference,
    save,
    score_case,
    sha,
)

ANCHOR = "expanded_r4"
COLS = (
    "candidate_own",
    "posterior_own",
    "posterior_uniform",
    "candidate_wrong",
    "candidate_uniform",
)


def root(a, method):
    return a.root / ("render_inputs" if method == ANCHOR else "refinements") / method


def cases(a, reg, methods):
    result = {}
    for sub in SUBS:
        for method in methods:
            source = root(a, method)
            folder = source / "renders" / PANEL / sub
            s = read(folder / "settings.json")
            rows, folds, words, donors, support, vec, radius, signature = packet(source, sub, PANEL)
            validate_packet(
                SimpleNamespace(
                    conditions=source / "conditions",
                    panel=PANEL,
                    evaluations=root(a, ANCHOR) / "evaluations",
                ),
                sub,
                reg,
                s,
            )
            for k, v in dict(
                rows=rows,
                folds=folds,
                predicted_words=words,
                donor_rows=donors,
                support_rows=support,
                seeds=923761 + rows,
            ).items():
                check(np.array_equal(s[k], v), f"{method}/{sub}: JPEGs differ from {k}")
            check(
                np.array_equal(rows, reg["rows"])
                and np.array_equal(folds, reg["folds"])
                and s["design_sha256"] == sha(source / "design.json")
                and s["packet_sha256"] == signature
                and s["radius"] == radius,
                "Registration, scale or packet differs from JPEGs",
            )
            result[method, sub] = s
            if method == ANCHOR:
                stamp = read(a.root / "render_packets.json")
                check(
                    signature == stamp["packets"][method][f"{sub}/{PANEL}"],
                    "Different native packet",
                )
                native = vec
                continue
            base = result[ANCHOR, sub]
            design = read(source / "design.json")
            check(
                design["method"] == method
                and design["baseline_design_sha256"] == base["design_sha256"]
                and design["wrong_direction_mode"] == "recipient_native_angle"
                and np.array_equal(native["uniform"], vec["uniform"])
                and all(
                    s[k] == base[k]
                    for k in (
                        "support_rows",
                        "predicted_words",
                        "donor_rows",
                        "seeds",
                        "radius",
                        "radius_sha256",
                        "adapter_sha256",
                        "dictionary_sha256",
                        "turbo_sha256",
                        "batch_size",
                    )
                ),
                "Candidate differs on a frozen factor",
            )
            with np.load(
                source / "conditions" / sub / PANEL / "conditions_512.npz", allow_pickle=False
            ) as z:
                axis = native["uniform"].astype(float)
                axis /= np.linalg.norm(axis, axis=1, keepdims=True)
                v = native["own"].astype(float)
                dot = (v * axis).sum(1)
                angle = np.arctan2(np.linalg.norm(v - dot[:, None] * axis, axis=1), dot)
                check(
                    str(z["baseline_packet_sha256"]) == base["packet_sha256"]
                    and np.max(abs(z["recipient_angle_radians"] - angle)) < 2e-6,
                    "Candidate changed the native recipient amplitude",
                )
    return result


def dino(a, methods):
    opts = SimpleNamespace(**(vars(a) | dict(out=a.work, references=a.root / "references")))
    encode = encoder(opts)
    table = reference(opts)
    with ZipFile(a.originals) as z:
        ref = originals(opts, encode, table, z, PANEL)
    for method in methods:
        bundle = root(a, method)
        for sub in SUBS:
            shots, pairs, provenance = score_case(
                bundle / "renders" / PANEL / sub, encode, ref, bundle / "conditions"
            )
            out = a.work / method / sub / "per_query.json"
            save(
                out,
                dict(
                    images=shots,
                    pairs=pairs,
                    **provenance,
                    model_sha256=FILES["model.safetensors"],
                    reference_gallery_sha256=sha(a.work / "original_cls" / (PANEL + ".npz")),
                ),
            )
            print(
                method,
                sub,
                "DINO own, uniform, wrong",
                [sum(r["intra10"][arm] for r in shots) for arm in ARMS],
                flush=True,
            )


def uncertainty(entries, people):
    """A resampled true object always brings every participant, image and comparison with it."""
    n = np.zeros((len(people), 12), int)
    sums = np.zeros((*n.shape, len(COLS)))
    for sub, category, v in entries:
        if sub in people:
            i = people.index(sub)
            n[i, category] += 1
            sums[i, category] += v
    if not n.sum():
        return dict(n=0)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(12, size=(10000, 12))
    n_boot = n.sum(0)[idx].sum(1)
    draws = sums.sum(0)[idx].sum(1)[n_boot > 0] / n_boot[n_boot > 0, None]
    counts = sums.sum((0, 1))
    return dict(
        n=int(n.sum()),
        credits=dict(zip(COLS, counts.tolist(), strict=True)),
        candidate_minus={
            k: dict(
                delta_credit=float(counts[0] - counts[j]),
                delta_rate=float((counts[0] - counts[j]) / n.sum()),
                class_ci95=np.quantile(draws[:, 0] - draws[:, j], [0.025, 0.975]).tolist(),
            )
            for j, k in enumerate(COLS)
            if j
        },
    )


def scoring_units(truth, rows, donors, strict_pairs):
    for j, t in enumerate(truth):
        yield "dino_intra10", 0, j, t
        yield "clip_full1854", 1, j, t
        if rows[j] != donors[j]:
            yield "dino_intra10_nonself_donor", 0, j, t
    for pair, t in strict_pairs.items():
        yield "dino_strict", 2, pair, t
        yield "clip_strict", 3, pair, t


def join(a, reg, cfg, methods):
    clips = {m: read(root(a, m) / "clip.json") for m in methods}
    labels = {w: j for j, w in enumerate(reg["vocab"])}
    out = dict(
        status="Method development on an already examined fourth panel. Both refinements were selected on generic calibration data.",
        primary="Exact original top-1 among ten true-class photographs using frozen DINOv2-S/14 CLS cosine, evaluated on all queries.",
        secondary="Independent OpenAI CLIP-L/14 naming over 1854 labels and strict correct-word, same-fold two-photo assignment by each encoder.",
        intervals="Paired 95% percentile intervals conditional on these three people. The same twelve true-object clusters are resampled for all people, folds, arms and candidates, 10000 draws, seed 260509. Rates are fractions. No multiplicity adjustment.",
        inputs=dict(
            registration_sha256=reg["sha"],
            code_sha256=sha(Path(__file__)),
            dino_sha256=FILES["model.safetensors"],
            clip_sha256=CLIP_SHA,
            clip_reports={m: sha(root(a, m) / "clip.json") for m in methods},
        ),
        methods={},
    )
    for m in methods:
        check(
            clips[m]["clip_sha256"] == CLIP_SHA
            and all(
                clips[m][k] == clips[ANCHOR][k]
                for k in ("dictionary_sha256", "manifest_sha256", "scorer_sha256")
            ),
            "CLIP scorers differ",
        )
    audit = []
    for m in methods[1:]:
        endpoints = {
            k: []
            for k in (
                "dino_intra10",
                "clip_full1854",
                "dino_strict",
                "clip_strict",
                "dino_intra10_nonself_donor",
            )
        }
        sources = {}
        for sub in SUBS:
            arrays = []
            for name in (ANCHOR, m):
                s = cfg[name, sub]
                c = clips[name]["cases"][PANEL + "/" + sub]
                q = a.work / name / sub / "per_query.json"
                d = read(q)
                digest = sha(root(a, name) / "renders" / PANEL / sub / "settings.json")
                rows, folds, truth, words = (
                    reg["rows"],
                    reg["folds"],
                    reg["labels"],
                    np.array(s["predicted_words"]),
                )
                check(
                    c["settings_sha256"] == d["settings_sha256"] == digest
                    and clips[name]["design_sha256"] == s["design_sha256"]
                    and d["condition_sha256"] == s["packet_sha256"]
                    and d["model_sha256"] == FILES["model.safetensors"]
                    and d["reference_gallery_sha256"]
                    == sha(a.work / "original_cls" / (PANEL + ".npz"))
                    and all(
                        np.array_equal(c[k], v)
                        for k, v in dict(
                            rows=rows, folds=folds, truth=truth, predicted_words=words
                        ).items()
                    )
                    and all(
                        np.array_equal([r[k] for r in d["images"]], v)
                        for k, v in dict(row=rows, object=truth, word=words, fold=folds).items()
                    ),
                    "Scorer provenance differs",
                )
                pairs, dp, cp = strict(d, c, truth, folds, rows, words)
                arrays.append(
                    (
                        np.array([[r["intra10"][k] for k in ARMS] for r in d["images"]]),
                        np.column_stack([c["per_row"]["full1854"][k] for k in ARMS]),
                        dp,
                        cp,
                        pairs,
                    )
                )
                sources[name + "/" + sub] = dict(
                    settings_sha256=digest,
                    packet_sha256=s["packet_sha256"],
                    dino_report_sha256=sha(q),
                )
            base, candidate = arrays
            check(base[4] == candidate[4], "Correct-word pairs changed")

            for metric, index, identity, t in scoring_units(truth, rows, s["donor_rows"], base[4]):
                v, w = base[index][identity], candidate[index][identity]
                values = [float(x) for x in (w[0], v[0], v[1], w[2], w[1])]
                endpoints[metric].append((sub, labels[t], values))
                audit.append(
                    dict(
                        candidate=m,
                        subject=sub,
                        metric=metric,
                        category=str(t),
                        rows=[int(rows[identity])] if index < 2 else list(identity),
                        credits=values,
                    )
                )
            sources[m + "/" + sub]["identical_posterior_uniform_jpegs"] = int(
                np.sum(
                    np.array(cfg[m, sub]["jpeg_sha256"]["uniform"])
                    == cfg[ANCHOR, sub]["jpeg_sha256"]["uniform"]
                )
            )
        out["methods"][m] = dict(
            pooled={k: uncertainty(v, SUBS) for k, v in endpoints.items()},
            subjects={s: {k: uncertainty(v, (s,)) for k, v in endpoints.items()} for s in SUBS},
            cases=sources,
        )
    raw = a.work / "candidate_paired_rows.json"
    save(raw, dict(columns=COLS, records=audit))
    out["inputs"]["paired_rows_sha256"] = sha(raw)
    save(a.work / "candidate_summary.json", out)
    if a.release:
        save(a.release, out)
    for m, v in out["methods"].items():
        print(m, v["pooled"]["dino_intra10"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("dino", "join", "all"))
    p.add_argument(
        "--candidates", choices=("repeat", "shared"), nargs="+", default=["repeat", "shared"]
    )
    for k, v in dict(
        root="outputs/prospective-photo",
        work="outputs/prospective-photo/paired_scores",
        model="outputs/photo-structure/dino/facebook-dinov2-small",
        originals="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--release", type=Path)
    a = p.parse_args()
    check(len(set(a.candidates)) == len(a.candidates), "Repeated method")
    reg = registered(a.root / "references", PANEL)
    methods = [ANCHOR, *a.candidates]
    cfg = cases(a, reg, methods)
    if a.stage in ("dino", "all"):
        dino(a, a.candidates)
    if a.stage in ("join", "all"):
        join(a, reg, cfg, methods)
