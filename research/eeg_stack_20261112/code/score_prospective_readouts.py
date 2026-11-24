"""Compare frozen posterior and full-scalp JPEG reconstructions on the same prospective photographs."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from render_within_photo_ip import check, packet
from score_fresh_photo import validate_packet
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_within_photo_dino import (
    ARMS,
    FILES,
    REV,
    encoder,
    originals,
    read,
    reference,
    save,
    score_case,
    sha,
)

METHODS = ("expanded_r4", "full_head_512")
SUBS = ("sub-04", "sub-05", "sub-01")
PANEL = "prospective"
SEED = 260509


def load_cases(a):
    reg = registered(a.root / "references", PANEL)
    protocol = read(a.root / "panel_lock.json")
    stamp = read(a.root / "render_packets.json")
    check(
        protocol["participants"] == list(SUBS)
        and protocol["registration_sha256"] == reg["sha"]
        and protocol["words"] == reg["vocab"].tolist()
        and stamp["public_lock_sha256"] == sha(a.root / "panel_lock.json"),
        "Different prospective registration",
    )
    cases = {}
    for m in METHODS:
        bundle = a.root / "render_inputs" / m
        for sub in SUBS:
            folder = bundle / "renders" / PANEL / sub
            settings = read(folder / "settings.json")
            rows, folds, words, donors, support, _, _, signature = packet(bundle, sub, PANEL)
            check(
                signature == stamp["packets"][m][f"{sub}/{PANEL}"]
                and settings["design_sha256"] == sha(bundle / "design.json")
                and np.array_equal(settings["seeds"], 923761 + rows)
                and np.array_equal(rows, reg["rows"])
                and np.array_equal(folds, reg["folds"])
                and np.array_equal(settings["rows"], rows)
                and np.array_equal(settings["predicted_words"], words)
                and np.array_equal(settings["donor_rows"], donors)
                and np.array_equal(settings["support_rows"], support),
                "JPEG registration differs from a frozen readout",
            )
            validate_packet(
                SimpleNamespace(
                    conditions=bundle / "conditions",
                    panel=PANEL,
                    evaluations=bundle / "evaluations",
                ),
                sub,
                reg,
                settings,
            )
            cases[m, sub] = dict(
                folder=folder,
                settings=settings,
                words=words,
                signature=signature,
                settings_sha=sha(folder / "settings.json"),
            )
    for sub in SUBS:
        old, new = (cases[m, sub]["settings"] for m in METHODS)
        same = np.asarray(old["predicted_words"]) == np.asarray(new["predicted_words"])
        paths = [
            a.root / "render_inputs" / m / "conditions" / sub / PANEL / "conditions_512.npz"
            for m in METHODS
        ]
        with np.load(paths[0], allow_pickle=False) as x, np.load(paths[1], allow_pickle=False) as y:
            check(
                all(
                    np.array_equal(x[k][same], y[k][same])
                    for k in ("support_rows", "uniform", "own")
                ),
                "Same-word queries did not receive the same matched conditions",
            )
        check(
            all(
                old[k] == new[k]
                for k in (
                    "adapter_sha256",
                    "turbo_sha256",
                    "driver_sha256",
                    "dictionary_sha256",
                    "radius",
                    "seeds",
                    "rows",
                    "folds",
                    "batch_size",
                )
            ),
            "Generator, row seeds or registration vary by readout",
        )
    return reg, cases


def dino(a, cases):
    # All six frozen render manifests have been validated before opening an original.
    opts = SimpleNamespace(**(vars(a) | dict(out=a.work, references=a.root / "references")))
    encode = encoder(opts)
    table = reference(opts)
    with ZipFile(a.originals) as z:
        ref = originals(opts, encode, table, z, PANEL)
    for m in METHODS:
        root = a.root / "render_inputs" / m
        for sub in SUBS:
            folder = cases[m, sub]["folder"]
            shots, pairs, inputs = score_case(folder, encode, ref, root / "conditions")
            save(
                a.work / m / sub / "per_query.json",
                dict(
                    images=shots,
                    pairs=pairs,
                    **inputs,
                    model_sha256=FILES["model.safetensors"],
                    reference_gallery_sha256=sha(a.work / "original_cls" / (PANEL + ".npz")),
                ),
            )
            print(
                m,
                sub,
                "DINO within-10 own",
                sum(r["intra10"]["own"] for r in shots),
                "/120",
                flush=True,
            )


def arm_vector(v):
    return np.array([v[arm] for arm in ARMS], dtype=float)


def estimate(entries, people):
    """One row/pair contributes once per arm. Resample participants and their shared object classes."""
    chosen = [r for r in entries if r[0] in people]
    if not chosen:
        return dict(n=0)
    num = np.zeros((len(people), 12), int)
    sums = np.zeros((len(people), 12, 2, 3))
    for s, c, both in chosen:
        i = people.index(s)
        num[i, c] += 1
        sums[i, c] += both
    rng = np.random.default_rng(SEED)
    si = rng.integers(len(people), size=(10000, len(people), 1))
    ci = rng.integers(12, size=(10000, 1, 12))
    n = num.sum()
    den = num[si, ci].sum((1, 2))
    draws = sums[si, ci].sum((1, 2))[den > 0] / den[den > 0, None, None]
    totals = sums.sum((0, 1))

    def delta(change, boot):
        return dict(
            delta_credit=float(change),
            delta_rate=float(change / n),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        )

    return dict(
        n=int(n),
        credits={
            m: {arm: float(totals[i, k]) for k, arm in enumerate(ARMS)}
            for i, m in enumerate(METHODS)
        },
        full_minus_posterior_own=delta(
            totals[1, 0] - totals[0, 0], draws[:, 1, 0] - draws[:, 0, 0]
        ),
        own_minus_control={
            m: {
                ARMS[k]: delta(totals[i, 0] - totals[i, k], draws[:, i, 0] - draws[:, i, k])
                for k in (1, 2)
            }
            for i, m in enumerate(METHODS)
        },
    )


def strict(d, c, truth, folds, rows, words):
    expected = {}
    for k in np.unique(truth):
        for f in range(5):
            ix = np.flatnonzero((truth == k) & (folds == f))
            check(len(ix) == 2, "Unbalanced prospective fold")
            if (words[ix] == k).all():
                expected[tuple(sorted(rows[ix].tolist()))] = k
    dino = {tuple(sorted(r["rows"])): arm_vector(r["assignment"]) for r in d["pairs"]}
    clip = {
        tuple(sorted(rr)): np.array([c["per_pair"]["assignment"][arm][j] for arm in ARMS])
        for j, rr in enumerate(c["pair_rows"])
        if c["correct_word_pair"][j]
    }
    check(
        set(dino) == set(clip) == set(expected),
        "DINO/CLIP two-photo pairs must exactly equal independently derived eligibility",
    )
    return expected, dino, clip


def join(a, reg, cases):
    clip_paths = {m: a.root / "render_inputs" / m / "clip.json" for m in METHODS}
    if a.posterior_clip:
        clip_paths[METHODS[0]] = a.posterior_clip
    if a.full_clip:
        clip_paths[METHODS[1]] = a.full_clip
    clips = {m: read(f) for m, f in clip_paths.items()}
    left, right = clips.values()
    check(
        left["clip_sha256"] == right["clip_sha256"] == CLIP_SHA
        and all(
            left[k] == right[k] for k in ("dictionary_sha256", "manifest_sha256", "scorer_sha256")
        ),
        "CLIP evaluations use different galleries, scorer or dictionary",
    )
    rows = reg["rows"]
    truth = reg["labels"]
    folds = reg["folds"]
    label = {w: i for i, w in enumerate(reg["vocab"])}
    endpoints = {
        k: []
        for k in (
            "clip_full1854",
            "clip_closed12",
            "dino_intra10",
            "clip_common_strict",
            "dino_common_strict",
        )
    }
    subsets = {}
    individual = {}
    provenance = {}
    per_rows = []
    own_counts = {}
    eeg = []
    for sub in SUBS:
        per_model = {}
        for m in METHODS:
            case = cases[m, sub]
            c = clips[m]["cases"][PANEL + "/" + sub]
            file = a.work / m / sub / "per_query.json"
            d = read(file)
            check(
                clips[m]["design_sha256"] == case["settings"]["design_sha256"]
                and c["settings_sha256"] == case["settings_sha"] == d["settings_sha256"]
                and d["condition_sha256"] == case["signature"]
                and d["model_sha256"] == FILES["model.safetensors"]
                and np.array_equal(c["rows"], rows)
                and np.array_equal(c["truth"], truth)
                and np.array_equal(c["folds"], folds)
                and np.array_equal(case["words"], c["predicted_words"])
                and np.array_equal([r["row"] for r in d["images"]], rows)
                and np.array_equal([r["object"] for r in d["images"]], truth)
                and np.array_equal([r["word"] for r in d["images"]], case["words"]),
                "Scorer inputs or row identities disagree",
            )
            pair_labels, dp, cp = strict(d, c, truth, folds, rows, case["words"])
            data = dict(
                clip_full1854=np.stack([c["per_row"]["full1854"][arm] for arm in ARMS], axis=1),
                clip_closed12=np.stack([c["per_row"]["closed12"][arm] for arm in ARMS], axis=1),
                dino_intra10=np.stack([arm_vector(r["intra10"]) for r in d["images"]]),
            )
            per_model[m] = (case, data, pair_labels, cp, dp)
            provenance[m + "/" + sub] = dict(
                settings_sha256=case["settings_sha"],
                packet_sha256=case["signature"],
                dino_report_sha256=sha(file),
            )
            own_counts[m, sub] = dict(
                n=len(pair_labels),
                dino={arm: float(sum(v[k] for v in dp.values())) for k, arm in enumerate(ARMS)},
                clip={arm: float(sum(v[k] for v in cp.values())) for k, arm in enumerate(ARMS)},
            )
        (b, bd, bp, bc, bi), (h, hd, hp, hc, hi) = per_model.values()
        same = b["words"] == h["words"]
        correct0 = b["words"] == truth
        correct1 = h["words"] == truth
        for j, w in enumerate(truth):
            eeg.append(
                (
                    sub,
                    label[w],
                    np.broadcast_to(np.array([correct0[j], correct1[j]])[:, None], (2, 3)),
                )
            )
        groups = dict(
            same_correct=same & correct0,
            same_wrong=same & ~correct0,
            corrected=~correct0 & correct1,
            lost=correct0 & ~correct1,
            changed_both_wrong=~same & ~correct0 & ~correct1,
        )
        check(
            np.all(np.sum(list(groups.values()), axis=0) == 1), "Nonexhaustive word-change groups"
        )
        subsets[sub] = {}
        jpeg_equal = (
            np.array(b["settings"]["jpeg_sha256"]["own"]) == h["settings"]["jpeg_sha256"]["own"]
        )
        for k, mask in groups.items():
            subsets[sub][k] = dict(
                n=int(mask.sum()),
                identical_own_jpegs=int(jpeg_equal[mask].sum()),
                **{
                    metric: dict(
                        posterior=float(bd[metric][mask, 0].sum()),
                        full_head=float(hd[metric][mask, 0].sum()),
                    )
                    for metric in bd
                },
            )
        for j, r in enumerate(rows):
            per_rows.append(
                dict(
                    subject=sub,
                    row=int(r),
                    fold=int(folds[j]),
                    truth=truth[j],
                    posterior_word=b["words"][j],
                    full_word=h["words"][j],
                    same_own_jpeg=bool(jpeg_equal[j]),
                    measures={
                        metric: {
                            METHODS[0]: bd[metric][j].tolist(),
                            METHODS[1]: hd[metric][j].tolist(),
                        }
                        for metric in bd
                    },
                )
            )
            for metric in bd:
                endpoints[metric].append(
                    (sub, label[truth[j]], np.stack([bd[metric][j], hd[metric][j]]))
                )
        common = sorted(bp.keys() & hp.keys())
        for pair in common:
            endpoints["clip_common_strict"].append(
                (sub, label[bp[pair]], np.stack([bc[pair], hc[pair]]))
            )
            endpoints["dino_common_strict"].append(
                (sub, label[bp[pair]], np.stack([bi[pair], hi[pair]]))
            )
        individual[sub] = dict(
            eeg_n=120,
            posterior_eeg_hits=int(correct0.sum()),
            full_head_eeg_hits=int(correct1.sum()),
            common_strict_n=len(common),
            own_strict_eligibility={m: own_counts[m, sub] for m in METHODS},
        )
    for sub in SUBS:
        individual[sub]["matched_image_scores"] = {
            k: estimate(v, [sub]) for k, v in endpoints.items()
        }
        individual[sub]["paired_eeg_difference"] = estimate(eeg, [sub])["full_minus_posterior_own"]
    raw = a.work / "paired_rows.json"
    save(raw, dict(arm_order=list(ARMS), rows=per_rows))
    result = dict(
        panel=PANEL,
        subjects=list(SUBS),
        unique_original_photos=120,
        participant_queries=360,
        models=list(METHODS),
        primary="DINOv2-S/14 top-1 original photograph among ten pictures of the true object, for all queries including incorrect predicted words.",
        object_endpoint="Independent OpenAI CLIP-L/14 recognition of all saved JPEGs against 1,854 object labels. Closed-12 scores restrict the same similarities to the twelve registered object labels. Tied names share top-1 credit.",
        strict="Compare readouts only on the intersection of correctly worded pairs matched by both original row IDs, class and fold. Separate readout-specific counts are descriptive.",
        uncertainty="Paired whole-object percentile intervals within each subject. The pooled bootstrap resamples the three people and twelve shared objects independently with replacement, 10,000 draws, seed 260509. Rates and intervals are fractions.",
        individual=individual,
        pooled_eeg_difference=estimate(eeg, list(SUBS))["full_minus_posterior_own"],
        pooled={k: estimate(v, list(SUBS)) for k, v in endpoints.items()},
        word_change_groups=subsets,
        inputs=dict(
            panel_lock_sha256=sha(a.root / "panel_lock.json"),
            registration_sha256=reg["sha"],
            original_zip_sha256=sha(a.originals),
            row_manifest_sha256=sha(a.manifest),
            dino_model=REV,
            dino_weights_sha256=FILES["model.safetensors"],
            clip_weights_sha256=CLIP_SHA,
            clip_report_sha256={m: sha(p) for m, p in clip_paths.items()},
            cases=provenance,
            paired_rows_sha256=sha(raw),
            script_sha256=sha(Path(__file__)),
        ),
    )
    save(a.work / "paired_summary.json", result)
    if a.release:
        save(a.release, result)
    for k, v in result["pooled"].items():
        print(k, v["n"], v.get("credits"), v.get("full_minus_posterior_own"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("dino", "join", "all"))
    for k, d in dict(
        root="outputs/prospective-photo",
        work="outputs/prospective-photo/paired_scores",
        model="outputs/photo-structure/dino/facebook-dinov2-small",
        originals="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(d))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--release", type=Path)
    p.add_argument("--posterior-clip", type=Path)
    p.add_argument("--full-clip", type=Path)
    a = p.parse_args()
    reg, cases = load_cases(a)
    if a.stage in ("dino", "all"):
        dino(a, cases)
    if a.stage in ("join", "all"):
        join(a, reg, cases)
