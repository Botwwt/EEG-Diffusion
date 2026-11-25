"""Evaluate three fixed individual EEG reconstruction menus and pool their overlapping objects."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_shared_photo_eeg import read, require, sha
from register_fresh_photo_panel import freeze
from render_within_photo_ip import packet
from score_fresh_photo import run as pixel_run
from score_fresh_photo import validate_packet
from score_prospective_readouts import strict
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_session_matched_images import image_rows
from score_within_photo_dino import ARMS, FILES

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity"
DESIGN = PUBLIC / "personal12_image_evaluation_design.json"
EXPECTED = "d304654ec73a5845241fe26d3a64b80c88912fe66b1b499ffa2dc281afa64938"
RENDER_PROTOCOL = PUBLIC / "single_person_confirmation_image_design.json"
PANEL = "personal12"
CLIP_MANIFEST = "91e75aa52fbb0addc004eeb38a87aa237f48e0d6bdbd5f5cc3d0d0e839d1b145"
DICTIONARY = "1aebc7a2c35edcfcfffc63934754440c90534ccc46df0537db41dc6197450132"
REFNAMES = "acc026e65dbe2fcb80f446e4039729282d89c986c547de5d2a2bafd38bc4efd7"
SEED = 260554


def aliases(a):
    d = read(DESIGN)
    lock = a.root / "candidate_panel_lock.json"
    require(
        sha(DESIGN) == EXPECTED and sha(lock) == d["registration_sha256"],
        "Different original design",
    )
    require(
        all(sha(CODE / k) == h for k, h in d["source_sha256"].items()),
        "Scoring dependencies changed",
    )
    raw = read(lock)
    regs = {}
    for sub in d["participants"]:
        reg = raw["panels"][sub]["registration"]
        folder = a.out / "references" / sub
        path = folder / "personal12_ablation.json"
        freeze(path, reg)
        require(sha(path) == d["reference_sha256"][sub], "Invalid reference registration")
        regs[sub] = registered(folder, PANEL)
        sel = {
            w: int(regs[sub]["rows"][regs[sub]["labels"] == w].min())
            for w in sorted(regs[sub]["vocab"])
        }
        require(
            sel == d["frozen_gallery_lowest_row_by_object"][sub], "Different gallery registration"
        )
    require(
        len(set.union(*(set(v["rows"]) for v in regs.values()))) == 290
        and len(set.union(*(set(v["labels"]) for v in regs.values()))) == 29,
        "Unexpected number of distinct objects or references",
    )
    return d, regs


def validate_render(a, sub, reg, driver):
    inputs = a.root / "render_inputs"
    render = inputs / "renders" / PANEL / sub
    cfg = read(render / "settings.json")
    meta = read(a.root / "render_packets.json")
    require(
        meta["study_sha256"] == driver["registration_sha256"]
        and meta["protocol_sha256"] == sha(RENDER_PROTOCOL)
        and meta["registrations_sha256"][sub] == reg["sha"]
        and meta["render_design_sha256"] == cfg["design_sha256"] == sha(inputs / "design.json")
        and meta["radius_sha256"] == sha(inputs / "radius.json")
        and meta["packets_sha256"][f"{sub}/{PANEL}"] == cfg["packet_sha256"],
        "Different render protocol, packet, or EEG registration",
    )
    packet_data = packet(inputs, sub, PANEL)
    for field, values in zip(
        ("rows", "folds", "predicted_words", "donor_rows", "support_rows"),
        packet_data[:5],
        strict=True,
    ):
        require(np.array_equal(cfg[field], values), f"Different frozen {field}")
    require(
        np.array_equal(cfg["seeds"], reg["rows"] + 923761)
        and cfg["driver_sha256"] == driver["source_sha256"]["render_within_photo_ip.py"],
        "Different diffusion seed or driver",
    )
    validate_packet(
        SimpleNamespace(
            panel=PANEL, conditions=inputs / "conditions", evaluations=inputs / "evaluations"
        ),
        sub,
        reg,
        cfg,
    )
    return cfg


def pixels(a, regs, d):
    for sub in a.subjects:
        validate_render(a, sub, regs[sub], d)
        opts = SimpleNamespace(
            panel=PANEL,
            subjects=[sub],
            panel_lock=None,
            renders=a.root / "render_inputs/renders",
            conditions=a.root / "render_inputs/conditions",
            evaluations=a.root / "render_inputs/evaluations",
            references=a.out / "references" / sub,
            originals=a.originals,
            manifest=a.manifest,
            device=a.device,
            batch_size=a.batch_size,
            model=a.model,
            out=a.out / "pixels" / sub,
            gallery_dir=a.out / "gallery",
            release=None,
            private_gallery=True,
        )
        pixel_run(opts)


def table(values, labels, universe, credit):
    """Photo-weighted estimates; a sampled object retains all its participating people and images."""
    a = np.asarray(values, dtype=float).reshape(-1, 3)
    labels = np.asarray(labels)
    require(len(labels) == len(a) and np.isfinite(a).all(), "Malformed independent outcome")
    base = {s: i for i, s in enumerate(universe)}
    ix = np.array([base[k] for k in labels], dtype=int)
    n = np.bincount(ix, minlength=len(universe))
    out = dict(n=int(n.sum()), objects_observed=int((n > 0).sum()), arms={}, own_minus={})
    if not len(labels):
        return out
    sums = np.zeros((len(universe), 3))
    np.add.at(sums, ix, a)
    draw = np.random.default_rng(SEED).integers(len(universe), size=(10000, len(universe)))
    sizes = n[draw].sum(1)
    valid = sizes > 0
    samples = sums[draw].sum(1)[valid] / sizes[valid, None]
    avg = a.mean(0)
    for j, arm in enumerate(ARMS):
        out["arms"][arm] = dict(
            mean=float(avg[j]), ci95=np.quantile(samples[:, j], [0.025, 0.975]).tolist()
        )
        if credit:
            out["arms"][arm]["credit"] = float(a[:, j].sum())
        if j:
            # Subtract cluster-level credits before division, including exact zero bootstrap draws.
            delta = (sums[:, 0] - sums[:, j])[draw].sum(1)[valid] / sizes[valid]
            out["own_minus"][arm] = dict(
                delta=float((a[:, 0] - a[:, j]).mean()),
                ci95=np.quantile(delta, [0.025, 0.975]).tolist(),
            )
    return out


def collect(a, sub, reg, driver):
    cfg = validate_render(a, sub, reg, driver)
    pixel_root = a.out / "pixels" / sub
    px = pixel_root / PANEL / sub / "per_query.json"
    cm = a.out / "clip" / (sub + ".json")
    rn = a.out / "clip" / (sub + "-refnames.json")
    summary = read(pixel_root / "summary.json")
    sdir = a.root / "render_inputs/renders" / PANEL / sub
    d, clip, ref = read(px), read(cm), read(rn)
    c = clip["cases"][PANEL + "/" + sub]
    require(
        clip["clip_sha256"] == ref["clip_sha256"] == CLIP_SHA
        and clip["dictionary_sha256"] == ref["dictionary_sha256"] == DICTIONARY
        and clip["manifest_sha256"] == ref["image_manifest_sha256"] == CLIP_MANIFEST
        and clip["scorer_sha256"] == driver["source_sha256"]["score_within_photo_ip.py"]
        and ref["scorer_sha256"] == REFNAMES
        and ref["registration_sha256"] == reg["sha"]
        and ref["targets"] == len(reg["rows"])
        and clip["design_sha256"] == cfg["design_sha256"]
        and c["settings_sha256"] == d["settings_sha256"] == sha(sdir / "settings.json")
        and d["condition_sha256"] == cfg["packet_sha256"],
        "Different source scorer, photo metadata, or image bytes",
    )
    require(
        summary["registration_sha256"] == reg["sha"]
        and summary["dino"]["sha256"] == FILES
        and summary["inputs"]["scorer_sha256"] == driver["source_sha256"]["score_fresh_photo.py"]
        and summary["inputs"]["original_zip_sha256"]
        == driver["stimulus_sha256"]["things_training_images_zip"]
        and summary["inputs"]["row_manifest_sha256"]
        == driver["stimulus_sha256"]["original_row_manifest"],
        "Different pixel/DINO references",
    )
    rows, folds, truth, words = (
        reg["rows"],
        reg["folds"],
        reg["labels"],
        np.asarray(cfg["predicted_words"]),
    )
    donors = np.array(cfg["donor_rows"])
    for key, val in dict(
        rows=rows, folds=folds, truth=truth, predicted_words=words, donor_rows=donors
    ).items():
        require(np.array_equal(c[key], val), "Different JPEG rows at " + key)
    for key, val in dict(row=rows, fold=folds, object=truth, word=words).items():
        require(
            np.array_equal([r[key] for r in d["images"]], val), "Different pixel rows at " + key
        )
    pos = {int(r): i for i, r in enumerate(rows)}
    donor = np.array([pos[int(r)] for r in donors])
    require(
        np.array_equal(folds, folds[donor]) and np.array_equal(words, words[donor]),
        "Wrong EEG is outside fold/word",
    )
    same = (rows != donors) & (truth == truth[donor])
    expected, dp, cp = strict(d, c, truth, folds, rows, words)
    keys = list(expected)
    reciprocal = np.array(
        [donors[pos[i]] == j and donors[pos[j]] == i for i, j in keys], dtype=bool
    )
    cl_reciprocal = {
        tuple(sorted(r))
        for r, yes in zip(c["pair_rows"], c["reciprocal_correct_pair"], strict=True)
        if yes
    }
    require(
        cl_reciprocal == {k for k, yes in zip(keys, reciprocal, strict=True) if yes},
        "Wrong paired-donor eligibility",
    )
    values = image_rows(d, c)
    values["clip_intra10"] = np.column_stack([c["per_row"]["intra10"][arm] for arm in ARMS])
    groups = {}
    for mark, mask in (
        ("all", np.ones(len(rows), bool)),
        ("correct_word", truth == words),
        ("same_truth_word_fold_distinct_donor", same),
    ):
        groups[mark] = {k: (v[mask], truth[mask]) for k, v in values.items()}
    labels = np.array([expected[k] for k in keys])
    for name, mask in (("strict", np.ones(len(keys), bool)), ("strict_reciprocal", reciprocal)):
        groups[name] = {
            k: (np.array([v[x] for x in keys]).reshape(-1, 3)[mask], labels[mask])
            for k, v in (("dino_assignment", dp), ("clip_assignment", cp))
        }
    with np.load(pixel_root / "original_cls" / (PANEL + ".npz"), allow_pickle=False) as z:
        require(np.array_equal(z["rows"], rows), "Wrong original hash order")
        originals = dict(zip(rows.tolist(), z["jpeg_sha256"].tolist(), strict=True))
    gallery = summary["cases"][sub]["gallery"]
    f = a.out / "gallery" / f"photo_angle_{sub.replace('-', '')}_{PANEL}.jpg"
    require(
        gallery["rows"] == list(driver["frozen_gallery_lowest_row_by_object"][sub].values())
        and sha(f) == gallery["sha256"],
        "Different fixed gallery",
    )
    meta = dict(
        settings_sha256=c["settings_sha256"],
        condition_sha256=cfg["packet_sha256"],
        pixel_rows_sha256=sha(px),
        clip_rows_sha256=sha(cm),
        reference_score_sha256=sha(rn),
        original_sha256_in_row_order=d["original_jpeg_sha256_in_row_order"],
        gallery=gallery,
        private_gallery_sha256=sha(px.with_name("private_comparison.jpg")),
        reference_n=ref["targets"],
        reference_full1854=ref["full1854_credit"],
        reference_closed12=ref["closed12_credit"],
        reference_full1854_by_object=ref["class_full1854_credit"],
        word_hits=int((truth == words).sum()),
        word_n=len(rows),
        wrong_donor_self_rows=int((rows == donors).sum()),
    )
    return groups, meta, originals


def summarize(a, regs, d):
    groups = {}
    meta = {}
    originals = {}
    for s in d["participants"]:
        groups[s], meta[s], hashes = collect(a, s, regs[s], d)
        require(
            all(r not in originals or originals[r] == h for r, h in hashes.items()),
            "Shared photo has inconsistent reference hashes",
        )
        originals.update(hashes)
    require(len(originals) == 290, "Expected 290 distinct references")

    def make(people):
        universe = sorted(set.union(*(set(regs[s]["labels"]) for s in people)))
        cases = {}
        for category, metrics in groups[people[0]].items():
            cases[category] = {}
            for key in metrics:
                items = [groups[s][category][key] for s in people]
                v, labels = (np.concatenate(a) for a in zip(*items, strict=True))
                cases[category][key] = table(
                    v, labels, universe, key not in ("rgb_rmse32", "luma_pearson32")
                )
        return cases

    original_credit = {}
    eeg_hits = []
    eeg_objects = []
    for sub, subject in meta.items():
        for word, credit in subject["reference_full1854_by_object"].items():
            require(
                word not in original_credit or original_credit[word] == credit,
                "Naming the same original photos against the full vocabulary gave different credit",
            )
            original_credit[word] = credit
        words = read(a.root / "render_inputs/renders" / PANEL / sub / "settings.json")[
            "predicted_words"
        ]
        eeg_hits.extend(regs[sub]["labels"] == words)
        eeg_objects.extend(regs[sub]["labels"])
    require(len(original_credit) == 29, "Incomplete original vocabulary")
    replicated_hits = np.repeat(np.asarray(eeg_hits, dtype=float)[:, None], 3, axis=1)
    eeg = table(replicated_hits, eeg_objects, sorted(original_credit), True)["arms"]["own"]
    result = dict(
        design_sha256=sha(DESIGN),
        source_sha256=sha(Path(__file__)),
        object_registry_sha256=d["registration_sha256"],
        render_protocol_sha256=sha(RENDER_PROTOCOL),
        render_receipt_sha256=sha(a.root / "render_packets.json"),
        counts=d["cohort_counts"],
        bootstrap_seed=SEED,
        bootstrap_draws=10000,
        by_subject={s: make([s]) for s in groups},
        pooled=make(d["participants"]),
        provenance=meta,
        eeg_word_hits=sum(x["word_hits"] for x in meta.values()),
        eeg_word_rate_ci95=eeg["ci95"],
        reference_n=sum(x["reference_n"] for x in meta.values()),
        reference_full1854=sum(x["reference_full1854"] for x in meta.values()),
        reference_closed12=sum(x["reference_closed12"] for x in meta.values()),
        unique_original_full1854=dict(n=290, credit=sum(original_credit.values())),
    )
    freeze(a.out / "summary.json", result)
    if a.release:
        freeze(a.release, result)
    print(
        "Closed 12",
        result["pooled"]["all"]["clip_closed12"]["arms"],
        "full 1854",
        result["pooled"]["all"]["clip_full1854"]["arms"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["register", "pixels", "summarize"])
    p.add_argument("--root", type=Path, default=Path("outputs/object-head-confirmation"))
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--subjects",
        nargs="+",
        choices=["sub-04", "sub-05", "sub-01"],
        default=["sub-04", "sub-05", "sub-01"],
    )
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument(
        "--originals", type=Path, default=Path("outputs/photo-structure/training_images.zip")
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--release", type=Path)
    a = p.parse_args()
    a.out = a.out or a.root / "evaluation"
    d, regs = aliases(a)
    if a.stage == "pixels":
        pixels(a, regs, d)
    elif a.stage == "summarize":
        summarize(a, regs, d)
    else:
        print(
            "Three frozen 12-object reference menus;",
            len(set.union(*(set(v["labels"]) for v in regs.values()))),
            "unique objects",
        )


if __name__ == "__main__":
    main()
