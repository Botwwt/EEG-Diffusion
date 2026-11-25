"""Register session-matched replications on the same 35 objects in two more people."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from decode_eeg_layout import CHANNELS
from prepare_expanded_personal import design_block
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE, SIZE, axes, digest, http
from prepare_trial_object_inputs import atomic, check, convert
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
SUBJECTS = ("sub-05", "sub-01")
SEED = 260629
DESIGN = PUBLIC / "fresh_people_design.json"


def register(data, out):
    first = read(PUBLIC / "fresh_mix_design.json")
    result = read(PUBLIC / "fresh_mix.json")
    sources = first["sources"]
    schedule = data / "object-transfer/sessions.json"
    manifest = data / "official-object-readout/sub-03/atm_train_rows.csv"
    frame = data / "fresh-photo/registration/exclusion_sets.json"
    check(
        sha(schedule) == sources["sessions"]
        and sha(manifest) == sources["original_photo_rows"]
        and sha(frame) == sources["exclusions"]
        and result["images_authorized_by_registered_gate"],
        "Discovery source or gate changed",
    )
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        labels = [r["unique_id"] for r in csv.DictReader(f)][::10]
    db = read(PUBLIC / "double_calibration_design.json")
    blocked = set(db["old_objects"] + db["extra_objects"] + db["fifth_objects"])
    exclusions = read(frame)["excluded_concept_ids"]
    for key, vals in exclusions.items():
        if key != "historical_rgb_gain_calibration":
            blocked.update(vals)
    for target in (
        PUBLIC / "cohort_session_matched_design.json",
        data / "object-head-confirmation/candidate_panel_lock.json",
    ):
        blocked.update(
            c for panel in read(target)["panels"].values() for c in panel["selected_concepts"]
        )
    for target in (
        "fresh-photo/references/fresh_ablation.json",
        "prospective-photo/references/prospective_ablation.json",
    ):
        blocked.update(v // 10 for v in read(data / target)["rows"])
    strict = {c for p in first["panels"].values() for c in p["primary_object_ids"]}
    decoys = set(range(1654)) - blocked - strict
    check(
        len(strict) == 35
        and len(decoys) == 147
        and not strict & blocked
        and decoys <= set(exclusions["historical_rgb_gain_calibration"]),
        "Incorrect candidate pool",
    )
    panels = {}
    evidence = {}
    for i, sub in enumerate(SUBJECTS):
        rng = np.random.default_rng(SEED + i)
        sessions = read(schedule)["subjects"][sub]["session_pairs"]
        panels[sub] = {}
        evidence[sub] = {
            key: sha(path)
            for key, path in {
                "old_trials": data / "trial-object-replication" / sub / "trials/receipt.json",
                "old_fit": data / "trial-object-replication" / sub / "fit/receipt.json",
                "extra_physical": data / "double-calibration/features" / (sub + ".json"),
                "old_physical": data / "photo-structure/expanded/features" / (sub + ".json"),
            }.items()
        }
        for pair in ((1, 3), (1, 4), (2, 3), (2, 4)):
            primary = sorted(c for c in strict if tuple(sessions[c]) == pair)
            pool = sorted(c for c in decoys if tuple(sessions[c]) == pair)
            fill = rng.choice(pool, 12 - len(primary), replace=False).tolist()
            ids = rng.permutation(primary + fill)
            order = np.stack([10 * c + rng.permutation(10) for c in ids]).reshape(12, 5, 2)
            name = f"mix{pair[0]}{pair[1]}"
            path = out / sub / "references" / (name + "_ablation.json")
            freeze(
                path,
                dict(
                    words=[labels[c] for c in ids],
                    rows=order.transpose(1, 0, 2).ravel().tolist(),
                    folds=np.repeat(np.arange(5), 24).tolist(),
                    truth=np.tile(np.repeat(np.arange(12), 2), 5).tolist(),
                    calibration_rows_by_fold_and_word=[
                        order[:, np.arange(5) != f].reshape(12, 8).tolist() for f in range(5)
                    ],
                ),
            )
            panels[sub][name] = dict(
                object_ids=ids.tolist(),
                primary_object_ids=primary,
                session_pair=list(pair),
                registration_sha256=sha(path),
            )
    spec = dict(
        seed=SEED,
        subjects=list(SUBJECTS),
        discovery_design_sha256=sha(PUBLIC / "fresh_mix_design.json"),
        discovery_word_sha256=sha(PUBLIC / "fresh_mix.json"),
        discovery_image_protocol_sha256=sha(PUBLIC / "fresh_mix_images_design.json"),
        sources=evidence,
        session_sha256=sha(schedule),
        panels=panels,
        neural_weight=result["weight"],
        acquisition="Use the discovery 35 objects in both participants. Regroup by each person's actual session pair. Fill each menu to 12 from the 147 supplemental-only objects. Ten queries per primary object, eight disjoint support photos per candidate in each fold.",
        fit="Retrain the 1024-class neural encoder in sorted object order for 1500 updates separately in each person, preserving the discovery seed, split-trial loss, architecture and optimizer. Fit the 1024 Fisher projection on that person's same objects. Transfer the single sub-04 neural weight without tuning. Compare with the previously trained personal 512 heads and equal weight.",
        scoring="Report all 700 primary participant-photo decisions and 960 including fillers. Keep matched and wrong-query cyclic controls in each menu. Resample 35 object clusters with both subjects and all 10 photos together, 10000 draws, seed 260631. The two new people, with fixed menus, are the replication sample.",
        advance="Take both full models to matched image generation only if pooled primary learned-minus-legacy and learned-minus-exchanged word accuracy have strictly positive 95% lower bounds and each participant has nonnegative net improvement. Report all failures and secondary branches without selecting a new weight.",
    )
    freeze(DESIGN, spec)
    freeze(out / "design.json", spec)


def acquire(a):
    spec = read(DESIGN)
    check(sha(a.out / "design.json") == sha(DESIGN), "Replication design mismatch")
    sub = a.subject
    root = a.out / sub
    record = a.data / "double-calibration/features" / (sub + ".json")
    check(
        sha(record) == spec["sources"][sub]["extra_physical"], "Wrong external training EEG source"
    )
    base = read(record)
    cache = a.data / "object-transfer/ranges" / REV / sub
    tail = OFFSET + SHAPE[0] * ROW_BYTES
    channels, times = axes(
        (cache / f"0-{OFFSET - 1}.bin").read_bytes(),
        (cache / f"{tail}-{SIZE - 1}.bin").read_bytes(),
    )
    check(
        np.allclose(times[50:175], np.arange(125) / 250, rtol=0, atol=1e-7),
        "Unexpected EEG time axis",
    )
    if a.stage == "extra":
        folder = root / "extra"
        folder.mkdir(parents=True, exist_ok=True)
        ids = base["range_class_order"]
        options = SimpleNamespace(out=folder, raw_cache=None, offline=a.offline)

        def one(c):
            expected = base["raw_sha256"][str(c)]
            receipt = folder / f"{c:04d}.json"
            if receipt.is_file():
                v = read(receipt)
                check(
                    v["source_raw_sha256"] == expected
                    and sha(receipt.with_suffix(".npy")) == v["output_sha256"],
                    "Changed cached crop",
                )
            else:
                v = convert(c, expected, cache, base["source"], options)
                freeze(receipt, v)
            return v

        with ThreadPoolExecutor(a.workers) as workers:
            outputs = dict(zip(map(str, ids), workers.map(one, ids), strict=True))
        freeze(
            folder / "receipt.json", dict(subject=sub, design_sha256=sha(DESIGN), blocks=outputs)
        )
        return
    ids = sorted(c for p in spec["panels"][sub].values() for c in p["object_ids"])
    check(
        len(set(ids)) == len(ids) == 48 and not set(ids) & set(base["range_class_order"]),
        "Unexpected target classes",
    )
    folder = root / "eeg/raw"
    folder.mkdir(parents=True, exist_ok=True)
    post = [channels.tolist().index(c) for c in CHANNELS]

    def target(c):
        path = folder / f"{c:04d}.bin"
        receipt = path.with_suffix(".json")
        lo, hi = OFFSET + 10 * c * ROW_BYTES, OFFSET + 10 * (c + 1) * ROW_BYTES - 1
        interval = f"{lo}-{hi}"
        if receipt.is_file():
            raw = path.read_bytes()
            check(
                read(receipt) == dict(source=base["source"], interval=interval, sha256=digest(raw)),
                "Changed target range",
            )
        else:
            check(not a.offline, "Missing verified target block")
            url = base["source"] + "?range=" + interval
            kwargs = ({"Range": "bytes=" + interval}, 10 * ROW_BYTES, f"bytes {interval}/{SIZE}")
            raw = http(url, *kwargs)
            check(
                http(url + "&verify=fresh-people", *kwargs) == raw, "Independent EEG ranges differ"
            )
            atomic(path, raw)
            freeze(receipt, dict(source=base["source"], interval=interval, sha256=digest(raw)))
        wave = np.frombuffer(raw, "<f8").reshape(10, *SHAPE[1:])
        return (
            np.ascontiguousarray(wave[:, :, :, :125], dtype=np.float32),
            design_block(raw, post),
            digest(raw),
        )

    with ThreadPoolExecutor(a.workers) as workers:
        values = list(workers.map(target, ids))
    packet = root / "eeg" / (sub + ".npz")
    frozen_npz(
        packet,
        dict(
            rows=(10 * np.array(ids)[:, None] + np.arange(10)).ravel(),
            trials=np.concatenate([v[0] for v in values]),
            features=np.concatenate([v[1] for v in values]),
            subject=sub,
            design_sha256=sha(DESIGN),
        ),
    )
    freeze(
        packet.with_suffix(".json"),
        dict(
            subject=sub,
            design_sha256=sha(DESIGN),
            packet_sha256=sha(packet),
            source=base["source"],
            raw_sha256={str(c): v[2] for c, v in zip(ids, values, strict=True)},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("register", "extra", "target"))
    p.add_argument("--subject", choices=SUBJECTS)
    p.add_argument("--out", type=Path, default=Path("outputs/fresh-people"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--offline", action="store_true")
    p.add_argument("--workers", type=int, default=6, choices=range(1, 13))
    args = p.parse_args()
    if args.stage == "register":
        register(args.data, args.out)
    elif args.subject:
        acquire(args)
    else:
        p.error("--subject is required for acquisition")
