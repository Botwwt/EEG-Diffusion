"""Acquire sub-04's pre-registered extra 512 four-trial crops with per-range validation."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from prepare_trial_object_inputs import RECEIPTS, SOURCE_SHA, check, convert, inputs
from register_fresh_photo_panel import freeze, read, sha

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "scale_trial_design.json"


def metadata(data):
    config = read(DESIGN)["sources"]
    expected = {
        "original_generic_receipt": data / "trial-object-encoder/generic/receipt.json",
        "original_generic_score": data / "trial-object-encoder/generic/score.json",
        "original_generic_manifest": data / "trial-object-inputs/sub-04/generic512.json",
        "extra_design": PUBLIC / "double_calibration_design.json",
        "extra_subject_receipt": data / "double-calibration/features/sub-04.json",
        "personal_menus": data / "object-head-confirmation/candidate_panel_lock.json",
        "sessions": data / "object-transfer/sessions.json",
        "factorial_trainer": Path(__file__).with_name("ablate_trial_object_encoder.py"),
    }
    check({k: sha(p) for k, p in expected.items()} == config, "Changed frozen source")
    base = read(expected["original_generic_receipt"])
    double = read(expected["extra_design"])
    native = read(expected["extra_subject_receipt"])
    personal = {c for d in read(expected["personal_menus"])["panels"].values()
                for c in d["selected_concepts"]}
    fit, held, extra = (set(v) for v in (base["train_classes"], base["test_classes"],
                                         double["extra_objects"]))
    check((len(fit), len(held), len(extra), len(personal)) == (416, 96, 512, 29)
          and not fit & held and not (fit | held) & extra and not (fit | held | extra) & personal
          and fit | held == set(double["old_objects"])
          and set(native["raw_sha256"]) == {str(c) for c in extra}
          and native["source_full_lfs_sha256"] == SOURCE_SHA, "Class partition or source changed")
    return base, sorted(extra), native


def acquire(args):
    _, extra, native = metadata(args.data)
    groups, _, cached, source, _, _ = inputs(args.data)
    check(native["source"] == source and set(groups["stress12"]) <= set(extra), "Release mismatch")
    original = args.data / "trial-object-inputs/sub-04"
    reuse = read(original / "stress12.json")
    check(sha(original / "stress12.json") ==
          "07367d90c79371d3cd5d80d784ee3b2b38da5c87d4e38663bf199309a0421061", "Reuse receipt")
    out = args.out / "extra"
    out.mkdir(parents=True, exist_ok=True)
    options = SimpleNamespace(out=out, raw_cache=None, offline=args.offline)

    def one(c):
        raw_sha = native["raw_sha256"][str(c)]
        dest = out / f"{c:04d}.npy"
        rec = out / f"{c:04d}.json"
        if rec.exists():
            value = read(rec)
            check(value["source_raw_sha256"] == raw_sha and value["output_sha256"] == sha(dest),
                  f"Incorrect cached EEG for {c}")
            return value
        if c in groups["stress12"]:
            record = reuse["blocks"][str(c)]
            src = original / dest.name
            check(record["source_raw_sha256"] == raw_sha and sha(src) == record["output_sha256"],
                  f"Cached stimulus mismatch {c}")
            if not dest.exists():
                # The files are on the same research drive. No second copy of the raw EEG.
                try:
                    dest.hardlink_to(src)
                except OSError:
                    import shutil
                    shutil.copyfile(src, dest)
            check(sha(dest) == record["output_sha256"], "Bad linked crop")
            value = {k: record[k] for k in ("source_raw_sha256", "output_sha256")}
        else:
            value = convert(c, raw_sha, cached, source, options)
        freeze(rec, value)
        return value

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        queue = {pool.submit(one, c): c for c in extra}
        for j, future in enumerate(as_completed(queue), 1):
            results[str(queue[future])] = future.result()
            if j % 32 == 0:
                print(j, "/512 verified", flush=True)
    freeze(out / "receipt.json", dict(subject="sub-04", design_sha256=sha(DESIGN),
           source_sha256=SOURCE_SHA, raw_manifest_sha256=RECEIPTS["double-calibration/features"],
           object_ids=extra, shape=[10, 4, 63, 125], dtype="float32",
           blocks={str(c): results[str(c)] for c in extra}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/scale-trial"))
    p.add_argument("--workers", type=int, default=6, choices=range(1, 13))
    p.add_argument("--offline", action="store_true")
    acquire(p.parse_args())
