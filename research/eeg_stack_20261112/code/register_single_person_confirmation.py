"""Replay three participant-specific R4 object panels using only public metadata.

The companion JSON retains names and session codes for the 64 candidate objects.
The full 512-class training list comes from double_calibration_design.json. No
EEG, photographs, per-query predictions, fitted models or scores are used.
Optionally compare the public metadata with the original local CSV and schedules.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from register_fresh_photo_panel import freeze, read, sha

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
LOCK_SHA = "b6a7413bfc9c36e7035e52028cc37935f10c49f0b5f614d16f956c64ba6940ae"


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def verify_original_sources(root, spec, ids, names, cells, original):
    sources = spec["protocol"]["source_sha256"]
    paths = {
        "double": "double-calibration/design.json",
        "cohorts": "session-matched-cohort/design.json",
        "sessions": "object-transfer/sessions.json",
        "conservative": "fresh-photo/registration/exclusion_sets.json",
        "training_rows": "official-object-readout/sub-03/atm_train_rows.csv",
        "official_personal": "official-repeat-object/design.json",
    }
    check(all(sha(root / v) == sources[k] for k, v in paths.items()), "Source metadata changed")
    pairs = read(root / paths["sessions"])["subjects"]
    with (root / paths["training_rows"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for c in ids:
        check(
            all(
                int(rows[r]["train_row"]) == r and rows[r]["unique_id"] == names[c]
                for r in range(c * 10, c * 10 + 10)
            ),
            "Mismatch with the original stimulus manifest",
        )
    for sub, subject_cells in cells.items():
        expanded = root / "photo-structure/expanded/features" / (sub + ".json")
        physical = root / "object-transfer" / (sub + ".json")
        check(
            sha(expanded) == sources[sub]["expanded"]
            and sha(physical) == sources[sub]["physical"]
            and read(expanded)["training_concepts"] == original
            and [pairs[sub]["session_pairs"][c] for c in ids] == subject_cells,
            "Training class identities or original recording sessions differ",
        )


def register(out, source_root):
    spec = read(PUBLIC / "single_person_confirmation_panel.json")
    pre = spec["protocol"]
    source = pre["source_sha256"]
    a = PUBLIC / "double_calibration_design.json"
    b = PUBLIC / "cohort_session_matched_design.json"
    for p, k in ((a, "double"), (b, "cohorts")):
        check(
            hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == source[k],
            f"Changed source registration {k}",
        )
    d = read(a)
    used = read(b)["panels"]["new_three4"]["selected_concepts"]
    ids = sorted(set(d["reserve_objects"]) - set(d["fifth_objects"]) - set(used))
    check(
        len(ids) == 64 and not set(ids).intersection(d["old_objects"] + d["extra_objects"]),
        "Changed eligible confirmation frame",
    )
    meta = spec["candidate_metadata"]
    names = dict(zip(ids, meta["words"], strict=True))
    table = meta["session_codebook"]
    cells = {
        sub: [table[int(k)] for k in meta["session_codes"][sub]] for sub in pre["participants"]
    }
    if source_root is not None:
        verify_original_sources(source_root, spec, ids, names, cells, d["old_objects"])
    rng = np.random.default_rng(pre["selection_seed"])
    panels = {}
    for sub in pre["participants"]:
        groups = {}
        for c, pair in zip(ids, cells[sub], strict=True):
            groups.setdefault(tuple(pair), []).append(c)
        pair, eligible = min(groups.items(), key=lambda x: (-len(x[1]), x[0]))
        chosen = sorted(rng.choice(eligible, 12, replace=False).tolist())
        panel = dict(
            session_pair=list(pair), source_cell_size=len(eligible), selected_concepts=chosen
        )
        check(panel == spec["selected"][sub], "Selected menu differs")
        fold_rng = np.random.default_rng(pre["fold_seed_base"] + int(sub[-2:]))
        by_class = np.array([c * 10 + fold_rng.permutation(10) for c in chosen]).reshape(12, 5, 2)
        rows = by_class.transpose(1, 0, 2).ravel()
        support = [by_class[:, np.arange(5) != f].reshape(12, 8).tolist() for f in range(5)]
        check(
            all(
                len(set(np.ravel(support[f]))) == 96
                and not set(np.ravel(support[f])).intersection(rows[24 * f : 24 * (f + 1)])
                for f in range(5)
            )
            and len(set(rows)) == 120,
            "Repeated or leaking photo",
        )
        panel["registration"] = dict(
            words=[names[c] for c in chosen],
            rows=rows.tolist(),
            truth=np.tile(np.repeat(np.arange(12), 2), 5).tolist(),
            folds=np.repeat(np.arange(5), 24).tolist(),
            calibration_rows_by_fold_and_word=support,
        )
        panels[sub] = panel
    n = len({c for panel in panels.values() for c in panel["selected_concepts"]})
    check(n == spec["archive_scope"]["distinct_concepts"] == 29, "Changed pooled object identities")
    result = dict(
        **pre,
        training_classes=sorted(d["old_objects"]),
        remaining_eligible_classes=ids,
        panels=panels,
        **spec["archive_scope"],
    )
    freeze(out, result)
    check(
        sha(out) == spec["expanded_lock_sha256"] == LOCK_SHA,
        "Historical expanded registration differs",
    )
    print(f"Confirmed 3 single-participant panels, {n} distinct objects, SHA256 {LOCK_SHA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/object-head-confirmation/candidate_panel_lock.json"),
    )
    parser.add_argument(
        "--verify-source-root",
        type=Path,
        help="Optional local outputs directory containing source metadata",
    )
    args = parser.parse_args()
    register(args.out, args.verify_source_root)
