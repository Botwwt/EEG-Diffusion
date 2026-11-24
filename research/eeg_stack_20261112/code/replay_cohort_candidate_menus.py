"""Replay the registered EEG choices with an explicit candidate menu for each query ordinal."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_shared_photo_eeg import load_photos, read, require, sha
from fuse_same_stimulus_r4 import PEOPLE
from register_fresh_photo_panel import freeze
from test_cohort_session_matched import NAMES, compact

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
ORIGINAL = PUBLIC / "cohort_session_matched_design.json"
MENUS = PUBLIC / "cohort_session_menus.json"
EVIDENCE = PUBLIC / "cohort_session_menu_replay.json"
RESULTS = {"stress12": "cohort_session_stress12.json", "new_three4": "cohort_session_new4.json"}


def canonical_json(data):
    return (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf8")


def menus():
    design = read(ORIGINAL)
    rendered = read(PUBLIC / "cohort_session_render_inputs.json")
    out = {}
    for name in NAMES:
        panel = design["panels"][name]
        reg = panel["registration"]
        size = panel["n_words_per_primary_decision"]
        folds = np.asarray(reg["folds"])
        require(
            np.array_equal(folds, np.repeat(np.arange(5), 24)),
            "Unexpected registered ordinal order",
        )
        require(size in (4, 12) and len(reg["words"]) == 12, "Invalid menu width")
        signature = hashlib.sha256(canonical_json(reg)).hexdigest()
        require(
            signature == rendered["reference_sha256"][name],
            "Registration hash differs from actual generation",
        )
        alternatives = np.arange(12).reshape(-1, size)
        # Each registered fold listed two queries per vocabulary position, grouped by menu.
        # Every query in the same eight-position block gets exactly the same four alternatives.
        primary = np.tile(np.repeat(np.arange(len(alternatives)), 2 * size), 5)
        settings = {
            "ordinary_12way": dict(alternatives=[list(range(12))], menu_at_ordinal=[0] * 120)
        }
        if name == "new_three4":
            settings["within_matched_four"] = dict(
                alternatives=alternatives.tolist(), menu_at_ordinal=primary.tolist()
            )
        out[name] = dict(registration_sha256=signature, decisions=settings)
    return dict(
        original_design_sha256=sha(ORIGINAL),
        provenance="Derived after the run from its immutable fold and query ordinal schedule. Every restricted query receives its registered four-word menu before EEG scoring.",
        panels=out,
    )


def classify(queries, supports, folds, available, ordinal_menu, null_order):
    """Input contains personal query EEG, calibration EEG, folds, menus and frozen permutation order."""
    full = np.full((len(PEOPLE), len(folds), 12), -np.inf)
    shifted = full.copy()
    rng = np.random.default_rng(260534)
    for f in range(5):
        for group, choices in enumerate(available):
            ix = np.flatnonzero((folds == f) & (ordinal_menu == group))
            ix = ix[np.argsort(null_order[ix])]
            require(len(ix) == 2 * len(choices), "Missing registered menu ordinals")
            offsets = rng.choice(np.arange(1, len(ix)), 2, replace=False)
            for j in range(len(PEOPLE)):
                scores = similarities(queries[j][ix], supports[j][f, choices])
                full[j][np.ix_(ix, choices)] = scores
                shifted[j][np.ix_(ix, choices)] = (
                    scores if j == 0 else np.roll(scores, offsets[j - 1], axis=0)
                )
    require(np.all(np.isfinite(full).sum(2) == len(available[0])), "Wrong number of alternatives")
    return dict(
        cohort=full.sum(0).argmax(1),
        misaligned=shifted.sum(0).argmax(1),
        **{s: v.argmax(1) for s, v in zip(PEOPLE, full, strict=True)},
    )


def fill_original(path, payload):
    if path.exists():
        require(path.read_bytes() == payload, "Existing published-run inputs differ")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def replay(a, menu):
    design = read(ORIGINAL)
    old_driver = Path(__file__).with_name("test_cohort_session_matched.py")
    published_inputs = read(PUBLIC / "cohort_session_render_inputs.json")
    require(sha(a.out / "design.json") == sha(ORIGINAL), "Incorrect registered experiment")
    checks = {}
    for name in NAMES:
        published = PUBLIC / RESULTS[name]
        record = read(published)
        require(
            sha(published) == published_inputs["comparisons"][name]["source_result_sha256"]
            and record["driver_sha256"] == sha(old_driver),
            "Original as-run report or source changed",
        )
        panel = design["panels"][name]
        reg = panel["registration"]
        folds = np.asarray(reg["folds"])
        rows = np.asarray(reg["rows"])
        cal = np.asarray(reg["calibration_rows_by_fold_and_word"])
        concepts = np.asarray(panel["selected_concepts"])
        source_rows = (concepts[:, None] * 10 + np.arange(10)).ravel()
        order = np.argsort(source_rows)
        by_row = np.sort(source_rows)
        q = np.searchsorted(by_row, rows)
        support = np.searchsorted(by_row, cal)
        require(
            np.array_equal(by_row[q], rows)
            and np.array_equal(by_row[support], cal)
            and all(
                set(cal[f].ravel()) == set(rows[folds != f]) and len(np.unique(cal[f])) == 96
                for f in range(5)
            ),
            "Invalid eight-support row sets",
        )
        queries = []
        supports = []
        old = np.asarray(read(a.double / "design.json")["old_objects"])
        require(
            len(old) == 512 and not np.intersect1d(concepts, old).size, "Target class in frozen fit"
        )
        for sub in PEOPLE:
            require(
                np.array_equal(
                    read(a.expanded / "features" / (sub + ".json"))["training_concepts"], old
                ),
                "Different training classes",
            )
            x = load_photos(a, sub, old)
            xt, signature = compact(a, sub, name, concepts)
            require(
                record["target_feature_receipts"][sub] == signature,
                "Unexpected test waveform features",
            )
            projected = fisher(x, xt[order])
            queries.append(projected[q])
            supports.append(projected[support])
        data = dict(rows=rows, folds=folds)
        report = {}
        for protocol, options in menu["panels"][name]["decisions"].items():
            choices = np.asarray(options["alternatives"])
            ordinal_menu = np.asarray(options["menu_at_ordinal"])
            result = classify(
                queries, supports, folds, choices, ordinal_menu, np.argsort(np.argsort(rows))
            )
            data.update({protocol + "_" + s: v for s, v in result.items()})
            report[protocol] = dict(
                candidates=choices.shape[1], n=len(rows), exact_arrays=list(result)
            )
        buf = io.BytesIO()
        np.savez(buf, **data)
        payload = buf.getvalue()
        signature = hashlib.sha256(payload).hexdigest()
        require(
            signature == record["predictions_sha256"],
            "Predictions differ from the actual generation source",
        )
        fill_original(a.out / name / "predictions.npz", payload)
        # The historical record keeps its generating script digest; the new digest is in this replay's receipt.
        fill_original(a.out / name / "result.json", published.read_bytes())
        checks[name] = dict(
            predictions_sha256=signature, as_run_result_sha256=sha(published), comparisons=report
        )
    freeze(
        EVIDENCE,
        dict(
            menu_sha256=sha(MENUS),
            original_design_sha256=sha(ORIGINAL),
            as_run_driver_sha256=sha(old_driver),
            replay_driver_sha256=sha(Path(__file__)),
            checked_panels=checks,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("menus", "replay"), nargs="?", default="menus")
    p.add_argument("--out", type=Path, default=Path("outputs/session-matched-cohort"))
    p.add_argument("--double", type=Path, default=Path("outputs/double-calibration"))
    p.add_argument("--expanded", type=Path, default=Path("outputs/photo-structure/expanded"))
    a = p.parse_args()
    registered = menus()
    freeze(MENUS, registered)
    if a.stage == "replay":
        replay(a, registered)
