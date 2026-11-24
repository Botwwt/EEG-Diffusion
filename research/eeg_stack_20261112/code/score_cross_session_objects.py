"""Cross early and late calibration EEG with frozen sub-03 registered-object heads.

Each directional projected score is standardized over all 12 classes for that
query, then equally averaged across the two directions. The primary fused score
is 0.75 times that average plus 0.25 times the original standardized R4 visual
score. Session IDs check the design and never enter the classifier.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from decode_eeg_layout import CHANNELS, eeg_design, registration
from fit_photo_conditions import normalize, restore_cone
from fit_registered_transfer import (
    SOURCE,
    apply_ridge,
    digest,
    packet,
    per_query_standard,
    read,
    registered_panel,
    save,
    sha,
    verify_waveforms,
)
from score_source_prototypes import transform


def load_halves(path, subject, required, ref_times, mean4, planned):
    with np.load(path, allow_pickle=False) as z:
        rows, chans = z["rows"], z["channels"].tolist()
        x, times = z["half_eeg"], z["times"]
        if (
            z["subject"].item() != subject
            or z["half_names"].tolist() != ["early", "late"]
            or z["repeats_per_half"].item() != 2
            or not np.array_equal(z["stored_slots"], [[1, 3], [0, 2]])
            or len(set(rows)) != len(rows)
            or not np.issubdtype(rows.dtype, np.integer)
            or set(chans) != set(CHANNELS)
            or len(chans) != 17
            or x.dtype != np.float32
            or x.shape != (2, len(rows), 17, 250)
            or not np.isfinite(x).all()
            or times.shape != (250,)
            or not np.allclose(times, ref_times, rtol=0, atol=1e-7)
        ):
            raise ValueError("Invalid split-half packet for " + subject)
        if (
            rows.min() < 0
            or rows.max() >= 16540
            or not np.array_equal(z["session_ids"], planned[rows // 10].T)
        ):
            raise ValueError("Public acquisition schedule and ordered half metadata disagree")
        local = {int(r): i for i, r in enumerate(rows)}
        x = x[:, [local[int(r)] for r in required]][:, :, [chans.index(c) for c in CHANNELS], :]
    # Public R4 and R2 packets may average float32 trials in different orders.
    check = verify_waveforms((x[0] + x[1]) * 0.5, mean4)
    return np.ascontiguousarray(x), check


def scores(p, m, eeg4, halves, required):
    rows, folds = p["rows"], p["folds"]
    pos = np.searchsorted(required, rows)
    channels = np.arange(17)
    half_projected = [transform(eeg_design(x, pos, channels), m) for x in halves]
    full = eeg_design(eeg4, pos, channels)
    full_projected = transform(full, m)
    hp, fp = np.split(apply_ridge(m, full, "visual"), 2, axis=1)
    directions = ("early_to_late", "late_to_early", "early_to_early", "late_to_late")
    names = (*directions, "crossed", "matched", "r4")
    pred = {
        f"{arm}_{branch}": np.empty(len(rows), np.int64)
        for arm in names
        for branch in ("projected", "fused")
    }
    pred["r4_visual_h14"] = np.empty(len(rows), np.int64)
    local = {int(r): i for i, r in enumerate(rows)}
    local_supports = []
    for fold, support in enumerate(p["support"]):
        q = np.flatnonzero(folds == fold)
        c = np.array([[local[int(r)] for r in grp] for grp in support])
        if np.intersect1d(q, c).size:
            raise ValueError("Query photograph entered a calibration half")
        local_supports.append(c)
        part = {}
        for name, (i, j) in zip(directions, ((0, 1), (1, 0), (0, 0), (1, 1)), strict=True):
            # Source and destination always refer to query half and support half.
            qi = normalize(half_projected[i][q])
            centers = normalize(half_projected[j][c].mean(1))
            part[name] = per_query_standard(qi @ centers.T)
        part["crossed"] = (part["early_to_late"] + part["late_to_early"]) * 0.5
        part["matched"] = (part["early_to_early"] + part["late_to_late"]) * 0.5
        part["r4"] = per_query_standard(
            normalize(full_projected[q]) @ normalize(full_projected[c].mean(1)).T
        )
        visual = per_query_standard(
            normalize(hp[q] - m["center_h"]) @ normalize(m["cal_h"][fold] - m["center_h"]).T
        )
        pred["r4_visual_h14"][q] = visual.argmax(1)
        for name, projected in part.items():
            pred[name + "_projected"][q] = projected.argmax(1)
            pred[name + "_fused"][q] = (0.75 * projected + 0.25 * visual).argmax(1)
    return pred, fp, local_supports


def recondition(base, p, m, picks, old_picks, fp, supports):
    # Exact R4 layout and original continuous features for all unchanged choices.
    out = {k: base[k].copy() for k in ("rows", "folds", "start", "condition", "class_only")}
    if not np.array_equal(out["rows"], p["rows"]) or not np.array_equal(out["folds"], p["folds"]):
        raise ValueError("R4 condition registration disagrees with the scored panel")
    if not np.array_equal(base["words"], p["words"][old_picks]):
        raise ValueError("R4 conditioned words differ from the original frozen EEG prediction")
    out["words"] = p["words"][picks]
    for fold, cal in enumerate(supports):
        changed = np.flatnonzero((p["folds"] == fold) & (picks != old_picks))
        if not len(changed):
            continue
        ix = picks[changed]
        centers = m["cal_f"][fold, ix]
        residual = fp[changed] - fp[cal].mean(1)[ix]
        cone = (m["global_f"], m["axis"], m["alpha"], m["radius"])
        out["condition"][changed] = restore_cone(centers + residual, *cone)
        out["class_only"][changed] = restore_cone(centers, *cone)
    return out


def pairs(a, b, truth):
    da, db = a == truth, b == truth
    d = da.astype(float) - db
    group = np.array([d[truth == k].mean() for k in np.unique(truth)])
    boot = np.random.default_rng(260427).integers(len(group), size=(10000, len(group)))
    return dict(
        delta=float(d.mean()),
        ci95=np.quantile(group[boot].mean(1), [0.025, 0.975]).tolist(),
        a_only=int((da & ~db).sum()),
        b_only=int((db & ~da).sum()),
    )


def run(args):
    if args.out.resolve() == args.transfer.resolve():
        raise ValueError("Use a separate output directory to preserve the R4 fit")
    gate = read(args.transfer / "source_verification.json")
    records, labels, _, posterior = registration(args.archive)
    layout = args.archive / "data/recovery_20261108_v236/sub-03_layout.json"
    if sha(layout) != gate["archive_sha256"]["data/recovery_20261108_v236/sub-03_layout.json"]:
        raise ValueError("Archive differs from the gated R4 source fit")
    names = read(layout)["channels"]
    if [names[i] for i in posterior] != list(CHANNELS):
        raise ValueError("Posterior waveform axes disagree with the source fit")
    cw = [r["concept_id"].split("_", 1)[1] for r in records[::10]]
    panels = [registered_panel(d, labels, cw) for d in args.panel_dirs]
    required = np.unique(np.concatenate([p["rows"] for p in panels]))
    if (
        len(required) != 240
        or {p["name"]: p["ev"]["exact_hits"] for p in panels} != gate["verified_panels"]
    ):
        raise ValueError("Expected both original replayed R4 panels")
    models = {}
    for p in panels:
        cache = args.transfer / "cache" / p["name"]
        receipt = read(cache / "generic_model.json")
        sig = digest(
            dict(archive=gate["archive_sha256"], code=gate["code_sha256"], panel=p["receipt"])
        )
        if receipt["signature"] != sig or sha(cache / "generic_model.npz") != receipt["sha256"]:
            raise ValueError("Generic model checksum or registration mismatch")
        with np.load(cache / "generic_model.npz", allow_pickle=False) as z:
            models[p["name"]] = ({k: z[k] for k in z.files if not k.startswith("latent_")}, sig)
    schedule = read(args.sessions)
    schedule_sha = sha(args.sessions)
    subjects = args.subjects or sorted(p.stem for p in args.halves_dir.glob("sub-*.npz"))
    if not subjects:
        raise ValueError("No split-half participant packets")
    for s in dict.fromkeys(subjects):
        if not re.fullmatch(r"sub-(0[1-9]|10)", s):
            raise ValueError("Invalid participant ID")
        planned = np.asarray(schedule["subjects"][s]["session_pairs"])
        if (
            planned.shape != (1654, 2)
            or not np.isin(planned[required // 10, 0], [1, 2]).all()
            or not np.isin(planned[required // 10, 1], [3, 4]).all()
        ):
            raise ValueError("Registered classes do not span the early and late acquisition blocks")
        full_path, half_path = args.subjects_dir / (s + ".npz"), args.halves_dir / (s + ".npz")
        full_sha, half_sha = sha(full_path), sha(half_path)
        if s == SOURCE and full_sha != gate["source_packet_sha256"]:
            raise ValueError("Source waveform packet differs from historical replay")
        all_eeg, times = packet(full_path, required, names)
        eeg = all_eeg[:, posterior, :]
        halves, check = load_halves(half_path, s, required, times, eeg, planned)
        for p in panels:
            m, sig = models[p["name"]]
            baseline = args.transfer / p["name"] / s
            orig = read(baseline / "evaluation.json")
            if (
                orig["packet_sha256"] != full_sha
                or orig["model_signature"] != sig
                or any(orig[k] != p["ev"][k] for k in ("rows", "words", "folds", "truth"))
            ):
                raise ValueError(
                    "Original four-repeat baseline does not match this model and packet"
                )
            predictions, fp, supports = scores(p, m, eeg, halves, required)
            for new, old in (
                ("r4_projected", "projected_eeg"),
                ("r4_fused", "fused"),
                ("r4_visual_h14", "visual_h14"),
            ):
                if not np.array_equal(predictions[new], orig["results"][old]["predicted"]):
                    raise ValueError(f"Four-repeat row-level baseline replay failed for {s}/{new}")
            truth = p["truth"]
            result = dict(
                subject=s,
                primary="crossed_fused",
                model_signature=sig,
                full_packet_sha256=full_sha,
                half_packet_sha256=half_sha,
                session_plan_sha256=schedule_sha,
                reconstruction_of_mean4=check,
                projected_single_direction_query_repeats=2,
                crossed_or_matched_query_repeats=4,
                visual_query_repeats=4,
                score_definition="Within each direction standardize scores across all 12 objects for the query; average directions, then combine 0.75 projected + 0.25 standardized original R4 H14.",
                session_usage="Session IDs verify the two acquisition blocks. Candidate classes are never filtered or weighted by session.",
                words=p["ev"]["words"],
                rows=p["ev"]["rows"],
                folds=p["ev"]["folds"],
                truth=p["ev"]["truth"],
                results={
                    k: dict(predicted=v.tolist(), n=len(v), hits=int((v == truth).sum()))
                    for k, v in predictions.items()
                },
                paired_ci_unit="12 concepts with paired photos; 10000 percentile bootstrap draws, seed 260427; fraction of queries.",
                paired={},
            )
            for branch in ("projected", "fused"):
                primary = predictions["crossed_" + branch]
                for control in ("matched_" + branch, "r4_" + branch):
                    result["paired"]["crossed_" + branch + " minus " + control] = pairs(
                        primary, predictions[control], truth
                    )
            out = args.out / p["name"] / s
            if args.export_conditions:
                base = baseline / "conditions.npz"
                if sha(base) != orig["conditions_sha256"]:
                    raise ValueError("R4 condition payload failed receipt validation")
                with np.load(base, allow_pickle=False) as z:
                    c = recondition(
                        z, p, m, predictions["crossed_fused"], predictions["r4_fused"], fp, supports
                    )
                save(out / "conditions.npz", c)
                result["conditions_sha256"] = sha(out / "conditions.npz")
            save(out / "evaluation.json", result)
            print(
                s,
                p["name"],
                {
                    k: result["results"][k]["hits"]
                    for k in ("r4_fused", "matched_fused", "crossed_fused")
                },
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("archive", "subjects-dir", "transfer", "halves-dir", "sessions", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--panel-dirs", nargs=2, type=Path, required=True)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--export-conditions", action="store_true")
    run(p.parse_args())
