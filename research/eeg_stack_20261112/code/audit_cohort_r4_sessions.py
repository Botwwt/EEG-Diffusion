"""Exact joint-recording-session audit of the frozen four-panel same-stimulus readout."""

import argparse
from pathlib import Path

import numpy as np
from audit_registered_sessions import conditional_null
from fit_shared_photo_eeg import read, require, sha, write
from fuse_same_stimulus_r4 import PANELS, PEOPLE, paths


def one(truth, preds, pair, scores):
    _, group = np.unique(pair, axis=0, return_inverse=True)
    groups = [np.flatnonzero(group == i) for i in range(max(group) + 1)]
    sizes = np.array([len(g) for g in groups])
    same = group[truth] == group[preds]
    multi = sizes[group[truth]] > 1
    expected = same / sizes[group[truth]]
    correct = preds == truth
    hist = conditional_null(truth, preds, groups)
    require(
        abs(np.arange(len(hist)) @ hist / hist.sum() - expected.sum()) < 1e-8, "Exact null differs"
    )
    # Keep every original 12-way winner for the principal audit. The restricted
    # score is reported separately because the decision's chance level changes.
    restricted = np.argmax(np.where(group[None, :] == group[truth, None], scores, -np.inf), axis=1)
    return dict(
        group_sizes=sorted(sizes.tolist(), reverse=True),
        groups_per_size={str(k): int((sizes == k).sum()) for k in np.unique(sizes)},
        session_only_expected_hits=int(10 * len(groups)),
        observed_hits=int(correct.sum()),
        eeg_minus_session_only=float(correct.sum() - 10 * len(groups)),
        fixed_predictions_conditional_expected_hits=float(expected.sum()),
        excess_over_label_permutation=float(correct.sum() - expected.sum()),
        exact_permutations=int(hist.sum()),
        one_sided_exact_p=float(hist[correct.sum() :].sum() / hist.sum()),
        nonsingleton_original_12way=dict(
            n=int(multi.sum()),
            correct=int(correct[multi].sum()),
            conditional_expected_hits=float(expected[multi].sum()),
        ),
        nonsingleton_restricted=dict(
            n=int(multi.sum()),
            correct=int((restricted[multi] == truth[multi]).sum()),
            session_only_expected_hits=float((1 / sizes[group[truth[multi]]]).sum()),
        ),
        groups=[
            dict(
                tuple=pair[g[0]].tolist(),
                word_indices=g.tolist(),
                n=len(g) * 10,
                original_12way_correct=int(correct[np.isin(truth, g)].sum()),
            )
            for g in groups
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k, d in dict(
        out="outputs/cohort-r4",
        inputs="outputs/object-transfer",
        angle="outputs/photo-structure/residual_angle",
        fresh="outputs/fresh-photo",
        prospective="outputs/prospective-photo",
        references="outputs/object-fidelity",
        release="research/eeg_stack_20261112/results/object_fidelity/cohort_r4_sessions.json",
    ).items():
        parser.add_argument("--" + k, type=Path, default=Path(d))
    a = parser.parse_args()
    f = a.inputs / "sessions.json"
    schedule = read(f)
    result = read(a.out / "result.json")
    all_panels = {}
    joint_hist = np.ones(1, dtype=np.int64)
    for p in PANELS:
        src = paths(a, p, PEOPLE[0])[0]
        reg = read(src)
        rows, truth = np.array(reg["rows"]), np.array(reg["truth"])
        calibration = np.array(reg["calibration_rows_by_fold_and_word"])
        concept = np.array([np.unique(rows[truth == i] // 10).item() for i in range(12)])
        pf = a.out / p / "predictions.npz"
        require(sha(pf) == result["panels"][p]["predictions_sha256"], "Frozen scores changed")
        with np.load(pf, allow_pickle=False) as z:
            scores = z["individual_scores"]
            require(np.array_equal(rows, z["rows"]), "Query identity changed")
            oneperson = {}
            signatures = []
            for i, s in enumerate(PEOPLE):
                pair = np.array(schedule["subjects"][s]["session_pairs"])
                obs = paths(a, p, s)[1]
                with np.load(obs, allow_pickle=False) as packet:
                    rr = packet["rows"]
                    require(
                        np.array_equal(packet["session_ids"].T, pair[rr // 10]),
                        "Physical packet vs official session mismatch",
                    )
                for fold in range(5):
                    require(
                        np.array_equal(
                            pair[calibration[fold] // 10],
                            np.repeat(pair[concept, None, :], 8, axis=1),
                        ),
                        "A class's eight calibration photos have different sessions",
                    )
                oneperson[s] = one(truth, z[s], pair[concept], scores[i])
                signatures.append(pair[concept])
            combined = np.concatenate(signatures, axis=1)
            cohort = one(truth, z["cohort"], combined, scores.sum(0))
            groups = [v["word_indices"] for v in cohort["groups"]]
            joint_hist = np.convolve(joint_hist, conditional_null(truth, z["cohort"], groups))
        all_panels[p] = dict(words=reg["words"], individuals=oneperson, cohort=cohort)
    total = {
        k: sum(p["cohort"][k] for p in all_panels.values())
        for k in (
            "observed_hits",
            "session_only_expected_hits",
            "fixed_predictions_conditional_expected_hits",
            "excess_over_label_permutation",
        )
    }
    total["exact_permutations"] = int(joint_hist.sum())
    total["joint_one_sided_exact_p"] = float(
        joint_hist[int(total["observed_hits"]) :].sum() / joint_hist.sum()
    )
    write(
        a.release,
        dict(
            study="Three participants' same-photo R4 data. Session-only classifier has the query's three session pairs and each registered label's permitted supports. Uniform ties imply 10 hits per occupied joint session cell in a balanced 12x10 panel.",
            conditional_null="Hold all EEG predictions fixed and permute whole class names within the exact joint tuple across all ten exemplars. This uses a different null from the session-only classifier. Singleton cells contain no contrast for object information beyond sessions.",
            source_schedule_sha256=sha(f),
            source_revision=schedule["revision"],
            recorded_packet_session_ids_equal_source=True,
            cohort_result_sha256=sha(a.out / "result.json"),
            panels=all_panels,
            combined=total,
        ),
    )
    print(total)


if __name__ == "__main__":
    main()
