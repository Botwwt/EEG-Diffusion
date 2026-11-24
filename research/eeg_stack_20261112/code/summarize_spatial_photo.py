"""Publish compact participant-level outcomes from the frozen spatial-transfer image experiment."""

import argparse
from pathlib import Path

from fit_spatial_photo import ARMS, PLAN_SHA, save
from render_within_photo_ip import check, read, sha

SUBJECTS = ("sub-04", "sub-05", "sub-01")
MEASURES = {
    "pixels": {
        "correct": ["gray_ssim", "luma_correlation", "rgb_rmse", "gradient_cosine"],
        "correct_strict_donor": ["gray_ssim", "luma_correlation", "rgb_rmse", "gradient_cosine"],
    },
    "dino": {
        "correct_pair": ["assignment", "cosine_margin"],
        "correct_reciprocal": ["assignment"],
        "all": ["intra10"],
    },
}


def stats(block, group, name):
    q = block[group][name]
    count = name in ("intra10", "assignment")
    out = dict(queries_or_pairs=q["n"], true_objects=q["objects"])
    out["credit" if count else "mean"] = {
        a: (round(q["n"] * v, 9) if count and v is not None else v) for a, v in q["means"].items()
    }
    for control in ("prior", "wrong", "old_own", "neutral"):
        out["own_minus_" + control] = q["comparisons"]["own_minus_" + control]
    return out


def run(a):
    lock = read(a.root / "fresh_lock.json")
    results = {}
    photo_sets = {}
    check(lock["method_sha256"] == sha(a.root / "design.json") == PLAN_SHA, "Wrong locked model")
    for age, panels in (("development", ("registered", "replicate")), ("fresh", ("fresh",))):
        results[age] = {}
        for backend, groups in MEASURES.items():
            name = backend if age == "development" else "fresh_" + backend
            p = a.root / (name + ".json")
            r = read(p)
            check(
                r["design_sha256"] == PLAN_SHA and set(r["subjects"]) == set(SUBJECTS),
                "Wrong scorer cohort",
            )
            if age == "development":
                check(
                    sha(p) == lock["old_metrics_sha256"][backend],
                    "Development result changed since lock",
                )
            cases = {}
            for s in SUBJECTS:
                d = r["subjects"][s]
                check(set(d["panels"]) == set(panels), "Wrong evaluation panels")
                settings = {}
                for panel in panels:
                    marker = a.root / "renders" / panel / s / "settings.json"
                    cfg = read(marker)
                    rows = set(cfg["rows"])
                    check(
                        photo_sets.setdefault(panel, rows) == rows,
                        "Panel photo identities differ between people",
                    )
                    check(
                        sha(marker) == d["panels"][panel]["settings_sha256"]
                        and cfg["subject_model_sha256"] == lock["generic_model_sha256"][s]
                        and cfg["subject"] == s
                        and cfg["panel"] == panel
                        and len(set(cfg["rows"])) == 120
                        and set(cfg["jpeg_sha256"]) == set(ARMS),
                        "Render/source/model identity mismatch",
                    )
                    check(
                        all(len(hs) == 120 for hs in cfg["jpeg_sha256"].values()),
                        "Missing JPEG fingerprints",
                    )
                    if age == "fresh":
                        check(
                            cfg["confirmation_sha256"] == sha(a.root / "fresh_lock.json")
                            and cfg["packet_sha256"] == lock["new_condition_sha256"][s],
                            "Confirmation input differs",
                        )
                    settings[panel] = sha(marker)
                cases[s] = dict(
                    settings_sha256=settings,
                    metrics={
                        g: {m: stats(d["pooled"], g, m) for m in ms} for g, ms in groups.items()
                    },
                )
            results[age][backend] = dict(
                score_sha256=sha(p), model_sha256=r["model_sha256"], subjects=cases
            )
    old = photo_sets["registered"] | photo_sets["replicate"]
    new = photo_sets["fresh"]
    check(
        len(old) == 240
        and len(new) == 120
        and {r // 10 for r in old}.isdisjoint({r // 10 for r in new}),
        "Panels share objects or photos",
    )
    n_people = len(SUBJECTS)
    n_photos = len(old | new)
    return dict(
        training_design_sha256=PLAN_SHA,
        fresh_lock_sha256=sha(a.root / "fresh_lock.json"),
        unique_photos=n_photos,
        participant_photo_queries=n_people * n_photos,
        old_panels=dict(
            objects=len({r // 10 for r in old}),
            distinct_photos=len(old),
            queries=n_people * len(old),
        ),
        new_panel=dict(
            objects=len({r // 10 for r in new}),
            distinct_photos=len(new),
            queries=n_people * len(new),
        ),
        jpeg_records=n_people * n_photos * len(ARMS),
        historical_lock_erratum="The narrative of the frozen fresh lock quoted 78 for development own-EEG ten-photo hits. Its authenticated DINO file gives 30 + 24 + 25 = 79 of 720; the spatial prior and unedited baseline each give 78.",
        arms=list(ARMS),
        scope="The same three participants contribute to the development and fresh panels. Test objects and photographs differ across those panels. Correct-word two-photo scores include only pairs with both expanded EEG labels correct; ties split. Wrong donors retain predicted word and fold. Strict donor rows share the true class, correct predicted word and fold with a different donor.",
        scale="DINO assignment and intra-object retrieval are listed as credits; paired differences and percentile 95% intervals remain rates. Continuous differences use raw units, lower RGB RMSE is better. The bootstrap resamples entire true objects within each participant; participants view the same photographs.",
        studies=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/spatial-photo"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    save(a.out, run(a))
