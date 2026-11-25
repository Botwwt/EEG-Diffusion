"""Validate the two object-prior replays and export aggregate evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(ok, message):
    if not ok:
        raise ValueError(message)


def receipt(directory, names):
    return {name: digest(directory / name) for name in names}


def summaries(root, strict):
    code = Path(__file__).resolve().parent
    public = code.parent / "results/object_fidelity"
    cov = root / "covariance"
    external = root / "external"
    c = read(cov / "report.json")
    e = read(external / "result.json")
    audit = read(external / "collision_audit.json")
    calibration = read(external / "calibration.json")
    sources = receipt(code, ("fit_support_image_covariance.py", "fit_external_word_prior.py"))
    check(
        c["source_sha256"] == sources["fit_support_image_covariance.py"]
        and e["code_sha256"] == sources["fit_external_word_prior.py"],
        "Changed fitting code",
    )
    check(
        c["design_sha256"] == digest(public / "support_image_covariance_design.json")
        and e["design_sha256"]
        == audit["design_sha256"]
        == calibration["design_sha256"]
        == digest(public / "external_word_prior_design.json"),
        "Changed design",
    )
    cd = receipt(cov, ("predictions.json",))
    ed = receipt(
        external,
        (
            "matched_conditions.npz",
            "evaluation_rows.npz",
            "calibration.json",
            "calibration.npz",
            "collision_audit.json",
        ),
    )
    check(
        cd["predictions.json"] == c["predictions_sha256"]
        and ed["matched_conditions.npz"] == e["matched_conditions_sha256"]
        and ed["calibration.json"] == e["calibration_sha256"]
        and ed["calibration.npz"] == calibration["condition_parameters_sha256"]
        and ed["collision_audit.json"] == calibration["audit_sha256"],
        "Changed decision artifacts",
    )
    decisions = read(cov / "predictions.json")
    check(
        decisions["design_sha256"] == c["design_sha256"] and len(decisions["decisions"]) == 9,
        "Incomplete covariance replay",
    )
    recorded = {
        "decisions": cd["predictions.json"]
        == "5376420bdc1ff1e8c2adc8586f71696cb41fff390998818e965642ab3ca73d3c",
        "conditions": ed["matched_conditions.npz"]
        == "95851701de52756adae9f2cccefa067b8448ea82c4e7384c6d24e8d75f2ec43a",
        "row_evaluations": ed["evaluation_rows.npz"]
        == "bae49bad51f3127019d4deeb843cacc52271992a3b90d0fff039fa4e1c14d8b2",
    }
    if strict:
        check(all(recorded.values()), "Replay bytes differ from the recorded development run")
    keys = ("object_top1", "intra10_top1", "photo_top1", "text64_top1", "text1854_top1")
    hits = {}
    with np.load(external / "evaluation_rows.npz", allow_pickle=False) as rows:
        check(len(np.unique(rows["rows"])) == e["photos"] == 640, "Incomplete external query rows")
        for arm, metrics in e["results"].items():
            hits[arm] = {key: int(rows[f"{arm}_{key}"].sum()) for key in keys}
            check(
                all(
                    rows[f"{arm}_{key}"].shape == (640,) and hits[arm][key] == metrics[key]["hits"]
                    for key in keys
                ),
                "External decisions and counts disagree",
            )
    primary = c["primary_pooled"]
    check(
        primary["n"] == sum(v["n"] for v in c["primary_by_subject"].values()) == 720
        and all(
            sum(v["hits"][arm] for v in c["primary_by_subject"].values()) == hits_total
            for arm, hits_total in primary["hits"].items()
        ),
        "Inconsistent primary denominator or counts",
    )
    return dict(
        support_image_covariance=dict(
            advance=c["advance"],
            primary_unique_photos=240,
            primary_pooled=primary,
            primary_hits_by_subject={s: v["hits"] for s, v in c["primary_by_subject"].items()},
            hits_by_task={s: v["hits"] for s, v in c["per_case"].items()},
            generic_unique_photos=1280,
            generic_pooled=c["generic_pooled"],
            design_sha256=c["design_sha256"],
            original_fitting_source_sha256="97094f5b27a35a4b68af381f1571a6549efbd4ed49da4098bb4f11e5f9367657",
            replay_artifacts=cd,
        ),
        external_word_prior=dict(
            advance=e["advance"],
            subject="sub-04",
            objects=e["objects"],
            photos=e["photos"],
            gate=e["gate"],
            feature_top1_hits=hits,
            target_cosine={k: v["target_cosine"]["mean"] for k, v in e["results"].items()},
            matched_minus={
                k: {m: v[m] for m in (*keys[:3], "target_cosine")}
                for k, v in e["matched_minus"].items()
            },
            eeg_word_head={
                k: e["full_eeg_posterior"][k]
                for k in ("full1854_top1", "full1854_top5", "names64_top1", "mean_effective_labels")
            },
            calibration={k: calibration[k] for k in ("n", "temperature", "beta")},
            collision_check=dict(pairs=audit["comparisons"], flagged_pairs=len(audit["flagged"])),
            design_sha256=e["design_sha256"],
            original_fitting_source_sha256="eadbea005100d4740a2d2e1858b7e42d761eabc60dad1120384e3a36a99bf3f9",
            original_collision_source_sha256="a808b1a212bd8027079842f80288a6c351e6e49f598e1f5dd3df2bef2c55dc66",
            replay_artifacts=ed,
        ),
        byte_matches_original_run=recorded,
        replay_source_sha256=sources,
        interval_units="fractions or cosine; paired whole objects",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/object-prior-replay"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--strict-replay",
        action="store_true",
        help="require exact historical decision and score bytes",
    )
    args = parser.parse_args()
    data = summaries(args.root, args.strict_replay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )
