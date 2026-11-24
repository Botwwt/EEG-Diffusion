"""Measure foreground geometry and structural similarity on frozen, rendered JPEGs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fit_eeg_foreground_geometry import centered
from measure_foreground_geometry import MODEL_SHA, measure
from PIL import Image
from score_support_structure import ZIP_SHA, check, instance, score_case, sha

ARMS = ("baseline", "neutral", "prior", "query", "swapped")
AXES = ("center_x", "center_y", "log_sd_x", "log_sd_y")


def load_bank(args):
    with np.load(args.geometry, allow_pickle=False) as bank:
        rows, geom, jpeg_hash = (bank[k] for k in ("rows", "geometry", "jpeg_sha256"))
        check(
            str(bank["model_sha256"]) == MODEL_SHA
            and str(bank["images_sha256"]) == ZIP_SHA
            and tuple(bank["keys"]) == AXES,
            "Geometry bank/model identity mismatch",
        )
    with np.load(args.inputs / "generic/sub-04.npz", allow_pickle=False) as generic:
        gr = generic["rows"]
    check(
        geom.shape == (1520, 4)
        and len(np.unique(rows)) == 1520
        and np.isfinite(geom).all()
        and len(gr) == 1280
        and np.array_equal(gr % 10, np.tile(np.arange(10), 128))
        and len(set(gr // 10)) == 128,
        "Expected 1,280 generic and 240 distinct panel photos",
    )
    loc = {r: i for i, r in enumerate(rows)}
    scale = centered(geom[[loc[r] for r in gr]]).std(0).clip(1e-8)
    return loc, geom, jpeg_hash, scale, gr


def assess(args, panel, subject, bank, session):
    root = args.renders / panel / subject
    cfg_file = root / "settings.json"
    cfg = json.loads(cfg_file.read_bytes())
    spec = args.references / (panel + "_ablation.json")
    reg = json.loads(spec.read_bytes())
    photo = args.inputs / "photo_residual" / panel / "sub-04/report.json"
    prior_file = (
        args.inputs / "personal_render/personal_conditions" / panel / subject / "results.json"
    )
    prior = json.loads(prior_file.read_bytes())
    prediction = args.predictions / subject / panel / "geometry.npz"
    check(
        cfg["geometry_sha256"] == sha(prediction)
        and cfg["source_results_sha256"] == sha(prior_file)
        and cfg["conditions_sha256"] == prior["condition_sha"]
        and set(cfg["jpeg_sha256"]) == set(ARMS)
        and cfg["source_jpeg_sha256"] == cfg["jpeg_sha256"]["baseline"],
        "Wrong prediction, prior render or image arms",
    )
    truth = np.asarray(reg["words"])[reg["truth"]]
    rows, folds, words, donors = (
        np.asarray(cfg[k]) for k in ("rows", "folds", "predicted_words", "donor_rows")
    )
    check(
        reg["rows"] == prior["rows"] == rows.tolist()
        and reg["folds"] == folds.tolist()
        and truth.tolist() == prior["labels"]
        and words.tolist() == prior["predicted_words"]
        and set(donors) == set(rows)
        and np.array_equal(cfg["swap_eligible"], donors != rows),
        "Original photos, frozen words, folds or donors are misaligned",
    )
    with np.load(prediction, allow_pickle=False) as d:
        check(
            np.array_equal(rows, d["rows"]) and np.array_equal(donors, d["donor_rows"]),
            "Wrong EEG registration",
        )
    pos = {r: i for i, r in enumerate(rows)}
    donor = np.array([pos[r] for r in donors])
    loc, geom, hashes, scale, generic = bank
    idx = np.array([loc[r] for r in rows])
    check(
        not np.intersect1d(generic // 10, rows // 10).size
        and hashes[idx].tolist() == json.loads(photo.read_bytes())["original_sha256"],
        "Original foreground geometry is not from the registered unseen photos",
    )
    target = geom[idx]
    paths = {arm: [root / "images" / arm / f"{i:03}.jpg" for i in range(len(rows))] for arm in ARMS}
    measured = {}

    def geometry_scores(arm, peer):
        moments = []
        for path, expected in zip(paths[arm], cfg["jpeg_sha256"][arm], strict=True):
            blob = path.read_bytes()
            check(
                hashlib.sha256(blob).hexdigest() == expected, "JPEG changed during mask measurement"
            )
            if expected not in measured:
                with Image.open(io.BytesIO(blob)) as im:
                    measured[expected] = measure(session, im)
            moments.append(measured[expected])
        squared = ((np.stack(moments)[:, None, :] - target[None, :, :]) / scale) ** 2
        distances = squared.mean(2)
        similarity = np.where(truth[:, None] == truth[None, :], -distances, -np.inf)
        result = instance("mask_similarity", similarity, peer)
        result["mask_mse"] = -result.pop("mask_similarity")
        for i, axis in enumerate(AXES):
            result[f"mask_{axis}_mse"] = squared[np.arange(len(rows)), np.arange(len(rows)), i]
        print(subject, panel, arm, "measured", len(moments), "JPEGs", flush=True)
        return result

    shared = SimpleNamespace(
        train_zip=args.train_zip,
        train_rows=args.train_rows,
        photo_report=photo,
        registration=spec,
        primary="query",
        swap="swapped",
        baseline="baseline",
        extra_comparisons=(("neutral", "baseline"), ("prior", "neutral")),
    )
    report = score_case(
        shared, rows, folds, truth, words, donor, paths, cfg["jpeg_sha256"], geometry_scores
    )
    report["sources_sha256"] = {
        "settings": sha(cfg_file),
        "original_photo_report": sha(photo),
        "prediction": sha(prediction),
        "baseline_results": sha(prior_file),
        "registration": sha(spec),
    }
    for group, control in (("all", "prior"), ("verified_wrong_eeg_same_object", "swapped")):
        effects = report["groups"][group]["primary_minus"][control]
        print(
            subject,
            panel,
            group,
            "query minus",
            control,
            {k: effects[k]["mean"] for k in ("mask_mse", "luma_ssim", "luma_ssim_pair_assignment")},
            flush=True,
        )
    return report


def run(args):
    import onnxruntime as ort

    check(sha(args.model) == MODEL_SHA, "Unexpected segmentation model")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False
    session = ort.InferenceSession(str(args.model), sess_options=opts, providers=[args.provider])
    check(
        session.get_providers()[0] == args.provider, "Requested ONNX inference provider unavailable"
    )
    bank = load_bank(args)
    panels = {p: {s: assess(args, p, s, bank, session) for s in args.subjects} for p in args.panels}
    return dict(
        panels=panels,
        generic_within_object_sd=dict(zip(AXES, bank[3].tolist(), strict=True)),
        geometry_bank_sha256=sha(args.geometry),
        generic_rows_packet_sha256=sha(args.inputs / "generic/sub-04.npz"),
        train_zip_sha256=ZIP_SHA,
        foreground_model_sha256=MODEL_SHA,
        provider=args.provider,
        endpoints=dict(
            mask_mse="Mean squared error across saliency centroid x/y and log horizontal/vertical spread. Each error is in units of within-object generic residual SD; lower is better.",
            geometry_comparison="The frozen U2NetP is rerun on decoded final JPEGs; the readout and warp share this saliency model. Grayscale SSIM and paired photo assignment provide independent pixel comparisons.",
            assignment="The two held-out photos share object, fold and predicted word. Credit is one when the two correct similarities sum to more than the swapped sum, half for a tie. Complete reciprocal pairs only.",
            mask_similarity="The photo-identification similarity is the negative four-axis standardized MSE. Intra10 competes with eight support photos plus both held-out photos.",
            statistics="query minus each control on matched rows. Wrong-EEG comparisons require distinct EEG, same true object, predicted word and fold. Intervals resample the represented true classes 10,000 times within each participant and panel.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    root = Path("outputs/photo-structure")
    for name, default in (
        ("renders", root / "geometry_warp"),
        ("geometry", root / "foreground_geometry.npz"),
        ("predictions", root / "geometry_readout"),
        ("model", root / "u2netp.onnx"),
        ("inputs", Path("outputs/object-transfer")),
        ("references", Path("outputs/object-fidelity")),
        ("train-zip", root / "training_images.zip"),
        ("train-rows", Path("outputs/official-object-readout/sub-03/atm_train_rows.csv")),
    ):
        p.add_argument("--" + name, type=Path, default=default)
    p.add_argument("--provider", default="CPUExecutionProvider")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--subjects", nargs="+", choices=("sub-04", "sub-05"), default=["sub-04", "sub-05"]
    )
    p.add_argument(
        "--panels",
        nargs="+",
        choices=("registered", "replicate"),
        default=["registered", "replicate"],
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = run(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".partial")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(a.out)
