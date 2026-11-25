"""Reproduce fixed native H/14 conditions for two encoders of the same sub-04 R80 EEG.

Prerequisites: the official repeat-object (1012) and global-named packets, the
frozen H/14 training targets, the word dictionary and the external visual anchors.
Run acquire, register, prepare, then score. Only score opens official test labels
and original photo embeddings. All public results refer to the same participant.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np
from calibrate_official_repeat_angles import expmap, tangent
from calibrate_official_repeat_angles import rows as within
from fit_official_repeat_objects import eval_images, fit, metrics, ranks, standard, training
from fit_shared_photo_eeg import require
from official_object_readout import acquire, sha, tensor_storage, unit
from register_fresh_photo_panel import freeze, read
from run_prospective_readout import frozen_npz
from scipy.linalg import cho_factor, cho_solve

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
REPO = PUBLIC.parents[3]
DATA = REPO / "outputs"
BASE = DATA / "official-repeat-object"
SUB = BASE / "sub-04"
STUDY = SUB / "dual-r80"
OUT = STUDY
SOURCE = DATA / "official-object-readout/sub-03"
A = SimpleNamespace(
    source=SOURCE,
    expanded=DATA / "photo-structure/expanded",
    extra=DATA / "double-calibration",
    external=DATA / "external-object-anchors/sub-03/external_anchors.npz",
)
TR = STUDY / "assets/sub04_atm_train.pt"
TE = DATA / "official-visual-anchors/assets/ATM_S_eeg_features_sub-04_test.pt"
NATIVE = SUB / "global-named/render_packet.npz"
if not NATIVE.is_file():
    NATIVE = SUB / "global_angle_packet.npz"
WORDS = SUB / "1012/predictions.npz"
EVIDENCE = BASE / "inference_receipt.json"
HISTORICAL_SHA = "54f32ff772430d53dfc5b000714bb7763d1768f350aed45c160a05b71f37bb18"
DESIGN_SHA = "63a4a4a5f34fb8ff211a6c9252977dbaedb054dacbb0754dc593b9a73f5ecc81"
TRSHA = "0f6dc610c68f45a0e6c89ab5d5a90d52d443d45fe19ffab90925e0aa269f4233"
TESHA = "1878b71916e0ff2cec8639198e005f1646f90fcf5c0f86cf488dd472e2f715df"
GEOMETRY_ATOL = 1e-9
MRR_ATOL = 1e-6


def validate_generic(actual, expected, field="generic"):
    """Match decisions and discrete metrics exactly; allow roundoff in secondary summaries."""
    require(type(actual) is type(expected), f"Different type at {field}")
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"Different keys at {field}")
        for key, value in expected.items():
            validate_generic(actual[key], value, f"{field}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"Different item count at {field}")
        for index, value in enumerate(expected):
            validate_generic(actual[index], value, f"{field}[{index}]")
    elif isinstance(expected, float):
        name = field.rsplit(".", 1)[-1]
        tolerance = 0 if name in ("within10_hits", "within_pair", "median_rank") else GEOMETRY_ATOL
        if name == "mrr":
            tolerance = MRR_ATOL
        require(
            math.isfinite(actual) and math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0),
            f"Different {field}: {actual!r} vs {expected!r} (absolute tolerance {tolerance})",
        )
    else:
        require(actual == expected, f"Different {field}: {actual!r} vs {expected!r}")


def report_bytes(report):
    return (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()


def save_latest(path, source):
    """Replace a convenience copy; immutable run receipts live under fit_replays."""
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(source.read_bytes())
    Path(tmp.name).replace(path)


def verify_design():
    published = PUBLIC / "dual_r80_fit_design.json"
    require(
        hashlib.sha256(published.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == DESIGN_SHA,
        "Different registered design",
    )
    spec = read(published)
    freeze(OUT / "fit_design.json", spec)
    freeze(OUT / "image_protocol.json", read(PUBLIC / "dual_r80_image_protocol.json"))
    for key, path in dict(
        angle_split_sha256=BASE / "angles/design.json",
        physical_selection_sha256=BASE / "selection.json",
        physical_inference_sha256=EVIDENCE,
        source_packet_sha256=NATIVE,
        physical_word_scores_sha256=WORDS,
        image_protocol_sha256=OUT / "image_protocol.json",
        training_manifest_sha256=SOURCE / "atm_train_rows.csv",
    ).items():
        require(sha(path) == spec["sources"][key], f"Wrong {key} at {path}")
    return spec


def acquire_sources():
    # SHA-verified storage is parsed without executing pickle; the source files are documentation.
    spec = read(PUBLIC / "dual_r80_fit_design.json")
    acquire(TR, (spec["sources"]["training_url"], TRSHA))
    acquire(TE, (spec["sources"]["test_url"], TESHA))
    rev = spec["third_party"]["repository_revision"]
    prefix = f"https://raw.githubusercontent.com/dongyangli-del/EEG_Image_decode/{rev}/"
    for filename, digest in spec["third_party"]["source_hashes"].items():
        folder, _, name = filename.partition("_")
        acquire(STUDY / "upstream" / filename, (prefix + folder + "/" + name, digest))


def transform_fit(x, y):
    x = unit(x.astype(float))
    y = unit(y.astype(float))
    mu = x.mean(0)
    sd = x.std(0).clip(1e-6)
    mean = y.mean(0)
    z = (x - mu) / sd
    gram = z.T @ z
    gram.flat[:: len(gram) + 1] += len(z)
    return mu, sd, mean, cho_solve(cho_factor(gram, lower=True), z.T @ (y - mean))


def mapped(x, m):
    mu, sd, mean, b = m
    return unit(mean + ((unit(x.astype(float)) - mu) / sd) @ b)


def angle(x, axis):
    return np.arccos(np.clip(unit(x.astype(float)) @ axis, -1, 1))


def spread(x, axis, gain, cap):
    return expmap(tangent(x, axis) * gain, axis, cap)


def audit(pred, truth, axis):
    x, y = unit(pred.astype(float)), unit(truth.astype(float))
    matched = (x * y).sum(1)
    other = (x * np.roll(y, 170, axis=0)).sum(1)
    m = within(x, y)
    return dict(
        n=len(x),
        mean_match=float(matched.mean()),
        mean_cross_object=float(other.mean()),
        margin=float((matched - other).mean()),
        within_pair=float(m["within_pair"].mean()),
        within10_hits=float(m["top1"].sum()),
        median_theta=float(np.median(angle(x, axis))),
        median_target_theta=float(np.median(angle(y, axis))),
    )


def correct(s, y):
    return metrics(ranks(s, y))


def prepare():
    cfg = verify_design()
    require(sha(TR) == TRSHA and sha(TE) == TESHA, "ATM release checksums differ")
    trials = tensor_storage(TR, 66160, False).reshape(16540, 4, 1024)
    all_eeg = unit(trials.mean(1, dtype=np.float64))
    h = read(BASE / "angles/design.json")
    cal, val = np.array(h["angle_calibration_classes"]), np.array(h["angle_validation_classes"])
    hold = np.r_[cal, val]
    b = read(BASE / "design.json")
    full = np.array(b["extended_classes"])
    train = np.setdiff1d(full, hold)
    require(
        len(train) == 884
        and len(set(hold)) == 128
        and len(cal) == len(val) == 64
        and len(full) == 1012,
        "Wrong visual calibration split",
    )

    def rid(c):
        return (10 * c[:, None] + np.arange(10)).ravel()

    tc, cc, vc = map(rid, (train, cal, val))
    d = training(A, b)
    blocks, y, names, ids, _, text, _ = d
    scored, _, own = fit(train, d, np.concatenate([blocks[c] for c in hold]))
    target = unit(y.reshape(-1, 1024)[tc].astype(float))
    mean = target.mean(0)
    axis = mean / np.linalg.norm(mean)
    require(np.allclose(own["mean"], mean, atol=2e-6), "Training H14 coordinates differ")
    cy, vy = [unit(y.reshape(-1, 1024)[v].astype(float)) for v in (cc, vc)]
    raw = all_eeg[cc]
    raw_audit = audit(raw, cy, axis)
    direct = (
        abs(raw_audit["median_theta"] - raw_audit["median_target_theta"]) <= 0.15
        and raw_audit["margin"] > 0.02
    )
    fitted = None if direct else transform_fit(all_eeg[tc], target)

    def atm(v):
        return v if direct else mapped(v, fitted)

    capable = atm(raw)
    gain = float(np.clip(np.median(angle(cy, axis)) / np.median(angle(capable, axis)), 0.25, 8))
    past = read(BASE / "angles/calibration.json")["parameters"]
    cap = past["real_angle_cap_radians"]
    physical = spread(own["mean"] + own["ridge"], axis, past["global_gain"], cap)
    derived = spread(atm(all_eeg[vc]), axis, gain, cap)
    alone = physical[640:]
    dual = unit(alone.astype(float) + derived)
    diag = {
        k: audit(v, vy, axis)
        for k, v in {"ours": alone, "atm_raw": all_eeg[vc], "atm": derived, "dual": dual}.items()
    }
    a, o, f = diag["atm"], diag["ours"], diag["dual"]
    gate = (
        a["margin"] >= 0.02
        and a["within_pair"] > 0.5
        and f["margin"] >= o["margin"]
        and f["margin"] > 0
        and f["within_pair"] >= o["within_pair"]
    )
    only = scored["joint_32"][640:]
    other = all_eeg[vc] @ text.T
    named = {"ours": only, "atm": other, "dual": (standard(only) + standard(other)) / 2}
    cal_result = dict(
        mode="direct" if direct else "ridge",
        raw_generic64_calibration=raw_audit,
        gain=gain,
        cap=cap,
        validation=diag,
        validation_naming={k: correct(v, names[val].repeat(10)) for k, v in named.items()},
        upstream_order_check="Held generic own versus within-class and cross-class alignment by official row ID; features are read as float32 raw Torch storage, no pickle execution.",
        passed=bool(gate),
        design_sha256=sha(OUT / "fit_design.json"),
    )
    archive = read(PUBLIC / "dual_r80_fit_packet.json")
    reference = read(PUBLIC / "dual_r80_feature_evidence.json")["generic"]
    require(
        hashlib.sha256(report_bytes(reference)).hexdigest() == archive["generic_validation_sha256"],
        "Published generic values differ from the archived registration",
    )
    validate_generic(cal_result, reference)
    source_hash = sha(Path(__file__))
    validation_hash = hashlib.sha256(report_bytes(cal_result)).hexdigest()
    replay = OUT / "fit_replays" / f"{source_hash[:16]}-{validation_hash[:16]}"
    freeze(replay / "generic_validation.json", cal_result)
    first_validation = OUT / "generic_validation.json"
    if first_validation.exists():
        validate_generic(read(first_validation), reference)
    else:
        freeze(first_validation, cal_result)
    print("Generic gate", gate, cal_result, flush=True)
    if not gate:
        return
    r = rid(full)
    real = unit(y.reshape(-1, 1024)[r].astype(float))
    mean = real.mean(0)
    axis = mean / np.linalg.norm(mean)
    with np.load(SUB / "1012/visual_coordinates.npz", allow_pickle=False) as z:
        require(np.allclose(z["mean"], mean, atol=2e-6), "Final native H14 axes disagree")
    test = unit(tensor_storage(TE, 200, False).astype(float))
    native_atm = test if direct else mapped(test, transform_fit(all_eeg[r], real))
    atm_dir = spread(native_atm, axis, gain, cap)
    with np.load(NATIVE, allow_pickle=False) as z:
        old = dict(z)
    own = old["r80"]
    both = unit(own.astype(float) + atm_dir.astype(float)).astype(np.float32)
    donor = old["donor"]
    require(np.array_equal(donor, np.roll(np.arange(200), 67)), "Changed official donor")
    packet = dict(
        rows=old["rows"],
        donor=donor,
        radius=old["radius"],
        r4=atm_dir,
        r80=both,
        r4_words=old["r80_words"],
        r80_words=old["r80_words"],
    )
    require(
        all(
            np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-6) and np.isfinite(v).all()
            for v in (atm_dir, both)
        ),
        "Bad H14 conditions",
    )
    frozen_npz(OUT / "render_packet.npz", packet)
    with np.load(WORDS, allow_pickle=False) as z:
        personal = z["chosen"][0, -1].astype(float)
        require(
            np.array_equal(z["ids"], ids)
            and np.array_equal(ids[personal.argmax(1)], old["r80_words"]),
            "Original R80 word reproduces incorrectly",
        )
    words_atm = test @ text.T
    mixed = (standard(personal) + standard(words_atm)) / 2
    frozen_npz(
        OUT / "word_and_gallery_scores.npz", dict(ids=ids, ours=personal, ATM=words_atm, dual=mixed)
    )
    computed = dict(
        packet_sha256=sha(OUT / "render_packet.npz"),
        fit_design_sha256=sha(OUT / "fit_design.json"),
        image_protocol_sha256=sha(OUT / "image_protocol.json"),
        source_sha256=source_hash,
        data_sha256=cfg["sources"],
        source_packet_sha256=sha(NATIVE),
        word_scores_sha256=sha(OUT / "word_and_gallery_scores.npz"),
        generic_validation_sha256=sha(replay / "generic_validation.json"),
        arms=["r4", "r80", "wrong"],
        aliases=cfg["aliases"],
        radius=float(old["radius"]),
        same_observations_per_image=80,
        check=dict(
            words_unchanged=True,
            donor_shift=67,
            unit_cosine_ours_atm_mean=float(np.mean(np.sum(own * atm_dir, axis=1))),
        ),
    )
    run_fields = {"source_sha256", "generic_validation_sha256"}
    require(
        archive["source_sha256"] == HISTORICAL_SHA
        and computed.keys() == archive.keys()
        and all(computed[k] == archive[k] for k in archive.keys() - run_fields),
        "Public replay does not reproduce the original packet and its inputs",
    )
    # Downstream image registrations bind the historical receipt, including its
    # as-run driver. Preserve it after exact replay and record this run separately.
    freeze(OUT / "render_packet.json", archive)
    freeze(replay / "public_fit_receipt.json", computed)
    freeze(
        replay / "public_fit_replay.json",
        dict(
            replay_dir=replay.relative_to(OUT).as_posix(),
            public_source_sha256=source_hash,
            archived_source_sha256=HISTORICAL_SHA,
            current_receipt_sha256=sha(replay / "public_fit_receipt.json"),
            archived_receipt_sha256=sha(PUBLIC / "dual_r80_fit_packet.json"),
            current_generic_sha256=computed["generic_validation_sha256"],
            archived_generic_sha256=archive["generic_validation_sha256"],
            generic_float_absolute_tolerance=dict(geometry=GEOMETRY_ATOL, mrr=MRR_ATOL),
            packet_sha256=computed["packet_sha256"],
            full_native_word_scores_sha256=computed["word_scores_sha256"],
        ),
    )
    for name in ("public_fit_receipt.json", "public_fit_replay.json"):
        save_latest(OUT / name, replay / name)
    print("Frozen", sha(OUT / "render_packet.npz"), flush=True)


def evaluate():
    r = read(OUT / "render_packet.json")
    require(
        r["word_scores_sha256"] == sha(OUT / "word_and_gallery_scores.npz")
        and r["packet_sha256"] == sha(OUT / "render_packet.npz"),
        "Inference not frozen",
    )
    with (SOURCE / "evaluation/originals_manifest.csv").open(encoding="utf8", newline="") as f:
        refs = list(csv.DictReader(f))
    require(
        [int(v["eeg_row"]) for v in refs] == list(range(200))
        and len({v["unique_id"] for v in refs}) == 200,
        "Official row registry differs",
    )
    real = eval_images(SOURCE)
    with (
        np.load(OUT / "word_and_gallery_scores.npz", allow_pickle=False) as z,
        np.load(OUT / "render_packet.npz", allow_pickle=False) as v,
        np.load(NATIVE, allow_pickle=False) as old,
    ):
        ids = z["ids"].tolist()
        truth = np.array([ids.index(x["unique_id"]) for x in refs])
        donor = v["donor"]
        naming = {k: z[k] for k in ("ours", "ATM", "dual")}
        naming["wrong_dual"] = z["dual"][donor]
        visual = {
            "ours_global": old["r80"],
            "ATM": v["r4"],
            "dual": v["r80"],
            "wrong_dual": v["r80"][donor],
        }
    result = dict(
        naming={k: correct(v, truth) for k, v in naming.items()},
        official200_word_only={k: correct(v[:, truth], np.arange(200)) for k, v in naming.items()},
        native_direction_photo200={
            k: correct(unit(v.astype(float)) @ real.T, np.arange(200)) for k, v in visual.items()
        },
    )
    rng = np.random.default_rng(261127)
    draw = rng.integers(0, 200, (10000, 200))
    paired = {}
    for n in (1, 5):
        choices = {k: ranks(v, truth) <= n for k, v in naming.items()}
        paired["top" + str(n)] = {}
        for other in ("ours", "ATM", "wrong_dual"):
            diff = choices["dual"].astype(float) - choices[other]
            paired["top" + str(n)][other] = dict(
                gained=int((diff > 0).sum()),
                lost=int((diff < 0).sum()),
                delta=float(diff.mean()),
                ci95=np.quantile(diff[draw].mean(1), [0.025, 0.975]).tolist(),
            )
    result.update(
        n=200,
        paired_word_readout=paired,
        packet_sha256=r["packet_sha256"],
        note="Full vocabulary naming readout is evaluated offline. Generated image prompts stay locked to the historical physical R80 word for both matched arms. Native_direction_photo200 uses true H14 embeddings only in evaluation. Equal 65/200 from the historical report refers to different readout coordinates.",
    )
    freeze(OUT / "official_evaluation.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("acquire", "register", "prepare", "score"))
    p.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="Private replay output; reads shared frozen input caches from the repository outputs directory",
    )
    args = p.parse_args()
    OUT = args.out.resolve()
    if args.stage == "acquire":
        acquire_sources()
    elif args.stage == "register":
        verify_design()
        print(DESIGN_SHA)
    elif args.stage == "prepare":
        prepare()
    else:
        evaluate()
