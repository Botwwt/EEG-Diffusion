"""Paired independent scoring of three R80 image conditions and the previous R80 named model."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from compare_official_angle_clip import read_scores
from fit_dual_r80 import TE as ATM_TEST_FILE
from fit_dual_r80 import TR as ATM_TRAIN_FILE
from render_official_repeat_images import ADAPTER_SHA, DICT_SHA, TURBO_SHA, inputs
from score_official_named_factorial import NAMED, PUBLIC, contrast, prior_dino
from score_official_repeat_images import (
    BOOTSTRAP_SEED,
    FILES,
    MANIFEST_SHA,
    ORIGINALS,
    ROWS,
    encoder,
    gallery_metrics,
    generated,
    metrics,
    read,
    ref_features,
    references,
    require,
    sha,
    thumbnail,
)

REPO = Path(__file__).resolve().parents[3]
SEMANTICS = dict(zip(NAMED, ("atm_only_r80", "dual_r80", "wrong_dual_r80"), strict=True))
PREVIOUS = "physical_global_r80"
PROTOCOL = "da220cc643eb979ac249d7b1360192adce8a1ac88531363b99999304aa74959e"
RENDERER = "880b25ec272f26753c088f86c8e7c6ad687782d261ca189033f18090b8f71681"
PACKET = "dd07fc687c8904f822656138644cb5ab564fb361af41764d994dc1766d3338e5"
ATM_TRAIN = "0f6dc610c68f45a0e6c89ab5d5a90d52d443d45fe19ffab90925e0aa269f4233"
ATM_TEST = "1878b71916e0ff2cec8639198e005f1646f90fcf5c0f86cf488dd472e2f715df"


def check_case(root, design, dictionary, arms):
    packet = root / "render_packet.npz"
    settings = read(root / "render/settings.json")
    registered = read(design)
    require(registered["arms"] == list(NAMED), "Expected the three named generator conditions")
    expected, seeds, _ = inputs(
        SimpleNamespace(packet=packet, design=design, dictionary=dictionary)
    )
    require(
        settings["rows"] == ROWS.tolist()
        and settings["seeds"] == seeds
        and settings["packet_sha256"] == sha(packet) == registered["packet_sha256"]
        and settings["design_sha256"] == sha(design)
        and all(
            set(settings[k]) == set(NAMED)
            for k in ("jpeg_sha256", "arm_receipts", "prompts", "eeg_rows")
        ),
        "Prompt, arm, packet, row or seed registration differs",
    )
    for key in arms:
        item = root / "render" / (key + ".json")
        meta = read(item)
        eeg = expected[key][2].tolist()
        prompts = expected[key][1]
        require(
            sha(item) == settings["arm_receipts"][key]
            and meta["arm"] == key
            and meta["packet_sha256"] == settings["packet_sha256"]
            and meta["design_sha256"] == sha(design)
            and meta["eeg_rows"] == settings["eeg_rows"][key] == eeg
            and meta["prompts"] == settings["prompts"][key] == prompts
            and meta["seeds"] == settings["seeds"]
            and meta["adapter_sha256"] == ADAPTER_SHA
            and meta["turbo_sha256"] == TURBO_SHA
            and meta["driver_sha256"] == RENDERER
            and meta["batch"] == 8,
            "Render arm metadata differs from the frozen EEG inputs",
        )
        paths = root / "render" / key
        require(
            {p.name for p in paths.glob("*.jpg")} == {f"{i:03}.jpg" for i in ROWS}
            and settings["jpeg_sha256"][key]
            == meta["jpeg_sha256"]
            == [sha(paths / f"{i:03}.jpg") for i in ROWS],
            "Missing or changed generated JPEG",
        )
    return settings


def verify_registration(a):
    new = a.case
    prev = a.previous
    design = a.design
    if a.design_sha256:
        require(sha(design) == a.design_sha256.lower(), "Independent design digest differs")
    reg = read(design)
    require(
        sha(PUBLIC / "dual_r80_image_protocol.json") == PROTOCOL,
        "Original three-model image protocol changed",
    )
    receipt = new / "render_packet.json"
    require(
        reg["export_receipt_sha256"] == sha(receipt)
        and reg["physical_trials_per_photo"] == 80
        and reg["participant"] == "sub-04",
        "Private export receipt, participant or physical budget differs",
    )
    source = read(receipt)
    protocol = read(PUBLIC / "dual_r80_image_protocol.json")
    training = read(new / "fit_design.json")
    assets = training["sources"]
    historical = read(PUBLIC / "dual_r80_fit_packet.json")
    require(
        {k: v for k, v in historical.items() if k != "source_sha256"}
        == {k: v for k, v in source.items() if k != "source_sha256"}
        and training == read(PUBLIC / "dual_r80_fit_design.json"),
        "Decoder inputs, coefficients or model packet differ from the registered fit",
    )
    require(
        reg["image_protocol_sha256"] == PROTOCOL
        and source["image_protocol_sha256"] == PROTOCOL
        and source["same_observations_per_image"] == 80
        and source["data_sha256"] == assets
        and source["aliases"]
        == training["aliases"]
        == {"r4": "ATM_only_R80", "r80": "dual_R80", "wrong": "wrong_dual_R80"}
        and assets["train_lfs_sha256"] == ATM_TRAIN == sha(ATM_TRAIN_FILE)
        and assets["test_lfs_sha256"] == ATM_TEST == sha(ATM_TEST_FILE)
        and reg["historical_names"] == protocol["condition_format"]
        and reg["fit_design_sha256"] == source["fit_design_sha256"] == sha(new / "fit_design.json")
        and reg["packet_sha256"]
        == source["packet_sha256"]
        == sha(new / "render_packet.npz")
        == PACKET
        and reg["baseline_packet_sha256"] == sha(prev / "render_packet.npz"),
        "Source fit, exported vectors, image protocol or matched baseline differs",
    )


def verify(a):
    verify_registration(a)
    new, prev, design = a.case, a.previous, a.design
    native_design = PUBLIC / "official_global_named_images_design.json"
    dictionary = a.source / "dictionary.csv"
    require(sha(dictionary) == DICT_SHA, "Private EEG scoring requires the known dictionary")
    old = check_case(prev, native_design, dictionary, ["r80"])
    cfg = check_case(new, design, dictionary, NAMED)
    require(
        cfg["seeds"] == old["seeds"] and cfg["packet_sha256"] != old["packet_sha256"],
        "Expected a newly derived packet on the same original recipients",
    )
    donor = np.roll(ROWS, 67)
    matched = old["prompts"]["r80"]
    require(
        cfg["prompts"]["r4"] == cfg["prompts"]["r80"] == matched
        and cfg["prompts"]["wrong"] == [matched[i] for i in donor],
        "Object names changed across models or the wrong control",
    )
    with (
        np.load(new / "render_packet.npz", allow_pickle=False) as ins,
        np.load(prev / "render_packet.npz", allow_pickle=False) as source,
    ):
        fused = ins["r4"].astype(np.float64) + source["r80"].astype(np.float64)
        fused /= np.linalg.norm(fused, axis=1, keepdims=True)
        require(
            np.array_equal(ins["r4_words"], ins["r80_words"])
            and np.array_equal(ins["r80_words"], source["r80_words"])
            and np.array_equal(ins["donor"], source["donor"])
            and not np.array_equal(ins["r4"], ins["r80"]),
            "Private model inputs, words or donor differ from the intended comparison",
        )
        require(
            np.allclose(fused, ins["r80"], rtol=0, atol=3e-7),
            "Dual condition differs from the equal-weight fusion",
        )
    return cfg, old


def gallery(sheet, tiles, ordinals, columns):
    from PIL import Image, ImageDraw, ImageFont

    names, titles = zip(*columns, strict=True)
    size = 160
    image = Image.new("RGB", (size * len(names), 25 + len(ordinals) * (size + 18)), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13)
    for j, title in enumerate(titles):
        draw.text((size * j + 4, 4), title, font=font, fill="black")
    for k, i in enumerate(ordinals):
        y = 25 + k * (size + 18)
        draw.text((4, y), f"Official row {i:03d}", font=font, fill="black")
        for j, name in enumerate(names):
            image.paste(
                tiles[name][k].resize((size, size), Image.Resampling.LANCZOS), (size * j, y + 18)
            )
    image.save(sheet, "JPEG", quality=92, subsampling=0)


def dino(a, cfg, old):
    originals = references(a.source)
    encode = encoder(a)
    bank = ref_features(a, encode, originals)
    require(gallery_metrics(bank, bank)["dino_top1"].sum() == 200, "Original self-test failed")
    refs = [thumbnail(im) for im in originals]
    scores = {}
    ordinals = list(range(0, 200, 20))
    tiles = {"Original": [originals[i] for i in ordinals]}
    saved = prior_dino(a.previous, old, PUBLIC / "official_global_named_images.json", ["r80"])[
        "r80"
    ]
    for arm, (case, key, settings) in {
        **{v: (a.case, k, cfg) for k, v in SEMANTICS.items()},
        PREVIOUS: (a.previous, "r80", old),
    }.items():
        images = generated(case / "render", key, settings["jpeg_sha256"][key])
        tiles[arm] = [images[i] for i in ordinals]
        score = gallery_metrics(encode(images), bank)
        pixels = np.array(
            [
                metrics(thumbnail(im), reference)[1:]
                for im, reference in zip(images, refs, strict=True)
            ]
        )
        score["rgb_rmse32"], score["luma_pearson32"] = pixels.T
        scores[arm] = score
    require(
        set(saved) == set(scores[PREVIOUS])
        and all(
            np.allclose(v, scores[PREVIOUS][k], rtol=1e-6, atol=1e-6) for k, v in saved.items()
        ),
        "Previous physical-only R80 row scores did not reproduce",
    )
    columns = [
        (PREVIOUS, "Physical R80"),
        ("atm_only_r80", "ATM R80"),
        ("dual_r80", "Dual R80"),
        ("wrong_dual_r80", "Wrong dual R80"),
    ]
    sheet = a.out / "fixed_gallery.jpg"
    gallery(sheet, tiles, ordinals, [("Original", "Original"), *columns])
    generated_sheet = a.out / "official_dual_r80.jpg"
    gallery(generated_sheet, tiles, ordinals, columns)
    if a.publish:
        (PUBLIC.parents[1] / "gallery" / generated_sheet.name).write_bytes(
            generated_sheet.read_bytes()
        )
    return scores, dict(
        fixed_gallery_sha256=sha(sheet),
        generated_only_gallery_sha256=sha(generated_sheet),
        gallery_rows=ordinals,
        dino_weights=FILES,
        official_originals_sha256=ORIGINALS[1],
        reference_manifest_sha256=MANIFEST_SHA,
        previous_summary_sha256=sha(PUBLIC / "official_global_named_images.json"),
    )


def clip(a):
    scores, original, receipt = read_scores(a.case, a.case / "clip/summary.json", NAMED)
    old, previous, old_receipt = read_scores(
        a.previous, PUBLIC / "official_global_named_clip.json", ["r80"]
    )
    require(
        all(np.array_equal(x, y) for x, y in zip(original, previous, strict=True)),
        "Original CLIP galleries or dictionary changed",
    )
    metadata = read(a.case / "clip/summary.json")["provenance"]
    prior = read(a.previous / "clip/summary.json")["provenance"]
    for k in (
        "original_zip_sha256",
        "manifest_sha256",
        "dictionary_sha256",
        "model_sha256",
        "vocab_sha256",
        "merges_sha256",
        "archive_clip_implementation_sha256",
    ):
        require(metadata[k] == prior[k], f"CLIP scoring environment differs in {k}")
    with (a.source / "evaluation/originals_manifest.csv").open(encoding="utf8", newline="") as f:
        truth = [r["unique_id"] for r in csv.DictReader(f)]
    require(original[2].tolist() == truth, "Wrong official CLIP photo order")
    refs = read(a.case / "clip/summary.json")["summary"]["original"]
    return {**{v: scores[k] for k, v in SEMANTICS.items()}, PREVIOUS: old["r80"]}, dict(
        independent_clip_sha256=metadata["model_sha256"],
        private_scoring_receipt=receipt,
        previous_scoring_receipt=old_receipt,
        original_image_reference={
            k: {"top1_credit": refs[k]["top1_credit"], "top5_credit": refs[k]["top5_credit"]}
            for k in ("full1854", "restricted200", "photo200")
        },
    )


def result(a, scores, receipts):
    generated = ("dual_r80", "atm_only_r80", "wrong_dual_r80", PREVIOUS)
    pairs = [
        ("dual_r80", "atm_only_r80"),
        ("dual_r80", PREVIOUS),
        ("dual_r80", "wrong_dual_r80"),
        ("atm_only_r80", PREVIOUS),
    ]
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(200, size=(10000, 200))
    observed = {}
    for arm in generated:
        observed[arm] = {}
        for name, v in scores[arm].items():
            require(v.shape == (200,) and np.isfinite(v).all(), "Missing paired scientific result")
            values = dict(
                mean=float(v.mean()), ci95=np.quantile(v[draws].mean(1), [0.025, 0.975]).tolist()
            )
            if name.endswith(("top1", "top5")):
                values["hits"] = float(v.sum())
            observed[arm][name] = values
    raw = a.out / (a.stage + "_rows.npz")
    np.savez(
        raw,
        rows=ROWS,
        donor_rows=np.roll(ROWS, 67),
        **{f"{arm}_{k}": v for arm, group in scores.items() for k, v in group.items()},
    )
    output = dict(
        participant="sub-04",
        physical_presentations_per_image=80,
        distinct_photos=200,
        distinct_objects=200,
        interpretation="All models use the same participant and same 80 EEG presentations. The decoder supplies the same object-word prompt for each matched recipient. The wrong dual condition shifts both the word and the continuous direction together. One original per object makes photo and object identity inseparable in retrieval.",
        models=dict(
            atm_only_r80="Released ATM visual readout plus the frozen physical decoder's word.",
            dual_r80="Fused ATM and physical readouts plus the same frozen word.",
            wrong_dual_r80="Exchanged complete dual condition from another row, recipient seed retained.",
            physical_global_r80="Previous global-angle physical readout plus the same word.",
        ),
        paired_uncertainty="Fixed person and 200 paired stimulus IDs, ten thousand percentile bootstrap draws; intervals are marginal. CLIP comparisons also use the original two-sided paired swap procedure.",
        observed=observed,
        paired={
            left + "_minus_" + right: contrast(scores[left], scores[right], a.stage)
            for left, right in pairs
        },
        provenance=dict(
            **receipts,
            private_render_settings_sha256=sha(a.case / "render/settings.json"),
            previous_settings_sha256=sha(a.previous / "render/settings.json"),
            image_protocol_sha256=PROTOCOL,
            released_atm_train_sha256=ATM_TRAIN,
            released_atm_test_sha256=ATM_TEST,
            fitted_decoder_design_sha256=sha(a.case / "fit_design.json"),
            frozen_render_design_sha256=sha(a.design),
            condition_export_receipt_sha256=sha(a.case / "render_packet.json"),
            packet_sha256=sha(a.case / "render_packet.npz"),
            per_row_sha256=sha(raw),
            scoring_code_sha256=sha(Path(__file__)),
        ),
    )
    path = a.out / (a.stage + "_paired.json")
    payload = (json.dumps(output, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(payload)
    if a.publish:
        name = (
            "official_dual_r80_images.json"
            if a.stage == "dino"
            else "official_dual_r80_clip_paired.json"
        )
        (PUBLIC / name).write_bytes(payload)
    print(
        a.stage,
        [
            (arm, {k: v["hits"] for k, v in obs.items() if "hits" in v})
            for arm, obs in observed.items()
        ],
        path,
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("dino", "clip"))
    p.add_argument(
        "--root",
        type=Path,
        default=REPO / "outputs/official-repeat-object/sub-04",
        help="Participant directory containing dual-r80 and global-named",
    )
    p.add_argument(
        "--previous",
        type=Path,
        help="Previous registered global-named directory when replaying in a new output root",
    )
    p.add_argument(
        "--publish",
        action="store_true",
        help="Write only the aggregate evidence to the repository results directory",
    )
    p.add_argument("--source", type=Path, default=REPO / "outputs/official-object-readout/sub-03")
    p.add_argument("--design", type=Path)
    p.add_argument(
        "--design-sha256",
        help="Optionally pin an exact historical render receipt; defaults to verifying the current registration, frozen packet and protocol",
    )
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--model", type=Path, default=REPO / "outputs/photo-structure/dino/facebook-dinov2-small"
    )
    p.add_argument("--batch-size", type=int, choices=(16,), default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.case = args.root / "dual-r80"
    args.previous = args.previous or args.root / "global-named"
    args.design = args.design or args.case / "render_design.json"
    args.out = args.out or args.case / "independent_scores"
    args.out.mkdir(exist_ok=True, parents=True)
    cfg, old = verify(args)
    measures, source = dino(args, cfg, old) if args.stage == "dino" else clip(args)
    result(args, measures, source)
