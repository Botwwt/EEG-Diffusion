"""Score a developmental completion of the native/calibrated by named/fixed-prompt comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from compare_official_angle_clip import read_scores
from PIL import Image, ImageDraw, ImageFont
from render_official_repeat_images import TURBO_SHA
from render_within_photo_ip import ADAPTER_SHA
from score_official_objects import paired_arms
from score_official_repeat_images import (
    ARMS,
    BOOTSTRAP_SEED,
    FILES,
    GALLERY_ROWS,
    MANIFEST_SHA,
    ORIGINALS,
    PUBLIC,
    ROWS,
    VISUAL,
    decoded,
    encoder,
    fixed_gallery,
    gallery_metrics,
    generated,
    metrics,
    read,
    ref_features,
    references,
    require,
    sha,
    summary,
    thumbnail,
    verify_render,
)

NAMED = ARMS[:3]
NEW_NAME = "official_global_named"
HISTORY = "Third stage, added after examining the first six arms and the three calibrated-angle fixed-prompt arms on these same 200 official photographs. Conditional contrasts for this participant and this fixed gallery."


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf8"))


def verify_case(a):
    native, native_root, donors = verify_render(
        a.root / "render", a.source, PUBLIC / "official_repeat_images_design.json", ARMS
    )
    generic, generic_root, other = verify_render(
        a.root / "global-angle/render",
        a.source,
        PUBLIC / "official_global_angle_images_design.json",
        VISUAL,
    )
    case = a.root / "global-named"
    out = case / "render"
    settings = read(out / "settings.json")
    registration = PUBLIC / (NEW_NAME + "_images_design.json")
    design = read(registration)
    packet = case / "render_packet.npz"
    require(
        tuple(design["arms"]) == NAMED
        and design["packet_sha256"] == sha(packet)
        and generic["packet_sha256"] == settings["packet_sha256"] == sha(packet)
        and native["packet_sha256"] != settings["packet_sha256"]
        and settings["design_sha256"] == sha(registration)
        and settings["rows"] == native["rows"] == generic["rows"] == ROWS.tolist()
        and settings["seeds"] == native["seeds"] == generic["seeds"]
        and np.array_equal(donors, other)
        and set(settings["jpeg_sha256"]) == set(NAMED),
        "Calibrated named arm uses different recipients, relation to the reference packets, design, seeds or arms",
    )
    for arm in NAMED:
        for cfg, folder, key in (
            (settings, out, arm),
            (native, native_root, arm),
            (generic, generic_root, arm + "_visual"),
        ):
            receipt = folder / (key + ".json")
            rec = read(receipt)
            require(
                sha(receipt) == cfg["arm_receipts"][key]
                and rec["batch"] == 8
                and rec["adapter_sha256"] == ADAPTER_SHA
                and rec["turbo_sha256"] == TURBO_SHA
                and rec["jpeg_sha256"] == cfg["jpeg_sha256"][key],
                "Changed generator, adapter, batch or per-arm JPEG receipt",
            )
        require(
            settings["eeg_rows"][arm]
            == native["eeg_rows"][arm]
            == generic["eeg_rows"][arm + "_visual"]
            and settings["prompts"][arm] == native["prompts"][arm]
            and len(settings["jpeg_sha256"][arm]) == 200
            and {p.name for p in (out / arm).glob("*.jpg")} == {f"{r:03d}.jpg" for r in ROWS},
            "Prompts, donor, recipient order or number of generated images differs",
        )
    return settings, ((native, native_root), (generic, generic_root)), donors


def contrast(left, right, kind):
    if kind == "clip":
        result = paired_arms(left, right)
    else:
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(200, size=(10000, 200))
        result = {}
        for k, v in left.items():
            diff = v - right[k]
            result[k] = dict(
                n=200,
                delta=float(diff.mean()),
                ci95=np.quantile(diff[draws].mean(1), [0.025, 0.975]).tolist(),
            )
    for k, v in left.items():
        if k.endswith(("top1", "top5")):
            result[k].update(
                left_credit=float(v.sum()),
                right_credit=float(right[k].sum()),
                rows_gaining_credit=int((v > right[k]).sum()),
                rows_losing_credit=int((v < right[k]).sum()),
            )
    return result


def cross_settings(named, prior_named, prior_generic, native_generic, kind):
    def diff(a, b):
        return {k: v - b[k] for k, v in a.items()}

    result = {}
    for arm in NAMED:
        generic = arm + "_visual"
        result[arm] = dict(
            calibrated_named_minus_native_named=contrast(named[arm], prior_named[arm], kind),
            calibrated_named_minus_calibrated_generic=contrast(
                named[arm], prior_generic[generic], kind
            ),
            change_in_naming_effect_after_calibration=contrast(
                diff(named[arm], prior_generic[generic]),
                diff(prior_named[arm], native_generic[generic]),
                kind,
            ),
        )
    return result


def prior_dino(case, expected, public, subset):
    directory = case / "independent_images"
    raw = directory / "per_row.npz"
    report = read(directory / "summary.json")
    require(
        report == read(public)
        and report["receipts"]["dino_files_sha256"] == FILES
        and report["receipts"]["original_archive_sha256"] == ORIGINALS[1]
        and report["receipts"]["rows_manifest_sha256"] == MANIFEST_SHA
        and report["receipts"]["settings_sha256"] == sha(case / "render/settings.json")
        and report["receipts"]["packet_sha256"] == expected["packet_sha256"]
        and sha(raw) == report["receipts"]["per_row_scores_sha256"],
        "Earlier DINO scores changed",
    )
    with np.load(raw, allow_pickle=False) as z:
        require(
            np.array_equal(z["rows"], ROWS) and np.array_equal(z["donor_rows"], np.roll(ROWS, 67)),
            "DINO rows differ",
        )
        return {arm: {k: z[arm + "_" + k] for k in report["observed"][arm]} for arm in subset}


def private_comparison(dest, panels, prior):
    cols = ("Original", "Native angle + word", "Calibrated, fixed text", "Calibrated + word")
    size = 176
    sheet = Image.new("RGB", (size * len(cols), 32 + len(GALLERY_ROWS) * (size + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, col in enumerate(cols):
        draw.text((j * size + 3, 7), col, fill="black", font=ImageFont.load_default(size=13))
    for i, r in enumerate(GALLERY_ROWS):
        y = 32 + i * (size + 20)
        draw.text((4, y + 2), f"Official row {r:03d}", fill="black")
        images = [panels["original"][r]]
        for (settings, root), arm in zip(prior, ("r80", "r80_visual"), strict=True):
            raw = (root / arm / f"{r:03d}.jpg").read_bytes()
            require(
                hashlib.sha256(raw).hexdigest() == settings["jpeg_sha256"][arm][r],
                "Comparison JPEG changed",
            )
            images.append(decoded(raw, True))
        images.append(panels["r80"][r])
        for j, im in enumerate(images):
            sheet.paste(im.resize((size, size), Image.Resampling.LANCZOS), (size * j, y + 20))
    sheet.save(dest / "gallery_r80_factorial.jpg", quality=92, subsampling=0)


def dino(a, settings, prior, donors):
    dest = a.root / "global-named/independent_images"
    dest.mkdir(parents=True, exist_ok=True)
    a.out = dest
    originals = references(a.source)
    encode = encoder(a)
    bank = ref_features(a, encode, originals)
    require(gallery_metrics(bank, bank)["dino_top1"].sum() == 200, "Original identity check failed")
    ref_pixels = [thumbnail(x) for x in originals]
    store = {"original": {r: originals[r] for r in GALLERY_ROWS}}
    scores = {}
    for arm in NAMED:
        imgs = generated(a.root / "global-named/render", arm, settings["jpeg_sha256"][arm])
        scores[arm] = gallery_metrics(encode(imgs), bank)
        pix = np.array(
            [metrics(thumbnail(x), t)[1:] for x, t in zip(imgs, ref_pixels, strict=True)]
        )
        scores[arm]["rgb_rmse32"], scores[arm]["luma_pearson32"] = pix.T
        store[arm] = {r: imgs[r] for r in GALLERY_ROWS}
    fixed_gallery(dest, store)
    private_comparison(dest, store, prior)
    raw = dest / "per_row.npz"
    np.savez(
        raw,
        rows=ROWS,
        donor_rows=donors,
        **{f"{a}_{k}": v for a, d in scores.items() for k, v in d.items()},
    )
    old_path, gen_path = (
        PUBLIC / n for n in ("official_repeat_images.json", "official_global_angle_images.json")
    )
    old = prior_dino(a.root, prior[0][0], old_path, ARMS)
    generic = prior_dino(a.root / "global-angle", prior[1][0], gen_path, VISUAL)
    receipts = dict(
        packet_sha256=settings["packet_sha256"],
        render_design_sha256=settings["design_sha256"],
        settings_sha256=sha(a.root / "global-named/render/settings.json"),
        original_archive_sha256=ORIGINALS[1],
        rows_manifest_sha256=MANIFEST_SHA,
        dino_files_sha256=FILES,
        native_summary_sha256=sha(old_path),
        generic_summary_sha256=sha(gen_path),
        per_row_scores_sha256=sha(raw),
        scorer_sha256=sha(Path(__file__)),
    )
    path = dest / "summary.json"
    summary(scores, path, receipts, {})
    report = read(path)
    report.pop("global_angle_minus_native_angle")
    report.update(
        history=HISTORY, across_settings=cross_settings(scores, old, generic, old, "dino")
    )
    write(path, report)
    write(PUBLIC / (NEW_NAME + "_images.json"), report)
    print(
        "DINO top-1", {arm: scores[arm]["dino_top1"].sum() for arm in NAMED}, sha(path), flush=True
    )


def clip(a):
    native, nref, nsha = read_scores(a.root, PUBLIC / "official_repeat_clip.json", ARMS)
    generic, gref, gsha = read_scores(
        a.root / "global-angle", PUBLIC / "official_global_angle_clip.json", VISUAL
    )
    case = a.root / "global-named"
    source = case / "clip/summary.json"
    named, ref, csha = read_scores(case, source, NAMED)
    require(
        all(
            np.array_equal(x, y) and np.array_equal(x, z)
            for x, y, z in zip(ref, nref, gref, strict=True)
        ),
        "The original-image score matrices or dictionary order changed across the three runs",
    )
    release = PUBLIC / (NEW_NAME + "_clip.json")
    require(
        not release.exists() or release.read_bytes() == source.read_bytes(),
        "Existing public CLIP summary differs",
    )
    release.write_bytes(source.read_bytes())
    out = PUBLIC / (NEW_NAME + "_clip_paired.json")
    write(
        out,
        dict(
            history=HISTORY,
            participant="sub-04",
            unique_photos=200,
            generated_named=600,
            source=dict(
                native=nsha,
                calibrated_generic=gsha,
                calibrated_named=csha,
                driver_sha256=sha(Path(__file__)),
            ),
            uncertainty="10,000 paired row bootstrap and two-sided sign flips, seed 20261125, fixed person and gallery; unadjusted.",
            across_settings=cross_settings(named, native, generic, native, "clip"),
        ),
    )
    print("CLIP paired results", sha(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("dino", "clip"))
    p.add_argument("--root", type=Path, default=Path("outputs/official-repeat-object/sub-04"))
    p.add_argument("--source", type=Path, default=Path("outputs/official-object-readout/sub-03"))
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    cfg, prior, donors = verify_case(args)
    if args.stage == "dino":
        dino(args, cfg, prior, donors)
    else:
        clip(args)
