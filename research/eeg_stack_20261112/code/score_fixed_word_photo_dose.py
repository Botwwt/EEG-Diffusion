"""Independently score the saved 1/3/9-observer JPEGs on the two fixed-word panels."""

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_fixed_word_photo_dose import (
    GROUPS,
    LOCK,
    PANELS,
    PHOTO,
    PUBLIC,
    REF,
    ROOT,
    SOURCE,
    sign_hit,
    summarize,
)
from fit_shared_photo_eeg import require
from PIL import Image
from readout_external_images import MODEL_SHA, h14_batch, h14_pixels, load_h14
from refine_registered_color import metrics, thumbnail
from register_fresh_photo_panel import freeze, read
from render_fixed_word_photo_dose import packet
from render_official_words import crop512
from render_within_photo_ip import ADAPTER_SHA, ARMS, sha
from score_object_panel import score_object_panel
from score_within_photo_dino import FILES, REV, encoder, reference

JPEG_LOCK = PUBLIC / "fixed_word_photo_dose_jpeg_design.json"


def score_design(root):
    return dict(
        study_design_sha256=sha(LOCK),
        packets_design_sha256=sha(root / "render_inputs/design.json"),
        protocol="Score every saved 512-square RGB JPEG in the six predeclared cases. There are 240 unique test stimuli shared by 1, 3 and 9 observers. Word predictions and eight supports come from the same frozen sub-04 readout. The earlier H14 condition-space outcomes were known before specifying this pixel evaluation.",
        primary=[
            "DINOv2-S/14 final normalized CLS, raw cosine to ten originals of the same true object, top-one retrieval on all 240 stimuli.",
            "DINOv2-S/14 summed matched versus exchanged cosine for the two held-out original photos of the same true object and fold when both sub-04 object words are correct. One joint decision per pair; 33 registered and 39 replicate pairs.",
        ],
        comparisons="Compare own JPEGs across nested acquisition budgets and own against uniform and same-word/same-fold wrong at each budget. Cluster-bootstrap hit differences over true objects jointly within panel, 10000 draws seed 261125. The same individuals and pictures contribute to all budgets. Check every shared uniform JPEG SHA.",
        secondary=[
            "Ten-way retrieval on correct-word rows; assignment for reciprocal same-object donors; independent two-alternative choices for the strict pair members.",
            "Repeat the photo retrieval metrics on reopened generated JPEGs and held-out originals using Laion H/14, the visual representation used in the EEG supervision.",
            "Original full-resolution photo center crop to 512; compare generated and original 32-square BOX RGB thumbnails for RMSE and spatial luminance Pearson.",
        ],
        extra="Separate planned guided-sampling diagnostic takes the identical nine-person packet on the same row seeds and applies eps_uniform + 2*(eps_condition-eps_uniform). Require the same condition SHA and identical uniform JPEGs. Analyze in a separate supplement.",
        dino_revision=REV,
        dino_assets=FILES,
        h14_weights_sha256=MODEL_SHA,
    )


def jpeg(raw, generated):
    with Image.open(io.BytesIO(raw)) as im:
        if generated:
            require(
                im.format == "JPEG" and im.mode == "RGB" and im.size == (512, 512),
                "Expected saved generated RGB JPEG",
            )
            return im.copy()
        return crop512(im.convert("RGB"))


def registry(dose, panel, root, base):
    extra = dose == "cohort-9-extra"
    folder = root / "renders" / panel / ("cohort-9" if extra else dose)
    cfg = read(folder / "settings.json")
    reference_subject = "cohort-9" if extra else dose
    rows, folds, words, donors, support, _, _, signature = packet(base, reference_subject, panel)
    design = read(root / "design.json")
    if extra:
        require(
            signature == design["packets_sha256"][reference_subject + "/" + panel],
            "Alternate sampler altered the frozen nine-person packet",
        )
        location = root / "conditions" / reference_subject / panel
        records = [p for p in location.glob("*.npz") if sha(p) == signature]
        require(len(records) == 1, "Missing extra comparison's independent packet")
        with np.load(records[0], allow_pickle=False) as f:
            require(
                all(
                    np.array_equal(f[k], v)
                    for k, v in (
                        ("rows", rows),
                        ("folds", folds),
                        ("predicted_words", words),
                        ("donor_rows", donors),
                        ("support_rows", support),
                    )
                ),
                "Extra comparison changed the frozen words, rows, donors or supports",
            )
    require(
        cfg["subject"] == reference_subject
        and cfg["panel"] == panel
        and all(
            np.array_equal(cfg[k], v)
            for k, v in (
                ("rows", rows),
                ("folds", folds),
                ("predicted_words", words),
                ("donor_rows", donors),
                ("support_rows", support),
                ("seeds", 923761 + rows),
            )
        )
        and cfg["packet_sha256"] == signature
        and cfg["design_sha256"] == sha(root / "design.json")
        and cfg["radius_sha256"] == sha(root / "radius.json")
        and cfg["adapter_sha256"] == ADAPTER_SHA
        and cfg["turbo_sha256"]
        == "e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26"
        and cfg["batch_size"] == 8
        and set(cfg["jpeg_sha256"]) == set(ARMS),
        "Generated settings differ from the independently frozen conditions",
    )
    for arm in ARMS:
        require(
            len(cfg["jpeg_sha256"][arm]) == 120
            and {f.name for f in (folder / "images" / arm).glob("*.jpg")}
            == {f"{i:03}.jpg" for i in range(120)},
            "Missing or extra generated files",
        )
    return folder, cfg


def measure(refs, gen, facts):
    rows, truth, words, folds, donors = facts
    pc = truth == words
    pairs = []
    for k in np.unique(truth):
        for f in range(5):
            q = np.flatnonzero((truth == k) & (folds == f))
            require(len(q) == 2, "Incomplete true-object fold")
            if all(pc[q]):
                pairs.append(q)
    i, j = np.asarray(pairs).T
    reciprocal = (donors[i] == rows[j]) & (donors[j] == rows[i])
    result = {}
    for field in ("dino", "h14"):
        sim = gen[field].astype(float) @ refs[field].astype(float).T
        shots = score_object_panel(sim, rows, truth)["per_photo"]
        hit = np.array([s["intra10_top1"] for s in shots])
        result[field + "_ten"] = (rows[:, None], truth, hit)
        result[field + "_ten_correct"] = (rows[pc, None], truth[pc], hit[pc])
        left = sim[i, i] - sim[i, j]
        right = sim[j, j] - sim[j, i]
        pairrows = np.column_stack((rows[i], rows[j]))
        ph = sign_hit((left + right) / 2)
        result[field + "_strict_pair"] = (pairrows, truth[i], ph)
        result[field + "_strict_reciprocal"] = (
            pairrows[reciprocal],
            truth[i[reciprocal]],
            ph[reciprocal],
        )
        result[field + "_pair_single_query"] = (
            rows[np.asarray(pairs).ravel(), None],
            np.repeat(truth[i], 2),
            np.column_stack((sign_hit(left), sign_hit(right))).ravel(),
        )
    pix = np.asarray([metrics(g, t)[1:] for g, t in zip(gen["rgb"], refs["rgb"], strict=True)])
    for col, k in enumerate(("rgb_rmse32", "luma_pearson32")):
        result[k] = (rows[:, None], truth, pix[:, col])
        result[k + "_correct"] = (rows[pc, None], truth[pc], pix[pc, col])
    return result


def extract_features(images, encoders):
    return dict(
        dino=encoders[0](images),
        h14=encoders[1](images),
        rgb=np.stack([thumbnail(im) for im in images]),
    )


def run(a):
    import torch
    import transformers

    require(read(JPEG_LOCK) == score_design(a.root), "Independent JPEG score design differs")
    specs = {k: a.root / "render_inputs" for k in GROUPS}
    if a.extra_root:
        specs["cohort-9-extra"] = a.extra_root
    if a.extra_root and a.release == PUBLIC / "fixed_word_photo_dose_jpegs.json":
        raise ValueError("Use a distinct --release for the post-registration comparison")
    cases = {
        p: {k: registry(k, p, specs[k], a.root / "render_inputs") for k in specs} for p in PANELS
    }
    # Read references only after every requested generation has its frozen completion receipt.
    table = reference(a)
    dino = encoder(a)
    h, mean, std = load_h14(a.h14, a.device)

    def h14(images):
        return np.concatenate(
            [
                h14_batch(h, mean, std, [h14_pixels(im) for im in images[i : i + a.batch_size]])
                for i in range(0, len(images), a.batch_size)
            ]
        )

    encoders = (dino, h14)
    panel_data = {}
    provenance = {}
    with ZipFile(a.originals) as zipfile:
        for panel, groups in cases.items():
            reg = read(REF / (panel + "_ablation.json"))
            rows, truth, folds = (
                np.array(reg["rows"]),
                np.array(reg["words"])[reg["truth"]],
                np.array(reg["folds"]),
            )
            raw = []
            for rid, word in zip(rows, truth, strict=True):
                m = table[rid]
                require(m["unique_id"] == word, "Wrong original reference")
                raw.append(
                    zipfile.read(f"training_images/{m['official_folder']}/{m['stimulus_stem']}.jpg")
                )
            refhash = [hashlib.sha256(x).hexdigest() for x in raw]
            refs = extract_features([jpeg(x, False) for x in raw], encoders)
            first = groups["single-1"][1]
            facts = (
                rows,
                truth,
                np.array(first["predicted_words"]),
                folds,
                np.array(first["donor_rows"]),
            )
            output = {}
            provenance[panel] = dict(
                original_ordered_jpeg_sha256=hashlib.sha256("".join(refhash).encode()).hexdigest(),
                cases={},
            )
            feature_cache = {}
            for k, (folder, cfg) in groups.items():
                require(
                    all(
                        cfg[f] == first[f]
                        for f in (
                            "rows",
                            "folds",
                            "predicted_words",
                            "donor_rows",
                            "seeds",
                            "support_rows",
                            "dictionary_sha256",
                            "batch_size",
                            "turbo_sha256",
                            "adapter_sha256",
                        )
                    ),
                    "Rows, words, supports, seeds or model changed between arms",
                )
                require(
                    cfg["jpeg_sha256"]["uniform"] == first["jpeg_sha256"]["uniform"],
                    "Equal inputs and row seeds produced different uniform JPEGs",
                )
                if k == "cohort-9-extra":
                    ordinary_folder, ordinary = groups["cohort-9"]
                    require(
                        cfg["driver_sha256"] != ordinary["driver_sha256"]
                        and cfg["guidance_factor"] == 2.0
                        and cfg["uniform_parity_hits"] == 120
                        and cfg["guidance_design_sha256"]
                        == "74d41ca021a7ba9bbbd9abb9dc2ea148b0616af84bfab370256d7314dfbf8bc0"
                        and cfg["ordinary_settings_sha256"]
                        == sha(ordinary_folder / "settings.json")
                        and cfg["ordinary_driver_sha256"] == ordinary["driver_sha256"],
                        "Guidance strength, design, source sampler, uniform parity or reference receipt changed",
                    )
                provenance[panel]["cases"][k] = dict(
                    settings_sha256=sha(folder / "settings.json"),
                    packet_sha256=cfg["packet_sha256"],
                )
                for arm in ARMS:
                    hashes = tuple(cfg["jpeg_sha256"][arm])
                    saved = [
                        (folder / "images" / arm / f"{i:03}.jpg").read_bytes() for i in range(120)
                    ]
                    require(
                        all(
                            hashlib.sha256(raw).hexdigest() == sh
                            for raw, sh in zip(saved, hashes, strict=True)
                        ),
                        "Generated JPEG changed after freezing",
                    )
                    if hashes not in feature_cache:
                        feature_cache[hashes] = extract_features(
                            [jpeg(x, True) for x in saved], encoders
                        )
                    data = measure(refs, feature_cache[hashes], facts)
                    for metric, (rr, labels, value) in data.items():
                        if metric not in output:
                            output[metric] = [
                                dict(panel=panel, object=str(label), rows=row.tolist(), scores={})
                                for label, row in zip(labels, rr, strict=True)
                            ]
                        for rec, label, row, v in zip(
                            output[metric], labels, rr, value, strict=True
                        ):
                            require(
                                rec["object"] == label and rec["rows"] == row.tolist(),
                                "Mismatched scoring denominator",
                            )
                            rec["scores"][k + "/" + arm] = float(v)
                print(panel, k, "verified and measured", flush=True)
            panel_data[panel] = output
    pairs = [(k + "/own", k + "/" + c) for k in specs for c in ("uniform", "wrong")]
    pairs.extend(
        (k + "/own", b + "/own")
        for k, b in (("cohort-9", "single-1"), ("cohort-9", "cohort-3"), ("cohort-3", "single-1"))
    )
    if a.extra_root:
        pairs.append(("cohort-9-extra/own", "cohort-9/own"))

    def summed(panel):
        return {m: summarize(x, pairs, m.startswith(("rgb_", "luma_"))) for m, x in panel.items()}

    combined = {m: [row for p in PANELS for row in panel_data[p][m]] for m in panel_data[PANELS[0]]}
    rawpath = a.root / ("jpeg_rows_extra.json" if a.extra_root else "jpeg_rows.json")
    freeze(rawpath, panel_data)
    freeze(
        a.release,
        dict(
            study_design_sha256=sha(LOCK),
            jpeg_score_design_sha256=sha(JPEG_LOCK),
            render_inputs_sha256=sha(a.root / "render_inputs/design.json"),
            cases=provenance,
            n_generated_jpegs=2 * len(specs) * 3 * 120,
            n_unique_originals=240,
            paired_over="Same stimulus or strict pair; fixed people; object-cluster resampling within panels",
            primary="Frozen DINOv2-S/14 raw CLS cosine: independent ten-way retrieval among same-object originals on every query; strict summed-cosine two-photo assignment within true class/fold, both previously decoded sub-04 words correct.",
            secondary="32-square BOX RGB RMSE (lower better), spatial luminance Pearson (higher better). H14 uses the visual target encoder on held-out decoded JPEGs and identically center-cropped original RGB, with separately verified weights; it is secondary given its representation was used to train the EEG readout.",
            dino_sha256=FILES,
            dino_revision=REV,
            h14_sha256=MODEL_SHA,
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            device=a.device,
            batch_size=a.batch_size,
            scorer_sha256=sha(Path(__file__)),
            per_row_sha256=sha(rawpath),
            panels={p: summed(x) for p, x in panel_data.items()},
            combined=summed(combined),
        ),
    )
    for m in ("dino_ten", "dino_strict_pair", "h14_ten", "rgb_rmse32", "luma_pearson32"):
        d = read(a.release)["combined"][m]
        print(m, d["n"], {k: v.get("hits", v.get("mean")) for k, v in d["arms"].items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in dict(
        root=ROOT,
        model=PHOTO / "dino/facebook-dinov2-small",
        h14=Path("outputs/external-object-anchors/assets"),
        originals=PHOTO / "training_images.zip",
        manifest=SOURCE / "atm_train_rows.csv",
        release=PUBLIC / "fixed_word_photo_dose_jpegs.json",
    ).items():
        p.add_argument("--" + k, type=Path, default=v)
    p.add_argument("--extra-root", type=Path)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--register",
        action="store_true",
        help="Freeze the JPEG scoring protocol without opening generated images",
    )
    args = p.parse_args()
    if args.register:
        freeze(JPEG_LOCK, score_design(args.root))
        print(JPEG_LOCK, sha(JPEG_LOCK))
    else:
        run(args)
