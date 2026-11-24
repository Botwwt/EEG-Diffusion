"""Score all 24 images of sub-04 registered fold zero against its full 120-photo gallery."""

import argparse
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from refine_registered_color import KEYS, metrics
from render_official_words import crop512
from render_within_photo_ip import check, packet
from score_within_photo_dino import (
    FILES,
    REV,
    credit,
    encoder,
    originals,
    read,
    reference,
    save,
    sha,
)

ARMS = ("own", "uniform", "wrong", "old_four_step")
SUB = "sub-04"
PANEL = "registered"


def tiles(img):
    return np.asarray(img, dtype=np.float32).reshape(32, 16, 32, 16, 3).mean((1, 3)) / 255


def report(values, groups):
    v = np.asarray(values, dtype=float)
    labels = np.asarray(groups)
    unique = np.unique(labels)
    sums = np.array([v[labels == c].sum() for c in unique])
    ns = np.array([(labels == c).sum() for c in unique])
    draws = np.random.default_rng(260512).integers(len(unique), size=(10000, len(unique)))
    return dict(
        n=len(v),
        sum=float(v.sum()),
        mean=float(v.mean()),
        concept_cluster_ci95=np.quantile(
            sums[draws].sum(1) / ns[draws].sum(1), [0.025, 0.975]
        ).tolist(),
    )


def directly_compare(by_arm, labels):
    return dict(
        arms={a: report(v, labels) for a, v in by_arm.items()},
        own_minus={c: report(by_arm["own"] - by_arm[c], labels) for c in ARMS[1:]},
    )


def first_sheet(imgs, path):
    width = 192
    canvas = Image.new("RGB", (5 * width, width + 24), "white")
    text = ImageDraw.Draw(canvas)
    for i, (name, im) in enumerate(imgs):
        canvas.paste(im.resize((width, width), Image.Resampling.LANCZOS), (i * width, 24))
        text.text((i * width + 4, 5), name, fill="black")
    canvas.save(path, format="PNG")


def run(a):
    # Confirm the complete locked fold and source conditions before any target image is opened.
    baseline = a.h14_root / "renders" / PANEL / SUB
    pilot = a.renders
    sm, old = (read(d / "settings.json") for d in (pilot, baseline))
    full, folds, words, donors, support, _, _, signature = packet(a.h14_root, SUB, PANEL)
    reg = read(a.references / (PANEL + "_ablation.json"))
    truth = np.array(reg["words"])[reg["truth"]]
    ix = np.flatnonzero(folds == 0)
    check(
        np.array_equal(ix, np.arange(24))
        and len(np.unique(truth[ix])) == 12
        and np.all(np.unique(truth[ix], return_counts=True)[1] == 2),
        "The 24 pilot rows must be precisely the first registered fold, two queries per object",
    )
    for manifest, sel in ((sm, ix), (old, np.arange(120))):
        check(
            manifest["subject"] == SUB
            and manifest["panel"] == PANEL
            and manifest["packet_sha256"] == signature
            and manifest["design_sha256"] == sha(a.h14_root / "design.json")
            and all(
                np.array_equal(manifest[k], v[sel])
                for k, v in dict(
                    rows=full,
                    folds=folds,
                    predicted_words=words,
                    donor_rows=donors,
                    support_rows=support,
                    seeds=923761 + full,
                ).items()
            ),
            "Rows, words, seeds, donor, support or H/14 packet changed",
        )
    check(
        sm["spatial_sha256"] == sha(a.spatial)
        and sm["limit"] == 24
        and sm["adapter_sha256"] == old["adapter_sha256"]
        and sm["turbo_sha256"] == old["turbo_sha256"]
        and reg["rows"] == full.tolist(),
        "Spatial pilot and original generator are not aligned",
    )
    imgs = {}
    source_hashes = {}
    for arm in ARMS:
        fld, name, info = (baseline, "own", old) if arm == "old_four_step" else (pilot, arm, sm)
        imgs[arm] = []
        source_hashes[arm] = []
        for j in ix:
            raw = (fld / "images" / name / f"{j:03d}.jpg").read_bytes()
            h = hashlib.sha256(raw).hexdigest()
            check(h == info["jpeg_sha256"][name][j], "Generated image failed its prior checksum")
            source_hashes[arm].append(h)
            with Image.open(io.BytesIO(raw)) as im:
                check(
                    im.mode == "RGB" and im.size == (512, 512) and im.format == "JPEG",
                    "Invalid generated JPEG",
                )
                imgs[arm].append(im.copy())
    encode = encoder(a)
    table = reference(a)
    original_imgs = []
    with ZipFile(a.originals) as z:
        _, gallery, original_hash = originals(
            SimpleNamespace(**(vars(a) | dict(out=a.reference_cache))), encode, table, z, PANEL
        )
        for j in ix:
            rec = table[full[j]]
            raw = z.read(f"training_images/{rec['official_folder']}/{rec['stimulus_stem']}.jpg")
            check(
                hashlib.sha256(raw).hexdigest() == original_hash[j],
                "Original photo differs from scoring gallery",
            )
            with Image.open(io.BytesIO(raw)) as im:
                original_imgs.append(crop512(im.convert("RGB")))
    out = {}
    per_query = []
    per_pair = []
    cos = {
        arm: encode(images).astype(float) @ gallery.astype(float).T for arm, images in imgs.items()
    }
    values = {
        k: {arm: np.zeros(24) for arm in ARMS}
        for k in ("dino_intra10", "dino_rank10", "dino_target_cosine", *KEYS)
    }
    for i in ix:
        rec = dict(
            row=int(full[i]), ordinal=int(i), object=truth[i], predicted_word=words[i], arms={}
        )
        bank = truth == truth[i]
        check(bank.sum() == 10, "Invalid ten-image gallery")
        for arm in ARMS:
            v = cos[arm][i]
            matches = abs(v[bank] - v[bank].max()) <= 1e-12
            fields = dict(
                dino_intra10=float((abs(v[i] - v[bank].max()) <= 1e-12) / matches.sum()),
                dino_rank10=1 + int((v[bank] > v[i] + 1e-12).sum()),
                dino_target_cosine=float(v[i]),
                **dict(
                    zip(KEYS, metrics(tiles(imgs[arm][i]), tiles(original_imgs[i])), strict=True)
                ),
            )
            rec["arms"][arm] = fields
            for k, x in fields.items():
                values[k][arm][i] = x
        per_query.append(rec)
    for i in range(0, 24, 2):
        j = i + 1
        check(truth[i] == truth[j], "Registered rows do not align neighboring fold-0 object pairs")
        paired = {}
        independent = {}
        for arm, v in cos.items():
            m1, m2 = v[i, i] - v[i, j], v[j, j] - v[j, i]
            paired[arm] = credit((m1 + m2) / 2)
            independent[arm] = (credit(m1) + credit(m2)) / 2
        per_pair.append(
            dict(
                rows=[int(full[i]), int(full[j])],
                object=truth[i],
                both_words_correct=bool(words[i] == words[j] == truth[i]),
                paired=paired,
                independent=independent,
            )
        )
    for k in values:
        out[k] = directly_compare(values[k], truth[ix])
    for subset, chosen in (
        ("all_pairs", per_pair),
        ("strict_pairs", [r for r in per_pair if r["both_words_correct"]]),
    ):
        out[subset] = {
            k: directly_compare(
                {arm: np.array([r[k][arm] for r in chosen]) for arm in ARMS},
                [r["object"] for r in chosen],
            )
            for k in ("paired", "independent")
        }
    # Before reporting an experimental effect, replay the independently published four-step DINO endpoint.
    prev_file = a.reference_cache / PANEL / SUB / "per_query.json"
    prev = read(prev_file)
    check(
        all(
            r["row"] == p["row"]
            and r["arms"]["old_four_step"]["dino_intra10"] == p["intra10"]["own"]
            for r, p in zip(per_query, prev["images"][:24], strict=True)
        ),
        "Historical DINO ten-photo scores differ",
    )
    pair_hist = {tuple(r["rows"]): r["assignment"]["own"] for r in prev["pairs"] if r["fold"] == 0}
    check(
        pair_hist
        == {
            tuple(r["rows"]): r["paired"]["old_four_step"]
            for r in per_pair
            if r["both_words_correct"]
        },
        "Historical DINO pair scores differ",
    )
    a.out.mkdir(parents=True, exist_ok=True)
    first_sheet(
        [
            ("Original row " + str(full[0]), original_imgs[0]),
            ("Old four-step", imgs["old_four_step"][0]),
            ("EEG spatial", imgs["own"][0]),
            ("Support mean", imgs["uniform"][0]),
            ("Wrong spatial EEG", imgs["wrong"][0]),
        ],
        a.out / "fixed_first_row.png",
    )
    save(
        a.out / "per_query.json",
        dict(images=per_query, pairs=per_pair, generated_sha256=source_hashes),
    )
    save(
        a.out / "summary.json",
        dict(
            subject=SUB,
            panel=PANEL,
            fold=0,
            n_queries=24,
            original_gallery=120,
            primary="DINOv2-S/14 final CLS, cosine within-class ten-photo top-1 on all 24. Mean rank and target cosine, image RGB and luma are additional measurements.",
            comparison="All new arms have the same matched angular H/14 and two denoising steps. The older own H/14 images have four denoising steps from noise. Rows and per-row seeds match.",
            uncertainty="Percentile intervals resample whole classes 10,000 times; strict pairs require both predicted words correct. Pixel values in [0,1]; RGB RMSE/rank lower, correlation/cosine/top1 higher.",
            model=REV,
            score_script_sha256=sha(Path(__file__)),
            model_sha256=FILES["model.safetensors"],
            new_settings_sha256=sha(pilot / "settings.json"),
            old_settings_sha256=sha(baseline / "settings.json"),
            registration_sha256=sha(a.references / (PANEL + "_ablation.json")),
            historical_scores_sha256=sha(prev_file),
            per_query_sha256=sha(a.out / "per_query.json"),
            first_row_sheet_sha256=sha(a.out / "fixed_first_row.png"),
            metrics=out,
        ),
    )
    if a.release:
        selection = read(a.selection)
        with np.load(a.spatial, allow_pickle=False) as z:
            check(
                str(z["spatial_selection_sha256"]) == sha(a.selection)
                and sm["driver_sha256"]
                == sha(Path(__file__).with_name("render_personal_spatial_latents.py")),
                "Release requires the real spatial renderer and its matched generic selection",
            )
        k = selection["ridges"].index(selection["ridge"])
        oof = selection["validation"][k]
        gain = selection["gain"]
        compact = dict(
            method="Current EEG H/14, class support VAE prior and identical two-step suffix in the three spatial arms. The four-step historical arm uses the same H/14 and noise seed, with a full noise start.",
            scope="Development on the complete first registered fold of one previously studied participant; 24 unique queries, 12 objects. Additional panels and participants are outside this experiment.",
            arms=list(ARMS),
            uncertainty="Whole-object percentile bootstrap with 10,000 draws in one participant. Rates in fractions; ties split.",
            generic_object_held_out=dict(
                objects=512,
                photos=5120,
                folds=4,
                selected_ridge=selection["ridge"],
                generation_gain=gain,
                selected_oof=oof,
                r2_at_generation_gain=(2 * gain * oof["mse_optimal_gain"] - gain**2)
                / oof["amplitude_matching_gain"] ** 2,
            ),
            queries=24,
            strict_pair_count=len(pair_hist),
            fold=0,
            participant=SUB,
            panel=PANEL,
            counts={
                k: {arm: v["arms"][arm]["sum"] for arm in ARMS}
                for k, v in {
                    "dino_intra10": out["dino_intra10"],
                    "all_pairs": out["all_pairs"]["paired"],
                    "strict_pairs": out["strict_pairs"]["paired"],
                }.items()
            },
            means={
                k: {arm: out[k]["arms"][arm]["mean"] for arm in ARMS}
                for k in (
                    "rgb_rmse32",
                    "luminance_layout_corr32",
                    "dino_rank10",
                    "dino_target_cosine",
                )
            },
            dino_intra10_own_minus={
                arm: {
                    k: out["dino_intra10"]["own_minus"][arm][k]
                    for k in ("mean", "concept_cluster_ci95")
                }
                for arm in ARMS[1:]
            },
            inputs_sha256=dict(
                summary=sha(a.out / "summary.json"),
                per_query=sha(a.out / "per_query.json"),
                renderer_settings=sha(pilot / "settings.json"),
                original_four_step_settings=sha(baseline / "settings.json"),
                spatial_packet=sha(a.spatial),
                selection=sha(a.selection),
                dino_checkpoint=FILES["model.safetensors"],
                registration=sha(a.references / (PANEL + "_ablation.json")),
            ),
        )
        save(a.release, compact)
    for key in ("dino_intra10", "rgb_rmse32", "luminance_layout_corr32"):
        print(key, {arm: out[key]["arms"][arm]["mean"] for arm in ARMS}, flush=True)
    print(
        "strict paired",
        {arm: out["strict_pairs"]["paired"]["arms"][arm]["sum"] for arm in ARMS},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, d in dict(
        h14_root="outputs/photo-structure/residual_angle",
        renders="outputs/latent-spatial/pilot24/registered/sub-04",
        spatial="outputs/latent-spatial/conditions/sub-04/registered/spatial.npz",
        out="outputs/latent-spatial/pilot_score",
        references="outputs/object-fidelity",
        model="outputs/photo-structure/dino/facebook-dinov2-small",
        reference_cache="outputs/photo-structure/dino/residual_angle_three",
        originals="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
    ).items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=Path(d))
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--selection", type=Path, default=Path("outputs/latent-spatial/selection.json"))
    p.add_argument(
        "--release",
        type=Path,
        help="Optional compact aggregate JSON, without originals or row-level arrays",
    )
    run(p.parse_args())
