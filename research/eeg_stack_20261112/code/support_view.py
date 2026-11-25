"""Use a single off-fold EEG-selected thumbnail to initialize a personal reconstruction."""

import argparse
import csv
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, table
from fit_support_structure import unit
from fit_within_photo_embedding import images
from PIL import Image
from register_fresh_photo_panel import freeze, read, sha
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA, check, packet
from run_prospective_readout import frozen_npz
from score_reconstruction_jpegs import registered
from score_within_photo_dino import ZIP_SHA

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "support_view_design.json"
ARMS = ("mean", "medoid", "own", "wrong")


def prepare(a):
    source = (
        a.data / "trial-object-encoder/personal12"
        if a.subject == "sub-04"
        else a.data / "trial-object-replication" / a.subject
    )
    rows, folds, words, donors, supports, vectors, radius, digest = packet(
        source / "render_inputs", a.subject, "personal12"
    )
    reg = registered(
        a.data / "object-head-confirmation/evaluation/references" / a.subject, "personal12"
    )
    spec = read(PUBLIC / "personal12_pose_evaluation_design.json")
    check(
        reg["sha"] == spec["registration_sha256"][a.subject]
        and np.array_equal(reg["rows"], rows)
        and np.array_equal(reg["folds"], folds),
        "Wrong population",
    )
    wi = [reg["vocab"].tolist().index(w) for w in words]
    check(np.array_equal(supports, reg["calibration"][folds, wi]), "Off-fold supports changed")
    bank = images(a.data / "official-object-readout/sub-03")
    feat = unit(bank[supports].astype(float))
    check(
        np.allclose(unit(feat.mean(1)), vectors["uniform"], atol=2e-5), "Support direction mismatch"
    )
    chosen = {}
    for arm, key in (("medoid", "uniform"), ("own", "own"), ("wrong", "wrong")):
        selection = (feat * vectors[key][:, None]).sum(-1).argmax(axis=1)
        chosen[arm] = supports[np.arange(len(rows)), selection]
    byrow = {r: i for i, r in enumerate(rows)}
    check(
        np.array_equal(chosen["wrong"], chosen["own"][[byrow[d] for d in donors]]), "Donor mismatch"
    )
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    tab = table(a)
    photo_zip = a.data / "photo-structure/training_images.zip"
    check(sha(photo_zip) == ZIP_SHA, "Unexpected source archive")
    with ZipFile(photo_zip) as archive:
        thumb = {
            r: np.asarray(original(archive, tab, int(r)).resize((32, 32), Image.Resampling.BOX))
            for r in np.unique(supports)
        }
    starts = {arm: np.stack([thumb[r] for r in selected]) for arm, selected in chosen.items()}
    starts["mean"] = np.stack(
        [np.rint(np.mean([thumb[r] for r in sr], axis=0)).astype(np.uint8) for sr in supports]
    )
    output = a.root / a.subject
    f = output / "inputs.npz"
    frozen_npz(
        f,
        dict(
            rows=rows,
            folds=folds,
            words=words,
            donors=donors,
            supports=supports,
            h14=vectors["own"],
            radius=radius,
            **starts,
            **{"selected_" + k: v for k, v in chosen.items()},
        ),
    )
    freeze(
        output / "inputs.json",
        dict(
            subject=a.subject,
            design_sha256=sha(DESIGN),
            source_packet_sha256=digest,
            sha256=sha(f),
            registration_sha256=reg["sha"],
            support_zip_sha256=ZIP_SHA,
            distinct_matched_vs_medoid=int((chosen["own"] != chosen["medoid"]).sum()),
            distinct_matched_vs_wrong=int((chosen["own"] != chosen["wrong"]).sum()),
        ),
    )


def render(a):
    import torch
    from render_fixed_word_guidance import embedding
    from render_personal_spatial_latents import pipeline

    base = a.root / a.subject
    meta = read(base / "inputs.json")
    check(
        meta["subject"] == a.subject
        and meta["design_sha256"] == sha(DESIGN)
        and sha(base / "inputs.npz") == meta["sha256"],
        "Unexpected render inputs",
    )
    check(
        sha(a.dictionary) == DICTIONARY
        and sha(a.adapter) == ADAPTER_SHA
        and sha(a.archive / "assets/native_generator_v243/turbo/sd_xl_turbo_1.0_fp16.safetensors")
        == TURBO,
        "Wrong generator assets",
    )
    with a.dictionary.open(encoding="utf-8-sig", newline="") as stream:
        names = {r["unique_id"]: r["word"] for r in csv.DictReader(stream)}
    pipe = pipeline(a)
    with np.load(base / "inputs.npz", allow_pickle=False) as z, torch.inference_mode():
        rows, words = z["rows"], z["words"]
        prompts = {}
        for w in np.unique(words):
            e, _, po, _ = pipe.encode_prompt(
                "A photo of " + names[w] + ".", device="cuda", do_classifier_free_guidance=False
            )
            prompts[w] = e, po
        pipe.text_encoder = pipe.text_encoder_2 = None
        torch.cuda.empty_cache()
        # SDXL's VAE requires its force-upcast path for finite posterior modes.
        pipe.vae.to(dtype=torch.float32)
        out = base / "renders"
        hashes = {}
        for arm in ARMS:
            folder = out / arm
            folder.mkdir(exist_ok=True, parents=True)
            thumbs = z[arm]
            hashes[arm] = []
            for lo in range(0, len(rows), a.batch_size):
                ix = np.arange(lo, min(lo + a.batch_size, len(rows)))
                pixels = np.stack(
                    [
                        np.asarray(
                            Image.fromarray(thumbs[i]).resize((512, 512), Image.Resampling.BILINEAR)
                        )
                        for i in ix
                    ]
                )
                batch = torch.from_numpy(pixels).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
                # vae.decode may leave the model in fp16 when diffusers restores its dtype.
                pipe.vae.to(dtype=torch.float32)
                start = (
                    pipe.vae.encode(batch).latent_dist.mode() * pipe.vae.config.scaling_factor
                ).half()
                check(
                    torch.isfinite(start).all().item() and start.shape[1:] == (4, 64, 64),
                    "Bad VAE mode",
                )
                # Match the standard pipeline's fp16 decode/upcast path.
                pipe.vae.to(dtype=torch.float16)
                e, po = [torch.cat([prompts[words[i]][k] for i in ix]) for k in (0, 1)]
                generated = pipe(
                    prompt_embeds=e,
                    pooled_prompt_embeds=po,
                    ip_adapter_image_embeds=[embedding(z["h14"], ix, float(z["radius"]))],
                    image=start,
                    strength=0.5,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    generator=[
                        torch.Generator(device="cuda").manual_seed(923761 + int(rows[i]))
                        for i in ix
                    ],
                    original_size=(512, 512),
                    target_size=(512, 512),
                    output_type="pil",
                ).images
                for i, pic in zip(ix, generated, strict=True):
                    p = folder / f"{i:03d}.jpg"
                    check(pic.size == (512, 512), "Unexpected output size")
                    pic.save(p, "JPEG", quality=95, subsampling=0)
                    hashes[arm].append(sha(p))
            print(arm, len(hashes[arm]), flush=True)
    freeze(
        out / "settings.json",
        dict(
            input_sha256=meta["sha256"],
            design_sha256=sha(DESIGN),
            renderer_sha256=sha(Path(__file__)),
            subject=a.subject,
            seeds=(923761 + rows).tolist(),
            jpeg_sha256=hashes,
            batch_size=a.batch_size,
            adapter_sha256=ADAPTER_SHA,
            turbo_sha256=TURBO,
        ),
    )


def encode_clip(a):
    from eval_instance_starts import load_visual_scorer
    from score_official_objects import load_dictionary, load_tokenizer, text_bank
    from score_reconstruction_jpegs import CLIP_SHA, raw_features

    out = a.root / a.subject / "renders"
    cfg = read(out / "settings.json")
    check(
        cfg["subject"] == a.subject
        and cfg["design_sha256"] == sha(DESIGN)
        and len(cfg["seeds"]) == 120,
        "Wrong render case",
    )
    model, weight = load_visual_scorer(a.archive.resolve())
    check(weight == CLIP_SHA and sha(a.dictionary) == DICTIONARY, "Unexpected evaluation assets")
    ids, common, inverse, _, _ = load_dictionary(a.dictionary)
    tok = load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer")
    texts = text_bank(model, tok, common, 16)
    scores = {}
    for arm in ARMS:
        files = [out / arm / f"{i:03d}.jpg" for i in range(120)]
        check([sha(p) for p in files] == cfg["jpeg_sha256"][arm], "Image changed")

        def open_image(i, paths=files):
            with Image.open(paths[i]) as pic:
                return pic.convert("RGB")

        scores[arm] = (unit(raw_features(model, open_image, len(files), 16)) @ texts.T)[:, inverse]
    frozen_npz(
        out / "clip_logits.npz",
        dict(
            ids=np.array(ids),
            settings_sha256=sha(out / "settings.json"),
            model_sha256=weight,
            **scores,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "render", "clip"))
    p.add_argument("--subject", default="sub-04", choices=("sub-04", "sub-05", "sub-01"))
    p.add_argument("--root", type=Path, default=Path("outputs/support-view"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--archive", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    {"prepare": prepare, "render": render, "clip": encode_clip}[a.stage](a)
