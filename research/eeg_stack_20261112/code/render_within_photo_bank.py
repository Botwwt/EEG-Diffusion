"""Prepare support-only H14 centroids, or independently generate the eight shared images per cell."""

import argparse
import csv
from pathlib import Path
from types import MethodType

import numpy as np
from render_within_photo_ip import PANELS, check, read, save, sha


def prepare(a):
    from fit_support_structure import population, unit
    from fit_within_photo_embedding import images
    from score_reconstruction_jpegs import registered

    plan = read(a.root / "design.json")
    bank = images(a.source)
    table = population(a)
    with (a.source / "dictionary.csv").open(encoding="utf-8-sig", newline="") as f:
        dictionary = {r["unique_id"]: r["word"] for r in csv.DictReader(f)}
    check(
        len(dictionary) == 1854 and sha(a.source / "dictionary.csv") == plan["dictionary_sha256"],
        "Unexpected prompt dictionary",
    )
    check(sha(a.source / "assets/text_train.pt") == plan["visual_sha256"], "Unexpected visual bank")
    prev_radius = a.root / "radius.json"
    check(sha(prev_radius) == plan["radius_sha256"], "Expected original generic-only radius")
    receipts = {}
    objects = set()
    for panel in PANELS:
        reg = registered(a.references, panel)
        check(reg["sha"] == plan["registration_sha256"][panel], "Changed folds or vocabulary")
        words = reg["vocab"]
        support = reg["calibration"]
        check(not objects.intersection(words), "Panel objects overlap")
        objects.update(words)
        uniform = np.empty((5, 12, 1024), np.float32)
        for fold, groups in enumerate(support):
            check(
                not np.intersect1d(groups, reg["rows"][reg["folds"] == fold]).size
                and len(set(groups.ravel())) == 96,
                "Fold includes held-out or duplicate image",
            )
            for word, group in zip(words, groups, strict=True):
                check(
                    all(table[r]["unique_id"] == word for r in group),
                    "Calibration photograph belongs to a different object",
                )
            uniform[fold] = unit(unit(bank[groups].astype(np.float64)).mean(1))
        from_packets = {}
        for subject in ("sub-04", "sub-05"):
            packet = a.conditions / subject / panel / "conditions_512.npz"
            with np.load(packet, allow_pickle=False) as z:
                wi = {w: i for i, w in enumerate(words)}
                fg = z["folds"]
                ci = [wi[w] for w in z["predicted_words"]]
                check(
                    str(z["registration_sha256"]) == reg["sha"]
                    and np.array_equal(reg["rows"], z["rows"])
                    and np.array_equal(reg["folds"], fg)
                    and np.array_equal(support[fg, ci], z["support_rows"])
                    and np.array_equal(uniform[fg, ci], z["uniform"]),
                    "Centroids differ from the frozen EEG packets",
                )
                from_packets[subject] = sha(packet)
        path = a.root / (panel + ".npz")
        np.savez(
            path,
            words=words,
            common=np.asarray([dictionary[w] for w in words]),
            support_rows=support,
            uniform=uniform,
        )
        receipts[panel] = dict(packet_sha256=sha(path), verified_against=from_packets)
    save(
        a.root / "conditions.json", dict(design_sha256=sha(a.root / "design.json"), panels=receipts)
    )


def validated(root, panel):
    plan = read(root / "design.json")
    receipt = read(root / "conditions.json")
    path = root / (panel + ".npz")
    radius = root / "radius.json"
    check(
        sha(path) == receipt["panels"][panel]["packet_sha256"]
        and sha(root / "design.json") == receipt["design_sha256"]
        and sha(radius) == plan["radius_sha256"],
        "Changed library inputs",
    )
    with np.load(path, allow_pickle=False) as z:
        check(
            set(z.files) == {"words", "common", "support_rows", "uniform"}, "Unexpected bank fields"
        )
        w, common, support, direction = (
            z[k] for k in ("words", "common", "support_rows", "uniform")
        )
    check(
        w.shape == common.shape == (12,)
        and len(set(w)) == 12
        and support.shape == (5, 12, 8)
        and direction.shape == (5, 12, 1024)
        and np.allclose(np.linalg.norm(direction, axis=-1), 1, rtol=0, atol=1e-6)
        and plan["seeds"] == list(range(740291, 740299))
        and plan["candidates_per_cell"] == 8
        and all(
            isinstance(i, str) and i and i not in (".", "..") and "/" not in i and "\\" not in i
            for i in w.tolist()
        ),
        "Invalid support-only bank",
    )
    return plan, w, common, direction, float(read(radius)["median_norm"])


def render(a):
    plan, words, common, directions, radius = validated(a.root, a.panel)
    if a.check_only:
        return
    import torch
    from diffusers import StableDiffusionXLPipeline
    from render_official_ip_adapter import batched_embeddings

    turbo = a.archive / "assets/native_generator_v243/turbo"
    weight = turbo / "sd_xl_turbo_1.0_fp16.safetensors"
    check(
        sha(a.adapter) == plan["adapter_sha256"] and sha(weight) == plan["turbo_sha256"],
        "Wrong generator weights",
    )
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(weight),
        config=str(turbo),
        torch_dtype=torch.float16,
        local_files_only=True,
        add_watermarker=False,
    )
    pipe.load_ip_adapter(
        str(a.adapter.parent.resolve()),
        subfolder="",
        weight_name=a.adapter.name,
        image_encoder_folder=None,
        local_files_only=True,
    )
    pipe.set_ip_adapter_scale(1.0)
    (layer,) = pipe.unet.encoder_hid_proj.image_projection_layers
    check(
        (layer.image_embeds.in_features, layer.image_embeds.out_features) == (1024, 8192),
        "Wrong H14 projection",
    )
    pipe.prepare_ip_adapter_image_embeds = MethodType(batched_embeddings, pipe)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(4, device="cuda")
    check(
        type(pipe.scheduler).__name__ == "EulerAncestralDiscreteScheduler"
        and pipe.scheduler.timesteps.tolist() == [999, 749, 499, 249]
        and pipe.scheduler.config.prediction_type == "epsilon",
        "Changed schedule",
    )
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dest = a.root / "renders" / a.panel
    dest.mkdir(parents=True, exist_ok=True)
    lock = dict(
        panel=a.panel,
        design_sha256=sha(a.root / "design.json"),
        condition_sha256=sha(a.root / (a.panel + ".npz")),
        radius_sha256=plan["radius_sha256"],
        adapter_sha256=plan["adapter_sha256"],
        turbo_sha256=plan["turbo_sha256"],
        driver_sha256=sha(Path(__file__)),
        seeds=plan["seeds"],
        radius=radius,
        batch_size=8,
    )
    marker = dest / "run.json"
    if marker.exists():
        check(read(marker) == lock, "Different bank already in output directory")
    else:
        check(not (dest / "images").exists(), "Unidentified partial generation")
        save(marker, lock)
    records = []
    with torch.inference_mode():
        embeds = []
        for name in common:
            v, _, pooled, _ = pipe.encode_prompt(
                "A photo of " + name + ".", device="cuda", do_classifier_free_guidance=False
            )
            embeds.append((v.repeat(8, 1, 1), pooled.repeat(8, 1)))
        pipe.text_encoder = pipe.text_encoder_2 = None
        torch.cuda.empty_cache()
        for fold in range(5):
            for col, word in enumerate(words.tolist()):
                paths = [dest / "images" / f"f{fold}" / word / f"{k:02d}.jpg" for k in range(8)]
                paths[0].parent.mkdir(parents=True, exist_ok=True)
                if not all(p.is_file() for p in paths):
                    x = (
                        torch.from_numpy((directions[fold, col] * radius).astype(np.float32))
                        .cuda()
                        .half()
                        .view(1, 1, 1024)
                        .repeat(8, 1, 1)
                    )
                    v, pooled = embeds[col]
                    imgs = pipe(
                        prompt_embeds=v,
                        pooled_prompt_embeds=pooled,
                        ip_adapter_image_embeds=[x],
                        generator=[
                            torch.Generator(device="cuda").manual_seed(s) for s in plan["seeds"]
                        ],
                        width=512,
                        height=512,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        num_images_per_prompt=1,
                        output_type="pil",
                    ).images
                    for path, im in zip(paths, imgs, strict=True):
                        check(im.size == (512, 512), "Unexpected output size")
                        tmp = path.with_suffix(".partial")
                        im.convert("RGB").save(tmp, "JPEG", quality=95, subsampling=0)
                        tmp.replace(path)
                for k, path in enumerate(paths):
                    records.append(
                        dict(
                            fold=fold,
                            word=word,
                            candidate=k,
                            seed=plan["seeds"][k],
                            path=path.relative_to(dest).as_posix(),
                            sha256=sha(path),
                        )
                    )
            print(a.panel, fold, len(records), flush=True)
    check(
        len(records) == 480
        and len({(r["fold"], r["word"], r["candidate"]) for r in records}) == 480,
        "Incomplete candidate grid",
    )
    save(
        dest / "settings.json",
        dict(**lock, distinct_jpeg_sha256=len({r["sha256"] for r in records}), records=records),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "render"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("outputs/official-object-readout/sub-03"))
    p.add_argument("--references", type=Path, default=Path("outputs/object-fidelity"))
    p.add_argument("--conditions", type=Path, default=Path("outputs/photo-structure/within_photo"))
    p.add_argument("--panel", choices=PANELS)
    p.add_argument("--archive", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--check-only", action="store_true")
    opts = p.parse_args()
    if opts.stage == "render" and (
        not opts.panel or (not opts.check_only and not (opts.adapter and opts.archive))
    ):
        p.error("Render needs a panel and offline weights")
    (prepare if opts.stage == "prepare" else render)(opts)
