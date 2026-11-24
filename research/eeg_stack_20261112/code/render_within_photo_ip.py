"""Generate or evaluate registered EEG H14 residuals using distinct, offline stages."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np

ARMS = ("own", "uniform", "wrong")
PANELS = ("registered", "replicate")
ADAPTER_SHA = "ebf05d918348aec7abb02a5e9ecef77e0aaea6914a5c4ea13f50d45eb1681831"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)


def check(v, msg):
    if not v:
        raise ValueError(msg)


def packet(root, subject, panel):
    design = read(root / "design.json")
    radius = read(root / "radius.json")
    file = root / "conditions" / subject / panel / "conditions_512.npz"
    check(
        sha(file) == design["packets_sha256"][f"{subject}/{panel}"]
        and radius["generic_rows"] == design["generic_radius_rows"]
        and np.median(radius["cosines_to_author"]) > 0.95,
        "Locked EEG or generic-only visual scale changed",
    )
    with np.load(file, allow_pickle=False) as d:
        rows, folds, words, donors, support = (
            d[k] for k in ("rows", "folds", "predicted_words", "donor_rows", "support_rows")
        )
        vectors = {a: d[a] for a in ARMS}
        check(
            str(d["visual_sha256"]) == design["visual_sha256"]
            and len(d["training_classes"]) == 512
            and not np.intersect1d(rows // 10, d["training_classes"]).size,
            "Unexpected visual space or overlap with panel",
        )
    check(
        rows.shape == folds.shape == words.shape == donors.shape == (120,)
        and len(set(rows)) == 120
        and set(rows) == set(donors)
        and set(folds) == set(range(5))
        and support.shape == (120, 8),
        "Malformed rows",
    )
    ix = {x: i for i, x in enumerate(rows)}
    j = np.array([ix[x] for x in donors])
    check(
        np.array_equal(words, words[j])
        and np.array_equal(folds, folds[j])
        and np.array_equal(vectors["wrong"], vectors["own"][j])
        and all(
            len(set(s)) == 8 and set(s) <= set(rows[folds != folds[i]])
            for i, s in enumerate(support)
        ),
        "Support or donor is incompatible with the registered fold",
    )
    check(
        all(
            v.shape == (120, 1024) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
            for v in vectors.values()
        ),
        "Expected unit projected H14 directions",
    )
    check(
        np.isclose(np.median(radius["norms"]), radius["median_norm"]) and radius["median_norm"] > 1,
        "Invalid generic image radius",
    )
    return rows, folds, words, donors, support, vectors, float(radius["median_norm"]), sha(file)


def run_render(args):
    packets = {p: packet(args.root, args.subject, p) for p in PANELS}
    if args.check_only:
        return
    import torch
    from diffusers import StableDiffusionXLPipeline
    from render_official_ip_adapter import batched_embeddings

    check(sha(args.adapter) == ADAPTER_SHA, "Wrong Tencent base ViT-H adapter")
    with args.dictionary.open(encoding="utf-8-sig", newline="") as f:
        readable = {r["unique_id"]: r["word"] for r in csv.DictReader(f)}
    check(len(readable) == 1854, "Incomplete vocabulary")
    turbo = args.archive / "assets/native_generator_v243/turbo"
    weight = turbo / "sd_xl_turbo_1.0_fp16.safetensors"
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(weight),
        config=str(turbo),
        torch_dtype=torch.float16,
        local_files_only=True,
        add_watermarker=False,
    )
    pipe.load_ip_adapter(
        str(args.adapter.parent.resolve()),
        subfolder="",
        weight_name=args.adapter.name,
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
        "Unexpected Turbo schedule",
    )
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    frozen = dict(
        adapter_sha256=ADAPTER_SHA,
        turbo_sha256=sha(weight),
        design_sha256=sha(args.root / "design.json"),
        radius_sha256=sha(args.root / "radius.json"),
        dictionary_sha256=sha(args.dictionary),
        driver_sha256=sha(Path(__file__)),
    )
    with torch.inference_mode():
        prompts = {}
        for word in sorted({w for data in packets.values() for w in data[2]}):
            embeds, _, pooled, _ = pipe.encode_prompt(
                "A photo of " + readable[word] + ".",
                device="cuda",
                do_classifier_free_guidance=False,
            )
            prompts[word] = (embeds, pooled)
        pipe.text_encoder = pipe.text_encoder_2 = None
        torch.cuda.empty_cache()
        for p, (rows, folds, words, donors, support, items, radius, signature) in packets.items():
            dest = args.root / "renders" / p / args.subject
            dest.mkdir(parents=True, exist_ok=True)
            lock = dict(
                subject=args.subject,
                panel=p,
                packet_sha256=signature,
                batch_size=args.batch_size,
                **frozen,
            )
            lockfile = dest / "run.json"
            if lockfile.exists():
                check(read(lockfile) == lock, "Different render in output directory")
            else:
                save(lockfile, lock)
            hashes = {a: [] for a in ARMS}
            for arm, matrix in items.items():
                folder = dest / "images" / arm
                folder.mkdir(exist_ok=True, parents=True)
                for lo in range(0, len(rows), args.batch_size):
                    ids = np.arange(lo, min(lo + args.batch_size, len(rows)))
                    paths = [folder / f"{i:03d}.jpg" for i in ids]
                    if not all(f.is_file() for f in paths):
                        feat = (
                            torch.from_numpy((radius * matrix[ids]).astype(np.float32))
                            .cuda()
                            .half()[:, None, :]
                        )
                        gens = [
                            torch.Generator(device="cuda").manual_seed(923761 + int(rows[i]))
                            for i in ids
                        ]
                        embs, pools = [
                            torch.cat([prompts[words[i]][k] for i in ids]) for k in (0, 1)
                        ]
                        imgs = pipe(
                            prompt_embeds=embs,
                            pooled_prompt_embeds=pools,
                            ip_adapter_image_embeds=[feat],
                            generator=gens,
                            width=512,
                            height=512,
                            num_inference_steps=4,
                            guidance_scale=0.0,
                            num_images_per_prompt=1,
                            output_type="pil",
                        ).images
                        for path, im in zip(paths, imgs, strict=True):
                            check(im.size == (512, 512), "Unexpected size")
                            tmp = path.with_suffix(".partial")
                            im.convert("RGB").save(tmp, "JPEG", quality=95, subsampling=0)
                            tmp.replace(path)
                    hashes[arm].extend(sha(f) for f in paths)
                print(args.subject, p, arm, "120 frozen JPEGs", flush=True)
            save(
                dest / "settings.json",
                dict(
                    **lock,
                    rows=rows.tolist(),
                    folds=folds.tolist(),
                    predicted_words=words.tolist(),
                    donor_rows=donors.tolist(),
                    support_rows=support.tolist(),
                    radius=radius,
                    seeds=(923761 + rows).tolist(),
                    jpeg_sha256=hashes,
                ),
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument("--subject", choices=("sub-01", "sub-04", "sub-05"), required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or (
        not args.check_only
        and any(getattr(args, k) is None for k in ("archive", "adapter", "dictionary"))
    ):
        p.error("Renderer needs archive, adapter, dictionary and a positive batch size")
    run_render(args)


if __name__ == "__main__":
    main()
