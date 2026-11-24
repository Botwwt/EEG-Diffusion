"""Render the current EEG H/14 condition with three matched continuous VAE starts."""

import argparse
import csv
from pathlib import Path
from types import MethodType

import numpy as np
from render_within_photo_ip import ADAPTER_SHA, ARMS, check, packet, read, save, sha


def load_case(a):
    rows, folds, words, donors, support, h14, radius, hsig = packet(a.h14_root, a.subject, a.panel)
    with np.load(a.spatial, allow_pickle=False) as z:
        check(
            str(z["subject"]) == a.subject
            and str(z["panel"]) == a.panel
            and str(z["h14_packet_sha256"]) == hsig
            and all(
                np.array_equal(z[k], v)
                for k, v in dict(
                    rows=rows,
                    folds=folds,
                    predicted_words=words,
                    donor_rows=donors,
                    support_rows=support,
                ).items()
            ),
            "Spatial packet disagrees with H/14 registration",
        )
        prior, residual = z["prior"], z["residual"]
        check(
            prior.shape == residual.shape == (120, 4, 8, 8)
            and np.isfinite(prior).all()
            and np.isfinite(residual).all()
            and np.max(abs(residual.mean((-2, -1)))) < 1e-5,
            "Invalid 8x8 scaled spatial directions",
        )
        selection = str(z["spatial_selection_sha256"])
    rmap = {r: i for i, r in enumerate(rows)}
    idx = np.array([rmap[d] for d in donors])
    check(np.array_equal(prior, prior[idx]), "Wrong control uses different support images")
    return (
        rows,
        folds,
        words,
        donors,
        support,
        hsig,
        h14["own"],
        radius,
        selection,
        dict(own=prior + residual, uniform=prior, wrong=prior + residual[idx]),
    )


def pipeline(a):
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
    from render_official_ip_adapter import batched_embeddings

    turbo = a.archive / "assets/native_generator_v243/turbo"
    check(sha(a.adapter) == ADAPTER_SHA, "Wrong H/14 IP-Adapter")
    txt = StableDiffusionXLPipeline.from_single_file(
        str(turbo / "sd_xl_turbo_1.0_fp16.safetensors"),
        config=str(turbo),
        torch_dtype=torch.float16,
        local_files_only=True,
        add_watermarker=False,
    )
    pipe = StableDiffusionXLImg2ImgPipeline(**txt.components, add_watermarker=False)
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
        "Wrong visual projection",
    )
    pipe.prepare_ip_adapter_image_embeds = MethodType(batched_embeddings, pipe)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(4, device="cuda")
    ts, n = pipe.get_timesteps(4, 0.5, "cuda")
    check(
        n == 2
        and ts.tolist() == [499, 249]
        and type(pipe.scheduler).__name__ == "EulerAncestralDiscreteScheduler"
        and pipe.scheduler.config.prediction_type == "epsilon",
        "Unexpected Euler img2img suffix",
    )
    return pipe


def run(a):
    rows, folds, words, donors, support, sig, h14, radius, ssig, starts = load_case(a)
    if a.check_only:
        print("Verified spatial/H14 inputs", a.subject, a.panel, sha(a.spatial), flush=True)
        return
    import torch
    import torch.nn.functional as F

    output = (
        a.out / "renders" / a.panel / a.subject
        if a.limit == 120
        else a.out / f"pilot{a.limit}" / a.panel / a.subject
    )
    output.mkdir(parents=True, exist_ok=True)
    weight = a.archive / "assets/native_generator_v243/turbo/sd_xl_turbo_1.0_fp16.safetensors"
    protocol = dict(
        subject=a.subject,
        panel=a.panel,
        packet_sha256=sig,
        spatial_sha256=sha(a.spatial),
        spatial_selection_sha256=ssig,
        design_sha256=sha(a.h14_root / "design.json"),
        dictionary_sha256=sha(a.dictionary),
        turbo_sha256=sha(weight),
        adapter_sha256=sha(a.adapter),
        driver_sha256=sha(Path(__file__)),
        batch_size=a.batch_size,
        limit=a.limit,
        schedule="Euler ancestral [499,249], img2img strength 0.5 on four-step grid, no CFG; bilinear 8x8-to-64x64 latent start; fresh same-seed generator for every arm",
        treatment="All arms use the identical own H/14 EEG condition, predicted word, eight-support VAE prior and per-row noise. Own adds matched spatial EEG, uniform adds zero, wrong adds the frozen same-fold/word donor's spatial EEG.",
    )
    record = output / "run.json"
    if record.exists():
        check(read(record) == protocol, "Existing run differs from registered inputs")
    else:
        save(record, protocol)
    with a.dictionary.open(encoding="utf-8-sig", newline="") as f:
        labels = {r["unique_id"]: r["word"] for r in csv.DictReader(f)}
    check(len(labels) == 1854 and set(words) <= set(labels), "Word dictionary differs")
    pipe = pipeline(a)
    with torch.inference_mode():
        prompts = {}
        for w in sorted(set(words[: a.limit])):
            embed, _, pooled, _ = pipe.encode_prompt(
                "A photo of " + labels[w] + ".", device="cuda", do_classifier_free_guidance=False
            )
            prompts[w] = (embed, pooled)
        pipe.text_encoder = pipe.text_encoder_2 = None
        torch.cuda.empty_cache()
        hashes = {}
        for arm in ARMS:
            dest = output / "images" / arm
            dest.mkdir(parents=True, exist_ok=True)
            hashes[arm] = []
            for lo in range(0, a.limit, a.batch_size):
                ix = np.arange(lo, min(lo + a.batch_size, a.limit))
                files = [dest / f"{i:03d}.jpg" for i in ix]
                if not all(f.exists() for f in files):
                    start = F.interpolate(
                        torch.from_numpy(starts[arm][ix]).cuda(),
                        size=(64, 64),
                        mode="bilinear",
                        align_corners=False,
                    ).half()
                    feat = (
                        torch.from_numpy((radius * h14[ix]).astype(np.float32))
                        .cuda()
                        .half()[:, None, :]
                    )
                    gens = [
                        torch.Generator(device="cuda").manual_seed(923761 + int(rows[i]))
                        for i in ix
                    ]
                    embeds, pools = [torch.cat([prompts[words[i]][k] for i in ix]) for k in (0, 1)]
                    imgs = pipe(
                        prompt_embeds=embeds,
                        pooled_prompt_embeds=pools,
                        ip_adapter_image_embeds=[feat],
                        image=start,
                        strength=0.5,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        generator=gens,
                        original_size=(512, 512),
                        target_size=(512, 512),
                        num_images_per_prompt=1,
                        output_type="pil",
                    ).images
                    for im, f in zip(imgs, files, strict=True):
                        check(im.size == (512, 512), "Unexpected output size")
                        tmp = f.with_suffix(".partial")
                        im.convert("RGB").save(tmp, "JPEG", quality=95, subsampling=0)
                        tmp.replace(f)
                hashes[arm].extend(sha(f) for f in files)
            print(a.subject, a.panel, arm, len(hashes[arm]), flush=True)
    save(
        output / "settings.json",
        dict(
            **protocol,
            rows=rows[: a.limit].tolist(),
            folds=folds[: a.limit].tolist(),
            predicted_words=words[: a.limit].tolist(),
            donor_rows=donors[: a.limit].tolist(),
            support_rows=support[: a.limit].tolist(),
            seeds=(923761 + rows[: a.limit]).tolist(),
            radius=radius,
            jpeg_sha256=hashes,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("h14-root", "spatial", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("archive", "dictionary", "adapter"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--subject", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=120, choices=range(1, 121), metavar="1..120")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or (
        not args.check_only
        and any(getattr(args, k) is None for k in ("archive", "dictionary", "adapter"))
    ):
        p.error("Positive batch size and generation assets required")
    run(args)
