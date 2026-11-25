"""Render paired repeat-budget comparisons from a frozen, target-blind sub-04 EEG export."""

import argparse
import csv
import json
from pathlib import Path
from types import MethodType

import numpy as np
from render_within_photo_ip import ADAPTER_SHA, check, read, save, sha

ARMS = ("r4", "r80", "wrong", "r4_visual", "r80_visual", "wrong_visual")
TURBO_SHA = "e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26"
DICT_SHA = "1aebc7a2c35edcfcfffc63934754440c90534ccc46df0537db41dc6197450132"


def inputs(a):
    d = read(a.design)
    arms_in_design = tuple(d["arms"])
    check(
        d["packet_sha256"] == sha(a.packet) and arms_in_design in (ARMS, ARMS[3:], ARMS[:3]),
        "Condition lock differs",
    )
    check(sha(a.dictionary) == DICT_SHA, "Wrong 1,854-word dictionary")
    with a.dictionary.open(encoding="utf-8-sig", newline="") as f:
        words = {r["unique_id"]: r["word"] for r in csv.DictReader(f)}
    arms = {}
    with np.load(a.packet, allow_pickle=False) as z:
        rows, donors, radius = z["rows"], z["donor"], z["radius"].item()
        check(
            np.array_equal(rows, np.arange(200))
            and np.array_equal(donors, np.roll(rows, 67))
            and radius == 21.945598602294922
            and len(words) == 1854,
            "Wrong population or radius",
        )
        for arm in arms_in_design:
            ids = donors if arm.startswith("wrong") else rows
            budget = "r4" if arm.startswith("r4") else "r80"
            features = z[budget][ids]
            labels = z[budget + "_words"][ids]
            check(
                features.shape == (200, 1024)
                and np.isfinite(features).all()
                and np.max(abs(np.linalg.norm(features, axis=1) - 1)) < 1e-6
                and all(k in words for k in labels),
                "Invalid continuous direction or label",
            )
            prompts = (
                ["A photograph."] * 200
                if arm.endswith("_visual")
                else ["A photo of " + words[k] + "." for k in labels]
            )
            arms[arm] = (features, prompts, ids)
    return arms, (923761 + rows).tolist(), radius


def render(a, data, seeds, radius):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from render_official_ip_adapter import batched_embeddings

    turbo = a.archive / "assets/native_generator_v243/turbo"
    weight = turbo / "sd_xl_turbo_1.0_fp16.safetensors"
    check(sha(weight) == TURBO_SHA and sha(a.adapter) == ADAPTER_SHA, "Wrong generator weights")
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
    pipe.set_ip_adapter_scale(1)
    pipe.prepare_ip_adapter_image_embeds = MethodType(batched_embeddings, pipe)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(4, device="cuda")
    check(
        type(pipe.scheduler).__name__ == "EulerAncestralDiscreteScheduler"
        and pipe.scheduler.timesteps.tolist() == [999, 749, 499, 249],
        "Wrong noise schedule",
    )
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        prompts = {}
        for text in sorted({t for arm in a.arms for t in data[arm][1]}):
            emb, _, pooled, _ = pipe.encode_prompt(
                text, device="cuda", do_classifier_free_guidance=False
            )
            prompts[text] = (emb, pooled)
        pipe.text_encoder = pipe.text_encoder_2 = None
        torch.cuda.empty_cache()
        for arm in a.arms:
            feats, words, ids = data[arm]
            folder = a.out / arm
            folder.mkdir(exist_ok=True, parents=True)
            hashes = []
            for lo in range(0, 200, a.batch):
                sel = range(lo, min(lo + a.batch, 200))
                gens = [torch.Generator(device="cuda").manual_seed(seeds[i]) for i in sel]
                emb, pooled = (torch.cat([prompts[words[i]][k] for i in sel]) for k in (0, 1))
                ip = torch.as_tensor(
                    feats[sel.start : sel.stop] * radius, device="cuda", dtype=torch.float16
                )
                imgs = pipe(
                    prompt_embeds=emb,
                    pooled_prompt_embeds=pooled,
                    ip_adapter_image_embeds=[ip[:, None]],
                    generator=gens,
                    width=512,
                    height=512,
                    num_inference_steps=4,
                    guidance_scale=0,
                    output_type="pil",
                ).images
                for i, im in zip(sel, imgs, strict=True):
                    path = folder / f"{i:03d}.jpg"
                    check(im.size == (512, 512), "Wrong image size")
                    im.convert("RGB").save(path, quality=95, subsampling=0)
                    hashes.append(sha(path))
            save(
                a.out / (arm + ".json"),
                dict(
                    arm=arm,
                    jpeg_sha256=hashes,
                    prompts=words,
                    eeg_rows=ids.tolist(),
                    seeds=seeds,
                    packet_sha256=sha(a.packet),
                    design_sha256=sha(a.design),
                    batch=a.batch,
                    driver_sha256=sha(Path(__file__)),
                    adapter_sha256=ADAPTER_SHA,
                    turbo_sha256=TURBO_SHA,
                ),
            )
            print("Finished 200 JPEGs", arm, flush=True)


def join(a, data, seeds):
    report = dict(
        rows=list(range(200)),
        seeds=seeds,
        packet_sha256=sha(a.packet),
        design_sha256=sha(a.design),
        jpeg_sha256={},
        eeg_rows={},
        prompts={},
        arm_receipts={},
    )
    for arm in data:
        path = a.out / (arm + ".json")
        rec = read(path)
        hashes = [sha(a.out / arm / f"{i:03d}.jpg") for i in range(200)]
        check(
            rec["arm"] == arm
            and rec["jpeg_sha256"] == hashes
            and rec["seeds"] == seeds
            and rec["packet_sha256"] == report["packet_sha256"]
            and rec["design_sha256"] == report["design_sha256"]
            and rec["adapter_sha256"] == ADAPTER_SHA
            and rec["turbo_sha256"] == TURBO_SHA
            and rec["batch"] == 8
            and rec["eeg_rows"] == data[arm][2].tolist()
            and rec["prompts"] == data[arm][1],
            "Render provenance differs",
        )
        for key in ("jpeg_sha256", "eeg_rows", "prompts"):
            report[key][arm] = rec[key]
        report["arm_receipts"][arm] = sha(path)
    # This receipt links the independent Linux and Windows image scorers.
    temp = a.out / "settings.partial"
    temp.write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf8"))
    temp.replace(a.out / "settings.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("render", "join", "check"))
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/official-repeat-object/sub-04/render"))
    p.add_argument("--archive", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--arms", nargs="+", choices=ARMS)
    p.add_argument("--batch", type=int, choices=(8,), default=8)
    a = p.parse_args()
    data, seeds, radius = inputs(a)
    a.arms = a.arms or list(data)
    check(len(set(a.arms)) == len(a.arms) and set(a.arms) <= data.keys(), "Arm outside the design")
    if a.stage == "render":
        if not a.archive or not a.adapter:
            p.error("Provide both --archive and --adapter to render")
        render(a, data, seeds, radius)
    elif a.stage == "join":
        join(a, data, seeds)
