"""Official ATM R80 H14 directions through the base Tencent SDXL IP-Adapter and SDXL-Turbo.

CPU preparation: --prepare-base PATH --conditions FILE.npz
Offline GPU render: --archive ROOT --adapter FILE --conditions FILE.npz --prior FILE.npy --out DIR
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MethodType
from zipfile import ZipFile

import numpy as np
from render_official_words import file_sha, fixed_plan, output_image, write_json

ADAPTER_SHA = "ebf05d918348aec7abb02a5e9ecef77e0aaea6914a5c4ea13f50d45eb1681831"
DEMO_PRIOR_SHA = "39d36619a296cb5179cde07098aea7e0021fd93f9521ad5d05227eeaf3c0c9e5"
RAW_FLOAT32_SHA = "f089c837c36fbaa8ac372e19ad0b82697d355f88d8569c4f5aa1d27d9a240067"
UPSTREAM = "https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/Generation/"
WEIGHTS = "https://huggingface.co/h94/IP-Adapter/blob/018e402774aeeddd60609b4ecdb7e298259dc729/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"


def prepare(base, path):
    from fit_official_eeg_anchors import TRAIN_EEG
    from official_object_readout import ASSETS, acquire, tensor_storage, unit

    x = (
        tensor_storage(acquire(base / "assets/eeg_train.pt", TRAIN_EEG), 66160, False)
        .reshape(16540, 4, 1024)
        .mean(1)
    )
    visual = acquire(base / "assets/text_train.pt", ASSETS["text_train.pt"])
    with ZipFile(visual) as z:
        (entry,) = [s for s in z.namelist() if s.endswith("/data/1")]
        y = np.frombuffer(z.read(entry), dtype="<f4").reshape(16540, 1024)
    if not np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-4):
        raise ValueError("Expected the released unit-normalized training images")
    r = float(np.median(np.linalg.norm(x, axis=1)))
    x = unit(x)
    mu_x, mu_y = x.mean(0), y.mean(0)
    x -= mu_x
    gram = x.T @ x
    # Fixed ridge: one mean covariance eigenvalue, with an unpenalized intercept.
    lam = np.trace(gram) / len(gram)
    gram.flat[:: len(gram) + 1] += lam
    w = np.linalg.solve(gram, x.T @ (y - mu_y))
    eeg = tensor_storage(acquire(base / "assets/eeg.pt", ASSETS["eeg.pt"]), 200, False)
    inferred = (unit(eeg) - mu_x) @ w + mu_y
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rows=np.arange(200),
        raw=eeg,
        decoded=r * unit(inferred),
        mean=r * unit(mu_y[None])[0],
        train_radius=r,
        ridge_lambda=lam,
    )
    print(
        f"Prepared {path}; training-only auxiliary norm {r:.4f}; native raw EEG range "
        f"{np.linalg.norm(eeg, axis=1).min():.4f}..{np.linalg.norm(eeg, axis=1).max():.4f}",
        flush=True,
    )


def load_conditions(path, prior_path, seed):
    with np.load(path, allow_pickle=False) as pack:
        if not np.array_equal(pack["rows"], np.arange(200)):
            raise ValueError("Expected official rows 0..199")
        raw, decoded, mean = (pack[k].astype(np.float32) for k in ("raw", "decoded", "mean"))
        if mean.shape != (1024,):
            raise ValueError("Training mean must contain 1,024 entries")
    if hashlib.sha256(raw.astype("<f4", copy=False).tobytes()).hexdigest() != RAW_FLOAT32_SHA:
        raise ValueError("Raw rows differ from the published sub-03 ATM R80 tensor")
    if file_sha(prior_path) != DEMO_PRIOR_SHA:
        raise ValueError("Prior differs from the frozen 10-step/CFG-2, seed-1729 export")
    prior = np.load(prior_path, allow_pickle=False).astype(np.float32)
    plan = fixed_plan(
        [dict(predicted_word="continuous condition", predicted_id="continuous")] * 200, seed
    )
    order = np.array([row["shuffled_row"] for row in plan])
    if len(set(order)) != 200 or np.any(order == np.arange(200)):
        raise ValueError("Invalid fixed permutation")
    arms = dict(
        predicted=prior,
        shuffled=prior[order],
        empty=np.broadcast_to(mean, raw.shape),
        raw=raw,
        train_ridge=decoded,
    )
    for name, v in arms.items():
        if (
            v.shape != (200, 1024)
            or not np.isfinite(v).all()
            or np.any(np.linalg.norm(v, axis=1) == 0)
        ):
            raise ValueError(f"Invalid H14 features in {name}")
    return arms, plan


def batched_embeddings(
    self,
    ip_adapter_image,
    ip_adapter_image_embeds,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
):
    # Diffusers normally repeats an adapter condition across the entire prompt
    # batch. Our precomputed B x 1 x 1024 already has one distinct condition per row.
    if (
        ip_adapter_image is not None
        or do_classifier_free_guidance
        or len(ip_adapter_image_embeds) != 1
    ):
        raise ValueError("Only one precomputed ViT-H adapter and CFG=0 are registered")
    (x,) = ip_adapter_image_embeds
    if x.ndim != 3 or x.shape[1:] != (1, 1024) or num_images_per_prompt not in (1, len(x)):
        raise ValueError(f"Unexpected adapter batching {x.shape}, {num_images_per_prompt}")
    return [x.to(device)]


def sheets(out, name):
    from PIL import Image

    # The fixed gallery uses precisely the same 20 rows as the official scorer.
    rows = [j * 199 // 19 for j in range(20)]
    canvas = Image.new("RGB", (5 * 162, 4 * 162), "white")
    for j, i in enumerate(rows):
        with Image.open(output_image(out, name, i)) as im:
            canvas.paste(
                im.convert("RGB").resize((160, 160), getattr(Image, "Resampling", Image).LANCZOS),
                ((j % 5) * 162, (j // 5) * 162),
            )
    canvas.save(out / f"gallery-{name}.jpg", quality=94)
    return rows


def render(args):
    import diffusers
    import torch
    from diffusers import StableDiffusionXLPipeline

    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    data, plan = load_conditions(args.conditions, args.prior, args.seed)
    if file_sha(args.adapter) != ADAPTER_SHA:
        raise ValueError("Adapter differs from the released SDXL ViT-H base adapter")
    turbo = args.archive / "assets/native_generator_v243/turbo"
    weight = turbo / "sd_xl_turbo_1.0_fp16.safetensors"
    config = dict(
        arms=list(data),
        seed=args.seed,
        samples_per_row=1,
        batch_size=args.batch_size,
        nfe=4,
        guidance_scale=0.0,
        ip_scale=1.0,
        resolution=512,
        text_prompt="",
        sha256=dict(
            adapter=ADAPTER_SHA,
            turbo=file_sha(weight),
            conditions=file_sha(args.conditions),
            prior=file_sha(args.prior),
            script=file_sha(Path(__file__)),
        ),
        diffusers=diffusers.__version__,
        torch=torch.__version__,
        adapter_source=WEIGHTS,
        atm_code=UPSTREAM + "custom_pipeline.py",
        atm_demo=UPSTREAM + "1x1024_reconstruct_sdxl.ipynb",
        roles=dict(
            predicted="single draw, original ATM ten-step prior, guidance 2; native norm",
            raw="published ATM sub-03 EEG, native norm as in author's notebook",
            shuffled="fixed derangement of prior rows with original destination noise",
            empty="unit direction of mean official training image, scaled to median training EEG norm",
            train_ridge="mean-four EEG-to-unit-H14 training-only ridge; output unit-normalized, scaled to median training EEG norm",
        ),
        native_norms={
            name: np.quantile(np.linalg.norm(v, axis=1), [0, 0.5, 1]).tolist()
            for name, v in data.items()
        },
    )
    args.out.mkdir(exist_ok=True, parents=True)
    target = args.out / "settings.json"
    if target.exists() and json.loads(target.read_text(encoding="utf8")) != config:
        raise ValueError("Existing output registration differs")
    write_json(target, config)
    write_json(args.out / "query_plan.json", plan)
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
    (proj,) = pipe.unet.encoder_hid_proj.image_projection_layers
    if proj.image_embeds.in_features != 1024 or proj.image_embeds.out_features != 8192:
        raise ValueError("Expected 1,024 H14 coordinates projected to four 2,048-wide tokens")
    pipe.prepare_ip_adapter_image_embeds = MethodType(batched_embeddings, pipe)
    if type(pipe.scheduler).__name__ != "EulerAncestralDiscreteScheduler":
        raise ValueError(f"Unexpected scheduler {type(pipe.scheduler)}")
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(4, device="cuda")
    if (
        pipe.scheduler.timesteps.tolist() != [999, 749, 499, 249]
        or pipe.scheduler.config.prediction_type != "epsilon"
    ):
        raise ValueError("Unexpected Turbo timesteps or prediction type")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        for arm, matrix in data.items():
            for lo in range(0, 200, args.batch_size):
                rows = list(range(lo, min(lo + args.batch_size, 200)))
                paths = [output_image(args.out, arm, i) for i in rows]
                if all(p.is_file() for p in paths):
                    continue
                features = torch.from_numpy(np.ascontiguousarray(matrix[rows])).to(
                    "cuda", torch.float16
                )[:, None, :]
                gens = [torch.Generator(device="cuda").manual_seed(plan[i]["seed"]) for i in rows]
                images = pipe(
                    prompt=[""] * len(rows),
                    ip_adapter_image_embeds=[features],
                    num_images_per_prompt=1,
                    width=512,
                    height=512,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    generator=gens,
                    output_type="pil",
                ).images
                for path, im in zip(paths, images, strict=True):
                    if im.size != (512, 512):
                        raise ValueError("Unexpected generated size")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    im.convert("RGB").save(tmp, "JPEG", quality=95, subsampling=0)
                    tmp.replace(path)
            print(f"Completed {arm}: 200 images", flush=True)
            selected = sheets(args.out, arm)
    write_json(
        args.out / "completion.json",
        dict(
            arms=list(data),
            rows=200,
            timesteps=[999, 749, 499, 249],
            gallery_rows=selected,
            gpu=torch.cuda.get_device_name(),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prepare-base",
        type=Path,
        help="Build --conditions locally from the official ATM training bundle",
    )
    for name in ("conditions", "archive", "prior", "adapter", "out"):
        p.add_argument("--" + name, type=Path, required=name == "conditions")
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--seed", default=20261125, type=int)
    args = p.parse_args()
    if args.prepare_base:
        prepare(args.prepare_base, args.conditions)
    else:
        if args.batch_size < 1 or any(
            getattr(args, k) is None for k in ("archive", "prior", "adapter", "out")
        ):
            p.error(
                "GPU rendering requires --archive --prior --adapter --out and a positive batch size"
            )
        render(args)


if __name__ == "__main__":
    main()
