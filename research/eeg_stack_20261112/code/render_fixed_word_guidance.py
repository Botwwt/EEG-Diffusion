"""Render the registered cohort-9 support-relative epsilon guidance experiment."""

import argparse
import csv
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import numpy as np
from render_fixed_word_photo_dose import PANELS, packet
from render_within_photo_ip import ADAPTER_SHA, check, read, save, sha

REGISTRATION = "74d41ca021a7ba9bbbd9abb9dc2ea148b0616af84bfab370256d7314dfbf8bc0"
SUBJECT = "cohort-9"
ARMS = ("uniform", "own", "wrong")
TIMESTEPS = [999, 749, 499, 249]


def load_cases(a):
    lock = read(a.registration)
    check(sha(a.registration) == REGISTRATION, "Guidance registration changed")
    check(
        sha(a.root / "design.json") == lock["design_sha256"]
        and sha(a.root / "study_design.json") == lock["original_study_sha256"]
        and sha(a.root / "radius.json") == lock["radius_sha256"],
        "Changed nine-observer design",
    )
    cases = {}
    for panel in a.panels:
        data = packet(a.root, SUBJECT, panel)
        rows, folds, words, donors, supports, _, radius, packet_sha = data
        check(packet_sha == lock["packet_sha256"][panel], "A different EEG packet was supplied")
        path = a.baseline / panel / SUBJECT / "settings.json"
        original = read(path)
        expected = dict(
            subject=SUBJECT,
            panel=panel,
            rows=rows.tolist(),
            folds=folds.tolist(),
            predicted_words=words.tolist(),
            donor_rows=donors.tolist(),
            support_rows=supports.tolist(),
            radius=radius,
            seeds=(923761 + rows).tolist(),
            packet_sha256=packet_sha,
            batch_size=8,
            driver_sha256=lock["original_renderer_sha256"],
            design_sha256=lock["design_sha256"],
            radius_sha256=lock["radius_sha256"],
            adapter_sha256=ADAPTER_SHA,
        )
        check(
            all(original.get(k) == v for k, v in expected.items())
            and set(original["jpeg_sha256"]) == set(ARMS)
            and all(len(original["jpeg_sha256"][arm]) == 120 for arm in ARMS),
            "Original render receipt and registered query rows disagree",
        )
        cases[panel] = data, original, sha(path)
    return cases


@contextmanager
def relative_epsilon(unet, uniform, selected):
    """Intercept only the U-Net prediction. Diffusers retains its full ordinary scheduler loop."""
    import torch

    observed = []
    ordinary = unet.forward
    had_local_forward = "forward" in vars(unet)
    same = torch.eq(uniform, selected).flatten(1).all(1)
    all_same = bool(same.all())

    def predict(sample, timestep, **kwargs):
        extra = kwargs["added_cond_kwargs"]
        (actual,) = extra["image_embeds"]
        check(
            actual.shape == selected.shape
            and torch.equal(actual, selected)
            and kwargs.get("return_dict") is False,
            "Unexpected original SDXL U-Net call",
        )
        observed.append(int(timestep.item()))
        (eps_base,) = ordinary(
            sample,
            timestep,
            **{**kwargs, "added_cond_kwargs": {**extra, "image_embeds": [uniform]}},
        )
        if all_same:
            return (eps_base,)
        (eps_selected,) = ordinary(sample, timestep, **kwargs)
        # Keep both denoiser calls at the original batch size. Only noise predictions coexist.
        adjusted = eps_base.float() + 2.0 * (eps_selected.float() - eps_base.float())
        return (torch.where(same[:, None, None, None], eps_base, adjusted.to(eps_base.dtype)),)

    unet.forward = predict
    try:
        yield
    finally:
        if had_local_forward:
            unet.forward = ordinary
        else:
            del unet.forward
    check(observed == TIMESTEPS, "Expected exactly four guidance steps")


def load_pipeline(a):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from render_official_ip_adapter import batched_embeddings

    turbo = a.archive / "assets/native_generator_v243/turbo"
    weight = turbo / "sd_xl_turbo_1.0_fp16.safetensors"
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
        "Wrong ViT-H adapter projection",
    )
    pipe.prepare_ip_adapter_image_embeds = MethodType(batched_embeddings, pipe)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(4, device="cuda")
    check(
        type(pipe.scheduler).__name__ == "EulerAncestralDiscreteScheduler"
        and pipe.scheduler.config.prediction_type == "epsilon"
        and pipe.scheduler.timesteps.tolist() == TIMESTEPS,
        "Wrong four-step scheduler",
    )
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return pipe


def embedding(matrix, ids, radius):
    import torch

    return torch.from_numpy((radius * matrix[ids]).astype(np.float32)).cuda().half()[:, None, :]


def generate(a, cases):
    import torch

    turbo = a.archive / "assets/native_generator_v243/turbo/sd_xl_turbo_1.0_fp16.safetensors"
    assets = dict(
        adapter_sha256=sha(a.adapter), turbo_sha256=sha(turbo), dictionary_sha256=sha(a.dictionary)
    )
    check(
        assets["adapter_sha256"] == ADAPTER_SHA
        and all(
            all(original[k] == v for k, v in assets.items()) for _, original, _ in cases.values()
        ),
        "Generator, adapter or dictionary differs from the ordinary run",
    )
    with a.dictionary.open(encoding="utf-8-sig", newline="") as stream:
        names = {r["unique_id"]: r["word"] for r in csv.DictReader(stream)}
    check(len(names) == 1854, "Unexpected word dictionary")
    pipe = load_pipeline(a)
    with torch.inference_mode():
        prompt = {}
        for word in sorted({w for data, _, _ in cases.values() for w in data[2]}):
            emb, _, pool, _ = pipe.encode_prompt(
                "A photo of " + names[word] + ".", device="cuda", do_classifier_free_guidance=False
            )
            prompt[word] = (emb, pool)
        pipe.text_encoder = pipe.text_encoder_2 = None
        torch.cuda.empty_cache()
        for panel, (data, original, base_sha) in cases.items():
            rows, _, words, _, _, vectors, radius, _ = data
            dst = a.out / "renders" / panel / SUBJECT
            check(
                dst.resolve() != (a.baseline / panel / SUBJECT).resolve(),
                "Use a separate guided output directory",
            )
            lock = dict(
                subject=SUBJECT,
                panel=panel,
                packet_sha256=data[-1],
                batch_size=8,
                **assets,
                design_sha256=original["design_sha256"],
                radius_sha256=original["radius_sha256"],
                driver_sha256=sha(Path(__file__)),
                guidance_design_sha256=REGISTRATION,
                ordinary_settings_sha256=base_sha,
                ordinary_driver_sha256=original["driver_sha256"],
                guidance_factor=2.0,
            )
            lockpath = dst / "run.json"
            if lockpath.exists():
                check(read(lockpath) == lock, "Incompatible guided output directory")
            else:
                check(not (dst / "images").exists(), "Existing unregistered images in output")
                save(lockpath, lock)
            hashes = {}
            for arm in ARMS:
                folder = dst / "images" / arm
                folder.mkdir(exist_ok=True, parents=True)
                hashes[arm] = []
                for lo in range(0, len(rows), 8):
                    ids = np.arange(lo, lo + 8)
                    paths = [folder / f"{i:03d}.jpg" for i in ids]
                    if not all(p.is_file() for p in paths):
                        selected = embedding(vectors[arm], ids, radius)
                        gens = [
                            torch.Generator(device="cuda").manual_seed(923761 + int(rows[i]))
                            for i in ids
                        ]
                        emb, pool = [torch.cat([prompt[words[i]][j] for i in ids]) for j in (0, 1)]
                        with (
                            nullcontext()
                            if arm == "uniform"
                            else relative_epsilon(
                                pipe.unet, embedding(vectors["uniform"], ids, radius), selected
                            )
                        ):
                            images = pipe(
                                prompt_embeds=emb,
                                pooled_prompt_embeds=pool,
                                ip_adapter_image_embeds=[selected],
                                generator=gens,
                                width=512,
                                height=512,
                                num_inference_steps=4,
                                guidance_scale=0.0,
                                num_images_per_prompt=1,
                                output_type="pil",
                            ).images
                        for path, im in zip(paths, images, strict=True):
                            check(im.size == (512, 512), "Unexpected resolution")
                            temp = path.with_suffix(".partial")
                            im.convert("RGB").save(temp, "JPEG", quality=95, subsampling=0)
                            temp.replace(path)
                    batch_hashes = [sha(p) for p in paths]
                    if arm == "uniform":
                        check(
                            batch_hashes == original["jpeg_sha256"]["uniform"][lo : lo + 8],
                            f"Uniform parity failed at {panel} recipient ordinals {lo}–{lo + 7}",
                        )
                    hashes[arm].extend(batch_hashes)
                print(panel, arm, len(hashes[arm]), "verified JPEGs", flush=True)
            fields = (
                "rows",
                "folds",
                "predicted_words",
                "donor_rows",
                "support_rows",
                "radius",
                "seeds",
            )
            save(
                dst / "settings.json",
                dict(
                    **lock,
                    **{k: original[k] for k in fields},
                    uniform_parity_hits=120,
                    jpeg_sha256=hashes,
                ),
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "baseline", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--registration",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/object_fidelity/fixed_word_guidance_design.json",
    )
    p.add_argument("--panels", nargs="+", choices=PANELS, default=PANELS)
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Verify ordinary render receipts and the exact registered EEG packets without loading CUDA",
    )
    for name in ("archive", "adapter", "dictionary"):
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    check(len(set(a.panels)) == len(a.panels), "Repeated panel")
    cases = load_cases(a)
    if a.check_only:
        print("Validated", list(cases), flush=True)
    else:
        if not all((a.archive, a.adapter, a.dictionary)):
            p.error("Rendering needs --archive, --adapter and --dictionary")
        generate(a, cases)


if __name__ == "__main__":
    main()
