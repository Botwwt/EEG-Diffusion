"""Apply a frozen EEG spatial readout to one clean-latent estimate on the native Turbo schedule."""

import argparse
import csv
from functools import wraps
from pathlib import Path
from types import MethodType
from zipfile import ZipFile

import numpy as np
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import ADAPTER_SHA, check
from run_prospective_readout import frozen_npz

CODE = Path(__file__).resolve().parent
DESIGN = CODE.parent / "results/object_fidelity/lowband_guidance_design.json"
LOCK = "0c34b45fb3e1b0f598465101b1000ee30bcdd22adbad78a5f002a615db9cc570"
SUB = "sub-05"
PANEL = "personal12"
ARMS = ("native", "prior", "own", "wrong")


def prepare(a):
    from fit_spatial_photo import LUMA, ZIP_SHA, eeg_design, original, spatial, table, waves
    from render_within_photo_ip import packet
    from scipy.fft import idctn
    from score_reconstruction_jpegs import registered

    sources = read(DESIGN)["sources"]
    src = a.data / "trial-object-replication" / SUB / "render_inputs"
    eeg = a.data / "object-head-confirmation/eeg" / (SUB + ".npz")
    model = a.data / "spatial-photo/models" / (SUB + ".npz")
    settings = src / "renders" / PANEL / SUB / "settings.json"
    regdir = a.data / "object-head-confirmation/evaluation/references" / SUB
    reg = registered(regdir, PANEL)
    old = a.data / "spatial-photo/renders" / PANEL / SUB / "settings.json"
    required = dict(
        registration=regdir / (PANEL + "_ablation.json"),
        eeg=eeg,
        frozen_model=model,
        model_receipt=model.with_suffix(".json"),
        previous_pixel_settings=old,
        native_settings=settings,
        native_run=settings.with_name("run.json"),
        radius=src / "radius.json",
        spatial_code=CODE / "fit_spatial_photo.py",
        native_code=CODE / "render_within_photo_ip.py",
        conditions=src / "conditions" / SUB / PANEL / "conditions_512.npz",
    )
    check(
        {k: sha(v) for k, v in required.items()} == sources,
        "Frozen experimental inputs have changed",
    )
    cfg = read(settings)
    layout = read(old)
    rows, folds, words, donors, support, vec, radius, sig = packet(src, SUB, PANEL)
    check(
        np.array_equal(reg["rows"], rows)
        and np.array_equal(reg["folds"], folds)
        and np.array_equal(
            reg["calibration"][folds, [reg["vocab"].tolist().index(w) for w in words]], support
        )
        and sig == layout["packet_sha256"] == cfg["packet_sha256"]
        and sha(eeg) == layout["eeg_sha256"]
        and cfg["seeds"] == (rows + 923761).tolist()
        and cfg["batch_size"] == 8,
        "Mismatch with frozen generator or spatial readout",
    )
    with np.load(model, allow_pickle=False) as z:
        weight, gain, classes = (z[k] for k in ("w", "gain", "classes"))
    check(
        len(set(classes)) == 512 and not set(classes) & set(rows // 10),
        "Spatial fit contains a target object",
    )
    ids, wave = waves(eeg, SUB)
    x = eeg_design(wave, np.arange(len(wave)), np.arange(17)).astype(float)
    qi, si = np.searchsorted(ids, rows), np.searchsorted(ids, support)
    check(
        np.array_equal(ids[qi], rows) and np.array_equal(ids[si], support), "EEG row lookup differs"
    )
    delta = ((x[qi] - x[si].mean(1)) @ weight).reshape(-1, 8, 8, 4) * gain
    dj = np.array([rows.tolist().index(d) for d in donors])
    check(
        np.array_equal(folds, folds[dj])
        and np.array_equal(words, words[dj])
        and np.array_equal(support, support[dj]),
        "Incorrect donor support",
    )
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    tab = table(a)
    zipfile = a.data / "photo-structure/training_images.zip"
    check(sha(zipfile) == ZIP_SHA, "Changed support image archive")
    dc = np.empty((len(rows), 8, 8, 4))
    with ZipFile(zipfile) as archive:
        for f in range(5):
            byrow = {}
            ix = np.flatnonzero(folds == f)
            held = set(rows[ix])
            for i in ix:
                check(
                    set(support[i]).isdisjoint(held) and len(set(support[i])) == 8,
                    "Held-out photo in candidate features",
                )
                for row in support[i]:
                    check(tab[row]["unique_id"] == words[i], "Wrong support word")
                    if row not in byrow:
                        byrow[row] = spatial(original(archive, tab, int(row)))[0]
                dc[i] = np.stack([byrow[r] for r in support[i]]).mean(0)
    anchors = {}
    for arm, target in (("prior", dc), ("own", dc + delta), ("wrong", dc + delta[dj])):
        lum, blue, red, _ = np.moveaxis(idctn(target, axes=(1, 2), norm="ortho"), -1, 0)
        rgb = np.stack(
            [lum + red, lum - (LUMA[0] * red + LUMA[2] * blue) / LUMA[1], lum + blue], axis=-1
        )
        anchors[arm] = np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), np.float32)
    check(
        np.array_equal(anchors["own"][dj], anchors["wrong"]) and np.isfinite(anchors["own"]).all(),
        "Wrong donor anchors",
    )
    dest = a.root / SUB
    dest.mkdir(exist_ok=True, parents=True)
    fn = dest / "inputs.npz"
    frozen_npz(
        fn,
        dict(
            rows=rows,
            folds=folds,
            words=words,
            supports=support,
            donors=donors,
            h14=vec["own"],
            radius=radius,
            **anchors,
        ),
    )
    freeze(
        dest / "inputs.json",
        dict(
            subject=SUB,
            design_sha256=LOCK,
            input_sha256=sha(fn),
            source_settings_sha256=sha(settings),
            source_packet_sha256=sig,
            expected_native_jpeg_sha256=cfg["jpeg_sha256"]["own"],
            batch_size=cfg["batch_size"],
            dictionary_sha256=cfg["dictionary_sha256"],
            adapter_sha256=cfg["adapter_sha256"],
            turbo_sha256=cfg["turbo_sha256"],
        ),
    )


def correction(epsilon, sample, sigma, anchor):
    import torch.nn.functional as F

    x0 = sample.float() - sigma * epsilon.float()
    residual = anchor.float() - F.avg_pool2d(x0, kernel_size=8, stride=8)
    low = F.interpolate(residual, size=(64, 64), mode="bilinear", align_corners=False)
    return (epsilon.float() - (0.35 / sigma) * low).to(epsilon.dtype)


def encode_anchors(vae, values):
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(values).to(device="cuda", dtype=torch.float32).permute(0, 3, 1, 2)
    image = (
        F.interpolate(tensor, size=(512, 512), mode="bilinear", align_corners=False).mul(2).sub(1)
    )
    latent = vae.encode(image).latent_dist.mode() * vae.config.scaling_factor
    check(
        torch.isfinite(latent).all().item() and latent.shape[1:] == (4, 64, 64),
        "Nonfinite anchor VAE mode",
    )
    return F.avg_pool2d(latent, kernel_size=8, stride=8)


def hook_step(scheduler, fn, anchor, seen):
    @wraps(fn)
    def step(model_output, timestep, sample, *args, **kwargs):
        t = int(timestep.item())
        seen.append(t)
        if t == 499:
            # scale_model_input initialized step_index in the original pipeline before the U-Net.
            check(scheduler.step_index == 2, "Unexpected location on Euler noise grid")
            sig = float(scheduler.sigmas[scheduler.step_index])
            check(sig > 0, "Invalid scheduler noise level")
            model_output = correction(model_output, sample, sig, anchor)
        return fn(model_output, timestep, sample, *args, **kwargs)

    return step


def render(a):
    import copy

    import torch
    from diffusers import StableDiffusionXLPipeline
    from render_official_ip_adapter import batched_embeddings

    base = a.root / SUB
    meta = read(base / "inputs.json")
    file = base / "inputs.npz"
    check(
        meta["design_sha256"] == LOCK
        and meta["input_sha256"] == sha(file)
        and meta["batch_size"] == 8,
        "Unexpected packet",
    )
    turbo = a.archive / "assets/native_generator_v243/turbo"
    weight = turbo / "sd_xl_turbo_1.0_fp16.safetensors"
    check(
        sha(weight) == meta["turbo_sha256"]
        and sha(a.adapter) == ADAPTER_SHA == meta["adapter_sha256"]
        and sha(a.dictionary) == meta["dictionary_sha256"],
        "Wrong frozen generator components",
    )
    with a.dictionary.open(encoding="utf-8-sig", newline="") as stream:
        name = {r["unique_id"]: r["word"] for r in csv.DictReader(stream)}
    # Match the original text-to-image loader and batch ordering, including the VAE decode path.
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
    pipe.prepare_ip_adapter_image_embeds = MethodType(batched_embeddings, pipe)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(4, device="cuda")
    check(
        type(pipe.scheduler).__name__ == "EulerAncestralDiscreteScheduler"
        and pipe.scheduler.timesteps.tolist() == [999, 749, 499, 249]
        and pipe.scheduler.config.prediction_type == "epsilon",
        "Unexpected native Euler scheduler",
    )
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = base / "renders"
    hashes = {}
    with np.load(file, allow_pickle=False) as z, torch.inference_mode():
        rows, words, vec, rad = (z[k] for k in ("rows", "words", "h14", "radius"))
        prompts = {}
        for w in sorted(set(words)):
            e, _, po, _ = pipe.encode_prompt(
                "A photo of " + name[w] + ".", device="cuda", do_classifier_free_guidance=False
            )
            prompts[w] = (e, po)
        pipe.text_encoder = pipe.text_encoder_2 = None
        torch.cuda.empty_cache()
        original_step = pipe.scheduler.step
        latent_vae = None
        for arm in ARMS:
            folder = out / arm
            folder.mkdir(parents=True, exist_ok=True)
            hashes[arm] = []
            if arm == "prior":
                # A separate anchor VAE protects the native decoder weights and dtype transitions.
                latent_vae = copy.deepcopy(pipe.vae).float().eval()
            for lo in range(0, len(rows), 8):
                ids = np.arange(lo, min(lo + 8, len(rows)))
                feat = (
                    torch.from_numpy((float(rad) * vec[ids]).astype(np.float32))
                    .cuda()
                    .half()[:, None, :]
                )
                gens = [
                    torch.Generator(device="cuda").manual_seed(923761 + int(rows[i])) for i in ids
                ]
                e, po = [torch.cat([prompts[words[i]][k] for i in ids]) for k in (0, 1)]
                seen = []
                if arm != "native":
                    anchor = encode_anchors(latent_vae, z[arm][ids])
                    pipe.scheduler.step = hook_step(pipe.scheduler, original_step, anchor, seen)
                try:
                    batch = pipe(
                        prompt_embeds=e,
                        pooled_prompt_embeds=po,
                        ip_adapter_image_embeds=[feat],
                        generator=gens,
                        width=512,
                        height=512,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        num_images_per_prompt=1,
                        output_type="pil",
                    ).images
                finally:
                    if arm != "native":
                        pipe.scheduler.step = original_step
                check(
                    arm == "native" or seen == [999, 749, 499, 249],
                    "Original four scheduler invocations did not replay",
                )
                for i, im in zip(ids, batch, strict=True):
                    f = folder / f"{i:03}.jpg"
                    check(im.size == (512, 512), "Bad output dimensions")
                    im.convert("RGB").save(f, "JPEG", quality=95, subsampling=0)
                    sig = sha(f)
                    hashes[arm].append(sig)
                    if arm == "native":
                        check(
                            sig == meta["expected_native_jpeg_sha256"][i],
                            f"Native parity failed at row {i}; check GPU, batch or software",
                        )
            print(arm, len(hashes[arm]), flush=True)
    freeze(
        out / "settings.json",
        dict(
            subject=SUB,
            design_sha256=LOCK,
            input_sha256=sha(file),
            renderer_sha256=sha(Path(__file__)),
            source_settings_sha256=meta["source_settings_sha256"],
            native_parity=120,
            seeds=(923761 + rows).tolist(),
            jpeg_sha256=hashes,
            step=499,
            gain=0.35,
        ),
    )


def clip(a):
    from eval_instance_starts import load_visual_scorer
    from fit_support_structure import unit
    from PIL import Image
    from score_official_objects import load_dictionary, load_tokenizer, text_bank
    from score_reconstruction_jpegs import CLIP_SHA, raw_features

    base = a.root / SUB
    out = base / "renders"
    cfg = read(out / "settings.json")
    meta = read(base / "inputs.json")
    check(
        cfg["design_sha256"] == LOCK
        and cfg["native_parity"] == 120
        and cfg["jpeg_sha256"]["native"] == meta["expected_native_jpeg_sha256"]
        and sha(a.dictionary) == meta["dictionary_sha256"],
        "Changed run",
    )
    model, weight = load_visual_scorer(a.archive.resolve())
    check(weight == CLIP_SHA, "Wrong scorer")
    ids, common, inverse, _, _ = load_dictionary(a.dictionary)
    tokens = load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer")
    txt = text_bank(model, tokens, common, 16)
    logits = {}
    for arm in ARMS:
        files = [out / arm / f"{i:03}.jpg" for i in range(120)]
        check([sha(f) for f in files] == cfg["jpeg_sha256"][arm], "Image changed")

        def opened(i, paths=files):
            with Image.open(paths[i]) as im:
                return im.convert("RGB")

        logits[arm] = (unit(raw_features(model, opened, 120, 16)) @ txt.T)[:, inverse]
    frozen_npz(
        out / "clip_logits.npz",
        dict(
            ids=np.asarray(ids),
            settings_sha256=sha(out / "settings.json"),
            model_sha256=weight,
            **logits,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "render", "clip"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--root", type=Path, default=Path("outputs/lowband-guidance"))
    for key in ("archive", "adapter", "dictionary"):
        p.add_argument("--" + key, type=Path)
    a = p.parse_args()
    check(sha(DESIGN) == LOCK, "Design not frozen")
    {"prepare": prepare, "render": render, "clip": clip}[a.stage](a)


if __name__ == "__main__":
    main()
