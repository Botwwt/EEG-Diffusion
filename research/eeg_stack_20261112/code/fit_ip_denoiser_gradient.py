"""Distill a paired, same-noise Turbo gradient into an angle-preserving H14 condition."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(p):
    return json.loads(p.read_bytes())


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(p, data):
    temp = p.with_suffix(".partial")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(p)


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    require(np.all(np.isfinite(n)) and np.all(n > 1e-10), "Undefined tangent direction")
    return x / n


def angular(prior, own):
    c = unit(prior.astype(np.float64))[:, None, :]
    x = unit(own.astype(np.float64))
    d = np.sum(x * c, axis=-1, keepdims=True)
    t = x - d * c
    return c, unit(t), np.arctan2(np.linalg.norm(t, axis=-1, keepdims=True), d)


def fit_heads(packet, gradient):
    c, tangents, _ = angular(packet["prior"][:128], packet["own"][:128])
    prior = c[:, 0]
    teacher = unit(-gradient + np.sum(gradient * prior, axis=1, keepdims=True) * prior)
    x = tangents[:, 0] - tangents[:, 1]
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    require(s[15] > 1e-6, "Degenerate cross-fitted EEG design")
    basis = vt[:16].T
    xr = x @ basis
    penalty = 0.25 * np.mean(s[:16] ** 2)
    projection = np.linalg.solve(xr.T @ xr + np.eye(16) * penalty, xr.T)
    signs = np.ones(128)
    signs[np.random.default_rng(260524).choice(128, 64, replace=False)] = -1
    return dict(
        basis=basis,
        learned=projection @ teacher,
        sham=projection @ (teacher * signs[:, None]),
        penalty=penalty,
    )


def apply_head(prior, own, basis, weights):
    """One condition per query. All leading indices are independent."""
    c, t, theta = angular(prior, own)
    delta = (t @ basis) @ weights
    direction = unit(t + 0.5 * unit(delta - np.sum(delta * c, axis=-1, keepdims=True) * c))
    return unit(np.cos(theta) * c + np.sin(theta) * direction).astype(np.float32)


def validate(root):
    design = read(root / "design.json")
    receipt = read(root / "inputs.json")
    packet = np.load(root / "generic_pairs.npz", allow_pickle=False)
    require(
        receipt["packet_sha256"] == digest(root / "generic_pairs.npz")
        and receipt["design_sha256"] == digest(root / "design.json"),
        "Packet signature mismatch",
    )
    classes = np.r_[design["train_classes"], design["eval_classes"]]
    rows = np.asarray(design["pair_rows"])
    supports = np.asarray(design["support_rows"])
    require(
        len(set(classes)) == 176
        and len(design["train_classes"]) == 128
        and len(design["eval_classes"]) == 48
        and design["subject"] == "sub-04",
        "Wrong generic split",
    )
    require(
        np.array_equal(classes, packet["classes"])
        and np.array_equal(rows, packet["rows"])
        and np.array_equal(
            np.sort(np.c_[rows, supports], axis=1), 10 * classes[:, None] + np.arange(10)
        ),
        "Photo split mismatch",
    )
    require(
        np.array_equal(packet["seeds"], design["noise_seeds"])
        and np.array_equal(packet["t"], design["timesteps"])
        and set(packet["t"]) == {499, 749},
        "Noise lock mismatch",
    )
    for k, shape in [("prior", (176, 1024)), ("own", (176, 2, 1024))]:
        x = packet[k]
        require(
            x.shape == shape and np.max(abs(np.linalg.norm(x, axis=-1) - 1)) < 1e-5, "Invalid " + k
        )
    return packet, design, receipt


def setup(a, design, words):
    import diffusers
    import torch
    from diffusers import StableDiffusionXLPipeline

    native = a.archive / "data/native_T_latents_v256"
    mapping = native / "registration.json"
    tensor = native / "native_VAE_T_modes.npy"
    turbo = a.archive / "assets/native_generator_v243/turbo"
    weights = turbo / "sd_xl_turbo_1.0_fp16.safetensors"
    require(
        digest(mapping) == design["registration_sha256"] and digest(tensor) == design["t_sha256"],
        "Historical mode bank mismatch",
    )
    require(
        digest(weights) == design["turbo_sha256"]
        and digest(a.adapter) == design["ip_adapter_sha256"],
        "Frozen generator mismatch",
    )
    registered = read(mapping)
    positions = {r: i for i, r in enumerate(registered["array_rows"])}
    bank = np.load(tensor, mmap_mode="r", allow_pickle=False)
    require(
        bank.shape == (13230, 4, 64, 64)
        and bank.dtype == np.float32
        and len(positions) == len(bank)
        and registered["VAE_scaling_factor"] == 0.13025,
        "Invalid native tensor",
    )
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(weights),
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
    require(
        (layer.image_embeds.in_features, layer.image_embeds.out_features) == (1024, 8192),
        "Unexpected IP-Adapter projection",
    )
    pipe.to("cuda")
    pipe.unet.requires_grad_(False).eval()
    pipe.scheduler.set_timesteps(4, device="cuda")
    require(
        type(pipe.scheduler).__name__ == "EulerAncestralDiscreteScheduler"
        and pipe.scheduler.config.prediction_type == "epsilon"
        and pipe.scheduler.timesteps.tolist() == [999, 749, 499, 249],
        "Unregistered denoiser schedule",
    )
    sigmas = dict(
        zip(pipe.scheduler.timesteps.tolist(), pipe.scheduler.sigmas.cpu().tolist(), strict=False)
    )
    times = pipe._get_add_time_ids(
        (512, 512),
        (0, 0),
        (512, 512),
        dtype=torch.float16,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).cuda()
    require(times.tolist() == [[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]], "Wrong size conditioning")
    prompts, pooled = [], []
    with torch.no_grad():
        for i in range(0, len(words), 8):
            p, _, v, _ = pipe.encode_prompt(
                ["A photo of " + w + "." for w in words[i : i + 8]],
                device="cuda",
                do_classifier_free_guidance=False,
            )
            prompts.append(p.cpu())
            pooled.append(v.cpu())
    pipe.text_encoder = pipe.text_encoder_2 = pipe.vae = None
    torch.cuda.empty_cache()
    runtime = dict(
        torch=str(torch.__version__),
        diffusers=diffusers.__version__,
        gpu=torch.cuda.get_device_name(),
        dtype="float16 denoiser; float32 physical error",
        sigmas={str(k): v for k, v in sigmas.items()},
    )
    return pipe.unet, torch.cat(prompts), torch.cat(pooled), times, bank, positions, sigmas, runtime


def evaluate_physical(a, data, design, receipt):
    import torch

    start = time.perf_counter()
    out = a.root / "out"
    out.mkdir(exist_ok=True)
    model, prompts, pooled, sizes, bank, positions, sigmas, runtime = setup(
        a, design, data["words"].tolist()
    )
    rows, steps, seeds = data["rows"], data["t"], data["seeds"]
    prior, own = data["prior"], data["own"]
    radius = float(design["radius"])

    def batch_inputs(ids):
        modes = torch.from_numpy(
            np.stack([[bank[positions[int(r)]] for r in rows[i]] for i in ids])
        ).cuda()
        noise = torch.stack(
            [
                torch.randn(
                    (4, 64, 64),
                    device="cuda",
                    dtype=torch.float32,
                    generator=torch.Generator(device="cuda").manual_seed(int(seeds[i])),
                )
                for i in ids
            ]
        )
        t = torch.as_tensor(steps[ids], device="cuda").repeat_interleave(2)
        s = torch.tensor([sigmas[int(steps[i])] for i in ids], device="cuda").reshape(
            -1, 1, 1, 1, 1
        )
        target = noise.repeat_interleave(2, dim=0)
        z = ((modes + s * noise[:, None]) / torch.sqrt(1 + s.square())).flatten(0, 1).half()
        p = prompts[ids].cuda().repeat_interleave(2, dim=0)
        v = pooled[ids].cuda().repeat_interleave(2, dim=0)

        def loss(condition):
            kw = dict(
                text_embeds=v,
                time_ids=sizes.expand(len(z), -1),
                image_embeds=[(condition * radius).half()[:, None, :]],
            )
            result = model(z, t, encoder_hidden_states=p, added_cond_kwargs=kw, return_dict=False)[
                0
            ]
            return (result.float() - target).square().mean((1, 2, 3)).reshape(-1, 2)

        return loss

    gradients = []
    for i in range(0, 128, a.class_batch):
        ids = np.arange(i, min(i + a.class_batch, 128))
        loss = batch_inputs(ids)
        center = torch.tensor(prior[ids], device="cuda", requires_grad=True)
        d = loss(center.repeat_interleave(2, 0))
        grad = torch.autograd.grad(1024.0 * (d[:, 0] - d[:, 1]).sum(), center)[0] / 1024.0
        gradients.append(grad.detach().float().cpu().numpy())
        del d, center, grad, loss
        if (i + len(ids)) % 16 == 0:
            print(
                "gradient classes",
                i + len(ids),
                "elapsed seconds",
                round(time.perf_counter() - start, 1),
                flush=True,
            )
    gradients = np.vstack(gradients)
    require(np.isfinite(gradients).all(), "Nonfinite teacher gradients")
    heads = fit_heads(data, gradients.astype(np.float64))
    np.savez(out / "head.npz", **heads)
    head_sha = digest(out / "head.npz")
    # Commit head before accessing the independent capability control or scoring the held-out modes.
    save(
        out / "fit.json",
        dict(
            **receipt,
            driver_sha256=digest(Path(__file__)),
            head_sha256=head_sha,
            n_classes=128,
            mean_gradient_norm=float(np.linalg.norm(gradients, axis=1).mean()),
            runtime=runtime,
        ),
    )
    c, _, theta = angular(prior[128:], own[128:])
    oracle = data["oracle_eval_only"]
    require(
        oracle.shape == (48, 2, 1024) and np.max(abs(np.linalg.norm(oracle, axis=-1) - 1)) < 1e-5,
        "Incorrect oracle assay",
    )
    _, oracle_t, _ = angular(prior[128:], oracle)
    conditions = {
        "native": own[128:],
        "learned": apply_head(prior[128:], own[128:], heads["basis"], heads["learned"]),
        "sham": apply_head(prior[128:], own[128:], heads["basis"], heads["sham"]),
        "oracle": unit(np.cos(theta) * c + np.sin(theta) * oracle_t).astype(np.float32),
    }
    for arm, mat in conditions.items():
        require(
            np.abs(angular(prior[128:], mat)[2] - theta).max() < 1e-6,
            "Unmatched angular magnitude " + arm,
        )
    case = {k: [] for k in conditions}
    with torch.no_grad():
        for j in range(0, 48, a.class_batch):
            ids = np.arange(j, min(j + a.class_batch, 48))
            loss = batch_inputs(ids + 128)
            for arm, mat in conditions.items():
                v = torch.from_numpy(mat[ids]).cuda()
                match = loss(v.flatten(0, 1)).cpu().numpy()
                swap = loss(v.flip(1).flatten(0, 1)).cpu().numpy()
                case[arm].extend(np.stack([match, swap], axis=1).tolist())
            if (j + len(ids)) % 12 == 0:
                print("held-out classes", j + len(ids), flush=True)
    save(
        out / "heldout.json",
        dict(
            classes=data["classes"][128:].tolist(),
            rows=rows[128:].tolist(),
            timesteps=steps[128:].tolist(),
            loss_axes=["class", "matched_or_swapped", "photo_A_or_B"],
            losses=case,
        ),
    )
    boot = np.random.default_rng(260525).integers(48, size=(10000, 48))

    def summary(x):
        return dict(
            mean=float(np.mean(x)),
            ci95=np.quantile(np.mean(x[boot], axis=1), [0.025, 0.975]).tolist(),
        )

    matches = {k: np.mean(np.asarray(v)[:, 0], axis=1) for k, v in case.items()}
    contrasts = {k: np.mean(np.asarray(v)[:, 1], axis=1) - matches[k] for k, v in case.items()}
    report = {
        k: dict(
            matched_mse=summary(matches[k]),
            swapped_minus_matched=summary(x),
            positive_classes=int((x > 0).sum()),
            ties=int((x == 0).sum()),
            sign_credit=float(np.mean((x > 0).astype(float) + 0.5 * (x == 0))),
        )
        for k, x in contrasts.items()
    }
    paired = {k: summary(contrasts["learned"] - contrasts[k]) for k in ["native", "sham"]}
    absolute = summary(matches["native"] - matches["learned"])
    advance = (
        all(v["ci95"][0] > 0 for v in paired.values())
        and absolute["mean"] >= 0
        and report["oracle"]["swapped_minus_matched"]["mean"] > 0
    )
    save(
        out / "report.json",
        dict(
            status="finished",
            subject="sub-04",
            n_training_classes=128,
            n_test_classes=48,
            n_test_photos=96,
            absolute_mse_baseline_minus_learned=absolute,
            paired_learned_contrast_advantage=paired,
            arms=report,
            advance_locked_generation_gate=bool(advance),
            runtime=runtime,
            elapsed_seconds=time.perf_counter() - start,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            driver_sha256=digest(Path(__file__)),
            head_sha256=head_sha,
            heldout_sha256=digest(out / "heldout.json"),
            **receipt,
        ),
    )
    print(json.dumps(dict(arms=report, paired=paired, advance=advance), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--class-batch", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    data, design, receipt = validate(a.root)
    if a.check_only:
        # Exercise the frozen geometry with synthetic teacher gradients, without the UNet or held-out oracle.
        g = np.random.default_rng(31).normal(size=(128, 1024))
        h = fit_heads(data, g)
        for k in ("learned", "sham"):
            pred = apply_head(data["prior"][:5], data["own"][:5], h["basis"], h[k])
            require(
                np.max(
                    abs(
                        angular(data["prior"][:5], pred)[2]
                        - angular(data["prior"][:5], data["own"][:5])[2]
                    )
                )
                < 1e-6,
                "Angular check failed",
            )
        print("Verified packet, teacher-to-head mapping and held-angle geometry", flush=True)
    else:
        require(a.archive and a.adapter, "Supply --archive and --adapter")
        evaluate_physical(a, data, design, receipt)


if __name__ == "__main__":
    main()
