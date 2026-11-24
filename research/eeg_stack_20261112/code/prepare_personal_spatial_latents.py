"""Register then encode missing 8x8 scaled Turbo VAE modes with historical numeric controls."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

FULL_SHA = "b1198b3e07361e3b941f87e3b60146b1cc7957293e813db69f7b59cfa8469a2d"
CONFIG_SHA = "85a181424a5fd0cc292e44e467be66a5263976eee1baf36d208133f9d49d4269"
SEED = 260512


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def save(p, data):
    p.parent.mkdir(exist_ok=True, parents=True)
    raw = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    if p.exists() and p.read_bytes() != raw:
        raise ValueError(f"Existing receipt differs at {p}")
    p.write_bytes(raw)


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def manifest(a):
    reg = read(a.out / "source/registration.json")
    complete = read(a.out / "source/completion.json")
    split = read(a.fresh / "source/current_split_manifest.json")["image_rows"]
    historical = np.asarray(reg["array_rows"])
    check(
        complete["sha256"] == FULL_SHA
        and complete["array_rows"] == split["T"] == reg["array_rows"]
        and reg["VAE_config_sha256"] == CONFIG_SHA
        and reg["VAE_scaling_factor"] == 0.13025,
        "Historical row identities, latent units or Turbo VAE registration differ",
    )
    feature_file = a.expanded / "features/sub-04.json"
    classes = np.asarray(read(feature_file)["training_concepts"])
    check(len(np.unique(classes)) == 512, "Invalid personal training receipt")
    generic = (classes[:, None] * 10 + np.arange(10)).ravel()
    panel = np.concatenate(
        [read(a.references / (p + "_ablation.json"))["rows"] for p in ("registered", "replicate")]
    )
    fresh = read(a.fresh / "references/fresh_ablation.json")["rows"]
    need = np.setdiff1d(np.r_[generic, panel, fresh], historical)
    check(
        len(np.intersect1d(generic, np.r_[panel, fresh])) == 0
        and len(need) == 1020
        and len(np.setdiff1d(generic, historical)) == 970,
        "Unexpected request set",
    )
    # Independent encoder calibration uses 24 generic, historical-T photographs only.
    parity = np.sort(
        np.random.default_rng(SEED).choice(np.intersect1d(generic, historical), 24, replace=False)
    )
    save(
        a.out / "source/extension_rows.json",
        dict(
            generic_classes=classes.tolist(),
            rows=need.tolist(),
            parity_rows=parity.tolist(),
            historical_registration_sha256=sha(a.out / "source/registration.json"),
            full_historical_sha256=FULL_SHA,
            generic_receipt_sha256=sha(feature_file),
            n_missing_generic=970,
            n_missing_old_panel_supports=50,
            source="Historical archive row indices, 8x8 spatial means of scaled 4x64x64 posterior modes, NCHW.",
        ),
    )
    print("Extension", len(need), "rows and", len(parity), "historical generic controls")


def encoder(a, reg):
    import torch
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors import safe_open

    root = a.archive / "assets/native_generator_v243/turbo"
    cfg = root / "vae/config.json"
    check(sha(cfg) == CONFIG_SHA, "Wrong VAE config")
    with safe_open(
        str(root / "sd_xl_turbo_1.0_fp16.safetensors"), framework="pt", device="cpu"
    ) as file:
        tensors = {k: file.get_tensor(k) for k in file.keys() if k.startswith("first_stage_model.")}
    values = convert_ldm_vae_checkpoint(tensors, read(cfg))
    h = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        h.update(name.encode())
        h.update(tensor.detach().contiguous().numpy().tobytes())
    check(
        h.hexdigest() == reg["original_FP16_VAE_state_sha256"],
        "Actual VAE weights differ from historical encoder",
    )
    vae = AutoencoderKL.from_config(read(cfg))
    vae.load_state_dict(values, strict=True)
    vae = vae.cuda().float().eval().requires_grad_(False)
    vae.enable_slicing()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return vae


def encode(a):
    import torch
    from PIL import Image
    from torchvision import transforms

    base = a.archive / "data/native_T_latents_v256"
    reg = read(base / "registration.json")
    req = read(a.out / "source/extension_rows.json")
    check(
        sha(base / "registration.json") == req["historical_registration_sha256"]
        and read(base / "completion.json")["sha256"] == FULL_SHA,
        "Archive and extension plan do not share a VAE bank",
    )
    rows = np.asarray(req["rows"] + req["parity_rows"])
    check(len(np.unique(rows)) == len(rows), "Duplicate encoding row")
    manifest_path = a.archive / "data/full_T_recovery_20261108_v238/image_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        records = {
            int(r["array_row"]): r for r in csv.DictReader(f) if r["dataset_split"] == "train"
        }
    check(len(records) == 16540, "Expected all 16540 manifest rows")
    vae = encoder(a, reg)
    pixels = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    values = []
    archive = a.archive / "assets/things_images_v254/training_images.zip"
    with ZipFile(archive) as z, torch.inference_mode():
        for lo in range(0, len(rows), 4):
            batch = []
            for row in rows[lo : lo + 4]:
                r = records[row]
                raw = z.read(r["image_member"])
                check(
                    len(raw) == int(r["image_bytes"])
                    and hashlib.sha256(raw).hexdigest() == r["image_sha256"],
                    "Original row/photo digest mismatch",
                )
                with Image.open(io.BytesIO(raw)) as im:
                    batch.append(pixels(im.convert("RGB")))
            latent = (
                vae.encode(torch.stack(batch).cuda()).latent_dist.mode().float()
                * vae.config.scaling_factor
            )
            v = latent.cpu().numpy()
            check(v.shape[1:] == (4, 64, 64) and np.isfinite(v).all(), "Invalid VAE modes")
            values.append(v.reshape(-1, 4, 8, 8, 8, 8).mean((3, 5)))
            if lo % 200 == 0:
                print("Encoded", lo + len(batch), "/", len(rows), flush=True)
    values = np.concatenate(values)
    actual = values[len(req["rows"]) :]
    ix = {r: i for i, r in enumerate(reg["array_rows"])}
    tbank = np.load(base / "native_VAE_T_modes.npy", mmap_mode="r")
    expected = tbank[[ix[r] for r in req["parity_rows"]]].reshape(-1, 4, 8, 8, 8, 8).mean((3, 5))
    error = np.abs(actual - expected)
    check(
        np.allclose(actual, expected, rtol=2e-4, atol=2e-4),
        "On overlapping generic photos re-encoded modes do not match historical values",
    )
    dest = a.out / "source/extension8.npz"
    np.savez(
        dest,
        rows=rows,
        latents=values,
        registration_sha256=sha(base / "registration.json"),
        request_sha256=sha(a.out / "source/extension_rows.json"),
    )
    save(
        a.out / "source/extension8.json",
        dict(
            output_sha256=sha(dest),
            max_abs_overlap_error=float(error.max()),
            mean_abs_overlap_error=float(error.mean()),
            n_overlap=len(actual),
            n_new=len(req["rows"]),
            image_manifest_sha256=sha(manifest_path),
            mode="FP32 scaled posterior modes, mean of 8x8 blocks",
            vae_state_sha256=reg["original_FP16_VAE_state_sha256"],
            new_rows_sha256=sha(a.out / "source/extension_rows.json"),
        ),
    )
    print("Validated and saved", dest, "max overlap error", error.max(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifest", "encode"))
    parser.add_argument("--archive", type=Path, help="Historical archive (encode)")
    for k, d in dict(
        out="outputs/latent-spatial",
        fresh="outputs/fresh-photo",
        expanded="outputs/photo-structure/expanded",
        references="outputs/object-fidelity",
    ).items():
        parser.add_argument("--" + k, type=Path, default=Path(d))
    args = parser.parse_args()
    if args.stage == "manifest":
        manifest(args)
    elif args.archive is not None:
        encode(args)
    else:
        parser.error("encode requires --archive")
