"""Move pixels in a fixed EEG reconstruction toward an EEG-predicted local DINO feature map."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from fit_shared_photo_eeg import load_photos, read, require, write
from fit_spatial_photo import original, other_eight
from PIL import Image
from scipy.linalg import cho_factor, cho_solve
from score_reconstruction_jpegs import registered
from score_session_matched_images import case
from score_within_photo_dino import FILES, reference, sha
from test_cohort_session_matched import NAMES, compact

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
PLAN = PUBLIC / "patch_transport_design.json"
SUBJECT = "sub-04"
BLIND = ("neutral", "prior", "own", "wrong")


def patch_encoder(a):
    from transformers import Dinov2Model

    require(all(sha(a.model / f) == v for f, v in FILES.items()), "DINO revision differs")
    model = Dinov2Model.from_pretrained(a.model, local_files_only=True).eval().to(a.device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=a.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=a.device)[None, :, None, None]

    @torch.inference_mode()
    def encode(images):
        raw = np.stack([np.array(im.resize((224, 224), Image.Resampling.BICUBIC)) for im in images])
        pixels = torch.tensor(raw, device=a.device).permute(0, 3, 1, 2).float() / 255
        feat = model(pixel_values=(pixels - mean) / std).last_hidden_state[:, 1:]
        return F.normalize(feat, dim=-1).transpose(1, 2).reshape(-1, 384, 16, 16)

    return encode


def pool(rows, encode, archive, tab, batch):
    chunks = []
    for off in range(0, len(rows), batch):
        images = [original(archive, tab, int(r)) for r in rows[off : off + batch]]
        chunks.append(F.avg_pool2d(encode(images), 4).flatten(1).cpu().numpy())
    return np.concatenate(chunks)


def ridge(x, y):
    dx = (x - x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])
    dy = (y - y.mean(1, keepdims=True)).reshape(-1, y.shape[-1])
    scale = dx.std(0).clip(1e-8)
    dx /= scale
    xx = dx.T @ dx
    xx.flat[:: len(xx) + 1] += 2 * len(dx)
    return cho_solve(cho_factor(xx), dx.T @ dy) / scale[:, None]


def fit(a, tab, archive, encode):
    classes = np.array(read(a.expanded / "features" / (SUBJECT + ".json"))["training_concepts"])
    split = read(PUBLIC / "shared_photo_eeg_design.json")
    tr, ev = (np.flatnonzero(np.isin(classes, split[k])) for k in ("train_classes", "test_classes"))
    require(len(tr) == 416 and len(ev) == 96 and not np.intersect1d(tr, ev).size, "Split differs")
    x = load_photos(a, SUBJECT, classes).astype("float32")
    all_rows = (10 * classes[:, None] + np.arange(10)).ravel()
    y = pool(all_rows, encode, archive, tab, a.batch).reshape(512, 10, -1)
    m = ridge(x[tr], y[tr])
    pred = (x[ev] - other_eight(x[ev])) @ m
    residual = y[ev] - other_eight(y[ev])
    gain = float(np.clip((pred * residual).sum() / np.square(pred).sum(), 0, 1))
    base_mse = np.square(residual).mean()
    fitted = a.out / "fit.npz"
    np.savez(fitted, weight=ridge(x, y), gain=gain, classes=classes)
    write(
        a.out / "fit.json",
        dict(
            plan=sha(PLAN),
            fit=sha(fitted),
            gain=gain,
            calibration_r2=float(1 - np.square(residual - gain * pred).mean() / base_mse),
            split=sha(PUBLIC / "shared_photo_eeg_design.json"),
            train_objects=416,
            gain_objects=96,
        ),
    )
    print("Fitted 512 objects with generic calibration gain", gain, flush=True)


def sampling_grid(knots, side, limited=False):
    # Knots have zero displacement on the boundary. Normalized coordinates span [-1, 1].
    field = F.pad(0.28 * knots.tanh(), (1, 1, 1, 1))
    if limited:
        dx = field.diff(dim=3).square().sum(1).amax(dim=(1, 2))
        dy = field.diff(dim=2).square().sum(1).amax(dim=(1, 2))
        # Bilinear derivatives interpolate two adjacent edges. The coarsest grid has
        # five knot intervals across a normalized width of 2 * 15 / 16.
        bound = (5 * 16 / 30) * (dx + dy).clamp_min(1e-12).sqrt()
        field = field / (bound / 0.25).clamp_min(1)[:, None, None, None]
    delta = F.interpolate(
        field,
        size=(side, side),
        mode="bilinear",
        align_corners=True,
    )
    coords = (torch.arange(side, device=knots.device) + 0.5) * (2 / side) - 1
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1) + delta.permute(0, 2, 3, 1)
    return grid, delta


def transport(source, patches, target, limited):
    knots = torch.zeros(len(source), 2, 4, 4, device=patches.device, requires_grad=True)
    optimizer = torch.optim.Adam([knots], lr=0.12)
    for _ in range(32):
        grid, delta = sampling_grid(knots, 16, limited)
        moved = F.grid_sample(patches, grid, padding_mode="border", align_corners=False)
        loss = (
            F.avg_pool2d(moved, 4) - target
        ).square().mean() * 384 + 0.02 * delta.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        grid, _ = sampling_grid(knots, 512, limited)
        moved = F.grid_sample(source, grid, padding_mode="border", align_corners=False)
        return moved.permute(0, 2, 3, 1).round().clamp(0, 255).byte().cpu().numpy()


def registered_case(a, panel):
    reg = registered(a.root / "references", panel)
    settings = case(
        SimpleNamespace(render=a.root / "render_inputs", root=a.root, ref=a.root / "references"),
        panel,
        SUBJECT,
        reg,
    )
    check = read(PUBLIC / "cohort_session_images.json")["panels"][panel]["inputs"][SUBJECT]
    base = a.root / "render_inputs/renders" / panel / SUBJECT
    require(sha(base / "settings.json") == check["settings_sha256"], "Base image settings changed")
    return reg, settings, base


def render(a, panel, tab, archive, encode, oracle):
    reg, settings, base = registered_case(a, panel)
    with np.load(a.fit_dir / "fit.npz", allow_pickle=False) as fit_file:
        weight, gain, classes = [fit_file[k] for k in ("weight", "gain", "classes")]
    require(read(a.fit_dir / "fit.json")["fit"] == sha(a.fit_dir / "fit.npz"), "Fit file changed")
    rows = reg["rows"]
    concepts = np.unique(rows // 10)
    require(not np.intersect1d(classes, concepts).size, "Calibration/target object overlap")
    all_rows = (10 * concepts[:, None] + np.arange(10)).ravel()
    x, eeg_digest = compact(SimpleNamespace(double=a.double, out=a.root), SUBJECT, panel, concepts)
    require(
        eeg_digest == read(a.root / panel / "result.json")["target_feature_receipts"][SUBJECT],
        "Target EEG source differs",
    )
    idx = np.searchsorted(all_rows, rows)
    sup = np.searchsorted(all_rows, settings["support_rows"])
    require(
        np.array_equal(all_rows[idx], rows)
        and np.array_equal(all_rows[sup], settings["support_rows"])
        and all(
            not np.intersect1d(all_rows[sup[i]], rows[reg["folds"] == f]).size
            for i, f in enumerate(reg["folds"])
        ),
        "Visual support outside allowed fold",
    )
    # A photo may be a support for a different fold. Only the eight permitted vectors
    # enter the prior for any one query. Oracle access is a separate invocation.
    bank = pool(all_rows, encode, archive, tab, a.batch)
    if oracle:
        targets = {"oracle": bank[idx]}
    else:
        prior = bank[sup].mean(1)
        predicted = ((x[idx] - x[sup].mean(1)).astype("float32") @ weight) * gain
        lookup = {int(r): i for i, r in enumerate(rows)}
        donors = [lookup[r] for r in settings["donor_rows"]]
        require(np.array_equal(sup, sup[donors]), "Donors have different visual supports")
        targets = dict(prior=prior, own=prior + predicted, wrong=prior + predicted[donors])
    arms = ["oracle"] if oracle else list(BLIND)
    root = a.out / panel
    for arm in arms:
        (root / arm).mkdir(exist_ok=True, parents=True)
    hashes = {arm: [] for arm in arms}
    for off in range(0, len(rows), a.batch):
        imgs = []
        for i in range(off, min(off + a.batch, len(rows))):
            src = base / "images/own" / f"{i:03d}.jpg"
            require(sha(src) == settings["jpeg_sha256"]["own"][i], "Base JPEG changed")
            with Image.open(src) as im:
                imgs.append(im.convert("RGB"))
        descriptors = encode(imgs).clone()
        pixels = (
            torch.tensor(np.stack([np.array(im) for im in imgs]), device=a.device)
            .permute(0, 3, 1, 2)
            .float()
        )
        for arm in arms:
            images = imgs
            if arm != "neutral":
                target = torch.tensor(targets[arm][off : off + len(imgs)], device=a.device).reshape(
                    -1, 384, 4, 4
                )
                images = [
                    Image.fromarray(arr)
                    for arr in transport(pixels, descriptors, target, a.limited)
                ]
            for j, im in enumerate(images, off):
                dest = root / arm / f"{j:03d}.jpg"
                im.save(dest, quality=95)
                hashes[arm].append(sha(dest))
    record = dict(
        plan=sha(a.plan),
        fit=sha(a.fit_dir / "fit.npz"),
        panel=panel,
        rows=rows.tolist(),
        settings=sha(base / "settings.json"),
        jpeg=hashes,
    )
    write(root / ("oracle.json" if oracle else "blind.json"), record)
    print(panel, arms, len(rows), flush=True)


def layout(a):
    a.fit_dir = a.out
    a.plan = PLAN
    if a.limited:
        a.out = a.out / "limited"
        a.plan = PUBLIC / "patch_transport_limited_design.json"


def main(a):
    layout(a)
    require(not (a.limited and a.stage == "fit"), "The limited warp reuses the original fit")
    a.out.mkdir(exist_ok=True, parents=True)
    if a.stage == "oracle":
        lock = read(a.out / "blind_lock.json")
        require(all(lock[p] == sha(a.out / p / "blind.json") for p in NAMES), "Missing blind lock")
    table = reference(a)
    encode = patch_encoder(a)
    with ZipFile(a.originals) as archive:
        if a.stage == "fit":
            fit(a, table, archive, encode)
        else:
            for panel in NAMES:
                render(a, panel, table, archive, encode, a.stage == "oracle")
            if a.stage == "blind":
                write(a.out / "blind_lock.json", {p: sha(a.out / p / "blind.json") for p in NAMES})


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "blind", "oracle"))
    for name, path in dict(
        out="outputs/patch-transport",
        root="outputs/session-matched-cohort",
        expanded="outputs/photo-structure/expanded",
        double="outputs/double-calibration",
        originals="outputs/photo-structure/training_images.zip",
        manifest="outputs/official-object-readout/sub-03/atm_train_rows.csv",
        model="outputs/photo-structure/dino/facebook-dinov2-small",
    ).items():
        p.add_argument("--" + name, type=Path, default=Path(path))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limited", action="store_true")
    return p


if __name__ == "__main__":
    main(arguments().parse_args())
