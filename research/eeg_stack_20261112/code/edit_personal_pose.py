"""Bounded affine transfer of predicted foreground pose onto frozen personal R4 JPEGs."""

import argparse
import io
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, save
from fit_support_contour import segmentation
from fit_support_structure import population
from measure_foreground_geometry import MODEL_SHA, ZIP_SHA
from PIL import Image
from render_official_words import crop512
from render_within_photo_ip import check, packet, read, sha
from scipy.ndimage import affine_transform, binary_dilation, binary_erosion

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
LOCK = "83e7551af17a3425233888d16942fc400740615f44dc58fcc576a09a74a72a5f"
SUBJECTS = ("sub-04", "sub-05", "sub-01")
ARMS = ("neutral", "prior", "own", "wrong")
KEYS = ("center_x", "center_y", "log_radius", "anisotropy_xx_yy", "anisotropy_xy")


def mask(model, im):
    px = np.asarray(
        crop512(im.convert("RGB")).resize((320, 320), Image.Resampling.LANCZOS), np.float32
    )
    t = (px / max(float(px.max()), 1) - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    pred = model.run(
        [model.get_outputs()[0].name],
        {model.get_inputs()[0].name: np.ascontiguousarray(t.transpose(2, 0, 1)[None])},
    )[0][0, 0]
    mass = (pred - pred.min()).clip(0)
    check(np.isfinite(mass).all() and mass.max() > 1e-8, "Degenerate U2NetP foreground")
    return mass / mass.max()


def pose(mass):
    h, w = mass.shape
    x = (np.arange(w) + 0.5) / w
    y = (np.arange(h) + 0.5) / h
    weight = mass.astype(float) / mass.sum(dtype=float)
    cx = weight.sum(0) @ x
    cy = weight.sum(1) @ y
    dx, dy = x - cx, y - cy
    xx = weight.sum(0) @ (dx * dx)
    yy = weight.sum(1) @ (dy * dy)
    xy = dy @ weight @ dx
    trace = xx + yy
    check(trace > 1e-10, "Foreground covariance is singular")
    return np.array([cx, cy, 0.5 * np.log(trace), (xx - yy) / trace, 2 * xy / trace])


def frame(p):
    a, b = p[3:]
    magnitude = min(float(np.hypot(a, b)), 0.999999)
    theta = 0.5 * np.arctan2(b, a)
    scale = np.exp(p[2]) * np.sqrt(np.array([1 + magnitude, 1 - magnitude]) * 0.5)
    return theta, magnitude, scale


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def geometry(src, dest, points, size):
    angle, anis, sd = frame(src)
    wanted, target_anis, target_sd = frame(dest)
    turn = (wanted - angle + np.pi / 2) % np.pi - np.pi / 2
    turn = np.clip(0.8 * turn * min(1, anis / 0.15, target_anis / 0.15), -np.pi / 6, np.pi / 6)
    ratio = np.exp(np.clip(0.8 * np.log(target_sd / sd), np.log(0.75), np.log(1.33)))
    linear = rotation(angle + turn) @ np.diag(ratio) @ rotation(-angle)
    center = src[:2]
    desired = center + np.clip(0.8 * (dest[:2] - center), -0.12, 0.12)
    limits = 0.5 / np.array(size)
    relative = points - center

    def feasible(weight):
        transform = np.eye(2) + weight * (linear - np.eye(2))
        mapped = relative @ transform.T
        lo = np.maximum(limits - mapped.min(0), center - 0.12)
        hi = np.minimum(1 - limits - mapped.max(0), center + 0.12)
        return np.all(lo <= hi + 1e-12), transform, lo, hi

    factor = 1.0
    if not feasible(1.0)[0]:
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) / 2
            if feasible(mid)[0]:
                lo = mid
            else:
                hi = mid
        factor = lo
    valid, matrix, lo, hi = feasible(factor)
    moved = np.clip(desired, lo, hi)
    check(valid and np.linalg.det(matrix) > 0, "Unable to preserve the foreground in frame")
    return (
        matrix,
        moved,
        dict(
            rotation_degrees_requested=float(np.degrees(turn)),
            principal_scale_requested=ratio.tolist(),
            matrix_fraction=float(factor),
            matrix=matrix.tolist(),
            center_applied=moved.tolist(),
        ),
    )


def resize_float(array, size, method):
    if array.ndim == 3:
        return np.stack([resize_float(v, size, method) for v in array.transpose(2, 0, 1)], axis=-1)
    return np.asarray(
        Image.fromarray(np.asarray(array, np.float32)).resize(size, method), np.float32
    )


def background(rgb, visible):
    """Fill covered pixels from the visible part of the SAME reconstruction."""
    mass = visible.astype(np.float32)
    levels = [(rgb * mass[:, :, None], mass)]
    while min(levels[-1][1].shape) > 1:
        weighted, mass = levels[-1]
        h, w = mass.shape
        size = ((w + 1) // 2, (h + 1) // 2)
        levels.append(
            (
                resize_float(weighted, size, Image.Resampling.BOX),
                resize_float(mass, size, Image.Resampling.BOX),
            )
        )
    weighted, mass = levels[-1]
    check(mass.mean() > 0, "No visible background remains")
    out = weighted / mass[:, :, None].clip(1e-12)
    for weighted, mass in reversed(levels[:-1]):
        up = resize_float(out, (mass.shape[1], mass.shape[0]), Image.Resampling.BILINEAR)
        out = weighted + (1 - mass[:, :, None]) * up
    return out


def scene(im, mass):
    h, w = im.height, im.width
    alpha = ((mass - 0.08) / (0.85 - 0.08)).clip(0, 1)
    alpha = resize_float(alpha * alpha * (3 - 2 * alpha), (w, h), Image.Resampling.BILINEAR)
    active = alpha > 0
    yy, xx = np.nonzero(active & ~binary_erosion(active, border_value=0))
    check(len(xx) > 1, "Missing foreground for transfer")
    boundary = np.column_stack(((xx + 0.5) / w, (yy + 0.5) / h))
    rgb = np.asarray(im, np.float32) / 255
    visible = ~binary_dilation(active, iterations=2)
    bg = background(rgb, visible) if np.any(visible) else None
    fg = None if bg is None else np.clip(rgb - (1 - alpha[:, :, None]) * bg, 0, alpha[:, :, None])
    return pose(mass), alpha, boundary, fg, bg


def warp(im, state, wanted):
    src, alpha, boundary, fg, bg = state
    matrix, moved, log = geometry(src, wanted, boundary, im.size)
    if bg is None or (
        np.allclose(matrix, np.eye(2), rtol=0, atol=1e-10)
        and np.allclose(moved, src[:2], rtol=0, atol=1e-10)
    ):
        log["identity"] = True
        return im.copy(), log
    size = np.array(im.size)
    pixel_matrix = np.diag(size) @ matrix @ np.diag(1 / size)
    inv = np.linalg.inv(pixel_matrix)
    offset = (src[:2] * size - 0.5) - inv @ (moved * size - 0.5)

    def transform(v):
        return affine_transform(
            v,
            inv[::-1, ::-1],
            offset=offset[::-1],
            order=1,
            mode="grid-constant",
            cval=0.0,
            prefilter=False,
        )

    opacity = transform(alpha).clip(0, 1)
    moved_fg = np.stack([transform(v) for v in fg.transpose(2, 0, 1)], axis=-1)
    edited = moved_fg + (1 - opacity[:, :, None]) * bg
    changed = (alpha > 0) | (opacity > 0)
    result = np.array(im)
    result[changed] = np.rint(255 * edited[changed]).clip(0, 255).astype(np.uint8)
    log["identity"] = False
    return Image.fromarray(result), log


def encoded(im):
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=95, subsampling=0)
    return b.getvalue()


def synthetic(a):
    h, w = 144, 160
    y, x = np.indices((h, w), dtype=float)

    def ellipse(cx, cy, t, sx, sy):
        uv = np.stack(((x + 0.5) / w - cx, (y + 0.5) / h - cy), axis=-1) @ rotation(np.deg2rad(t))
        return ((uv[:, :, 0] / sx) ** 2 + (uv[:, :, 1] / sy) ** 2 < 1).astype(np.float32)

    m = ellipse(0.45, 0.49, 9, 0.20, 0.075)
    test = ellipse(0.52, 0.51, 29, 0.23, 0.071)
    rgb = np.full((h, w, 3), [37, 103, 181], np.uint8)
    rgb[m > 0] = [223, 29, 31]
    im = Image.fromarray(rgb)
    st = scene(im, m)
    identity, _ = warp(im, st, st[0])
    check(np.array_equal(np.asarray(identity), rgb), "Identity failed")
    q = pose(test)
    out, log = warp(im, st, q)
    pix = np.array(out)
    recovered = pose((pix[:, :, 0] > 140).astype(float))
    before = abs(frame(st[0])[0] - frame(q)[0])
    after = abs(frame(recovered)[0] - frame(q)[0])
    check(
        after < 0.5 * before
        and np.linalg.norm(recovered[:2] - q[:2]) < 0.5 * np.linalg.norm(st[0][:2] - q[:2])
        and np.array_equal(pix[0], rgb[0])
        and np.array_equal(pix[:, -1], rgb[:, -1]),
        "Rotation or translation regression",
    )
    for location in ((0.02, 0.03), (0.99, 0.95), (0.1, 0.79)):
        edge = ellipse(*location, 70, 0.2, 0.12)
        original_state = scene(im, edge)
        matrix, center, _ = geometry(
            original_state[0], np.array([0.99, 0.01, -0.5, -0.8, 0.2]), original_state[2], im.size
        )
        points = (original_state[2] - original_state[0][:2]) @ matrix.T + center
        check(
            np.min(points - 0.5 / np.array(im.size)) > -2e-7
            and np.max(points - (1 - 0.5 / np.array(im.size))) < 2e-7,
            "Visible foreground escaped the frame",
        )
    save(
        a.root / "synthetic.json",
        dict(
            design_sha256=LOCK,
            identity_rgb_exact=True,
            centering_error_before=float(np.linalg.norm(st[0][:2] - q[:2])),
            centering_error_after=float(np.linalg.norm(recovered[:2] - q[:2])),
            angle_error_before_degrees=float(np.degrees(before)),
            angle_error_after_degrees=float(np.degrees(after)),
            edge_fixtures=3,
            fixed_background_edge_exact=True,
            fixture_transform=log,
        ),
    )


def oracle(a):
    from score_object_panel import score_object_panel
    from score_within_photo_dino import encoder

    gate = read(a.root / "synthetic.json")
    check(
        gate["design_sha256"] == LOCK and gate["identity_rgb_exact"], "Synthetic regression missing"
    )
    old = a.data / "fixed-word-photo-dose/render_inputs/renders/registered/single-1"
    s = read(old / "settings.json")
    check(
        s["subject"] == "single-1" and s["panel"] == "registered" and len(s["rows"]) == 120,
        "Oracle is restricted to the previous registered single-person panel",
    )
    refs = population(SimpleNamespace(source=a.data / "official-object-readout/sub-03"))
    archive = a.data / "photo-structure/training_images.zip"
    check(sha(archive) == ZIP_SHA, "Different historical reference bank")
    model = segmentation()
    root = a.root / "old_oracle"
    true = []
    measurements = {"neutral": [], "oracle": []}
    for arm in measurements:
        (root / arm).mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z:
        for i, r in enumerate(s["rows"]):
            p = old / "images/own" / f"{i:03d}.jpg"
            check(sha(p) == s["jpeg_sha256"]["own"][i], "Changed old source JPEG")
            with Image.open(p) as opened:
                im = opened.convert("RGB")
            real = original(z, refs, r)
            true.append(real)
            ref_pose = pose(mask(model, real))
            state = scene(im, mask(model, im))
            oracle_im, _ = warp(im, state, ref_pose)
            for arm, result in (("neutral", im), ("oracle", oracle_im)):
                f = root / arm / f"{i:03d}.jpg"
                f.write_bytes(encoded(result))
                with Image.open(f) as opened:
                    actual = opened.convert("RGB")
                now = pose(mask(model, actual))
                diff = now - ref_pose
                pixels = (np.asarray(actual, np.float32) - np.asarray(real, np.float32)) / 255
                measurements[arm].append(
                    [
                        np.linalg.norm(diff[:2]),
                        abs(diff[2]),
                        np.linalg.norm(diff[3:]),
                        np.sqrt(np.mean(pixels**2)),
                    ]
                )
    rows = np.array(s["rows"])
    labels = rows // 10
    classes = np.unique(labels)
    check(len(classes) == 12, "Wrong old photo population")
    dino = encoder(
        SimpleNamespace(
            model=a.data / "photo-structure/dino/facebook-dinov2-small",
            device=a.device,
            batch_size=16,
        )
    )
    real_features = dino(true).astype(float)
    rowscores = {}
    metrics = (
        "center_l2",
        "absolute_log_radius",
        "anisotropy_l2",
        "rgb_rmse512",
        "dino_within10",
        "dino_object12",
    )
    means = {}
    for arm, v in measurements.items():
        ims = []
        for i in range(120):
            with Image.open(root / arm / f"{i:03d}.jpg") as opened:
                ims.append(opened.convert("RGB"))
        d = score_object_panel(dino(ims).astype(float) @ real_features.T, rows, labels)["per_photo"]
        values = np.column_stack((v, [[t["intra10_top1"], t["object_top1"]] for t in d]))
        rowscores[arm] = values
        means[arm] = {
            k: float(values[:, j].mean() if j < 4 else values[:, j].sum())
            for j, k in enumerate(metrics)
        }
    delta = (rowscores["neutral"] - rowscores["oracle"]) * np.array([1, 1, 1, 1, -1, -1])
    class_means = np.stack([delta[labels == k].mean(0) for k in classes])
    boot = class_means[np.random.default_rng(261130).integers(12, size=(10000, 12))].mean(1)
    paired = {
        k: dict(
            favorable_delta=float(delta[:, j].mean()),
            ci95=np.quantile(boot[:, j], [0.025, 0.975]).tolist(),
        )
        for j, k in enumerate(metrics)
    }
    passed = bool(delta[:, 2].mean() > 0 and (delta[:, 0].mean() > 0 or delta[:, 1].mean() > 0))
    report = dict(
        design_sha256=LOCK,
        source_settings_sha256=sha(old / "settings.json"),
        stimulus_zip_sha256=ZIP_SHA,
        foreground_sha256=MODEL_SHA,
        n=120,
        objects=12,
        group="registered/single-1",
        metrics=means,
        units="DINO entries are correct credits out of 120. Paired DINO effects are rates. Center and RGB measures use [0,1] image units.",
        oracle_minus_neutral_favorable=paired,
        passes_mechanism_gate=passed,
    )
    np.savez(root / "score_rows.npz", rows=rows, **rowscores)
    save(root / "summary.json", report)
    if a.publish:
        save(PUBLIC / "personal12_pose_old_oracle.json", report)
    print("Old-panel mechanism", passed, means, flush=True)


def verify_packet(a, subject):
    src = (
        a.data / "trial-object-encoder/personal12"
        if subject == "sub-04"
        else a.data / "trial-object-replication" / subject
    ) / "render_inputs"
    rows, folds, words, donors, support, _, _, h14_sha = packet(src, subject, "personal12")
    folder = src / "renders/personal12" / subject
    sf = folder / "settings.json"
    s = read(sf)
    check(
        s["subject"] == subject
        and s["panel"] == "personal12"
        and s["packet_sha256"] == h14_sha
        and s["design_sha256"] == sha(src / "design.json")
        and s["rows"] == rows.tolist()
        and s["folds"] == folds.tolist()
        and s["predicted_words"] == words.tolist()
        and s["support_rows"] == support.tolist()
        and s["donor_rows"] == donors.tolist()
        and s["seeds"] == (923761 + rows).tolist(),
        "Changed source EEG/image conditions",
    )
    location = a.root / "personal12" / subject
    po = location / "pose.npz"
    receipt = location / "receipt.json"
    rec = read(receipt)
    with np.load(po, allow_pickle=False) as z:
        check(
            all(
                np.array_equal(z[k], v)
                for k, v in dict(
                    rows=rows,
                    folds=folds,
                    predicted_words=words,
                    support_rows=support,
                    donor_rows=donors,
                ).items()
            )
            and z["keys"].tolist() == list(KEYS),
            "Pose predictions and mixed-head image populations differ",
        )
        targets = {
            arm: z[k].copy()
            for arm, k in (("prior", "uniform"), ("own", "own"), ("wrong", "wrong"))
        }
    lookup = {r: j for j, r in enumerate(rows)}
    donor = np.array([lookup[r] for r in donors])
    check(
        all(
            v.shape == (120, 5)
            and np.isfinite(v).all()
            and np.all((v[:, :2] >= 0) & (v[:, :2] <= 1))
            and np.all(np.linalg.norm(v[:, 3:], axis=1) <= 0.9900001)
            for v in targets.values()
        )
        and np.allclose(targets["prior"], targets["prior"][donor], rtol=0, atol=1e-9)
        and np.allclose(targets["wrong"], targets["own"][donor], rtol=0, atol=1e-9),
        "Invalid shape or mismatched-EEG assignment",
    )
    check(sha(po) in rec.values(), "Unsealed pose packet")
    return (
        folder,
        s,
        targets,
        dict(
            subject=subject,
            panel="personal12",
            packet_sha256=sha(po),
            pose_receipt_sha256=sha(receipt),
            image_design_sha256=LOCK,
            source_settings_sha256=sha(sf),
            source_packet_sha256=h14_sha,
            source_arm="own",
            source_jpeg_sha256=s["jpeg_sha256"]["own"],
            **{
                k: s[k]
                for k in ("rows", "folds", "predicted_words", "support_rows", "donor_rows", "seeds")
            },
        ),
    )


def edit(a):
    gate = read(a.root / "old_oracle/summary.json")
    check(
        gate["design_sha256"] == LOCK and gate["passes_mechanism_gate"],
        "Old mechanism gate missing",
    )
    inputs = {p: verify_packet(a, p) for p in SUBJECTS}
    model = segmentation()
    root = a.root / "renders"
    panels = {}
    for sub, (folder, s, targets, meta) in inputs.items():
        output = root / "personal12" / sub
        for arm in ARMS:
            (output / "images" / arm).mkdir(parents=True, exist_ok=True)
        hashes = {k: [] for k in ARMS}
        logs = []
        for i, digest in enumerate(s["jpeg_sha256"]["own"]):
            src = folder / "images/own" / f"{i:03d}.jpg"
            check(sha(src) == digest, "Source reconstruction changed")
            with Image.open(src) as opened:
                check(
                    opened.size == (512, 512) and opened.format == "JPEG" and opened.mode == "RGB",
                    "Incorrect fixed JPEG format",
                )
                im = opened.copy()
            state = scene(im, mask(model, im))
            info = {}
            for k in ARMS:
                if k == "neutral":
                    result = im
                else:
                    result, info[k] = warp(im, state, targets[k][i])
                p = output / "images" / k / f"{i:03d}.jpg"
                temp = p.with_suffix(".partial")
                temp.write_bytes(encoded(result))
                temp.replace(p)
                hashes[k].append(sha(p))
            logs.append(info)
        meta.update(
            arm_names=list(ARMS),
            jpeg_sha256=hashes,
            foreground_model_sha256=MODEL_SHA,
            editor_sha256=sha(Path(__file__)),
        )
        save(output / "applied_transforms.json", dict(rows=s["rows"], transforms=logs))
        meta["applied_transforms_sha256"] = sha(output / "applied_transforms.json")
        save(output / "settings.json", meta)
        panels[sub] = sha(output / "settings.json")
        print("Sealed", sub, len(logs) * 4, "edited JPEGs", flush=True)
    save(
        root / "blind_images.json",
        dict(
            image_design_sha256=LOCK,
            editor_sha256=sha(Path(__file__)),
            cases=panels,
            gate_sha256=sha(a.root / "old_oracle/summary.json"),
            generated_jpegs=1440,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("synthetic", "old-oracle", "edit"))
    p.add_argument("--root", type=Path, default=Path("outputs/personal-pose"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--publish", action="store_true", help="Release old-panel oracle aggregates")
    a = p.parse_args()
    check(sha(PUBLIC / "personal12_pose_image_design.json") == LOCK, "Changed registered editor")
    if a.stage == "synthetic":
        synthetic(a)
    elif a.stage == "old-oracle":
        oracle(a)
    else:
        edit(a)
