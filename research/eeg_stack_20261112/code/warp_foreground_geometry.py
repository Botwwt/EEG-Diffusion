"""Apply frozen EEG foreground geometry to existing generated JPEGs using a boundary-preserving warp."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from eval_instance_starts import write_json
from measure_foreground_geometry import MODEL_SHA, digest, measure
from PIL import Image
from scipy.interpolate import PchipInterpolator

ARMS = ("baseline", "neutral", "prior", "query", "swapped")
STRENGTH = 0.8
MAX_SHIFT = 0.12
RATIO_LIMITS = (0.75, 1.33)
BORDER = 1 / 16


def load_case(args, panel, person):
    geom = args.geometry / person / panel / "geometry.npz"
    source = args.results / panel / person / "results.json"
    reg = args.registrations / panel / person / "evaluation.json"
    prior = json.loads(source.read_text(encoding="utf8"))
    registration = json.loads(reg.read_text(encoding="utf8"))
    with np.load(geom, allow_pickle=False) as z:
        rows, donors = z["rows"], z["donor_rows"]
        targets = {k: z[k] for k in ("prior", "query", "swapped")}
    folds = np.asarray(registration["folds"])
    words = np.asarray(prior["predicted_words"])
    if (
        rows.shape != (120,)
        or rows.dtype.kind not in "iu"
        or len(set(rows)) != 120
        or donors.shape != rows.shape
        or donors.dtype.kind not in "iu"
        or set(rows) != set(donors)
        or registration["subject"] != person
        or registration["rows"] != rows.tolist()
        or prior["rows"] != rows.tolist()
        or folds.shape != rows.shape
        or words.shape != rows.shape
        or set(folds) != set(range(5))
        or not np.array_equal(
            words,
            np.asarray(registration["words"])[registration["results"]["personal_r4"]["predicted"]],
        )
        or any(v.shape != (120, 4) or not np.isfinite(v).all() for v in targets.values())
    ):
        raise ValueError(f"Case registration mismatch in {geom}")
    pos = {row: i for i, row in enumerate(rows)}
    idx = np.array([pos[j] for j in donors])
    if not np.array_equal(words, words[idx]) or not np.array_equal(folds, folds[idx]):
        raise ValueError(f"Donor word or fold mismatch in {geom}")
    residual = targets["query"] - targets["prior"]
    if not np.allclose(targets["swapped"] - targets["prior"], residual[idx], rtol=0, atol=1e-12):
        raise ValueError(f"Swapped residual does not correspond to the recorded donor in {geom}")
    origin = dict(
        geometry_sha256=digest(geom),
        source_results_sha256=digest(source),
        conditions_sha256=prior["condition_sha"],
        registration_sha256=digest(reg),
    )
    expected = {}
    if args.jpeg_receipts:
        path = args.jpeg_receipts / panel / person / "photo_layout.json"
        receipt = json.loads(path.read_text(encoding="utf8"))
        if receipt["personal"]["results_sha256"] != origin["source_results_sha256"]:
            raise ValueError(f"Prior JPEG receipt differs from the frozen render: {path}")
        expected = {a["row"]: a["jpeg_sha256"]["personal_aligned"] for a in receipt["per_photo"]}
        if set(expected) != set(rows):
            raise ValueError(f"Incomplete prior JPEG receipt in {path}")
        origin["jpeg_receipt_sha256"] = digest(path)
    return rows, folds, words, donors, targets, origin, expected


def inverse_axis(size, center, log_sd, wanted_center, wanted_log_sd):
    pixels = np.arange(size, dtype=np.float64)
    # The fixed endpoints are the centers of the first and last image pixels.
    left, right = 0.5 / size, 1 - 0.5 / size
    if not left < center < right:
        raise ValueError("Foreground centroid lies outside the image interior")
    moved = np.clip(
        center + np.clip(STRENGTH * (wanted_center - center), -MAX_SHIFT, MAX_SHIFT),
        left + 1e-5,
        right - 1e-5,
    )
    ratio = np.exp(np.clip(STRENGTH * (wanted_log_sd - log_sd), *np.log(RATIO_LIMITS)))
    sd = np.exp(log_sd)
    # Shorten a landmark pair proportionally if ±1 SD reaches the frame before or after the warp.
    lo = min(sd, 0.95 * (center - left), 0.95 * (moved - left) / ratio)
    hi = min(sd, 0.95 * (right - center), 0.95 * (right - moved) / ratio)
    orig = [left, center - lo, center, center + hi, right]
    dest = [left, moved - ratio * lo, moved, moved + ratio * hi, right]
    coord = PchipInterpolator(dest, orig, extrapolate=False)((pixels + 0.5) / size) * size - 0.5
    coord[[0, -1]] = pixels[[0, -1]]
    if not np.isfinite(coord).all() or np.any(np.diff(coord) <= 0):
        raise ValueError("Nonmonotone displacement")
    return coord, float(moved), float(ratio)


def taper(t, n):
    d = (np.minimum(t, n - 1 - t) / ((n - 1) * BORDER)).clip(0, 1)
    return d * d * (3 - 2 * d)


def warp(rgb, source, target):
    h, w = rgb.shape[:2]
    xx = np.arange(w)[None, :]
    yy = np.arange(h)[:, None]
    mx, cx, rx = inverse_axis(w, source[0], source[2], target[0], target[2])
    my, cy, ry = inverse_axis(h, source[1], source[3], target[1], target[3])
    # Compose two monotone 1-D maps. At the outer frame both displacements vanish;
    # in the interior each reaches its prescribed center and spread landmarks.
    y = yy + (my[:, None] - yy) * taper(xx, w)
    x = xx + (mx[None, :] - xx) * taper(y, h)
    x, y = x.clip(0, w - 1), y.clip(0, h - 1)
    x0, y0 = x.astype(np.intp), y.astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0).astype(np.float32)[..., None], (y - y0).astype(np.float32)[..., None]
    top = rgb[y0, x0] * (1 - fx) + rgb[y0, x1] * fx
    bottom = rgb[y1, x0] * (1 - fx) + rgb[y1, x1] * fx
    output = np.rint(top * (1 - fy) + bottom * fy).clip(0, 255).astype(np.uint8)
    return output, [cx, cy, rx, ry]


def jpeg(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run_case(args, session, panel, person):
    rows, folds, words, donors, targets, origin, expected = load_case(args, panel, person)
    directory = args.out / panel / person
    input_dir = args.baselines / panel / person / "images/aligned"
    if {p.name for p in input_dir.glob("*.jpg")} != {f"{i:03d}.jpg" for i in range(len(rows))}:
        raise ValueError(f"Incomplete baseline JPEG inventory at {input_dir}")
    for arm in ARMS:
        (directory / "images" / arm).mkdir(parents=True, exist_ok=True)
    marker = directory / "settings.json"
    if marker.exists():
        done = json.loads(marker.read_text(encoding="utf8"))
        if (
            any(done.get(k) != v for k, v in origin.items())
            or done.get("script_sha256") != digest(Path(__file__))
            or done.get("provider") != args.provider
        ):
            raise ValueError(f"Existing output belongs to a different run: {directory}")
        if all(
            digest(directory / "images" / a / f"{i:03d}.jpg") == done["jpeg_sha256"][a][i]
            and (a != "baseline" or digest(input_dir / f"{i:03d}.jpg") == done["jpeg_sha256"][a][i])
            for a in ARMS
            for i in range(len(rows))
        ):
            print("Verified completed", panel, person, flush=True)
            return
        raise ValueError(f"Completed output or baseline failed its digest check: {directory}")
    hashes = {k: [] for k in ARMS}
    applied = {k: [] for k in targets}
    measured = []
    for i, row in enumerate(rows):
        name = f"{i:03d}.jpg"
        data = (input_dir / name).read_bytes()
        if expected and hashlib.sha256(data).hexdigest() != expected[int(row)]:
            raise ValueError(f"Frozen personal JPEG hash mismatch at {input_dir / name}")
        with Image.open(io.BytesIO(data)) as im:
            if im.format != "JPEG" or im.mode != "RGB" or im.size != (512, 512):
                raise ValueError("Expected a generated RGB 512-square JPEG")
            source = measure(session, im)
            rgb = np.asarray(im, dtype=np.uint8)
        measured.append(source.tolist())
        rendered = {"baseline": data, "neutral": jpeg(rgb)}
        for a, t in targets.items():
            result, params = warp(rgb, source, t[i])
            rendered[a] = jpeg(result)
            applied[a].append(params)
        for arm, blob in rendered.items():
            (directory / "images" / arm / name).write_bytes(blob)
            hashes[arm].append(hashlib.sha256(blob).hexdigest())
        if (i + 1) % 40 == 0:
            print(panel, person, i + 1, "of", len(rows), flush=True)
    eligible = rows != donors
    write_json(
        marker,
        dict(
            panel=panel,
            subject=person,
            **origin,
            script_sha256=digest(Path(__file__)),
            model_sha256=MODEL_SHA,
            provider=args.provider,
            rows=rows.tolist(),
            folds=folds.tolist(),
            predicted_words=words.tolist(),
            donor_rows=donors.tolist(),
            swap_eligible=eligible.tolist(),
            swap_eligible_rows=rows[eligible].tolist(),
            arms=list(ARMS),
            source_jpeg_sha256=hashes["baseline"],
            jpeg_sha256=hashes,
            baseline_geometry=measured,
            targets={k: v.tolist() for k, v in targets.items()},
            applied=applied,
            geometry_fields=["center_x", "center_y", "log_sd_x", "log_sd_y"],
            applied_fields=["center_x", "center_y", "sd_ratio_x", "sd_ratio_y"],
            recipe=dict(
                strength=STRENGTH,
                max_center_shift=MAX_SHIFT,
                sd_ratio_limits=RATIO_LIMITS,
                border_taper_fraction=BORDER,
                resampling="bilinear; inverse axis PCHIP through fixed frame endpoints, center and one-SD landmarks; smooth edge taper",
                jpeg_quality=95,
                baseline="Original personal aligned JPEG bytes",
                neutral="Identity pixels with the same JPEG encoder as the geometry arms",
            ),
        ),
    )
    print("Frozen", marker, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry", "baselines", "results", "registrations", "model", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--jpeg-receipts",
        type=Path,
        help="Verify each existing generated JPEG against prior photo_layout.json",
    )
    p.add_argument(
        "--provider",
        choices=("DmlExecutionProvider", "CPUExecutionProvider", "CUDAExecutionProvider"),
        default="CPUExecutionProvider",
    )
    p.add_argument("--subjects", nargs="+", default=["sub-04", "sub-05"])
    p.add_argument(
        "--panels",
        nargs="+",
        choices=("registered", "replicate"),
        default=["registered", "replicate"],
    )
    a = p.parse_args()
    if digest(a.model) != MODEL_SHA:
        raise ValueError("U2NetP checkpoint SHA differs from the fixed geometry predictor")
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False
    session = ort.InferenceSession(str(a.model), sess_options=opts, providers=[a.provider])
    if session.get_providers()[0] != a.provider:
        raise RuntimeError(f"Requested provider {a.provider} unavailable")
    for panel in a.panels:
        for subject in a.subjects:
            run_case(a, session, panel, subject)


if __name__ == "__main__":
    main()
