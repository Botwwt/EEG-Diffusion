"""Generate new images from a held-out EEG query and its eight allowed calibration photographs.

Run one or more --case CONDITIONS SELECTION ABLATION OLD_RENDER OUT pairs. The old
aligned images are copied only as the previously generated comparison. A separate
evaluator may open original query photographs once settings.json has been written.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ARMS = ("matched", "mean", "swapped", "random")
SEED = 923761
RANDOM_SEED = 260435
RECIPE = dict(
    schedule="last 2 of 4 archived Turbo Euler ancestral steps",
    latent_pool=4,
    support_fraction=0.8,
    eeg_start_fraction=0.2,
    first_hold=0,
    tether=False,
    noise_base_seed=SEED,
    random_base_seed=RANDOM_SEED,
    jpeg_quality=95,
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def load_case(paths):
    from render_registered_objects import load_packet

    conditions, selection, panel, baseline, out = paths
    ablation = panel / "ablation.json" if panel.is_dir() else panel
    packet = load_packet(conditions)
    with np.load(selection, allow_pickle=False) as s:
        for name, actual in (
            ("rows", packet.rows),
            ("folds", packet.folds),
            ("predicted_words", packet.words),
        ):
            if not np.array_equal(s[name], actual):
                raise ValueError(f"{name} differs between selection and conditions at {selection}")
        supports, weights = s["support_rows"], s["weights"]
    if (
        supports.shape != (120, 8)
        or supports.dtype.kind not in "iu"
        or weights.shape != supports.shape
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.allclose(weights.sum(1), 1, atol=1e-7, rtol=0)
    ):
        raise ValueError(f"Invalid support rows or voting weights at {selection}")
    weights = weights.astype(np.float64) / weights.sum(1, keepdims=True)
    reg = read(ablation)
    words = reg["words"]
    calibrated = np.asarray(reg["calibration_rows_by_fold_and_word"])
    if (
        not np.array_equal(reg["rows"], packet.rows)
        or not np.array_equal(reg["folds"], packet.folds)
        or set(packet.folds) != set(range(5))
        or len(words) != 12
        or len(set(words)) != 12
        or set(packet.words) - set(words)
        or calibrated.shape != (5, 12, 8)
        or calibrated.dtype.kind not in "iu"
    ):
        raise ValueError(f"Condition rows, folds or vocabulary differ from {ablation}")
    for fold, allowed in enumerate(calibrated):
        query = packet.rows[packet.folds == fold]
        if (
            len(query) != 24
            or np.unique(allowed).size != 96
            or set(allowed.ravel()) != set(packet.rows) - set(query)
        ):
            raise ValueError(f"Fold {fold} does not partition 24 queries and 96 calibration rows")
    for i, (f, w) in enumerate(zip(packet.folds, packet.words, strict=True)):
        if set(supports[i]) != set(calibrated[f, words.index(w)]):
            raise ValueError(f"Calibration support escaped the predicted class/fold at query {i}")
    prior = read(baseline / "results.json")
    fingerprint = digest(conditions)
    if (
        prior["rows"] != packet.rows.tolist()
        or prior["predicted_words"] != packet.words.tolist()
        or prior["condition_sha"] != fingerprint
        or prior["results"]["aligned"]["n"] != 120
    ):
        raise ValueError(f"Original render does not correspond to the frozen packet: {baseline}")
    positions = np.arange(120)
    donors = positions.copy()
    shuffled = np.empty_like(weights)
    random = np.empty_like(weights)
    for f in range(5):
        for word in np.unique(packet.words[packet.folds == f]):
            group = positions[(packet.folds == f) & (packet.words == word)]
            donors[group] = np.roll(group, 1)
    for i, j in enumerate(donors):
        lookup = dict(zip(supports[j], weights[j], strict=True))
        shuffled[i] = [lookup[r] for r in supports[i]]
        rng = np.random.default_rng(RANDOM_SEED + int(packet.rows[i]))
        random[i] = weights[i, rng.permutation(8)]
    by_arm = dict(
        matched=weights, mean=np.full_like(weights, 1 / 8), swapped=shuffled, random=random
    )
    provenance = dict(
        conditions_sha256=fingerprint,
        selection_sha256=digest(selection),
        ablation_sha256=digest(ablation),
        baseline_results_sha256=digest(baseline / "results.json"),
    )
    return SimpleNamespace(
        packet=packet,
        support=supports,
        weights=by_arm,
        donors=donors,
        provenance=provenance,
        baseline=baseline,
        out=out,
    )


def photo_index(path, required):
    records = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for record in csv.DictReader(f):
            if record["dataset_split"] != "train":
                continue
            row = int(record["array_row"])
            if row in required:
                if row in records:
                    raise ValueError(f"Duplicate manifest row {row}")
                records[row] = record["image_member"], record["image_sha256"]
    if set(records) != required or len({v[0] for v in records.values()}) != len(required):
        raise ValueError("Manifest does not contain distinct calibration photographs for all rows")
    return records


def encode_fold(pipe, zipped, records, rows, size):
    import torch
    import torch.nn.functional as F
    from render_official_words import crop512

    encodings = []
    for lo in range(0, len(rows), size):
        pixels = []
        for row in rows[lo : lo + size]:
            name, expected = records[row]
            raw = zipped.read(name)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Calibration JPEG disagrees with manifest at row {row}")
            with Image.open(io.BytesIO(raw)) as im:
                pixels.append(np.asarray(crop512(im.convert("RGB")), dtype=np.uint8))
        x = torch.from_numpy(np.stack(pixels)).cuda().permute(0, 3, 1, 2).float().div_(255)
        z = pipe.vae.encode(2 * x - 1).latent_dist.mode().float() * pipe.vae.config.scaling_factor
        if z.shape[1:] != (4, 64, 64) or not torch.isfinite(z).all():
            raise ValueError("Unexpected Turbo calibration VAE features")
        encodings.append(F.avg_pool2d(z, kernel_size=RECIPE["latent_pool"]).cpu())
    return torch.cat(encodings)


def write_jpeg(tensor, path):
    from torchvision.transforms import ToPILImage

    raw = io.BytesIO()
    ToPILImage()(tensor.cpu()).save(raw, format="JPEG", quality=RECIPE["jpeg_quality"])
    data = raw.getvalue()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def generate(case, modules, zipped, records, batch_size, manifest_sha):
    import torch
    import torch.nn.functional as F
    from eval_instance_starts import rollout, write_json

    packet = case.packet
    pipe, adapter, proj, empty, prompts = modules
    # Keep the low-noise suffix of the original four-step grid with its original
    # Euler coefficients. The last sigma is the terminal zero.
    steps, sigmas = pipe.scheduler.timesteps, pipe.scheduler.sigmas
    if len(steps) != 4 or len(sigmas) != 5 or float(sigmas[-1]) != 0:
        raise ValueError("Expected the archived four-step trailing Euler ancestral schedule")
    suffix = SimpleNamespace(
        unet=pipe.unet, scheduler=SimpleNamespace(timesteps=steps[2:], sigmas=sigmas[2:])
    )
    noise = torch.stack(
        [
            torch.randn(5, 4, 64, 64, generator=torch.Generator().manual_seed(SEED + int(r)))
            for r in packet.rows
        ]
    )
    noise = noise[:, [0, 3, 4]]
    tokens = proj(adapter(torch.from_numpy(packet.condition).cuda().half()))
    hashes = {k: [None] * 120 for k in ("baseline", *ARMS)}
    for name in hashes:
        (case.out / "images" / name).mkdir(parents=True, exist_ok=True)
    marker = case.out / "settings.json"
    if marker.exists():
        old = read(marker)
        if any(old.get(k) != v for k, v in case.provenance.items()) or old.get("recipe") != RECIPE:
            raise ValueError(
                f"Existing case has different inputs or generation settings at {case.out}"
            )
        marker.unlink()
    for i in range(120):
        file = f"{i:03d}.jpg"
        data = (case.baseline / "images/aligned" / file).read_bytes()
        (case.out / "images/baseline" / file).write_bytes(data)
        hashes["baseline"][i] = hashlib.sha256(data).hexdigest()
    for f in range(5):
        queries = np.flatnonzero(packet.folds == f)
        eligible = np.unique(case.support[queries])
        if np.intersect1d(eligible, packet.rows[queries]).size:
            raise ValueError("Query pixels were requested as fold calibration")
        # This cache is local to one fold. No held-out photo is encoded for its own fold.
        layouts = encode_fold(pipe, zipped, records, eligible, batch_size)
        for lo in range(0, len(queries), batch_size):
            ix = queries[lo : lo + batch_size]
            supports = layouts[np.searchsorted(eligible, case.support[ix])].cuda()
            query_start = F.interpolate(
                torch.from_numpy(packet.start[ix]).cuda(),
                (64, 64),
                mode="bilinear",
                align_corners=False,
            )
            prompt = tuple(
                torch.cat([prompts[w][j] for w in packet.words[ix]]).cuda() for j in (0, 1)
            ) + (empty[2],)
            row_noise = noise[ix].cuda()
            for arm in ARMS:
                w = torch.as_tensor(case.weights[arm][ix], dtype=torch.float32, device="cuda")
                support_start = F.interpolate(
                    (supports * w[:, :, None, None, None]).sum(1),
                    (64, 64),
                    mode="bilinear",
                    align_corners=False,
                )
                initial = (
                    RECIPE["support_fraction"] * support_start
                    + RECIPE["eeg_start_fraction"] * query_start
                )
                z = rollout(
                    suffix,
                    prompt,
                    tokens[ix.tolist()],
                    initial,
                    row_noise,
                    first_hold=0,
                    tether=False,
                )
                rgb = (
                    0.5 * pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample + 0.5
                ).clamp(0, 1)
                if not torch.isfinite(rgb).all():
                    raise ValueError(f"Non-finite {arm} output in fold {f}")
                for index, im in zip(ix, rgb, strict=True):
                    hashes[arm][index] = write_jpeg(
                        im, case.out / "images" / arm / f"{index:03d}.jpg"
                    )
        print(case.out, "completed fold", f, flush=True)
    swap_eligible = case.donors != np.arange(120)
    if any(any(v is None for v in hs) for hs in hashes.values()):
        raise RuntimeError("Missing registered output")
    write_json(
        marker,
        dict(
            **case.provenance,
            recipe=RECIPE,
            script_sha256=digest(Path(__file__)),
            manifest_sha256=manifest_sha,
            batch_size=batch_size,
            rows=packet.rows.tolist(),
            folds=packet.folds.tolist(),
            predicted_words=packet.words.tolist(),
            support_rows=case.support.tolist(),
            weights=case.weights["matched"].tolist(),
            random_weights=case.weights["random"].tolist(),
            swapped_weights=case.weights["swapped"].tolist(),
            donor_rows=packet.rows[case.donors].tolist(),
            swap_eligible=swap_eligible.tolist(),
            swap_eligible_rows=packet.rows[swap_eligible].tolist(),
            timesteps=steps[2:].tolist(),
            sigmas=sigmas[2:].tolist(),
            arms=list(hashes),
            jpeg_sha256=hashes,
            baseline="Byte copy of the previously generated personal aligned JPEG.",
            weights_source="Frozen selection.npz weights. Mean is uniform. Random permutes the same magnitudes across the eight supports. Swapped transfers weights by support row from another query with the same predicted word and fold; singleton rows keep their weights.",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--dictionary", required=True, type=Path)
    p.add_argument("--code-dir", type=Path, default=Path("research/eeg_stack_20261112/code"))
    p.add_argument(
        "--case",
        type=Path,
        action="append",
        nargs=5,
        required=True,
        metavar=("CONDITIONS", "SELECTION", "ABLATION", "OLD_RENDER", "OUT"),
    )
    p.add_argument("--images-zip", type=Path, help="Override archive training_images.zip")
    p.add_argument(
        "--manifest", type=Path, help="Override archive SHA-indexed v238 image_manifest.csv"
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate packets, split boundaries, manifest and baseline JPEG inventory without loading models",
    )
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("Positive --batch-size required")
    sys.path.insert(0, str(a.code_dir.resolve()))
    from render_registered_objects import load_generator

    root = a.archive.resolve()
    cases = [load_case(paths) for paths in a.case]
    outputs = {c.out.resolve() for c in cases}
    if len(outputs) != len(cases) or outputs.intersection(c.baseline.resolve() for c in cases):
        p.error("Use distinct output directories, separate from all prior renders")
    manifest = a.manifest or root / "data/full_T_recovery_20261108_v238/image_manifest.csv"
    photos = a.images_zip or root / "assets/things_images_v254/training_images.zip"
    records = photo_index(manifest, set(np.concatenate([c.support.ravel() for c in cases])))
    with zipfile.ZipFile(photos) as zipped:
        for case in cases:
            if any(
                not (case.baseline / "images/aligned" / f"{i:03d}.jpg").is_file()
                for i in range(120)
            ):
                raise ValueError(f"Incomplete aligned JPEG comparator at {case.baseline}")
        if a.check_only:
            for member, _ in records.values():
                zipped.getinfo(member)
            print(
                "Validated support sets, conditions, manifest, ZIP members and baseline inventory for",
                len(cases),
                "cases",
            )
            return
        import torch

        with torch.inference_mode():
            modules = load_generator(
                root, a.dictionary, np.concatenate([c.packet.words for c in cases])
            )
            manifest_sha = digest(manifest)
            for case in cases:
                generate(case, modules, zipped, records, a.batch_size, manifest_sha)


if __name__ == "__main__":
    main()
