"""Export the sub-04 protected directions with no access to query labels or visual targets."""

import argparse
import io
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from calibrate_official_repeat_angles import tangent
from fit_canonical_neighbor import angle_set, model, native, nhood, physical, sources
from fit_protected_photo import alter, semantic_axes
from official_object_readout import unit
from package_personal_confirmation import scientific_code
from register_fresh_photo_panel import freeze, read, sha
from render_protected_photo import LOCK, PANEL, SUB
from render_protected_photo import packet as verified
from render_within_photo_ip import check, packet
from run_prospective_readout import frozen_npz

PUB = Path(__file__).resolve().parents[1] / "results/object_fidelity"


def prepare(a):
    reg = PUB / "protected_photo_images_design.json"
    check(sha(reg) == LOCK, "Unexpected pilot")
    spec = read(reg)["source_sha256"]
    native_root = a.data / "fresh-mix/render_inputs/learned"
    old = native_root / "renders" / PANEL / SUB / "settings.json"
    original = native_root / "conditions" / SUB / PANEL / "conditions_512.npz"
    eeg = a.data / "fresh-mix/eeg/sub-04.npz"
    checks = dict(
        generic_result=PUB / "protected_photo.json",
        generic_design=PUB / "protected_photo_design.json",
        eeg=eeg,
        native_settings=old,
        native_packet=original,
        historical_images=PUB / "fresh_mix_images.json",
        renderer=Path(__file__).with_name("render_within_photo_ip.py"),
    )
    check(
        all(sha(path) == spec[k] for k, path in checks.items())
        and read(checks["generic_result"])["advance"]
        and sha(Path(__file__).with_name("fit_protected_photo.py"))
        == read(checks["generic_result"])["driver_sha256"],
        "Source study changed",
    )
    rows, folds, words, donors, support, v, rad, _ = packet(native_root, SUB, PANEL)
    index = {r: i for i, r in enumerate(rows)}
    d = np.array([index[r] for r in donors])
    si = np.array([[index[r] for r in s] for s in support])
    check(
        np.array_equal(support, support[d])
        and np.array_equal(folds, folds[d])
        and np.array_equal(words, words[d])
        and np.all(folds[:, None] != folds[si])
        and np.all(rows[:, None] != support),
        "Incompatible support or donor",
    )
    p, _, train, _ = sources(a)
    check(
        len(train) == 884 and not np.intersect1d(rows // 10, train).size,
        "Training contains pilot object",
    )
    # Use the archive's lazy tensor mapping, indexed strictly by external training class.
    # Reading its full distributed-file digest is independent of any photo-conditioned algorithm.
    trrows = (train[:, None] * 10 + np.arange(10)).ravel()
    check(not np.intersect1d(trrows, rows).size, "Query entered H14 supervision")
    image_storage = torch.load(
        p["generic_images"], weights_only=True, mmap=True, map_location="cpu"
    )["img_features"]
    check(
        tuple(image_storage.shape) == (16540, 1024) and image_storage.dtype == torch.float32,
        "Unknown H14 storage",
    )
    y = unit(image_storage.numpy()[trrows])
    estimator = model(physical(p, train), y)
    q = semantic_axes(y)
    h = read(PUB / "protected_photo_design.json")["sources"]["estimator"]
    sfile = a.data / "canonical-neighbor/selection.json"
    check(sha(sfile) == h, "Wrong generic canonical estimator")
    selection = read(sfile)
    check(selection["training_classes"] == train.tolist(), "Estimator fit IDs differ")
    with np.load(eeg, allow_pickle=False) as z:
        r = z["rows"]
        select = np.searchsorted(r, rows)
        check(
            str(z["subject"]) == SUB and np.all(np.diff(r) > 0) and np.array_equal(r[select], rows),
            "EEG order mismatch",
        )
        x = z["features"][select].astype(float)
    check(x.shape == (120, 1003) and np.isfinite(x).all(), "Invalid recipient EEG")
    direction, coords = native(estimator, x)
    near = nhood(estimator, coords, y, (selection["k"],))[selection["k"]]
    absolute = angle_set(
        tangent(direction, estimator["axis"]),
        near,
        selection["mix"],
        estimator["axis"],
        selection["gain"],
        selection["cap"],
    ).astype(float)
    delta = absolute - absolute[si].mean(1)
    base, prior = v["own"].astype(float), v["uniform"].astype(float)
    # The uniform packet is the direction of the published eight-support H14 mean.
    # Make it unit to double precision for the projection operation.
    prior = unit(prior)
    estimates, gerr, perr = alter(base, prior, delta, delta[d], base, q)
    zero = np.zeros_like(delta)
    identity, _, _ = alter(base, prior, zero, zero, base, q)
    check(
        all(np.array_equal(a.astype(np.float32), v["own"]) for a in identity.values())
        and np.array_equal(estimates["reference"].astype(np.float32), v["own"]),
        "Identity failed",
    )
    arms = dict(
        native=v["own"],
        own=estimates["matched"].astype(np.float32),
        wrong=estimates["wrong"].astype(np.float32),
    )
    # Check the actual serialized precision, including preservation of singleton donor rows.
    max_global = max(np.abs((a.astype(float) - base) @ q).max() for a in arms.values())
    max_prior = max(abs(((a.astype(float) - base) * prior).sum(1)).max() for a in arms.values())
    check(
        max(max_global, max_prior) < 2e-6
        and np.array_equal(arms["wrong"][d == np.arange(120)], arms["own"][d == np.arange(120)]),
        "Projection changed protected coordinates",
    )
    root = a.out / "inputs"
    frozen_npz(
        root / "conditions.npz",
        dict(
            rows=rows, folds=folds, words=words, donors=donors, support=support, radius=rad, **arms
        ),
    )
    for src, name in (
        (reg, "protocol.json"),
        (old, "native_settings.json"),
        (native_root / "radius.json", "radius.json"),
    ):
        dest = root / name
        check(not dest.exists() or sha(src) == sha(dest), "Cached study differs")
        shutil.copyfile(src, dest)
    freeze(
        root / "design.json",
        dict(
            registration_sha256=LOCK,
            packet_sha256=sha(root / "conditions.npz"),
            radius_sha256=sha(root / "radius.json"),
            prep_source_sha256=sha(Path(__file__)),
            max_precast_global_error=gerr,
            max_precast_prior_error=perr,
            max_float32_global_error=float(max_global),
            max_float32_prior_error=float(max_prior),
            zero_direction_exact_rows=120,
            nonself_donors=int((rows != donors).sum()),
        ),
    )
    verified(root)
    files = scientific_code("render_protected_photo")
    files.update(
        {
            "inputs/" + f.name: f
            for f in (
                root / "conditions.npz",
                root / "protocol.json",
                root / "native_settings.json",
                root / "design.json",
                root / "radius.json",
            )
        }
    )
    out = a.out / "protected_photo.tar"
    temp = out.with_suffix(".partial")
    with tarfile.open(temp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, src in sorted(files.items()):
            b = src.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(b)
            tar.addfile(info, io.BytesIO(b))
    check(not out.exists() or sha(temp) == sha(out), "Archive changed")
    temp.replace(out)
    print(
        "HPC payload",
        sha(out),
        out.stat().st_size,
        "condition",
        sha(root / "conditions.npz"),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/protected-photo/images"))
    prepare(p.parse_args())
