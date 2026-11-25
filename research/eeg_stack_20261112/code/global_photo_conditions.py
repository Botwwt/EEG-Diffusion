"""Prepare absolute EEG-to-image predictions and a separately packaged reference intervention."""

import argparse
import io
import shutil
import tarfile
from pathlib import Path

import numpy as np
from calibrate_official_repeat_angles import tangent
from fit_canonical_neighbor import angle_set, model, native, nhood, photo_bank, physical, sources
from fit_within_photo_embedding import images
from package_personal_confirmation import scientific_code
from prepare_expanded_personal import extract
from register_fresh_photo_panel import freeze, read, sha
from render_global_photo import LOCK, PANEL, SUB, verify
from render_within_photo_ip import check, packet
from run_prospective_readout import frozen_npz

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    check(np.isfinite(norm).all() and (norm > 1e-10).all(), "Undefined conditioning direction")
    return x / norm


def prepare(a):
    protocol = PUBLIC / "global_photo_design.json"
    check(sha(protocol) == LOCK, "Global photo protocol changed")
    expected = read(protocol)["source_sha256"]
    native_root = a.data / "trial-object-encoder/personal12/render_inputs"
    ref = native_root / "renders" / PANEL / SUB / "settings.json"
    physical_file = a.data / "object-head-confirmation/eeg/sub-04.npz"
    sel = a.data / "canonical-neighbor/selection.json"
    source = native_root / "conditions" / SUB / PANEL / "conditions_512.npz"
    files = {
        "native_settings": ref,
        "condition": source,
        "radius": native_root / "radius.json",
        "eeg": physical_file,
        "canonical_selection": sel,
        "global_fit": Path(__file__).with_name("fit_canonical_neighbor.py"),
    }
    check(all(sha(v) == expected[k] for k, v in files.items()), "Changed registered sources")
    rows, folds, words, donors, supports, vec, rad, _ = packet(native_root, SUB, PANEL)
    check(
        all(set(ss) <= set(rows[folds != f]) for ss, f in zip(supports, folds, strict=True)),
        "Support leakage",
    )
    work = a.root / a.stage
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "eeg":
        p, author, train, _ = sources(a)
        check(
            len(train) == 884 and not set(train) & set(rows // 10), "Target classes in global fit"
        )
        yy = photo_bank(author, train)
        m = model(physical(p, train), yy)
        with np.load(physical_file, allow_pickle=False) as z:
            rawrows = z["rows"]
            check(
                str(z["subject"]) == SUB
                and int(z["repeats_per_half"]) == 2
                and np.array_equal(rawrows, np.unique(rawrows))
                and np.array_equal(np.sort(rows), rawrows),
                "Unexpected physical EEG",
            )
            x = extract(z["half_eeg"].mean(0))
        s = read(sel)
        check(
            s["k"] == 128 and s["mix"] == 0.5 and s["training_classes"] == train.tolist(),
            "Wrong external global estimator",
        )
        v, query = native(m, x)
        near = nhood(m, query, yy, (s["k"],))[s["k"]]
        absolute = angle_set(
            tangent(v, m["axis"]), near, s["mix"], m["axis"], s["gain"], s["cap"]
        ).astype(float)
        base = vec["own"].astype(float)

        def mix(second):
            return unit(base + second).astype(np.float32)

        estimates = dict(
            native=vec["own"],
            matched=mix(absolute[np.searchsorted(rawrows, rows)]),
            prior=mix(unit(absolute[np.searchsorted(rawrows, supports)].mean(1))),
            wrong=mix(absolute[np.searchsorted(rawrows, donors)]),
        )
    else:
        verify(a.root / "eeg", "eeg")
        reg = (
            a.data
            / "object-head-confirmation/evaluation/references"
            / SUB
            / (PANEL + "_ablation.json")
        )
        check(sha(reg) == expected["ref"], "Reference registration mismatch")
        labels = np.array(read(reg)["truth"])
        check(np.array_equal(read(reg)["rows"], rows), "Reference rows differ")
        foil = np.empty_like(rows)
        for i in range(len(rows)):
            pair = rows[(labels == labels[i]) & (folds == folds[i]) & (rows != rows[i])]
            check(
                len(pair) == 1 and pair[0] // 10 == rows[i] // 10, "Wrong held-out comparison image"
            )
            foil[i] = pair[0]
        author = a.data / "official-object-readout/sub-03"
        bank = images(author)
        estimates = {
            k: unit(bank[r].astype(float)).astype(np.float32)
            for k, r in (("reference", rows), ("foil", foil))
        }
    common = dict(rows=rows, folds=folds, words=words, donors=donors, supports=supports, radius=rad)
    frozen_npz(work / "conditions.npz", common | estimates)
    for src, name in ((protocol, "protocol.json"), (native_root / "radius.json", "radius.json")):
        dest = work / name
        check(not dest.exists() or sha(dest) == sha(src), "Existing immutable source differs")
        shutil.copyfile(src, dest)
    original = read(ref)
    freeze(
        work / "design.json",
        dict(
            registration_sha256=LOCK,
            stage=a.stage,
            packet_sha256=sha(work / "conditions.npz"),
            radius_sha256=sha(work / "radius.json"),
            native_settings_sha256=sha(ref),
            native_hashes=original["jpeg_sha256"]["own"],
            generator={
                k: original[k]
                for k in ("adapter_sha256", "turbo_sha256", "dictionary_sha256", "driver_sha256")
            },
        ),
    )
    verify(work, a.stage)
    files = scientific_code("render_global_photo")
    files.update(
        {
            "inputs/" + name: work / name
            for name in ("conditions.npz", "design.json", "protocol.json", "radius.json")
        }
    )
    tar = a.root / (a.stage + ".tar")
    temp = tar.with_suffix(".partial")
    with tarfile.open(temp, "w", format=tarfile.USTAR_FORMAT) as t:
        for name, src in sorted(files.items()):
            b = src.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(b)
            t.addfile(info, io.BytesIO(b))
    check(not tar.exists() or sha(tar) == sha(temp), "Conflicting input archive")
    temp.replace(tar)
    print(a.stage, sha(tar), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("eeg", "oracle"))
    p.add_argument("--root", type=Path, default=Path("outputs/global-photo"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    prepare(p.parse_args())
