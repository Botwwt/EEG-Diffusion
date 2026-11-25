"""Prepare the support-centered sub-04 canonical-neighbor packet and deterministic offline input tar."""

import argparse
import io
import shutil
import tarfile
from pathlib import Path

import numpy as np
from calibrate_official_repeat_angles import expmap
from fit_canonical_neighbor import (
    CODE,
    DESIGN,
    PUB,
    metrics,
    model,
    native,
    nhood,
    photo_bank,
    physical,
    sources,
    summary,
)
from fit_repeat_photo_subspace import centered, ridge
from prepare_expanded_personal import extract
from register_fresh_photo_panel import freeze, read, sha
from render_canonical_neighbor_personal12 import LOCK, PACKET, direction, packet, visual_update
from render_within_photo_ip import check
from run_prospective_readout import frozen_npz

REGISTRATION = PUB / "canonical_neighbor_personal12_design.json"
WRAPPER = CODE / "render_canonical_neighbor_personal12.py"
DEPENDENCIES = (
    "render_canonical_neighbor_personal12.py",
    "render_within_photo_ip.py",
    "render_official_ip_adapter.py",
    "render_official_words.py",
)


def local_features(m, x, bank):
    _, query = native(m, x)
    return expmap(nhood(m, query, bank, (128,))[128], m["axis"], np.pi)


def tangents(prior, contrast):
    a = direction(prior.astype(float))
    difference = contrast.astype(float)
    v = difference - (difference * a).sum(1, keepdims=True) * a
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm > 1e-10, v / norm.clip(1e-12), 0).astype(np.float32)


def generic(a, p, source, tr, held, x, y, m):
    xx = physical(p, held)
    yy = photo_bank(source, held)
    predicted = ridge(x, centered(y.reshape(len(tr), 10, -1).astype(float)), xx.reshape(-1, 1003))
    loc = local_features(m, xx, y)
    row = np.arange(len(yy))
    donor = row ^ 1
    supports = np.array(
        [[i // 10 * 10 + j for j in range(10) if j // 2 != (i % 10) // 2] for i in row]
    )
    support_mean = yy[supports].mean(1)
    prior = direction(support_mean)
    base = direction(support_mean + predicted - predicted[supports].mean(1)).astype(np.float32)
    delta = tangents(prior, loc - loc[supports].mean(1))
    own, wrong, _ = visual_update(base, prior, delta, donor)
    result = {key: metrics(v, yy) for key, v in dict(native=base, own=own, wrong=wrong).items()}
    counts = {k: summary(v) for k, v in result.items()}
    draws = np.random.default_rng(260574).integers(len(held), size=(10000, len(held)))
    contrasts = {}
    for k in ("native", "wrong"):
        contrasts[k] = {}
        for metric in ("within10", "within_pair", "mrr", "target_cosine"):
            d = (result["own"][metric] - result[k][metric]).reshape(-1, 10).mean(1)
            contrasts[k][metric] = dict(
                delta=float(d.mean()), ci95=np.quantile(d[draws].mean(1), [0.025, 0.975]).tolist()
            )
    receipt = dict(
        design_sha256=LOCK,
        fit_objects=len(tr),
        held_objects=len(held),
        queries=len(yy),
        n_support=8,
        metrics=counts,
        own_minus=contrasts,
    )
    freeze(a.out / "generic_support_check.json", receipt)
    print("Generic support-centered check", counts, flush=True)
    return receipt


def tar_bundle(out, root):
    targets = {f"code/{name}": CODE / name for name in DEPENDENCIES}
    for rel in (Path("design.json"), Path("radius.json"), Path("protocol.json"), PACKET):
        targets["inputs/" + rel.as_posix()] = root / rel
    path = out / "canonical-neighbor-personal12.tar"
    temp = path.with_suffix(".partial")
    with tarfile.open(temp, "w", format=tarfile.USTAR_FORMAT) as t:
        for label, file in sorted(targets.items()):
            raw = file.read_bytes()
            info = tarfile.TarInfo(label)
            info.size = len(raw)
            info.mode = 0o644
            t.addfile(info, io.BytesIO(raw))
    if path.exists():
        check(sha(temp) == sha(path), "Existing bundle differs")
        temp.unlink()
    else:
        temp.replace(path)
    return path


def run(a):
    root = a.out / "render_inputs"
    b = a.data / "trial-object-encoder/personal12/render_inputs"
    ref = b / "renders/personal12/sub-04/settings.json"
    word = b / "evaluations/sub-04/personal12/word_predictions.json"
    wave = a.data / "object-head-confirmation/eeg/sub-04.npz"
    refs = dict(
        canonical_registration=DESIGN,
        generic_estimator=CODE / "fit_canonical_neighbor.py",
        canonical_metrics=PUB / "canonical_neighbor.json",
        selection=a.data / "canonical-neighbor/selection.json",
        old_design=b / "design.json",
        radius=b / "radius.json",
        baseline=b / PACKET,
        word_manifest=word,
        target_waveforms=wave,
        prior_jpeg_manifest=ref,
        renderer=CODE / "render_within_photo_ip.py",
    )
    protocol = read(REGISTRATION)
    check(
        sha(REGISTRATION) == LOCK
        and {k: sha(f) for k, f in refs.items()} == protocol["source_sha256"],
        "Registered source differs",
    )
    sel = read(refs["selection"])
    check(sel["k"] == 128 and sel["mix"] == 0.5, "Historical OOF selection differs")
    p, source, tr, held = sources(a)
    x = physical(p, tr)
    y = photo_bank(source, tr)
    m = model(x, y)
    g = generic(a, p, source, tr, held, x, y, m)
    old = read(ref)
    words_manifest = read(word)
    with np.load(b / PACKET, allow_pickle=False) as z:
        out = {
            k: z[k]
            for k in (
                "rows",
                "folds",
                "support_rows",
                "predicted_words",
                "donor_rows",
                "subject",
                "physical_trials_per_photo",
                "training_classes",
                "visual_sha256",
                "target_eeg_sha256",
                "word_prediction_sha256",
                "registration_sha256",
            )
        }
        rows, support, donors = (out[k] for k in ("rows", "support_rows", "donor_rows"))
        unit_prior = z["uniform"]
        base = z["own"]
    check(
        not np.intersect1d(rows // 10, tr).size and len(set(rows // 10)) == 12,
        "Query-class photo present in fitted gallery",
    )
    check(
        all(
            np.array_equal(out[k], old[k])
            for k in ("rows", "folds", "support_rows", "donor_rows", "predicted_words")
        )
        and np.array_equal(rows, words_manifest["rows"])
        and np.array_equal(
            np.asarray(words_manifest["words"])[words_manifest["predicted"]], out["predicted_words"]
        )
        and np.array_equal(923761 + rows, old["seeds"])
        and old["packet_sha256"] == sha(b / PACKET)
        and old["batch_size"] == 8,
        "Original words/seeds/packet identity mismatch",
    )
    with np.load(wave, allow_pickle=False) as z:
        wave_rows = z["rows"]
        check(
            set(wave_rows) == set(rows)
            and z["half_eeg"].shape == (2, 120, 17, 250)
            and int(z["repeats_per_half"]) == 2
            and str(z["subject"]) == "sub-04",
            "Waveform population differs",
        )
        xx = extract(z["half_eeg"].mean(0))
    check(
        str(out["target_eeg_sha256"]) == sha(wave)
        and str(out["word_prediction_sha256"]) == sha(word),
        "Waveform or word receipt mismatch",
    )
    order = {r: i for i, r in enumerate(wave_rows)}
    e = local_features(m, xx, y).astype(float)
    diff = e[[order[r] for r in rows]] - e[[[order[r] for r in ss] for ss in support]].mean(1)
    delta = tangents(unit_prior, diff)
    index = {r: i for i, r in enumerate(rows)}
    j = np.array([index[r] for r in donors])
    own, wrong, angles = visual_update(base, unit_prior, delta, j)
    out.update(
        native=base,
        own=own,
        wrong=wrong,
        zero=base,
        prior=unit_prior,
        local_tangent=delta,
        recipient_angle_radians=angles,
        canonical_training_classes=tr,
        original_conditions_sha256=sha(b / PACKET),
    )
    frozen_npz(root / PACKET, out)
    for src, name in ((b / "radius.json", "radius.json"), (REGISTRATION, "protocol.json")):
        dest = root / name
        if dest.exists():
            check(sha(dest) == sha(src), "Existing lock differs")
        else:
            shutil.copyfile(src, dest)
    design = dict(
        registration_sha256=LOCK,
        packet_sha256=sha(root / PACKET),
        panels=["personal12"],
        subjects=["sub-04"],
        physical_trials_per_photo=4,
        batch_size=old["batch_size"],
        original_native_jpeg_sha256=old["jpeg_sha256"]["own"],
        original_generator={
            k: old[k]
            for k in ("adapter_sha256", "dictionary_sha256", "turbo_sha256", "driver_sha256")
        },
    )
    freeze(root / "design.json", design)
    packet(root)
    tar = tar_bundle(a.out, root)
    receipt = dict(
        registration_sha256=LOCK,
        design_sha256=sha(root / "design.json"),
        packet_sha256=sha(root / PACKET),
        bundle_sha256=sha(tar),
        package_script_sha256=sha(Path(__file__)),
        renderer_sha256=sha(WRAPPER),
        slurm_sha256=sha(CODE.parent / "jobs/run_canonical_neighbor_personal12.sbatch"),
        generic_check_sha256=sha(a.out / "generic_support_check.json"),
        generic_check=g,
        source_native_packet_sha256=sha(b / PACKET),
        source_eeg_sha256=sha(wave),
        radius_sha256=sha(root / "radius.json"),
        n_queries=len(rows),
        n_nonself_donor=int(np.sum(rows != donors)),
        n_distinct_conditions=3,
        zero_alias="native",
        native_byte_identity=True,
        recipient_angle_range_radians=[float(angles.min()), float(angles.max())],
    )
    freeze(a.release, receipt)
    print(
        "Packet",
        receipt["packet_sha256"],
        "tar",
        receipt["bundle_sha256"],
        "donors",
        receipt["n_nonself_donor"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/canonical-neighbor/personal12"))
    parser.add_argument(
        "--release", type=Path, default=PUB / "canonical_neighbor_personal12_packet.json"
    )
    run(parser.parse_args())
