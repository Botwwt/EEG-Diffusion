"""Render recipient-matched native, local EEG update and swapped local EEG update."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from render_within_photo_ip import check, draw_packets, read, save, sha

LOCK = "1c3a68f20f8dc8da7208bb47754314f70860509d3b658a6e2ce4ff9771b9a1a1"
PANEL = "personal12"
SUBJECT = "sub-04"
PACKET = Path("conditions") / SUBJECT / PANEL / "conditions_512.npz"


def direction(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def visual_update(native, prior, local, donors):
    a = direction(prior.astype(float))
    b = native.astype(float)
    dot = np.sum(a * b, axis=1, keepdims=True)
    t = b - dot * a
    theta = np.arctan2(np.linalg.norm(t, axis=1, keepdims=True), dot)
    base = direction(t)

    def adjust(delta):
        mix = (base + delta) / 2
        good = (np.linalg.norm(delta, axis=1) > 1e-10) & (np.linalg.norm(mix, axis=1) > 1e-10)
        adjusted = np.cos(theta) * a + np.sin(theta) * direction(mix)
        return np.where(good[:, None], adjusted, b).astype(np.float32)

    return adjust(local), adjust(local[donors]), theta[:, 0]


def packet(root):
    plan = read(root / "design.json")
    check(
        sha(root / "protocol.json") == LOCK == plan["registration_sha256"], "Changed registration"
    )
    check(sha(root / PACKET) == plan["packet_sha256"], "Changed frozen conditions")
    old = read(root / "protocol.json")["source_sha256"]
    check(
        sha(root / "radius.json") == old["radius"]
        and sha(Path(__file__).with_name("render_within_photo_ip.py")) == old["renderer"],
        "Changed scale or renderer",
    )
    radius = read(root / "radius.json")
    with np.load(root / PACKET, allow_pickle=False) as z:
        rows, folds, words, donors, supports = (
            z[k] for k in ("rows", "folds", "predicted_words", "donor_rows", "support_rows")
        )
        native, prior, delta = (z[k] for k in ("native", "prior", "local_tangent"))
        actual = {key: z[key] for key in ("native", "own", "wrong")}
        check(
            rows.shape == folds.shape == words.shape == donors.shape == (120,)
            and supports.shape == (120, 8)
            and len(set(rows)) == 120
            and set(rows) == set(donors)
            and set(folds) == set(range(5)),
            "Invalid query or support identity",
        )
        check(
            len(z["canonical_training_classes"]) == 884
            and not np.intersect1d(rows // 10, z["canonical_training_classes"]).size,
            "Visual gallery contains a query class",
        )
        check(
            str(z["original_conditions_sha256"]) == old["baseline"]
            and str(z["target_eeg_sha256"]) == old["target_waveforms"],
            "Changed upstream condition dependencies",
        )
        index = {r: i for i, r in enumerate(rows)}
        j = np.array([index[r] for r in donors])
        check(
            np.array_equal(words, words[j])
            and np.array_equal(folds, folds[j])
            and np.array_equal(supports, supports[j])
            and np.array_equal(prior, prior[j]),
            "Invalid donor substitution",
        )
        check(
            all(
                len(set(s)) == 8 and set(s) <= set(rows[folds != f])
                for s, f in zip(supports, folds, strict=True)
            ),
            "Support fold mismatch",
        )
        check(
            np.array_equal(native, z["zero"])
            and np.array_equal(
                actual["wrong"][j == np.arange(120)], actual["own"][j == np.arange(120)]
            ),
            "Zero term or self-donor changed an image direction",
        )
        for v in [prior, *actual.values()]:
            check(
                v.shape == (120, 1024) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=2e-6),
                "Invalid visual direction",
            )
        check(
            delta.shape == (120, 1024)
            and np.isfinite(delta).all()
            and np.max(np.abs((direction(prior.astype(float)) * delta).sum(1))) < 2e-6,
            "Invalid local tangent",
        )
        norm = np.linalg.norm(delta, axis=1)
        check(
            np.all((norm < 1e-10) | (np.abs(norm - 1) < 2e-6)),
            "Local update must be zero or a unit tangent",
        )
        own, wrong, angle = visual_update(native, prior, delta, j)
        check(
            np.allclose(angle, z["recipient_angle_radians"], atol=1e-8, rtol=0)
            and np.allclose(own, actual["own"], atol=2e-7, rtol=0)
            and np.allclose(wrong, actual["wrong"], atol=2e-7, rtol=0),
            "Recipient direction, angle or local donor differs",
        )
        for arm in ["own", "wrong"]:
            a = direction(prior.astype(float))
            b = actual[arm].astype(float)
            dot = (a * b).sum(1, keepdims=True)
            got = np.arctan2(np.linalg.norm(b - dot * a, axis=1), dot[:, 0])
            check(np.max(np.abs(got - angle)) < 2e-6, "Angle was not retained")
    return (
        rows,
        folds,
        words,
        donors,
        supports,
        actual,
        float(radius["median_norm"]),
        plan["packet_sha256"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    data = packet(args.root)
    plan = read(args.root / "design.json")
    if args.check_only:
        print("Verified native, own, wrong and exact zero alias", data[-1])
        return
    args.subject = SUBJECT
    args.batch_size = plan["batch_size"]
    check(args.archive and args.adapter and args.dictionary, "Offline generation assets required")
    check(
        sha(args.dictionary) == plan["original_generator"]["dictionary_sha256"]
        and sha(
            args.archive / "assets/native_generator_v243/turbo/sd_xl_turbo_1.0_fp16.safetensors"
        )
        == plan["original_generator"]["turbo_sha256"],
        "Historical dictionary or model differs",
    )
    draw_packets(args, {PANEL: data})
    dest = args.root / "renders" / PANEL / SUBJECT
    settings = read(dest / "settings.json")
    hashes = settings["jpeg_sha256"]
    alias = dest / "images/zero"
    alias.mkdir(exist_ok=True)
    for i, h in enumerate(hashes["native"]):
        path = alias / f"{i:03d}.jpg"
        if not path.is_file():
            shutil.copyfile(dest / "images/native" / path.name, path)
        check(sha(path) == h, "Zero alias JPEG mismatch")
    hashes["zero"] = hashes["native"]
    settings.update(
        zero_alias="native",
        registration_sha256=LOCK,
        wrapper_sha256=sha(Path(__file__)),
        native_identical_to_previous=sum(
            a == b
            for a, b in zip(hashes["native"], plan["original_native_jpeg_sha256"], strict=True)
        ),
    )
    save(dest / "settings.json", settings)
    print("Native parity with historical batch:", settings["native_identical_to_previous"], "/120")


if __name__ == "__main__":
    main()
