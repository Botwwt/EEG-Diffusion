"""Bind two decoders of the same 80 observations to fixed image prompts and seeds."""

import argparse
from pathlib import Path

import numpy as np
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"


def register(root):
    case = root / "dual-r80"
    protocol = PUBLIC / "dual_r80_image_protocol.json"
    design = read(protocol)
    export = case / "render_packet.json"
    training = case / "fit_design.json"
    packet = case / "render_packet.npz"
    source = root / "global_angle_packet.npz"
    receipt = read(export)
    check(receipt["packet_sha256"] == sha(packet), "EEG export differs from its receipt")
    check(receipt["fit_design_sha256"] == sha(training), "Decoder fit design differs")
    check(
        sha(source) == read(PUBLIC / "official_global_named_images_design.json")["packet_sha256"],
        "Historical fixed-word baseline packet differs",
    )
    with np.load(source, allow_pickle=False) as old, np.load(packet, allow_pickle=False) as new:
        for key in ("rows", "donor", "radius"):
            check(np.array_equal(old[key], new[key]), f"Historical {key} differs")
        for slot in ("r4", "r80"):
            check(
                np.array_equal(old["r80_words"], new[slot + "_words"]),
                "Both visual heads must receive the identical EEG-predicted R80 word",
            )
            vectors = new[slot]
            check(
                vectors.shape == (200, 1024)
                and np.isfinite(vectors).all()
                and np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-6),
                f"Invalid generated-image condition for {slot}",
            )
    freeze(
        case / "render_design.json",
        dict(
            participant="sub-04",
            physical_trials_per_photo=80,
            arms=list(design["condition_format"]),
            historical_names=design["condition_format"],
            image_protocol_sha256=sha(protocol),
            fit_design_sha256=sha(training),
            export_receipt_sha256=sha(export),
            baseline_packet_sha256=sha(source),
            packet_sha256=sha(packet),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/official-repeat-object/sub-04"))
    register(parser.parse_args().root)
