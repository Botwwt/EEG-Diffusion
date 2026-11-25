"""Package target-blind generation inputs separately from original-photo scoring inputs."""

import argparse
import json
import tarfile
from pathlib import Path

from render_within_photo_ip import check, read, sha

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity"


def package(a):
    inputs = {
        "repeat": (PUBLIC / "official_repeat_images_design.json", "render_packet.npz"),
        "global_angle": (
            PUBLIC / "official_global_angle_images_design.json",
            "global_angle_packet.npz",
        ),
        "global_named": (
            PUBLIC / "official_global_named_images_design.json",
            "global_angle_packet.npz",
        ),
        "dual_r80": (a.subject / "dual-r80/render_design.json", "dual-r80/render_packet.npz"),
    }
    design, packet_name = inputs[a.series]
    packet = a.subject / packet_name
    check(
        sha(packet) == read(design)["packet_sha256"], "EEG export differs from the selected design"
    )
    a.out.mkdir(exist_ok=True, parents=True)
    shared = {"packet.npz": packet, "design.json": design}
    core = ("render_official_repeat_images", "render_within_photo_ip", "render_official_words")
    receipts = {}
    for phase, extra in (
        ("input", ("render_official_ip_adapter",)),
        (
            "score-input",
            ("score_official_repeat_jpegs", "score_official_objects", "eval_instance_starts"),
        ),
    ):
        files = shared.copy()
        for name in (*core, *extra):
            files["code/" + name + ".py"] = CODE / (name + ".py")
        if phase == "score-input":
            for name in ("test_images.zip", "originals_manifest.csv"):
                files["references/" + name] = a.references / name
        path = a.out / f"{a.series}-{phase}.tar"
        with tarfile.open(path, "w") as archive:
            for member, src in files.items():
                archive.add(src, arcname=member, recursive=False)
        receipts[path.name] = sha(path)
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--series", choices=("repeat", "global_angle", "global_named", "dual_r80"), required=True
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--subject", type=Path, default=Path("outputs/official-repeat-object/sub-04"))
    p.add_argument(
        "--references", type=Path, default=Path("outputs/official-object-readout/sub-03/evaluation")
    )
    package(p.parse_args())
