"""Bundle the successful background attribute confirmation and its label-blind image inputs."""

import argparse
import io
import shutil
import tarfile
from pathlib import Path

from package_personal_confirmation import scientific_code
from register_fresh_photo_panel import freeze, read, sha
from render_background_prompt import LOCK, PANELS, SUB, controls
from render_within_photo_ip import check

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"


def package(a):
    protocol = PUBLIC / "background_prompt_image_design.json"
    confirm = PUBLIC / "background_attribute_confirmation.json"
    source = a.data / "eeg-background-confirmation/target"
    flags = read(source / "receipt.json")
    scored = read(PUBLIC / "fresh_mix_images.json")["panels"]
    check(
        read(confirm)["gate_passed"]
        and sha(protocol) == LOCK
        and flags["design"] == read(protocol)["confirmation_sha256"]
        and flags["suffix"] == read(protocol)["suffix"]
        and sha(confirm) == flags["confirmation_sha256"],
        "Wrong confirmation or prompt intervention",
    )
    root = a.out / "worker"
    root.mkdir(exist_ok=True, parents=True)
    b = a.data / "fresh-mix/render_inputs/learned"
    paths = {
        "protocol.json": protocol,
        "input/design.json": b / "design.json",
        "input/radius.json": b / "radius.json",
    }
    hashes = {}
    conditions = {}
    source_jpegs = {}
    for panel in PANELS:
        flag = source / (panel + ".npz")
        f = b / "conditions" / SUB / panel / "conditions_512.npz"
        cfg = b / "renders" / panel / SUB / "settings.json"
        meta = flags["panels"][panel]
        check(
            sha(flag) == meta["sha256"]
            and sha(cfg)
            == meta["source_settings_sha256"]
            == scored[panel]["settings_sha256"]["learned"]
            and sha(f) == meta["source_h14_sha256"] == scored[panel]["condition_sha256"]["learned"],
            "Changed source class, feature or word",
        )
        hashes[panel] = sha(flag)
        conditions[panel] = sha(f)
        source_jpegs[panel] = read(cfg)["jpeg_sha256"]["own"]
        paths["flags/" + panel + ".npz"] = flag
        paths["input/" + f.relative_to(b).as_posix()] = f
    for name, src in paths.items():
        dest = root / name
        dest.parent.mkdir(exist_ok=True, parents=True)
        check(not dest.exists() or sha(dest) == sha(src), "A frozen input differs")
        shutil.copyfile(src, dest)
    freeze(
        root / "worker.json",
        dict(
            protocol_sha256=LOCK,
            confirmation_sha256=sha(confirm),
            policy_sha256=sha(source / "receipt.json"),
            flags=hashes,
            conditions=conditions,
            baseline_jpeg_sha256=source_jpegs,
        ),
    )
    for panel in PANELS:
        controls(root, panel)
    members = scientific_code("render_background_prompt")
    members.update({"inputs/" + name: root / name for name in [*paths, "worker.json"]})
    path = a.out / "background-prompts.tar"
    tmp = path.with_suffix(".partial")
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as t:
        for name, p in sorted(members.items()):
            b = p.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(b)
            t.addfile(info, io.BytesIO(b))
    check(not path.exists() or sha(tmp) == sha(path), "Different bundle exists")
    tmp.replace(path)
    print(sha(path), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/background-prompt"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    package(p.parse_args())
