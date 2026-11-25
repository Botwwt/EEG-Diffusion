"""Make separate reconstruction and reference-scoring bundles for three single-person panels."""

import argparse
import ast
import io
import tarfile
from pathlib import Path

from render_within_photo_ip import check, packet, read, sha

CODE = Path(__file__).resolve().parent
PEOPLE = ("sub-04", "sub-05", "sub-01")


def scientific_code(entry):
    pending, files = [entry], {}
    while pending:
        name = pending.pop()
        src = CODE / f"{name}.py"
        if not src.is_file() or name in files:
            continue
        files[name] = src
        for node in ast.walk(ast.parse(src.read_bytes())):
            if isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                pending.extend(x.name.split(".")[0] for x in node.names)
    return {"code/" + p.name: p for p in files.values()}


def package(root, reference_root, out, series, people, stages):
    inputs = root / "render_inputs"
    common = {f"inputs/{n}.json": inputs / (n + ".json") for n in ("design", "radius")}
    for sub in people:
        packet(inputs, sub, "personal12")
        rel = f"conditions/{sub}/personal12/conditions_512.npz"
        common["inputs/" + rel] = inputs / rel
    evaluation = read(
        CODE.parent / "results/object_fidelity/personal12_image_evaluation_design.json"
    )
    out.mkdir(parents=True, exist_ok=True)
    entries = {
        "render": "render_within_photo_ip",
        "clip": "score_within_photo_ip",
        "names": "score_reference_names",
    }
    for stage in stages:
        members = scientific_code(entries[stage])
        if stage != "names":
            members.update(common)
        if stage in ("clip", "names"):
            for sub in people:
                path = reference_root / sub / "personal12_ablation.json"
                check(path.is_file(), f"Missing scoring registration for {sub}")
                check(
                    sha(path) == evaluation["reference_sha256"][sub],
                    f"Reference for {sub} differs from the frozen design",
                )
                members[f"references/{sub}/personal12_ablation.json"] = path
        target = out / f"{series}-{stage}.tar"
        partial = target.with_suffix(".partial")
        with tarfile.open(partial, "w") as tar:
            for member, path in sorted(members.items()):
                data = path.read_bytes()
                info = tarfile.TarInfo(member)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        partial.replace(target)
        print(target.name, sha(target))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/object-head-confirmation"))
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("outputs/object-head-confirmation/evaluation/references"),
    )
    parser.add_argument(
        "--series",
        choices=("personal12", "trial_personal12", "trial_replication05", "trial_replication01"),
        default="personal12",
    )
    parser.add_argument("--subjects", nargs="+", choices=PEOPLE, default=PEOPLE)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("render", "clip", "names"),
        default=("render", "clip", "names"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.root, args.reference_root, args.out, args.series, args.subjects, args.stages)
