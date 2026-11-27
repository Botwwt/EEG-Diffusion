"""Rebuild the original common-renderer statistics from the bound score packets."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

import numpy as np

from collect_results import BUDGETS, METHODS


FILES = ("render.json", "scores.json", "observations.npz", "inception_official.json", "inception_official.npz")


def verify(packet, canonical, stack_code):
    sys.path.insert(0, str(stack_code))
    from build_paper_uniform_results import METRIC, read_arm

    references = {}
    verified = {}
    with tarfile.open(packet) as source, TemporaryDirectory(dir=packet.parent) as temporary:
        cached = Path(temporary)
        members = [member.name for member in source.getmembers() if member.isfile()]
        expected = {
            f"images/{method}/{budget}/sub-{person:02d}/{name}"
            for budget in BUDGETS
            for method in METHODS
            for person in range(1, 11)
            for name in FILES
        }
        previews = {
            f"images/{method}/{budget}/sub-08/{i:05d}.png"
            for budget in BUDGETS for method in METHODS for i in range(6)
        }
        if set(members) != expected | previews or len(members) != len(expected) + len(previews):
            raise ValueError("Expected 120 unique score sets and 72 registered example images")
        for name in sorted(previews):
            folder, image = name.rsplit("/", 1)
            receipt = json.load(source.extractfile(folder + "/render.json"))
            query = int(image.removesuffix(".png"))
            if sha256(source.extractfile(name).read()).hexdigest() != receipt["pixels_sha256"][query]:
                raise ValueError(f"Different rendered image bytes for {name}")
        for name in sorted(expected):
            target = cached / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.extractfile(name).read())

        for budget in BUDGETS:
            conditions = None
            for method in METHODS:
                entry = canonical[budget][method]
                receipts = {record["subject"]: record for record in entry["subject_receipts"]}
                directories = [cached / "images" / method / budget / f"sub-{p:02d}" for p in range(1, 11)]
                for person, path in enumerate(directories, 1):
                    for name, key in (("scores.json", "score_sha256"), ("inception_official.json", "inception_sha256")):
                        if sha256((path / name).read_bytes()).hexdigest() != receipts[person][key]:
                            raise ValueError(f"Different registered scores for {budget}/{method}/sub-{person:02d}")
                result, data, protocol = read_arm(directories, references, common_budget=budget.removeprefix("official-"))
                if result["n"] != entry["n"] or result["n"] != 2000 or len(data) != 10 or result["missing"]:
                    raise ValueError(f"Incomplete score packets for {budget}/{method}")
                if conditions is not None and protocol != conditions:
                    raise ValueError(f"Changed generation protocol for {budget}/{method}")
                conditions = protocol
                means = result["mean"]
                comparables = {"inception_2way" if key == METRIC else key: value for key, value in means.items()}
                if entry["mean"].keys() != comparables.keys() or any(
                    not np.isclose(value, comparables[key], atol=1e-9, rtol=1e-7)
                    for key, value in entry["mean"].items()
                ):
                    raise ValueError(f"Table means differ for {budget}/{method}")
                verified[budget, method] = result["n"]
    print(f"Verified {sum(verified.values()):,} image scores in {len(verified)} table rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-code", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--packet", type=Path, default=Path(__file__).with_name("common-canonical-scores.tar"))
    args = parser.parse_args()
    verify(args.packet, json.loads(args.canonical.read_bytes()), args.stack_code)
