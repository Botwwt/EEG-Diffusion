"""Build the source-only fresh-mix worker archive with reproducible tar headers."""

import argparse
import io
import tarfile
from pathlib import Path

from package_personal_confirmation import scientific_code
from render_fresh_mix_images import LOCK
from render_within_photo_ip import check, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/fresh-mix/cluster/fresh_mix_image_code.tar")
    )
    args = parser.parse_args()
    public = (
        Path(__file__).resolve().parents[1] / "results/object_fidelity/fresh_mix_images_design.json"
    )
    check(sha(public) == LOCK, "Image evaluation registration changed")
    members = scientific_code("name_fresh_mix_images") | scientific_code("render_fresh_mix_images")
    members["image_protocol.json"] = public
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(".partial")
    with tarfile.open(temp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, path in sorted(members.items()):
            data = path.read_bytes()
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    if args.out.exists():
        check(sha(temp) == sha(args.out), "Existing worker archive differs")
        temp.unlink()
    else:
        temp.replace(args.out)
    print(args.out, sha(args.out), len(members))


if __name__ == "__main__":
    main()
