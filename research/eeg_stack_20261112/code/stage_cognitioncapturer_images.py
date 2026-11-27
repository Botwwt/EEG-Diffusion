"""Bind CognitionCapturer's released pixels to the registered THINGS-EEG2 test rows."""

import argparse
import csv
import hashlib
import io
import re
import tarfile
from pathlib import Path, PurePosixPath

ARCHIVE_SHA256 = "fc8b4bfddb11375d9185b3e37268783a2748d3b03ab423bf6e5765611bf5a84d"
SUBJECTS = tuple(f"sub-{i:02d}" for i in range(1, 11))


def pack_release(source, gallery, destination):
    with source.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Unexpected CognitionCapturer author archive")
    with gallery.open(encoding="utf-8-sig", newline="") as f:
        photo_rows = list(csv.DictReader(f))
    match = {r["filename"]: r for r in photo_rows}
    if len(match) != 200 or [int(r["row0"]) for r in photo_rows] != list(range(200)):
        raise ValueError("Reference manifest must contain 200 unique and ordered photos")
    manifests = {sub: {} for sub in SUBJECTS}
    originals = set()

    def add_file(output, name, data):
        meta = tarfile.TarInfo(name)
        meta.size = len(data)
        output.addfile(meta, io.BytesIO(data))

    partial = destination.with_name(destination.name + ".partial")
    try:
        with tarfile.open(source, "r|gz") as archive, tarfile.open(partial, "w") as output:
            for member in archive:
                if not member.isfile():
                    continue
                folder, name = PurePosixPath(member.name).parts[-2:]
                if name.startswith(".fuse_hidden"):
                    continue
                ref = match[name]
                data = archive.extractfile(member).read()
                digest = hashlib.sha256(data).hexdigest()
                if folder == "original_image":
                    if name in originals or digest != ref["image_sha256"]:
                        raise ValueError(f"Reference image differs for {name}")
                    originals.add(name)
                    continue
                subject_match = re.fullmatch(r"(sub-\d{2})_reconstructed_image", folder)
                if not subject_match or subject_match[1] not in manifests:
                    raise ValueError(f"Unknown subject folder {folder}")
                subject = subject_match[1]
                entries = manifests[subject]
                row = int(ref["row0"])
                if row in entries:
                    raise ValueError(f"Duplicate reconstruction in {subject} at {row}")
                relative = f"{row:03d}.jpg"
                location = "cogcap-" + subject.replace("-", "")
                add_file(output, f"{location}/{relative}", data)
                entries[row] = dict(eeg_row=row, unique_id=ref["unique_id"], filename=relative,
                                    subject=subject, draw="published", reconstruction_member=member.name,
                                    reconstruction_sha256=digest, reference_member=ref["archive_member"],
                                    reference_sha256=ref["image_sha256"], archive_sha256=ARCHIVE_SHA256)
            if originals != set(match) or any(set(entries) != set(range(200)) for entries in manifests.values()):
                raise ValueError("Incomplete source coverage")
            for subject, entries in manifests.items():
                output_csv = io.StringIO(newline="")
                writer = csv.DictWriter(output_csv, fieldnames=list(entries[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(entries[i] for i in range(200))
                location = "cogcap-" + subject.replace("-", "")
                add_file(output, f"{location}/manifest.csv", output_csv.getvalue().encode())
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("gallery", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    pack_release(args.source, args.gallery, args.destination)


if __name__ == "__main__":
    main()
