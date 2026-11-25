"""Stage immutable Brain-HiVE input files and verify each published SHA-256."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


RECIPE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PAPER_INPUTS_ROOT", "outputs/paper-benchmark/inputs")).resolve()
MANIFEST = RECIPE / "public_manifest.json"
AUTHOR_SOURCES = RECIPE / "author_sources.json"
MIB = 1024 * 1024
CHUNK = 32 * MIB


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches(path, asset):
    return path.is_file() and path.stat().st_size == asset["bytes"] and digest(path) == asset["sha256"]


def chunk_url(url, first, last, attempt):
    if url.startswith("https://osf.io/"):
        return url + "?" + urllib.parse.urlencode({"paper_range": f"{first}-{last}-{attempt}"})
    if not url.startswith("https://huggingface.co/"):
        return url
    return url + "?" + urllib.parse.urlencode({"download": "true", "data_range": f"{first}-{last}-{attempt}"})


def transfer_part(asset, parts, first):
    size = asset["bytes"]
    last = min(size, first + CHUNK) - 1
    length = last - first + 1
    path = parts / str(first)
    if path.is_file() and path.stat().st_size == length:
        return
    temp = parts / (str(first) + ".partial")
    for attempt in range(9):
        try:
            url = chunk_url(asset["url"], first, last, attempt)
            headers = {"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity",
                       "User-Agent": "Brain-HiVE research input verification"}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=50) as reply:
                expected = f"bytes {first}-{last}/{size}"
                single = first == 0 and length == size and reply.status == 200
                if not single and (reply.status != 206 or reply.headers.get("Content-Range") != expected):
                    raise OSError(f"invalid range {reply.status} {reply.headers.get('Content-Range')} for {expected}")
                with temp.open("wb") as stream:
                    remaining = length
                    while remaining:
                        chunk = reply.read(min(MIB, remaining))
                        if not chunk:
                            raise OSError(f"short chunk, missing {remaining} bytes")
                        stream.write(chunk)
                        remaining -= len(chunk)
                    if reply.read(1):
                        raise OSError("oversize chunk")
            temp.replace(path)
            return
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            if attempt == 8:
                raise
            if attempt in (2, 5):
                print(f"Retry {attempt + 1}: {asset['path']} @ {first}: {error}", flush=True)
            time.sleep(min(8, 0.8 * (attempt + 1)))


def acquire(asset, connections):
    dest = ROOT / asset["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not matches(dest, asset):
            raise ValueError(f"Existing file differs from the published checksum: {dest}")
        print("Verified", asset["path"], flush=True)
        return
    if not asset["url"].startswith("https://huggingface.co/") and asset["bytes"] <= CHUNK:
        request = urllib.request.Request(asset["url"], headers={"Accept-Encoding": "identity"})
        with tempfile.TemporaryDirectory(dir=dest.parent) as temp:
            file = Path(temp) / dest.name
            with urllib.request.urlopen(request, timeout=90) as reply, file.open("wb") as stream:
                shutil.copyfileobj(reply, stream, length=4 * MIB)
            if not matches(file, asset):
                raise ValueError(f"Published checksum mismatch for {asset['path']}")
            file.replace(dest)
        print("READY", asset["path"], asset["sha256"], flush=True)
        return
    parts = dest.with_name(dest.name + ".parts")
    parts.mkdir(exist_ok=True)
    offsets = range(0, asset["bytes"], CHUNK)
    print("Fetching", asset["path"], asset["bytes"], flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
        for _ in pool.map(lambda offset: transfer_part(asset, parts, offset), offsets):
            pass
    hasher = hashlib.sha256()
    temp = dest.with_name(dest.name + ".verifying")
    with temp.open("wb") as out:
        for offset in offsets:
            with (parts / str(offset)).open("rb") as src:
                for buf in iter(lambda: src.read(4 * MIB), b""):
                    hasher.update(buf)
                    out.write(buf)
    if hasher.hexdigest() != asset["sha256"] or temp.stat().st_size != asset["bytes"]:
        temp.unlink()
        for offset in offsets:
            (parts / str(offset)).unlink()
        raise ValueError(f"Published checksum mismatch: {dest}")
    temp.replace(dest)
    for offset in offsets:
        (parts / str(offset)).unlink()
    parts.rmdir()
    print("READY", asset["path"], hasher.hexdigest(), flush=True)


def stage_author_code():
    for folder, record in json.loads(AUTHOR_SOURCES.read_bytes()).items():
        out = ROOT / folder
        if all((out / name).is_file() and digest(out / name) == expected for name, expected in record["files"].items()):
            print("Verified", folder, record["revision"], flush=True)
            continue
        with urllib.request.urlopen(record["url"], timeout=90) as r:
            payload = r.read()
        if hashlib.sha256(payload).hexdigest() != record["archive_sha256"]:
            raise ValueError(f"Author source archive differs for {folder}")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name, expected in record["files"].items():
                content = archive.read(f"Brain-HIVE-{record['revision']}/{name}")
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Author source file differs: {folder}/{name}")
                file = out / name
                if file.is_file():
                    if digest(file) != expected:
                        raise ValueError(f"Existing author file differs: {file}")
                    continue
                file.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=file.parent) as temp:
                    local = Path(temp) / file.name
                    local.write_bytes(content)
                    local.replace(file)
        print("READY", folder, record["revision"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=("eeg", "images", "visual", "code", "fusion-encoder", "fusion-adapters"), default=["eeg"])
    parser.add_argument("--subjects", nargs="+", type=int, choices=range(1, 11))
    parser.add_argument("--splits", nargs="+", choices=("test", "train"), default=["test", "train"])
    parser.add_argument("--files", type=int, default=3, help="parallel files")
    parser.add_argument("--connections", type=int, default=4, help="parallel ranges per file")
    args = parser.parse_args()
    if args.files < 1 or args.connections < 1:
        parser.error("parallel counts must be positive")
    ROOT.mkdir(parents=True, exist_ok=True)
    if "code" in args.groups:
        stage_author_code()
    assets = json.loads(MANIFEST.read_bytes())["assets"]
    chosen = []
    for asset in assets:
        if asset["group"] not in args.groups:
            continue
        path = Path(asset["path"])
        if asset["group"] == "eeg":
            if args.subjects and int(path.parent.name[4:]) not in args.subjects:
                continue
            if path.stem not in args.splits:
                continue
        chosen.append(asset)
    missing = [a["bytes"] for a in chosen if not (ROOT / a["path"]).exists()]
    if missing and shutil.disk_usage(ROOT).free < sum(missing) + sum(sorted(missing, reverse=True)[:args.files]):
        raise OSError("Insufficient space for final files and parallel verification")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.files) as pool:
        futures = [pool.submit(acquire, asset, args.connections) for asset in chosen]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print("All requested inputs verified", len(chosen), sum(a["bytes"] for a in chosen), flush=True)


if __name__ == "__main__":
    main()
