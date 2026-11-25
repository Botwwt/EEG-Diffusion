"""Input-bound layout and PNG transactions for prospective paper renders."""

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

from filelock import FileLock
from PIL import Image


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def key(data):
    return hashlib.sha256(canonical(data)).hexdigest()


def atomic_file(path, write):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + "-", delete=False) as f:
        tmp = Path(f.name)
        try:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path, data):
    raw = (json.dumps(data, indent=1, ensure_ascii=True, allow_nan=False) + "\n").encode()
    atomic_file(path, lambda f: f.write(raw))


def track_for(runs, subject):
    require(1 <= subject <= 10, "Expected a participant from 1 to 10")
    name = Path(runs).name
    for kind in ("personal", "native-full"):
        if name.startswith(kind + "-"):
            require(name == f"{kind}-{subject:02d}", f"Participant {subject:02d} disagrees with the selected fit root {name}")
            return kind
    return "shared" if name == "shared" else "personal"


def stamp(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class PinnedFiles:
    """Hash actual input bytes once; detect source replacement or mutation before commit."""

    def __init__(self):
        self.records = {}

    def pin(self, path):
        path = Path(path).absolute()
        current = stamp(path)
        if path not in self.records:
            digest = sha(path)
            require(stamp(path) == current, f"Input changed while hashing: {path}")
            self.records[path] = current, digest
        before, digest = self.records[path]
        require(before == current, f"Pinned input changed: {path}")
        return digest

    def verify(self):
        for path, (before, _) in self.records.items():
            require(stamp(path) == before, f"Input changed while producing the cache: {path}")


def cached_layout(base, request, fit, validate, verify):
    """Publish a layout and its receipt under the hash of its actual training inputs."""
    import torch

    identity = key(request)
    folder = Path(base) / "v2" / identity
    folder.mkdir(parents=True, exist_ok=True)
    payload, receipt = folder / "ridge.pt", folder / "ridge.json"
    with FileLock(str(folder / ".lock")):
        verify()
        try:
            saved = json.loads(receipt.read_bytes()) if receipt.is_file() else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            saved = None
        if saved is not None:
            require(saved.get("request") == request and saved.get("request_sha256") == identity, "Layout receipt belongs to another input")
        if saved is not None and payload.is_file() and sha(payload) == saved["ridge_sha256"]:
            raw = torch.load(payload, map_location="cpu", weights_only=True)
            require(raw["request_sha256"] == identity, "Layout checkpoint and manifest disagree")
            validate(raw["layout"])
            verify()
            return raw["layout"], saved
        # A missing receipt is an interrupted fit. Recompute from the versioned inputs.
        # The existing registered hash, when present, stays authoritative during repair.
        pack = fit()
        validate(pack)
        with tempfile.TemporaryDirectory(dir=folder, prefix=".fit-") as tmp:
            stage = Path(tmp) / "ridge.pt"
            with stage.open("wb") as f:
                torch.save({"request_sha256": identity, "layout": pack}, f)
                f.flush()
                os.fsync(f.fileno())
            digest = sha(stage)
            require(saved is None or saved["ridge_sha256"] == digest,
                    "Refitted layout differs from its committed checkpoint; the previous receipt was preserved")
            verify()
            stage.replace(payload)
            result = {"cache_version": 2, "request": request, "request_sha256": identity, "ridge_sha256": digest}
            save_json(receipt, result)
        return pack, result


def png_hash(path):
    """Return None for an absent, incomplete, or incorrectly formatted 512-pixel RGB PNG."""
    if not path.is_file():
        return None
    raw = path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.mode != "RGB" or im.size != (512, 512):
                return None
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
    except (OSError, ValueError, SyntaxError):
        return None
    return hashlib.sha256(raw).hexdigest()


def render_transaction(out, request, generate, verify, batch_size=8):
    """Resume exact original batches. Keep legacy/different-input folders and existing scores intact."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = out.with_name("." + out.name + ".render-v2")
    final = out / "render.json"
    if final.is_file() and not state.exists():
        require(json.loads(final.read_bytes()).get("render_cache_version") == 2, f"Historical render has no v2 input binding: {out}")
    state.mkdir(exist_ok=True)
    proof, backup = state / "request.json", state / "final.json"
    identity, n = key(request), request["n"]
    names = [f"{i:05d}.png" for i in range(n)]
    receipt_base = {**request, "request_sha256": identity}
    with FileLock(str(state / ".lock")):
        verify()
        marker = json.loads(proof.read_bytes()) if proof.is_file() else None
        if marker is not None:
            require(marker == {"cache_version": 2, "request_sha256": identity},
                    f"Existing render transaction uses different inputs; use a distinct snapshot: {out}")
        try:
            committed = json.loads(final.read_bytes()) if final.is_file() else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            require(marker is not None and backup.is_file(), f"Cannot establish the source of a damaged receipt: {out}")
            committed = None
        stored = json.loads(backup.read_bytes()) if backup.is_file() else None
        if committed is not None:
            require(committed.get("render_cache_version") == 2, f"Historical render has no v2 input binding; preserve this folder: {out}")
        if committed is not None and stored is not None:
            if stored != committed:
                require(marker is not None and committed.get("request_sha256") == identity, f"Render and recovery receipts disagree: {out}")
                committed = None
        prior = committed if committed is not None else stored
        if prior is not None:
            require({k: v for k, v in prior.items() if k != "pixels_sha256"} == receipt_base,
                    f"Existing render has different inputs or sampling settings; preserve this folder: {out}")
            expected = prior.get("pixels_sha256", [])
            require(len(expected) == n and all(isinstance(x, str) and re.fullmatch("[0-9a-f]{64}", x) for x in expected),
                    f"Incomplete committed PNG hash list: {out}")
        else:
            expected = None
            require(marker is not None or not out.is_dir() or not any(out.iterdir()), f"Unbound existing render files: {out}")
        if marker is None:
            save_json(proof, {"cache_version": 2, "request_sha256": identity})
        if out.is_dir():
            allowed = set(names)
            extras = [p.name for p in out.glob("*.png") if p.stem.isdecimal() and p.name not in allowed]
            require(not extras, f"Unexpected numbered PNGs in {out}: {extras[:4]}")
        original = [png_hash(out / name) for name in names]
        if committed is not None and original == expected:
            verify()
            return False
        stage = state / "pending"
        stage.mkdir(exist_ok=True)
        digests, pending = [], []
        for first in range(0, n, batch_size):
            end = min(n, first + batch_size)
            ledger = state / f"batch-{first:05d}.json"
            want = expected[first:end] if expected is not None else None
            if want is not None and original[first:end] == want:
                digests.extend(want)
                continue
            try:
                saved = json.loads(ledger.read_bytes()) if ledger.exists() else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                saved = None
            if saved is not None:
                require(saved["request_sha256"] == identity and saved["first"] == first, "Render batch journal differs")
                recorded = saved.get("sha256")
                if want is None and isinstance(recorded, list) and len(recorded) == end - first and all(
                    isinstance(d, str) and re.fullmatch("[0-9a-f]{64}", d) for d in recorded
                ):
                    want = saved["sha256"]
            actual = [png_hash(stage / name) for name in names[first:end]]
            if want is None or len(want) != end - first or any(d is None for d in actual) or actual != want:
                pending.append(first)
            digests.extend(want if want is not None else [None] * (end - first))

        def batch_complete(first, end):
            actual = [png_hash(stage / name) for name in names[first:end]]
            require(len(actual) == end - first and all(actual), "Incomplete generated PNG batch")
            previous = digests[first:end]
            require(all(old is None or old == new for old, new in zip(previous, actual, strict=True)),
                    f"Regenerated pixels differ from the committed same-input batch {first} in {out}")
            digests[first:end] = actual
            save_json(state / f"batch-{first:05d}.json", {"request_sha256": identity, "first": first, "sha256": actual})

        if pending:
            generate(stage, pending, batch_complete)
        require(len(digests) == n and all(digests) and (expected is None or digests == expected),
                "Render did not complete every requested original batch")
        result = {**receipt_base, "pixels_sha256": digests}
        verify()
        for i, name in enumerate(names):
            if original[i] != digests[i]:
                require(png_hash(stage / name) == digests[i], f"Staged PNG differs before publication: {name}")
        if (out / "scores.json").is_file():
            score = json.loads((out / "scores.json").read_bytes())
            require(prior is not None and score.get("ordered_image_hashes_sha256") == hashlib.sha256("\n".join(digests).encode()).hexdigest(),
                    "Existing image scoring is not bound to the recoverable render")
            raw = (json.dumps(result, indent=1, ensure_ascii=True, allow_nan=False) + "\n").encode()
            require(score.get("render_sha256", hashlib.sha256(raw).hexdigest()) == hashlib.sha256(raw).hexdigest(),
                    "Repair would invalidate an existing score receipt")
        save_json(backup, result)
        if not out.exists():
            save_json(stage / "render.json", result)
            stage.rename(out)
        else:
            for i, name in enumerate(names):
                if original[i] != digests[i]:
                    (stage / name).replace(out / name)
            if committed is None:
                save_json(final, result)
        return True
