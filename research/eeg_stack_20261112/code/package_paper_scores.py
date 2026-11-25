"""Export complete score receipts and fixed examples from a Slurm run for private review."""
import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def export(root, dest):
    receipts = []
    dino = []
    preview_rows = {"official": np.linspace(0, 199, 16, dtype=int).tolist(),
                    "fresh": [10 * i + j for i in (0, 4, 10, 18, 26, 36, 43, 49) for j in (0, 1)]}
    with zipfile.ZipFile(dest.with_suffix(".partial"), "w", zipfile.ZIP_DEFLATED, compresslevel=2) as z:
        for name in ("data/prep/paper_split.json", "data/prep/paper_development.json", "data/prep/train_ids.json",
                     "data/prep/train_concept.npy", "data/prep/test_ids.json", "data/prep/paper_test_F_receipt.json",
                     "expected-inputs.json", "input-integrity.json"):
            if (root / name).is_file():
                z.write(root / name, name)
        for fit in sorted(root.glob("fits/*/enc/native/sub-*/fold*_seed2025/fit.json")):
            z.write(fit, fit.relative_to(root).as_posix())
        folders = [*root.glob("images/*/*/*/sub-*/scores.json"), *root.glob("side/*/*/scored/scores.json"),
                   *root.glob("external/atm-sub*/scores.json"), *root.glob("calibration/dev/*/scores.json"),
                   *root.glob("exploratory/*/dev-r4/sub-*/scores.json")]
        for p in sorted(folders):
            folder = p.parent
            blob = (folder / "observations.npz").read_bytes()
            score_bytes = p.read_bytes()
            s = json.loads(score_bytes)
            assert hashlib.sha256(blob).hexdigest() == s["observations_sha256"]
            rel = folder.relative_to(root)
            z.writestr((rel / "observations.npz").as_posix(), blob)
            z.writestr((rel / "scores.json").as_posix(), score_bytes)
            receipts.append(dict(folder=rel.as_posix(), scores_sha256=hashlib.sha256(score_bytes).hexdigest(),
                                 n=s["n"], observations_sha256=s["observations_sha256"]))
            for name in ("render.json", "manifest.csv"):
                if (folder / name).is_file():
                    z.write(folder / name, (rel / name).as_posix())
            if (folder / "dino.json").is_file():
                report = (folder / "dino.json").read_bytes()
                data = json.loads(report)
                payload = (folder / "dino.npz").read_bytes()
                assert hashlib.sha256(payload).hexdigest() == data["dino_npz_sha256"]
                with np.load(io.BytesIO(payload), allow_pickle=False) as original:
                    compact = io.BytesIO()
                    names = ["row", "concept", *data["per_row_metrics"]]
                    np.savez_compressed(compact, **{k: original[k] for k in names})
                packed = compact.getvalue()
                z.writestr((rel / "dino.json").as_posix(), report)
                z.writestr((rel / "dino_rows.npz").as_posix(), packed)
                dino.append(dict(folder=rel.as_posix(), n=s["n"], scores_sha256=hashlib.sha256(report).hexdigest(),
                                 observations_sha256=hashlib.sha256(packed).hexdigest(), complete_matrix_sha256=data["dino_npz_sha256"]))
            if rel.parts[0] != "images" or rel.parts[-1] not in {"sub-01", "sub-08"}:
                continue
            if rel.parts[-3] not in {"native-paper", "matched", "d2", "wrong", "response"}:
                continue
            if rel.parts[-2] not in {"official-r4", "fresh-r4"}:
                continue
            meta = json.loads((folder / "render.json").read_bytes())
            for i in preview_rows[s["panel"]]:
                image = (folder / f"{i:05d}.png").read_bytes()
                assert hashlib.sha256(image).hexdigest() == meta["pixels_sha256"][i]
                with Image.open(io.BytesIO(image)) as im:
                    thumb = im.convert("RGB").resize((192, 192), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    thumb.save(buffer, format="JPEG", quality=90)
                    z.writestr((rel / "preview" / f"{i:05d}.jpg").as_posix(), buffer.getvalue())
        features = []
        for path in sorted(root.glob("features/*/*/*/sub-*/features.json")):
            meta = path.read_bytes()
            obs = path.with_name("features.npz").read_bytes()
            record = json.loads(meta)
            assert hashlib.sha256(obs).hexdigest() == record["observations_sha256"]
            folder = path.parent.relative_to(root)
            z.writestr((folder / "features.json").as_posix(), meta)
            z.writestr((folder / "features.npz").as_posix(), obs)
            features.append(dict(folder=folder.as_posix(), n=record["n"], scores_sha256=hashlib.sha256(meta).hexdigest(),
                                 observations_sha256=record["observations_sha256"]))
        z.writestr("coverage.json", json.dumps({"rows": receipts, "features": features, "dino": dino,
                                               "preview_rows": preview_rows}, indent=1))
    dest.with_suffix(".partial").replace(dest)
    print("Saved", len(receipts), "complete scoring runs to", dest, flush=True)


def unpack(archive, dest):
    """Validate the score manifests, then publish the small export locally."""
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or source.testzip() is not None:
            raise ValueError("Duplicate or corrupt entry in score export")
        if any(dest not in (dest / name).resolve().parents for name in names):
            raise ValueError("Export member outside the destination")
        coverage = json.loads(source.read("coverage.json"))
        for section, receipt, obs in (("rows", "scores.json", "observations.npz"), ("features", "features.json", "features.npz")):
            groups = coverage.get(section, [])
            if len(groups) != len({row["folder"] for row in groups}):
                raise ValueError(f"Duplicate {section} folder")
            for row in groups:
                folder = row["folder"]
                metadata = source.read(f"{folder}/{receipt}")
                observations = source.read(f"{folder}/{obs}")
                meta = json.loads(metadata)
                if not (hashlib.sha256(metadata).hexdigest() == row["scores_sha256"]
                        and hashlib.sha256(observations).hexdigest() == row["observations_sha256"] == meta["observations_sha256"]
                        and meta["n"] == row["n"]):
                    raise ValueError(f"Score or observation receipt mismatch in {folder}")
        for row in coverage.get("dino", []):
            folder = row["folder"]
            meta = source.read(f"{folder}/dino.json")
            rows = source.read(f"{folder}/dino_rows.npz")
            if (hashlib.sha256(meta).hexdigest() != row["scores_sha256"]
                    or hashlib.sha256(rows).hexdigest() != row["observations_sha256"]
                    or json.loads(meta)["dino_npz_sha256"] != row["complete_matrix_sha256"]):
                raise ValueError(f"DINO receipt or per-stimulus observations differ in {folder}")
        for name in [n for n in names if n != "coverage.json"] + ["coverage.json"]:
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                tmp.write(source.read(name))
            Path(tmp.name).replace(path)
    print("Verified", len(coverage["rows"]), "RGB,", len(coverage.get("features", [])), "EEG-feature and",
          len(coverage.get("dino", [])), "DINO runs")


def scores_archive(source_path, output):
    """Publish checksums, metadata and per-stimulus metrics for independent recalculation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".partial")
    split_files = {"paper_split.json", "paper_development.json", "train_ids.json", "test_ids.json",
                   "train_concept.npy", "paper_test_F_receipt.json"}
    with zipfile.ZipFile(source_path) as src, zipfile.ZipFile(tmp, "w") as dst:
        for name in sorted(src.namelist()):
            parts = Path(name).parts
            keep = name in ("coverage.json", "input-integrity.json", "expected-inputs.json")
            if len(parts) >= 3:
                keep |= parts[:2] == ("data", "prep") and parts[-1] in split_files
                keep |= parts[0] == "fits" and parts[-1] == "fit.json"
                keep |= parts[0] == "features" and parts[-1] in ("features.json", "features.npz")
                keep |= parts[0] in ("images", "side", "external", "calibration", "exploratory") and parts[-1] in (
                    "scores.json", "observations.npz", "render.json", "manifest.csv", "dino.json", "dino_rows.npz")
            if keep:
                info = zipfile.ZipInfo(name)
                dst.writestr(info, src.read(name), compress_type=zipfile.ZIP_DEFLATED, compresslevel=8)
    tmp.replace(output)
    print(output, hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--unpack", action="store_true", help="Read a score ZIP and verify it into a local result directory")
    mode.add_argument("--scores-only", action="store_true", help="Create the public score archive from an exported ZIP")
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.scores_only:
        scores_archive(a.root, a.output)
    else:
        (unpack if a.unpack else export)(a.root, a.output)
