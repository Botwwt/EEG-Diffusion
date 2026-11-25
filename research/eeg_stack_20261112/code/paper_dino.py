"""Rescore complete official/fresh paper folders against their registered photo galleries.

python paper_dino.py FOLDER [FOLDER ...] --model /path/to/facebook-dinov2-small --device cuda
The default --panel auto reads each folder's scores.json. Reference inputs use ARCHIVE and
RAD_ROOT/data/prep. Only dino.npz and dino.json are published in a scored folder.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import PIL
import skimage
import torch
import transformers
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from transformers import AutoImageProcessor, Dinov2Model


REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_FILES = {
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
}
OFFICIAL_ZIP = "57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a"
OFFICIAL_MANIFEST = "74773a1ef288b8d1c974a256a6b1c39940949c36842fca1302dc76e641943089"
TRAIN_ZIPS = {
    "b91edf1a316142406a2ae28bdbb673712f1d5c38f57c0aec9dcfb5da446cabf1",
    "dc9cc627141ba230641f729f84b80eb01f0a963d288c971eefcf76b4a7581f3a",
}
# SHA-256 of newline-joined "row<TAB>member<TAB>concept<TAB>original JPEG SHA-256",
# independently assembled from the public photo index and registered panel rows.
GALLERY_SHA = {
    "official": "31d85c5dbd8e0f158d7546d141b55b89f07a027217cd1a3752bec0fcfd57c441",
    "fresh": "23ff3bf5edb4456d0962af940f2647f6350309cb840edc90be5d8f3b7b6be89b",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_csv(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def rgb_tile(raw):
    with Image.open(io.BytesIO(raw)) as im:
        require(im.width == im.height and im.width >= 256, f"Expected a square photo or reconstruction, got {im.size}")
        return im.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)


def build_encoder(model_path, device, batch_size):
    for name, value in MODEL_FILES.items():
        require(sha(model_path / name) == value, f"DINO release file differs: {name}")
    proc = AutoImageProcessor.from_pretrained(str(model_path), use_fast=False, local_files_only=True)
    model = Dinov2Model.from_pretrained(str(model_path), local_files_only=True, use_safetensors=True)
    require((model.config.patch_size, model.config.hidden_size, model.config.num_hidden_layers) == (14, 384, 12),
            "Expected the frozen DINOv2-S/14 final CLS representation")
    model = model.eval().requires_grad_(False).float().to(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    @torch.inference_mode()
    def encode(images):
        blocks = []
        for lo in range(0, len(images), batch_size):
            pixels = proc(images=images[lo:lo + batch_size], return_tensors="pt")["pixel_values"]
            require(tuple(pixels.shape[1:]) == (3, 224, 224), "Unexpected DINO processor geometry")
            cls = model(pixel_values=pixels.to(device)).last_hidden_state[:, 0].float().cpu().numpy()
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            require(cls.shape[1] == 384 and np.isfinite(norms).all() and np.all(norms > 0), "Invalid DINO features")
            blocks.append(cls / norms)
        return np.concatenate(blocks).astype(np.float64)

    return encode


@dataclass
class Gallery:
    rows: np.ndarray
    concepts: np.ndarray
    members: list
    hashes: list
    tiles: list
    features: np.ndarray
    split_sha: str
    source: dict


def gallery(panel, archive, prep, encode):
    split_bytes = (prep / "paper_split.json").read_bytes()
    split = json.loads(split_bytes)
    rows = np.asarray(split[f"{panel}_rows"])
    n = 200 if panel == "official" else 500
    require(rows.shape == (n,) and rows.dtype.kind in "iu" and np.all(rows >= 0) and len(set(rows.tolist())) == n,
            f"Invalid registered {panel} query rows")
    expected = None
    if panel == "official":
        off = archive / "runs/visual_fidelity_20261125_000818/official"
        path = off / "test_images.zip"
        index = (off / "originals_manifest.csv").read_bytes()
        require(digest(index) == OFFICIAL_MANIFEST, "Official originals manifest SHA-256 differs")
        records = read_csv(index)
        require(np.array_equal(rows, np.arange(n)) and [int(r["eeg_row"]) for r in records] == rows.tolist(),
                "Official reference row order differs")
        members = [r["archive_member"] for r in records]
        labels = [r["unique_id"] for r in records]
        expected = [r["image_sha256"] for r in records]
        allowed_zip = {OFFICIAL_ZIP}
    else:
        path = archive / "assets/things_images_v254/training_images.zip"
        index = (prep / "train_ids.json").read_bytes()
        ids = json.loads(index)
        require(len(ids["ids"]) == len(ids["paths"]) == 16540 and np.max(rows) < 16540, "Invalid training photo index")
        members = [ids["paths"][i] for i in rows]
        require([PurePosixPath(m).stem for m in members] == [ids["ids"][i] for i in rows], "Training JPEG identity mismatch")
        labels = [PurePosixPath(m).parent.name.split("_", 1)[1] for m in members]
        allowed_zip = TRAIN_ZIPS
    file_sha = sha(path)
    require(file_sha in allowed_zip, f"Unexpected original photo archive SHA-256: {file_sha}")
    require(len(set(members)) == n and set(Counter(labels).values()) == {1 if panel == "official" else 10},
            f"Unexpected concept multiplicity in {panel}")
    tiles, hashes = [], []
    with zipfile.ZipFile(path) as z:
        for i, member in enumerate(members):
            raw = z.read(member)
            hashes.append(digest(raw))
            require(expected is None or hashes[-1] == expected[i], f"Official JPEG SHA-256 mismatch: {member}")
            tiles.append(rgb_tile(raw))
    bound = digest("\n".join(f"{r}\t{m}\t{c}\t{h}" for r, m, c, h in zip(rows, members, labels, hashes, strict=True)).encode())
    require(bound == GALLERY_SHA[panel], f"{panel} row/member/concept/JPEG hashes differ from the frozen gallery")
    print(f"Encoding verified {panel} reference gallery ({n} original photos)", flush=True)
    sources = {"gallery_sha256": bound, "archive_sha256": file_sha, "reference_index_sha256": digest(index),
               "paper_split_sha256": digest(split_bytes)}
    return Gallery(rows, np.array(labels), members, hashes, tiles, encode(tiles), digest(split_bytes), sources)


def query_input(folder, panel, ref, score_bytes, score):
    n = len(ref.rows)
    require(score.get("raw_RGB") is True and score["n"] == n and score["panel"] == panel, "Primary score panel or size differs")
    inputs = {"scores.json": digest(score_bytes)}
    observed = (folder / "observations.npz").read_bytes()
    inputs["observations.npz"] = digest(observed)
    require(inputs["observations.npz"] == score["observations_sha256"], "Primary observations SHA-256 differs")
    with np.load(io.BytesIO(observed), allow_pickle=False) as obs:
        require(obs["row"].dtype.kind in "iu" and np.array_equal(obs["row"], ref.rows)
                and np.array_equal(obs["concept"], ref.concepts), "Primary observation row/concept does not match reference gallery")
    declared, names, identity = [], [], {}
    render_path, manifest_path = folder / "render.json", folder / "manifest.csv"
    for path, key in [(render_path, "render_sha256"), (manifest_path, "source_manifest_sha256")]:
        if not path.is_file():
            require(key not in score, f"Primary scorer recorded a missing {path.name}")
            continue
        raw = path.read_bytes()
        inputs[path.name] = digest(raw)
        require(key not in score or score[key] == digest(raw), f"Primary scorer's {path.name} digest differs")
        if path == render_path:
            meta = json.loads(raw)
            require(meta["row"] == ref.rows.tolist() and meta["n"] == n and meta["budget"].split("-")[0] == panel,
                    "Render row order or panel differs")
            require(meta.get("split_sha256", ref.split_sha) == ref.split_sha, "Render registered a different split")
            declared.append(meta["pixels_sha256"])
            identity.update({k: meta[k] for k in ("training", "arm", "subject", "budget") if k in meta})
        else:
            manifest = read_csv(raw)
            require([int(r["eeg_row"]) for r in manifest] == ref.rows.tolist()
                    and [r["unique_id"] for r in manifest] == ref.concepts.tolist()
                    and [r["reference_member"] for r in manifest] == ref.members
                    and [r["reference_sha256"] for r in manifest] == ref.hashes, "External manifest row/label/original SHA-256 differs")
            declared.append([r["reconstruction_sha256"] for r in manifest])
            names = [r["filename"] for r in manifest]
            require(len(set(names)) == n and all(Path(x).name == x for x in names), "Invalid external reconstruction names")
            for key in ("subject", "draw"):
                if key in manifest[0]:
                    values = {r[key] for r in manifest}
                    require(len(values) == 1, f"Mixed external {key} in one folder")
                    identity[key] = next(iter(values))
    require(declared and all(len(v) == n and v == declared[0] for v in declared), "Missing or inconsistent render/manifest PNG hashes")
    formats = [fmt for fmt in ("{:05d}.png", "{:03d}.png", "{:03d}.jpg") if (folder / fmt.format(0)).is_file()]
    require(len(formats) <= 1, "Ambiguous numbered reconstruction files")
    if formats:
        names = [formats[0].format(i) for i in range(n)]
    require(len(names) == n, "No complete numbered or manifest-based query filename mapping")
    tiles = []
    for i, (name, expected) in enumerate(zip(names, declared[0], strict=True)):
        raw = (folder / name).read_bytes()
        require(digest(raw) == expected, f"Reconstruction SHA-256 mismatch at row {ref.rows[i]}: {name}")
        tiles.append(rgb_tile(raw))
    ordered_sha = digest("\n".join(declared[0]).encode())
    require(score.get("ordered_image_hashes_sha256", ordered_sha) == ordered_sha, "Scored image hash sequence differs")
    return tiles, declared[0], inputs, identity, ordered_sha


def rank_metrics(matrix):
    positive = np.diag(matrix)[:, None]
    above = (matrix > positive).sum(1)
    tied = (matrix == positive).sum(1)
    require(np.all(tied > 0) and np.isfinite(matrix).all(), "Unrankable DINO matrix")
    return {"top1": np.clip((1 - above) / tied, 0, 1), "top5": np.clip((5 - above) / tied, 0, 1),
            "correct_cosine": positive.ravel(), "rank": 1 + above + .5 * (tied - 1),
            "2way": (len(matrix) - above - tied + .5 * (tied - 1)) / (len(matrix) - 1)}


def metrics(tiles, ref, encode, panel):
    matrix = encode(tiles) @ ref.features.T
    values = {f"dino_{key}": v for key, v in rank_metrics(matrix).items()}
    if panel == "fresh":
        for indices in (np.flatnonzero(ref.concepts == c) for c in np.unique(ref.concepts)):
            require(len(indices) == 10, "Expected ten fresh originals per concept")
            for k, v in rank_metrics(matrix[np.ix_(indices, indices)]).items():
                if k == "correct_cosine":
                    continue
                key = "dino_within10" if k == "top1" else f"dino_within10_{k}"
                if key not in values:
                    values[key] = np.empty(len(ref.rows))
                values[key][indices] = v
    pairs = []
    for generated, original in zip(tiles, ref.tiles, strict=True):
        a, b = [np.asarray(im, dtype=np.float32) / 255 for im in (generated, original)]
        pairs.append((np.corrcoef(a.ravel(), b.ravel())[0, 1],
                      structural_similarity(rgb2gray(a), rgb2gray(b), data_range=1., gaussian_weights=True,
                                            sigma=1.5, use_sample_covariance=False)))
    values["pixcorr256"], values["ssim256"] = np.asarray(pairs).T
    require(all(np.isfinite(v).all() for v in values.values()), "Non-finite score")
    return matrix, values


def summary_from_payload(path, binding):
    with np.load(path, allow_pickle=False) as obj:
        saved = json.loads(str(obj["metadata_json"].item()))
        require(saved["binding"] == binding, f"Existing DINO payload belongs to different inputs: {path}")
        means = {k: float(np.mean(obj[k])) for k in saved["per_row_metrics"]}
        require(all(np.isfinite(list(means.values()))), "Nonfinite DINO means")
        return {**saved, "mean": means, "dino_npz_sha256": sha(path)}


def publish(src, dst):
    # Temporary and final files share a filesystem. An exclusive hard link
    # lets concurrent writers publish complete immutable files without a lock.
    try:
        os.link(src, dst)
    except FileExistsError:
        pass


def score_folder(folder, panel, ref, score_bytes, score, encode, args):
    tiles, png_hashes, inputs, identity, ordered_sha = query_input(folder, panel, ref, score_bytes, score)
    binding = {"panel": panel, "n": len(ref.rows), "identity": identity, "input_sha256": inputs, **ref.source,
               "ordered_image_hashes_sha256": ordered_sha, "model_revision": REVISION, "model_files_sha256": MODEL_FILES,
               "scorer_sha256": sha(Path(__file__))}
    npz, receipt = folder / "dino.npz", folder / "dino.json"
    with tempfile.TemporaryDirectory(dir=folder, prefix=".dino-tmp-") as work:
        work = Path(work)
        if not npz.is_file():
            matrix, values = metrics(tiles, ref, encode, panel)
            meta = {"binding": binding, "panel": panel, "n": len(ref.rows), "per_row_metrics": list(values),
                    "gallery": "All original panel photos, in row order on both axes of dino_cosine.",
                    "dino": "Raw uncentered cosine of final L2-normalized float32 DINOv2-S/14 CLS; float64 dot products. "
                            "Ties receive fractional top-k credit and half pairwise credit. Rank is one-based midrank.",
                    "fresh_within10": "Same raw cosine submatrix among the ten originals with each query's concept. "
                                      "dino_within10 is top-1." if panel == "fresh" else None,
                    "pixels": "Official source JPEG and generated RGB each PIL Lanczos -> 256x256. RGB Pearson; grayscale "
                              "Gaussian SSIM, sigma=1.5, population covariance, range=1. DINO's frozen processor center-crops to 224.",
                    "device": str(args.device), "batch_size": args.batch_size, "torch": str(torch.__version__),
                    "transformers": transformers.__version__, "pillow": PIL.__version__, "numpy": np.__version__, "skimage": skimage.__version__}
            tmp = work / "dino.npz"
            np.savez_compressed(tmp, row=ref.rows, concept=ref.concepts, reference_member=np.asarray(ref.members),
                                reference_sha256=np.asarray(ref.hashes), query_sha256=np.asarray(png_hashes),
                                dino_cosine=matrix, metadata_json=np.array(json.dumps(meta, sort_keys=True)), **values)
            if receipt.is_file():
                require(json.loads(receipt.read_bytes())["dino_npz_sha256"] == sha(tmp), "Missing DINO payload differs from its saved receipt")
            require(all(sha(folder / name) == value for name, value in inputs.items()), "Primary inputs changed during scoring")
            publish(tmp, npz)
        report = summary_from_payload(npz, binding)
        if not receipt.is_file():
            tmp = work / "dino.json"
            tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            publish(tmp, receipt)
        require(json.loads(receipt.read_bytes()) == report, f"Existing DINO receipt does not match the payload in {folder}")
    print(folder, json.dumps(report["mean"], allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rad = Path(os.environ.get("RAD_ROOT", "."))
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--panel", choices=("auto", "official", "fresh"), default="auto")
    parser.add_argument("--archive", type=Path, default=Path(os.environ.get("ARCHIVE", ".")))
    parser.add_argument("--prep", type=Path, default=rad / "data/prep")
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("DINO_MODEL", rad / "models/facebook-dinov2-small")))
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda, cuda:1, or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    require(args.batch_size > 0 and args.cpu_threads > 0, "Batch size and CPU thread count must be positive")
    args.device = torch.device(args.device)
    require(args.device.type in ("cuda", "cpu") and (args.device.type == "cpu" or torch.cuda.is_available()), "Requested device unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    encode = build_encoder(args.model, args.device, args.batch_size)
    banks = {}
    for folder in dict.fromkeys(f.resolve() for f in args.folders):
        score_bytes = (folder / "scores.json").read_bytes()
        score = json.loads(score_bytes)
        panel = score.get("panel")
        require(panel in GALLERY_SHA and args.panel in ("auto", panel), f"Unexpected panel for {folder}: {panel}")
        if panel not in banks:
            banks[panel] = gallery(panel, args.archive, args.prep, encode)
        score_folder(folder, panel, banks[panel], score_bytes, score, encode, args)


if __name__ == "__main__":
    main()
