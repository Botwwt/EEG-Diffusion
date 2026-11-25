"""Recompute only Brain-HIVE's pretrained Inception-v3 identification from bound full RGB.

The two inception_official sidecars can be copied verbatim alongside the original
score receipts. read_sidecar(folder) verifies them using NumPy and returns the rows.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

METHOD = "brain-hive-inception-v3-avgpool-transform-input-512-342-strict-v1"
METRIC = "inception_2way_official"
LEGACY = "inception_2way_legacy"
STEM = "inception_official"
CHECKPOINT = "inception_v3_google-0cc3c7bd.pth"
WEIGHTS_SHA256 = "0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef"
GALLERY = {
    # SHA-256 of newline-joined row<TAB>member<TAB>concept<TAB>original JPEG SHA-256.
    "official": "31d85c5dbd8e0f158d7546d141b55b89f07a027217cd1a3752bec0fcfd57c441",
    "fresh": "23ff3bf5edb4456d0962af940f2647f6350309cb840edc90be5d8f3b7b6be89b",
    "dev": "69bf8decd3d6ae3c1605d9d4c8781bc36ee478399d58e048d37a08c43c4ca703",
}
PATTERNS = ("images/*/*/*/sub-*/scores.json", "external/atm-sub*/scores.json",
            "calibration/dev/*/scores.json", "exploratory/*/dev-r4/sub-*/scores.json")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def chain(values):
    return digest("\n".join(values).encode())


def weights_sha(path):
    value = file_sha(path)
    require(value == WEIGHTS_SHA256, f"Unexpected official Inception checkpoint SHA-256: {value}")
    return value


def read_primary(folder):
    raw = (folder / "scores.json").read_bytes()
    score = json.loads(raw)
    panel = score["panel"]
    require(panel in GALLERY and score["raw_RGB"] is True, f"Unknown source panel in {folder}")
    n = 500 if panel == "fresh" else 200
    require(score["n"] == n, f"Unexpected primary score denominator in {folder}")
    observed = (folder / "observations.npz").read_bytes()
    hashes = {"scores.json": digest(raw), "observations.npz": digest(observed)}
    require(hashes["observations.npz"] == score["observations_sha256"], f"Primary observations digest differs in {folder}")
    with np.load(io.BytesIO(observed), allow_pickle=False) as source:
        row, concept, legacy = (source[k] for k in ("row", "concept", "inception_2way"))
    require(row.shape == (n,) and row.dtype.kind in "iu" and len(np.unique(row)) == n
            and concept.shape == (n,) and legacy.shape == (n,), f"Invalid primary rows in {folder}")
    require(np.isfinite(legacy).all() and np.all((0 <= legacy) & (legacy <= 1))
            and np.isclose(legacy.mean(), score["mean"]["inception_2way"], rtol=0, atol=1e-12),
            f"Historical Inception mean differs in {folder}")
    return score, hashes, row, concept, legacy


@dataclass
class Gallery:
    rows: np.ndarray
    concepts: np.ndarray
    members: list
    hashes: list
    source: dict
    features: np.ndarray


def load_gallery(panel, prep, archive, encode):
    split_raw = (prep / "paper_split.json").read_bytes()
    split = json.loads(split_raw)
    source = {"paper_split_sha256": digest(split_raw)}
    if panel == "dev":
        raw = (prep / "paper_development.json").read_bytes()
        source["development_split_sha256"] = digest(raw)
        rows = np.asarray(json.loads(raw)["dev_rows"])
        require(set(rows).isdisjoint(split["fresh_rows"]) and set(rows).isdisjoint(split["train_rows"]),
                "Development overlaps the registered training or fresh rows")
    else:
        rows = np.asarray(split[f"{panel}_rows"])
    n = 500 if panel == "fresh" else 200
    require(rows.shape == (n,) and rows.dtype.kind in "iu" and np.all(rows >= 0) and len(np.unique(rows)) == n,
            f"Unexpected registered {panel} rows")
    if panel == "official":
        off = archive / "runs/visual_fidelity_20261125_000818/official"
        path = off / "test_images.zip"
        index = (off / "originals_manifest.csv").read_bytes()
        data = csv_rows(index)
        require(np.array_equal(np.arange(n), rows) and [int(r["eeg_row"]) for r in data] == rows.tolist(),
                "Official reference row order differs")
        members = [r["archive_member"] for r in data]
        concepts = [r["unique_id"] for r in data]
        expected = [r["image_sha256"] for r in data]
    else:
        path = archive / "assets/things_images_v254/training_images.zip"
        index = (prep / "train_ids.json").read_bytes()
        data = json.loads(index)
        require(len(data["ids"]) == len(data["paths"]) == 16540 and max(rows) < 16540, "Invalid training image index")
        members = [data["paths"][i] for i in rows]
        require([PurePosixPath(m).stem for m in members] == [data["ids"][i] for i in rows], "Training image IDs differ")
        concepts = [PurePosixPath(m).parent.name.split("_", 1)[1] for m in members]
        expected = None
    require(len(set(members)) == n and set(Counter(concepts).values()) == {1 if panel == "official" else 10},
            f"Unexpected {panel} gallery multiplicity")
    with zipfile.ZipFile(path) as z:
        pictures = [z.read(m) for m in members]
    hashes = [digest(raw) for raw in pictures]
    bound = chain(f"{r}\t{m}\t{c}\t{h}" for r, m, c, h in zip(rows, members, concepts, hashes, strict=True))
    require(bound == GALLERY[panel] and (expected is None or hashes == expected),
            f"Registered {panel} JPEG identities, labels or bytes differ")
    source.update(gallery_sha256=bound, reference_index_sha256=digest(index))
    return Gallery(rows, np.asarray(concepts), members, hashes, source, encode(pictures))


def query(folder, primary, gallery):
    score, input_sha, rows, concepts, _ = primary
    panel, n = score["panel"], len(rows)
    require(np.array_equal(rows, gallery.rows) and np.array_equal(concepts, gallery.concepts),
            f"Primary rows or concepts differ from registered {panel} gallery in {folder}")
    declared, names = [], []
    for name, key in (("render.json", "render_sha256"), ("manifest.csv", "source_manifest_sha256")):
        path = folder / name
        if not path.is_file():
            require(key not in score, f"Missing bound {name} in {folder}")
            continue
        raw = path.read_bytes()
        input_sha[name] = digest(raw)
        require(score.get(key) == input_sha[name], f"Primary scorer did not bind this {name} in {folder}")
        if name == "render.json":
            render = json.loads(raw)
            require(render["row"] == rows.tolist() and render.get("n", n) == n, f"Render rows differ in {folder}")
            if "dev_sha256" in render:  # The soft-text development renderer has its own receipt schema.
                require(panel == "dev" and render["dev_sha256"] == gallery.source["development_split_sha256"]
                        and folder.parent.name == "dev-r4", f"Exploratory development split differs in {folder}")
            elif "budget" in render:  # EEG reconstruction arms, including development.
                require(render["budget"].split("-")[0] == panel and render["split_sha256"] == gallery.source["paper_split_sha256"],
                        f"EEG render budget or split differs in {folder}")
                if panel == "dev":
                    require(render.get("development_split_sha256", gallery.source["development_split_sha256"])
                            == gallery.source["development_split_sha256"], "Development split changed")
            else:  # Generator-only upper bound and training-mean calibration.
                require(panel == render["panel"] == "dev" and render["eeg"] is False
                        and render["split_sha256"] == gallery.source["development_split_sha256"], f"Calibration split differs in {folder}")
            declared.append(render["pixels_sha256"])
        else:
            external = csv_rows(raw)
            require(panel == "official" and [int(r["eeg_row"]) for r in external] == rows.tolist()
                    and [r["unique_id"] for r in external] == concepts.tolist()
                    and [r["reference_member"] for r in external] == gallery.members
                    and [r["reference_sha256"] for r in external] == gallery.hashes,
                    f"External reconstruction/source alignment differs in {folder}")
            expected_subject = re.fullmatch(r"atm-sub(\d{2})", folder.name)
            require(expected_subject and {r["subject"] for r in external} == {f"sub-{expected_subject[1]}"}
                    and {r["draw"] for r in external} == {"0"}, f"External subject or draw differs in {folder}")
            declared.append([r["reconstruction_sha256"] for r in external])
            names = [r["filename"] for r in external]
            require(len(set(names)) == n and all(Path(s).name == s and "\\" not in s and "/" not in s for s in names),
                    "Invalid external image filenames")
    require(declared and all(len(v) == n and v == declared[0] for v in declared), f"Missing or inconsistent RGB hash lists in {folder}")
    ordered = chain(declared[0])
    require(score.get("ordered_image_hashes_sha256") == ordered, f"Primary scored RGB sequence differs in {folder}")
    patterns = [fmt for fmt in ("{:05d}.png", "{:03d}.jpg", "{:03d}.png") if (folder / fmt.format(0)).is_file()]
    require(len(patterns) <= 1, f"Ambiguous reconstruction filenames in {folder}")
    if patterns:
        names = [patterns[0].format(i) for i in range(n)]
    require(len(names) == n, f"Missing full-resolution RGB files in {folder}")
    pictures = []
    for i, (name, h) in enumerate(zip(names, declared[0], strict=True)):
        raw = (folder / name).read_bytes()
        require(digest(raw) == h, f"Reconstruction bytes differ at row {rows[i]} in {folder}")
        pictures.append(raw)
    return pictures, ordered


def strict_counts(real, predicted):
    """Brain-HIVE uses column-query NumPy corrcoef followed by strict less-than."""
    require(real.ndim == 2 and real.shape == predicted.shape and len(real) > 1, "Invalid feature matrices")
    n = len(real)
    require(np.isfinite(real).all() and np.isfinite(predicted).all() and np.all(np.ptp(real, axis=1) > 0)
            and np.all(np.ptp(predicted, axis=1) > 0), "Nonfinite or constant Inception features")
    r = np.corrcoef(real, predicted)[:n, n:]
    require(np.isfinite(r).all(), "Invalid Pearson matrix")
    correct = np.diag(r)
    return (r < correct).sum(0).astype(np.int32), ((r == correct).sum(0) - 1).astype(np.int32)


def offline_network(state, transform_input=True):
    from torchvision.models import inception_v3

    model = inception_v3(weights=None, init_weights=False, transform_input=transform_input)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def make_encoder(state, device, batch_size):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.models.feature_extraction import create_feature_extractor

    net = create_feature_extractor(offline_network(state), return_nodes=["avgpool"]).eval().to(device)
    bilinear = T.InterpolationMode.BILINEAR
    prep = T.Compose([T.Resize(512, interpolation=bilinear), T.CenterCrop(512), T.ToTensor()])
    features = T.Compose([T.Resize(342, interpolation=bilinear, antialias=True),
                          T.Normalize([.485, .456, .406], [.229, .224, .225])])

    @torch.inference_mode()
    def encode(pictures):
        parts = []
        for start in range(0, len(pictures), batch_size):
            tiles = []
            for raw in pictures[start:start + batch_size]:
                with Image.open(io.BytesIO(raw)) as im:
                    tiles.append(prep(im.convert("RGB")))
            # eval_images first transfers the 512 tensors to the scoring device.
            # two_way_identification then applies Resize(342) and Normalize image by image.
            batch = torch.stack([features(x) for x in torch.stack(tiles).to(device)])
            parts.append(net(batch)["avgpool"].float().flatten(1).cpu().numpy())
        matrix = np.concatenate(parts)
        require(matrix.shape == (len(pictures), 2048) and np.isfinite(matrix).all(), "Invalid Inception avgpool output")
        return matrix

    return encode


def verify_graph(state, device, weight_id):
    """Compare both FX graphs and the offline FX output against the real factory's eager avgpool."""
    from unittest.mock import patch

    import torch
    import torchvision
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.feature_extraction import create_feature_extractor

    # Only the factory's download is substituted. The installed torchvision factory
    # still sets its own kwargs, loads the whole state and constructs the eager model.
    with patch.object(Inception_V3_Weights, "get_state_dict", return_value=state):
        author = inception_v3(weights=Inception_V3_Weights.DEFAULT).eval().to(device)
    require(author.transform_input is True, "Installed pretrained factory did not enable transform_input")
    reference = create_feature_extractor(author, return_nodes=["avgpool"]).eval()
    fixed = create_feature_extractor(offline_network(state).to(device), return_nodes=["avgpool"]).eval()
    old = create_feature_extractor(offline_network(state, False).to(device), return_nodes=["avgpool"]).eval()
    require(str(fixed.graph) == str(reference.graph) and str(old.graph) != str(fixed.graph), "Unexpected traced FX graphs")
    x = torch.rand((2, 3, 342, 342), generator=torch.Generator().manual_seed(719)).to(device)
    mean, sd = x.new_tensor([.485, .456, .406])[None, :, None, None], x.new_tensor([.229, .224, .225])[None, :, None, None]
    x = (x - mean) / sd
    eager = []
    with torch.inference_mode():
        hook = author.avgpool.register_forward_hook(lambda module, args, output: eager.append(output.detach().clone()))
        try:
            author(x)
        finally:
            hook.remove()
        official = reference(x)["avgpool"]
        corrected = fixed(x)["avgpool"]
        legacy = old(x)["avgpool"]
    torch.testing.assert_close(official, eager[0])
    torch.testing.assert_close(corrected, official)
    baseline = official.abs().max().item()
    legacy_delta = (legacy - official).abs().max().item()
    require(baseline > 0 and legacy_delta / baseline > 1e-5, "Legacy FX unexpectedly agrees with official on nonuniform inputs")
    return dict(weight_source=weight_id, device=str(device), torch=str(torch.__version__), torchvision=torchvision.__version__,
                identical_corrected_factory_graph=True, images=2, shape=list(corrected.shape),
                eager_factory_fx_max_abs=(official - eager[0]).abs().max().item(),
                corrected_factory_fx_max_abs=(corrected - official).abs().max().item(),
                legacy_factory_fx_max_abs=legacy_delta, legacy_factory_max_relative_to_peak=legacy_delta / baseline)


def read_payload(path, primary):
    score, hashes, rows, concepts, historical = primary
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        data = {k: z[k] for k in ("row", "concept", METRIC, LEGACY, "wins", "ties")}
    n = len(rows)
    require(meta["method"] == METHOD and meta["panel"] == score["panel"] and meta["n"] == n and meta["denominator"] == n - 1
            and meta["gallery_sha256"] == GALLERY[score["panel"]] and meta["weights_sha256"] == WEIGHTS_SHA256,
            f"Unexpected Inception sidecar protocol in {path}")
    required = {"scores.json", "observations.npz"}
    required.update(name for name, key in (("render.json", "render_sha256"), ("manifest.csv", "source_manifest_sha256")) if key in score)
    require(set(meta["source_sha256"]) == required and len(required) > 2, f"Missing primary source binding in {path}")
    for name in required:
        expected = hashes[name] if name in hashes else file_sha(path.parent / name)
        require(meta["source_sha256"][name] == expected, f"Primary source {name} differs for {path}")
    for name, key in (("render.json", "render_sha256"), ("manifest.csv", "source_manifest_sha256")):
        if key in score:
            require(meta["source_sha256"][name] == score[key], f"RGB metadata binding differs for {path}")
    require(meta["ordered_image_hashes_sha256"] == score["ordered_image_hashes_sha256"], f"RGB order changed in {path}")
    require(np.array_equal(data["row"], rows) and np.array_equal(data["concept"], concepts)
            and np.array_equal(data[LEGACY], historical), f"Historical per-row binding differs for {path}")
    for k in ("wins", "ties"):
        value = data[k]
        require(value.shape == (n,) and value.dtype.kind in "iu" and np.all((0 <= value) & (value < n)), f"Invalid {k}")
    require(np.all(data["wins"] + data["ties"] <= n - 1) and np.array_equal(data[METRIC], data["wins"] / (n - 1)),
            f"Invalid strict Inception counts in {path}")
    summary = {**meta, "mean": {key: float(data[key].mean()) for key in (METRIC, LEGACY)}, "observations_sha256": file_sha(path)}
    return summary, data


def read_sidecar(folder):
    folder = Path(folder)
    summary, data = read_payload(folder / f"{STEM}.npz", read_primary(folder))
    require(json.loads((folder / f"{STEM}.json").read_bytes()) == summary, f"Mismatched Inception receipt in {folder}")
    return summary, data


def link_new(src, dst):
    # Publishing an already-complete file with a hard link also works on shared Linux filesystems.
    try:
        os.link(src, dst)
    except FileExistsError:
        pass


def score_folder(folder, ref, primary, encode, runtime):
    score, hashes, rows, concepts, legacy = primary
    pictures, ordered = query(folder, primary, ref)
    meta = dict(method=METHOD, panel=score["panel"], n=len(rows), denominator=len(rows) - 1,
                source_sha256=hashes, ordered_image_hashes_sha256=ordered, **ref.source, **runtime)
    payload, receipt = (folder / f"{STEM}.{ext}" for ext in ("npz", "json"))
    with tempfile.TemporaryDirectory(dir=folder, prefix=".inception-") as td:
        if not payload.is_file():
            wins, ties = strict_counts(ref.features, encode(pictures))
            tmp = Path(td) / payload.name
            np.savez_compressed(tmp, row=rows, concept=concepts, wins=wins, ties=ties, metadata_json=np.array(json.dumps(meta, sort_keys=True)),
                                **{METRIC: wins / (len(rows) - 1), LEGACY: legacy})
            require(all(file_sha(folder / k) == v for k, v in hashes.items()), f"Primary receipts changed during scoring in {folder}")
            link_new(tmp, payload)
        result, _ = read_payload(payload, primary)
        for key in ("source_sha256", "gallery_sha256", "weights_sha256", "scorer_sha256", "method"):
            require(result[key] == meta[key], f"Existing Inception {key} differs in {folder}")
        if not receipt.is_file():
            tmp = Path(td) / receipt.name
            tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            link_new(tmp, receipt)
        require(json.loads(receipt.read_bytes()) == result, f"Inception receipt changed in {folder}")
    return result


def select_folders(root, explicit, shard_index, shard_count):
    require(shard_count > 0 and 0 <= shard_index < shard_count, "Invalid zero-based shard index/count")
    folders = {(root / f).resolve() for f in explicit} if explicit else {p.parent.resolve() for pat in PATTERNS for p in root.glob(pat)}
    chosen = []
    for folder in sorted(folders):
        rel = folder.relative_to(root).as_posix()
        require(not rel.startswith("side/"), "These historical side panels lack an original per-row RGB binding")
        if int(digest(rel.encode())[:16], 16) % shard_count == shard_index:
            chosen.append(folder)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="*", type=Path, help="Explicit primary score directories; otherwise discover all bound panels.")
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("RAD_ROOT", ".")))
    ap.add_argument("--archive", type=Path, default=Path(os.environ.get("ARCHIVE", ".")))
    ap.add_argument("--prep", type=Path)
    ap.add_argument("--weights", type=Path, help="Official torchvision checkpoint; defaults to ARCHIVE/assets/evaluation_v272.")
    ap.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List discovered folders for this shard without loading PyTorch.")
    mode.add_argument("--check", action="store_true", help="Verify all selected numeric sidecars and primary receipts without RGB or PyTorch.")
    mode.add_argument("--verify-graph", action="store_true", help="Compare the real pretrained factory's eager and FX avgpool with offline and legacy graphs.")
    ap.add_argument("--synthetic-model", action="store_true", help="Only with --verify-graph. Use a fixed random state for an offline graph regression.")
    a = ap.parse_args()
    require(not a.synthetic_model or a.verify_graph, "Synthetic weights are allowed only in the graph self-test")
    root = a.root.resolve()
    folders = select_folders(root, a.folders, a.shard_index, a.shard_count)
    if a.list:
        for folder in folders:
            print(folder.relative_to(root).as_posix())
        return
    if a.check:
        for folder in folders:
            summary, _ = read_sidecar(folder)
            print(folder.relative_to(root).as_posix(), summary["n"], summary["mean"], flush=True)
        print(f"Verified {len(folders)} sidecars.")
        return
    if not a.verify_graph and not folders:
        print("No complete primary score directories selected.")
        return
    import PIL
    import torch
    import torchvision

    dev = torch.device(a.device)
    require(dev.type in {"cpu", "cuda"} and (dev.type == "cpu" or torch.cuda.is_available()), f"Unavailable device {dev}")
    require(a.cpu_threads > 0 and a.batch_size > 0, "Threads and batch size must be positive")
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if a.synthetic_model:
        torch.manual_seed(103)
        state = torchvision.models.inception_v3(weights=None, init_weights=False).state_dict()
        weight_id = "synthetic-seed-103"
    else:
        path = a.weights or a.archive / "assets/evaluation_v272" / CHECKPOINT
        weight_id = weights_sha(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
    if a.verify_graph:
        print(json.dumps(verify_graph(state, dev, weight_id), indent=2), flush=True)
        return
    encode = make_encoder(state, dev, a.batch_size)
    del state
    runtime = dict(weights_sha256=weight_id, scorer_sha256=file_sha(Path(__file__)), device=str(dev), batch_size=a.batch_size,
                   torch=str(torch.__version__), torchvision=torchvision.__version__, pillow=PIL.__version__, numpy=np.__version__)
    banks = {}
    for folder in folders:
        primary = read_primary(folder)
        panel = primary[0]["panel"]
        if panel not in banks:
            banks[panel] = load_gallery(panel, a.prep or root / "data/prep", a.archive, encode)
        result = score_folder(folder, banks[panel], primary, encode, runtime)
        print(folder.relative_to(root).as_posix(), result["n"], result["mean"], flush=True)


if __name__ == "__main__":
    main()
