"""Score frozen generated JPEGs with the published DINOv2-S/14 final CLS token, without fitting."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from fit_within_photo_embedding import uncertainty
from PIL import Image
from prepare_registered_subjects import ROWS_SHA
from render_official_words import crop512
from score_object_panel import score_object_panel
from score_support_structure import ZIP_SHA

ARMS = ("own", "uniform", "wrong")
PANELS = ("registered", "replicate")
REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
FILES = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")


def encoder(args):
    import torch
    from transformers import AutoImageProcessor, Dinov2Model

    if any(sha(args.model / name) != value for name, value in FILES.items()):
        raise ValueError(
            "DINOv2-S/14 model or preprocessing differs from the fixed official revision"
        )
    processor = AutoImageProcessor.from_pretrained(
        str(args.model), use_fast=False, local_files_only=True
    )
    model = (
        Dinov2Model.from_pretrained(str(args.model), local_files_only=True)
        .eval()
        .requires_grad_(False)
        .to(args.device)
    )
    if (model.config.patch_size, model.config.hidden_size, model.config.num_hidden_layers) != (
        14,
        384,
        12,
    ):
        raise ValueError("Expected DINOv2-S/14 without a task-specific head")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def encode(images):
        blocks = []
        with torch.inference_mode():
            for lo in range(0, len(images), args.batch_size):
                pixels = processor(images=images[lo : lo + args.batch_size], return_tensors="pt")[
                    "pixel_values"
                ]
                if tuple(pixels.shape[1:]) != (3, 224, 224):
                    raise ValueError("Unexpected DINO image preprocessing")
                cls = (
                    model(pixel_values=pixels.to(args.device))
                    .last_hidden_state[:, 0]
                    .float()
                    .cpu()
                    .numpy()
                )
                norms = np.linalg.norm(cls, axis=1, keepdims=True)
                if cls.shape[1] != 384 or not np.isfinite(norms).all() or np.any(norms <= 0):
                    raise ValueError("Invalid final DINO CLS feature")
                blocks.append(cls / norms)
        return np.concatenate(blocks)

    return encode


def reference(args):
    if sha(args.originals) != ZIP_SHA or sha(args.manifest) != ROWS_SHA:
        raise ValueError(
            "Original THINGS images or published row mapping differ from the frozen release"
        )
    with args.manifest.open(encoding="utf8", newline="") as f:
        table = list(csv.DictReader(f))
    if [int(rec["train_row"]) for rec in table] != list(range(16540)):
        raise ValueError("Reference table is not in original EEG order")
    return table


def originals(args, encode, table, z, panel):
    reg_path = args.references / (panel + "_ablation.json")
    reg = read(reg_path)
    rows = np.asarray(reg["rows"])
    truth = np.asarray(reg["words"])[reg["truth"]]
    folds = np.asarray(reg["folds"])
    if (
        len(set(rows)) != 120
        or len(set(truth)) != 12
        or any(np.sum((folds == f) & (truth == t)) != 2 for f in range(5) for t in set(truth))
    ):
        raise ValueError("Incomplete original twelve-object panel")
    images, checks = [], []
    for row, label in zip(rows, truth, strict=True):
        r = table[row]
        if r["unique_id"] != label:
            raise ValueError("Truth and original photo manifest disagree")
        member = f"training_images/{r['official_folder']}/{r['stimulus_stem']}.jpg"
        raw = z.read(member)
        checks.append(hashlib.sha256(raw).hexdigest())
        with Image.open(io.BytesIO(raw)) as im:
            images.append(crop512(im.convert("RGB")))
    cache = args.out / "original_cls" / (panel + ".npz")
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as c:
            if (
                str(c["registration_sha"]) == sha(reg_path)
                and str(c["model_sha"]) == FILES["model.safetensors"]
                and str(c["processor_sha"]) == FILES["preprocessor_config.json"]
                and np.array_equal(c["rows"], rows)
                and np.array_equal(c["jpeg_sha256"], checks)
                and str(c["device"]) == args.device
                and str(c["script_sha"]) == sha(Path(__file__))
                and int(c["batch_size"]) == args.batch_size
                and hashlib.sha256(c["cls"].tobytes()).hexdigest() == str(c["feature_sha"])
            ):
                return reg, c["cls"], checks
    features = encode(images)
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        cache,
        rows=rows,
        cls=features,
        jpeg_sha256=checks,
        registration_sha=sha(reg_path),
        device=args.device,
        batch_size=args.batch_size,
        script_sha=sha(Path(__file__)),
        feature_sha=hashlib.sha256(features.tobytes()).hexdigest(),
        model_sha=FILES["model.safetensors"],
        processor_sha=FILES["preprocessor_config.json"],
    )
    return reg, features, checks


def credit(x):
    return float((x > 1e-12) + 0.5 * (abs(x) <= 1e-12))


def summarize(entries, field):
    classes = [r["panel"] + "/" + r["object"] for r in entries]
    output = {}
    for arm in ARMS:
        v = np.array([r[field][arm] for r in entries])
        s = uncertainty(v, classes)
        output[arm] = dict(
            correct_credit=float(v.sum()), n=len(v), rate=float(v.mean()), rate_ci95=s["ci95"]
        )
    for control in ("uniform", "wrong"):
        d = uncertainty([r[field]["own"] - r[field][control] for r in entries], classes)
        output["own_minus_" + control] = dict(
            delta_rate=d["mean"],
            ci95=d["ci95"],
            class_signflip_one_sided_p=d["cluster_signflip_one_sided_p"],
        )
    return output


def score_case(folder, encode, ref, condition_root):
    reg, bank, original_hashes = ref
    s = read(folder / "settings.json")
    panel, subject = s["panel"], s["subject"]
    rows, folds, words, donors = [
        np.asarray(s[k]) for k in ("rows", "folds", "predicted_words", "donor_rows")
    ]
    path = condition_root / subject / panel / "conditions_512.npz"
    if sha(path) != s["packet_sha256"]:
        raise ValueError(f"Frozen condition checksum differs at {folder}")
    with np.load(path, allow_pickle=False) as packet:
        for k, v in (
            ("rows", rows),
            ("folds", folds),
            ("predicted_words", words),
            ("donor_rows", donors),
        ):
            if not np.array_equal(packet[k], v):
                raise ValueError("Render manifest differs from frozen packet")
    if (
        reg["rows"] != rows.tolist()
        or reg["folds"] != folds.tolist()
        or not np.array_equal(s["seeds"], 923761 + rows)
        or folder.name != subject
        or folder.parent.name != panel
        or set(rows) != set(donors)
    ):
        raise ValueError(f"Render and evaluation registration differ at {folder}")
    truth = np.asarray(reg["words"])[reg["truth"]]
    lookup = {r: i for i, r in enumerate(rows)}
    donor = np.array([lookup[r] for r in donors])
    if not np.array_equal(words, words[donor]) or not np.array_equal(folds, folds[donor]):
        raise ValueError("Incorrect wrong-EEG control")
    shots = [
        dict(
            panel=panel,
            row=int(r),
            object=str(t),
            word=str(w),
            fold=int(f),
            intra10={},
            object12={},
        )
        for r, t, w, f in zip(rows, truth, words, folds, strict=True)
    ]
    pair = {}
    for arm in ARMS:
        images = []
        expected = s["jpeg_sha256"][arm]
        if len(expected) != 120:
            raise ValueError("Missing generated-image digests")
        for i, digest in enumerate(expected):
            name = folder / "images" / arm / f"{i:03d}.jpg"
            raw = name.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"Generated JPEG changed: {name}")
            with Image.open(io.BytesIO(raw)) as im:
                if im.format != "JPEG" or im.mode != "RGB" or im.size != (512, 512):
                    raise ValueError("Unexpected generated image format or size")
                images.append(im.copy())
        generated = encode(images)
        cos = generated.astype(np.float64) @ bank.astype(np.float64).T
        gallery = score_object_panel(cos, rows, truth)["per_photo"]
        for rec, hit in zip(shots, gallery, strict=True):
            rec["intra10"][arm] = hit["intra10_top1"]
            rec["object12"][arm] = hit["object_top1"]
        for f in range(5):
            for word in np.unique(truth):
                i, j = np.flatnonzero((folds == f) & (truth == word))
                if words[i] != word or words[j] != word:
                    continue
                key = (f, str(word))
                record = pair.setdefault(
                    key,
                    dict(
                        panel=panel,
                        object=str(word),
                        fold=f,
                        rows=[int(rows[i]), int(rows[j])],
                        assignment={},
                        independent_two={},
                        margin={},
                    ),
                )
                m1 = cos[i, i] - cos[i, j]
                m2 = cos[j, j] - cos[j, i]
                record["margin"][arm] = float((m1 + m2) / 2)
                record["assignment"][arm] = credit((m1 + m2) / 2)
                record["independent_two"][arm] = (credit(m1) + credit(m2)) / 2
    return (
        shots,
        list(pair.values()),
        dict(
            settings_sha256=sha(folder / "settings.json"),
            condition_sha256=sha(path),
            original_jpeg_sha256_in_row_order=hashlib.sha256(
                "".join(original_hashes).encode()
            ).hexdigest(),
        ),
    )


def main():
    import torch
    import transformers

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument(
        "--originals", type=Path, default=Path("outputs/photo-structure/training_images.zip")
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    )
    p.add_argument("--references", type=Path, default=Path("outputs/object-fidelity"))
    p.add_argument("--conditions", type=Path, default=Path("outputs/photo-structure/within_photo"))
    p.add_argument(
        "--renders",
        nargs="+",
        type=Path,
        default=[Path("outputs/photo-structure/h14_residual_render/renders")],
    )
    p.add_argument("--out", type=Path, default=Path("outputs/photo-structure/dino/scoring"))
    p.add_argument("--subjects", nargs="+", default=["sub-04", "sub-05"])
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--references-only", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("Positive batch size required")
    table = reference(args)
    encode = encoder(args)
    with zipfile.ZipFile(args.originals) as z:
        refs = {panel: originals(args, encode, table, z, panel) for panel in PANELS}
    if args.references_only:
        print(
            "Verified original galleries", [(panel, data[1].shape) for panel, data in refs.items()]
        )
        return
    summary = {}
    for subject in args.subjects:
        all_images, all_pairs, panels = [], [], {}
        for panel in PANELS:
            paths = [
                root / panel / subject
                for root in args.renders
                if (root / panel / subject / "settings.json").is_file()
            ]
            if len(paths) != 1:
                raise ValueError(
                    f"Expected exactly one frozen render manifest for {panel}/{subject}"
                )
            images, pairs, inputs = score_case(paths[0], encode, refs[panel], args.conditions)
            record = dict(
                case=inputs,
                n_queries=len(images),
                n_strict_pairs=len(pairs),
                intra10=summarize(images, "intra10"),
                object12=summarize(images, "object12"),
                strict_pair=summarize(pairs, "assignment"),
                strict_independent_two=summarize(pairs, "independent_two"),
            )
            save(
                args.out / panel / subject / "per_query.json",
                dict(images=images, pairs=pairs, **inputs),
            )
            panels[panel] = record
            all_images.extend(images)
            all_pairs.extend(pairs)
            print(
                subject,
                panel,
                len(pairs),
                "pairs; own full-panel intra10",
                record["intra10"]["own"],
                flush=True,
            )
        summary[subject] = dict(
            panels=panels,
            combined=dict(
                strict_pair=summarize(all_pairs, "assignment"),
                intra10=summarize(all_images, "intra10"),
                object12=summarize(all_images, "object12"),
                strict_independent_two=summarize(all_pairs, "independent_two"),
            ),
        )
    save(
        args.out / "summary.json",
        dict(
            model="facebook/dinov2-small",
            revision=REV,
            model_files_sha256=FILES,
            torch=torch.__version__,
            transformers=transformers.__version__,
            device=args.device,
            batch_size=args.batch_size,
            original_zip_sha256=ZIP_SHA,
            row_manifest_sha256=ROWS_SHA,
            script_sha256=sha(Path(__file__)),
            features="Final layer normalized CLS token, unit-normalized, raw cosine; no layers, seeds or parameters fitted. Apply registered 512 center crop to originals and the official 256-resize/224-center-crop processor to all images.",
            endpoints="All rows: single generated image versus ten true-class original photos (chance .1); object12 uses gallery centroids excluding the query ordinal. Strict pairs: two held-out originals share truth and fold, both predicted words correct. One-to-one assignment and independent two-option comparison. Uniform and wrong still use distinct row-specific render seeds.",
            uncertainty="Percentile intervals over 10,000 concept-cluster bootstrap samples; descriptive within each fixed participant, rates in fractions.",
            subjects=summary,
        ),
    )


if __name__ == "__main__":
    main()
