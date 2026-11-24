"""Full-vocabulary text readout for published sub-03 R80 ATM EEG embeddings.

python official_object_readout.py --out outputs/official-object-readout/sub-03
Stages can run separately with --stage prepare|predict|evaluate. Predict opens
only the full text dictionary and the EEG .npy. Standard library and NumPy.
"""

import argparse
import csv
import hashlib
import json
import os
import pickletools
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

ATM = "https://huggingface.co/datasets/LidongYang/EEG_Image_decode/resolve/dbe34bb2407164f70883c661c76664f0d596e522/"
MIRROR = "https://huggingface.co/datasets/gasparyanartur/things-eeg2/resolve/0e2ad986a26d90b6192cd69daeac32ff069c2460/"
THINGS = "https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/57e91c9754dfea8579ffc58c0ae42d9dc60080af/data/files/things_concepts.tsv"
ASSETS = {
    "concepts.tsv": (THINGS, "f6eb4ec6fcad61e50a746bee820f98d1595fe655e9e210a9a407e9a17fd8cee4"),
    "metadata.npy": (
        MIRROR + "image_metadata.npy",
        "681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c",
    ),
    "text_train.pt": (
        ATM + "ViT-H-14_features_train.pt",
        "b8156300f59d058600fc9a2c9e1586c5fbd4a113f236031acc64c6511d56b432",
    ),
    "text_test.pt": (
        ATM + "ViT-H-14_features_test.pt",
        "2d772e733ae9afe01f60547c38f0026a094347aa56fefd77cf375cf53667cdc9",
    ),
    "eeg.pt": (
        ATM + "emb_eeg/ATM_S_eeg_features_sub-03_test.pt",
        "b2d1ba91a2c8d4a630e1700bb84dfd5f3d6ee260541d43a64a3de5e89e26a1ab",
    ),
}
CATEGORY = "Top-down Category (manual selection)"
ORIGINALS = (
    "https://osf.io/download/znu7b/",
    "57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a",
)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def acquire(path, source):
    url, digest = source
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", dir=path.parent, delete=False) as f:
            temp = Path(f.name)
            try:
                with urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, f, 1 << 20)
            except BaseException:
                f.close()
                temp.unlink(missing_ok=True)
                raise
        if sha(temp) != digest:
            temp.unlink()
            raise ValueError(f"Checksum mismatch for {path.name}")
        os.replace(temp, path)
    if sha(path) != digest:
        raise ValueError(f"Checksum mismatch for {path.name}")
    return path


def unit(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Nonfinite or empty embedding")
    return x / norms


def tensor_storage(path, rows, text):
    # These SHA-pinned Torch archives use contiguous float32 row-major storage.
    # Decode the declared first tensor only. The second tensor is never opened.
    with ZipFile(path) as z:
        (meta,) = [m for m in z.namelist() if m.endswith("/data.pkl")]
        ops = [(op.name, v) for op, v, _ in pickletools.genops(z.read(meta))]
        words = [v for op, v in ops if op == "BINUNICODE"]
        if text and not (
            words[0] == "text_features" and words.index("0") < words.index("img_features")
        ):
            raise ValueError("First storage must describe the published text embeddings")
        integral = [v for op, v in ops if op in {"BININT", "BININT1", "BININT2"}]
        if ("GLOBAL", "torch FloatStorage") not in ops or integral[:6] != [
            rows * 1024,
            0,
            rows,
            1024,
            1024,
            1,
        ]:
            raise ValueError("Unexpected tensor dtype, shape, offset or strides")
        arr = np.frombuffer(z.read(meta.removesuffix("data.pkl") + "data/0"), dtype="<f4")
    return arr.reshape(rows, 1024)


def to_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def object_ids(names):
    return [s.split("_", 1)[1] for s in names]


def prepare(out):
    cache = out / "assets"
    paths = {k: acquire(cache / k, v) for k, v in ASSETS.items()}
    # The original release uses an object-format .npy. Only read it after
    # verifying the SHA independently published in the source OSF image set.
    meta = np.load(paths["metadata.npy"], allow_pickle=True).item()
    with paths["concepts.tsv"].open(encoding="utf8", newline="") as f:
        table = list(csv.DictReader(f, delimiter="\t"))
    unique_ids = [r["uniqueID"] for r in table]
    if len(unique_ids) != len(set(unique_ids)) or len(unique_ids) != 1854:
        raise ValueError("Expected the original complete THINGS vocabulary")
    global_ids = {}
    by_id = {}
    for part, count in (("train", 1654), ("test", 200)):
        names = sorted(set(meta[part + "_img_concepts"]))
        if len(names) != count or list(dict.fromkeys(meta[part + "_img_concepts"])) != names:
            raise ValueError("Image set ordering differs from the author text cache")
        ids = object_ids(names)
        values = tensor_storage(paths["text_" + part + ".pt"], count, text=True)
        for name, vector in zip(ids, values, strict=True):
            if name in by_id:
                raise ValueError(f"Ambiguous stimulus folder name {name}")
            by_id[name] = vector
        for name in set(meta[part + "_img_concepts_THINGS"]):
            num, word = name.split("_", 1)
            global_ids[word] = int(num)
    if (
        set(by_id) != set(unique_ids)
        or set(global_ids) != set(unique_ids)
        or set(global_ids.values()) != set(range(1, 1855))
    ):
        raise ValueError("Text cache is missing dictionary entries")
    embeds = unit(np.stack([by_id[k] for k in unique_ids]))
    groups = [r[CATEGORY] for r in table]
    words = [r["Word"] for r in table]
    np.savez(
        out / "dictionary.npz",
        unique_ids=unique_ids,
        words=words,
        things_ids=[global_ids[k] for k in unique_ids],
        text_embeddings=embeds,
        manual_categories=groups,
        model="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        prompt="This picture is {uniqueID}",
    )
    to_csv(
        out / "dictionary.csv",
        ["things_id", "unique_id", "word", "manual_categories"],
        (
            dict(things_id=global_ids[k], unique_id=k, word=words[i], manual_categories=groups[i])
            for i, k in enumerate(unique_ids)
        ),
    )
    np.save(out / "eeg_r80.npy", tensor_storage(paths["eeg.pt"], 200, text=False))
    source = {
        "dataset": "sub-03 official THINGS-EEG2 test, authors' released mean-80 EEG embeddings",
        "text_model": "OpenCLIP ViT-H-14 laion2b_s32b_b79k",
        "prompt": "This picture is {uniqueID}",
        "text_encoder_source": "https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/Retrieval/eegdatasets_leaveone.py",
        "training_source": "https://github.com/dongyangli-del/EEG_Image_decode/blob/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/Retrieval/ATMS_retrieval.py",
        "independent_metadata_source": "https://osf.io/download/qkgtf/",
        "inputs": {k: {"url": v[0], "sha256": v[1]} for k, v in ASSETS.items()},
        "consumer_inputs": ["dictionary.npz", "eeg_r80.npy"],
        "encoder_note": "Frozen released EEG features; authors' public retrieval code monitors official test performance across epochs. Their exact export epoch was not independently reconstructed.",
    }
    (out / "provenance.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf8")


def predict(out, dictionary_path=None, eeg_path=None):
    with np.load(
        dictionary_path if dictionary_path is not None else out / "dictionary.npz",
        allow_pickle=False,
    ) as bank:
        ids, words = bank["unique_ids"], bank["words"]
        text = bank["text_embeddings"]
        categories = [
            set(x.strip() for x in s.split(",") if x.strip()) for s in bank["manual_categories"]
        ]
    eeg = unit(
        np.load(eeg_path if eeg_path is not None else out / "eeg_r80.npy", allow_pickle=False)
    )
    scores = eeg @ text.T
    order = np.argsort(-scores, axis=1, kind="stable")
    names = sorted(set().union(*categories))
    prototypes = unit(np.stack([text[[n in s for s in categories]].mean(axis=0) for n in names]))
    broad = np.asarray(names)[(eeg @ prototypes.T).argmax(axis=1)]
    np.savez(
        out / "readout.npz", ranked_ids=ids[order], broad_predictions=broad, category_names=names
    )
    to_csv(
        out / "predictions.csv",
        ["eeg_row", "predicted_id", "predicted_word", "top5_ids", "broad_category"],
        (
            dict(
                eeg_row=i,
                predicted_id=ids[r[0]],
                predicted_word=words[r[0]],
                top5_ids="|".join(ids[r[:5]]),
                broad_category=broad[i],
            )
            for i, r in enumerate(order)
        ),
    )


def evaluate_readout(out, originals, reference=None, broad_space="text"):
    reference = out if reference is None else reference
    path = acquire(reference / "assets/metadata.npy", ASSETS["metadata.npy"])
    meta = np.load(path, allow_pickle=True).item()
    truth = np.asarray(object_ids(meta["test_img_concepts"]))
    with np.load(out / "readout.npz", allow_pickle=False) as f:
        ranking, broad, names = f["ranked_ids"], f["broad_predictions"], f["category_names"]
    if ranking.shape != (200, 1854) or not np.all((ranking == truth[:, None]).sum(axis=1) == 1):
        raise ValueError("Evaluator found an incomplete prediction or label mismatch")
    ranks = (ranking == truth[:, None]).argmax(axis=1) + 1
    with np.load(reference / "dictionary.npz", allow_pickle=False) as f:
        categories = {
            k: set(x.strip() for x in v.split(",") if x.strip())
            for k, v in zip(f["unique_ids"], f["manual_categories"], strict=True)
        }
    hits = np.array([p in categories[t] for p, t in zip(broad, truth, strict=True)])
    eligible = np.array([bool(categories[t]) for t in truth])
    out_eval = out / "evaluation"
    out_eval.mkdir(exist_ok=True)
    to_csv(
        out_eval / "scored.csv",
        [
            "eeg_row",
            "truth_id",
            "truth_file",
            "predicted_id",
            "truth_rank",
            "broad_truth",
            "broad_prediction",
            "broad_hit",
        ],
        (
            dict(
                eeg_row=i,
                truth_id=t,
                truth_file=meta["test_img_files"][i],
                predicted_id=ranking[i, 0],
                truth_rank=ranks[i],
                broad_truth="|".join(sorted(categories[t])),
                broad_prediction=broad[i],
                broad_hit=int(hits[i]) if eligible[i] else "",
            )
            for i, t in enumerate(truth)
        ),
    )
    # Exact expectation from pairing every EEG query with every different row.
    perm_hits = ranking[:, 0, None] == truth[None, :]
    wrong = ~np.eye(len(truth), dtype=bool)
    broad_perm = np.array([[p in categories[t] for t in truth] for p in broad])
    summary = {
        "n_eeg": len(truth),
        "vocabulary": ranking.shape[1],
        "top1_hits": int((ranks == 1).sum()),
        "top5_hits": int((ranks <= 5).sum()),
        "top10_hits": int((ranks <= 10).sum()),
        "median_rank": float(np.median(ranks)),
        "uniform_top1_chance": 1 / ranking.shape[1],
        "wrong_eeg_pairing_top1": float(perm_hits[wrong].mean()),
        "broad_task": f"Nearest normalized {broad_space}-space centroid of concepts sharing a THINGS manually assigned category, full vocabulary",
        "broad_classes": len(names),
        "broad_eligible_n": int(eligible.sum()),
        "broad_correct": int(hits.sum()),
        "broad_fraction": float(hits[eligible].mean()),
        "broad_wrong_eeg_pairing": float(broad_perm[wrong & eligible[None, :]].mean()),
    }
    if originals:
        orig = acquire(out_eval / "test_images.zip", ORIGINALS)
        with ZipFile(orig) as z:
            files = [
                p
                for p in z.namelist()
                if p.lower().endswith(".jpg") and not Path(p).name.startswith("._")
            ]
            lookup = {(Path(p).parent.name, Path(p).name): p for p in files}
            records = []
            for i, (t, c, f) in enumerate(
                zip(truth, meta["test_img_concepts"], meta["test_img_files"], strict=True)
            ):
                member = lookup[(c, f)]
                records.append(
                    dict(
                        eeg_row=i,
                        unique_id=t,
                        archive_member=member,
                        image_sha256=hashlib.sha256(z.read(member)).hexdigest(),
                    )
                )
            if len(records) != 200 or len(lookup) != 200:
                raise ValueError("Ground truth archive differs from EEG test manifest")
            to_csv(
                out_eval / "originals_manifest.csv",
                ["eeg_row", "unique_id", "archive_member", "image_sha256"],
                records,
            )
        summary["evaluation_originals"] = {
            "file": "test_images.zip",
            "official": "https://osf.io/download/znu7b/",
            "sha256": ORIGINALS[1],
        }
    (out_eval / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "predict", "evaluate"), default="all")
    parser.add_argument(
        "--originals",
        action="store_true",
        help="Verify and index ground truth JPEGs for evaluation",
    )
    args = parser.parse_args()
    if args.stage in ("all", "prepare"):
        prepare(args.out)
    if args.stage in ("all", "predict"):
        predict(args.out)
    if args.stage in ("all", "evaluate"):
        evaluate_readout(args.out, args.originals)
