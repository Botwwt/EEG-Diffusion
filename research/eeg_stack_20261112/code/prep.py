"""One-off preparation.

  python prep.py eeg 8        # sub-08: keep the 17 O+P channels, all single trials -> data/prep/sub-08_{train,test}.npy
  python prep.py targets      # visual embeddings aligned to the EEG image order + frozen fused target F

Downloaded files are read as data only: EEG .pt through a restricted unpickler, embeddings through parquet.
"""
import glob
import sys
from pathlib import Path

import numpy as np

from common import CH17, EMB, EMB_KEYS, HIVE, PREP, PRIOR, RAW, channel_indices, safe_load_pt, write_json


def image_ids(img):
    img = np.asarray(img)
    if img.ndim == 2:
        assert (img == img[:, :1]).all(), "repeats of one row must share the image"
        img = img[:, 0]
    paths = [str(p) for p in img.reshape(-1).tolist()]
    return paths, [Path(p).stem for p in paths]


def prep_eeg(sid):
    PREP.mkdir(parents=True, exist_ok=True)
    idx = channel_indices(CH17)
    for split, n_img, n_rep in (("train", 16540, 4), ("test", 200, 80)):
        d = safe_load_pt(RAW / f"sub-{sid:02d}/{split}.pt")
        x = np.asarray(d["eeg"])
        print(split, "keys", sorted(d.keys()), "eeg", x.shape, x.dtype, flush=True)
        assert x.shape == (n_img, n_rep, 63, 250), x.shape
        paths, ids = image_ids(d["img"])
        assert len(ids) == n_img and len(set(ids)) == n_img
        x17 = np.ascontiguousarray(x[:, :, idx, :]).astype(np.float32)
        np.save(PREP / f"sub-{sid:02d}_{split}.npy", x17)
        meta = PREP / f"{split}_ids.json"
        if meta.exists():
            import json
            assert json.loads(meta.read_text())["ids"] == ids, "image order differs between subjects"
        else:
            write_json(meta, {"ids": ids, "paths": paths})
        print(f"sub-{sid:02d} {split}: saved {x17.shape} float32; per-trial std {x17.std():.4f}; "
              f"mean-over-repeats std {x17.mean(1).std():.4f}", flush=True)


def load_parquet_emb(split, key, ids):
    import pandas as pd
    files = sorted(glob.glob(str(EMB / f"things_{split}_{key}-part-*-of-*.parquet")))
    assert files, (split, key)
    df = pd.concat([pd.read_parquet(f, columns=["image_id", "emb"]) for f in files], ignore_index=True)
    print(split, key, "parquet rows", len(df), "unique ids", df["image_id"].nunique(), flush=True)
    col = df.drop_duplicates("image_id").set_index("image_id")["emb"].reindex(ids)
    assert col.notna().all(), f"missing embeddings for {int(col.isna().sum())} images"

    def flat(v):  # nested lists / object arrays -> 1-D float32 in C order (same as tensor.reshape(bsz, -1))
        a = np.asarray(v)
        if a.dtype == object:
            return np.concatenate([flat(u) for u in a])
        return a.astype(np.float32).reshape(-1)

    return np.stack([flat(v) for v in col.values])


def prep_targets():
    import json
    import torch
    sys.path.insert(0, str(HIVE))
    from main.models_adapter import FusionEncoderModel
    fe = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").eval().requires_grad_(False)
    print("fusion_encoder proj_meta:", dict(fe.config.proj_meta), flush=True)
    for split in ("train", "test"):
        ids = json.loads((PREP / f"{split}_ids.json").read_text())["ids"]
        embs = {k: load_parquet_emb(split, k, ids) for k in EMB_KEYS}
        for k, v in embs.items():
            print(split, k, v.shape, f"norm {np.linalg.norm(v, axis=1).mean():.3f}", flush=True)
        torch.save({k: torch.from_numpy(v) for k, v in embs.items()}, PREP / f"emb_{split}.pt")
        with torch.no_grad():
            t = {k: torch.from_numpy(v) for k, v in embs.items()}
            f = fe.vision_model(**t)               # z_f: what the EEG encoder is aligned to
            a = fe.adapter(f)                       # z_c = z_f + phi_c(z_f): what the IP projection saw in prior training
        np.save(PREP / f"F_{split}.npy", f.numpy().astype(np.float32))
        np.save(PREP / f"AF_{split}.npy", a.numpy().astype(np.float32))
        print(split, "F", tuple(f.shape), f"norm {f.norm(dim=1).mean():.3f}", "A(F) norm", f"{a.norm(dim=1).mean():.3f}", flush=True)
    paths = json.loads((PREP / "train_ids.json").read_text())["paths"]
    concepts = [Path(p).parent.name if Path(p).parent.name else None for p in paths]
    if any(c is None or c == "" for c in concepts):
        import re
        concepts = [re.sub(r"_\d+[a-z]?$", "", Path(p).stem) for p in paths]
    names = sorted(set(concepts), key=concepts.index)
    cid = np.array([names.index(c) for c in concepts])
    counts = np.bincount(cid)
    print("concepts:", len(names), "images per concept min/max:", counts.min(), counts.max(), flush=True)
    assert len(names) == 1654 and counts.min() == counts.max() == 10
    np.save(PREP / "train_concept.npy", cid)
    write_json(PREP / "train_concept_names.json", names)


if __name__ == "__main__":
    if sys.argv[1] == "eeg":
        for s in sys.argv[2:]:
            prep_eeg(int(s))
    elif sys.argv[1] == "targets":
        prep_targets()
