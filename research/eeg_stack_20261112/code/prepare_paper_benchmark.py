"""Prepare original RGB references, train-only visual targets, and fixed query IDs on Slurm."""
import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from common import CH17, EMB_KEYS, PREP, PRIOR, TURBO, concept_folds, write_json


def generators(archive):
    from diffusers import StableDiffusionXLPipeline
    source = archive / "assets/native_generator_v243"
    for sub in ("attn_adapter", "proj", "fusion_encoder"):
        dest = PRIOR / sub
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "diffusion_pytorch_model.safetensors"):
            origin = source / "prior_H14" / sub / name
            if sub == "fusion_encoder" and name.endswith("safetensors"):
                origin = archive / "assets/native_teacher_v239/fusion_encoder.safetensors"
            assert origin.is_file()
            if not (dest / name).exists():
                (dest / name).symlink_to(origin)
    if not (TURBO / "model_index.json").is_file():
        # The archived checkpoint uses the single-file SDXL format. Export once for the paper scripts.
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(source / "turbo/sd_xl_turbo_1.0_fp16.safetensors"), config=str(source / "turbo"),
            torch_dtype=torch.float16, local_files_only=True, add_watermarker=False)
        temp = TURBO.with_name("sdxl-turbo-exporting")
        pipe.save_pretrained(str(temp), variant="fp16", safe_serialization=True)
        temp.rename(TURBO)


def photos(archive):
    from diffusers import AutoencoderKL
    from main.models_adapter import FusionEncoderModel
    PREP.mkdir(exist_ok=True, parents=True)
    train = archive / "data/native_training_v239"
    stems = json.loads((train / "train_image_ids.json").read_text())
    assert len(set(stems)) == len(stems) == 16540
    keys = torch.load(train / "train_visual_embeddings.pt", map_location="cpu", weights_only=True)
    assert set(keys) == set(EMB_KEYS) and all(v.shape[0] == len(stems) for v in keys.values())
    emb_path = PREP / "emb_train.pt"
    if not emb_path.exists():
        emb_path.symlink_to(train / "train_visual_embeddings.pt")
    fe = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.cuda().float().eval()
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor()])
    image_file = archive / "assets/things_images_v254/training_images.zip"
    out = np.lib.format.open_memmap(PREP / "lat512_train.partial.npy", mode="w+", dtype=np.float16, shape=(16540, 4, 64, 64))
    ok = np.zeros(len(stems), dtype=bool)
    with torch.inference_mode(), zipfile.ZipFile(image_file) as z:
        files = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg")}
        assert set(stems) == set(files)
        paths = [files[s] for s in stems]
        objects = [Path(p).parent.name for p in paths]
        names = list(dict.fromkeys(objects))
        mapping = {n: i for i, n in enumerate(names)}
        cids = np.array([mapping[n] for n in objects])
        assert len(names) == 1654 and (np.bincount(cids) == 10).all()
        write_json(PREP / "train_ids.json", {"ids": stems, "paths": paths})
        np.save(PREP / "train_concept.npy", cids)
        for start in range(0, len(stems), 12):
            batch, rows = [], []
            for row in range(start, min(start + 12, len(stems))):
                try:
                    with z.open(paths[row]) as fp, Image.open(fp) as im:
                        batch.append(tf(im.convert("RGB")))
                    rows.append(row)
                except (OSError, ValueError) as exc:
                    print("unreadable training image", row, type(exc).__name__, flush=True)
            if rows:
                lat = vae.encode(torch.stack(batch).cuda() * 2 - 1).latent_dist.mode() * vae.config.scaling_factor
                assert torch.isfinite(lat).all()
                out[rows] = lat.cpu().half().numpy()
                ok[rows] = True
            if start % 1200 == 0:
                print("encoded original training images", start, flush=True)
        out.flush()
        del out
        (PREP / "lat512_train.partial.npy").replace(PREP / "lat512_train.npy")
        np.save(PREP / "lat512_train_ok.npy", ok)
        train_idx = np.flatnonzero((concept_folds(1654)[cids] != 0) & ok)
        rng = torch.Generator().manual_seed(20261118)
        target, semantic = [], torch.zeros((len(stems), 1024))
        # Fit the semantic marginal only to the training fold. Held-out image values never enter it.
        sources = torch.from_numpy(train_idx)
        for step in range(16):
            donors = sources[torch.randint(len(sources), (len(stems),), generator=rng)]
            for start in range(0, len(stems), 512):
                sl = slice(start, start + 512)
                v = {k: a[sl].cuda().float() for k, a in keys.items()}
                if step == 0:
                    target.append(fe(**v).cpu())
                v[EMB_KEYS[2]] = keys[EMB_KEYS[2]][donors[sl]].cuda().float()
                semantic[sl] += fe(**v).cpu() / 16
        np.save(PREP / "F_train.npy", torch.cat(target).numpy())
        torch.save({"F_sem": semantic.half(), "m_draws": 16}, PREP / "F_sem_train.pt")
    official = archive / "runs/visual_fidelity_20261125_000818/official"
    with (official / "originals_manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["eeg_row"]) for r in rows] == list(range(200))
    write_json(PREP / "test_ids.json", {"ids": [Path(r["archive_member"]).stem for r in rows],
                                         "paths": [r["archive_member"] for r in rows]})
    lock_split()


def lock_split():
    cids = np.load(PREP / "train_concept.npy")
    ok = np.load(PREP / "lat512_train_ok.npy")
    folds = concept_folds(1654)
    fold0 = np.flatnonzero(folds == 0)
    assert len(fold0) == 331
    # The archived 50 were drawn from 327 eligible classes. Their four missing IDs
    # need not match today's JPEG decoder. A selected rank r can move at most four slots.
    old_ranks = np.random.default_rng(20261122).choice(327, 50, replace=False)
    possibly_seen = np.unique(fold0[(old_ranks[:, None] + np.arange(5)).ravel()])
    pool = [c for c in np.setdiff1d(fold0, possibly_seen) if ok[cids == c].all()]
    fresh = np.sort(np.random.default_rng(20261125).choice(pool, 50, replace=False))
    train_idx = np.flatnonzero((folds[cids] != 0) & ok)
    write_json(PREP / "paper_split.json", dict(train_rows=train_idx.tolist(), fresh_rows=np.flatnonzero(np.isin(cids, fresh)).tolist(),
              excluded_possible_historical_concepts=possibly_seen.tolist(), fresh_concepts=fresh.tolist(), official_rows=list(range(200)),
              trial_indices_R4=list(range(4)), trial_indices_R4_balanced=[0, 20, 40, 60], trial_indices_R80=list(range(80)),
              channels=CH17, seed=20261125, historical_eligible=327, historical_fold_size=331))


def lock_development():
    plan = json.loads((PREP / "paper_split.json").read_bytes())
    concepts = np.load(PREP / "train_concept.npy")
    ok = np.load(PREP / "lat512_train_ok.npy")
    blocked = set(plan["fresh_concepts"]) | set(plan["excluded_possible_historical_concepts"])
    eligible = [int(c) for c in np.flatnonzero(concept_folds(1654) == 0)
                if c not in blocked and ok[concepts == c].all()]
    selected = np.sort(np.random.default_rng(287311).choice(eligible, 20, replace=False))
    rows = np.flatnonzero(np.isin(concepts, selected))
    assert len(rows) == 200 and not (set(rows) & (set(plan["fresh_rows"]) | set(plan["train_rows"])))
    write_json(PREP / "paper_development.json", dict(dev_rows=rows.tolist(), concepts=selected.tolist(), seed=287311,
               subjects=[1, 3], conditions=["d2", "response-positive", "response"],
               primary="generated_name1854", secondary=["clip_image_top1", "clip_within10", "clip_2way", "ssim"],
               selection="Positive naming change with preserved within-object image identification on development; confirm across subjects."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generators", "photos", "lock", "development"))
    args = parser.parse_args()
    if args.stage == "development":
        lock_development()
    elif args.stage == "lock":
        lock_split()
    else:
        stage = generators if args.stage == "generators" else photos
        stage(Path(os.environ["ARCHIVE"]))
