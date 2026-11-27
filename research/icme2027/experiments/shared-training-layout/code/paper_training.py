"""Train fixed-step benchmark arms on either one participant or the shared cohort."""
import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from common import EMB_KEYS, PREP, PRIOR, ROOT, RUNS, TURBO, write_json
from exp0_intra10_swap_gate import ALL_SUBS, OUT_DIR, get_concept_splits, train_or_load_arm0_tr14
from exp1_swapdiff_3arm import build_tr14_resources, train_arm1_rep_intra
from exp3_swapdiff_v3 import train_raw_swap_v3
from exp4_heldout6_and_condctrl import train_arm1b_rep_condlyr
from exp6_draft_intra4 import DRAFT_DIR, precompute_or_load_tr14_z0_oof, train_draft_arm
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel

BASE = OUT_DIR / "arm0_tr14_xbrain_s2025.pt"
REP = OUT_DIR / "arm1_rep_intra_st800_s2025.pt"
MATCHED = OUT_DIR / "arm1b_rep_condlyr_st600_s2025.pt"
SWAP = OUT_DIR / "arm3b_rep_rawswap_condlyr_st600_s2025.pt"
REC = OUT_DIR / "arm3b_rec_only_st600_s2025.pt"
D1 = DRAFT_DIR / "d1_simonly_cond_st120_s2025.pt"
D2 = DRAFT_DIR / "d2_compete4_cond_st120_s2025.pt"
FULL_BASE = OUT_DIR / "arm0_full_xbrain_s2025.pt"


def check_eeg(subjects, split="train"):
    ids = json.loads((PREP / f"eeg_{split}_ids.json").read_text())["ids"]
    refs = json.loads((PREP / f"{split}_ids.json").read_text())["ids"]
    assert list(ids) == list(refs)
    for subject in subjects:
        base = PREP / f"sub-{subject:02d}_{split}"
        meta = json.loads(base.with_suffix(".json").read_text())
        with base.with_suffix(".npy").open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == meta["output_sha256"]


def generator(dtype, training=False):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dtype, variant="fp16", add_watermarker=False)
    IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").bind_unet(pipe.unet)
    pipe.to("cuda", dtype)
    pipe.scheduler.set_timesteps(4, device="cuda")
    with torch.no_grad():
        prompt, _, pooled, _ = pipe.encode_prompt("", device="cuda", do_classifier_free_guidance=False)
        dims = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=dtype,
                                      text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).cuda()
    pipe.text_encoder = pipe.text_encoder_2 = None
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    if training:
        pipe.unet.enable_gradient_checkpointing()
        pipe.vae.enable_gradient_checkpointing()
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to("cuda", dtype).eval().requires_grad_(False)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to("cuda", dtype).eval().requires_grad_(False)
    return pipe, adapter, proj, prompt, pooled, dims


def vision(key, dtype):
    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / key), torch_dtype=dtype).cuda().eval().requires_grad_(False)
    model.gradient_checkpointing_enable()
    return model


@torch.inference_mode()
def b32_alignment():
    # Historical b32fix sampled every fourth photo. Exclude the full confirmation fold before fitting.
    model = vision(EMB_KEYS[1], torch.float16)
    rows = np.array(json.loads((PREP / "paper_split.json").read_text())["train_rows"])[::4]
    names = json.loads((PREP / "train_ids.json").read_text())["paths"]
    target = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)[EMB_KEYS[1]][rows].double()
    tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor()])
    norm = T.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])
    vectors = []
    with zipfile.ZipFile(Path(os.environ["ARCHIVE"]) / "assets/things_images_v254/training_images.zip") as z:
        for i in range(0, len(rows), 64):
            data = []
            for row in rows[i:i+64]:
                with Image.open(io.BytesIO(z.read(names[row]))) as im:
                    data.append(tf(im.convert("RGB")))
            px = torch.nn.functional.interpolate(torch.stack(data).cuda(), size=224, mode="bicubic", antialias=True).clamp(0, 1)
            vectors.append(model(pixel_values=norm(px).half()).image_embeds.cpu().double())
    fix = torch.linalg.lstsq(torch.cat(vectors), target).solution.float()
    assert fix.shape == (512, 512) and torch.isfinite(fix).all()
    torch.save(fix, PREP / "b32_fix.pt")


def check_full_training():
    labels, available, rows, *_ = get_concept_splits(full_training=True)
    ids = json.loads((PREP / "train_ids.json").read_bytes())["ids"]
    test = json.loads((PREP / "test_ids.json").read_bytes())["ids"]
    classes, counts = np.unique(labels, return_counts=True)
    if len(ids) != 16540 or len(set(ids)) != len(ids) or len(test) != 200 or set(ids).intersection(test):
        raise ValueError("Expected the complete and disjoint official train and test photo registrations")
    if len(classes) != 1654 or not np.all(counts == 10) or not np.all(available) or not np.array_equal(rows, np.arange(16540)):
        raise ValueError("Every official training concept needs ten registered photos and available layout targets")
    return dict(scope="official-training", photos=len(ids), concepts=len(classes), people=list(ALL_SUBS),
                seed=2025, encoder_epochs=150, rep_steps=800, swap_steps=600, d2_steps=120)


def train(stage, subjects, *, full_training=False, crossbrain_weight=0.25, training_layout="individual"):
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    if stage not in ("backbone", "rep", "swap", "rec") and crossbrain_weight != 0.25:
        raise ValueError("Set the cross-person weight for an encoder training stage")
    if training_layout != "individual" and stage != "d2":
        raise ValueError("The group-mean layout experiment is specific to the assignment stage")
    if stage == "b32":
        b32_alignment()
        return
    if stage == "backbone":
        check_eeg(subjects)
        train_or_load_arm0_tr14(subjects=subjects, full_training=full_training, crossbrain_weight=crossbrain_weight)
        return
    resource_scope = {"full_training": True} if full_training else {}
    res = build_tr14_resources("cuda", subjects=subjects, **resource_scope)
    res["crossbrain_weight"] = crossbrain_weight
    backbone = FULL_BASE if full_training else BASE
    if stage == "rep":
        train_arm1_rep_intra(backbone, res, steps=800)
    elif stage == "matched":
        train_arm1b_rep_condlyr(REP, res, steps=600)
    elif stage == "oof":
        precompute_or_load_tr14_z0_oof(res["tr_idx_np"], res["concept_np"], subjects=subjects)
    else:
        pipe, adapter, proj, prompt, pooled, dims = generator(torch.bfloat16, True)
        if stage in ("swap", "rec"):
            name = "arm3b_rep_rawswap_condlyr" if stage == "swap" else "arm3b_rec_only"
            train_raw_swap_v3(name, REP, res, pipe.unet, pipe.scheduler.alphas_cumprod, adapter, proj, prompt, pooled, dims,
                              use_cond_layer=True, w_rep_intra=.5, w_rec=1.5, w_swap=.45 if stage == "swap" else 0., steps=600)
        else:
            mode = {"d1": "d1_simonly_cond", "d2-positive-half": "d2_positive_half_cond", "d2": "d2_compete4_cond"}[stage]
            oof = precompute_or_load_tr14_z0_oof(res["tr_idx_np"], res["concept_np"], subjects=subjects)
            train_draft_arm(mode, SWAP, res, oof, pipe.unet, pipe.scheduler, pipe.vae, adapter, proj,
                            vision(EMB_KEYS[0], torch.bfloat16), vision(EMB_KEYS[1], torch.bfloat16),
                            torch.load(PREP / "b32_fix.pt", weights_only=True).cuda(), prompt, pooled, dims, steps=120,
                            training_layout=training_layout)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("backbone", "rep", "matched", "swap", "rec", "d1", "d2", "d2-positive-half", "b32", "oof"))
    p.add_argument("--subject", type=int, choices=range(1, 11))
    p.add_argument("--full-training", action="store_true")
    p.add_argument("--crossbrain-weight", type=float, choices=(0., .25), default=.25,
                   help="Cross-person alignment weight for the current encoder training stage")
    p.add_argument("--training-layout", choices=("individual", "group-mean"), default="individual",
                   help="Use either one predicted layout per photograph or their group mean during assignment training")
    a = p.parse_args()
    expected = "shared-full" if a.full_training else "shared" if not a.subject else f"personal-{a.subject:02d}"
    if a.stage != "b32" and Path(os.environ["EEG_RUNS_ROOT"]).name != expected:
        raise ValueError("Use an isolated output root for the requested participant group")
    if a.full_training:
        if a.subject or a.stage == "b32":
            raise ValueError("Full training changes only the shared EEG and D2 fits")
        protocol = check_full_training()
        completed = RUNS / f"{a.stage}-training.json"
    train(a.stage, ALL_SUBS if not a.subject else (a.subject,), full_training=a.full_training,
          crossbrain_weight=a.crossbrain_weight, training_layout=a.training_layout)
    if a.full_training:
        write_json(completed, dict(stage=a.stage, **protocol))
