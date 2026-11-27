"""Extract strictly zero-leakage LAION CLIP-ViT-H/14 + IP-Adapter-Plus (16 spatial tokens)
+ Layer-12 4x4 Spatial Grid + Spatial Color targets for THINGS-EEG2 (16,540 train + 200 test oracle).
ZERO evaluator models (DINOv2 / OpenAI CLIP / SigLIP / Inception / EffNet / AlexNet) touched!
"""
import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionXLPipeline

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
PREP = RAD / "data/prep"
IMG_ROOT = RAD / "data/things-eeg/Image_set"
H14_DIR = RAD / "models/CLIP-ViT-H-14-laion2B-s32B-b79K"
TURBO = RAD / "models/sdxl-turbo"
IPA_DIR = RAD / "models/IP-Adapter"
OUT_DIR = RAD / "runs/photo_eeg/targets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PathDataset(Dataset):
    def __init__(self, rel_paths):
        self.paths = []
        for p in rel_paths:
            fp = IMG_ROOT / p
            if not fp.exists():
                fp = IMG_ROOT / p.replace("train_images/", "training_images/")
            self.paths.append(str(fp))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            im = Image.open(self.paths[idx]).convert("RGB").resize((224, 224), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        except Exception:
            arr = np.full((224, 224, 3), 0.5, dtype=np.float32)
        x = torch.from_numpy(arr).permute(2, 0, 1)  # [3, 224, 224]
        return x


def extract_split(split_name: str, rel_paths: list, vis_enc, resampler, dev: str = "cuda:0"):
    t0 = time.time()
    n = len(rel_paths)
    ds = PathDataset(rel_paths)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    plus16 = np.empty((n, 16, 2048), dtype=np.float16)
    h14_l12 = np.empty((n, 4, 4, 1280), dtype=np.float16)
    h14_fin = np.empty((n, 1024), dtype=np.float16)
    rgb16 = np.empty((n, 3, 16, 16), dtype=np.float16)

    pos = 0
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        for xb in dl:
            bs = xb.shape[0]
            xb_gpu = xb.to(dev, non_blocking=True)
            # Vivid 16x16 spatial RGB map
            rgb16[pos:pos + bs] = F.adaptive_avg_pool2d(xb_gpu, (16, 16)).half().cpu().numpy()

            xn = (xb_gpu - mean) / std
            out = vis_enc(pixel_values=xn.half(), output_hidden_states=True)
            # Penultimate hidden state [B, 257, 1280] -> IP-Adapter-Plus Resampler -> [B, 16, 2048]
            penult = out.hidden_states[-2]
            toks16 = resampler(penult)
            if toks16.ndim == 4:
                toks16 = toks16.reshape(bs, -1, toks16.shape[-1])
            plus16[pos:pos + bs] = toks16.half().cpu().numpy()

            # Layer-12 mid-level spatial grid [B, 256, 1280] -> [B, 4, 4, 1280]
            l12 = out.hidden_states[12][:, 1:, :].float()  # [B, 256, 1280]
            l12_grid = l12.transpose(1, 2).reshape(bs, 1280, 16, 16)
            l12_4x4 = F.avg_pool2d(l12_grid, 4).permute(0, 2, 3, 1)  # [B, 4, 4, 1280]
            l12_4x4 = F.normalize(l12_4x4, dim=-1)
            h14_l12[pos:pos + bs] = l12_4x4.half().cpu().numpy()

            # Final CLIP-H/14 global embedding [B, 1024]
            fin = F.normalize(out.image_embeds.float(), dim=-1)
            h14_fin[pos:pos + bs] = fin.half().cpu().numpy()
            pos += bs
            if pos % 3200 == 0 or pos == n:
                print(f"  [{split_name}] {pos}/{n} ({time.time()-t0:.1f}s)", flush=True)

    np.save(OUT_DIR / f"plus16_{split_name}.npy", plus16)
    np.save(OUT_DIR / f"h14_l12_{split_name}.npy", h14_l12)
    np.save(OUT_DIR / f"h14_fin_{split_name}.npy", h14_fin)
    np.save(OUT_DIR / f"rgb16_{split_name}.npy", rgb16)
    print(f"[DONE {split_name}] saved plus16 {plus16.shape}, h14_l12 {h14_l12.shape}, h14_fin {h14_fin.shape}", flush=True)


def main():
    dev = "cuda:0"
    print("Loading CLIP-ViT-H/14 vision encoder...", flush=True)
    vis_enc = CLIPVisionModelWithProjection.from_pretrained(str(H14_DIR), torch_dtype=torch.float16).to(dev).eval()

    print("Loading SDXL-Turbo + IP-Adapter-Plus-SDXL Resampler (exact diffusers module)...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=torch.float16, variant="fp16")
    pipe.load_ip_adapter(
        str(IPA_DIR), subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_folder=None,
    )
    resampler = pipe.unet.encoder_hid_proj.image_projection_layers[0].to(dev, dtype=torch.float16).eval()
    print(f"Resampler type: {type(resampler).__name__}", flush=True)
    del pipe
    torch.cuda.empty_cache()

    train_raw = json.load(open(PREP / "train_ids.json"))
    test_raw = json.load(open(PREP / "test_ids.json"))

    if not (OUT_DIR / "plus16_test.npy").exists():
        extract_split("test", test_raw["paths"], vis_enc, resampler, dev=dev)
    if not (OUT_DIR / "plus16_train.npy").exists():
        extract_split("train", train_raw["paths"], vis_enc, resampler, dev=dev)


if __name__ == "__main__":
    main()
