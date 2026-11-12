"""Image -> fused embedding F (CLIP ViT-H/14 + CLIP ViT-B/32 + SDXL-VAE latents at 128 px -> Fusion Prior vision model),
following Brain-HIVE build_embeddings.py. Used to score generated candidates against the EEG condition.

python fenc.py   -> checks the embedder against the published embeddings of the 200 test images
"""
import json
import sys

import numpy as np
import torch
from torchvision import transforms as T

from common import EMB_KEYS, HIVE, IMG, PREP, PRIOR, ROOT, TURBO

nz = torch.nn.functional.normalize


class FEmbedder:
    def __init__(self, dev="cuda"):
        from diffusers import AutoencoderKL
        from transformers import CLIPVisionModelWithProjection
        sys.path.insert(0, str(HIVE))
        from main.models_adapter import FusionEncoderModel
        self.dev = dev
        self.h14 = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[0]), torch_dtype=torch.float16).to(dev).eval()
        self.b32 = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / "models" / EMB_KEYS[1]), torch_dtype=torch.float16).to(dev).eval()
        self.vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval()
        self.fuse = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(dev).eval()
        # The published "CLIP-ViT-B-32" embeddings are a fixed linear function of this checkpoint's output (held-out cosine
        # 0.998 after the fit, 0.28 before); the matrix was fitted on 4132 training images.
        self.b32_fix = torch.load(PREP / "b32_fix.pt", weights_only=True).to(dev)
        self.cn = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def parts(self, img):  # [n, 3, H, H] in [0, 1]
        img = img.to(self.dev).float()
        x = self.cn(torch.nn.functional.interpolate(img, size=224, mode="bicubic", antialias=True).clamp(0, 1)).half()
        v = torch.nn.functional.interpolate(img, size=128, mode="bilinear", antialias=True) * 2 - 1
        lat = self.vae.encode(v).latent_dist.mode() * self.vae.config.scaling_factor  # the published ones are samples
        return {EMB_KEYS[0]: self.h14(pixel_values=x).image_embeds.float(), EMB_KEYS[1]: self.b32(pixel_values=x).image_embeds.float() @ self.b32_fix,
                EMB_KEYS[2]: lat.reshape(len(img), -1).float()}

    @torch.no_grad()
    def __call__(self, img, bs=40):
        return torch.cat([self.fuse(**self.parts(img[i:i + bs])).float() for i in range(0, len(img), bs)])


if __name__ == "__main__":
    from PIL import Image
    ids = json.loads((PREP / "test_ids.json").read_text())["ids"]
    tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor()])
    files = {f.stem: f for f in (IMG / "test_images").glob("*/*.jpg")}
    gt = torch.stack([tf(Image.open(files[i]).convert("RGB")) for i in ids])
    fe = FEmbedder()
    ref = torch.load(PREP / "emb_test.pt", weights_only=True)
    got = {k: [] for k in EMB_KEYS}
    for i in range(0, 200, 40):
        for k, v in fe.parts(gt[i:i + 40]).items():
            got[k].append(v.cpu())
    for k in EMB_KEYS:
        print(k, "cos with published:", float((nz(torch.cat(got[k]), dim=1) * nz(ref[k].float(), dim=1)).sum(1).mean()))
    F = fe(gt).cpu()
    F_te = torch.from_numpy(np.load(PREP / "F_test.npy"))
    s = nz(F, dim=1) @ nz(F_te, dim=1).T
    print("F cos with published:", float(s.diag().mean()), "| 200-way top1:", float((s.argmax(1) == torch.arange(200)).float().mean()),
          "| norm", float(F.norm(dim=1).mean()))
