"""Unified Blind Photo-Likeness + ICME-2027 + Object-Recognizability Evaluator.
100% BLIND to training/guidance/verification (uses ONLY DINOv2-base, OpenAI CLIP ViT-L/14,
SigLIP-SO400M, Inception-v3, EfficientNet-B1, AlexNet, SSIM, PixCorr, Spatial Color).
"""
import os
import re
import json
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    alexnet, AlexNet_Weights,
    inception_v3, Inception_V3_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import clip

RAD = os.environ.get("RAD_ROOT", "/path/to/rad")
os.environ.setdefault("TORCH_HOME", f"{RAD}/cache/torch")
SIGLIP_DIR = os.environ.get("SIGLIP_DIR", "/data/shared/liujiawei2/models/siglip-so400m-patch14-384")

CAT27 = [
    "animal", "bird", "body part", "clothing", "clothing accessory",
    "container", "dessert", "drink", "electronic device", "food",
    "fruit", "furniture", "home decor", "insect", "kitchen appliance",
    "kitchen tool", "medical equipment", "musical instrument", "office supply", "part of car",
    "plant", "sports equipment", "tool", "toy", "vegetable", "vehicle", "weapon",
]


def _p2c(p: str) -> str:
    folder = p.split("/")[1]
    name = folder.split("_", 1)[1] if "_" in folder else folder
    return re.sub(r"\d+$", "", name).replace("_", " ").strip()


def _two_way(feat_gen: np.ndarray, feat_gt: np.ndarray) -> float:
    fg = feat_gen - feat_gen.mean(axis=1, keepdims=True)
    ft = feat_gt - feat_gt.mean(axis=1, keepdims=True)
    fg = fg / (np.linalg.norm(fg, axis=1, keepdims=True) + 1e-8)
    ft = ft / (np.linalg.norm(ft, axis=1, keepdims=True) + 1e-8)
    sim = fg @ ft.T
    diag = np.diag(sim)[:, None]
    n = sim.shape[0]
    wins = (diag > sim).sum() + 0.5 * ((diag == sim).sum() - n)
    return float(100.0 * wins / (n * (n - 1)))


def _centered_retrieval(feat_gen: torch.Tensor, feat_gt: torch.Tensor):
    fg = F.normalize(feat_gen - feat_gen.mean(0, keepdim=True), dim=-1)
    ft = F.normalize(feat_gt - feat_gt.mean(0, keepdim=True), dim=-1)
    sim = fg @ ft.T
    n = sim.shape[0]
    idx = torch.arange(n, device=sim.device)
    t1 = float((sim.argmax(dim=-1) == idx).float().mean().item() * 100.0)
    t5 = float((sim.topk(5, dim=-1).indices == idx[:, None]).any(dim=-1).float().mean().item() * 100.0)
    return t1, t5


class PhotoFidelityEvaluator:
    def __init__(self, device: str = "cuda:0", use_siglip: bool = True):
        self.dev = torch.device(device)
        test_raw = json.load(open(f"{RAD}/data/prep/test_ids.json"))
        train_raw = json.load(open(f"{RAD}/data/prep/train_ids.json"))
        self.test_paths = test_raw["paths"]
        self.test_names = [_p2c(p) for p in self.test_paths]

        seen = set(self.test_names)
        train_only = []
        for p in train_raw["paths"]:
            c = _p2c(p)
            if c not in seen:
                seen.add(c)
                train_only.append(c)
        self.all_1823 = self.test_names + train_only
        self.img_root = f"{RAD}/data/things-eeg/Image_set"

        # Ground-truth test images [200, 3, 512, 512] float in [0, 1]
        gt_list = []
        for p in self.test_paths:
            im = Image.open(os.path.join(self.img_root, p)).convert("RGB").resize((512, 512), Image.BICUBIC)
            gt_list.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        self.gt_512 = torch.stack(gt_list, dim=0)  # CPU [200, 3, 512, 512]
        self.gt_425 = F.interpolate(self.gt_512, size=(425, 425), mode="bilinear", align_corners=False)
        self.gt_gray = rgb2gray(self.gt_425.permute(0, 2, 3, 1).numpy())
        self.gt_flat = self.gt_425.reshape(200, -1).numpy()
        self.gt_col16 = F.avg_pool2d(self.gt_512, 32).reshape(200, -1)  # [200, 768]

        self.im_mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.dev).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.dev).view(1, 3, 1, 1)

        # 1. DINOv2-base (100% blind photo likeness & spatial patch similarity!)
        self.dino = AutoModel.from_pretrained(f"{RAD}/models/dinov2-base").to(self.dev).eval().requires_grad_(False)
        with torch.no_grad():
            self.gt_dino_cls, self.gt_dino_p8 = self._dino_embed(self.gt_512)

        # 2. OpenAI CLIP ViT-L/14 (100% blind!)
        self.clip_m, _ = clip.load(f"{RAD}/cache/clip/ViT-L-14.pt", device=self.dev)
        self.clip_m.eval().requires_grad_(False)
        with torch.no_grad():
            def enc_clip_txt(words):
                toks = clip.tokenize([f"a photo of a {w}" for w in words]).to(self.dev)
                out = [F.normalize(self.clip_m.encode_text(toks[i:i+256]).float(), dim=-1) for i in range(0, len(words), 256)]
                return torch.cat(out, dim=0)
            self.txt_1823 = enc_clip_txt(self.all_1823)
            self.txt_200 = self.txt_1823[:200]
            self.txt_cat27 = enc_clip_txt(CAT27)
            self.gt_clip = self._clip_embed(self.gt_512)
            self.gt_cat27_idx = (self.gt_clip @ self.txt_cat27.T).argmax(dim=-1)

        # 3. SigLIP-SO400M (100% blind independent VLM!)
        self.use_siglip = False
        if use_siglip and os.path.isdir(SIGLIP_DIR):
            try:
                self.siglip_tok = AutoTokenizer.from_pretrained(SIGLIP_DIR)
                self.siglip = AutoModel.from_pretrained(SIGLIP_DIR, torch_dtype=torch.float16).to(self.dev).eval().requires_grad_(False)
                with torch.no_grad():
                    t_parts = []
                    prompts = [f"a photo of a {w}" for w in self.all_1823]
                    for i in range(0, len(prompts), 256):
                        inp = self.siglip_tok(prompts[i:i+256], padding="max_length", truncation=True, return_tensors="pt").to(self.dev)
                        feat = self.siglip.get_text_features(**inp)
                        if not isinstance(feat, torch.Tensor):
                            feat = feat.pooler_output
                        t_parts.append(F.normalize(feat.float(), dim=-1))
                    self.sig_txt_1823 = torch.cat(t_parts, dim=0)
                    self.sig_txt_200 = self.sig_txt_1823[:200]
                    self.gt_siglip = self._siglip_embed(self.gt_512)
                self.use_siglip = True
            except Exception as e:
                print(f"[WARN] SigLIP skipped: {e}", flush=True)

        # 4. TorchVision CNNs (Inception-v3, EfficientNet-B1, AlexNet)
        self.incep_full = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(self.dev).eval().requires_grad_(False)
        self.incep_feat = create_feature_extractor(self.incep_full, return_nodes=["avgpool"]).to(self.dev).eval().requires_grad_(False)
        self.eff_full = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT).to(self.dev).eval().requires_grad_(False)
        self.eff_feat = create_feature_extractor(self.eff_full, return_nodes=["avgpool"]).to(self.dev).eval().requires_grad_(False)
        self.alex = create_feature_extractor(
            alexnet(weights=AlexNet_Weights.DEFAULT),
            return_nodes={"features.4": "alex2", "features.11": "alex5"},
        ).to(self.dev).eval().requires_grad_(False)

        with torch.no_grad():
            self.gt_incep_f, gt_incep_l = self._incep_embed(self.gt_512)
            self.gt_incep_top1 = gt_incep_l.argmax(dim=-1)
            self.gt_eff_f, gt_eff_l = self._eff_embed(self.gt_512)
            self.gt_eff_top1 = gt_eff_l.argmax(dim=-1)
            self.gt_a2, self.gt_a5 = self._alex_embed(self.gt_512)

    @torch.no_grad()
    def _dino_embed(self, imgs_512: torch.Tensor):
        cls_list, p8_list = [], []
        for i in range(0, len(imgs_512), 40):
            x = imgs_512[i:i+40].to(self.dev)
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
            x = (x - self.im_mean) / self.im_std
            hs = self.dino(pixel_values=x).last_hidden_state.float()  # [B, 257, 768]
            cls = F.normalize(hs[:, 0], dim=-1)
            p16 = hs[:, 1:].transpose(1, 2).reshape(-1, 768, 16, 16)
            p8 = F.avg_pool2d(p16, 2)  # [B, 768, 8, 8]
            p8 = F.normalize(p8, dim=1)
            cls_list.append(cls)
            p8_list.append(p8)
        return torch.cat(cls_list, 0), torch.cat(p8_list, 0)

    @torch.no_grad()
    def _clip_embed(self, imgs_512: torch.Tensor):
        out = []
        for i in range(0, len(imgs_512), 50):
            x = imgs_512[i:i+50].to(self.dev)
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
            x = (x - self.clip_mean) / self.clip_std
            out.append(F.normalize(self.clip_m.encode_image(x).float(), dim=-1))
        return torch.cat(out, 0)

    @torch.no_grad()
    def _siglip_embed(self, imgs_512: torch.Tensor):
        out = []
        for i in range(0, len(imgs_512), 25):
            x = imgs_512[i:i+25].to(self.dev)
            x = F.interpolate(x, size=(384, 384), mode="bicubic", align_corners=False).clamp(0, 1)
            x = (x - 0.5) / 0.5
            feat = self.siglip.get_image_features(pixel_values=x.half())
            if not isinstance(feat, torch.Tensor):
                feat = feat.pooler_output
            out.append(F.normalize(feat.float(), dim=-1))
        return torch.cat(out, 0)

    @torch.no_grad()
    def _incep_embed(self, imgs_512: torch.Tensor):
        fs, ls = [], []
        for i in range(0, len(imgs_512), 50):
            x = imgs_512[i:i+50].to(self.dev)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False).clamp(0, 1)
            x = (x - self.im_mean) / self.im_std
            fs.append(self.incep_feat(x)["avgpool"].flatten(1).float())
            ls.append(self.incep_full(x).float())
        return torch.cat(fs, 0), torch.cat(ls, 0)

    @torch.no_grad()
    def _eff_embed(self, imgs_512: torch.Tensor):
        fs, ls = [], []
        for i in range(0, len(imgs_512), 50):
            x = imgs_512[i:i+50].to(self.dev)
            x = F.interpolate(x, size=(255, 255), mode="bilinear", align_corners=False).clamp(0, 1)
            x = (x - self.im_mean) / self.im_std
            fs.append(self.eff_feat(x)["avgpool"].flatten(1).float())
            ls.append(self.eff_full(x).float())
        return torch.cat(fs, 0), torch.cat(ls, 0)

    @torch.no_grad()
    def _alex_embed(self, imgs_512: torch.Tensor):
        a2s, a5s = [], []
        for i in range(0, len(imgs_512), 50):
            x = imgs_512[i:i+50].to(self.dev)
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 1)
            x = (x - self.im_mean) / self.im_std
            o = self.alex(x)
            a2s.append(o["alex2"].flatten(1).float().cpu().numpy())
            a5s.append(o["alex5"].flatten(1).float().cpu().numpy())
        return np.concatenate(a2s, 0), np.concatenate(a5s, 0)

    @torch.no_grad()
    def evaluate(self, imgs_u8_or_f: torch.Tensor, compute_lowlevel: bool = True) -> dict:
        if imgs_u8_or_f.dtype == torch.uint8:
            imgs = imgs_u8_or_f.float() / 255.0
        else:
            imgs = imgs_u8_or_f.float().clamp(0, 1)
        if imgs.shape[-1] != 512:
            imgs = F.interpolate(imgs, size=(512, 512), mode="bicubic", align_corners=False).clamp(0, 1)

        idx = torch.arange(200, device=self.dev)
        res = {}

        # 1. DINOv2 exact-photo & spatial patch likeness
        d_cls, d_p8 = self._dino_embed(imgs)
        res["dino_t1"], res["dino_t5"] = _centered_retrieval(d_cls, self.gt_dino_cls)
        res["dino_2way"] = _two_way(d_cls.cpu().numpy(), self.gt_dino_cls.cpu().numpy())
        res["dino_cos"] = float((d_cls * self.gt_dino_cls).sum(-1).mean().item())
        res["dino_patch_cos"] = float((d_p8 * self.gt_dino_p8).sum(dim=1).mean().item())

        # 2. OpenAI CLIP ViT-L/14
        gc = self._clip_embed(imgs)
        res["clip_img200_t1"], res["clip_img200_t5"] = _centered_retrieval(gc, self.gt_clip)
        res["clip_2way"] = _two_way(gc.cpu().numpy(), self.gt_clip.cpu().numpy())
        res["clip_cos"] = float((gc * self.gt_clip).sum(-1).mean().item())

        s200 = gc @ self.txt_200.T
        res["zs_obj200_t1"] = float((s200.argmax(-1) == idx).float().mean().item() * 100.0)
        res["zs_obj200_t5"] = float((s200.topk(5, -1).indices == idx[:, None]).any(-1).float().mean().item() * 100.0)
        s1823 = gc @ self.txt_1823.T
        res["zs_obj1823_t1"] = float((s1823.argmax(-1) == idx).float().mean().item() * 100.0)
        res["zs_obj1823_t5"] = float((s1823.topk(5, -1).indices == idx[:, None]).any(-1).float().mean().item() * 100.0)
        sc27 = gc @ self.txt_cat27.T
        res["coarse27_t1"] = float((sc27.argmax(-1) == self.gt_cat27_idx).float().mean().item() * 100.0)

        # 3. SigLIP-SO400M (if loaded)
        if self.use_siglip:
            gs = self._siglip_embed(imgs)
            res["siglip_img200_t1"], res["siglip_img200_t5"] = _centered_retrieval(gs, self.gt_siglip)
            ss200 = gs @ self.sig_txt_200.T
            ss1823 = gs @ self.sig_txt_1823.T
            res["siglip_200_t1"] = float((ss200.argmax(-1) == idx).float().mean().item() * 100.0)
            res["siglip_1823_t1"] = float((ss1823.argmax(-1) == idx).float().mean().item() * 100.0)

        # 4. CNNs + Crispness
        inc_f, inc_l = self._incep_embed(imgs)
        inc_p = F.softmax(inc_l, dim=-1)
        res["in1k_inc_t1"] = float((inc_l.argmax(-1) == self.gt_incep_top1).float().mean().item() * 100.0)
        res["in1k_inc_t5"] = float((inc_l.topk(5, -1).indices == self.gt_incep_top1[:, None]).any(-1).float().mean().item() * 100.0)
        res["obj_crisp"] = float(inc_p.max(dim=-1).values.mean().item() * 100.0)
        res["incep_2way"] = _two_way(inc_f.cpu().numpy(), self.gt_incep_f.cpu().numpy())

        eff_f, eff_l = self._eff_embed(imgs)
        res["in1k_eff_t1"] = float((eff_l.argmax(-1) == self.gt_eff_top1).float().mean().item() * 100.0)
        ef1 = eff_f - eff_f.mean(1, keepdim=True)
        ef2 = self.gt_eff_f - self.gt_eff_f.mean(1, keepdim=True)
        res["eff_dist"] = float(1.0 - (F.normalize(ef1, dim=-1) * F.normalize(ef2, dim=-1)).sum(-1).mean().item())

        a2, a5 = self._alex_embed(imgs)
        res["alex2"] = _two_way(a2, self.gt_a2)
        res["alex5"] = _two_way(a5, self.gt_a5)

        # 5. Spatial color fidelity + Low-level PixCorr & SSIM
        col16 = F.avg_pool2d(imgs.cpu(), 32).reshape(200, -1)
        res["color16_corr"] = float(
            (F.normalize(col16 - col16.mean(1, keepdim=True), dim=-1) *
             F.normalize(self.gt_col16 - self.gt_col16.mean(1, keepdim=True), dim=-1)).sum(-1).mean().item()
        )
        if compute_lowlevel:
            g425 = F.interpolate(imgs.cpu(), size=(425, 425), mode="bilinear", align_corners=False)
            gflat = g425.reshape(200, -1).numpy()
            pcs = [np.corrcoef(gflat[i], self.gt_flat[i])[0, 1] for i in range(200)]
            res["pixcorr"] = float(np.nanmean(pcs))
            ggray = rgb2gray(g425.permute(0, 2, 3, 1).numpy())
            ssims = [structural_similarity(ggray[i], self.gt_gray[i], data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False) for i in range(200)]
            res["ssim"] = float(np.mean(ssims))
        return res
