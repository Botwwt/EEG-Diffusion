"""Unified Object-Recognizability + Standard Top-Conference Evaluator (GPU, strictly OpenAI ViT-L/14 + TorchVision CNNs)."""
import os, re, json, time, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    alexnet, AlexNet_Weights,
    inception_v3, Inception_V3_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import clip

RAD = os.environ.get("RAD_ROOT", "/path/to/rad")
os.environ.setdefault("TORCH_HOME", f"{RAD}/cache/torch")

CAT27 = [
    "animal", "bird", "body part", "clothing", "clothing accessory",
    "container", "dessert", "drink", "electronic device", "food",
    "fruit", "furniture", "home decor", "insect", "kitchen appliance",
    "kitchen tool", "medical equipment", "musical instrument", "office supply", "part of car",
    "plant", "sports equipment", "tool", "toy", "vegetable", "vehicle", "weapon"
]

class ObjReconEvaluator:
    def __init__(self, device="cuda:0"):
        self.dev = torch.device(device)
        test_raw = json.load(open(f"{RAD}/data/prep/test_ids.json"))
        train_raw = json.load(open(f"{RAD}/data/prep/train_ids.json"))
        def p2c(p):
            folder = p.split("/")[1]
            name = folder.split("_", 1)[1] if "_" in folder else folder
            return re.sub(r"\d+$", "", name).replace("_", " ").strip()
        self.test_names = [p2c(p) for p in test_raw["paths"]]
        seen = set(self.test_names)
        train_only = []
        for p in train_raw["paths"]:
            c = p2c(p)
            if c not in seen:
                seen.add(c); train_only.append(c)
        self.all_1823 = self.test_names + train_only
        self.img_root = f"{RAD}/data/things-eeg/Image_set"

        # Load GT test images [200, 3, 512, 512] float in [0, 1]
        gt_list = []
        for p in test_raw["paths"]:
            im = Image.open(os.path.join(self.img_root, p)).convert("RGB").resize((512, 512), Image.BICUBIC)
            gt_list.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        self.gt_512 = torch.stack(gt_list, dim=0)  # [200, 3, 512, 512]
        self.gt_425 = F.interpolate(self.gt_512, size=(425, 425), mode="bilinear", align_corners=False)
        self.gt_gray = rgb2gray(self.gt_425.permute(0, 2, 3, 1).numpy())
        self.gt_flat = self.gt_425.reshape(200, -1).numpy()

        # 1. OpenAI CLIP ViT-L/14 (strictly evaluation-only!)
        self.clip_m, _ = clip.load(f"{RAD}/cache/clip/ViT-L-14.pt", device=self.dev)
        self.clip_m.eval().requires_grad_(False)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.dev).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.dev).view(1, 3, 1, 1)

        with torch.no_grad():
            def enc_txt(words):
                toks = clip.tokenize([f"a photo of a {w}" for w in words]).to(self.dev)
                parts = [F.normalize(self.clip_m.encode_text(toks[i:i+256]).float(), dim=-1) for i in range(0, len(words), 256)]
                return torch.cat(parts, dim=0)
            self.txt_1823 = enc_txt(self.all_1823)     # [1823, 768]
            self.txt_200 = self.txt_1823[:200]          # [200, 768]
            self.txt_cat27 = enc_txt(CAT27)             # [27, 768]
            self.gt_clip = self._clip_embed(self.gt_512) # [200, 768]
            self.gt_cat27_idx = (self.gt_clip @ self.txt_cat27.T).argmax(dim=-1)  # [200]

        # 2. Standard vision backbones (Inception-v3, EfficientNet-B1, AlexNet)
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

        self.incep_full = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(self.dev).eval().requires_grad_(False)
        self.incep_feat = create_feature_extractor(self.incep_full, return_nodes=["avgpool"]).to(self.dev).eval().requires_grad_(False)

        self.eff_full = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT).to(self.dev).eval().requires_grad_(False)
        self.eff_feat = create_feature_extractor(self.eff_full, return_nodes=["avgpool"]).to(self.dev).eval().requires_grad_(False)

        self.alex_feat = create_feature_extractor(
            alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=["features.4", "features.11"]
        ).to(self.dev).eval().requires_grad_(False)

        with torch.no_grad():
            self.gt_incep_f, self.gt_incep_p = self._incep_run(self.gt_512)
            self.gt_eff_f, self.gt_eff_p = self._eff_run(self.gt_512)
            self.gt_a2, self.gt_a5 = self._alex_run(self.gt_512)

    @torch.no_grad()
    def _clip_embed(self, imgs_bchw):
        out = []
        for i in range(0, len(imgs_bchw), 50):
            x = imgs_bchw[i:i+50].to(self.dev).float()
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
            x = (x - self.clip_mean) / self.clip_std
            out.append(F.normalize(self.clip_m.encode_image(x).float(), dim=-1))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def _incep_run(self, imgs_bchw):
        fs, ps = [], []
        for i in range(0, len(imgs_bchw), 50):
            x = imgs_bchw[i:i+50].to(self.dev).float()
            x = F.interpolate(x, size=(342, 342), mode="bilinear", align_corners=False)
            x = x[:, :, 21:320, 21:320] # center crop 299x299
            x = (x - self.im_mean) / self.im_std
            fs.append(self.incep_feat(x)["avgpool"].float().flatten(1).cpu())
            logits = self.incep_full(x).float()
            ps.append(F.softmax(logits, dim=-1).cpu())
        return torch.cat(fs, 0).numpy(), torch.cat(ps, 0)

    @torch.no_grad()
    def _eff_run(self, imgs_bchw):
        fs, ps = [], []
        for i in range(0, len(imgs_bchw), 50):
            x = imgs_bchw[i:i+50].to(self.dev).float()
            x = F.interpolate(x, size=(255, 255), mode="bilinear", align_corners=False)
            x = x[:, :, 7:247, 7:247] # center crop 240x240
            x = (x - self.im_mean) / self.im_std
            fs.append(self.eff_feat(x)["avgpool"].float().flatten(1).cpu())
            logits = self.eff_full(x).float()
            ps.append(F.softmax(logits, dim=-1).cpu())
        return torch.cat(fs, 0).numpy(), torch.cat(ps, 0)

    @torch.no_grad()
    def _alex_run(self, imgs_bchw):
        a2, a5 = [], []
        for i in range(0, len(imgs_bchw), 50):
            x = imgs_bchw[i:i+50].to(self.dev).float()
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x = (x - self.im_mean) / self.im_std
            o = self.alex_feat(x)
            a2.append(o["features.4"].float().flatten(1).cpu())
            a5.append(o["features.11"].float().flatten(1).cpu())
        return torch.cat(a2, 0).numpy(), torch.cat(a5, 0).numpy()

    @staticmethod
    def _two_way_corr(reals, preds):
        n = len(reals)
        r = np.corrcoef(reals, preds)[:n, n:]
        cong = np.diag(r)
        return float(np.mean(np.sum(r < cong, axis=0)) / (n - 1) * 100.0)

    @staticmethod
    def _topk_200(sim):
        diag = torch.arange(200, device=sim.device)
        t1 = float((sim.argmax(dim=-1) == diag).float().mean().item() * 100.0)
        t5 = float((sim.topk(5, dim=-1).indices == diag.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0)
        return t1, t5

    def evaluate(self, imgs_bchw: torch.Tensor, compute_lowlevel: bool = True) -> dict:
        """imgs_bchw: [200, 3, H, W] float in [0, 1] or uint8 in [0, 255]."""
        if imgs_bchw.dtype == torch.uint8:
            imgs = imgs_bchw.float() / 255.0
        else:
            imgs = imgs_bchw.float().clamp(0, 1)
        if imgs.shape[-1] == 3 and imgs.shape[1] != 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()

        # 1. CLIP ViT-L/14 object-recognizability & image retrieval
        gf = self._clip_embed(imgs)  # [200, 768]
        sim_img = gf @ self.gt_clip.T
        img200_t1, img200_t5 = self._topk_200(sim_img)
        zs200_t1, zs200_t5 = self._topk_200(gf @ self.txt_200.T)
        zs1823_t1, zs1823_t5 = self._topk_200(gf @ self.txt_1823.T)
        # Coarse-27 super-category agreement with GT
        cat_sim = gf @ self.txt_cat27.T
        cat27_t1 = float((cat_sim.argmax(dim=-1) == self.gt_cat27_idx).float().mean().item() * 100.0)
        cat27_t3 = float((cat_sim.topk(3, dim=-1).indices == self.gt_cat27_idx.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0)
        clip_2way = self._two_way_corr(self.gt_clip.cpu().numpy(), gf.cpu().numpy())

        # 2. Non-CLIP Supervised CNN Object Recognition (Inception-v3 & EfficientNet-B1 ImageNet-1K)
        inc_f, inc_p = self._incep_run(imgs)
        eff_f, eff_p = self._eff_run(imgs)
        gt_inc_top1 = self.gt_incep_p.argmax(dim=-1)
        gt_eff_top1 = self.gt_eff_p.argmax(dim=-1)
        in1k_inc_t1 = float((inc_p.argmax(dim=-1) == gt_inc_top1).float().mean().item() * 100.0)
        in1k_inc_t5 = float((inc_p.topk(5, dim=-1).indices == gt_inc_top1.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0)
        in1k_eff_t1 = float((eff_p.argmax(dim=-1) == gt_eff_top1).float().mean().item() * 100.0)
        in1k_eff_t5 = float((eff_p.topk(5, dim=-1).indices == gt_eff_top1.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0)
        in1k_pcos = float(F.cosine_similarity(inc_p, self.gt_incep_p, dim=-1).mean().item() * 100.0)
        obj_crisp = float(inc_p.max(dim=-1).values.mean().item() * 100.0)

        inc_2way = self._two_way_corr(self.gt_incep_f, inc_f)
        # EffNet cosine distance (lower is better, standard Brain-HIVE convention)
        ef_n = eff_f / (np.linalg.norm(eff_f, axis=1, keepdims=True) + 1e-8)
        gt_ef_n = self.gt_eff_f / (np.linalg.norm(self.gt_eff_f, axis=1, keepdims=True) + 1e-8)
        eff_dist = float(np.mean(1.0 - np.sum(ef_n * gt_ef_n, axis=1)))

        out = {
            "zs_obj200_t1": zs200_t1, "zs_obj200_t5": zs200_t5,
            "zs_obj1823_t1": zs1823_t1, "zs_obj1823_t5": zs1823_t5,
            "coarse27_t1": cat27_t1, "coarse27_t3": cat27_t3,
            "in1k_inc_t1": in1k_inc_t1, "in1k_inc_t5": in1k_inc_t5,
            "in1k_eff_t1": in1k_eff_t1, "in1k_eff_t5": in1k_eff_t5,
            "in1k_pcos": in1k_pcos, "obj_crisp": obj_crisp,
            "clip_img200_t1": img200_t1, "clip_img200_t5": img200_t5,
            "clip_2way": clip_2way, "incep_2way": inc_2way, "eff_dist": eff_dist,
        }
        if compute_lowlevel:
            a2, a5 = self._alex_run(imgs)
            out["alex2"] = self._two_way_corr(self.gt_a2, a2)
            out["alex5"] = self._two_way_corr(self.gt_a5, a5)
            r425 = F.interpolate(imgs, size=(425, 425), mode="bilinear", align_corners=False)
            r_flat = r425.reshape(200, -1).numpy()
            out["pixcorr"] = float(np.mean([np.corrcoef(self.gt_flat[i], r_flat[i])[0, 1] for i in range(200)]))
            r_gray = rgb2gray(r425.permute(0, 2, 3, 1).numpy())
            out["ssim"] = float(np.mean([
                structural_similarity(r_gray[i], self.gt_gray[i], gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
                for i in range(200)
            ]))
        return out

if __name__ == "__main__":
    t0 = time.time()
    ev = ObjReconEvaluator("cuda:0")
    oracle = ev.evaluate(ev.gt_512, compute_lowlevel=False)
    print(f"ORACLE GT: ZS200={oracle['zs_obj200_t1']:.2f}/{oracle['zs_obj200_t5']:.2f} | ZS1823={oracle['zs_obj1823_t1']:.2f}/{oracle['zs_obj1823_t5']:.2f} | Cat27={oracle['coarse27_t1']:.1f}% | Crisp={oracle['obj_crisp']:.1f}%", flush=True)
    rows = []
    for s in range(1, 11):
        ac = torch.load(f"{RAD}/runs/o14_official/audit_cache/sub-{s:02d}_O14_audit.pt", map_location="cpu", weights_only=False)
        m = ev.evaluate(ac["k0_rgb_u8"], compute_lowlevel=True)
        rows.append(m)
        print(f"  sub-{s:02d} O14 | ZS200={m['zs_obj200_t1']:5.2f}/{m['zs_obj200_t5']:5.2f} | ZS1823={m['zs_obj1823_t1']:4.2f}/{m['zs_obj1823_t5']:5.2f} | Cat27={m['coarse27_t1']:4.1f}% | IN1K={m['in1k_inc_t1']:4.1f}/{m['in1k_inc_t5']:4.1f} | Crisp={m['obj_crisp']:4.1f}% | Img200={m['clip_img200_t1']:5.2f}% | 2wC={m['clip_2way']:5.2f}% | SSIM={m['ssim']:.3f}", flush=True)
    mean_o14 = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"\n=== O14 10-SUB MEAN ({time.time()-t0:.1f}s) ===", flush=True)
    for k, v in mean_o14.items():
        print(f"  {k:16s}: {v:.4f}")
    os.makedirs(f"{RAD}/runs/obj_bench/evals", exist_ok=True)
    json.dump({"oracle": oracle, "per_sub": rows, "mean": mean_o14}, open(f"{RAD}/runs/obj_bench/evals/O14_all10.json", "w"), indent=2)
