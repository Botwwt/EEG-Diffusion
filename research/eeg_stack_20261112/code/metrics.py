"""MindEye-style reconstruction metrics, definitions of Brain-HIVE main/utils_eval.py (SwAV omitted: weights unreachable).

load_gt() -> ids, gt [200, 3, 512, 512];  score(rec [K, N, 3, 512, 512], gt) -> list of K metric dicts
"""
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from common import IMG, PREP, RUNS

dev = "cuda"


def load_gt():
    ids = json.loads((PREP / "test_ids.json").read_text())["ids"]
    tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor()])
    files = {f.stem: f for f in (IMG / "test_images").glob("*/*.jpg")}
    return ids, torch.stack([tf(Image.open(files[i]).convert("RGB")) for i in ids])


_NETS = None


def _nets():
    global _NETS
    if _NETS is None:
        import clip
        from torchvision.models import (AlexNet_Weights, EfficientNet_B1_Weights, Inception_V3_Weights, alexnet,
                                        efficientnet_b1, inception_v3)
        from torchvision.models.feature_extraction import create_feature_extractor
        IMN = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def rs(n):
            return T.Resize(n, interpolation=T.InterpolationMode.BILINEAR)
        alex = create_feature_extractor(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), ["features.4", "features.11"]).to(dev).eval()
        inc = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), ["avgpool"]).to(dev).eval()
        eff = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), ["avgpool"]).to(dev).eval()
        clip_model, _ = clip.load("ViT-L/14", device=dev, download_root=str(RUNS.parent / "cache/clip"))
        CLN = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        nets = {
            "alex2": (alex, T.Compose([rs(256), IMN]), "features.4"), "alex5": (alex, T.Compose([rs(256), IMN]), "features.11"),
            "inception": (inc, T.Compose([rs(342), IMN]), "avgpool"), "clip": (clip_model.encode_image, T.Compose([rs(224), CLN]), None),
            "effnet": (eff, T.Compose([rs(255), IMN]), "avgpool"),
        }
        _NETS = nets
    return _NETS


def score(rec, gt):
    K, N = rec.shape[:2]
    import clip
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity
    from torchvision.models import (AlexNet_Weights, EfficientNet_B1_Weights, Inception_V3_Weights, alexnet,
                                    efficientnet_b1, inception_v3)
    from torchvision.models.feature_extraction import create_feature_extractor


    IMN = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def rs(n):
        return T.Resize(n, interpolation=T.InterpolationMode.BILINEAR)


    @torch.no_grad()
    def feats(model, pre, x, layer=None):
        o = []
        for i in range(0, len(x), 50):
            y = model(pre(x[i:i + 50].to(dev)))
            o.append((y if layer is None else y[layer]).float().flatten(1).cpu())
        return torch.cat(o).numpy()


    def two_way(real, pred):
        n = len(real)
        r = np.corrcoef(real, pred)[:n, n:]
        return float(np.mean(np.sum(r < np.diag(r), 0)) / (n - 1))


    def corr_dist(real, pred):
        rc, pc = real - real.mean(1, keepdims=True), pred - pred.mean(1, keepdims=True)
        return float(np.mean(1 - (rc * pc).sum(1) / (np.linalg.norm(rc, axis=1) * np.linalg.norm(pc, axis=1))))


    nets = _nets()
    gt_f = {k: feats(m, pre, gt, layer) for k, (m, pre, layer) in nets.items()}
    gt425 = rs(425)(gt)
    gt_gray = rgb2gray(gt425.permute(0, 2, 3, 1).numpy())
    per_sample = []
    for k in range(K):
        r425 = rs(425)(rec[k])
        m = {"pixcorr": float(np.mean([np.corrcoef(gt425[i].reshape(-1), r425[i].reshape(-1))[0, 1] for i in range(N)]))}
        rg = rgb2gray(r425.permute(0, 2, 3, 1).numpy())
        m["ssim"] = float(np.mean([structural_similarity(rg[i], gt_gray[i], gaussian_weights=True, sigma=1.5,
                                                         use_sample_covariance=False, data_range=1.0) for i in range(N)]))
        for name, (mod, pre, layer) in nets.items():
            f = feats(mod, pre, rec[k], layer)
            m[name] = corr_dist(gt_f[name], f) if name == "effnet" else two_way(gt_f[name], f)
            if name == "clip":  # extra: exact-image identification among the 200 test images
                r = np.corrcoef(gt_f[name], f)[:N, N:]
                m["clip_top1"] = float(np.mean(r.argmax(0) == np.arange(N)))
        per_sample.append(m)
    return per_sample
