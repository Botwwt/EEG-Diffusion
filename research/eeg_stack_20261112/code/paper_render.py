"""Render one image per EEG query with a stimulus-blind seed and fixed sample budget."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import PREP, ROOT, RUNS, TURBO, write_json
from exp1_swapdiff_3arm import ResidualIPCondLayer, per_sample_cone
from exp6_draft_intra4 import forward_branches_and_trunk, rollout_4step_step_ips
from exp8_trial_precision_swapdiff import prec_erp_analytic
from main.models_brain import BrainEncoder
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from paper_training import D1, D2, MATCHED, REC, SWAP, check_eeg, generator

BUDGETS = ("official-r4", "official-r4-balanced", "official-r80", "fresh-r4", "dev-r4")
ARMS = {"matched": MATCHED, "rec": REC, "swap": SWAP, "d1": D1, "d2": D2, "wrong": D2}
ARMS.update({"response": D2.parent / "response_s20261125.pt", "response-positive": D2.parent / "response_positive_s20261125.pt"})
# The two crossed arms reuse the identical condition checkpoints of their parent arms.
FACTOR_CONDITION = {"native-layout": "native-paper", "d2-no-layout": "d2"}
INSTANCE_ARMS = ("instance-native", "instance-direct", "instance-residual")
NO_LAYOUT = {"native-code", "native-paper", "d2-no-layout", *INSTANCE_ARMS}
WINDOWS = ((12, 62), (10, 75), (15, 90), (10, 125), (0, 250))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def query_seed(panel, i):
    return int.from_bytes(hashlib.sha256(f"paper-20261125/{panel}/{i}".encode()).digest()[:8], "big") % (2**63 - 1)


def groups(budget):
    if budget == "official-r80":
        return np.arange(80).reshape(20, 4).tolist()
    return [[0, 20, 40, 60]] if budget == "official-r4-balanced" else [[0, 1, 2, 3]]


def precision_mean(x, indices):
    # Average the 20 R4 predictions before the single R80 render. A linear ridge
    # commutes with this group average; nonlinear semantic predictions are averaged later.
    output = []
    for start in range(0, len(x), 64):
        b = torch.from_numpy(np.array(x[start:start + 64], copy=True)).cuda().float()
        output.append(torch.stack([prec_erp_analytic(b[:, group], p=.5) for group in indices]).mean(0).cpu())
    return torch.cat(output)


def low(x, f):
    return F.interpolate(F.avg_pool2d(x.reshape(-1, 4, 64, 64), f), size=64, mode="bilinear", align_corners=False).flatten(1)


def fit_layout(sub, plan, train):
    folder = ROOT / "layouts" / f"sub-{sub:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "ridge.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)
    idx = np.asarray(plan["train_rows"])
    concepts = np.load(PREP / "train_concept.npy")[idx]
    calib = np.isin(concepts, np.unique(concepts)[::8])
    assert 0 < calib.sum() < len(idx)
    x = precision_mean(train[idx], [[0, 1, 2, 3]])
    y = torch.from_numpy(np.load(PREP / "lat512_train.npy", mmap_mode="r")[idx]).cuda().float().flatten(1)
    mu = y.mean(0)
    y -= mu
    torch.manual_seed(20261115 + sub)
    _, _, vec = torch.pca_lowrank(y + 1.5 * low(y, 4) + .75 * low(y, 2), q=288, center=False, niter=4)
    basis = torch.linalg.qr(vec[:, :256].double(), mode="reduced")[0]
    tgt = y.double() @ basis
    ref_rms = F.avg_pool2d(y[calib].view(-1, 4, 64, 64), 4).flatten(1).square().sum(1).mean().sqrt()
    windows, pred_cal = [], []
    for start, stop in WINDOWS:
        a = x[:, :, start:stop].flatten(1).cuda().double()
        for final in (False, True):
            fit, t = (a, tgt) if final else (a[~calib], tgt[~calib])
            center = fit.mean(0)
            scale = fit.std(0).clamp_min(1e-6)
            a_fit = (fit - center) / scale
            cov = a_fit.T @ a_fit / len(fit)
            ridge = 3 * cov.diagonal().mean()
            cov.diagonal().add_(ridge)
            w = torch.linalg.solve(cov, a_fit.T @ t / len(fit))
            if final:
                windows.append((center.float().cpu(), scale.float().cpu(), w.float().cpu()))
            else:
                pred_cal.append(((a[calib] - center) / scale @ w).float())
    cal_img = (torch.stack(pred_cal).mean(0) @ basis.float().T).view(-1, 4, 64, 64)
    pred_rms = F.avg_pool2d(cal_img, 4).flatten(1).square().sum(1).mean().sqrt()
    assert pred_rms > 0
    pack = {"windows": windows, "basis": basis.float().cpu(), "mean": mu.cpu(),
            "gamma": float(.80 * ref_rms / pred_rms), "calibration_rows": int(calib.sum())}
    tmp = path.with_name(f"ridge.{os.getpid()}.partial")
    torch.save(pack, tmp)
    tmp.replace(path)
    return pack


def layout(pack, x, group):
    erp = precision_mean(x, group)
    z = []
    for (start, stop), (center, scale, w) in zip(WINDOWS, pack["windows"]):
        z.append(((erp[:, :, start:stop].flatten(1).cuda() - center.cuda()) / scale.cuda()) @ w.cuda())
    return ((torch.stack(z).mean(0) @ pack["basis"].cuda().T) * pack["gamma"] + pack["mean"].cuda()).reshape(-1, 4, 64, 64).cpu()


def condition(arm, sub, x, group, adapter=None, proj=None, *, features_only=False):
    if arm in INSTANCE_ARMS:
        from paper_instance_residual import apply_condition
        features, digest = apply_condition(arm, sub, x, group)
        if features_only:
            return features, digest
        return torch.cat([proj(adapter(batch.cuda().half())).cpu() for batch in features.split(20)]), digest
    if arm.startswith("native"):
        fold = -1 if RUNS.name.startswith("native-full-") else 0
        runs = RUNS.parent / f"personal-{sub:02d}" if RUNS.name == "shared" else RUNS
        file = runs / f"enc/native/sub-{sub:02d}/fold{fold}_seed2025/brain_model.pt"
        enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=250, hidden_size=1024, embedding_dim=1440, dropout=.1)
        enc.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
        enc.cuda().float().eval()
    else:
        file = ARMS[arm]
        meta = torch.load(file, map_location="cpu", weights_only=True)
        base = meta if "state_dict" in meta else torch.load(meta["backbone_ckpt"], map_location="cpu", weights_only=True)
        enc = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=.20, in_drop=.15).cuda().float().eval()
        enc.load_state_dict(base["state_dict"])
        if not features_only:
            residual = ResidualIPCondLayer().cuda().float().eval()
            residual.load_state_dict(meta["cond_layer"])
        cone = meta["cone"]
    ips = []
    for start in range(0, len(x), 20):
        b = torch.from_numpy(np.array(x[start:start + 20], copy=True)).cuda().float()
        if arm.startswith("native"):
            c = enc.backbone(b[:, np.asarray(group).ravel().tolist()].mean(1))
            ip = c if features_only else proj(adapter(c.half()) if arm == "native-paper" else c.half())
        else:
            fs, hs = [], []
            for g in group:
                bb = b[:, g]
                f, h, _ = forward_branches_and_trunk(enc, prec_erp_analytic(bb), sub - 1, 4.)
                ft, ht, _ = forward_branches_and_trunk(enc, bb.flatten(0, 1), sub - 1, 1.)
                fs.append(.72 * f + .28 * ft.view(len(bb), 4, -1).mean(1))
                hs.append(.72 * h + .28 * ht.view(len(bb), 4, -1).mean(1))
            c = per_sample_cone(torch.stack(fs).mean(0), cone["R"], cone["alpha"], cone["beta"], cone["u_tr"].cuda().float())
            ip = c if features_only else residual(torch.stack(hs).mean(0), proj(adapter(c.half())))
        ips.append(ip.cpu())
    return torch.cat(ips), sha(file)


def draw_images(out, pipe, vae, prompts, tokens, seeds, z0s=None):
    prompt, pooled, dims = prompts
    if prompt.shape[0] != pooled.shape[0] or prompt.shape[0] not in (1, len(seeds)):
        raise ValueError("Expected a shared prompt or a separate text condition for each EEG query")
    with_layout = z0s is not None
    if not with_layout:
        assert math.isclose(float(pipe.scheduler.init_noise_sigma), float(pipe.scheduler.sigmas[0]), rel_tol=1e-6)
    for start in range(0, len(seeds), 8):
        batch = slice(start, start + 8)
        noise = [torch.randn(5, 4, 64, 64, generator=torch.Generator().manual_seed(s)) for s in seeds[batch]]
        eps, *steps = torch.stack(noise).cuda().unbind(1)
        ip = tokens[batch].cuda()
        z0 = z0s[batch].cuda() if with_layout else torch.zeros_like(eps)
        text, pooled_text = (prompt, pooled) if len(prompt) == 1 else (prompt[batch], pooled[batch])
        text = text.to(device=ip.device, dtype=ip.dtype)
        pooled_text = pooled_text.to(device=ip.device, dtype=ip.dtype)
        z = rollout_4step_step_ips(pipe.unet, pipe.scheduler, [ip] * 4, z0, eps, steps, text, pooled_text, dims,
                                   .375 if with_layout else 0., (8, .11, .02) if with_layout else None)
        image = (vae.decode(z.float() / vae.config.scaling_factor).sample.float() * .5 + .5).clamp(0, 1)
        assert torch.isfinite(image).all()
        pixels = image.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for j, pixel in enumerate(pixels, start):
            path = out / f"{j:05d}.png"
            tmp = path.with_suffix(".partial")
            Image.fromarray(pixel).save(tmp, format="PNG")
            tmp.replace(path)


@torch.inference_mode()
def render(a):
    from diffusers import AutoencoderKL
    sub = a.subject
    track = "native-full" if RUNS.name.startswith("native-full-") else ("shared" if RUNS.name == "shared" else "personal")
    if track == "native-full" and (any(not b.startswith("official-") for b in a.budgets)
                                   or any(arm not in ("native-paper", "native-code") for arm in a.arms)):
        raise ValueError("The full 1,654-concept native fit is for the official 200-stimulus split")
    ref = json.loads((PREP / "paper_split.json").read_text())
    train = np.load(PREP / f"sub-{sub:02d}_train.npy", mmap_mode="r")
    test = np.load(PREP / f"sub-{sub:02d}_test.npy", mmap_mode="r")
    check_eeg((sub,), "test")
    need_layout = any(arm not in NO_LAYOUT for arm in a.arms)
    ridge = fit_layout(sub, ref, train) if need_layout else None
    pipe, adapter, proj, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    for budget in a.budgets:
        panel = budget.split("-")[0]
        if panel == "dev":
            qrows = json.loads((PREP / "paper_development.json").read_bytes())["dev_rows"]
        else:
            qrows = ref["fresh_rows"] if panel == "fresh" else ref["official_rows"]
        x = test if panel == "official" else train[qrows]
        g = groups(budget)
        zero = layout(ridge, x, g) if need_layout else None
        seeds = [query_seed(panel, i) for i in range(len(x))]
        assert len(set(seeds)) == len(x)
        for arm in a.arms:
            out = ROOT / f"images/{track}/{arm}/{budget}/sub-{sub:02d}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "render.json").is_file():
                continue
            source_arm = FACTOR_CONDITION.get(arm, arm)
            landmark, ckpt_sha = condition(source_arm, sub, x, g, adapter, proj)
            donors = np.arange(len(x))
            if arm == "wrong":
                donors = np.roll(donors, 1) if panel == "official" else np.roll(donors.reshape(-1, 10), 1, axis=1).ravel()
            with_layout = arm not in NO_LAYOUT
            draw_images(out, pipe, vae, (prompt, pooled, dims), landmark[donors], seeds, zero[donors] if with_layout else None)
            write_json(out / "render.json", dict(training=track, subject=sub, arm=arm, budget=budget, n=len(x), prompt="",
                      draws=1, unet_calls_per_draw=4, repeats=g, row=qrows, eeg_donor=donors.tolist(), seeds=seeds,
                      checkpoint_sha256=ckpt_sha, split_sha256=sha(PREP / "paper_split.json"),
                      semantic_condition=source_arm, layout_mode="ridge-shell" if with_layout else "noise",
                      development_split_sha256=sha(PREP / "paper_development.json") if panel == "dev" else None,
                      pixels_sha256=[sha(out / f"{j:05d}.png") for j in range(len(x))]))
            print("rendered", out, len(x), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, required=True, choices=range(1, 11))
    p.add_argument("--arms", nargs="+", choices=("native-code", "native-paper", *ARMS, *FACTOR_CONDITION, *INSTANCE_ARMS), required=True)
    p.add_argument("--budgets", nargs="+", choices=BUDGETS, required=True)
    render(p.parse_args())
