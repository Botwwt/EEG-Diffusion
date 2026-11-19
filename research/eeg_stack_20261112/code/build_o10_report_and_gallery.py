"""Replay paired O9 vs O10 CLIP 200-way rankings, render pre-registered comparison galleries,
and compile the consolidated O10 scientific report (build_o10_report_and_gallery.py).

Zero test-set tuning: replay uses exact unchanged gen_exp.py flags (K=4 indep, nsel=1, bs=3,
smap c0.625,0.5,1.0, low_keep 8,0.22,0.04) and asserts exact numerical parity with stored metrics.json.
"""
import argparse
import copy
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from common import HIVE, IMG, PREP, PRIOR, RUNS, TURBO, write_json

sys.path.insert(0, str(HIVE))
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel  # noqa: E402

TMP_ROOT = RUNS / "tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TMP_ROOT)

ALL_SUBS = tuple(range(1, 11))
KEYS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "effnet", "clip_top1"]


def detect_gpus(gpus_arg: str = "", min_free_mb: int = 12000):
    if gpus_arg:
        return [int(g.strip()) for g in gpus_arg.split(",") if g.strip()]
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus = [int(ln.split(",")[0]) for ln in out.strip().splitlines() if int(ln.split(",")[1]) >= min_free_mb]
    if not gpus:
        raise RuntimeError(f"No GPUs satisfy min_free_mb={min_free_mb}; nvidia-smi:\n{out}")
    return gpus


def _lp_op(z: torch.Tensor, f: int):
    if f <= 1:
        return z
    return F.interpolate(F.avg_pool2d(z, f), size=z.shape[-1], mode="bilinear", align_corners=False)


def replay_subs_worker(subs, dev: str = "cuda:0"):
    """Exact deterministic gen_exp.py replay for O9 and O10 across subjects, saving rank matrices + k=0 tiles."""
    cache_dir = RUNS / "o10_audit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    need = [s for s in subs if not (cache_dir / f"sub-{s:02d}.pt").exists()]
    if not need:
        return

    import clip  # noqa: E402
    import diffusers  # noqa: E402
    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline  # noqa: E402

    diffusers.utils.logging.set_verbosity_error()
    dt = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(str(TURBO), torch_dtype=dt, variant="fp16")
    attn = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter")
    attn.bind_unet(pipe.unet)
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").to(dev, dt).eval()
    A = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.to(dev, dt).eval()
    pipe.to(dev, dt)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    vae32 = copy.deepcopy(pipe.vae).float()
    with torch.inference_mode():
        pe, _, ppe, _ = pipe.encode_prompt("", device=dev, do_classifier_free_guidance=False, num_images_per_prompt=1)
    p2 = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    p2.set_progress_bar_config(disable=True)

    clip_model, _ = clip.load("ViT-L/14", device=dev, download_root=str(RUNS.parent / "cache/clip"))
    clip_model.eval()
    cln = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    pre_clip = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), cln])

    ids = json.loads((PREP / "test_ids.json").read_text())["ids"]
    gt_tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.ToTensor()])
    files = {f.stem: f for f in (IMG / "test_images").glob("*/*.jpg")}
    gt = torch.stack([gt_tf(Image.open(files[i]).convert("RGB")) for i in ids])

    with torch.no_grad():
        gt_clip = torch.cat([clip_model.encode_image(pre_clip(gt[i:i + 50].to(dev))).float().cpu() for i in range(0, 200, 50)], 0)
    gt_clip_np = gt_clip.numpy()

    smap_c, s_rel, s_unrel = 0.625, 0.5, 1.0
    rho = torch.full((200, 1, 64, 64), smap_c)
    smap = s_unrel - (s_unrel - s_rel) * rho
    hold = (4 - 4 * smap).clamp(min=0)
    K, bs = 4, 3

    def _run_arm(cond_path: Path, lat_path: Path):
        c = torch.from_numpy(np.load(cond_path)).float()
        z0_all = torch.from_numpy(np.load(lat_path)).float()
        with torch.no_grad():
            init = torch.cat([
                (vae32.decode(z0_all[j:j + 20].to(dev) / vae32.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1).cpu()
                for j in range(0, 200, 20)
            ])
        rec = torch.empty(K, 200, 3, 512, 512)
        with torch.inference_mode():
            for s in range(0, 200, bs):
                q = list(range(s, min(200, s + bs)))
                Q = len(q)
                cc = c[q].to(dev, dt)
                ips = proj(A(cc)).repeat_interleave(K, dim=0)
                pe_q = pe.expand(Q * K, -1, -1)
                ppe_q = ppe.expand(Q * K, -1)
                gens = [torch.Generator(dev).manual_seed(1_000_003 * i + k + 12345) for i in q for k in range(K)]
                i0 = init[q].repeat_interleave(K, dim=0).to(dev, dt)
                z0 = z0_all[q].to(dev).float().repeat_interleave(K, dim=0)
                eps = torch.stack([
                    torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(7_000_003 * i + k + 99))
                    for i in q for k in range(K)
                ]).to(dev)
                h_ = hold[q].repeat_interleave(K, dim=0).to(dev)

                def cb(pp, i_step, t_cur, ck):
                    lt = ck["latents"].float()
                    m_ = (h_ - i_step).clamp(0, 1)
                    if float(m_.max()) > 0:
                        sig = pp.scheduler.sigmas[pp.scheduler.step_index].to(dev)
                        tgt = z0 + sig * eps
                        lt = m_ * tgt + (1.0 - m_) * lt
                    if i_step in (2, 3):
                        w_lk = 0.22 if i_step == 2 else 0.04
                        sig = pp.scheduler.sigmas[pp.scheduler.step_index].to(dev)
                        tgt = z0 + sig * eps
                        lt = lt + w_lk * (_lp_op(tgt, 8) - _lp_op(lt, 8))
                    ck["latents"] = lt.to(ck["latents"].dtype)
                    return ck

                imgs = p2(
                    image=i0, strength=s_unrel, generator=gens,
                    prompt_embeds=pe_q, pooled_prompt_embeds=ppe_q,
                    num_inference_steps=4, guidance_scale=0.0, output_type="pt",
                    cross_attention_kwargs={"ip_hidden_states": ips},
                    callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"],
                ).images
                imgs = (imgs.float().clamp(0, 1) * 255).round().div(255)
                rec[:, q] = imgs.cpu().view(Q, K, 3, 512, 512).transpose(0, 1)

        r_mats = np.zeros((K, 200, 200), dtype=np.float32)
        ranks = np.zeros((K, 200), dtype=np.int32)
        preds = np.zeros((K, 200), dtype=np.int32)
        two_ways, top1s = [], []
        for k in range(K):
            with torch.no_grad():
                f_k = torch.cat([clip_model.encode_image(pre_clip(rec[k][i:i + 50].to(dev))).float().cpu() for i in range(0, 200, 50)], 0).numpy()
            r = np.corrcoef(gt_clip_np, f_k)[:200, 200:]
            r_mats[k] = r.astype(np.float32)
            diag = np.diag(r)
            ranks[k] = np.sum(r >= diag[None, :], axis=0).astype(np.int32)
            preds[k] = r.argmax(axis=0).astype(np.int32)
            two_ways.append(float(np.mean(np.sum(r < diag[None, :], axis=0)) / 199.0))
            top1s.append(float(np.mean(preds[k] == np.arange(200))))

        k0_224 = (F.interpolate(rec[0], size=(224, 224), mode="bilinear", align_corners=False) * 255.0).round().clamp(0, 255).to(torch.uint8)
        return r_mats, ranks, preds, float(np.mean(top1s)), float(np.mean(two_ways)), k0_224

    for s in need:
        t0 = time.time()
        o9_c = RUNS / "cond" / "cone_o9_uni7_kvtta_ddpm_w1.00" / f"sub-{s:02d}.npy"
        o9_l = RUNS / "low" / f"sub-{s:02d}" / "lat512_o9_v2_ddpm_n0.80.npy"
        o10_c = RUNS / "cond" / "cone_o10_bifurc_kvtta_w1.00" / f"sub-{s:02d}.npy"
        o10_l = RUNS / "low" / f"sub-{s:02d}" / "lat512_o10_bifurc_v2_n0.80.npy"

        r9, rk9, pr9, t1_9, tw9, k0_9 = _run_arm(o9_c, o9_l)
        r10, rk10, pr10, t1_10, tw10, k0_10 = _run_arm(o10_c, o10_l)

        ref9 = json.loads((RUNS / "img" / f"s{s:02d}_o9_w1_notmf" / "metrics.json").read_text())["mean"]
        ref10 = json.loads((RUNS / "img" / f"s{s:02d}_o10_bifurc_w1_notmf" / "metrics.json").read_text())["mean"]
        d9 = abs(t1_9 - ref9["clip_top1"]) + abs(tw9 - ref9["clip"])
        d10 = abs(t1_10 - ref10["clip_top1"]) + abs(tw10 - ref10["clip"])
        assert d9 < 1e-4 and d10 < 1e-4, f"Replay parity mismatch sub-{s:02d}: d9={d9:.2e}, d10={d10:.2e}"

        payload = {
            "sub": s,
            "arm": "o10_bifurc_w1_notmf",
            "o9_rank": torch.from_numpy(rk9),
            "o10_rank": torch.from_numpy(rk10),
            "o9_pred": torch.from_numpy(pr9),
            "o10_pred": torch.from_numpy(pr10),
            "o9_t1": t1_9,
            "o9_2w": tw9,
            "o10_t1": t1_10,
            "o10_2w": tw10,
            "o9_img_k0": k0_9,
            "o10_img_k0": k0_10,
            "o9_r_diag": torch.from_numpy(np.diagonal(r9, axis1=1, axis2=2).copy()),
            "o10_r_diag": torch.from_numpy(np.diagonal(r10, axis1=1, axis2=2).copy()),
            "o9_r": torch.from_numpy(r9),
            "o10_r": torch.from_numpy(r10),
        }
        tmp_p = cache_dir / f".sub-{s:02d}.{os.getpid()}.pt"
        torch.save(payload, tmp_p)
        tmp_p.replace(cache_dir / f"sub-{s:02d}.pt")
        print(f"[REPLAY] sub-{s:02d} verified (d9={d9:.2e}, d10={d10:.2e}) | O9={t1_9:.4f} -> O10={t1_10:.4f} ({time.time()-t0:.1f}s)", flush=True)


def _replay_cache_valid(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        return torch.load(p, map_location="cpu", weights_only=True).get("arm") == "o10_bifurc_w1_notmf"
    except Exception:
        return False


def run_replay_all(gpus):
    need = [s for s in ALL_SUBS if not _replay_cache_valid(RUNS / "o10_audit_cache" / f"sub-{s:02d}.pt")]
    if not need:
        return
    with tempfile.TemporaryDirectory(prefix="o10_replay_", dir=str(TMP_ROOT)) as td:
        tdir = Path(td)
        buckets = [need[i::len(gpus)] for i in range(len(gpus))]
        procs = []
        for gi, gpu in enumerate(gpus):
            if not buckets[gi]:
                continue
            in_f = tdir / f"replay_gpu{gpu}.json"
            in_f.write_text(json.dumps({"subs": buckets[gi]}))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TMPDIR"] = str(TMP_ROOT)
            procs.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker_replay", str(in_f)], env=env))
        for pr in procs:
            rc = pr.wait()
            if rc != 0:
                raise RuntimeError(f"Replay worker failed with rc={rc}")


def _load_font(size: int, bold: bool = False):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for fp in cands:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _short_concept(stim_id: str, max_len: int = 18) -> str:
    name = stim_id.rsplit("_", 1)[0].replace("_", " ")
    return name[:max_len - 1] + "." if len(name) > max_len else name


def _render_triplet_card(
    sub: int,
    stim_idx: int,
    ids: list,
    gt_u8: torch.Tensor,
    o9_u8: torch.Tensor,
    o10_u8: torch.Tensor,
    r9: int,
    p9: int,
    r10: int,
    p10: int,
    tag_label: str,
    accent_rgb=(26, 127, 55),
) -> Image.Image:
    tile = 224
    gap = 6
    pad = 8
    header_h = 46
    card_w = pad * 2 + tile * 3 + gap * 2
    card_h = header_h + tile + pad

    card = Image.new("RGB", (card_w, card_h), (250, 250, 252))
    draw = ImageDraw.Draw(card)
    f_bold = _load_font(13, bold=True)
    f_reg = _load_font(12, bold=False)

    # Top accent bar
    draw.rectangle([0, 0, card_w - 1, card_h - 1], outline=(215, 220, 226), width=1)
    draw.rectangle([0, 0, 5, header_h], fill=accent_rgb)

    concept_str = _short_concept(ids[stim_idx], 22)
    title = f"sub-{sub:02d} | #{stim_idx:03d} {concept_str} ({ids[stim_idx]}) | k=0"
    draw.text((12, 6), title, fill=(24, 24, 28), font=f_bold)

    # Badge at top-right
    tw = len(tag_label) * 7 + 12
    draw.rectangle([card_w - tw - 8, 5, card_w - 8, 22], fill=accent_rgb)
    draw.text((card_w - tw - 2, 7), tag_label, fill=(255, 255, 255), font=_load_font(11, bold=True))

    p9_str = "exact" if r9 == 1 else _short_concept(ids[p9], 14)
    p10_str = "exact" if r10 == 1 else _short_concept(ids[p10], 14)
    subline = f"Orig Stim (THINGS)     |     O9 rank #{r9}/200 ({p9_str})     |     O10 rank #{r10}/200 ({p10_str})"
    draw.text((12, 26), subline, fill=(65, 72, 82), font=f_reg)

    for col_i, arr_u8 in enumerate((gt_u8, o9_u8, o10_u8)):
        img_pil = Image.fromarray(arr_u8.permute(1, 2, 0).numpy())
        x0 = pad + col_i * (tile + gap)
        card.paste(img_pil, (x0, header_h))
        # Highlight winner column if rank == 1
        if (col_i == 1 and r9 == 1) or (col_i == 2 and r10 == 1):
            draw.rectangle([x0, header_h, x0 + tile - 1, header_h + tile - 1], outline=accent_rgb, width=3)

    return card


def _compose_grid(cards: list, ncols: int = 2, bg=(255, 255, 255)) -> Image.Image:
    cw, ch = cards[0].size
    pad = 12
    nrows = (len(cards) + ncols - 1) // ncols
    canvas = Image.new("RGB", (pad * (ncols + 1) + ncols * cw, pad * (nrows + 1) + nrows * ch), bg)
    for idx, c in enumerate(cards):
        r, col = divmod(idx, ncols)
        canvas.paste(c, (pad + col * (cw + pad), pad + r * (ch + pad)))
    return canvas


def build_gallery_and_confusion(out_dir: Path, subs_cache: dict):
    """Run deterministic stratified sampling (seed k=0), save gallery JPEGs & GALLERY.md, return confusion summary."""
    gal_dir = out_dir / "gallery"
    gal_dir.mkdir(parents=True, exist_ok=True)
    ids = json.loads((PREP / "test_ids.json").read_text())["ids"]
    gt_tf = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(512), T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor()])
    files = {f.stem: f for f in (IMG / "test_images").glob("*/*.jpg")}
    gt_u8 = torch.stack([(gt_tf(Image.open(files[i]).convert("RGB")) * 255).round().clamp(0, 255).to(torch.uint8) for i in ids])

    # 1. Exact item-level confusion analysis over ALL 10 x 4 x 200 = 8,000 samples
    rk9_all = np.stack([subs_cache[s]["o9_rank"].numpy() for s in ALL_SUBS], axis=0)    # [10, 4, 200]
    rk10_all = np.stack([subs_cache[s]["o10_rank"].numpy() for s in ALL_SUBS], axis=0)  # [10, 4, 200]
    pr9_all = np.stack([subs_cache[s]["o9_pred"].numpy() for s in ALL_SUBS], axis=0)    # [10, 4, 200]
    pr10_all = np.stack([subs_cache[s]["o10_pred"].numpy() for s in ALL_SUBS], axis=0)  # [10, 4, 200]

    hit9 = rk9_all == 1
    hit10 = rk10_all == 1
    n_total = hit9.size
    n_both = int((hit9 & hit10).sum())
    n_solved = int((~hit9 & hit10).sum())
    n_regressed = int((hit9 & ~hit10).sum())
    n_neither = int((~hit9 & ~hit10).sum())

    # Hubness S_10 averaged across 10 subjects x 4 seeds
    def _mean_hub10(r_key):
        s10s = []
        for s in ALL_SUBS:
            rm = subs_cache[s][r_key].numpy()  # [4, 200, 200]
            for k in range(4):
                top10 = np.argsort(-rm[k], axis=0)[:10, :]
                cnt = np.bincount(top10.reshape(-1), minlength=200).astype(np.float64)
                s10s.append(float(((cnt - cnt.mean()) ** 3).mean() / max(cnt.std() ** 3, 1e-12)))
        return float(np.mean(s10s))

    hub9 = _mean_hub10("o9_r")
    hub10 = _mean_hub10("o10_r")

    # Per-stimulus net hits across 40 trials (10 subs x 4 seeds)
    stim_h9 = hit9.sum(axis=(0, 1))
    stim_h10 = hit10.sum(axis=(0, 1))
    stim_delta = stim_h10 - stim_h9
    gained_order = np.argsort(-stim_delta, kind="stable")[:12]
    lost_order = np.argsort(stim_delta, kind="stable")[:8]

    # Most frequent persistent confusion pairs when BOTH O9 and O10 fail (O10 pred != j)
    conf_counts = Counter()
    hard_stim_err = np.zeros(200, dtype=np.int32)
    for si in range(10):
        for k in range(4):
            for j in range(200):
                if (not hit9[si, k, j]) and (not hit10[si, k, j]):
                    hard_stim_err[j] += 1
                    conf_counts[(j, int(pr10_all[si, k, j]))] += 1

    # Zero-hit stimuli across all 40 attempts
    zero_both = [int(j) for j in range(200) if stim_h9[j] == 0 and stim_h10[j] == 0]
    zero_o9_only = [int(j) for j in range(200) if stim_h9[j] == 0 and stim_h10[j] > 0]

    # 2. Pre-registered Deterministic Sampling on Seed Index k=0
    k0_r9 = rk9_all[:, 0, :]    # [10, 200]
    k0_r10 = rk10_all[:, 0, :]
    k0_p9 = pr9_all[:, 0, :]
    k0_p10 = pr10_all[:, 0, :]

    overview_picks = []
    used_subs = set()

    def _pick_from_pool(pool, idx_fn, tag, accent, require_unused_sub=True):
        cand = [x for x in pool if (not require_unused_sub or x[0] not in used_subs)]
        if not cand:
            cand = pool
        if not cand:
            return
        pick = idx_fn(cand)
        used_subs.add(pick[0])
        s_num, j_idx, r9v, r10v = pick
        overview_picks.append((s_num, j_idx, tag, accent))

    # Stratum A: Newly solved (r9 > 1 -> r10 == 1), sorted by (r9, s, j)
    pool_solved = sorted([(si + 1, j, int(k0_r9[si, j]), int(k0_r10[si, j]))
                          for si in range(10) for j in range(200) if k0_r9[si, j] > 1 and k0_r10[si, j] == 1],
                         key=lambda x: (x[2], x[0], x[1]))
    _pick_from_pool(pool_solved, lambda c: c[len(c) // 2], "SOLVED (p50 O9 rank)", (26, 127, 55))
    _pick_from_pool(pool_solved, lambda c: c[int(0.85 * (len(c) - 1))], "SOLVED (p85 O9 rank)", (26, 127, 55))

    # Stratum B: Shared Rank-1 (r9 == 1 & r10 == 1), median stim_idx per unused sub
    for b_idx in range(2):
        sub_cands = []
        for s_num in ALL_SUBS:
            if s_num in used_subs:
                continue
            js = [j for j in range(200) if k0_r9[s_num - 1, j] == 1 and k0_r10[s_num - 1, j] == 1]
            if js:
                j_mid = js[len(js) // 2]
                sub_cands.append((s_num, j_mid, 1, 1))
        if sub_cands:
            _pick_from_pool(sub_cands, lambda c: c[0], "BOTH RANK #1", (9, 105, 218))

    # Stratum C: Near-Miss Top-5/10 (2 <= r10 <= 10), closest to r10=5
    pool_near = sorted([(si + 1, j, int(k0_r9[si, j]), int(k0_r10[si, j]))
                        for si in range(10) for j in range(200) if 2 <= k0_r10[si, j] <= 10],
                       key=lambda x: (abs(x[3] - 5), x[3], -x[2], x[0], x[1]))
    _pick_from_pool(pool_near, lambda c: c[0], "NEAR-MISS TOP-10", (154, 103, 0))
    _pick_from_pool(pool_near, lambda c: c[0], "NEAR-MISS TOP-10", (154, 103, 0))

    # Stratum D: 1 Regression (r9 == 1 -> r10 > 1, median r10) + 1 Persistent Hard Failure (r9 >= 100 & r10 >= 100, p75 r10)
    pool_reg = sorted([(si + 1, j, 1, int(k0_r10[si, j]))
                       for si in range(10) for j in range(200) if k0_r9[si, j] == 1 and k0_r10[si, j] > 1],
                      key=lambda x: (x[3], x[0], x[1]))
    _pick_from_pool(pool_reg, lambda c: c[len(c) // 2], "REGRESSION (O9 #1)", (207, 34, 46))

    pool_hard = sorted([(si + 1, j, int(k0_r9[si, j]), int(k0_r10[si, j]))
                        for si in range(10) for j in range(200) if k0_r9[si, j] >= 100 and k0_r10[si, j] >= 100],
                       key=lambda x: (x[3], x[2], x[0], x[1]))
    _pick_from_pool(pool_hard, lambda c: c[int(0.75 * (len(c) - 1))], "HARD FAILURE", (207, 34, 46))

    overview_cards = []
    for s_num, j_idx, tag, accent in overview_picks:
        sc = subs_cache[s_num]
        overview_cards.append(_render_triplet_card(
            s_num, j_idx, ids, gt_u8[j_idx], sc["o9_img_k0"][j_idx], sc["o10_img_k0"][j_idx],
            int(k0_r9[s_num - 1, j_idx]), int(k0_p9[s_num - 1, j_idx]),
            int(k0_r10[s_num - 1, j_idx]), int(k0_p10[s_num - 1, j_idx]),
            tag, accent,
        ))
    _compose_grid(overview_cards, ncols=2).save(gal_dir / "readme_overview_o9_vs_o10.jpg", quality=90, optimize=True)

    # Full 10-Subject Gallery: 5 sheets of 2 subjects x 4 strata = 40 triplets
    all_gal_rows = []
    for a_sub, b_sub in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]:
        pair_cards = {}
        for s_num in (a_sub, b_sub):
            si = s_num - 1
            chosen = set()
            items = []

            sol = sorted([j for j in range(200) if k0_r9[si, j] > 1 and k0_r10[si, j] == 1],
                         key=lambda j: (int(k0_r9[si, j]), j))
            if not sol:
                sol = sorted([j for j in range(200) if k0_r10[si, j] < k0_r9[si, j]], key=lambda j: (int(k0_r10[si, j]), j))
            j1 = sol[len(sol) // 2]
            chosen.add(j1)
            items.append((j1, "SOLVED (O10 #1)", (26, 127, 55)))

            b1 = [j for j in range(200) if k0_r9[si, j] == 1 and k0_r10[si, j] == 1 and j not in chosen]
            j2 = b1[len(b1) // 2] if b1 else [j for j in range(200) if k0_r10[si, j] == 1 and j not in chosen][0]
            chosen.add(j2)
            items.append((j2, "BOTH RANK #1", (9, 105, 218)))

            order_r10 = sorted([j for j in range(200) if j not in chosen], key=lambda j: (int(k0_r10[si, j]), int(k0_r9[si, j]), j))
            j3 = order_r10[len(order_r10) // 2]
            chosen.add(j3)
            items.append((j3, "SUBJECT MEDIAN", (154, 103, 0)))

            order_fail = [j for j in order_r10 if j not in chosen]
            j4 = order_fail[int(0.90 * (len(order_fail) - 1))]
            chosen.add(j4)
            items.append((j4, "HARD FAILURE (p90)", (207, 34, 46)))

            sc = subs_cache[s_num]
            pair_cards[s_num] = []
            for j_idx, tg, acc in items:
                pair_cards[s_num].append(_render_triplet_card(
                    s_num, j_idx, ids, gt_u8[j_idx], sc["o9_img_k0"][j_idx], sc["o10_img_k0"][j_idx],
                    int(k0_r9[si, j_idx]), int(k0_p9[si, j_idx]),
                    int(k0_r10[si, j_idx]), int(k0_p10[si, j_idx]),
                    tg, acc,
                ))
                all_gal_rows.append((f"gallery_sub{a_sub:02d}_{b_sub:02d}.jpg", s_num, j_idx, ids[j_idx], tg,
                                     int(k0_r9[si, j_idx]), ids[int(k0_p9[si, j_idx])],
                                     int(k0_r10[si, j_idx]), ids[int(k0_p10[si, j_idx])]))

        interleaved = []
        for r_idx in range(4):
            interleaved.append(pair_cards[a_sub][r_idx])
            interleaved.append(pair_cards[b_sub][r_idx])
        _compose_grid(interleaved, ncols=2).save(gal_dir / f"gallery_sub{a_sub:02d}_{b_sub:02d}.jpg", quality=90, optimize=True)

    # Write GALLERY.md
    g_lines = [
        "# Complete 10-Subject Side-by-Side Gallery: O9 vs O10 (`k = 0`, 200-Way CLIP Ranking)",
        "",
        "Each triplet shows **[Original Stimulus | O9 Reconstruction | O10 Reconstruction]** at identical random seed (`k = 0`, `nsel = 1`, `4` SDXL-Turbo steps, strictly single-subject EEG, `80` repeats).",
        "",
        "> **Attribution (THINGS-EEG2 Stimuli)**: Reference stimulus thumbnails in the left column come from the THINGS-EEG2 test set (`test_images`, Gifford et al., *NeuroImage* 2022, CC-BY 4.0) for visual verification alongside the corresponding `O9` and `O10` EEG reconstructions.",
        "",
        "## Pre-Registered Deterministic Sampling Rule (Zero Cherry-Picking)",
        "",
        "- **Seed fixed**: seed index `k = 0` (`gen_exp.py` generator seed `1_000_003 * j + 12345`, latent noise seed `7_000_003 * j + 99`) across all `10` subjects (`200` test stimuli each).",
        "- **README Overview Grid (`readme_overview_o9_vs_o10.jpg`)**: 8 triplets across 8 distinct subjects: `2` newly solved (`O9 rank > 1 -> O10 rank = 1` at exact 50th and 85th percentiles of `O9` rank), `2` shared rank-1 hits, `2` near-misses (`2 <= O10 rank <= 10`, closest to rank 5), `1` median regression (`O9 = 1 -> O10 > 1`), and `1` persistent hard failure (75th percentile among `O9 >= 100` & `O10 >= 100`).",
        "- **Per-Subject Sheets (`gallery_sub01_02.jpg` .. `gallery_sub09_10.jpg`)**: For each subject (`sub-01` .. `sub-10`), 4 stratified items: median-`r_O9` **SOLVED (`O9 > 1 -> O10 = 1`)**, median-index **BOTH #1**, exact **50th-percentile (`SUBJECT MEDIAN`)** of `O10` rank, and exact **90th-percentile (`HARD FAILURE`)** of `O10` rank.",
        "",
        "## Overview Sheet (`O9` vs `O10` Across 4 Rank Strata)",
        "",
        "![O9 vs O10 Overview Gallery](readme_overview_o9_vs_o10.jpg)",
        "",
        "## Complete 10-Subject Sheets",
        "",
    ]
    for a_sub, b_sub in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]:
        g_lines.append(f"### Subjects sub-{a_sub:02d} & sub-{b_sub:02d}")
        g_lines.append("")
        g_lines.append(f"![Subjects {a_sub:02d} and {b_sub:02d}](gallery_sub{a_sub:02d}_{b_sub:02d}.jpg)")
        g_lines.append("")

    g_lines.extend([
        "## Index of All 40 Displayed Items (`k = 0`)",
        "",
        "| Sheet | Subject | Stim # | Stimulus ID | Stratum | O9 Rank | O9 Top-1 Prediction | O10 Rank | O10 Top-1 Prediction |",
        "|---|---:|---:|---|---|---:|---|---:|---|",
    ])
    for sh, sn, jx, sid, tg, r9v, p9s, r10v, p10s in all_gal_rows:
        p9_disp = "**exact**" if r9v == 1 else f"`{p9s}`"
        p10_disp = "**exact**" if r10v == 1 else f"`{p10s}`"
        g_lines.append(f"| `{sh}` | sub-{sn:02d} | {jx:03d} | `{sid}` | {tg} | #{r9v}/200 | {p9_disp} | **#{r10v}/200** | {p10_disp} |")
    (gal_dir / "GALLERY.md").write_text("\n".join(g_lines) + "\n", encoding="utf-8")

    return {
        "n_total": int(n_total),
        "n_both": n_both,
        "n_solved": n_solved,
        "n_regressed": n_regressed,
        "n_neither": n_neither,
        "o9_top1_exact": float(hit9.mean()),
        "o10_top1_exact": float(hit10.mean()),
        "o9_top5": float((rk9_all <= 5).mean()),
        "o10_top5": float((rk10_all <= 5).mean()),
        "o9_mean_rank": float(rk9_all.mean()),
        "o10_mean_rank": float(rk10_all.mean()),
        "o9_median_rank": float(np.median(rk9_all)),
        "o10_median_rank": float(np.median(rk10_all)),
        "o9_hub_s10": hub9,
        "o10_hub_s10": hub10,
        "zero_both_count": len(zero_both),
        "zero_o9_rescued_count": len(zero_o9_only),
        "gained_stim": [(int(j), ids[int(j)], int(stim_h9[j]), int(stim_h10[j]), int(stim_delta[j])) for j in gained_order],
        "lost_stim": [(int(j), ids[int(j)], int(stim_h9[j]), int(stim_h10[j]), int(stim_delta[j])) for j in lost_order],
        "top_confusions": [(int(a), ids[a], int(b), ids[b], int(cnt)) for (a, b), cnt in conf_counts.most_common(12)],
        "hardest_stim": [(int(j), ids[int(j)], int(stim_h9[j]), int(stim_h10[j]), int(hard_stim_err[j]))
                         for j in np.argsort(-hard_stim_err, kind="stable")[:10]],
        "overview_picks": [(sn, jx, ids[jx], tg, int(k0_r9[sn - 1, jx]), ids[int(k0_p9[sn - 1, jx])],
                            int(k0_r10[sn - 1, jx]), ids[int(k0_p10[sn - 1, jx])])
                           for sn, jx, tg, _ in overview_picks],
    }


def _load_arm(pat: str, subs=ALL_SUBS):
    out = {}
    for s in subs:
        run, _, var = pat.format(s).partition("#")
        p = RUNS / "img" / run / "metrics.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        m = d[var.split(":", 1)[0]][var.split(":", 1)[1]] if var else d["mean"]
        tp = RUNS / "img" / run / "timing.json"
        t_info = json.loads(tp.read_text()) if tp.exists() else {}
        out[s] = {"vec": np.array([m[k] for k in KEYS], dtype=float), "args": d.get("args", {}), "timing": t_info}
    return out


def build_report(out_dir: Path, subs_cache: dict, conf: dict, dev_report_path: str = "", probe4_log_path: str = "", bifurc_log_path: str = ""):
    res_dir = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    arms_spec = [
        ("Brain-HIVE (repro, nsel=1, 4 NFE)", "s{:02d}_t0_s2025_A1"),
        ("ordinary head (per sub, nsel=1, 4 NFE)", "s{:02d}_t0_s2025_A1_ord"),
        ("O6 (prior original, 22 ckpts, nsel=1)", "s{:02d}_t0_s2025_A1_ordmw7bag10_lowrktta1_0.5_smap_c0.611_0.5_1.0"),
        ("O7a (w=1.00, 43+ ckpts, nsel=1)", "s{:02d}_w4_O7_w1_notmf"),
        ("O8 (w=1.00, 92 ckpts, nsel=1)", "s{:02d}_o8_w1_notmf"),
        ("O9-0tta (8 ckpts, 0 TTA, nsel=1, 4 NFE)", "s{:02d}_o9_0tta_w1_notmf"),
        ("O9 (8 ckpts, 350 kvTTA, nsel=1, 4 NFE)", "s{:02d}_o9_w1_notmf"),
        ("O9-zero00-0head (eta=0, 7 ckpts, nsel=1)", "s{:02d}_o10_ffull00_w1_notmf"),
        ("O10-sem-0tta (eta=1, 0 TTA, nsel=1)", "s{:02d}_o10_0tta_w1_notmf"),
        ("O10-sem-noPhysReSyn (eta=1, noPR, nsel=1)", "s{:02d}_o10_nophys_w1_notmf"),
        ("O10-sem-only (eta=1, PhysReSyn, nsel=1)", "s{:02d}_o10_w1_notmf"),
        ("O10-0tta (eta=0.5, 0 TTA, nsel=1, 4 NFE)", "s{:02d}_o10_bifurc_0tta_w1_notmf"),
        ("O10 (Headline: eta=0.5, 0-head, nsel=1)", "s{:02d}_o10_bifurc_w1_notmf"),
        ("O10-sel2 (budget: nsel=2, 8 NFE)", "s{:02d}_o10_bifurc_sel4_notmf#selvar:0.24,0.22@2"),
        ("O10-sel4 (budget: nsel=4, 16 NFE)", "s{:02d}_o10_bifurc_sel4_notmf"),
        ("B5-1a (full-stack, nsel=1, 4 NFE)", "s{:02d}_b5_n1_notmf"),
        ("B5a (full-stack, nsel=16, 64 NFE)", "s{:02d}_b5_sel16_notmf"),
    ]
    data = {lbl: _load_arm(pat) for lbl, pat in arms_spec}
    data = {k: v for k, v in data.items() if len(v) == 10}

    lines = []
    w = lines.append
    w("=" * 144)
    w("O10 BIFURCATED FORWARD-OPERATOR SYNTHESIS — DEV BOTTLENECK AUDIT, (0.35, 0.25) ELIMINATION & 10-SUBJECT OFFICIAL BENCHMARK")
    w("=" * 144)
    w("Official Protocol : THINGS-EEG2 test set (200 images x 80 repeats, 17 occipito-parietal channels, 0-1000 ms @ 250 Hz)")
    w("Parity Controls   : Single-subject training EEG only, w = 1.00 pure F_train[ok] cone marginal, nsel = 1 (4 UNet NFE/img),")
    w("                    strictly independent K = 4 draws (no --tmf), zero post-hoc Gaussian blur, OpenAI CLIP ViT-L/14 scorer.")
    w("O10 Architecture  : Stream 1 (7x KVEBPyrEncoder on F_train, (0,0) InfoNCE, 0 heads) + Stream 2 (7x on F_sem + FN-mask/margin")
    w("                    + PhysReSyn, 0 heads), 350-step 0-head L_kv kvTTA per stream, eta = 0.50 tangent bifurcation on F_train[ok].")

    # Include dev report if supplied
    if dev_report_path and Path(dev_report_path).exists():
        w("")
        w("=" * 144)
        w("SECTION A. LEAK-FREE DEVELOPMENT BENCHMARK (D_dev200 = fold 0, sub-{02,05,08,09}; ZERO OFFICIAL TEST ACCESS)")
        w("=" * 144)
        w(Path(dev_report_path).read_text(encoding="utf-8").strip())

    def _append_dev_log(log_path: str, tag_pat: str, header_title: str):
        if not log_path or not Path(log_path).exists():
            return
        txt = Path(log_path).read_text(encoding="utf-8", errors="ignore")
        rows = re.findall(tag_pat, txt)
        if not rows:
            return
        w("")
        w(header_title)
        for reg in ("r4", "r40"):
            reg_rows = [r for r in rows if r[1] == reg]
            if not reg_rows:
                continue
            w(f"\n--- Regime: {reg} ---")
            w(f"  {'Arm':<20} | {'A:F_t1':>7} {'A:Fs_t1':>7} | {'CLIP_t1':>8} {'CLIP_2w':>8} {'SSIM':>7} {'Extra':>7}")
            w("  " + "-" * 74)
            for a in sorted({r[2] for r in reg_rows}):
                vals = np.array([[float(x) for x in r[3:]] for r in reg_rows if r[2] == a])
                m = vals.mean(0)
                w(f"  {a:<20} | {m[0]:>7.4f} {m[1]:>7.4f} | {m[2]:>8.4f} {m[3]:>8.4f} {m[4]:>7.4f} {m[5]:>7.4f}")

    _append_dev_log(
        probe4_log_path,
        r"\[PROBE4\] s(\d+)\s+(r\d+)\s+(\S+)\s+\|\s+A_F=([0-9.]+)\s+A_Fs=([0-9.]+)\s+\|\s+CLIP_t1=([0-9.]+)\s+2w=([0-9.]+)\s+SSIM=([0-9.]+)\s+hub=([0-9.]+)",
        "=== Phase 2B: Factorized Forward-Operator Re-Synthesis (PhysReSyn) & Derangement Falsification on D_dev200 (mean sub-2,5,8,9; Extra=hubS10) ===",
    )
    _append_dev_log(
        bifurc_log_path,
        r"\[BIFURC\]\s+s(\d+)\s+(r\d+)\s+(\S+)\s+\|\s+A_F=([0-9.]+)\s+A_Fs=([0-9.]+)\s+\|\s+top1=([0-9.]+)\s+2w=([0-9.]+)\s+SSIM=([0-9.]+)\s+PixC=([0-9.]+)",
        "=== Phase 2C: Bifurcated Forward-Operator Synthesis (eta in {0.00, 0.35, 0.50}) on D_dev200 (mean sub-2,5,8,9; Extra=PixCorr) ===",
    )

    # Section B: O10 OOF & Stage-by-Stage Retrieval
    def _collect_stream_stats(enc_tag_name: str, raw_tta_dir: str):
        oof_r, tr_t, tt_t = [], [], []
        for s in ALL_SUBS:
            for f in range(5):
                mp = RUNS / "enc" / enc_tag_name / f"sub-{s:02d}" / f"fold{f}_seed2025" / "metrics.json"
                if mp.exists():
                    oof_r.append(json.loads(mp.read_text()))
            for sd in range(2025, 2032):
                mp = RUNS / "enc" / enc_tag_name / f"sub-{s:02d}" / f"fold-1_seed{sd}" / "metrics.json"
                if mp.exists():
                    tr_t.append(json.loads(mp.read_text()).get("train_s", 0.0))
            rc = RUNS / "cond" / raw_tta_dir / f"sub-{s:02d}.pt"
            if rc.exists():
                tt_t.append(float(torch.load(rc, map_location="cpu", weights_only=True).get("tta_s", 0.0)))
        return oof_r, tr_t, tt_t

    oof_full, enc_t_full, tta_t_full = _collect_stream_stats("o10_ffull_zero00", "raw_o10_ffull_kvtta")
    oof_sem, enc_t_sem, tta_t_sem = _collect_stream_stats("o10_fsem_fnm", "raw_o10_kvtta")
    phys_p = RUNS / "cond" / "o10_physresyn_timing.json"
    phys_s = float(json.loads(phys_p.read_text()).get("per_sub_s", 0.0)) if phys_p.exists() else 0.0
    bif_p = RUNS / "cond" / "o10_bifurc_timing.json"
    bif_s = float(json.loads(bif_p.read_text()).get("per_sub_s", 0.0)) if bif_p.exists() else 0.0

    w("")
    w("=" * 144)
    w("SECTION B. 10-SUBJECT OFFICIAL TEST RESULTS (sub-01 .. sub-10)")
    w("=" * 144)
    if oof_full:
        w(f"Stream-1 OOF (o10_ffull_zero00, n={len(oof_full)} folds, 200 distinct held-out concepts): "
          f"Perp_Ftrain_200w={np.mean([r['oof_perp_200w'] for r in oof_full]):.4f} | "
          f"KVc_200w={np.mean([r['kv_c_200w'] for r in oof_full]):.4f} | "
          f"7-ckpt train={7*np.mean(enc_t_full):.1f}s/sub | 350-step kvTTA={np.mean(tta_t_full) if tta_t_full else 0.0:.1f}s/sub")
    if oof_sem:
        w(f"Stream-2 OOF (o10_fsem_fnm,     n={len(oof_sem)} folds, ho_idx FN-masked): "
          f"Perp_Fsem_200w={np.mean([r['oof_perp_200w'] for r in oof_sem]):.4f} | "
          f"KVc_200w={np.mean([r['kv_c_200w'] for r in oof_sem]):.4f} | "
          f"Perp_Ftrain_200w={np.mean([r['oof_ftrain_perp_200w'] for r in oof_sem]):.4f} | "
          f"7-ckpt train={7*np.mean(enc_t_sem):.1f}s/sub | 350-step kvTTA={np.mean(tta_t_sem) if tta_t_sem else 0.0:.1f}s/sub")
    w(f"O10 Combine Overhead / Subject: PhysReSyn = {phys_s:.2f}s/sub | Tangent Bifurcation (eta=0.50) = {bif_s:.3f}s/sub | Zero BoundedConditionAdapter DDPM heads.")

    w("")
    w(f"{'Arm':<44} {'nsel':>4} {'NFE/im':>6} {'sec/sub':>7}" + "".join(f"{k[:8]:>10}" for k in KEYS))
    w("-" * 144)
    for lbl, pat in arms_spec:
        if lbl not in data:
            continue
        mat = np.stack([data[lbl][s]["vec"] for s in ALL_SUBS], 0)
        mn = mat.mean(0)
        sd = mat.std(0, ddof=1)
        nsel_v = data[lbl][1]["args"].get("nsel", 1)
        is_selvar = "#selvar:" in pat and "@" in pat
        if is_selvar:
            nsel_v = int(pat.split("@")[-1])
        nfe_v = 4 * int(nsel_v)
        ts = [data[lbl][s]["timing"].get("wall_clock_s") for s in ALL_SUBS]
        ts_ok = [x for x in ts if x is not None]
        t_str = f"{np.mean(ts_ok):>6.1f}s" if (ts_ok and not is_selvar) else "      -"
        w(f"{lbl:<44} {nsel_v:>4} {nfe_v:>6} {t_str:>7}" + "".join(f"{v:>10.4f}" for v in mn))
        w(f"{'  (10-subj SD)':<44} {'':>4} {'':>6} {'':>7}" + "".join(f"{v:>10.4f}" for v in sd))

    # Paired comparisons
    def _fmt_pair(title, subs_list, lbl_b, lbl_a):
        if lbl_b not in data or lbl_a not in data:
            return
        a = np.stack([data[lbl_a][s]["vec"] for s in subs_list], 0)
        b = np.stack([data[lbl_b][s]["vec"] for s in subs_list], 0)
        d = b - a
        dm = d.mean(0)
        n = len(subs_list)
        wins = [int((d[:, i] < 0).sum() if KEYS[i] == "effnet" else (d[:, i] > 0).sum()) for i in range(len(KEYS))]
        tvals = dm / (d.std(0, ddof=1) / (n ** 0.5) + 1e-12)
        w(f"\nPaired ({n} subs {title}): {lbl_b}  vs  {lbl_a}")
        w("  delta : " + " ".join(f"{KEYS[i][:8]}={dm[i]:+.4f}" for i in range(len(KEYS))))
        w("  wins  : " + " ".join(f"{KEYS[i][:8]}={wins[i]}/{n}" for i in range(len(KEYS))))
        w("  t-stat: " + " ".join(f"{KEYS[i][:8]}={tvals[i]:+.2f}" for i in range(len(KEYS))))

    S4, S6, S10 = [2, 5, 8, 9], [1, 3, 4, 6, 7, 10], list(ALL_SUBS)
    w("")
    w("=== Paired Screening ({2,5,8,9}) vs Untouched Held-Out ({1,3,4,6,7,10}) & Isolation Ablations ===")
    o9_k = "O9 (8 ckpts, 350 kvTTA, nsel=1, 4 NFE)"
    o9_z_k = "O9-zero00-0head (eta=0, 7 ckpts, nsel=1)"
    o10_s0_k = "O10-sem-0tta (eta=1, 0 TTA, nsel=1)"
    o10_snp_k = "O10-sem-noPhysReSyn (eta=1, noPR, nsel=1)"
    o10_sem_k = "O10-sem-only (eta=1, PhysReSyn, nsel=1)"
    o10_0_k = "O10-0tta (eta=0.5, 0 TTA, nsel=1, 4 NFE)"
    o10_k = "O10 (Headline: eta=0.5, 0-head, nsel=1)"
    o10_s2_k = "O10-sel2 (budget: nsel=2, 8 NFE)"
    o10_s4_k = "O10-sel4 (budget: nsel=4, 16 NFE)"
    b5a_k = "B5a (full-stack, nsel=16, 64 NFE)"

    _fmt_pair("[screened: 2,5,8,9]", S4, o10_k, o9_k)
    _fmt_pair("[held-out: 1,3,4,6,7,10]", S6, o10_k, o9_k)
    _fmt_pair("[all 10 subjects]", S10, o10_k, o9_k)
    _fmt_pair("[all 10 — Stream 1 isolation: (0,0) 0-head vs O9]", S10, o9_z_k, o9_k)
    _fmt_pair("[all 10 — Stream 2 isolation: sem-only vs O9]", S10, o10_sem_k, o9_k)
    _fmt_pair("[all 10 — Bifurcation gain over Stream 1]", S10, o10_k, o9_z_k)
    _fmt_pair("[all 10 — Bifurcation gain over Stream 2]", S10, o10_k, o10_sem_k)
    _fmt_pair("[all 10 — PhysReSyn inside Stream 2]", S10, o10_sem_k, o10_snp_k)
    _fmt_pair("[all 10 — O10-0tta vs O9-0tta]", S10, o10_0_k, "O9-0tta (8 ckpts, 0 TTA, nsel=1, 4 NFE)")
    _fmt_pair("[all 10 — O10-sel2 (8 NFE) vs O10 (4 NFE)]", S10, o10_s2_k, o10_k)
    _fmt_pair("[all 10 — O10-sel4 (16 NFE) vs O10 (4 NFE)]", S10, o10_s4_k, o10_k)
    _fmt_pair("[all 10 — O10-sel4 (16 NFE) vs B5a (64 NFE)]", S10, o10_s4_k, b5a_k)

    # Per-subject table with O9 vs O10 diff
    w("")
    w("=== Per-Subject Breakdown (sub-01 .. sub-10): O9 vs O10 (Identical nsel=1, 4 NFE, Indep K=4 Protocol) ===")
    w(f"{'Sub':<8} {'O9_top1':>9} {'O10_t1':>9} {'dTop1':>9} | {'O9_2way':>8} {'O10_2w':>8} | {'O9_SSIM':>8} {'O10_SSIM':>8} | {'O9_PixC':>8} {'O10_PixC':>8} | {'O10_s4':>8} {'B5a_t1':>8}")
    w("-" * 124)
    for s in ALL_SUBS:
        v9 = data[o9_k][s]["vec"]
        v10 = data[o10_k][s]["vec"]
        vs4 = data[o10_s4_k][s]["vec"] if o10_s4_k in data else np.zeros(8)
        vb5 = data[b5a_k][s]["vec"] if b5a_k in data else np.zeros(8)
        w(f"sub-{s:02d}   {v9[7]:>9.4f} {v10[7]:>9.4f} {v10[7]-v9[7]:>+9.4f} | {v9[5]:>8.4f} {v10[5]:>8.4f} | "
          f"{v9[1]:>8.4f} {v10[1]:>8.4f} | {v9[0]:>8.4f} {v10[0]:>8.4f} | {vs4[7]:>8.4f} {vb5[7]:>8.4f}")
    m9 = np.stack([data[o9_k][s]["vec"] for s in ALL_SUBS], 0).mean(0)
    m10 = np.stack([data[o10_k][s]["vec"] for s in ALL_SUBS], 0).mean(0)
    ms4 = np.stack([data[o10_s4_k][s]["vec"] for s in ALL_SUBS], 0).mean(0) if o10_s4_k in data else np.zeros(8)
    mb5 = np.stack([data[b5a_k][s]["vec"] for s in ALL_SUBS], 0).mean(0) if b5a_k in data else np.zeros(8)
    w("-" * 124)
    w(f"{'MEAN':<8} {m9[7]:>9.4f} {m10[7]:>9.4f} {m10[7]-m9[7]:>+9.4f} | {m9[5]:>8.4f} {m10[5]:>8.4f} | "
      f"{m9[1]:>8.4f} {m10[1]:>8.4f} | {m9[0]:>8.4f} {m10[0]:>8.4f} | {ms4[7]:>8.4f} {mb5[7]:>8.4f}")

    for lbl_extra in (o9_z_k, o10_s0_k, o10_snp_k, o10_sem_k, o10_0_k, o10_s2_k, o10_s4_k):
        if lbl_extra in data:
            w(f"\n--- {lbl_extra} ---")
            w("Sub     " + "".join(f"{k[:8]:>10}" for k in KEYS))
            for s in ALL_SUBS:
                w(f"sub-{s:02d}  " + "".join(f"{v:>10.4f}" for v in data[lbl_extra][s]["vec"]))
            mat = np.stack([data[lbl_extra][s]["vec"] for s in ALL_SUBS], 0)
            w("MEAN    " + "".join(f"{v:>10.4f}" for v in mat.mean(0)))

    # Stage A & Stage B retrieval on official 10 subjects
    F_te = torch.from_numpy(np.load(PREP / "F_test.npy")).float()
    kv_pack = torch.load(PREP / "unet_kv_metric.pt", map_location="cpu", weights_only=True)
    A_m = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.eval()
    P_m = IPProjectionModel(embed_dim=1024, cross_attention_dim=2048, sequence_length=4)
    P_m.load_state_dict(kv_pack["proj_state"])
    L_kv_t = kv_pack["L_kv"].float().T
    with torch.no_grad():
        kv_te = (P_m(A_m(F_te)) @ L_kv_t).flatten(1)
        kv_te_c = F.normalize(kv_te - kv_te.mean(0, keepdim=True), dim=1)

    w("")
    w("=== Stage-by-Stage Official Test Condition Retrieval (10-Subject Mean) ===")
    for cname, cdir in [
        ("O9 cond", "cone_o9_uni7_kvtta_ddpm_w1.00"),
        ("O9-zero00-0head (eta=0)", "cone_o10_ffull_kvtta_w1.00"),
        ("O10-sem-noPhysReSyn cond", "cone_o10_nophys_kvtta_w1.00"),
        ("O10-sem-only (eta=1)", "cone_o10_kvtta_w1.00"),
        ("O10-0tta (eta=0.5, 0TTA)", "cone_o10_bifurc_0tta_w1.00"),
        ("O10 bifurc (eta=0.5)", "cone_o10_bifurc_kvtta_w1.00"),
    ]:
        t1_f, t1_kv = [], []
        for s in ALL_SUBS:
            cp = RUNS / "cond" / cdir / f"sub-{s:02d}.npy"
            if not cp.exists():
                continue
            cv = torch.from_numpy(np.load(cp)).float()
            r_f = np.corrcoef(F_te.numpy(), cv.numpy())[:200, 200:]
            t1_f.append(float(np.mean(r_f.argmax(0) == np.arange(200))))
            with torch.no_grad():
                kv_p = (P_m(A_m(cv)) @ L_kv_t).flatten(1)
                kv_pc = F.normalize(kv_p - kv_p.mean(0, keepdim=True), dim=1)
                sc_kv = (kv_te_c @ kv_pc.T).numpy()
                t1_kv.append(float(np.mean(sc_kv.argmax(0) == np.arange(200))))
        if t1_f:
            w(f"  {cname:<28}: A:F_test_top1={np.mean(t1_f):.4f} | B:L_kv_top1={np.mean(t1_kv):.4f}")

    # Section C: Item-level confusion transitions
    w("")
    w("=" * 144)
    w("SECTION C. ITEM-LEVEL PAIRED CONFUSION & ERROR TRANSITION AUDIT (8,000 SAMPLES: 10 SUBS x 200 STIMULI x 4 SEEDS)")
    w("=" * 144)
    w(f"Exact Top-1 Hits : O9 = {int(conf['o9_top1_exact']*conf['n_total'])}/{conf['n_total']} ({conf['o9_top1_exact']:.4f})  ->  "
      f"O10 = {int(conf['o10_top1_exact']*conf['n_total'])}/{conf['n_total']} ({conf['o10_top1_exact']:.4f})  "
      f"[+{(conf['o10_top1_exact']-conf['o9_top1_exact'])*100:.2f}%pp, +{conf['n_solved']-conf['n_regressed']} net hits]")
    w(f"2x2 Transition   : Both Correct = {conf['n_both']} | Newly Solved (O9 wrong -> O10 #1) = {conf['n_solved']} | "
      f"Regressed (O9 #1 -> O10 wrong) = {conf['n_regressed']} | Persistent Both Wrong = {conf['n_neither']}")
    w(f"Top-5 / Rank     : Top-5 {conf['o9_top5']:.4f} -> {conf['o10_top5']:.4f} | "
      f"Mean Rank {conf['o9_mean_rank']:.2f} -> {conf['o10_mean_rank']:.2f} | "
      f"Median Rank {conf['o9_median_rank']:.1f} -> {conf['o10_median_rank']:.1f} | "
      f"Hubness S_10: {conf['o9_hub_s10']:.2f} -> {conf['o10_hub_s10']:.2f}")
    w(f"Stimulus Coverage: Zero-hit under BOTH (out of 200): {conf['zero_both_count']} | "
      f"Zero-hit under O9 rescued by O10: {conf['zero_o9_rescued_count']}")

    w("\nTop 12 Stimuli with Largest Gain in Rank-1 Hits (out of 40 trials = 10 subs x 4 seeds):")
    for j, sid, h9, h10, dg in conf["gained_stim"]:
        w(f"  stim #{j:03d} {sid:<28s}: O9={h9:>2d}/40 -> O10={h10:>2d}/40 (delta={dg:+d})")

    w("\nTop 8 Stimuli with Largest Regressions:")
    for j, sid, h9, h10, dg in conf["lost_stim"]:
        w(f"  stim #{j:03d} {sid:<28s}: O9={h9:>2d}/40 -> O10={h10:>2d}/40 (delta={dg:+d})")

    w("\nTop 12 Persistent Wrong Confusions When Both O9 and O10 Miss (True Stimulus -> O10 Predicted Top-1):")
    for a, sa, b, sb, cnt in conf["top_confusions"]:
        w(f"  #{a:03d} {sa:<26s} -> #{b:03d} {sb:<26s} ({cnt} / 40 trials)")

    w("\nTop 10 Hardest Stimuli Remaining Wrong Under Both Models:")
    for j, sid, h9, h10, err_cnt in conf["hardest_stim"]:
        w(f"  stim #{j:03d} {sid:<28s}: both_wrong={err_cnt:>2d}/40 (O9 hits={h9}, O10 hits={h10})")

    report_txt = "\n".join(lines) + "\n"
    (res_dir / "o10_factorized.txt").write_text(report_txt, encoding="utf-8")
    (res_dir / "o10_confusion_summary.json").write_text(json.dumps(conf, indent=1), encoding="utf-8")
    print(f"[REPORT] Saved {res_dir / 'o10_factorized.txt'} ({len(report_txt)} bytes)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--worker_replay", default="")
    ap.add_argument("--dev_report", default="")
    ap.add_argument("--probe4_log", default="")
    ap.add_argument("--bifurc_log", default="")
    args = ap.parse_args()

    if args.worker_replay:
        payload = json.loads(Path(args.worker_replay).read_text())
        replay_subs_worker(payload["subs"], dev="cuda:0")
        return

    if not args.out_dir:
        ap.error("--out_dir is required")

    gpus = detect_gpus(args.gpus, min_free_mb=12000)
    run_replay_all(gpus)

    subs_cache = {s: torch.load(RUNS / "o10_audit_cache" / f"sub-{s:02d}.pt", map_location="cpu", weights_only=True) for s in ALL_SUBS}
    out_dir = Path(args.out_dir)
    conf = build_gallery_and_confusion(out_dir, subs_cache)
    build_report(out_dir, subs_cache, conf, dev_report_path=args.dev_report, probe4_log_path=args.probe4_log, bifurc_log_path=args.bifurc_log)


if __name__ == "__main__":
    main()
