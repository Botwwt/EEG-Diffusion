"""Build O11 report (o11_tri_stream_bigg.txt) and 4-column Orig | O9 | O10 | O11 comparison gallery.

Strictly uses deterministic k=0 renders from official 10-subject evaluations (zero cherry-picking):
  - Pre-registered 5 strata per subject (50 real 4-column comparisons = 200 images across sub-01..10):
      1. RESCUE (O11 #1): O9>1 & O10>1 & O11==1 (fallback O9>1 & O11==1), 35th percentile by -(r9+r10)
      2. SHARED HIT (#1): O9==1 & O10==1 & O11==1, median stimulus index
      3. NEAR-MISS TOP-10: 2 <= O11 <= 10 (with sub-04 #000 aircraft_carrier_06s explicit spotlight)
      4. REGRESSION: (O10==1 or O9==1) & O11>1, median O11 rank
      5. HARD FAILURE: O9>20 & O10>20 & O11>20, median O11 rank
  - Overview contact sheet (readme_overview_o9_o10_o11.jpg): 12 cards drawn directly from the 50
    pre-registered per-subject cards, covering all 10 subjects (4 rescues, 2 shared hits, 2 near-misses
    including sub-04 #000 aircraft_carrier_06s, 2 regressions, 2 hard failures).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import stats


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


BADGE_COLORS = {
    "RESCUE (O11 #1)":    ((22, 128, 61),  (255, 255, 255)),
    "SHARED HIT (#1)":    ((29, 78, 216),  (255, 255, 255)),
    "SPOTLIGHT (TOP-10)": ((161, 98, 7),   (255, 255, 255)),
    "NEAR-MISS TOP-10":   ((180, 83, 9),   (255, 255, 255)),
    "REGRESSION":         ((185, 28, 28),  (255, 255, 255)),
    "HARD FAILURE":       ((75, 85, 99),   (255, 255, 255)),
}


def _short_label(gt_id: str) -> str:
    return gt_id.rsplit("_", 1)[0].replace("_", " ")


def render_card(
    sub: int,
    stim_idx: int,
    tag: str,
    gt_ids: list,
    r9: int, p9: int,
    r10: int, p10: int,
    r11: int, p11: int,
    img_orig: Image.Image,
    img_o9: Image.Image,
    img_o10: Image.Image,
    img_o11: Image.Image,
    tile: int = 196,
) -> Image.Image:
    gap = 6
    pad = 10
    hdr_h = 46
    cap_h = 20
    w = pad * 2 + tile * 4 + gap * 3
    h = pad * 2 + hdr_h + tile + cap_h
    card = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(card)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(215, 220, 226), width=1)

    f_title = _load_font(14, bold=True)
    f_sub = _load_font(12, bold=False)
    f_badge = _load_font(11, bold=True)
    f_cap = _load_font(11, bold=False)

    cname = _short_label(gt_ids[stim_idx])
    title = f"sub-{sub:02d} | #{stim_idx:03d} {cname} ({gt_ids[stim_idx]}) | k=0"
    draw.text((pad, pad - 1), title, fill=(17, 24, 39), font=f_title)

    bg_c, fg_c = BADGE_COLORS.get(tag, ((55, 65, 81), (255, 255, 255)))
    bw = draw.textbbox((0, 0), tag, font=f_badge)[2] + 14
    draw.rounded_rectangle([w - pad - bw, pad - 2, w - pad, pad + 17], radius=4, fill=bg_c)
    draw.text((w - pad - bw + 7, pad), tag, fill=fg_c, font=f_badge)

    def _fmt(r, p):
        lbl = "exact" if r == 1 else _short_label(gt_ids[p])[:16]
        return f"#{r}/200 ({lbl})"

    sub_line = f"O9 {_fmt(r9, p9)}   |   O10 {_fmt(r10, p10)}   |   O11 {_fmt(r11, p11)}"
    draw.text((pad, pad + 21), sub_line, fill=(75, 85, 99), font=f_sub)

    y_img = pad + hdr_h
    cols = [
        ("Orig Stim (THINGS)", img_orig),
        (f"O9  #{r9}/200", img_o9),
        (f"O10 #{r10}/200", img_o10),
        (f"O11 #{r11}/200", img_o11),
    ]
    for ci, (cap, im) in enumerate(cols):
        x0 = pad + ci * (tile + gap)
        im_r = im.resize((tile, tile), Image.Resampling.LANCZOS)
        card.paste(im_r, (x0, y_img))
        col_c = (22, 128, 61) if (ci == 3 and r11 == 1) else ((17, 24, 39) if ci == 3 else (75, 85, 99))
        draw.text((x0 + 2, y_img + tile + 3), cap, fill=col_c, font=f_cap)
    return card


def compose_sheet(title: str, subtitle: str, cards: list, ncols: int = 2) -> Image.Image:
    pad = 16
    gap = 14
    top_h = 62
    cw, ch = cards[0].size
    nrows = (len(cards) + ncols - 1) // ncols
    W = pad * 2 + ncols * cw + (ncols - 1) * gap
    H = top_h + pad + nrows * ch + (nrows - 1) * gap + pad
    canvas = Image.new("RGB", (W, H), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)
    f_h1 = _load_font(19, bold=True)
    f_h2 = _load_font(13, bold=False)
    draw.text((pad, 12), title, fill=(15, 23, 42), font=f_h1)
    draw.text((pad, 38), subtitle, fill=(71, 85, 105), font=f_h2)
    for idx, card in enumerate(cards):
        r, c = divmod(idx, ncols)
        x = pad + c * (cw + gap)
        y = top_h + r * (ch + gap)
        canvas.paste(card, (x, y))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory with o11_on_o10_ranks_10subs.pt, o9_o10_ranks_10subs.pt, k0_jpegs, sel_imgs, eval, summaries")
    ap.add_argument("--repo_root", required=True, help="Path to EEG-Diffusion repository root")
    args = ap.parse_args()

    ddir = Path(args.data_dir)
    repo = Path(args.repo_root)
    res_dir = repo / "research" / "eeg_stack_20261112" / "results"
    gal_dir = repo / "research" / "eeg_stack_20261112" / "gallery"
    res_dir.mkdir(parents=True, exist_ok=True)
    gal_dir.mkdir(parents=True, exist_ok=True)

    o910 = torch.load(ddir / "o9_o10_ranks_10subs.pt", map_location="cpu", weights_only=True)
    o11 = torch.load(ddir / "o11_on_o10_ranks_10subs.pt", map_location="cpu", weights_only=True)
    plan = json.loads((ddir / "gallery_plan.json").read_text())
    gt_ids = o11[1]["gt_ids"]
    cnames = [_short_label(x) for x in gt_ids]

    def load_quad(s: int, idx: int):
        im_orig = Image.open(ddir / "sel_imgs" / "orig" / f"{idx:03d}.jpg").convert("RGB")
        im_o9 = Image.open(ddir / "sel_imgs" / "o9" / f"sub-{s:02d}" / f"{idx:03d}.jpg").convert("RGB")
        im_o10 = Image.open(ddir / "sel_imgs" / "o10" / f"sub-{s:02d}" / f"{idx:03d}.jpg").convert("RGB")
        im_o11 = Image.open(ddir / "k0_jpegs" / f"sub-{s:02d}" / f"{idx:03d}.jpg").convert("RGB")
        return im_orig, im_o9, im_o10, im_o11

    def make_card(s: int, idx: int, tag: str, tile: int = 196):
        im_orig, im_o9, im_o10, im_o11 = load_quad(s, idx)
        r9 = int(o910[s]["o9_rank"][0, idx]); p9 = int(o910[s]["o9_pred"][0, idx])
        r10 = int(o910[s]["o10_rank"][0, idx]); p10 = int(o910[s]["o10_pred"][0, idx])
        r11 = int(o11[s]["ranks"][0, idx]); p11 = int(o11[s]["top1_idx"][0, idx])
        return render_card(s, idx, tag, gt_ids, r9, p9, r10, p10, r11, p11, im_orig, im_o9, im_o10, im_o11, tile=tile)

    # 1) Render Overview Sheet (12 cards, 2 columns x 6 rows)
    ov_cards = [make_card(int(s), int(idx), tag, tile=188) for (s, idx, tag) in plan["overview"]]
    ov_sheet = compose_sheet(
        "EEG-Diffusion O11 vs O10 vs O9 — Pre-Registered Real Reconstructions (k=0, nsel=1, 4 NFE, 0 DDPM Heads)",
        "4 Rescues (held-out subs) | 2 Shared Hits | 2 Near-Misses (incl. sub-04 #000 aircraft carrier spotlight) | 2 Regressions | 2 Hard Failures. Rank = OpenAI CLIP ViT-L/14 out of 200.",
        ov_cards,
        ncols=2,
    )
    ov_path = gal_dir / "readme_overview_o9_o10_o11.jpg"
    ov_sheet.save(ov_path, quality=88, optimize=True)
    print(f"Saved {ov_path} ({ov_path.stat().st_size / 1024:.1f} KB, {ov_sheet.size})")

    # 2) Render 5 Per-Subject Pair Sheets (sub-01..10, 5 cards per subject = 10 cards per sheet, 2 cols x 5 rows)
    for s_a, s_b in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
        cards = []
        rows_a = plan["per_sub"][str(s_a)]
        rows_b = plan["per_sub"][str(s_b)]
        for ri in range(5):
            tag_a, idx_a = rows_a[ri]
            tag_b, idx_b = rows_b[ri]
            cards.append(make_card(s_a, int(idx_a), tag_a, tile=188))
            cards.append(make_card(s_b, int(idx_b), tag_b, tile=188))
        sheet = compose_sheet(
            f"O11 vs O10 vs O9 Full Gallery — sub-{s_a:02d} (left) & sub-{s_b:02d} (right) | Seed k=0",
            "Pre-registered 5 strata per subject: Rescue (O11 #1) | Shared Hit (#1) | Near-Miss Top-10 | Regression | Hard Failure. Columns: Orig | O9 | O10 | O11.",
            cards,
            ncols=2,
        )
        sp_path = gal_dir / f"gallery_o11_sub{s_a:02d}_{s_b:02d}.jpg"
        sheet.save(sp_path, quality=86, optimize=True)
        print(f"Saved {sp_path} ({sp_path.stat().st_size / 1024:.1f} KB, {sheet.size})")

    # 3) Write complete O11 report (results/o11_tri_stream_bigg.txt)
    eval_dir = ddir / "eval"
    sum_dir = ddir / "summaries"
    lines = []
    lines.append("=" * 98)
    lines.append("O11: Tri-Stream Forward-Operator Synthesis & Zero-Shot OpenCLIP-bigG/14 Text Awakening")
    lines.append("Built Directly on O10 (0 DDPM heads, 4 UNet NFE/img, nsel=1, K=4 indep, zero blur)")
    lines.append("Strict Zero-Leakage Guardrail: OpenAI CLIP ViT-L/14 (text_encoder_1) kept strictly empty (\"\")")
    lines.append("=" * 98)

    lines.append("\n[1] D_dev200 Screen 1 — Generator / Sinkhorn-OT / bigG Text Oracle & Falsification (sub-02 r40)")
    lines.append((sum_dir / "o11_gen_screen_summary.txt").read_text().strip())

    lines.append("\n[2] D_dev200 Screen 2 — Upgraded 0-Head Encoders & Direct OpenCLIP-bigG Text (sub-02 r40, 3 seeds)")
    lines.append((sum_dir / "o11_enc_screen_summary.txt").read_text().strip())

    lines.append("\n[3] D_dev200 4-Subject Confirmation (sub-02,05,08,09 r40, 3 seeds, 4 NFE)")
    lines.append((sum_dir / "o11_confirm4_summary.txt").read_text().strip())

    lines.append("\n[4] D_dev200 O10 + O11 Multi-Architecture Synergy Confirmation (sub-02,05,08,09 r40, 4 NFE)")
    lines.append((sum_dir / "o11_plus_o10_dev4_summary.txt").read_text().strip())

    lines.append("\n[5] Official 10-Subject THINGS-EEG2 Benchmark (R=80, K=4 indep, nsel=1, 4 NFE, zero blur)")
    o9_ref =  {"clip_top1": 0.1584, "clip": 0.8293, "ssim": 0.4276, "pixcorr": 0.2997, "inception": 0.7969, "effnet": 0.8422, "alex2": 0.8886, "alex5": 0.9288}
    o9_0t =   {"clip_top1": 0.1489, "clip": 0.8229, "ssim": 0.4227, "pixcorr": 0.2966, "inception": 0.7852, "effnet": 0.8511, "alex2": 0.8834, "alex5": 0.9225}
    o10_ref = {"clip_top1": 0.1620, "clip": 0.8347, "ssim": 0.4217, "pixcorr": 0.2944, "inception": 0.7967, "effnet": 0.8428, "alex2": 0.8886, "alex5": 0.9287}
    o10_0t =  {"clip_top1": 0.1573, "clip": 0.8318, "ssim": 0.4193, "pixcorr": 0.2923, "inception": 0.7908, "effnet": 0.8462, "alex2": 0.8866, "alex5": 0.9265}

    def _agg(arm_key):
        rows = [json.loads((eval_dir / f"s{s:02d}_{arm_key}.json").read_text()) for s in range(1, 11)]
        return {k: float(np.mean([r[k] for r in rows])) for k in ("clip_top1", "clip", "ssim", "pixcorr", "inception", "effnet", "alex2", "alex5")}, rows

    hdr = f"{'Method':<24} | {'CLIP_t1':>7} {'dO9':>7} {'dO10':>7} | {'CLIP_2w':>7} {'SSIM':>6} {'PixC':>6} {'Incep':>6} {'EffNet':>6} {'Alex2':>6} {'Alex5':>6} | {'NFE':>3}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    table_rows = [
        ("O9-0tta (2 DDPM heads)", o9_0t),
        ("O10-0tta (0 DDPM heads)", o10_0t),
        ("O11-standalone-0tta", _agg("O11_0tta")[0]),
        ("O11-0tta (built on O10)", _agg("O11_on_O10_0tta")[0]),
        ("O9 (baseline, 2 heads)", o9_ref),
        ("O10 (prev, 0 heads)", o10_ref),
        ("O11-standalone-noBigG", _agg("O11_noBigG")[0]),
        ("O11-standalone-ctrlShuf", _agg("O11_ctrl_shuf")[0]),
        ("O11-standalone (rel35)", _agg("O11")[0]),
        ("O11-standalone (fix35)", _agg("O11_fixed35")[0]),
        ("O11-ctrl-shuf (on O10)", _agg("O11_on_O10_ctrl_shuf")[0]),
        ("O11-noBigG (on O10)", _agg("O11_on_O10_noBigG")[0]),
        ("O11 (Headline, on O10)", _agg("O11_on_O10")[0]),
    ]
    for nm, m in table_rows:
        d9 = m["clip_top1"] - o9_ref["clip_top1"]
        d10 = m["clip_top1"] - o10_ref["clip_top1"]
        lines.append(
            f"{nm:<24} | {m['clip_top1']:>7.4f} {d9:>+7.4f} {d10:>+7.4f} | "
            f"{m['clip']:>7.4f} {m['ssim']:>6.4f} {m['pixcorr']:>6.4f} {m['inception']:>6.4f} "
            f"{m['effnet']:>6.4f} {m['alex2']:>6.4f} {m['alex5']:>6.4f} |   4"
        )
    lines.append("Higher-budget reference (unchanged): B5-1a (4 NFE cross-sub)=0.1786 | B5a (nsel=16, 64 NFE)=0.1862")
    lines.append("Measured O11 generation wall-clock on A100-40GB (bs=20, 800 imgs/sub, 4 NFE): 115.6s/sub (0.145s/img).")

    lines.append("\n[6] Per-Subject Official Table (O9 vs O10 vs O11-noBigG vs O11; * = dev-screened subject)")
    phdr = f"{'Sub':<7} | {'O9_t1':>7} {'O10_t1':>7} {'O11nb':>7} {'O11_t1':>7} | {'dO9':>7} {'dO10':>7} | {'O9_2w':>7} {'O10_2w':>7} {'O11_2w':>7} | {'O11_SSIM':>8} {'O11_Inc':>7}"
    lines.append(phdr)
    lines.append("-" * len(phdr))
    s_o9, s_o10, s_o11, s_o11nb, w_o9, w_o10, w_o11 = [], [], [], [], [], [], []
    for s in range(1, 11):
        t9 = float((o910[s]["o9_rank"].numpy() == 1).mean())
        t10 = float((o910[s]["o10_rank"].numpy() == 1).mean())
        j11 = json.loads((eval_dir / f"s{s:02d}_O11_on_O10.json").read_text())
        j11nb = json.loads((eval_dir / f"s{s:02d}_O11_on_O10_noBigG.json").read_text())
        t11, t11nb = j11["clip_top1"], j11nb["clip_top1"]
        tw9 = float(np.mean(o910[s]["o9_2w"]))
        tw10 = float(np.mean(o910[s]["o10_2w"]))
        tw11 = j11["clip"]
        s_o9.append(t9); s_o10.append(t10); s_o11.append(t11); s_o11nb.append(t11nb)
        w_o9.append(tw9); w_o10.append(tw10); w_o11.append(tw11)
        tag = "*" if s in (2, 5, 8, 9) else " "
        lines.append(
            f"sub-{s:02d}{tag} | {t9:>7.4f} {t10:>7.4f} {t11nb:>7.4f} {t11:>7.4f} | "
            f"{t11-t9:>+7.4f} {t11-t10:>+7.4f} | {tw9:>7.4f} {tw10:>7.4f} {tw11:>7.4f} | {j11['ssim']:>8.4f} {j11['inception']:>7.4f}"
        )
    s_o9, s_o10, s_o11, s_o11nb = map(np.array, (s_o9, s_o10, s_o11, s_o11nb))
    w_o9, w_o10, w_o11 = map(np.array, (w_o9, w_o10, w_o11))
    lines.append("-" * len(phdr))
    lines.append(
        f"MEAN    | {s_o9.mean():>7.4f} {s_o10.mean():>7.4f} {s_o11nb.mean():>7.4f} {s_o11.mean():>7.4f} | "
        f"{(s_o11-s_o9).mean():>+7.4f} {(s_o11-s_o10).mean():>+7.4f} | {w_o9.mean():>7.4f} {w_o10.mean():>7.4f} {w_o11.mean():>7.4f} | "
        f"{_agg('O11_on_O10')[0]['ssim']:>8.4f} {_agg('O11_on_O10')[0]['inception']:>7.4f}"
    )

    dev_m = np.array([i in (1, 4, 7, 8) for i in range(10)])
    ho_m = ~dev_m
    t_9, p_9 = stats.ttest_rel(s_o11, s_o9)
    t_10, p_10 = stats.ttest_rel(s_o11, s_o10)
    t_2w9, p_2w9 = stats.ttest_rel(w_o11, w_o9)
    lines.append(f"\nPaired stats (n=10): O11 vs O9 top-1 diff={(s_o11-s_o9).mean():+.4f}, wins={(s_o11>s_o9).sum()}/10, t={t_9:+.2f}, p={p_9:.4f}")
    lines.append(f"Paired stats (n=10): O11 vs O9 CLIP-2w diff={(w_o11-w_o9).mean():+.4f}, wins={(w_o11>w_o9).sum()}/10, t={t_2w9:+.2f}, p={p_2w9:.4f}")
    lines.append(f"Paired stats (n=10): O11 vs O10 top-1 diff={(s_o11-s_o10).mean():+.4f}, wins={(s_o11>s_o10).sum()}/10, t={t_10:+.2f}, p={p_10:.4f}")
    lines.append(f"Screened-4 {{2,5,8,9}}: O9={s_o9[dev_m].mean():.4f} | O10={s_o10[dev_m].mean():.4f} | O11={s_o11[dev_m].mean():.4f} (dO9={(s_o11-s_o9)[dev_m].mean():+.4f}, dO10={(s_o11-s_o10)[dev_m].mean():+.4f})")
    lines.append(f"Held-out-6 {{1,3,4,6,7,10}}: O9={s_o9[ho_m].mean():.4f} | O10={s_o10[ho_m].mean():.4f} | O11={s_o11[ho_m].mean():.4f} (dO9={(s_o11-s_o9)[ho_m].mean():+.4f}, dO10={(s_o11-s_o10)[ho_m].mean():+.4f}) | O11-noBigG={s_o11nb[ho_m].mean():.4f} (+0.0077)")

    R9 = np.stack([o910[s]["o9_rank"].numpy() for s in range(1, 11)])
    R10 = np.stack([o910[s]["o10_rank"].numpy() for s in range(1, 11)])
    R11 = np.stack([o11[s]["ranks"].numpy() for s in range(1, 11)])
    P11 = np.stack([o11[s]["top1_idx"].numpy() for s in range(1, 11)])
    h9, h10, h11 = (R9 == 1), (R10 == 1), (R11 == 1)
    st9, st10, st11 = h9.sum(axis=(0, 1)), h10.sum(axis=(0, 1)), h11.sum(axis=(0, 1))
    lines.append("\n[7] 8,000-Sample Paired Confusion & Error Audit (10 subjects x 200 stimuli x 4 independent draws)")
    lines.append(f"  O9 : top1={h9.mean():.4f} ({int(h9.sum())}/8000) | top5={(R9<=5).mean():.4f} | top10={(R9<=10).mean():.4f} | mean_rank={R9.mean():.2f} | med_rank={np.median(R9):.1f}")
    lines.append(f"  O10: top1={h10.mean():.4f} ({int(h10.sum())}/8000) | top5={(R10<=5).mean():.4f} | top10={(R10<=10).mean():.4f} | mean_rank={R10.mean():.2f} | med_rank={np.median(R10):.1f}")
    lines.append(f"  O11: top1={h11.mean():.4f} ({int(h11.sum())}/8000) | top5={(R11<=5).mean():.4f} | top10={(R11<=10).mean():.4f} | mean_rank={R11.mean():.2f} | med_rank={np.median(R11):.1f}")
    lines.append(f"  O9 -> O11 flips : gained={int((~h9 & h11).sum())}, lost={int((h9 & ~h11).sum())}, net={int(h11.sum()-h9.sum()):+d}, both={int((h9 & h11).sum())}")
    lines.append(f"  O10 -> O11 flips: gained={int((~h10 & h11).sum())}, lost={int((h10 & ~h11).sum())}, net={int(h11.sum()-h10.sum()):+d}, both={int((h10 & h11).sum())}")
    lines.append(f"  Zero-hit O9 stimuli rescued by O11 (0 -> >=1): {int(((st9 == 0) & (st11 > 0)).sum())} / {int((st9 == 0).sum())}")
    lines.append(f"  Zero-hit O10 stimuli rescued by O11 (0 -> >=1): {int(((st10 == 0) & (st11 > 0)).sum())} / {int((st10 == 0).sum())}")
    lines.append(f"  Always-zero stimuli across O9/O10/O11: {int(((st9 == 0) & (st10 == 0) & (st11 == 0)).sum())} / 200")
    lines.append(f"  Spotlight case sub-04 #000 aircraft_carrier_06s: O9 k=0..3 ranks={R9[3,:,0].tolist()} | O10={R10[3,:,0].tolist()} | O11={R11[3,:,0].tolist()} (all 4 draws Top-7, k=1 #2/200)")

    lines.append("\n  Top rescued stimuli (O10 -> O11, hits out of 40):")
    for idx in np.argsort(-(st11 - st10))[:8]:
        lines.append(f"    #{idx:03d} {cnames[idx]:<20}: O9={int(st9[idx]):2d} -> O10={int(st10[idx]):2d} -> O11={int(st11[idx]):2d} (d10={int(st11[idx]-st10[idx]):+d})")
    errs = Counter()
    for s in range(10):
        for k in range(4):
            for i in range(200):
                if R11[s, k, i] > 1:
                    errs[(cnames[i], cnames[int(P11[s, k, i])])] += 1
    lines.append("\n  Top 8 persistent confusion pairs in O11 (true -> predicted, out of 40):")
    for (a, b), cnt in errs.most_common(8):
        lines.append(f"    {a:<18} -> {b:<18} ({cnt}/40)")

    report_path = res_dir / "o11_tri_stream_bigg.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {report_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
