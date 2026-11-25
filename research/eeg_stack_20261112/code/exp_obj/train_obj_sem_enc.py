"""Cross-Brain Semantic-Visual Decoupled Object Encoder (ObjSemEnc) with Cross-Exemplar Concept Alignment."""
import os, sys, gc, math, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAD = Path(os.environ.get("RAD_ROOT", "/path/to/rad"))
for sub in ("code/exp_w7", "code/exp_w6", "code/rad", "code/exp", "code/Brain-HIVE"):
    p = str(RAD / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from main.layers import ResNet1DBlock
from common import EMB_KEYS, PREP
from deep_enc import TargetSNRGate as EarlyBurstStat, sample_sub_repeats
from train_flex import ivw_mean, pyramid_feats, shift_batch
nz = F.normalize
ALL_SUBS = tuple(range(1, 11))

OUT_ROOT = RAD / "runs/obj_bench/obj_sem_enc"

def late_semantic_feats(x: torch.Tensor) -> torch.Tensor:
    """Multi-scale features from late semantic ERP window (160-880ms: idx 40..220)."""
    return pyramid_feats(x[..., 40:220])

class CrossSubObjSemEncoder(nn.Module):
    def __init__(self, n_subs: int = 10, hidden_size: int = 1024, num_layers: int = 3, dropout: float = 0.25, in_drop: float = 0.15):
        super().__init__()
        self.horizons = (250, 180, 120)
        dims = [pyramid_feats(torch.zeros(1, 17, t_h)).shape[1] for t_h in self.horizons]
        dim_late = late_semantic_feats(torch.zeros(1, 17, 250)).shape[1]
        self.all_dims = dims + [dim_late]  # 4 streams: 3 full-onset horizons + 1 late-semantic ERP stream

        self.ebs = nn.ModuleList([EarlyBurstStat() for _ in range(n_subs)])
        self.in_drop = nn.Dropout(in_drop)
        self.sub_stems = nn.ModuleList([
            nn.ModuleList([nn.Linear(d, hidden_size) for d in self.all_dims]) for _ in range(n_subs)
        ])
        self.stem_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in self.all_dims])
        self.horizon_logits = nn.Parameter(torch.zeros(len(self.all_dims)))
        self.sub_emb = nn.Embedding(n_subs, hidden_size)
        nn.init.zeros_(self.sub_emb.weight)

        self.block = ResNet1DBlock(hidden_size=hidden_size, intermediate_size=hidden_size * 2, num_layers=num_layers, dropout=dropout)
        self.head_h14_txt = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1024))
        self.head_bigg_txt = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1280))
        self.head_h14_vis = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1024))
        self.head_cat27 = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 27))

    def forward_sub(self, x: torch.Tensor, sid: int, r_eff=4.0):
        xg = self.ebs[sid](x, r_eff=r_eff)
        feats = [pyramid_feats(xg[..., :t_h]) for t_h in self.horizons] + [late_semantic_feats(xg)]
        hs = [
            self.stem_norms[i](self.sub_stems[sid][i](self.in_drop(feats[i]))) + self.sub_emb.weight[sid]
            for i in range(len(self.all_dims))
        ]
        cs = self.block(torch.cat(hs, dim=0)).chunk(len(self.all_dims), dim=0)
        w_h = F.softmax(self.horizon_logits, dim=0)
        b_af = self.block.layers[-1].norm.bias.to(cs[0].dtype)
        d0 = cs[0] - b_af
        d_mix = sum(w_h[i] * (cs[i] - b_af) for i in range(len(cs)))
        scale = d0.norm(dim=-1, keepdim=True) / d_mix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        c_ens = b_af + d_mix * scale
        return (
            self.head_h14_txt(c_ens),
            self.head_bigg_txt(c_ens),
            self.head_h14_vis(c_ens),
            self.head_cat27(c_ens),
        )

def supcon_concept_loss(pred_u, tgt_u, cid_b, scale, margin=None):
    """Multi-positive concept-level symmetric InfoNCE: same training concept = positive!"""
    # pred_u, tgt_u: [B, D] normalized
    logits = (pred_u @ tgt_u.T) * scale
    if margin is not None:
        logits = logits - margin
    same_c = (cid_b[:, None] == cid_b[None, :]).float()  # [B, B]
    # Forward: pred -> tgt (multi-positive over all same-concept targets in batch)
    pos_w_f = same_c / same_c.sum(dim=1, keepdim=True).clamp_min(1.0)
    l_f = -(pos_w_f * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    # Backward: tgt -> pred
    pos_w_b = same_c / same_c.sum(dim=0, keepdim=True).clamp_min(1.0)
    l_b = -(pos_w_b * F.log_softmax(logits, dim=0)).sum(dim=0).mean()
    return 0.5 * (l_f + l_b)

def xbrain_sem_loss(pa, pb, cid_b, scale, margin=None):
    ua = nz(pa.float() - pa.float().mean(0, keepdim=True), dim=-1)
    ub = nz(pb.float() - pb.float().mean(0, keepdim=True), dim=-1)
    return supcon_concept_loss(ua, ub, cid_b, scale, margin=margin)

def train_seed(seed: int, dev: str = "cuda:0", epochs: int = 160, xex_prob: float = 0.5, tag: str = "objsem_v1"):
    out_d = OUT_ROOT / tag / f"s{seed}"
    if (out_d / "model_fp16.pt").exists() and (out_d / "test_0tta.pt").exists():
        print(f"[CACHED] {tag} s{seed}", flush=True)
        return
    out_d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    tr_cid = tb["tr_img2cid"].long().to(dev)  # [16540] in 0..1626
    n_tr = len(tr_cid)

    # Build concept -> list of image indices for cross-exemplar pairing!
    cid_np = tr_cid.cpu().numpy()
    c2imgs = {}
    for idx_i, c_i in enumerate(cid_np):
        c2imgs.setdefault(int(c_i), []).append(idx_i)
    # Pre-build padded array [16540, 10] of same-concept exemplars
    max_k = max(len(v) for v in c2imgs.values())
    same_pool = np.zeros((n_tr, max_k), dtype=np.int64)
    same_len = np.zeros(n_tr, dtype=np.int64)
    for idx_i, c_i in enumerate(cid_np):
        v = c2imgs[int(c_i)]
        same_pool[idx_i, :len(v)] = v
        same_len[idx_i] = len(v)

    # Strictly train-only centered targets!
    h14_tr_raw = tb["h14_train"][tr_cid.cpu()].float()    # [16540, 1024]
    bigg_tr_raw = tb["bigg_train"][tr_cid.cpu()].float()  # [16540, 1280]
    mu_h14_t = h14_tr_raw.mean(0, keepdim=True)
    mu_bigg_t = bigg_tr_raw.mean(0, keepdim=True)
    T_h14 = nz(h14_tr_raw - mu_h14_t, dim=-1).to(dev)
    T_bigg = nz(bigg_tr_raw - mu_bigg_t, dim=-1).to(dev)

    # Coarse-27 soft distribution on train concepts ONLY
    cat27_raw = tb["h14_cat27"].float()  # [27, 1024]
    C27_soft = F.softmax((h14_tr_raw @ cat27_raw.T) / 0.04, dim=-1).to(dev)

    # Concept-level H14 visual prototype + instance residual (train ONLY)
    emb_tr = torch.load(PREP / "emb_train.pt", map_location="cpu", weights_only=True)
    H14_img = nz(emb_tr[EMB_KEYS[0]].float(), dim=-1)  # [16540, 1024]
    n_uniq = len(tb["uniq_train"])
    psum = torch.zeros(n_uniq, 1024); pcnt = torch.zeros(n_uniq, 1)
    psum.index_add_(0, tr_cid.cpu(), H14_img)
    pcnt.index_add_(0, tr_cid.cpu(), torch.ones(n_tr, 1))
    H14_proto = nz((psum / pcnt.clamp_min(1.0))[tr_cid.cpu()], dim=-1)
    # Blend 0.65 concept prototype + 0.35 instance visual (centered)
    V_mix = 0.65 * H14_proto + 0.35 * H14_img
    mu_v = V_mix.mean(0, keepdim=True)
    V_h14 = nz(V_mix - mu_v, dim=-1).to(dev)
    del emb_tr, H14_img, H14_proto, V_mix

    x_tr_all = [torch.from_numpy(np.load(PREP / f"sub-{s:02d}_train.npy")).half() for s in ALL_SUBS]

    model = CrossSubObjSemEncoder(n_subs=10, hidden_size=1024, num_layers=3, dropout=0.25, in_drop=0.15).to(dev)
    for sid in range(10):
        model.ebs[sid].fit_from_fold(x_tr_all[sid].float().to(dev))

    ls_h = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_b = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_v = nn.Parameter(torch.tensor(2.6592, device=dev))
    ls_x = nn.Parameter(torch.tensor(2.6592, device=dev))
    extra = [("ls_h", ls_h), ("ls_b", ls_b), ("ls_v", ls_v), ("ls_x", ls_x)]
    dec, ndec = [], []
    for n, p_ in list(model.named_parameters()) + extra:
        (ndec if (n.endswith("bias") or "norm" in n or n.startswith("ls") or "emb" in n or "logit" in n) else dec).append(p_)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.05}, {"params": ndec, "weight_decay": 0.0}], lr=5e-4, betas=(0.9, 0.995))

    bs = 1024
    spr = math.ceil(n_tr / bs)
    total = spr * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * s / total))))
    g = torch.Generator(device=dev).manual_seed(1000 + seed)
    rng = np.random.default_rng(1000 + seed)

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=dev, generator=g)
        sub_order = rng.permutation(10)
        for b in range(spr):
            idx = perm[b * bs:(b + 1) * bs]
            idx_cpu = idx.cpu().numpy()
            c_b = tr_cid[idx]
            sid_a = int(sub_order[(2 * b) % 10])
            sid_b = int(sub_order[(2 * b + 1) % 10])
            if sid_b == sid_a:
                sid_b = (sid_a + 1) % 10

            # Cross-exemplar pairing for subject b with probability xex_prob!
            if xex_prob > 0:
                use_xex = rng.random(len(idx_cpu)) < xex_prob
                r_pos = (rng.random(len(idx_cpu)) * same_len[idx_cpu]).astype(np.int64)
                alt_idx = same_pool[idx_cpu, r_pos]
                idx_b_cpu = np.where(use_xex, alt_idx, idx_cpu)
            else:
                idx_b_cpu = idx_cpu

            xa, ka = sample_sub_repeats(x_tr_all[sid_a][idx_cpu].float().to(dev), g)
            xb, kb = sample_sub_repeats(x_tr_all[sid_b][idx_b_cpu].float().to(dev), g)
            xa, xb = shift_batch(xa, 2, g), shift_batch(xb, 2, g)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                sh = ls_h.float().exp().clamp(max=100.0)
                sb = ls_b.float().exp().clamp(max=100.0)
                sv = ls_v.float().exp().clamp(max=100.0)
                sx = ls_x.float().exp().clamp(max=100.0)

                with torch.no_grad():
                    th = T_h14[idx]; tb_t = T_bigg[idx]; vh = V_h14[idx]; c27_t = C27_soft[idx]
                    diff_c = (c_b[:, None] != c_b[None, :]).float()
                    sim_t = th @ th.T
                    sbar = (sim_t * diff_c).sum() / diff_c.sum().clamp_min(1.0)
                    mh = 0.15 * sh.detach() * F.relu(sim_t - sbar) * diff_c
                    mb = 0.15 * sb.detach() * F.relu(sim_t - sbar) * diff_c
                    mx = 0.15 * sx.detach() * F.relu(sim_t - sbar) * diff_c

                ha, ba, va, ca27 = model.forward_sub(xa, sid_a, r_eff=ka)
                hb, bb, vb, cb27 = model.forward_sub(xb, sid_b, r_eff=kb)

                def _one_sub(h_, b_, v_, c27_, use_vis=True):
                    hu = nz(h_.float() - h_.float().mean(0, keepdim=True), dim=-1)
                    bu = nz(b_.float() - b_.float().mean(0, keepdim=True), dim=-1)
                    vu = nz(v_.float() - v_.float().mean(0, keepdim=True), dim=-1)
                    l_ht = supcon_concept_loss(hu, th, c_b, sh, margin=mh) + 0.25 * (1.0 - (hu * th).sum(-1)).mean()
                    l_bt = supcon_concept_loss(bu, tb_t, c_b, sb, margin=mb) + 0.25 * (1.0 - (bu * tb_t).sum(-1)).mean()
                    l_vt = supcon_concept_loss(vu, vh, c_b, sv) if use_vis else 0.0
                    l_c27 = F.kl_div(F.log_softmax(c27_.float(), dim=-1), c27_t, reduction="batchmean")
                    return 0.40 * l_ht + 0.40 * l_bt + 0.20 * l_vt + 0.20 * l_c27

                # Note: for cross-exemplar b, concept text & cat27 are IDENTICAL to idx! Vis matches idx_b!
                l_a = _one_sub(ha, ba, va, ca27, use_vis=True)
                idx_b_dev = torch.from_numpy(idx_b_cpu).to(dev)
                v_b_tgt = V_h14[idx_b_dev]
                vbu = nz(vb.float() - vb.float().mean(0, keepdim=True), dim=-1)
                l_b = _one_sub(hb, bb, vb, cb27, use_vis=False) + 0.20 * supcon_concept_loss(vbu, v_b_tgt, c_b, sv)

                l_xb = 0.50 * xbrain_sem_loss(ha, hb, c_b, sx, margin=mx) + 0.50 * xbrain_sem_loss(ba, bb, c_b, sx, margin=mx)
                loss = 0.5 * (l_a + l_b) + 0.30 * l_xb

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec + ndec, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    train_s = time.time() - t0
    model.eval()
    pack = {"train_s": train_s, "mu_h14_t": mu_h14_t, "mu_bigg_t": mu_bigg_t, "mu_v": mu_v}
    with torch.no_grad():
        for s in ALL_SUBS:
            sid = s - 1
            x_te = torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")).float().to(dev)
            full_in = ivw_mean(x_te)
            h_f, b_f, v_f, c27_f = model.forward_sub(full_in, sid, r_eff=72.0)
            pack[f"sub-{s:02d}"] = {
                "h14_txt": h_f.float().cpu(), "bigg_txt": b_f.float().cpu(),
                "h14_vis": v_f.float().cpu(), "cat27": c27_f.float().cpu(),
            }
            del x_te, full_in
    torch.save({k: (v.half().cpu() if v.is_floating_point() else v.cpu()) for k, v in model.state_dict().items()}, out_d / "model_fp16.pt")
    torch.save(pack, out_d / "test_0tta.pt")

    # Quick diagnostic on 0-TTA semantic decoding across all 10 subs!
    h14_e = nz(tb["h14_eval1823"].float() - mu_h14_t, dim=-1)
    bg_e = nz(tb["bigg_eval1823"].float() - mu_bigg_t, dim=-1)
    diag = torch.arange(200)
    r200, r1823 = [], []
    for s in ALL_SUBS:
        d = pack[f"sub-{s:02d}"]
        hq = nz(d["h14_txt"] - d["h14_txt"].mean(0, keepdim=True), dim=-1)
        bq = nz(d["bigg_txt"] - d["bigg_txt"].mean(0, keepdim=True), dim=-1)
        sh, sb = hq @ h14_e.T, bq @ bg_e.T
        def z(m): return (m - m.mean(-1, keepdim=True)) / (m.std(-1, keepdim=True) + 1e-6)
        se = 0.5 * z(sh) + 0.5 * z(sb)
        s200 = se[:, :200]
        r200.append(((s200.argmax(-1)==diag).float().mean().item()*100, (s200.topk(5,-1).indices==diag[:,None]).any(-1).float().mean().item()*100))
        r1823.append(((se.argmax(-1)==diag).float().mean().item()*100, (se.topk(5,-1).indices==diag[:,None]).any(-1).float().mean().item()*100))
    a2, aA = np.array(r200), np.array(r1823)
    print(f"[{tag} s{seed}] {train_s:.1f}s | 0-TTA 10-sub: 200w={a2[:,0].mean():.2f}/{a2[:,1].mean():.2f}% (s10={a2[9,0]:.1f}/{a2[9,1]:.1f}) | 1823w={aA[:,0].mean():.2f}/{aA[:,1].mean():.2f}% (s10={aA[9,0]:.1f}/{aA[9,1]:.1f})", flush=True)

def run_tta_and_ensemble(tag: str, seeds: list, dev: str = "cuda:0", steps: int = 50):
    tb = torch.load(RAD / "runs/obj_bench/text_banks/text_banks_laion.pt", map_location="cpu", weights_only=False)
    p0 = torch.load(OUT_ROOT / tag / f"s{seeds[0]}" / "test_0tta.pt", map_location="cpu", weights_only=False)
    mu_h14_t, mu_bigg_t = p0["mu_h14_t"], p0["mu_bigg_t"]
    out_ens = OUT_ROOT / tag / "ens_tta_all10.pt"

    res_all = {"mu_h14_t": mu_h14_t, "mu_bigg_t": mu_bigg_t}
    tgt = torch.arange(200, device=dev)
    for s in ALL_SUBS:
        sid = s - 1
        x_te = torch.from_numpy(np.load(PREP / f"sub-{s:02d}_test.npy")).float().to(dev)
        full_in = ivw_mean(x_te)
        hs, bs_, vs, cs = [], [], [], []
        for sd in seeds:
            ckpt = torch.load(OUT_ROOT / tag / f"s{sd}" / "model_fp16.pt", map_location="cpu", weights_only=True)
            ckpt = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt.items()}
            m = CrossSubObjSemEncoder(n_subs=10, hidden_size=1024, num_layers=3, dropout=0.25, in_drop=0.0).to(dev)
            m.load_state_dict(ckpt)
            m.eval().requires_grad_(False)
            for p in m.sub_stems[sid].parameters():
                p.requires_grad_(True)
            opt = torch.optim.Adam(m.sub_stems[sid].parameters(), lr=1e-4)
            g = torch.Generator(device=dev).manual_seed(9000 + sid * 37 + sd)
            with torch.no_grad():
                h0, b0, v0, _ = m.forward_sub(full_in, sid, r_eff=72.0)
            for _ in range(steps):
                perm = torch.rand(200, 80, device=dev, generator=g).argsort(1)
                xa = ivw_mean(torch.gather(x_te, 1, perm[:, :40, None, None].expand(-1, -1, 17, 250)))
                xb = ivw_mean(torch.gather(x_te, 1, perm[:, 40:, None, None].expand(-1, -1, 17, 250)))
                ha, ba, va, _ = m.forward_sub(xa, sid, r_eff=36.0)
                hb, bb, vb, _ = m.forward_sub(xb, sid, r_eff=36.0)
                def _c(z): return nz(z.float() - z.float().mean(0, keepdim=True), dim=-1)
                lg = (0.45 * (_c(ha) @ _c(hb).T) + 0.45 * (_c(ba) @ _c(bb).T) + 0.10 * (_c(va) @ _c(vb).T)) / 0.1
                hf, bf, vf, _ = m.forward_sub(full_in, sid, r_eff=72.0)
                anc = 0.5 * (((hf - h0).pow(2).sum(1) / h0.pow(2).sum(1).clamp_min(1e-6)).mean() +
                             ((bf - b0).pow(2).sum(1) / b0.pow(2).sum(1).clamp_min(1e-6)).mean())
                (0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt)) + 1.0 * anc).backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                hf, bf, vf, cf = m.forward_sub(full_in, sid, r_eff=72.0)
                hs.append(nz(hf.float() - hf.float().mean(0, keepdim=True), dim=-1).cpu())
                bs_.append(nz(bf.float() - bf.float().mean(0, keepdim=True), dim=-1).cpu())
                vs.append(nz(vf.float() - vf.float().mean(0, keepdim=True), dim=-1).cpu())
                cs.append(F.softmax(cf.float(), dim=-1).cpu())
        res_all[f"sub-{s:02d}"] = {
            "h14_txt": nz(torch.stack(hs).mean(0), dim=-1),
            "bigg_txt": nz(torch.stack(bs_).mean(0), dim=-1),
            "h14_vis": nz(torch.stack(vs).mean(0), dim=-1),
            "cat27_prob": torch.stack(cs).mean(0),
        }
    torch.save(res_all, out_ens)

    # Report 10-subject zero-shot text accuracy of ObjSemEnc ensemble!
    h14_e = nz(tb["h14_eval1823"].float() - mu_h14_t, dim=-1)
    bg_e = nz(tb["bigg_eval1823"].float() - mu_bigg_t, dim=-1)
    h14_cv = nz(tb["h14_cv_open"].float() - mu_h14_t, dim=-1)
    bg_cv = nz(tb["bigg_cv_open"].float() - mu_bigg_t, dim=-1)
    diag = torch.arange(200)
    r200, r1823 = [], []
    for s in ALL_SUBS:
        d = res_all[f"sub-{s:02d}"]
        def z(m): return (m - m.mean(-1, keepdim=True)) / (m.std(-1, keepdim=True) + 1e-6)
        se = 0.5 * z(d["h14_txt"] @ h14_e.T) + 0.5 * z(d["bigg_txt"] @ bg_e.T)
        s200 = se[:, :200]
        r200.append(((s200.argmax(-1)==diag).float().mean().item()*100, (s200.topk(5,-1).indices==diag[:,None]).any(-1).float().mean().item()*100))
        r1823.append(((se.argmax(-1)==diag).float().mean().item()*100, (se.topk(5,-1).indices==diag[:,None]).any(-1).float().mean().item()*100))
        if s == 10:
            scv = 0.5 * z(d["h14_txt"] @ h14_cv.T) + 0.5 * z(d["bigg_txt"] @ bg_cv.T)
            print("\n  [sub-10 ObjSemEnc -> ZERO-LEAKAGE CV-OpenVocab (8,141 words) Top-3 for #000..#011]:")
            for i in range(12):
                top3 = [tb["cv_open"][j] for j in scv[i].topk(3).indices.tolist()]
                topc = tb["cat27"][d["cat27_prob"][i].argmax().item()]
                print(f"    #{i:03d} GT={tb['test_names'][i]:18s} | cat27={topc:18s} | CV-8141={top3}")
    a2, aA = np.array(r200), np.array(r1823)
    print(f"\n=== [{tag}] 5-SEED + TTA 10-SUB MEAN ===", flush=True)
    print(f"  EEG -> Text 200w : Top1={a2[:,0].mean():.2f}% Top5={a2[:,1].mean():.2f}% (sub10={a2[9,0]:.1f}/{a2[9,1]:.1f})")
    print(f"  EEG -> Text 1823w: Top1={aA[:,0].mean():.2f}% Top5={aA[:,1].mean():.2f}% (sub10={aA[9,0]:.1f}/{aA[9,1]:.1f})", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--tag", type=str, default="objsem_v1")
    ap.add_argument("--xex_prob", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for sd in seeds:
        train_seed(sd, dev="cuda:0", epochs=args.epochs, xex_prob=args.xex_prob, tag=args.tag)
    if args.tta:
        run_tta_and_ensemble(args.tag, seeds, dev="cuda:0")
