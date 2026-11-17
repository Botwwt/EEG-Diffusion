"""Fast millisecond-locked unified EEG encoder in 70-layer UNet Key-Value space (deep_enc.py).

Combines O8's phase-locked channel x time expressive power (~3s/fold) with three foundational
lower-level innovations:
  1. TargetSNRGate (G_R): within-fold Empirical-Bayes signal/noise gate across (17 ch x 250 timepoints)
     with exact repeat-variance matching from training R_tr=4 to test R_eff (r_eff ~ 72 for IVW 80 repeats).
  2. Unified Multi-Horizon Pyramid Stem: jointly encodes 0-1000 ms (250), 0-700 ms (175), and
     0-480 ms (120) with softmax horizon weights inside a single model (eliminating O8's 14-window zoo).
  3. CenteredKVLoss: prunes O8's conc_w, soft_KL, and 3 aux heads down to 2 foundational terms operating
     directly in the centered 70-layer SDXL-Turbo cross-attention Key-Value subspace L_kv (eff_rank 237/2048).
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import HIVE
from train_flex import pyramid_feats

sys.path.insert(0, str(HIVE))
from main.layers import ResNet1DBlock, ResNet1DLayer  # noqa: E402
from main.models_adapter import IPProjectionModel  # noqa: E402

nz = F.normalize


class TargetSNRGate(nn.Module):
    """Within-fold Empirical-Bayes spatiotemporal reliability gate & repeat-variance scaler."""

    def __init__(self, c_in: int = 17, t_len: int = 250):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, c_in, t_len))
        self.register_buffer("v_s", torch.ones(1, c_in, t_len))
        self.register_buffer("v_eps", torch.zeros(1, c_in, t_len))
        self.register_buffer("w_snr", torch.ones(1, c_in, t_len))

    @torch.no_grad()
    def fit_from_fold(self, x_raw_fold: torch.Tensor):
        """Fit statistics strictly on training-fold EEG [N_tr, R_tr, 17, 250]."""
        n_tr, r_tr, _, _ = x_raw_fold.shape
        assert r_tr >= 2 and n_tr >= 10
        x = x_raw_fold.float()
        x_bar = x.mean(dim=1)  # [N_tr, 17, 250]
        mu = x_bar.mean(dim=0, keepdim=True)
        v_eps = ((x - x_bar.unsqueeze(1)).pow(2).sum(dim=1) / float(r_tr - 1)).mean(dim=0, keepdim=True)
        v_bar = x_bar.var(dim=0, unbiased=True, keepdim=True)
        v_s = (v_bar - v_eps / float(r_tr)).clamp(min=0.0)

        v_s_sm = F.avg_pool1d(v_s, kernel_size=5, stride=1, padding=2)
        v_eps_sm = F.avg_pool1d(v_eps, kernel_size=5, stride=1, padding=2)
        snr_tr = v_s_sm / (v_s_sm + v_eps_sm / float(r_tr) + 1e-6)
        w_raw = torch.sqrt(snr_tr + 0.15)
        w_snr = w_raw / w_raw.mean().clamp_min(1e-6)

        self.mu.copy_(mu.to(self.mu.device))
        self.v_s.copy_(v_s.to(self.v_s.device))
        self.v_eps.copy_(v_eps.to(self.v_eps.device))
        self.w_snr.copy_(w_snr.to(self.w_snr.device))

    def forward(self, x: torch.Tensor, r_eff=4.0) -> torch.Tensor:
        if isinstance(r_eff, torch.Tensor):
            r_val = r_eff.to(dtype=x.dtype, device=x.device).view(-1, 1, 1).clamp_min(1.0)
        else:
            r_val = max(1.0, float(r_eff))
        var_scale = torch.sqrt(
            (self.v_s + self.v_eps / 4.0 + 1e-6) / (self.v_s + self.v_eps / r_val + 1e-6)
        ).clamp(0.5, 2.0)
        return self.w_snr * var_scale * (x - self.mu)


class KVEBPyrEncoder(nn.Module):
    """Single unified millisecond-locked EEG encoder with Empirical-Bayes gate + multi-horizon stems."""

    def __init__(
        self,
        t_len: int = 250,
        c_in: int = 17,
        hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.15,
        multi_horizon: bool = True,
    ):
        super().__init__()
        self.c_in = c_in
        self.t_len = t_len
        self.eb = TargetSNRGate(c_in=c_in, t_len=t_len)
        self.horizons = tuple(h for h in ((250, 175, 120) if multi_horizon else (250,)) if h <= t_len)
        if not self.horizons:
            self.horizons = (t_len,)

        stems = []
        for t_h in self.horizons:
            dim_h = pyramid_feats(torch.zeros(1, c_in, t_h)).shape[1]
            stems.append(nn.Sequential(nn.Linear(dim_h, hidden_size), nn.LayerNorm(hidden_size)))
        self.stems = nn.ModuleList(stems)
        self.horizon_logits = nn.Parameter(torch.zeros(len(self.horizons)))
        self.block = ResNet1DBlock(
            hidden_size=hidden_size,
            intermediate_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, r_eff: float = 4.0) -> torch.Tensor:
        xg = self.eb(x, r_eff=r_eff)
        w_h = F.softmax(self.horizon_logits, dim=0)
        h = 0.0
        for i, t_h in enumerate(self.horizons):
            h = h + w_h[i] * self.stems[i](pyramid_feats(xg[..., :t_h]))
        return self.block(h)

    def last_norm_affine(self):
        norm = self.block.layers[-1].norm
        return norm.weight.detach().float(), norm.bias.detach().float()


class CenteredKVLoss(nn.Module):
    """Centered Contrastive & Cosine Alignment directly in the 70-layer UNet KV subspace."""

    def __init__(self, kv_pack_path: str, device: str = "cuda:0"):
        super().__init__()
        pack = torch.load(kv_pack_path, map_location="cpu", weights_only=True)
        self.adapter = ResNet1DLayer(hidden_size=1024, intermediate_size=4096, norm=False)
        self.adapter.load_state_dict(pack["adapter_state"])
        self.adapter.to(device).eval().requires_grad_(False)

        self.proj = IPProjectionModel(embed_dim=1024, cross_attention_dim=2048, sequence_length=4)
        self.proj.load_state_dict(pack["proj_state"])
        self.proj.to(device).eval().requires_grad_(False)

        self.register_buffer("L_kv_t", pack["L_kv"].float().T.contiguous().to(device))
        self.register_buffer("mu_kv", torch.zeros(1, 8192, device=device))
        self.register_buffer("mu_clip", torch.zeros(1, 1024, device=device))
        self.nu_bar = 32.0

    def project_kv(self, c_cone: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            tok = self.proj(self.adapter(c_cone.float())).float()
            return (tok @ self.L_kv_t).flatten(1)

    @torch.no_grad()
    def calibrate_fold(self, F_tr_all: torch.Tensor, tr_idx: torch.Tensor):
        f_sub = F_tr_all[tr_idx].float()
        self.nu_bar = float(f_sub.norm(dim=1).mean().item())
        u_sub = nz(f_sub, dim=-1)
        self.mu_clip.copy_(u_sub.mean(dim=0, keepdim=True))

        kv_sum = torch.zeros((1, 8192), dtype=torch.float64, device=self.mu_kv.device)
        for s in range(0, len(u_sub), 2048):
            kv_b = self.project_kv(u_sub[s:s + 2048] * self.nu_bar).double()
            kv_sum += kv_b.sum(dim=0, keepdim=True)
        self.mu_kv.copy_((kv_sum / float(len(u_sub))).float())

    def forward(self, c_raw: torch.Tensor, F_batch: torch.Tensor, logit_scale: torch.Tensor):
        u_hat = nz(c_raw.float(), dim=-1)
        with torch.no_grad():
            u_tgt = nz(F_batch.float(), dim=-1)
            kv_tgt = self.project_kv(u_tgt * self.nu_bar)
            y_kv_c = nz(kv_tgt - self.mu_kv, dim=-1)

        kv_hat = self.project_kv(u_hat * self.nu_bar)
        z_kv_c = nz(kv_hat - self.mu_kv, dim=-1)

        scale = logit_scale.exp().clamp(max=100.0)
        lg = scale * (0.5 * (u_hat @ u_tgt.T) + 0.5 * (z_kv_c @ y_kv_c.T))
        tb = torch.arange(len(c_raw), device=c_raw.device)
        loss_nce = 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))
        loss_align = (
            0.35 * (1.0 - (z_kv_c * y_kv_c).sum(-1)).mean()
            + 0.25 * (1.0 - (u_hat * u_tgt).sum(-1)).mean()
        )
        return loss_nce + loss_align


class CenteredGNKVLoss(nn.Module):
    """Centered Contrastive + Alignment Loss jointly in Active-Step UNet Gauss-Newton (L_gn) & 70-layer KV (L_kv) space."""

    def __init__(self, kv_pack_path: str, gn_pack_path: str, device: str = "cuda:0"):
        super().__init__()
        pack = torch.load(kv_pack_path, map_location="cpu", weights_only=True)
        gn_pack = torch.load(gn_pack_path, map_location="cpu", weights_only=True)

        self.adapter = ResNet1DLayer(hidden_size=1024, intermediate_size=4096, norm=False)
        self.adapter.load_state_dict(pack["adapter_state"])
        self.adapter.to(device).eval().requires_grad_(False)

        self.proj = IPProjectionModel(embed_dim=1024, cross_attention_dim=2048, sequence_length=4)
        self.proj.load_state_dict(pack["proj_state"])
        self.proj.to(device).eval().requires_grad_(False)

        self.register_buffer("L_kv_t", pack["L_kv"].float().T.contiguous().to(device))   # [2048, 2048]
        self.register_buffer("L_gn_t", gn_pack["L_gn"].float().T.contiguous().to(device))  # [8192, 512]
        self.register_buffer("mu_kv", torch.zeros(1, 8192, device=device))
        self.register_buffer("mu_gn", torch.zeros(1, gn_pack["L_gn"].shape[0], device=device))
        self.register_buffer("mu_clip", torch.zeros(1, 1024, device=device))
        self.nu_bar = 32.0

    def project_tok(self, c_cone: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            return self.proj(self.adapter(c_cone.float())).float()

    def project_kv_gn(self, c_cone: torch.Tensor):
        with torch.amp.autocast("cuda", enabled=False):
            tok = self.proj(self.adapter(c_cone.float())).float()
            kv = (tok @ self.L_kv_t).flatten(1)
            gn = tok.flatten(1) @ self.L_gn_t
            return kv, gn

    @torch.no_grad()
    def calibrate_fold(self, F_tr_all: torch.Tensor, tr_idx: torch.Tensor):
        f_sub = F_tr_all[tr_idx].float()
        self.nu_bar = float(f_sub.norm(dim=1).mean().item())
        u_sub = nz(f_sub, dim=-1)
        self.mu_clip.copy_(u_sub.mean(dim=0, keepdim=True))

        kv_sum = torch.zeros_like(self.mu_kv, dtype=torch.float64)
        gn_sum = torch.zeros_like(self.mu_gn, dtype=torch.float64)
        for s in range(0, len(u_sub), 2048):
            kv_b, gn_b = self.project_kv_gn(u_sub[s:s + 2048] * self.nu_bar)
            kv_sum += kv_b.double().sum(dim=0, keepdim=True)
            gn_sum += gn_b.double().sum(dim=0, keepdim=True)
        n_f = float(len(u_sub))
        self.mu_kv.copy_((kv_sum / n_f).float())
        self.mu_gn.copy_((gn_sum / n_f).float())

    def forward(self, c_raw: torch.Tensor, F_batch: torch.Tensor, logit_scale: torch.Tensor):
        u_hat = nz(c_raw.float(), dim=-1)
        with torch.no_grad():
            u_tgt = nz(F_batch.float(), dim=-1)
            kv_tgt, gn_tgt = self.project_kv_gn(u_tgt * self.nu_bar)
            y_kv_c = nz(kv_tgt - self.mu_kv, dim=-1)
            y_gn_c = nz(gn_tgt - self.mu_gn, dim=-1)

        kv_hat, gn_hat = self.project_kv_gn(u_hat * self.nu_bar)
        z_kv_c = nz(kv_hat - self.mu_kv, dim=-1)
        z_gn_c = nz(gn_hat - self.mu_gn, dim=-1)

        scale = logit_scale.exp().clamp(max=100.0)
        lg = scale * (0.40 * (u_hat @ u_tgt.T) + 0.40 * (z_gn_c @ y_gn_c.T) + 0.20 * (z_kv_c @ y_kv_c.T))
        tb = torch.arange(len(c_raw), device=c_raw.device)
        loss_nce = 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))
        loss_align = (
            0.30 * (1.0 - (z_gn_c * y_gn_c).sum(-1)).mean()
            + 0.15 * (1.0 - (z_kv_c * y_kv_c).sum(-1)).mean()
            + 0.20 * (1.0 - (u_hat * u_tgt).sum(-1)).mean()
        )
        return loss_nce + loss_align


def sample_sub_repeats(x_raw_b: torch.Tensor, g: torch.Generator):
    """Draw random repeat subsets k in {2, 3, 4} (probs 0.15, 0.35, 0.50) per trial for EB SNR training."""
    bsz = x_raw_b.shape[0]
    dev = x_raw_b.device
    u = torch.rand((bsz,), device=dev, generator=g)
    k_keep = torch.where(u < 0.15, 2, torch.where(u < 0.50, 3, 4))
    ranks = torch.rand((bsz, 4), device=dev, generator=g).argsort(dim=1).argsort(dim=1)
    mask = (ranks < k_keep[:, None]).to(x_raw_b.dtype)
    xb = (x_raw_b * mask[:, :, None, None]).sum(dim=1) / k_keep.to(x_raw_b.dtype).view(bsz, 1, 1)
    return xb, k_keep.float()


class DecoupledHorizonKVEncoder(nn.Module):
    """Unified decoupled-horizon encoder with TargetSNRGate, independent stem linears/norms, shared trunk."""

    def __init__(
        self,
        t_len: int = 250,
        c_in: int = 17,
        hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.15,
        in_drop: float = 0.10,
        horizons=(115, 160, 225),
    ):
        super().__init__()
        self.c_in = c_in
        self.t_len = t_len
        self.horizons = tuple(h for h in horizons if h <= t_len) or (t_len,)
        self.eb = TargetSNRGate(c_in=c_in, t_len=t_len)
        self.in_drop = nn.Dropout(in_drop)

        dims = [pyramid_feats(torch.zeros(1, c_in, t_h)).shape[1] for t_h in self.horizons]
        self.stem_linears = nn.ModuleList([nn.Linear(dim_h, hidden_size) for dim_h in dims])
        self.stem_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in dims])
        self.horizon_logits = nn.Parameter(torch.zeros(len(self.horizons)))
        self.block = ResNet1DBlock(
            hidden_size=hidden_size,
            intermediate_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, r_eff=4.0, return_branches: bool = False):
        xg = self.eb(x, r_eff=r_eff)
        hs = [
            self.stem_norms[i](self.stem_linears[i](self.in_drop(pyramid_feats(xg[..., :t_h]))))
            for i, t_h in enumerate(self.horizons)
        ]
        cs = self.block(torch.cat(hs, dim=0)).chunk(len(self.horizons), dim=0)
        w_h = F.softmax(self.horizon_logits, dim=0)
        b_af = self.block.layers[-1].norm.bias.to(cs[0].dtype)
        d0 = cs[0] - b_af
        d_mix = sum(w_h[i] * (cs[i] - b_af) for i in range(len(cs)))
        scale = d0.norm(dim=-1, keepdim=True) / d_mix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        c_ens = b_af + d_mix * scale
        return (c_ens, cs) if return_branches else c_ens

    def last_norm_affine(self):
        norm = self.block.layers[-1].norm
        return norm.weight.detach().float(), norm.bias.detach().float()


class TangentKVPullbackLoss(nn.Module):
    """Zero-bloat on-cone tangent + 70-layer UNet Key-Value pullback loss with relational KL."""

    def __init__(self, kv_pack_path: str, device: str = "cuda:0"):
        super().__init__()
        pack = torch.load(kv_pack_path, map_location="cpu", weights_only=True)
        self.adapter = ResNet1DLayer(hidden_size=1024, intermediate_size=4096, norm=False)
        self.adapter.load_state_dict(pack["adapter_state"])
        self.adapter.to(device).eval().requires_grad_(False)

        self.proj = IPProjectionModel(embed_dim=1024, cross_attention_dim=2048, sequence_length=4)
        self.proj.load_state_dict(pack["proj_state"])
        self.proj.to(device).eval().requires_grad_(False)

        self.register_buffer("L_kv_t", pack["L_kv"].float().T.contiguous().to(device))
        self.register_buffer("mu_kv", torch.zeros(1, 8192, device=device))
        self.register_buffer("u_tr", torch.zeros(1, 1024, device=device))
        self.register_buffer("mu_clip", torch.zeros(1, 1024, device=device))
        self.nu_bar = 32.0
        self.alpha_tr = 0.616
        self.beta_tr = 0.788

    def project_kv(self, c_cone: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            tok = self.proj(self.adapter(c_cone.float())).float()
            return (tok @ self.L_kv_t).flatten(1)

    @torch.no_grad()
    def calibrate_fold(self, F_tr_all: torch.Tensor, tr_idx: torch.Tensor):
        f_sub = F_tr_all[tr_idx].float()
        self.nu_bar = float(f_sub.norm(dim=1).mean().item())
        u_sub = nz(f_sub, dim=-1)
        mu_u = u_sub.mean(dim=0, keepdim=True)
        self.mu_clip.copy_(mu_u)
        self.alpha_tr = float(mu_u.norm().item())
        self.beta_tr = float(math.sqrt(max(1e-8, 1.0 - self.alpha_tr ** 2)))
        self.u_tr.copy_(mu_u / max(self.alpha_tr, 1e-8))

        kv_sum = torch.zeros((1, 8192), dtype=torch.float64, device=self.mu_kv.device)
        for s in range(0, len(f_sub), 2048):
            kv_sum += self.project_kv(f_sub[s:s + 2048]).double().sum(dim=0, keepdim=True)
        self.mu_kv.copy_((kv_sum / float(len(f_sub))).float())

    def cone_from_raw(self, c_raw: torch.Tensor):
        cu = nz(c_raw.float(), dim=-1)
        cu_c = cu - cu.mean(dim=0, keepdim=True)
        d_hat = nz(cu_c - (cu_c @ self.u_tr.T) * self.u_tr, dim=-1)
        c_cone = self.nu_bar * nz(self.alpha_tr * self.u_tr + self.beta_tr * d_hat, dim=-1)
        return cu, d_hat, c_cone

    def single_loss(
        self,
        c_raw: torch.Tensor,
        u_tgt: torch.Tensor,
        d_tgt: torch.Tensor,
        y_kv_c: torch.Tensor,
        p_teach: torch.Tensor,
        scale: torch.Tensor,
    ):
        cu, d_hat, c_cone = self.cone_from_raw(c_raw)
        z_kv_c = nz(self.project_kv(c_cone) - self.mu_kv, dim=-1)
        lg = scale * (0.55 * (d_hat @ d_tgt.T) + 0.45 * (z_kv_c @ y_kv_c.T))
        tb = torch.arange(len(c_raw), device=c_raw.device)
        loss_nce = 0.5 * (F.cross_entropy(lg, tb) + F.cross_entropy(lg.T, tb))
        loss_kl = 0.5 * (
            F.kl_div(F.log_softmax(lg, dim=-1), p_teach, reduction="batchmean")
            + F.kl_div(F.log_softmax(lg.T, dim=-1), p_teach, reduction="batchmean")
        )
        loss_align = (
            0.35 * (1.0 - (d_hat * d_tgt).sum(-1)).mean()
            + 0.25 * (1.0 - (z_kv_c * y_kv_c).sum(-1)).mean()
            + 0.10 * ((cu @ self.u_tr.T) - 0.18).pow(2).mean()
        )
        return 0.75 * loss_nce + 0.25 * loss_kl + loss_align

    def forward(self, c_ens: torch.Tensor, cs_branches, F_batch: torch.Tensor, logit_scale: torch.Tensor):
        with torch.amp.autocast("cuda", enabled=False):
            fb = F_batch.float()
            with torch.no_grad():
                u_tgt = nz(fb, dim=-1)
                d_tgt = nz(u_tgt - (u_tgt @ self.u_tr.T) * self.u_tr, dim=-1)
                y_kv_c = nz(self.project_kv(fb) - self.mu_kv, dim=-1)
                p_teach = F.softmax((0.55 * (d_tgt @ d_tgt.T) + 0.45 * (y_kv_c @ y_kv_c.T)) / 0.07, dim=-1)
            scale = logit_scale.float().exp().clamp(max=100.0)
            l_ens = self.single_loss(c_ens, u_tgt, d_tgt, y_kv_c, p_teach, scale)
            l_br = sum(self.single_loss(cm, u_tgt, d_tgt, y_kv_c, p_teach, scale) for cm in cs_branches) / float(len(cs_branches))
            return 0.50 * l_ens + 0.50 * l_br
