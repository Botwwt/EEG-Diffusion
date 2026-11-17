"""Exact closed-form 70-layer SDXL-Turbo IP-Adapter Key-Value metric (kv_metric.py).

Constructs the exact quadratic operator induced by all L=70 IPAttnProcessor2_0
cross-attention layers of Brain-HIVE's frozen SDXL-Turbo UNet adapter:
    sum_{l=1..70} (1 / d_l) * (||K_l(U1) - K_l(U2)||_F^2 + scale_l^2 * ||V_l(U1) - V_l(U2)||_F^2)
    == trace_scale * ||(U1 - U2) @ L_kv.T||_F^2,
where U = P(A(c)) in R^{B x 4 x 2048}, M_kv = L_kv.T @ L_kv is trace-normalized,
and zero test data or evaluation networks are touched.
"""
import argparse
import sys

import torch
import torch.nn.functional as F

from common import HIVE, PREP, PRIOR, write_json

sys.path.insert(0, str(HIVE))
from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(PREP / "unet_kv_metric.pt"))
    args = p.parse_args()

    print("Loading frozen FusionEncoder.adapter, IPProjectionModel, and IPAttnAdapterModel...", flush=True)
    attn = IPAttnAdapterModel.from_pretrained(str(PRIOR), subfolder="attn_adapter").eval()
    proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder="proj").eval()
    adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").adapter.eval()

    procs = list(attn.attn_processors.values())
    n_layers = len(procs)
    assert n_layers > 0, "Expected non-empty IPAttnAdapterModel.attn_processors"

    M_kv = torch.zeros((2048, 2048), dtype=torch.float64)
    scales = set()
    dims = []
    for proc in procs:
        assert hasattr(proc, "to_k_ip") and hasattr(proc, "to_v_ip")
        assert proc.to_k_ip.bias is None and proc.to_v_ip.bias is None, "Expected bias=False in IPAttnProcessor2_0"
        wk = proc.to_k_ip.weight.detach().double()  # [d_l, 2048]
        wv = proc.to_v_ip.weight.detach().double()  # [d_l, 2048]
        d_l = int(wk.shape[0])
        sc = float(getattr(proc, "scale", 1.0))
        scales.add(round(sc, 6))
        dims.append(d_l)
        M_kv += (wk.T @ wk + (sc ** 2) * (wv.T @ wv)) / float(d_l * n_layers)

    M_kv = 0.5 * (M_kv + M_kv.T)
    trace_scale = float(torch.trace(M_kv).item() / 2048.0)
    assert trace_scale > 0.0
    M_norm = M_kv / trace_scale
    eigvals, Q = torch.linalg.eigh(M_norm)
    eigvals = torch.flip(eigvals, dims=[0])
    Q = torch.flip(Q, dims=[1])
    eigvals_pos = eigvals.clamp(min=1e-8)
    L_kv = torch.diag(torch.sqrt(eigvals_pos)) @ Q.T  # M_norm == L_kv.T @ L_kv

    # Effective participation ratio (intrinsic rank in 2048-d UNet cross-attn space)
    ep = eigvals.clamp(min=0.0)
    eff_rank = float((ep.sum() ** 2) / ep.pow(2).sum().clamp_min(1e-12))

    # Algebraic equivalence self-check against explicit 70-layer UNet K/V projections
    g = torch.Generator().manual_seed(20261117)
    c1 = F.normalize(torch.randn((8, 1024), generator=g), dim=1) * 32.0
    c2 = F.normalize(torch.randn((8, 1024), generator=g), dim=1) * 32.0
    with torch.no_grad():
        u1 = proj(adapter(c1)).double()  # [8, 4, 2048]
        u2 = proj(adapter(c2)).double()
        du = u1 - u2
        emp_total = 0.0
        for proc in procs:
            sc = float(getattr(proc, "scale", 1.0))
            dk = proc.to_k_ip(du.float()).double()
            dv = sc * proc.to_v_ip(du.float()).double()
            d_l = dk.shape[-1]
            emp_total += float((dk.pow(2).sum() + dv.pow(2).sum()).item()) / float(d_l * n_layers)
        quad_total = float((du @ L_kv.T).pow(2).sum().item()) * trace_scale
    rel_err = abs(emp_total - quad_total) / max(abs(emp_total), 1e-12)
    print(f"Verified 70-layer KV metric: layers={n_layers}, scales={sorted(scales)}, "
          f"eff_rank={eff_rank:.1f}/2048, trace_scale={trace_scale:.6f}, rel_err={rel_err:.2e}", flush=True)
    assert rel_err < 1e-4, f"KV metric mismatch: rel_err={rel_err}"

    payload = {
        "M_kv": M_norm.float().cpu(),
        "L_kv": L_kv.float().cpu(),
        "eigvals": eigvals.float().cpu(),
        "trace_scale": trace_scale,
        "eff_rank": eff_rank,
        "rel_err": rel_err,
        "n_layers": n_layers,
        "scales": sorted(scales),
        "dim_counts": {str(d): dims.count(d) for d in sorted(set(dims))},
        "proj_state": {k: v.float().cpu() for k, v in proj.state_dict().items()},
        "adapter_state": {k: v.float().cpu() for k, v in adapter.state_dict().items()},
    }
    torch.save(payload, args.out)
    write_json(str(args.out) + ".json", {
        "n_layers": n_layers,
        "scales": sorted(scales),
        "dim_counts": payload["dim_counts"],
        "trace_scale": trace_scale,
        "eff_rank": eff_rank,
        "rel_err": rel_err,
        "top10_eigvals": [float(x) for x in eigvals[:10].tolist()],
        "bottom5_eigvals": [float(x) for x in eigvals[-5:].tolist()],
    })
    print("Saved:", args.out, flush=True)


if __name__ == "__main__":
    main()
