"""Meaningful CPU numerical checks. Run on OLD, never local Torch/NumPy."""
import json
import torch
from torch import nn
from torch.nn import functional as F
import eeg_cross_host_gpu_runtime_20261106_v231 as A


def serial(raw, visual, ids, groups, params):
    scores = F.normalize(raw, dim=-1) @ visual.T
    ii = torch.arange(len(raw), device=raw.device)
    own = scores[ii, ids]
    negative = scores.clone()
    negative[ii, ids] = -torch.inf
    same = groups[ids, None] == groups[None, :]
    wi = negative.masked_fill(~same, -torch.inf).argmax(-1)
    gi = negative.argmax(-1)
    margins = torch.cat((own - scores[ii, wi], own - scores[ii, gi]))
    J = torch.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(m, params, retain_graph=True)]).double() for m in margins])
    return margins.detach(), J, wi, gi


def checks(device='cpu'):
    torch.set_num_threads(2)
    results = []
    # Test both dimensions/final partial chunks and deliberately tied negatives.
    for dtype in (torch.float64, torch.float32):
        for dim, group_size, chunk, tied in ((7, 3, 1, False), (9, 2, 4, False), (7, 3, 4, True)):
            torch.manual_seed(1729)
            net = nn.Sequential(nn.Linear(5, 11), nn.GELU(), nn.Linear(11, dim)).to(device=device, dtype=dtype).eval()
            x = torch.randn(5, 5, device=device, dtype=dtype)
            visual = F.normalize(torch.randn(12, dim, device=device, dtype=dtype), dim=-1)
            if tied:
                visual[1] = visual[2] = visual[0]
            ids = torch.arange(5, device=device)
            groups = torch.arange(12, device=device) // group_size
            raw = net(x)
            parameters = tuple(net.parameters())
            m0, j0, wi, gi = serial(raw, visual, ids, groups, parameters)
            m1, j1, info = A.factorized_margin_jacobian(raw, visual, ids, groups, parameters, chunk_size=chunk)
            assert torch.equal(m0, m1)
            assert torch.equal(wi, info['within_negative']) and torch.equal(gi, info['global_negative'])
            error = float((j0-j1).abs().max())
            limit = 2e-12 if dtype == torch.float64 else 2e-6
            assert error <= limit, (dtype, error)
            results.append(dict(case='chain_rule_J', dtype=str(dtype), dimension=dim,
                                chunk=chunk, ties=tied, max_abs_error=error))
    class Hive(nn.Module):
        def __init__(self):
            super().__init__(); self.to_k_ip=nn.Linear(8, 4); self.to_v_ip=nn.Linear(8, 4)
        def forward(self, x): return self.to_k_ip(x) + self.to_v_ip(x)
    class Atm(nn.Module):
        def __init__(self):
            super().__init__(); self.image_projection_layers=nn.ModuleList([nn.Linear(6, 8)])
            self.to_k_ip=nn.ModuleList([nn.Linear(8, 4)]); self.to_v_ip=nn.ModuleList([nn.Linear(8, 4)])
        def forward(self,x):
            z=self.image_projection_layers[0](x)
            return self.to_k_ip[0](z)+self.to_v_ip[0](z)
    for kind, cls, dimension in (('HIVE-layout', Hive, 8), ('ATM-layout', Atm, 6)):
        net=cls().to(device).eval().requires_grad_(False)
        state={k:v.detach().clone() for k,v in net.state_dict().items()}
        originals={n:id(m) for n,m in net.named_modules()}
        x=torch.randn(1, 4, dimension, device=device)
        with torch.no_grad():
            reference=net(x).clone()
            with A.FrozenConditionRequest(net) as cache:
                assert torch.equal(net(x), reference)
                assert torch.equal(net(x), reference) and cache.hits > 0
                x.add_(.01)
                changed=net(x).clone()
            assert torch.equal(net(x), changed)
        assert originals=={n:id(m) for n,m in net.named_modules()}
        assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items())
        # A differentiable upstream input must bypass the detached inference cache.
        x=x.detach().requires_grad_(True)
        baseline=torch.autograd.grad(net(x).sum(), x)[0]
        with A.FrozenConditionRequest(net) as cache:
            actual=torch.autograd.grad(net(x).sum(), x)[0]
            assert torch.equal(baseline,actual) and cache.hits==0 and cache.bypasses>0
        results.append(dict(case='cache_invalidation_restore_and_grad_bypass', layout=kind))
    return dict(status='PASS_CPU_SYNTHETIC_DERIVATIVES_AND_CACHE_CONTRACT_ONLY',
                device=device, tests=results, actual_host_or_GPU_benchmark=False,
                new_training_updates=0, new_images=0)


if __name__ == '__main__':
    print(json.dumps(checks(), allow_nan=False))
