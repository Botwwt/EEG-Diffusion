"""Train and score an all-candidate, support-conditioned EEG comparator on generic objects."""

import argparse
from pathlib import Path

import numpy as np
import torch
from prepare_support_conditioned_object import LOCK, PUBLIC, context, incl
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

MODES = ("pooled", "conditioned")
SPLITS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def standard(x):
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-8)


def anchor(q, s):
    m = F.normalize(s.mean(2), dim=-1)
    return standard(torch.einsum("bqd,bkd->bqk", F.normalize(q, dim=-1), m))


class Matcher(nn.Module):
    def __init__(self, pooled):
        super().__init__()
        self.pooled = pooled
        self.project = nn.Linear(315, 48)
        self.gate = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 1))
        self.time_bias = nn.Parameter(torch.zeros(5))
        self.logit_gain = nn.Parameter(torch.tensor(np.log(0.2 / 0.8), dtype=torch.float32))

    def encode(self, tokens, partition):
        # [...,physical trial,window,coordinate] -> [...,complementary view,window,unit feature].
        ix, iy = SPLITS[partition]
        x = torch.stack(
            [tokens[..., list(ix), :, :].mean(-3), tokens[..., list(iy), :, :].mean(-3)], dim=-3
        )
        return F.normalize(F.gelu(self.project(x)), dim=-1)

    def forward(self, q, s, base):
        q0, q1 = q.unbind(2)
        s0, s1 = s.unbind(3)
        a, b = s0.mean(2), s1.mean(2)
        s01 = torch.einsum("bqtd,bktd->bqkt", q0, b)
        s10 = torch.einsum("bqtd,bktd->bqkt", q1, a)
        diag = (a * b).sum(-1)
        utility = (s01 + s10 - diag[:, None]) / 2
        margin = utility.unsqueeze(3) - utility.unsqueeze(2)
        ca = torch.einsum("bitd,bjtd->bijt", a, b)
        separation = diag.unsqueeze(2) + diag.unsqueeze(1) - ca - ca.transpose(1, 2)
        scatter = ((s0 - a[:, :, None]) * (s1 - b[:, :, None])).sum(-1).mean(2)
        spread = (scatter.unsqueeze(2) + scatter.unsqueeze(1)) / 2
        disag = s01 - s10
        disag = (disag.unsqueeze(3) - disag.unsqueeze(2)).abs()
        stats = torch.stack(
            [separation[:, None].expand_as(disag), spread[:, None].expand_as(disag), disag], dim=-1
        )
        k = a.shape[1]
        if self.pooled:
            same = torch.eye(k, device=q.device, dtype=torch.bool)[None, None, :, :, None, None]
            stats = stats.masked_fill(same, 0).sum((2, 3), keepdim=True) / (k * (k - 1))
        weights = (self.gate(stats).squeeze(-1) + self.time_bias).softmax(-1)
        vote = (margin * weights).sum(-1).sum(-1) / (k - 1)
        return base + self.logit_gain.sigmoid() * standard(vote)


def group_folds(v):
    k = v.shape[0]
    query = v.reshape(k, 5, 2, *v.shape[2:]).transpose(0, 1).reshape(5, k * 2, *v.shape[2:])
    slots = torch.arange(10, device=v.device)
    supports = torch.stack([v[:, slots // 2 != f] for f in range(5)])
    return query, supports


def load(a, stage, device):
    p = a.root / "features" / (stage + ".npz")
    meta = read(p.with_suffix(".json"))
    check(
        meta["design_sha256"] == LOCK
        and sha(p) == meta["features_sha256"]
        and meta["stage"] == stage,
        "Invalid input identities",
    )
    with np.load(p, allow_pickle=False) as z:
        ids = z["ids"]
        fold = z["fold"]
        cell = z["cell"]
        tokens = torch.from_numpy(z["tokens"]).to(device)
        f = torch.from_numpy(z["fisher"]).to(
            device, dtype=torch.float32 if stage == "train" else torch.float64
        )
        d = torch.from_numpy(z["deep"]).to(device, dtype=f.dtype)
    return ids, fold, cell, tokens, f, d


def train(a):
    _, device = incl(a)
    context(a)
    ids, fold, cell, tokens, features, deep = load(a, "train", device)
    cells = [np.flatnonzero((fold == f) & (cell == j)) for f in range(4) for j in np.unique(cell)]
    check(len(ids) == 416 and min(map(len, cells)) >= 12, "Cannot create 12-object OOF episodes")
    dest = a.root / "models"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        design_sha256=LOCK,
        driver_sha256=sha(Path(__file__)),
        training_features_sha256=sha(a.root / "features/train.npz"),
        updates=750,
        seed=260569,
        participant="sub-04",
        training_objects=416,
    )
    if (dest / "receipt.json").exists():
        rec = read(dest / "receipt.json")
        check(
            all(rec[k] == v for k, v in manifest.items())
            and all(
                sha(dest / (mode + ".safetensors")) == rec["models"][mode]["weights_sha256"]
                for mode in MODES
            ),
            "Different sealed model",
        )
        return
    models = {}
    for mode in MODES:
        torch.manual_seed(260569)
        rng = np.random.default_rng(260569)
        net = Matcher(mode == "pooled").to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.02)
        lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=750)
        label = torch.arange(12, device=device).repeat_interleave(2).expand(5, -1)
        losses = []
        for step in range(750):
            chosen = rng.choice(cells[step % len(cells)], 12, replace=False)
            order = np.stack([rng.permutation(10) for _ in chosen])
            cc = torch.as_tensor(chosen, device=device)[:, None]
            pp = torch.as_tensor(order, device=device)
            base = (anchor(*group_folds(features[cc, pp])) + anchor(*group_folds(deep[cc, pp]))) / 2
            encoded = net.encode(tokens[cc, pp], step % 3)
            logits = net(*group_folds(encoded), base)
            loss = F.cross_entropy((logits * 2).reshape(-1, 12), label.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
            lr.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 150 == 0:
                print(mode, step + 1, float(np.mean(losses[-150:])), flush=True)
        check(np.isfinite(losses).all(), "Nonfinite training objective")
        f = dest / (mode + ".safetensors")
        save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, f)
        models[mode] = dict(
            weights_sha256=sha(f),
            gain=float(net.logit_gain.sigmoid().detach()),
            first_150_loss=float(np.mean(losses[:150])),
            last_150_loss=float(np.mean(losses[-150:])),
            parameters=sum(p.numel() for p in net.parameters()),
        )
    freeze(dest / "receipt.json", manifest | dict(models=models))


def intervals(data):
    names = ("original", "pooled", "conditioned", "other_query")
    obj = data["rows"] // 10
    checks = {k: (data[k] == data["truth"]) for k in names}
    counts = np.stack([[v[obj == c].sum() for c in np.unique(obj)] for v in checks.values()])
    draws = np.random.default_rng(260552).integers(96, size=(10000, 96))
    effects = {}
    for j, arm in enumerate(names):
        if arm == "conditioned":
            continue
        diff = counts[2] - counts[j]
        effects[arm] = dict(
            net_hits=int(diff.sum()),
            delta=float(diff.sum() / 960),
            ci95=np.quantile(diff[draws].sum(1) / 960, [0.025, 0.975]).tolist(),
            gains=int(np.sum(checks["conditioned"] & ~checks[arm])),
            losses=int(np.sum(checks[arm] & ~checks["conditioned"])),
        )
    return dict(n=960, hits={k: int(v.sum()) for k, v in checks.items()}, conditioned_minus=effects)


def evaluate(a):
    from fit_episodic_object import SEED
    from fit_hard_object_pairs import episodes

    _, device = incl(a)
    paths, _, held, _ = context(a)
    ids, _, cells, tok, f, d = load(a, "held", device)
    check(np.array_equal(ids, held), "Held-out class list differs")
    root = a.root / "models"
    receipt = read(root / "receipt.json")
    check(
        receipt["design_sha256"] == LOCK and receipt["driver_sha256"] == sha(Path(__file__)),
        "Unknown model protocol",
    )
    nets = {}
    for mode in MODES:
        file = root / (mode + ".safetensors")
        check(sha(file) == receipt["models"][mode]["weights_sha256"], "Model changed after freeze")
        net = Matcher(mode == "pooled").to(device)
        net.load_state_dict(load_file(file, device=str(device)), strict=True)
        nets[mode] = net.eval()
    raw = (ids[:, None] * 10 + np.arange(10)).ravel()
    tok = tok.flatten(0, 1)
    f = f.flatten(0, 1)
    d = d.flatten(0, 1)
    report = {}
    packets = {}
    with torch.inference_mode():
        encoded = {mode: [net.encode(tok, j) for j in range(3)] for mode, net in nets.items()}
        for k in (4, 12):
            labels = []
            queries = []
            donor = []
            supports = []
            for _, _, _, y, qr, dr, sr in episodes(
                raw.reshape(-1, 10), ids, cells, k, SEED + 1000 + k
            ):
                check(not np.isin(np.r_[qr, dr], sr).any(), "Leaking query EEG")
                labels.extend([y, y])
                queries.append(qr)
                donor.append(dr)
                supports.append(sr)
            qord = np.stack(queries)
            dord = np.stack(donor)
            supp = np.stack(supports)
            qi = np.searchsorted(raw, np.concatenate([qord, dord], axis=1))
            si = np.searchsorted(raw, supp)
            pred = {m: [] for m in ("original", "pooled", "conditioned", "other_query")}
            old = paths["prior_generic_receipt"].parent / (f"generic_{k}.npz")
            check(
                sha(old)
                == read(paths["prior_generic_scores"])["predictions_sha256"][f"generic_{k}"],
                "Unknown public decisions",
            )
            with np.load(old, allow_pickle=False) as z:
                check(
                    np.array_equal(z["rows"], qord.ravel())
                    and np.array_equal(z["donor_rows"], dord.ravel())
                    and np.array_equal(z["support_rows"], np.repeat(supp, 2, axis=0)),
                    "Changed menus, donors or supports",
                )
                historical = z["mixed_slot"]
                reference = z["mixed_scores"]
                old_wrong = z["wrong_eeg_slot"]
            maxdiff = 0.0
            for start in range(0, len(qi), 16):
                stop = min(len(qi), start + 16)
                q = torch.as_tensor(qi[start:stop], device=device)
                s = torch.as_tensor(si[start:stop], device=device)
                base = (anchor(f[q], f[s]) + anchor(d[q], d[s])) / 2
                exp = reference[2 * start : 2 * stop]
                maxdiff = max(
                    maxdiff, float(np.max(abs(base[:, :2].cpu().numpy().reshape(-1, k) - exp)))
                )
                check(
                    np.array_equal(
                        base[:, 2:].argmax(-1).cpu().numpy().ravel(),
                        old_wrong[2 * start : 2 * stop],
                    ),
                    "Whole-query wrong-EEG baseline changed",
                )
                pred["original"].extend(base[:, :2].argmax(-1).cpu().numpy().ravel())
                for mode, net in nets.items():
                    scores = sum(net(z[q], z[s], base.float()) for z in encoded[mode]) / 3
                    pred[mode].extend(scores[:, :2].argmax(-1).cpu().numpy().ravel())
                    if mode == "conditioned":
                        pred["other_query"].extend(scores[:, 2:].argmax(-1).cpu().numpy().ravel())
            check(
                maxdiff < 2e-6 and np.array_equal(historical, pred["original"]),
                "Historical baseline did not replay",
            )
            packed = {k: np.asarray(v) for k, v in pred.items()} | dict(
                rows=qord.ravel(), truth=np.array(labels), donor_rows=dord.ravel()
            )
            file = a.root / "decisions" / (f"generic_{k}.npz")
            frozen_npz(file, packed)
            packets[str(k)] = sha(file)
            report[str(k)] = intervals(packed) | dict(
                historical_menu_sha256=sha(old), max_original_score_abs_replay=maxdiff
            )
    intervals12 = report["12"]["conditioned_minus"]
    gate = (
        all(intervals12[k]["ci95"][0] > 0 for k in ("original", "pooled"))
        and report["4"]["conditioned_minus"]["original"]["net_hits"] >= 0
        and intervals12["other_query"]["net_hits"] > 0
    )
    result = dict(
        protocol_sha256=LOCK,
        source_sha256=sha(Path(__file__)),
        preparation_sha256=sha(Path(__file__).with_name("prepare_support_conditioned_object.py")),
        fit_sha256=sha(root / "receipt.json"),
        private_decisions_sha256=packets,
        n_train=416,
        n_eval=96,
        independent_unit="held-out object",
        models=receipt["models"],
        menus=report,
        advance=gate,
    )
    freeze(a.release, result)
    print(report, "advance", gate, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["train", "evaluate"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--root", type=Path, default=Path("outputs/support-conditioned"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--release", type=Path, default=PUBLIC / "support_conditioned_object.json")
    a = p.parse_args()
    (train if a.stage == "train" else evaluate)(a)
