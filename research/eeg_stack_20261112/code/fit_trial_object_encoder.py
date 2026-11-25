"""Learn single-person object representations using four EEG presentations per photograph."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "trial_object_encoder_design.json"


class TrialObjectEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv1d(63, 48, 1, bias=False)
        self.temporal = nn.Conv1d(48, 48, 9, padding=4, groups=48, bias=False)
        self.log_precision = nn.Parameter(torch.zeros(1, 1, 48))
        self.head = nn.Sequential(
            nn.Conv1d(48, 64, 5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64 * 25, 96),
        )

    def trials(self, x):
        spatial = self.spatial(x.flatten(0, 1))
        temporal = F.avg_pool1d(spatial + self.temporal(spatial), 5)
        return temporal.reshape(len(x), x.shape[1], 48, 25)

    def aggregate(self, trials):
        errors = (trials - trials.mean(1, keepdim=True)).square().mean(-1)
        attention = (-F.softplus(self.log_precision) * errors).softmax(1)
        pooled = (attention[..., None] * trials).sum(1)
        return F.normalize(self.head(pooled), dim=-1)

    def forward(self, x):
        return self.aggregate(self.trials(x))


def source_data(directory, ids, records, device):
    values = []
    for cid in ids:
        f = directory / f"{cid:04d}.npy"
        check(sha(f) == records[str(cid)]["output_sha256"], f"Changed raw trial packet for {cid}")
        x = np.load(f, allow_pickle=False)
        check(
            x.shape == (10, 4, 63, 125) and x.dtype == np.float32 and np.isfinite(x).all(),
            "Invalid four-repeat input",
        )
        values.append(x)
    return torch.from_numpy(np.stack(values)).to(device)


def get_records(directory, names):
    result = {}
    for name in names:
        d = read(directory / (name + ".json"))
        check(
            d["subject"] == "sub-04" and d["shape"] == [10, 4, 63, 125],
            "Wrong input subject or axes",
        )
        for key, val in d["blocks"].items():
            check(key not in result or result[key] == val, "Conflicting source manifests")
            result[key] = val
    return result


def episode_loss(model, samples, permutations, full_query):
    # Each row is a distinct photograph and its four physical presentations.
    features = model.trials(samples.flatten(0, 1))
    index = torch.arange(len(features), device=features.device)[:, None]
    left = model.aggregate(features[index, permutations[:, :2]]).reshape(12, 10, 96)
    right = model.aggregate(features[index, permutations[:, 2:]]).reshape(12, 10, 96)
    query = model.aggregate(features).reshape(12, 10, 96) if full_query else right
    # A pair of held-out photos contributes no EEG to its own class prototype.
    support = left.reshape(12, 5, 2, 96)
    centers = F.normalize(left.sum(1)[:, None] - support.sum(2), dim=-1)
    scores = torch.einsum("cfpd,kfd->cfpk", query.reshape(12, 5, 2, 96), centers)
    objects = torch.arange(12, device=scores.device).repeat_interleave(10)
    photo_scores = torch.bmm(left, right.transpose(1, 2))
    photos = torch.arange(10, device=scores.device).repeat(12)
    return F.cross_entropy(scores.reshape(-1, 12) * 12, objects) + 0.2 * F.cross_entropy(
        photo_scores.reshape(-1, 10) * 12, photos
    )


def fit(data, sessions, spec, device):
    torch.manual_seed(spec["seed"])
    rng = np.random.default_rng(spec["seed"])
    model = TrialObjectEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.02)
    steps = 750
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    cells = [np.flatnonzero(sessions == i) for i in np.unique(sessions)]
    check(min(map(len, cells)) >= 12, "Each training episode needs twelve same-session classes")
    logs = []
    for step in range(steps):
        classes = rng.choice(cells[step % len(cells)], size=12, replace=False)
        photo_order = np.stack([rng.permutation(10) for _ in classes])
        sample = data[
            torch.as_tensor(classes, device=device)[:, None],
            torch.as_tensor(photo_order, device=device),
        ]
        splits = torch.rand(120, 4, device=device).argsort(dim=1)
        optimizer.zero_grad(set_to_none=True)
        loss = episode_loss(model, sample, splits, full_query=step % 2 == 1)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        logs.append(float(loss.detach()))
        if (step + 1) % 150 == 0:
            print("updates", step + 1, "mean training loss", np.mean(logs[-150:]), flush=True)
    check(np.isfinite(logs).all(), "Non-finite training loss")
    return model.eval(), logs


def run(a):
    spec = read(DESIGN)
    source = a.inputs / "sub-04"
    with np.load(a.baseline / "sub-04/model.npz", allow_pickle=False) as z:
        ids, held = z["train_ids"], z["generic_heldout_ids"]
    check(
        len(ids) == len(set(ids.tolist())) == 512 and len(held) == 96, "Unexpected historical split"
    )
    if a.stage == "generic":
        train_ids, test_ids = ids[~np.isin(ids, held)], ids[np.isin(ids, held)]
        records = get_records(source, ("generic512",))
    else:
        gate = read(a.out / "generic/score.json")
        check(
            gate["advance"] and gate["design_sha256"] == sha(DESIGN),
            "Generic comparison did not advance",
        )
        train_ids = ids
        panels = read(PUBLIC / "cohort_session_matched_design.json")["panels"]
        test_ids = np.unique(
            np.concatenate([panels[n]["selected_concepts"] for n in ("stress12", "new_three4")])
        )
        records = get_records(source, ("generic512", "stress12", "new_three4"))
    check(not np.intersect1d(train_ids, test_ids).size, "Evaluation class present in optimization")
    folder = a.out / a.stage
    if folder.is_dir():
        receipt = read(folder / "receipt.json")
        check(
            receipt["source_sha256"] == sha(Path(__file__))
            and receipt["design_sha256"] == sha(DESIGN)
            and receipt["session_manifest_sha256"] == sha(a.sessions)
            and receipt["baseline_split_sha256"] == sha(a.baseline / "sub-04/model.npz")
            and receipt["train_classes"] == train_ids.tolist()
            and receipt["test_classes"] == test_ids.tolist()
            and receipt["sources"]
            == {str(k): records[str(k)]["output_sha256"] for k in np.r_[train_ids, test_ids]}
            and receipt["packet_sha256"] == sha(folder / "embeddings.npz")
            and receipt["weights_sha256"] == sha(folder / "model.safetensors"),
            "Existing fit differs from the requested registered replay",
        )
        print("Verified existing fit", folder, flush=True)
        return
    device = torch.device(a.device)
    data = source_data(source, train_ids, records, device)
    # Per-electrode scale and offset from the training photos only.
    sd, mu = torch.std_mean(data, dim=(0, 1, 2, 4), unbiased=False)
    sd = sd.clamp_min(1e-5)[:, None]
    mu = mu[:, None]
    data.sub_(mu).div_(sd).clamp_(-12, 12)
    dates = np.array(read(a.sessions)["subjects"]["sub-04"]["session_pairs"])
    _, sessions = np.unique(dates[train_ids], axis=0, return_inverse=True)
    model, losses = fit(data, sessions, spec, device)
    del data
    test = source_data(source, test_ids, records, device)
    with torch.inference_mode():
        y = torch.cat(
            [model(((block - mu) / sd).clamp(-12, 12)) for block in test.flatten(0, 1).split(128)]
        )
    y = y.reshape(len(test_ids), 10, 96).cpu().numpy()
    check(
        np.isfinite(y).all() and np.allclose(np.linalg.norm(y, axis=-1), 1, atol=1e-5),
        "Invalid photo representations",
    )
    a.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=a.out) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        weights.update(input_mean=mu.cpu(), input_scale=sd.cpu())
        save_file(weights, stage / "model.safetensors")
        np.savez(stage / "embeddings.npz", ids=test_ids, features=y)
        freeze(
            stage / "receipt.json",
            dict(
                subject="sub-04",
                train_classes=train_ids.tolist(),
                test_classes=test_ids.tolist(),
                design_sha256=sha(DESIGN),
                source_sha256=sha(Path(__file__)),
                packet_sha256=sha(stage / "embeddings.npz"),
                weights_sha256=sha(stage / "model.safetensors"),
                baseline_split_sha256=sha(a.baseline / "sub-04/model.npz"),
                session_manifest_sha256=sha(a.sessions),
                sources={
                    str(k): records[str(k)]["output_sha256"] for k in np.r_[train_ids, test_ids]
                },
                updates=750,
                first_150_training_loss=float(np.mean(losses[:150])),
                last_150_training_loss=float(np.mean(losses[-150:])),
                torch_version=str(torch.__version__),
                device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            ),
        )
        stage.rename(folder)
    print(folder, sha(folder / "embeddings.npz"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("generic", "target"))
    p.add_argument("--inputs", type=Path, default=Path("outputs/trial-object-inputs"))
    p.add_argument("--baseline", type=Path, default=Path("outputs/episodic-object"))
    p.add_argument("--sessions", type=Path, default=Path("outputs/object-transfer/sessions.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/trial-object-encoder"))
    p.add_argument("--device", default="cuda")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    run(p.parse_args())
