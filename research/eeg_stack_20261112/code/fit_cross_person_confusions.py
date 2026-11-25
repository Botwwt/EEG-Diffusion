"""Distil photo-specific errors of two independent viewers into a personal EEG encoder."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from fit_trial_object_encoder import TrialObjectEncoder, episode_loss
from prepare_cross_person_confusions import (
    DESIGN,
    LOCK,
    PEOPLE,
    cli,
    frame,
    normfit,
    participant,
    setup,
)
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

ARMS = ("zero", "sham", "matched")
STEPS = 1500
STUDENT_SEED = 260553


def wrong_logits(z):
    """Cosines to the eleven alternative classes, centered within each held photo pair."""
    width = z.shape[-1]
    grouped = z.reshape(*z.shape[:-3], 12, 5, 2, width)
    centers = F.normalize(z.sum(-2).unsqueeze(-2) - grouped.sum(-2), dim=-1)
    score = torch.einsum("...ifpd,...jfd->...ifpj", F.normalize(grouped, dim=-1), centers)
    others = ~torch.eye(12, dtype=torch.bool, device=z.device)
    score = score[..., others[:, None, None, :].expand(12, 5, 2, 12)].reshape(
        *z.shape[:-3], 12, 5, 2, 11
    )
    score = (score - score.mean(-1, keepdim=True)) / score.std(
        -1, unbiased=False, keepdim=True
    ).clamp_min(1e-6)
    return score - score.mean(-2, keepdim=True)


def verify_logits():
    """Swapping the two queries preserves all eight supports and swaps negative labels."""
    z = torch.randn(2, 12, 10, 96, generator=torch.Generator().manual_seed(13))
    out = wrong_logits(z)
    swapped = z.reshape(2, 12, 5, 2, 96).flip(-2).reshape(z.shape)
    assert out.shape == (2, 12, 5, 2, 11)
    torch.testing.assert_close(wrong_logits(swapped), out.flip(-2), atol=4e-6, rtol=1e-5)
    torch.testing.assert_close(out.sum(-2), torch.zeros_like(out.sum(-2)), atol=4e-6, rtol=0)


def teacher_bank(a, ids, fold):
    features = np.empty((3, 2, len(ids), 10, 96), np.float32)
    receipts = {}
    teacher_driver = Path(__file__).with_name("prepare_cross_person_confusions.py")
    for who, sub in enumerate(PEOPLE):
        for k in range(4):
            path = a.out / "teachers" / sub / f"fold{k}.json"
            r = read(path)
            check(
                r["design_sha256"] == LOCK
                and r["subject"] == sub
                and r["source_sha256"] == sha(teacher_driver)
                and np.array_equal(r["train_ids"], ids[fold != k])
                and np.array_equal(r["encoded_ids"], ids[fold == k])
                and r["features_sha256"] == sha(path.with_suffix(".npz")),
                "Teacher split or features differ",
            )
            receipts[f"{sub}/{k}"] = sha(path)
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                check(np.array_equal(z["ids"], ids[fold == k]), "Teacher class order")
                for h, key in enumerate(("neural", "fisher")):
                    block = z[key]
                    check(
                        block.shape == (sum(fold == k), 10, 96) and np.isfinite(block).all(),
                        "Bad teacher embedding",
                    )
                    features[who, h, fold == k] = block
    return torch.from_numpy(features).to(a.device), receipts


def auxiliary_schedule(bank, eligible, who, seed):
    rng = np.random.default_rng(seed)
    other = [i for i in range(3) if i != who]
    peers = bank[other].reshape(4, bank.shape[2], 10, 96)
    # All four representations use the same images and one fold's coordinate system.
    classes, photos, targets = [], [], []
    stream = hashlib.sha256()
    for step in range(STEPS):
        c = rng.choice(eligible[step % len(eligible)], 12, replace=False)
        p = np.stack([rng.permutation(10) for _ in c])
        stream.update(c.tobytes())
        stream.update(p.tobytes())
        c, p = (torch.as_tensor(v, device=bank.device) for v in (c, p))
        z = peers[:, c[:, None], p]
        targets.append(wrong_logits(z).mean(0).softmax(-1))
        classes.append(c)
        photos.append(p)
    return torch.stack(classes), torch.stack(photos), torch.stack(targets), stream.hexdigest()


def train_one(raw, sessions, aux, arm, device):
    torch.manual_seed(STUDENT_SEED)
    rng = np.random.default_rng(STUDENT_SEED)
    model = TrialObjectEncoder().to(device)
    initial, stream, noise = (hashlib.sha256() for _ in range(3))
    for k, v in model.state_dict().items():
        initial.update(k.encode())
        initial.update(v.detach().cpu().contiguous().numpy().tobytes())
    cells = [np.flatnonzero(sessions == i) for i in np.unique(sessions)]
    check(min(map(len, cells)) >= 12, "Insufficient personal candidates")
    opt = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    auxclasses, auxphotos, targets, auxhash = aux
    summed = torch.zeros(2, device=device)
    for step in range(STEPS):
        c = rng.choice(cells[step % len(cells)], 12, replace=False)
        p = np.stack([rng.permutation(10) for _ in c])
        cc, pp = (torch.as_tensor(v, device=device) for v in (c, p))
        split = torch.rand(120, 4, device=device).argsort(1)
        for v in (c, p, split.cpu().numpy()):
            stream.update(v.tobytes())
        opt.zero_grad(set_to_none=True)
        model.train()
        supervised = episode_loss(model, raw[cc[:, None], pp], split, step % 2 == 1)
        model.eval()  # Independent differentiable pass, with no dropout draws.
        z = model(raw[auxclasses[step, :, None], auxphotos[step]].flatten(0, 1)).reshape(12, 10, 96)
        target = targets[step]
        if arm == "sham":
            target = target.flip(-2)
        kl = F.kl_div(wrong_logits(z).log_softmax(-1), target, reduction="none").sum(-1).mean()
        # The reference arm deliberately takes precisely the supervised-only gradient.
        loss = supervised if arm == "zero" else supervised + 0.25 * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        sched.step()
        summed += torch.stack((supervised.detach(), kl.detach()))
        state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        noise.update(state.numpy().tobytes())
    stats = (summed / STEPS).tolist()
    check(np.isfinite(stats).all(), "Nonfinite objective")
    trace = dict(
        initial=initial.hexdigest(),
        supervised=stream.hexdigest(),
        auxiliary=auxhash,
        dropout=noise.hexdigest(),
    )
    return model, trace, stats


def train(a):
    verify_logits()
    ids, _, schedules, fold, eligible = frame(a)
    check(
        all(len(np.unique(fold[c])) == 1 for c in eligible), "Auxiliary classes span teacher folds"
    )
    bank, teachers = teacher_bank(a, ids, fold)
    manifest = dict(
        design_sha256=LOCK, student_driver_sha256=sha(Path(__file__)), teachers=teachers
    )
    freeze(a.out / "teachers.json", manifest)
    models, traces = {}, {}
    device = torch.device(a.device)
    for who, sub in enumerate(PEOPLE):
        _, sessions = np.unique(schedules[who, ids], axis=0, return_inverse=True)
        data = aux = None
        for arm in ARMS:
            folder = a.out / "students" / sub / arm
            folder.mkdir(parents=True, exist_ok=True)
            p, rec = folder / "model.safetensors", folder / "receipt.json"
            header = dict(
                teachers_sha256=sha(a.out / "teachers.json"),
                subject=sub,
                arm=arm,
                train_ids=ids.tolist(),
            )
            if not rec.exists():
                if data is None:
                    data, phys = participant(a, sub, ids)
                    del phys
                    data, mu, sd = normfit(data)
                    aux = auxiliary_schedule(bank, eligible, who, read(DESIGN)["seed"])
                model, trace, values = train_one(data, sessions, aux, arm, device)
                state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
                state.update(input_mean=mu.cpu(), input_scale=sd.cpu())
                save_file(state, p)
                freeze(
                    rec,
                    header | dict(model_sha256=sha(p), trace=trace, mean_training_losses=values),
                )
                print(sub, arm, "frozen", flush=True)
            r = read(rec)
            check(
                all(r[k] == v for k, v in header.items()) and sha(p) == r["model_sha256"],
                "Wrong frozen student",
            )
            if sub not in traces:
                traces[sub] = r["trace"]
            check(r["trace"] == traces[sub], "Unmatched supervision, initialization, or dropout")
            models[f"{sub}/{arm}"] = sha(rec)
        del data, aux
    original = a.data / "scale-trial/objects928_steps1500/model.safetensors"
    replay = load_file(a.out / "students/sub-04/zero/model.safetensors")
    previous = load_file(original)
    check(replay.keys() == previous.keys(), "Reference model keys changed")
    err = max((replay[k] - previous[k]).abs().max().item() for k in replay)
    freeze(
        a.out / "fits.json",
        manifest | dict(students=models, trace=traces, sub04_zero_reference_max_abs=err),
    )
    print("All nine students frozen. Baseline max weight difference", err, flush=True)


if __name__ == "__main__":
    setup()
    train(cli().parse_args())
