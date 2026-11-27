"""Compare same-photo and cross-photo neural consistency for a temporal EEG encoder."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, require, units
from paper_raw_semantic_readout import leave_photo_out, normalize, read_json, sha, top_credit
from paper_render_cache import atomic_file, save_json
from paper_semantic_contrast import METRICS, contrast, data_registry, matched_montages, nearest_other


ARMS = ("visual-targets", "same-photo", "cross-photo")
SEED = 20261129
EPOCHS = 24
BATCH = 192
SAMPLES_PER_BIN = 2
PEER_WEIGHT = .4


class TemporalReadout(nn.Module):
    def __init__(self, width=96, bins=100, segments=1, channels=63):
        super().__init__()
        self.segments = segments
        self.stem = nn.Sequential(nn.Conv1d(channels, width, 9, stride=2, padding=4), nn.GroupNorm(8, width), nn.GELU())
        self.temporal = nn.Sequential(nn.Conv1d(width, width, 7, dilation=2, padding=6, groups=width),
                                      nn.Conv1d(width, width, 1), nn.GroupNorm(8, width), nn.GELU())
        self.pool = nn.Conv1d(width, 2, 1)
        self.pool_bias = nn.Parameter(torch.zeros(2, (bins + 1) // 2))
        pooled_width = width * segments
        projection_width = round(512 * (width + 1024) / (pooled_width + 1024))
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(pooled_width, projection_width), nn.LayerNorm(projection_width),
                                                       nn.GELU(), nn.Linear(projection_width, 1024)) for _ in range(2))

    def forward(self, eeg):
        hidden = self.stem(eeg)
        hidden = hidden + self.temporal(hidden)
        scores = self.pool(hidden).float() + self.pool_bias
        pooled = torch.cat([torch.bmm(w.softmax(-1), h.transpose(1, 2).float())
                            for w, h in zip(scores.tensor_split(self.segments, dim=-1),
                                            hidden.tensor_split(self.segments, dim=-1), strict=True)], dim=-1)
        return tuple(F.normalize(layer(pooled[:, i]).float(), dim=-1) for i, layer in enumerate(self.projections))


def paired_batches(groups, offsets, epoch, rng, batch_size=BATCH):
    """Both peer views use the same participant and the same object as the anchor."""
    order = np.take_along_axis(groups, rng.random(groups.shape).argsort(axis=-1), axis=-1)
    photos_per_object = groups.shape[1]
    offset = 1 + epoch % (photos_per_object - 1)
    for i in range(photos_per_object):
        shuffled = rng.permutation(len(groups))
        for first in range(0, len(groups), batch_size):
            g = shuffled[first:first + batch_size]
            anchor = order[g, i]
            sibling = order[g, (i + offset) % photos_per_object]
            person = (offsets[anchor] + epoch) % 10
            peer = (person + 1 + epoch % 9) % 10
            yield np.stack((person, peer, peer)), np.stack((anchor, anchor, sibling))


def visual_loss(photo, semantic, image, prototype, text):
    photo_fit = contrast(photo, image)
    object_fit = contrast(semantic, prototype)
    text_fit = contrast(semantic, text)
    return photo_fit + .50 * object_fit + .25 * text_fit


def predict(model, eeg):
    model.eval()
    axes = eeg.shape[:2]
    with torch.inference_mode():
        output = [tuple(q.cpu() for q in model(part.float())) for part in eeg.flatten(0, 1).split(256)]
    return tuple(torch.cat(group).reshape(*axes, -1).numpy() for group in zip(*output, strict=True))


def train(arm, eeg, held, image, prototype, text, groups, epochs, model_path, *, model_factory=TemporalReadout):
    torch.manual_seed(SEED)
    device = eeg.device
    model = model_factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=.025)
    steps = epochs * groups.shape[1] * ((len(groups) + BATCH - 1) // BATCH)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=4e-5)
    rng = np.random.default_rng(SEED)
    offsets = rng.integers(10, size=eeg.shape[1])
    indices = hashlib.sha256()
    records = []
    updates = 0
    for epoch in range(epochs):
        sums = torch.zeros(3, device=device)
        for subjects, rows in paired_batches(groups, offsets, epoch, rng):
            indices.update(subjects.astype("<i8").tobytes())
            indices.update(rows.astype("<i8").tobytes())
            s = torch.from_numpy(subjects).to(device)
            i = torch.from_numpy(rows).to(device)
            inputs = eeg[s, i].flatten(0, 1).float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                q, z = (out.reshape(*i.shape, -1) for out in model(inputs))
                supervised = torch.stack([visual_loss(q[j], z[j], image[i[j]], prototype[i[j]], text[i[j]])
                                          for j in range(len(i))]).mean()
                paired = supervised.new_zeros(())
                if arm != "visual-targets":
                    support = 1 if arm == "same-photo" else 2
                    paired = contrast(z[0], z[support])
                objective = supervised + PEER_WEIGHT * paired
            opt.zero_grad(set_to_none=True)
            objective.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            opt.step()
            schedule.step()
            sums += torch.stack((supervised.detach(), paired.detach(), grad.detach())) * i.shape[1]
            updates += 1
        row = (sums / eeg.shape[1]).cpu().tolist()
        records.append(dict(zip(("visual_loss", "peer_loss", "grad_norm"), row, strict=True)))
        print(f"{arm} {epoch + 1:02d}/{epochs}: {records[-1]}", flush=True)
    require(updates == steps, "Incomplete temporal EEG training schedule")
    q, z = predict(model, held)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    atomic_file(model_path, lambda f: torch.save(state, f))
    return q, z, dict(epochs=epochs, updates=updates, schedule_sha256=indices.hexdigest(), losses=records)


def held_concepts(benchmark, train, dev, label):
    path = benchmark / "verified/data/prep/paper_split.json"
    fresh = np.asarray(read_json(path)["fresh_rows"])
    pictures = np.concatenate((dev, fresh))
    require(fresh.shape == (500,) and len(set(pictures)) == 700
            and set(label[fresh]).isdisjoint(label[np.concatenate((train, dev))]), "The two held sets must be disjoint by concept")
    for selection, count in ((dev, 20), (fresh, 50)):
        freq = np.unique(label[selection], return_counts=True)[1]
        require(len(freq) == count and np.all(freq == 10), "Expected ten photographs per held concept")
    return pictures


def evaluate_arm(embedding, bank, names, photo, concepts, donor):
    row = np.arange(len(names))
    appearance = embedding @ photo.T
    logits = embedding @ bank.T
    same = concepts[:, None] == concepts[None, :]
    margin = logits[:, row, names] + logits[:, donor, names[donor]] - logits[:, row, names[donor]] - logits[:, donor, names]
    return np.stack((top_credit(logits, names, 1), top_credit(logits, names, 5), top_credit(appearance, row, 1),
                     top_credit(np.where(same, appearance, -np.inf), row, 1),
                     (margin > 0) + .5 * (margin == 0)), axis=-1)


def score(output, benchmark, held, labels, category, image, layout, proof, arms, *, comparisons=None,
          report_name="temporal-readout-verified.json"):
    path = benchmark / "semantic-energy-check/h14-text-bank.npz"
    bank_sha = sha(path)
    require(bank_sha == read_json(path.with_suffix(".json"))["bank_sha256"], "Frozen H/14 evaluator has changed")
    with np.load(path, allow_pickle=False) as z:
        bank, ids = normalize(z["text"]), z["concept_id"]
    require(bank.shape == (1854, 1024) and len(set(ids)) == 1854, "Invalid open vocabulary")
    lookup = {c: i for i, c in enumerate(ids)}
    source, estimate, schedules = {}, {}, set()
    for arm in arms:
        path = output / f"{arm}.npz"
        receipt = read_json(path.with_suffix(".json"))
        require(receipt["arm"] == arm and receipt["sources"] == proof and receipt["predictions_sha256"] == sha(path),
                "Held predictions use different training inputs")
        schedules.add(receipt["schedule_sha256"])
        source[arm] = receipt["predictions_sha256"]
        with np.load(path, allow_pickle=False) as z:
            require(np.array_equal(z["row"], held) and np.array_equal(z["concept"], category[labels[held]]), "Query identities changed")
            estimate[arm] = {k: normalize(z[key]) for k, key in (("semantics", "embedding"), ("photo", "photo_embedding"))}
    require(len(schedules) == 1, "Training data exposures differ between the arms")
    if comparisons is None:
        comparisons = (("cross-photo", "same-photo"), ("cross-photo", "visual-targets"), ("same-photo", "visual-targets"))
    overall = {}
    for panel, where in (("dev", slice(0, 200)), ("fresh", slice(200, 700))):
        rows = held[where]
        names = np.array([lookup[c] for c in category[labels[rows]]])
        concept = category[labels[rows]]
        photo = image[rows]
        donor = nearest_other(layout[rows], labels[rows])
        results = {f"{arm}.{kind}": evaluate_arm(x[:, where], bank, names, photo, concept, donor)
                   for arm in arms for kind, x in estimate[arm].items()}
        paired = {}
        for a, b in comparisons:
            if a not in arms or b not in arms:
                continue
            for kind in ("semantics", "photo"):
                delta = results[f"{a}.{kind}"] - results[f"{b}.{kind}"]
                lo, hi = bootstrap_ci(units(delta, concept, panel))
                paired[f"{a}.{kind} minus {b}.{kind}"] = {k: [float(delta[:, :, i].mean()), float(lo[i]), float(hi[i])]
                                                        for i, k in enumerate(METRICS)}
        overall[panel] = dict(queries=10 * len(rows), concepts=len(set(concept)),
                              truth_name_top1=float(top_credit(photo @ bank.T, names, 1).sum()), paired=paired,
                              hits={a: dict(zip(METRICS, x.sum((0, 1)).tolist(), strict=True)) for a, x in results.items()})
        atomic_file(output / f"{panel}-scores.npz", lambda f: np.savez_compressed(f, row=rows, concept=concept, **results))
    save_json(output / report_name, dict(sources=source, proof=proof, vocabulary_sha256=bank_sha,
                                         schedule_sha256=schedules.pop(), **overall))
    print(json.dumps({k: v["hits"] for k, v in overall.items()}, indent=2), flush=True)


def run(benchmark, output, device, arms, epochs, segments=1):
    require(epochs > 0 and len(arms) == len(set(arms)), "Choose distinct readout arms and a positive epoch count")
    raw, train_rows, dev, labels, category, image, text, layout, sources = data_registry(benchmark)
    held_rows = held_concepts(benchmark, train_rows, dev, labels)
    full_dir = benchmark / "native_baseline_candidate/enigma/prep"
    data, held, _, eeg_sources, norm = matched_montages(raw, full_dir, train_rows, held_rows, sources["photo_registry"],
                                                        bin_samples=SAMPLES_PER_BIN, dtype=np.float16, with_normalization=True)
    eeg = torch.from_numpy(data).to(device)
    queries = torch.from_numpy(held).to(device)
    del data, held
    group = np.unique(labels[train_rows], return_inverse=True)[1]
    require(np.array_equal(np.bincount(group), np.full(1323, 10)), "Expected ten training photos for each of 1,323 objects")
    groups = np.argsort(group, kind="stable").reshape(1323, 10)
    visual = image[train_rows]
    prototype = normalize(leave_photo_out(visual, group))
    target = tuple(torch.from_numpy(a).float().to(device) for a in (visual, prototype, text[labels[train_rows]]))
    output.mkdir(parents=True, exist_ok=True)
    norm_path = output / "normalization.npy"
    if norm_path.exists():
        require(np.array_equal(np.load(norm_path, allow_pickle=False), norm), "Training normalization changed")
    else:
        atomic_file(norm_path, lambda f: np.save(f, norm, allow_pickle=False))
    proof = dict(data=sources, eeg=eeg_sources, code=sha(Path(__file__)), loader=sha(Path(__file__).with_name("paper_semantic_contrast.py")),
                 epochs=epochs, batch_size=BATCH, physical_repeats=4, montage=63, bin_samples=SAMPLES_PER_BIN,
                 sample_hz=250, seed=SEED, peer_weight=PEER_WEIGHT, images=len(train_rows), subjects=10,
                 normalization_sha256=sha(norm_path), segments=segments)
    for arm in arms:
        predictions = output / f"{arm}.npz"
        model = output / f"{arm}.pt"
        meta = predictions.with_suffix(".json")
        if meta.exists():
            existing = read_json(meta)
            require(existing["sources"] == proof and existing["predictions_sha256"] == sha(predictions)
                    and existing["weights_sha256"] == sha(model), "An existing arm belongs to another experiment")
            continue
        q, z, history = train(arm, eeg, queries, *target, groups, epochs, model,
                              model_factory=lambda: TemporalReadout(segments=segments))
        atomic_file(predictions, lambda f: np.savez_compressed(f, embedding=z, photo_embedding=q, row=held_rows,
                                                               concept=category[labels[held_rows]]))
        save_json(meta, dict(arm=arm, sources=proof, predictions_sha256=sha(predictions), weights_sha256=sha(model), **history))
    score(output, benchmark, held_rows, labels, category, image, layout, proof, arms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--segments", type=int, choices=(1, 4), default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(6)
    run(args.benchmark, args.output, torch.device(args.device), args.arms, args.epochs, args.segments)
