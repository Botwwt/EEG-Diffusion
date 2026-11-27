"""Compare continuous, posterior mean, and posterior mode EEG semantics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, units
from paper_fusion_siglip_probe import measure, predict as apply_ridge, rank_credit, ridge_system, solve_ridge
from paper_layout_counterfactual import BINS, CHANNELS, physical_inputs
from paper_layout_mixture_composition import native_conditions
from paper_raw_semantic_readout import leave_photo_out, normalize
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_semantic_contrast import data_registry
from paper_tangent_calibration import read_inputs
from paper_temporal_readout import BATCH, SEED, TemporalReadout, held_concepts, paired_batches, visual_loss


DEPTH = 6
ENTRIES = 192
ARMS = ("continuous", "posterior-mean", "posterior-mode")
METRICS = ("h14_top1", "siglip2_top1", "h14_top5", "siglip2_top5", "h14_photo_top1")


def nearest(target, centers):
    return (2 * target @ centers.T - centers.square().sum(-1)).argmax(-1)


def quantize(target, codebooks):
    residual = target
    indices = []
    for centers in codebooks:
        choice = nearest(residual, centers)
        residual = residual - centers[choice]
        indices.append(choice)
    return torch.stack(indices, dim=-1)


def decode_posterior(probability, codebooks):
    """Return additive mean and the reconstruction of the most likely code tuple."""
    mean = torch.einsum("...lk,lkd->...d", probability, codebooks)
    choice = probability.argmax(-1)
    mode = codebooks[torch.arange(len(codebooks), device=choice.device), choice].sum(-2)
    return mean, mode


@torch.no_grad()
def fit_codebooks(target, depth=DEPTH, entries=ENTRIES, iterations=24):
    require(depth > 0 and 1 < entries <= len(target) and iterations > 0, "Invalid residual codebook dimensions")
    generator = torch.Generator(device=target.device).manual_seed(SEED)
    center = target.mean(0)
    residual = target - center
    codebooks = []
    for level in range(depth):
        centers = residual[torch.randperm(len(residual), generator=generator, device=residual.device)[:entries]]
        for _ in range(iterations):
            ids = nearest(residual, centers)
            membership = F.one_hot(ids, entries).T.to(residual)
            count = membership.sum(-1, keepdim=True)
            update = (membership @ residual) / count.clamp_min(1)
            centers = torch.where(count > 0, update, centers)
        residual = residual - centers[nearest(residual, centers)]
        codebooks.append(centers)
        print(f"Fitted visual residual codebook {level + 1}/{depth}, reconstruction loss {residual.square().mean():.7f}", flush=True)
    return center, torch.stack(codebooks)


class CategoricalReadout(nn.Module):
    def __init__(self, encoder=None, code_shape=(DEPTH, ENTRIES)):
        super().__init__()
        self.encoder = TemporalReadout(bins=BINS, channels=CHANNELS) if encoder is None else encoder
        self.code_shape = code_shape
        self.codes = nn.Linear(1024, code_shape[0] * code_shape[1])

    def forward(self, eeg):
        photo, subject = self.encoder(eeg)
        logits = self.codes(32 * subject).float().unflatten(-1, self.code_shape)
        return photo, subject, logits


def train(inputs, groups, photo, semantics, text, ids, epochs, *, encoder_factory=None, code_shape=(DEPTH, ENTRIES)):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    offsets = rng.integers(10, size=inputs.shape[1])
    encoder = encoder_factory() if encoder_factory is not None else None
    model = CategoricalReadout(encoder, code_shape).to(inputs.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=.025)
    nsteps = epochs * 10 * ((len(groups) + BATCH - 1) // BATCH)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nsteps, eta_min=4e-5)
    samples = hashlib.sha256()
    losses = []
    for epoch in range(epochs):
        totals = torch.zeros(3, device=inputs.device)
        for people, rows in paired_batches(groups, offsets, epoch, rng):
            person, photo_idx = people[0], rows[0]
            samples.update(person.astype("<i8").tobytes())
            samples.update(photo_idx.astype("<i8").tobytes())
            choose_person = torch.from_numpy(person).to(inputs.device)
            choose_photo = torch.from_numpy(photo_idx).to(inputs.device)
            with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=inputs.is_cuda):
                q, identity, logits = model(inputs[choose_person, choose_photo])
                vision = visual_loss(q, identity, photo[choose_photo], semantics[choose_photo], text[choose_photo])
                codes = F.cross_entropy(logits.flatten(0, 1), ids[choose_photo].flatten())
                objective = vision + .5 * codes
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            accuracy = logits.argmax(-1).eq(ids[choose_photo]).float().mean()
            totals += torch.stack((vision.detach(), codes.detach(), accuracy)) * len(photo_idx)
        losses.append((totals / len(photo)).tolist())
        print(f"Categorical EEG training {epoch + 1}/{epochs}: {losses[-1]}", flush=True)
    require(schedule.last_epoch == nsteps, "The registered training schedule did not complete")
    return model.eval(), dict(updates=nsteps, schedule=samples.hexdigest(), losses=losses)


@torch.inference_mode()
def decode_eeg(model, eeg, center, books):
    shape = eeg.shape[:3]
    prediction = [model(row) for row in eeg.flatten(0, 2).split(256)]
    photo, continuous, logits = (torch.cat(rows) for rows in zip(*prediction, strict=True))
    probability = logits.reshape(*shape, DEPTH, ENTRIES).mean(2).softmax(-1)
    mean, mode = decode_posterior(probability, books)
    vectors = [continuous.reshape(*shape, -1).mean(2), center + mean, center + mode]
    estimates = dict(zip(ARMS, (F.normalize(x, dim=-1) for x in vectors), strict=True))
    return F.normalize(photo.reshape(*shape, -1).mean(2), dim=-1), estimates, probability


def concepts(registry):
    with registry.open(encoding="utf-8-sig", newline="") as stream:
        return np.array([r["unique_id"] for r in csv.DictReader(stream)])


def evaluation_sources(benchmark, training, train_visual, names, pins, device):
    source, train, _, fused, _, _, _, _, proof, _ = read_inputs(benchmark, pins)
    require(np.array_equal(train, training), "The common visual bridge uses another training split")
    signature = source / "h14-text-bank.npz"
    require(pins.pin(signature) == json.loads(signature.with_suffix(".json").read_bytes())["bank_sha256"], "Changed H/14 bank")
    with np.load(signature, allow_pickle=False) as text:
        bank = torch.from_numpy(normalize(text["text"])).to(device)
        words = {word: i for i, word in enumerate(text["concept_id"])}
    reference = benchmark / "token-rank-check/siglip-independent/siglip-fused-evaluation-bridge.pt"
    receipt = json.loads(reference.with_name("siglip-fused-probe.json").read_bytes())
    registry = benchmark / "inputs/image_set/concepts1854.csv"
    require(pins.pin(reference) == receipt["evaluation_bridge_sha256"]
            and pins.pin(registry) == receipt["vision_and_text"]["test"]["concepts"], "Changed SigLIP2 evaluation bridge")
    independent = torch.load(reference, map_location=device, weights_only=True)
    names_sig = {word: i for i, word in enumerate(concepts(registry))}
    require(len(words) == len(names_sig) == 1854 and set(names).issubset(words) and set(names).issubset(names_sig),
            "Expected the full concept dictionary for evaluation")
    fused = torch.from_numpy(normalize(np.array(fused[train], copy=True))).to(device)
    bridge = solve_ridge(ridge_system(train_visual[None], fused), .01)
    proof = dict(split=proof["split"], h14=pins.pin(signature), siglip=pins.pin(reference), photo_bridge_ridge=.01)
    return bank, words, independent, names_sig, bridge, proof


def scores(vectors, photographs, labels, sources):
    bank, names, independent, names_sig, bridge, _ = sources
    h14 = measure(vectors, bank, np.arange(1854), np.asarray([names[k] for k in labels]))
    sg = apply_ridge(apply_ridge(vectors, bridge), independent["readout"])
    siglip = measure(sg, independent["text_bank"], independent["names"].cpu().numpy(), np.asarray([names_sig[k] for k in labels]))
    p = rank_credit((vectors @ photographs.T).cpu().numpy(), np.arange(len(labels)), 1)
    return np.stack((h14[..., 0], siglip[..., 0], h14[..., 1], siglip[..., 1], p), axis=-1)


@torch.inference_mode()
def evaluate(path, native, predicted, datasets, sources, pins):
    summary, observations, compositions = {}, {}, {}
    for panel, key, rows, labels, reference in datasets:
        base = F.normalize(native[key][:, rows].to(reference), dim=-1)
        conditions = {"native": base}
        for arm, value in predicted[key].items():
            current = value[:, rows]
            conditions[arm] = current
            conditions[f"mix-{arm}"] = F.normalize(.75 * base + .25 * current, dim=-1)
            compositions[f"{panel}/{arm}"] = current.cpu().numpy()
        shift = 1 if panel == "official" else 10
        require(np.all(labels != np.roll(labels, shift)), "Mismatch must change every stimulus concept")
        conditions["mix-posterior-mode-wrong"] = F.normalize(
            .75 * base + .25 * predicted[key]["posterior-mode"][:, rows].roll(shift, 1), dim=-1)
        measured = {key: scores(v, reference, labels, sources) for key, v in conditions.items()}
        paired = {}
        for arm, control in (("posterior-mode", "posterior-mean"), ("mix-posterior-mode", "mix-posterior-mean"),
                             ("mix-posterior-mode", "mix-continuous"), ("mix-posterior-mode", "native"),
                             ("mix-posterior-mode", "mix-posterior-mode-wrong")):
            effect = measured[arm] - measured[control]
            lo, hi = bootstrap_ci(units(effect, labels, panel))
            estimates = np.stack((effect.mean((0, 1)), lo, hi), axis=-1).tolist()
            paired[f"{arm} minus {control}"] = dict(zip(METRICS, estimates, strict=True))
        summary[panel] = dict(n=10 * len(labels), paired=paired, hits={
            k: dict(zip(METRICS, v.sum((0, 1)).tolist(), strict=True)) for k, v in measured.items()})
        observations.update({f"{panel}/{arm}": q for arm, q in measured.items()})
        print(panel, json.dumps(summary[panel]["hits"]), flush=True)
    file = path / "readout-scores.npz"
    atomic_file(file, lambda f: np.savez_compressed(f, **observations))
    condition = path / "eeg-readout-conditions.npz"
    atomic_file(condition, lambda f: np.savez_compressed(f, **compositions))
    return summary, dict(metrics=pins.pin(file), output=pins.pin(condition))


def run(benchmark, output, epochs, device):
    pins = PinnedFiles()
    raw, training, dev, label, category, visual, texts, _, vision_proof = data_registry(benchmark)
    held = held_concepts(benchmark, training, dev, label)
    data, development, official, physical = physical_inputs(raw, training, held, pins)
    data = data[:, :, 0].to(device)
    baseline, scale = data.mean(1, keepdim=True), data.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
    data = (data - baseline) / scale
    development = ((development.to(device) - baseline) / scale).unsqueeze(2)
    official = (official.to(device) - baseline[:, :, None]) / scale[:, :, None]
    category_ids = np.unique(label[training], return_inverse=True)[1]
    groups = np.argsort(category_ids, kind="stable").reshape(-1, 10)
    image = torch.from_numpy(visual[training]).to(device=device, dtype=torch.float32)
    prototype = torch.from_numpy(normalize(leave_photo_out(visual[training], category_ids))).to(image)
    name_target = torch.from_numpy(texts[label[training]]).to(image)
    center, codebooks = fit_codebooks(prototype)
    ids = quantize(prototype - center, codebooks)
    model, schedule = train(data, groups, image, prototype, name_target, ids, epochs)
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "categorical-readout.pt"
    params = dict(weights={k: v.detach().cpu() for k, v in model.state_dict().items()}, center=center.cpu(),
                  codebooks=codebooks.cpu(), mean=baseline.cpu(), scale=scale.cpu())
    atomic_file(weights, lambda f: torch.save(params, f))
    study = benchmark / "semantic-energy-check"
    proof = dict(visual=vision_proof, physical=physical)
    frozen, native_receipts = native_conditions(study / "horizon-readout-20261127/native-h14",
                                                 study / "background-invariant-20261127/guide", held, proof, pins)
    pack = study / "alignvis-20261127/alignvis-image-teachers.npz"
    receipt = json.loads(pack.with_suffix(".json").read_bytes())
    official_registry = benchmark / "inputs/image_set/test_photos.csv"
    require(pins.pin(pack) == receipt["array_sha256"] and pins.pin(official_registry) == receipt["photo_registries"]["test"],
            "The official image scoring files changed")
    with np.load(pack, allow_pickle=False) as references:
        official_names = references["test_word"]
        require(len(official_names) == 200 and np.array_equal(official_names, concepts(official_registry))
                and set(category[label[training]]).isdisjoint(official_names), "Invalid novel-concept evaluation rows")
        test_visual = torch.from_numpy(normalize(references["test/h14"])).to(device)
    held_names = category[label[held]]
    dev_visual = torch.from_numpy(visual[held]).to(device)
    eval_sources = evaluation_sources(benchmark, training, image, np.r_[held_names, official_names], pins, device)
    inputs = {"held": development, "official": official}
    pred, concentration, posterior = {}, {}, {}
    for key, eeg in inputs.items():
        _, pred[key], probability = decode_eeg(model, eeg, center, codebooks)
        concentration[key] = probability.amax(-1).mean((0, 1)).tolist()
        posterior[f"{key}/mode"] = probability.argmax(-1).cpu().numpy()
    atomic_file(output / "eeg-code-indices.npz", lambda f: np.savez_compressed(f, **posterior))
    testing = (("dev", "held", slice(200), held_names[:200], dev_visual[:200]),
               ("fresh", "held", slice(200, None), held_names[200:], dev_visual[200:]),
               ("official", "official", slice(None), official_names, test_visual))
    summary, outputs = evaluate(output, frozen, pred, testing, eval_sources, pins)
    oracle = {}
    for panel, _, _, names, targets in testing:
        conditions = quantize(targets - center, codebooks)
        selected = codebooks[torch.arange(len(codebooks), device=center.device), conditions]
        oracle[panel] = {}
        for k in (0, 1, 3, 6):
            vector = targets if k == 0 else F.normalize(center + selected[:, :k].sum(1), dim=-1)
            vals = scores(vector[None], targets, names, eval_sources)
            oracle[panel][str(k)] = dict(zip(METRICS, vals.sum((0, 1)).tolist(), strict=True))
    save_json(output / "readout.json", dict(results=summary, weights=pins.pin(weights), native=native_receipts,
                                            fit=proof, schema=dict(depth=DEPTH, entries=ENTRIES, seed=SEED, epochs=epochs,
                                                                   shape=(1024,), code=pins.pin(Path(__file__))),
                                            code_concentration=concentration, true_visual_conditions=oracle,
                                            scores=outputs, scorer=eval_sources[-1], training=schedule,
                                            codes=pins.pin(output / "eeg-code-indices.npz")))
    pins.verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    require(args.epochs > 0, "A positive training budget is required")
    run(args.benchmark, args.output, args.epochs, args.device)
