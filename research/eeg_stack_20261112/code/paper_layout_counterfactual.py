"""Replace the training EEG response to within-concept changes in image layout."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_aggregate import bootstrap_ci, units
from paper_raw_semantic_readout import leave_photo_out, normalize, read_json, top_credit
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_semantic_contrast import contrast, data_registry
from paper_temporal_readout import BATCH, SEED, TemporalReadout, held_concepts, paired_batches, predict, score, visual_loss


FOLDS = 5
RIDGE = .1
BINS = 40
CHANNELS = 17
EPOCHS = 12
ARMS = {"real-control": 0., "replace-layout": 1., "reverse-layout": -1.}


def concept_residual(values, groups):
    """Remove each concept's average along the penultimate, photographic axis."""
    counts = torch.bincount(groups).to(values)
    shape = (*values.shape[:-2], len(counts), values.shape[-1])
    sums = values.new_zeros(shape).index_add_(-2, groups, values)
    return values - (sums / counts[:, None]).index_select(-2, groups)


def response_scale(observed, estimated):
    """Calibrate the size of a layout change on separate real photographs."""
    squared = estimated.square().sum((1, 2), keepdim=True).clamp_min(1e-8)
    return ((observed * estimated).sum((1, 2), keepdim=True) / squared).clamp(0., 1.)


def fit_layout_response(layout, eeg, groups, folds=FOLDS, ridge=RIDGE, *, calibrate=False):
    """Use only other concepts to predict the layout response of a training photograph."""
    require(folds >= (3 if calibrate else 2), "Response calibration needs separate fit, calibration and query folds")
    design = concept_residual(layout, groups)
    response = concept_residual(eeg, groups)
    prediction = torch.empty_like(eeg)
    uncalibrated = torch.empty_like(eeg)

    def fit(variables, observed):
        covariance = variables.T @ variables
        variance = covariance.diagonal().mean().clamp_min(1e-8)
        covariance.diagonal().add_(ridge * variance)
        return torch.linalg.solve(covariance, variables.T @ observed)

    for fold in range(folds):
        held = groups % folds == fold
        require(bool(held.any()) and bool((~held).any()), "Each layout-response fold must hold out whole concepts")
        estimate = design[held] @ fit(design[~held], response[:, ~held])
        uncalibrated[:, held] = estimate
        if calibrate:
            calibrating = groups % folds == (fold + 1) % folds
            inner = ~(held | calibrating)
            validation = design[calibrating] @ fit(design[inner], response[:, inner])
            estimate = estimate * response_scale(response[:, calibrating], validation)
        prediction[:, held] = estimate
    operator = fit(design, response)
    if calibrate:
        operator = operator * response_scale(response, uncalibrated)
    return prediction, operator


def layout_donors(groups, folds, rng):
    """Assign every photograph a different object in the same auxiliary fit fold."""
    donors = np.empty(len(groups), dtype=np.int64)
    for fold in range(folds):
        rows = np.flatnonzero(groups % folds == fold)
        counts = np.unique(groups[rows], return_counts=True)[1]
        require(len(counts) > 1 and np.all(counts == counts[0]), "Layout swaps require equally sampled concepts")
        order = rows[np.argsort(groups[rows], kind="stable")].reshape(len(counts), counts[0])
        order = order[rng.permutation(len(order))]
        order = np.take_along_axis(order, rng.random(order.shape).argsort(-1), axis=-1)
        shift = int(rng.integers(1, len(order)))
        donors[order.ravel()] = np.roll(order, shift, axis=0).ravel()
    return donors


def replace_layout(eeg, original, alternate, gain):
    return eeg + gain * (alternate - original)


def group_indices(labels):
    return torch.as_tensor(np.unique(labels, return_inverse=True)[1], dtype=torch.long)


def pool_eeg(array, rows, group_size=4):
    """Read consecutive, disjoint groups of physical presentations."""
    require(group_size > 0 and array.shape[1] % group_size == 0, "Split must use every physical presentation once")
    repeats = array.shape[1] // group_size
    output = np.empty((len(rows), repeats, CHANNELS, BINS), dtype=np.float32)
    for first in range(0, len(rows), 64):
        section = rows[first:first + 64]
        scans = array[section, :, :, :200].astype(np.float32)
        scans = scans.reshape(len(section), repeats, group_size, CHANNELS, BINS, 5)
        output[first:first + len(section)] = scans.mean((2, 5))
    return torch.from_numpy(output)


def mix_real_halves(halves, people, anchor, donor):
    return (halves[people, anchor, 0] + halves[people, donor, 1]) * .5


def physical_inputs(raw, train, held, files, *, mix_photos=False):
    ids = {kind: files.pin(raw / f"{kind}_ids.json") for kind in ("train", "test")}
    shapes = {"train": (16540, 4, CHANNELS, 250), "test": (200, 80, CHANNELS, 250)}
    training, development, official, records = [], [], [], {}
    for person in range(1, 11):
        recordings = {}
        proof = {}
        for kind, shape in shapes.items():
            path = raw / f"sub-{person:02d}_{kind}.npy"
            receipt = path.with_suffix(".json")
            source = read_json(receipt)
            digest = files.pin(path)
            require(source["subject"] == f"sub-{person:02d}" and source["split"] == kind
                    and source["ids_sha256"] == ids[kind] and source["output_sha256"] == digest
                    and source["checked_physical_photo_ids"] == shape[0] * shape[1]
                    and source["sample_rate_hz"] == 250, "EEG no longer matches the physical repeat receipt")
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            require(data.shape == shape and data.dtype == np.float16, "Unexpected EEG coordinates")
            recordings[kind] = data
            proof[kind] = dict(array_sha256=digest, receipt_sha256=files.pin(receipt))
        training.append(pool_eeg(recordings["train"], train, group_size=2 if mix_photos else 4))
        development.append(pool_eeg(recordings["train"], held)[:, 0])
        official.append(pool_eeg(recordings["test"], np.arange(200)))
        records[str(person)] = proof
        print("Verified physical EEG", person, flush=True)
    return torch.stack(training), torch.stack(development), torch.stack(official), records


def response_metrics(eeg, predicted, groups):
    residual = concept_residual(eeg, groups)
    order = torch.argsort(groups, stable=True).reshape(-1, 10)
    donor = torch.empty_like(groups)
    donor[order.ravel()] = order.roll(1, dims=1).ravel()
    matched = ((residual - residual[:, donor]) * (predicted - predicted[:, donor])).sum(-1)
    mse = residual.square().mean()
    return dict(r2=float(1 - (residual - predicted).square().mean() / mse),
                response_rms=float(predicted.square().mean().sqrt()),
                remaining_eeg_rms=float(mse.sqrt()), pair_correct=int((matched > 0).sum()), pairs=matched.numel())


def train_model(eeg, response, layout_groups, targets, gain, epochs, *, halves=None):
    device = eeg.device
    torch.manual_seed(SEED)
    model = TemporalReadout(bins=BINS, channels=CHANNELS, segments=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=.025)
    groups = np.argsort(layout_groups, kind="stable").reshape(-1, 10)
    total = epochs * groups.shape[1] * ((len(groups) + BATCH - 1) // BATCH)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total, eta_min=4e-5)
    shuffle = np.random.default_rng(SEED)
    swaps = np.random.default_rng(SEED + 1)
    offsets = shuffle.integers(10, size=eeg.shape[1])
    schedule = sha256()
    mixing = halves is not None
    updates = 0
    for epoch in range(epochs):
        if not mixing:
            donors = layout_donors(layout_groups, FOLDS, swaps)
            schedule.update(donors.astype("<i8").tobytes())
        running = torch.zeros((), device=device)
        for subjects, rows in paired_batches(groups, offsets, epoch, shuffle):
            subject, photo = subjects[0], rows[0]
            if mixing:
                schedule.update(np.stack((subject, photo, rows[2])).astype("<i8").tobytes())
            else:
                schedule.update(np.stack((subject, photo)).astype("<i8").tobytes())
            s = torch.as_tensor(subject, device=device)
            i = torch.as_tensor(photo, device=device)
            original = eeg[s, i]
            if mixing:
                donor = torch.as_tensor(rows[2] if gain else photo, device=device)
                edited = mix_real_halves(halves, s, i, donor)
            else:
                j = torch.as_tensor(donors[photo], device=device)
                edited = replace_layout(original, response[s, i], response[s, j], gain)
            image, prototype, text = (value[i] for value in targets)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                p, c = model(torch.cat((original, edited), dim=0))
                regular, edited = c.chunk(2)
                loss = visual_loss(p[:len(regular)], regular, image, prototype, text)
                loss = loss + .50 * (contrast(edited, prototype) + .50 * contrast(edited, text))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            decay.step()
            running += loss.detach() * len(i)
            updates += 1
        label = "Real EEG mixture" if mixing else "Layout perturbation"
        print(f"{label} {gain:+.0f}, epoch {epoch+1:02d}/{epochs}: {float(running / eeg.shape[1]):.5f}", flush=True)
    require(updates == total, "Training schedule was truncated")
    return model, dict(epochs=epochs, updates=updates, schedule_sha256=schedule.hexdigest())


def score_official(benchmark, output, definitions, category, files, proof, comparisons):
    words_file = benchmark / "semantic-energy-check/h14-text-bank.npz"
    with np.load(words_file, allow_pickle=False) as reference:
        bank, words = normalize(reference["text"]), reference["concept_id"]
    require(files.pin(words_file) == read_json(words_file.with_suffix(".json"))["bank_sha256"], "Wrong naming words")
    registry = benchmark / "inputs/image_set/test_photos.csv"
    with registry.open(encoding="utf-8-sig", newline="") as catalog:
        photographs = list(csv.DictReader(catalog))
    ids = np.array([row["unique_id"] for row in photographs])
    raw_ids = benchmark / "inputs/prep17/test_ids.json"
    image_pack = benchmark / "semantic-energy-check/alignvis-20261127/alignvis-image-teachers.npz"
    image_manifest = read_json(image_pack.with_suffix(".json"))
    require(files.pin(registry) == image_manifest["photo_registries"]["test"]
            and files.pin(image_pack) == image_manifest["array_sha256"]
            and [row["image_id"] for row in photographs] == read_json(raw_ids)["ids"], "Wrong photo scoring archive")
    with np.load(image_pack, allow_pickle=False) as bank_file:
        gallery = normalize(bank_file["test/h14"])
        require(np.array_equal(ids, bank_file["test_word"]) and gallery.shape == (200, 1024)
                and set(category).isdisjoint(ids), "Official names overlap training concepts")
    index = {word: i for i, word in enumerate(words)}
    truth = np.array([index[row] for row in ids])
    metrics = ("name_top1", "name_top5", "original_top1")
    rows = np.arange(200)
    observed = {}
    for arm, name in definitions.items():
        receipt = read_json(name.with_suffix(".json"))
        require(receipt["sources"] == proof and receipt["official_sha256"] == files.pin(name), "Unregistered official prediction")
        with np.load(name, allow_pickle=False) as query:
            require(np.array_equal(query["row"], rows), "Official stimulus order changed")
            q, z = (normalize(query[f]) for f in ("photo_embedding", "embedding"))
        for key, value, photo in ((arm, z, q), (arm + "-wrong", np.roll(z, 1, axis=1), np.roll(q, 1, axis=1))):
            logits = value @ bank.T
            observed[key] = np.stack((top_credit(logits, truth, 1), top_credit(logits, truth, 5),
                                      top_credit(photo @ gallery.T, rows, 1)), axis=-1)
    paired = {}
    for a, b in comparisons:
        delta = observed[a] - observed[b]
        lo, hi = bootstrap_ci(units(delta, ids, "official"))
        paired[f"{a} minus {b}"] = dict(zip(metrics, np.stack((delta.mean((0, 1)), lo, hi), axis=-1).tolist(), strict=True))
    hits = {arm: dict(zip(metrics, measured.sum((0, 1)).tolist(), strict=True)) for arm, measured in observed.items()}
    atomic_file(output / "official-scores.npz", lambda stream: np.savez_compressed(stream, row=rows, concept=ids, **observed))
    save_json(output / "official-readout.json", dict(n=2000, hits=hits, paired=paired, score_sha256=files.pin(output / "official-scores.npz")))
    print("R80 matched subject readout", json.dumps(hits), flush=True)


def run(benchmark, output, device, epochs, dose, mix_photos=False):
    files = PinnedFiles()
    raw, train, dev, labels, category, image, text, layout, visual = data_registry(benchmark)
    held = held_concepts(benchmark, train, dev, labels)
    paired, query, official, recordings = physical_inputs(raw, train, held, files, mix_photos=mix_photos)
    paired, query, official = (value.to(device) for value in (paired, query, official))
    eeg = paired.mean(2)
    groups = group_indices(labels[train]).to(device)
    layout = torch.as_tensor(layout, device=device)
    output.mkdir(parents=True, exist_ok=True)
    center = eeg.mean(1, keepdim=True)
    scale = eeg.std(1, keepdim=True, unbiased=False).clamp_min(1e-5)
    if mix_photos:
        response = None
        halves = (paired - center[:, :, None]) / scale[:, :, None]
        arms = {"real-control": 0., "same-object-mixture": 1.}
        primary = "same-object-mixture"
        pairs = ((primary, "real-control"),)
        method = dict(real_repeat_halves=[[0, 1], [2, 3]], layout_response=None)
    else:
        raw_response, raw_operator = fit_layout_response(layout[train], eeg.flatten(2), groups)
        response, operator = fit_layout_response(layout[train], eeg.flatten(2), groups, calibrate=True)
        measurements = {"oof_train": response_metrics(eeg.flatten(2), response, groups),
                        "oof_train_uncalibrated": response_metrics(eeg.flatten(2), raw_response, groups),
                        "person_scales": (operator.flatten(1).norm(dim=1) / raw_operator.flatten(1).norm(dim=1)).tolist()}
        for panel, region in (("dev", slice(0, 200)), ("fresh", slice(200, 700))):
            selection = held[region]
            membership = group_indices(labels[selection]).to(device)
            vision = concept_residual(layout[selection], membership)
            measurements[panel] = response_metrics(query[:, region].flatten(2), vision @ operator, membership)
            measurements[panel + "_uncalibrated"] = response_metrics(query[:, region].flatten(2), vision @ raw_operator, membership)
        record = output / "layout-response.json"
        save_json(record, measurements)
        print("Layout response", json.dumps(measurements), flush=True)
        response = response.reshape_as(eeg) / scale
        halves = None
        arms = {name: dose * sign for name, sign in ARMS.items()}
        primary = "replace-layout"
        pairs = ((primary, "real-control"), (primary, "reverse-layout"))
        method = dict(response_sha256=files.pin(record), layout_folds=FOLDS, layout_ridge=RIDGE)
    del paired
    eeg, query = (eeg - center) / scale, (query - center) / scale
    official = (official - center[:, :, None]) / scale[:, :, None]
    proof = dict(visual=visual, physical=recordings, epochs=epochs, code=files.pin(Path(__file__)),
                 seed=SEED, bins=BINS, channels=CHANNELS, augmentation=arms, **method)
    visual = image[train]
    semantic = normalize(leave_photo_out(visual, groups.cpu().numpy()))
    targets = tuple(torch.from_numpy(value).to(device, dtype=torch.float32) for value in (visual, semantic, text[labels[train]]))
    official_files = {}
    for arm, gain in arms.items():
        group = output / f"{arm}.npz"
        weights = group.with_suffix(".pt")
        record = group.with_suffix(".json")
        official_name = output / f"{arm}-official.npz"
        official_files[arm] = official_name
        if record.is_file() and official_name.with_suffix(".json").is_file():
            existing = read_json(record)
            if existing["sources"] == proof and files.pin(group) == existing["predictions_sha256"] \
                    and files.pin(weights) == existing["weights_sha256"]:
                print("Reusing signed readout", arm, flush=True)
                continue
        model, schedule = train_model(eeg, response, groups.cpu().numpy(), targets, gain, epochs, halves=halves)
        photo, z = predict(model, query)
        atomic_file(group, lambda f: np.savez_compressed(f, row=held, concept=category[labels[held]],
                                                        photo_embedding=photo, embedding=z))
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        atomic_file(weights, lambda f: torch.save(state, f))
        axes = official.shape[:3]
        p, s = predict(model, official.flatten(1, 2))
        p, s = (normalize(value.reshape(*axes, -1).mean(2)) for value in (p, s))
        atomic_file(official_name, lambda f: np.savez_compressed(f, row=np.arange(200), photo_embedding=p, embedding=s))
        save_json(record, dict(arm=arm, sources=proof, weights_sha256=files.pin(weights),
                               predictions_sha256=files.pin(group), **schedule))
        save_json(official_name.with_suffix(".json"), dict(arm=arm, sources=proof, official_sha256=files.pin(official_name), repeats=80))
    score(output, benchmark, held, labels, category, image, layout.cpu().numpy(), proof, arms,
          comparisons=pairs, report_name="layout-readout.json")
    score_official(benchmark, output, official_files, category, files, proof, (*pairs, (primary, primary + "-wrong")))
    files.verify()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("benchmark", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    cli.add_argument("--epochs", type=int, default=EPOCHS)
    cli.add_argument("--dose", type=float, choices=(1., 4.), default=1., help="Compare the calibrated and amplified layout responses")
    cli.add_argument("--real-photo-mixture", action="store_true", help="Train with two repeats each from two same-object photos")
    args = cli.parse_args()
    require(args.epochs > 0, "Training must include at least one complete epoch")
    torch.set_num_threads(4)
    require(not args.real_photo_mixture or args.dose == 1., "The response dose applies only to synthetic layout changes")
    run(args.benchmark, args.output, torch.device(args.device), args.epochs, args.dose, args.real_photo_mixture)
