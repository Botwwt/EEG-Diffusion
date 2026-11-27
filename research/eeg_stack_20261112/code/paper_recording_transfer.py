"""Train a shared early readout directly on two visual EEG acquisitions."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paper_raw_semantic_readout import leave_photo_out, normalize, read_json
from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_semantic_contrast import contrast, data_registry
from paper_temporal_readout import BATCH, EPOCHS, SEED, TemporalReadout, held_concepts, paired_batches, score, visual_loss


ARMS = ("eeg2-same-photo", "eeg2-other-photo", "eeg1-same-photo", "eeg1-other-photo")
PEOPLE = 10
EXTERNAL_PEOPLE = 12
CHANNELS = 17
BIN = 2
BINS = 100
EARLY = 30
RATE = 250
AUX_VISUAL = .35
AUX_PAIR = .40


class AcquisitionReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = TemporalReadout(width=64, bins=EARLY, segments=2, channels=CHANNELS)
        self.late = TemporalReadout(width=64, bins=BINS - EARLY, segments=3, channels=CHANNELS)
        self.mix = nn.Parameter(torch.zeros(2))

    def forward(self, eeg):
        early = self.early(eeg[..., :EARLY])
        late = self.late(eeg[..., EARLY:])
        weight = self.mix.sigmoid().unbind()
        fused = tuple(F.normalize(torch.lerp(a, b, w), dim=-1) for a, b, w in zip(early, late, weight, strict=True))
        return early, fused


def select_external(present, photo, offset):
    candidates = present[:, photo].T
    order = np.argsort(~candidates, axis=1, kind="stable")
    count = candidates.sum(1)
    require((count > 0).all(), "Every selected photograph must have a measured external EEG")
    return order[np.arange(len(photo)), offset % count]


def bins(values, n):
    return values.reshape(*values.shape[:-1], n, BIN).mean(-1, dtype=np.float32)


def standardize(values, fit, axes):
    center = fit.mean(axes, dtype=np.float64).astype(np.float32)
    scale = fit.std(axes, dtype=np.float64).astype(np.float32).clip(1e-5)
    return ((values - center) / scale).astype(np.float16), np.stack((center, scale))


def read_eeg2(folder, rows, n_train, identities, pins):
    continuous = np.empty((PEOPLE, len(rows), CHANNELS, BINS), dtype=np.float16)
    singles = np.empty((PEOPLE, n_train, 4, CHANNELS, EARLY), dtype=np.float16)
    evidence, normalized, montage = {}, ([], []), None
    for subject in range(1, PEOPLE + 1):
        path = folder / f"sub-{subject:02d}_train.npy"
        manifest = path.with_suffix(".json")
        details = read_json(manifest)
        source = dict(array=pins.pin(path), manifest=pins.pin(manifest))
        channels = details["channel_names"]
        if montage is None:
            montage = channels
        require(source["array"] == details["output_sha256"] and details["ids_sha256"] == identities
                and details["subject"] == f"sub-{subject:02d}" and channels == montage and len(montage) == CHANNELS
                and details["sample_rate_hz"] == RATE and details["checked_physical_photo_ids"] == 66160
                and details["axes"] == ["stimulus", "physical_repeat", "channel", "sample"],
                "A registered THINGS-EEG2 recording has changed")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        require(raw.shape == (16540, 4, CHANNELS, 250) and raw.dtype == np.float16, "Expected four physical repeats")
        erp = np.empty(continuous.shape[1:], dtype=np.float32)
        trials = np.empty(singles.shape[1:], dtype=np.float32)
        for a in range(0, len(rows), 128):
            b = min(a + 128, len(rows))
            samples = raw[rows[a:b], :, :, :BINS * BIN].astype(np.float32)
            erp[a:b] = bins(samples.mean(1), BINS)
            if a < n_train:
                trials[a:min(b, n_train)] = bins(samples[:n_train - a, ..., :EARLY * BIN], EARLY)
        require(np.isfinite(erp).all() and np.isfinite(trials).all(), "Nonfinite EEG2 samples")
        continuous[subject - 1], erp_norm = standardize(erp, erp[:n_train], 0)
        singles[subject - 1], trial_norm = standardize(trials, trials, (0, 1))
        normalized[0].append(erp_norm)
        normalized[1].append(trial_norm)
        evidence[str(subject)] = source
        print(f"Verified EEG2 sub-{subject:02d} and four physical measurements", flush=True)
    return continuous, singles, montage, evidence, tuple(np.stack(s) for s in normalized)


def read_eeg1(folder, training, montage, proof, pins):
    output = np.zeros((EXTERNAL_PEOPLE, len(training), CHANNELS, EARLY), dtype=np.float16)
    present = np.zeros(output.shape[:2], dtype=bool)
    evidence, stats = {}, []
    data_release = None
    index = {photo: offset for offset, photo in enumerate(training)}
    for subject in range(1, EXTERNAL_PEOPLE + 1):
        path = folder / f"sub-{subject:02d}-registered-context.npz"
        manifest = path.with_suffix(".json")
        details = read_json(manifest)
        source = details["files"]
        if data_release is None:
            data_release = source["manifest"]
        checksum = pins.pin(path)
        require(checksum == details["sha256"] and details["subject"] == f"sub-{subject:02d}"
                and details["channels"] == montage and details["rate"] == RATE
                and details["context_excludes_all_unregistered_objects"] and details["context_seconds"] == [-.30, .24]
                and details["max_marker_discrepancy_ms"] < 4.
                and source["manifest"] == data_release and source["split"] == proof["split"]
                and source["concept_ids"] == proof["labels"] and source["photo_index"] == proof["identities"],
                "A registered THINGS-EEG1 recording has changed")
        with np.load(path, allow_pickle=False) as archive:
            rows, wave, event, contexts = (archive[k] for k in ("train_row", "eeg", "source_event", "context_fold_mask"))
        n = len(rows)
        require(len(set(rows)) == n == details["retained_photos"] and all(photo in index for photo in rows)
                and wave.shape == (n, CHANNELS, EARLY * BIN) and wave.dtype == np.float32 and np.isfinite(wave).all()
                and event.shape == contexts.shape == (n,) and np.all(np.diff(event) > 0)
                and np.all(contexts > 0) and not np.any(contexts & ~30),
                "External EEG must use only registered training photographs and events")
        positions = np.array([index[photo] for photo in rows])
        short = bins(wave, EARLY)
        output[subject - 1, positions], norm = standardize(short, short, 0)
        present[subject - 1, positions] = True
        stats.append(norm)
        evidence[str(subject)] = dict(array=checksum, manifest=pins.pin(manifest), events=n)
        print(f"Verified EEG1 sub-{subject:02d} and {n} independent training measurements", flush=True)
    return output, present, evidence, np.stack(stats)


def fit(arm, eeg, held, eeg2_peer, eeg1_peer, present, groups, targets, epochs, checkpoint):
    torch.manual_seed(SEED)
    device = eeg.device
    model = AcquisitionReadout().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=.025)
    updates = epochs * groups.shape[1] * ((len(groups) + BATCH - 1) // BATCH)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=4e-5)
    random = np.random.default_rng(SEED)
    offsets = random.integers(PEOPLE, size=eeg.shape[1])
    repeat_offsets = random.integers(4, size=eeg.shape[1])
    acquisition_offsets = random.integers(EXTERNAL_PEOPLE, size=eeg.shape[1])
    covered = present.any(0)
    anchor_schedule, peer_schedule = hashlib.sha256(), hashlib.sha256()
    external = arm.startswith("eeg1-")
    other_photo = arm.endswith("other-photo")
    history, stepped, paired_views = [], 0, 0
    for epoch in range(epochs):
        stats = torch.zeros(3, device=device)
        for people, rows in paired_batches(groups, offsets, epoch, random):
            photo = rows[0]
            sibling = rows[2]
            valid = np.flatnonzero(covered[photo] & covered[sibling])
            peer_rows = np.stack((photo[valid], sibling[valid]))
            anchor_schedule.update(np.concatenate((people, rows)).astype("<i8").tobytes())
            anchor_schedule.update(valid.astype("<i8").tobytes())
            peer_idx = torch.as_tensor(peer_rows, device=device)
            a = torch.as_tensor(photo, device=device)
            supported = torch.as_tensor(valid, device=device)
            if external:
                p = peer_rows.ravel()
                subject = select_external(present, p, acquisition_offsets[p] + epoch).reshape(peer_rows.shape)
                peer_schedule.update(subject.astype("<i8").tobytes())
                short = eeg1_peer[torch.as_tensor(subject, device=device), peer_idx]
            else:
                subject = people[1, valid]
                repeats = (repeat_offsets[peer_rows] + epoch) % 4
                peer_schedule.update(subject.astype("<i8").tobytes())
                peer_schedule.update(repeats.astype("<i8").tobytes())
                short = eeg2_peer[torch.as_tensor(subject[None, :], device=device), peer_idx,
                                  torch.as_tensor(repeats, device=device)]
            inputs = eeg[torch.as_tensor(people[0], device=device), a].float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                (_, early_semantic), (photo_fit, semantic_fit) = model(inputs)
                supervised = visual_loss(photo_fit, semantic_fit, *(x[a] for x in targets))
                require(len(valid) > 1, "Expected matched auxiliary data within each training batch")
                peer_photo, peer_semantic = (v.reshape(2, len(valid), -1) for v in model.early(short.flatten(0, 1).float()))
                visual = torch.stack([visual_loss(peer_photo[j], peer_semantic[j], *(x[peer_idx[j]] for x in targets)) for j in range(2)]).mean()
                alignment = contrast(early_semantic[supported], peer_semantic[int(other_photo)])
                objective = supervised + AUX_VISUAL * visual + AUX_PAIR * alignment
            opt.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            opt.step()
            schedule.step()
            stats += torch.stack((supervised.detach(), visual.detach(), alignment.detach())) * len(photo)
            stepped += 1
            paired_views += len(valid)
        history.append((stats / eeg.shape[1]).cpu().tolist())
        print(f"{arm} {epoch + 1:02d}/{epochs}: {history[-1]}", flush=True)
    require(stepped == updates, "Training did not follow its registered schedule")
    model.eval()
    with torch.inference_mode():
        both = [tuple(v.cpu() for v in model(x.float())[1]) for x in held.flatten(0, 1).split(256)]
    photo, semantic = (torch.cat(v).reshape(*held.shape[:2], -1).numpy() for v in zip(*both, strict=True))
    require(np.isfinite(photo).all() and np.isfinite(semantic).all(), "Nonfinite held-out readout")
    atomic_file(checkpoint, lambda f: torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f))
    return photo, semantic, dict(updates=updates, paired_examples=paired_views, schedule_sha256=anchor_schedule.hexdigest(),
                                 peer_schedule_sha256=peer_schedule.hexdigest(), losses=history,
                                 late_mix=model.mix.sigmoid().detach().cpu().tolist())


def run(benchmark, other_recording, output, device):
    source, training, dev, labels, category, image, text, layout, data = data_registry(benchmark)
    held = held_concepts(benchmark, training, dev, labels)
    rows = np.concatenate((training, held))
    n = len(training)
    pins = PinnedFiles()
    identity = pins.pin(benchmark / "verified/data/prep/train_ids.json")
    full, short, montage, eeg2_proof, (erp_norm, trial_norm) = read_eeg2(source, rows, n, data["physical_ids"], pins)
    external, present, eeg1_proof, outside_norm = read_eeg1(other_recording, training, montage, dict(data, identities=identity), pins)
    group, counts = np.unique(labels[training], return_inverse=True, return_counts=True)[1:]
    require(len(counts) == 1323 and np.all(counts == 10), "Expected ten separate training photos per concept")
    groups = np.argsort(group, kind="stable").reshape(1323, 10)
    photo = image[training]
    targets = [torch.from_numpy(a).to(device=device, dtype=torch.float32)
               for a in (photo, normalize(leave_photo_out(photo, group)), text[labels[training]])]
    output.mkdir(parents=True, exist_ok=True)
    parameters = output / "normalization.npz"
    if parameters.exists():
        with np.load(parameters, allow_pickle=False) as saved:
            require(all(np.array_equal(saved[key], item) for key, item in
                        (("full", erp_norm), ("single", trial_norm), ("external", outside_norm))), "Training normalization changed")
    else:
        atomic_file(parameters, lambda f: np.savez_compressed(f, full=erp_norm, single=trial_norm, external=outside_norm))
    proof = dict(data=data, eeg2=eeg2_proof, eeg1=eeg1_proof, identity=identity, head=pins.pin(Path(__file__)),
                 temporal_code=pins.pin(Path(__file__).with_name("paper_temporal_readout.py")), epochs=EPOCHS, seed=SEED,
                 full_repeats=4, peer_repeats=1, normalization=pins.pin(parameters), binned_samples=BIN,
                 sample_hz=RATE, early_bins=EARLY, bins=BINS, montage=montage, aux_visual=AUX_VISUAL, aux_pair=AUX_PAIR,
                 covered_external_photos=int(present.any(0).sum()), external_measurements=int(present.sum()), batch=BATCH)
    eeg, eeg2_peer, eeg1_peer = (torch.from_numpy(x).to(device) for x in (full, short, external))
    del full, short, external
    peer_schedule = {}
    counts = set()
    for arm in ARMS:
        predictions = output / f"{arm}.npz"
        receipt = predictions.with_suffix(".json")
        checkpoint = predictions.with_suffix(".pt")
        if receipt.exists():
            result = read_json(receipt)
            require(result["sources"] == proof and result["predictions_sha256"] == pins.pin(predictions)
                    and result["weights_sha256"] == pins.pin(checkpoint), "Existing checkpoint belongs to another run")
        else:
            p, s, log = fit(arm, eeg[:, :n], eeg[:, n:], eeg2_peer, eeg1_peer, present, groups, targets, EPOCHS, checkpoint)
            atomic_file(predictions, lambda f: np.savez_compressed(f, embedding=s, photo_embedding=p,
                                                                   row=held, concept=category[labels[held]]))
            result = dict(arm=arm, sources=proof, predictions_sha256=pins.pin(predictions),
                          weights_sha256=pins.pin(checkpoint), **log)
            save_json(receipt, result)
        counts.add(result["paired_examples"])
        peer_schedule[arm] = result["peer_schedule_sha256"]
    require(len(counts) == 1 and all(peer_schedule[f"eeg{a}-same-photo"] == peer_schedule[f"eeg{a}-other-photo"] for a in (1, 2)),
            "Comparison altered the number or sequence of auxiliary EEG measurements")
    pins.verify()
    comparisons = (("eeg1-same-photo", "eeg2-same-photo"), ("eeg1-other-photo", "eeg2-other-photo"),
                   ("eeg1-other-photo", "eeg1-same-photo"), ("eeg2-other-photo", "eeg2-same-photo"))
    score(output, benchmark, held, labels, category, image, layout, proof, ARMS,
          comparisons=comparisons, report_name="recording-transfer-verified.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("eeg1", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(5)
    run(args.benchmark, args.eeg1, args.output, torch.device(args.device))
