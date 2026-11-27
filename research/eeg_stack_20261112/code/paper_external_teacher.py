"""Train a single-person Brain-HiVE encoder with real cross-acquisition EEG targets."""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_paired_encoder import SUBJECTS
from paper_render_cache import PinnedFiles, atomic_file, require, save_json, sha
from paper_repeat_semantics import photo_loss
from paper_token_evidence import ridge_weights

EXTERNAL_SUBJECTS = tuple(range(1, 13))
CONTROL, INTERNAL, CANDIDATE, SIBLING = ("photo-control", "internal-peer", "external-photo", "external-sibling")
MODES = (CONTROL, INTERNAL, CANDIDATE, SIBLING)
TEMPERATURE, SEED = .07, 20261129
EPOCHS, BATCH = 6, 128


def safe_alignment(context, actual, fold):
    fit = (context & (1 << fold)) == 0
    predict = actual == fold
    require(not np.any(fit & predict), "A target training fold appears in the context of an alignment sample")
    return fit, predict


def covered_donors(labels, counts):
    rng = np.random.default_rng(SEED)
    donor = np.full(len(labels), -1, dtype=np.int64)
    support = np.zeros(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        valid = np.flatnonzero((labels == label) & (counts > 0))
        if len(valid) >= 2:
            order = rng.permutation(valid)
            donor[order] = np.roll(order, 1)
            support[order] = np.minimum(counts[order], counts[donor[order]])
    return donor, support


def fused_teacher(internal, extra, donor, count):
    direction = F.normalize(internal)
    external = F.normalize(extra)
    coefficient = .35 * (count / 3).clamp(max=1)
    substitute = external[donor.clamp_min(0)]
    return tuple(F.normalize(direction + coefficient[:, None] * item) for item in (external, substitute))


def early_bins(wave):
    return wave[..., :60].float().reshape(-1, 17, 12, 5).mean(-1).flatten(1)


def read_external(folder, rows, folds, proof, files):
    packets = []
    reference_manifest = None
    for person in EXTERNAL_SUBJECTS:
        path = folder / f"sub-{person:02d}-registered-context.npz"
        meta = path.with_suffix(".json")
        entry = json.loads(meta.read_bytes())
        sources = entry["files"]
        receipt = dict(array=files.pin(path), receipt=files.pin(meta))
        require(receipt["array"] == entry["sha256"] and entry["subject"] == f"sub-{person:02d}"
                and sources["split"] == proof["split"] and sources["concept_ids"] == proof["labels"]
                and sources["photo_index"] == proof["photo_index"]
                and entry["context_excludes_all_unregistered_objects"] and entry["context_seconds"] == [-.30, .24]
                and entry["channels"] == proof["channels"]
                and entry["rate"] == 250 and entry["max_marker_discrepancy_ms"] < 4.,
                "External acquisition used different photos or timing")
        if reference_manifest is None:
            reference_manifest = sources["manifest"]
        require(sources["manifest"] == reference_manifest, "External data release changed across participants")
        with np.load(path, allow_pickle=False) as package:
            raw, context = package["eeg"], package["context_fold_mask"]
            location = np.searchsorted(rows, package["train_row"])
            require(np.all(location < len(rows)) and np.array_equal(rows[location], package["train_row"]),
                    "External EEG includes a photograph outside the registered training rows")
            require(len(np.unique(location)) == len(location)
                    and raw.shape == (len(location), 17, 60)
                    and raw.dtype == np.float32 and np.isfinite(raw).all()
                    and context.shape == location.shape and not np.any(context & ~30)
                    and np.all((context & (1 << folds[location])) > 0),
                    "Invalid external waveform geometry or context")
        proof["external"][str(person)] = {**receipt, "events": int(len(location))}
        packets.append((early_bins(torch.from_numpy(raw).cuda()), location, context))
    return packets


def align_half(source, target, allowed, held):
    x, y = source[allowed], target[allowed]
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-5)
    target_mean = y.mean(0)
    weight = ridge_weights((x - mean) / scale, y - target_mean, .5)
    return ((source[held] - mean) / scale) @ weight + target_mean


def diagnostic(predicted, reference, labels, support):
    photos = F.normalize(reference)
    query = F.normalize(predicted)
    _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
    require(torch.all(counts == 10), "Each training object must have ten distinct photographs")
    member = inverse.argsort(stable=True).reshape(-1, 10)[inverse]
    scores = (query[:, None, :] * photos[member]).sum(2)
    i = torch.arange(len(labels), device=query.device)
    hits = member[i, scores.argmax(1)] == i
    correct = (query * photos).sum(1)
    average = (scores.sum(1) - correct) / 9.
    return dict(queries=int(support.sum()), photo_top1=int((hits & support).sum()),
                own_minus_siblings=(correct - average)[support].mean().item())


@torch.inference_mode()
def build(subjects):
    from common import CH17, PREP
    from paper_oof_conditions import registered_rows, study_root
    from paper_render import physical_eeg

    root, files = study_root(), PinnedFiles()
    rows, labels, folds = registered_rows(PREP)
    require(set(folds) == {1, 2, 3, 4} and np.all(np.diff(rows) > 0), "Unexpected concept fold registration")
    train_f = PREP / "F_train.npy"
    src = dict(split=files.pin(PREP / "paper_split.json"), labels=files.pin(PREP / "train_concept.npy"),
               photo_index=files.pin(PREP / "train_ids.json"), photos=files.pin(train_f), external={}, internal={},
               code={}, channels=CH17)
    for f in (build, registered_rows, physical_eeg, ridge_weights):
        file = Path(inspect.getfile(f))
        src["code"][file.name] = files.pin(file)
    recordings = {}
    for person in SUBJECTS:
        raw, receipt = physical_eeg(person, "train", files)
        src["internal"][str(person)] = receipt
        recordings[person] = torch.cat([early_bins(torch.from_numpy(raw[part][..., :60].copy()).cuda().float().mean(1))
                                        for part in np.array_split(rows, 55)])
    bank = read_external(root / "epochs", rows, folds, src, files)
    reference = torch.from_numpy(np.load(train_f, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    require(reference.shape == (len(rows), 1024) and torch.isfinite(reference).all(), "Invalid native visual bank")
    concepts = torch.from_numpy(labels).cuda()
    for recipient in subjects:
        others = [person for person in SUBJECTS if person != recipient]
        internal = torch.empty_like(reference)
        external = torch.zeros_like(reference)
        supports = torch.zeros(len(rows), device="cuda", dtype=torch.long)
        records = {}
        predicted_all = {}
        for held_fold in (1, 2, 3, 4):
            hold = folds == held_fold
            samples = ~hold
            views = [recordings[person] for person in others]
            mean = torch.stack([(v-v[samples].mean(0)) / v[samples].std(0).clamp_min(1e-5) for v in views]).mean(0)
            center = mean[samples].mean(0)
            translated = mean - center
            output = reference[samples]
            mapping = ridge_weights(translated[samples], output - output.mean(0), .3)
            internal[hold] = translated[hold] @ mapping
            records[str(held_fold)] = {}
            for e, (source, positions, contexts) in enumerate(bank, 1):
                fit_mask, infer = safe_alignment(contexts, folds[positions], held_fold)
                require(fit_mask.sum() >= 2 and infer.any(), "Insufficient verified cross-acquisition calibration rows")
                aligned = align_half(source, mean[positions], fit_mask, infer)
                idx = torch.from_numpy(positions[infer]).cuda()
                external.index_add_(0, idx, (aligned - center) @ mapping)
                supports.index_add_(0, idx, torch.ones_like(idx))
                records[str(held_fold)][str(e)] = [int(fit_mask.sum()), int(infer.sum())]
        average = external / supports.clamp_min(1)[:, None]
        donor, count = covered_donors(labels, supports.cpu().numpy())
        require(np.count_nonzero(count), "No pair of photos shares external EEG coverage")
        donor_gpu, count_gpu = (torch.from_numpy(x).cuda() for x in (donor, count))
        true, permuted = fused_teacher(internal, average, donor_gpu, count_gpu.float())
        teachers = {INTERNAL: F.normalize(internal), CANDIDATE: true, SIBLING: permuted}
        require(all(torch.isfinite(t).all() for t in teachers.values()), "Teacher latent directions are not finite")
        for arm, condition in teachers.items():
            predicted_all[arm] = diagnostic(condition, reference-reference.mean(0), concepts, count_gpu > 0)
        name = root / f"teachers-sub-{recipient:02d}.pt"
        content = dict(recipient=recipient, peers=others, rows=torch.from_numpy(rows), donor=torch.from_numpy(donor),
                       matches=supports.cpu(), support=torch.from_numpy(count),
                       directions={arm: value.cpu() for arm, value in teachers.items()}, proof=src)
        files.verify()
        atomic_file(name, lambda f: torch.save(content, f))
        receipt = dict(subject=recipient, peers=others, source_epochs=int(supports.sum()),
                       images_with_external=int((supports > 0).sum()), supported_comparison=int((count_gpu > 0).sum()),
                       coverage_histogram=torch.bincount(supports).tolist(), exact_teacher_diagnostic=predicted_all,
                       calibration=records, sha256=sha(name), proof=src)
        save_json(name.with_suffix(".json"), receipt)
        print("teacher bank", recipient, receipt["source_epochs"], receipt["supported_comparison"], predicted_all, flush=True)


def auxiliary(student, teacher, prototypes, weight):
    q = F.normalize(student)
    direction = 1 - (q * teacher).sum(1)
    student_logits = F.log_softmax(q @ prototypes.T / TEMPERATURE, dim=1)
    teacher_logits = F.softmax(teacher @ prototypes.T / TEMPERATURE, dim=1)
    objects = F.kl_div(student_logits, teacher_logits, reduction="none").sum(1)
    return ((direction + .25 * objects) * weight).sum() / weight.sum().clamp_min(1.)


def fit(person):
    from common import PREP, RUNS
    from paper_instance_residual import make_encoder
    from paper_object_tokens import epoch_batches, group_index
    from paper_oof_conditions import registered_rows, study_root
    from paper_render import physical_eeg

    root, files = study_root(), PinnedFiles()
    require(RUNS.name == f"personal-{person:02d}", "Expected the participant's native training directory")
    rows, labels, _ = registered_rows(PREP)
    checkpoint = RUNS / f"enc/native/sub-{person:02d}/fold0_seed2025/brain_model.pt"
    weights = root / f"teachers-sub-{person:02d}.pt"
    raw, receipt = physical_eeg(person, "train", files)
    fit_source = dict(base=files.pin(checkpoint), split=files.pin(PREP / "paper_split.json"),
                      photos=files.pin(PREP / "F_train.npy"), eeg=receipt, teachers=files.pin(weights),
                      code={}, training_subject=person)
    for f in (fit, photo_loss, epoch_batches, make_encoder, physical_eeg):
        file = Path(inspect.getfile(f))
        fit_source["code"][file.name] = files.pin(file)
    teachers = torch.load(weights, map_location="cuda", weights_only=True)
    require(teachers["recipient"] == person and teachers["peers"] == [s for s in SUBJECTS if s != person]
            and np.array_equal(teachers["rows"].cpu().numpy(), rows)
            and teachers["proof"]["split"] == fit_source["split"] and teachers["proof"]["photos"] == fit_source["photos"]
            and teachers["proof"]["internal"][str(person)] == receipt,
            "Teacher target and real EEG training photos differ")
    photo = torch.from_numpy(np.load(PREP / "F_train.npy", mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    eeg = torch.cat([torch.from_numpy(raw[part].copy()).cuda().float().mean(1) for part in np.array_split(rows, 60)])
    require(torch.isfinite(eeg).all(), "Nonfinite real EEG")
    initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
    loc = photo.mean(0)
    variance = (photo - loc).square().mean().clamp_min(1e-6)
    _, members = group_index(torch.from_numpy(labels).cuda())
    prototypes = F.normalize(photo[members].mean(1) - loc)
    weight = (teachers["support"].float() / 3).clamp(max=1.)
    steps = EPOCHS * members.shape[1] * math.ceil(len(members) / BATCH)
    out = root / f"fits/sub-{person:02d}"
    out.mkdir(parents=True, exist_ok=True)
    schedules = set()
    for arm in MODES:
        path = out / f"{arm}.pt"
        receipt_path = path.with_suffix(".json")
        if path.is_file() and receipt_path.is_file():
            previous = json.loads(receipt_path.read_bytes())
            require(previous["inputs"] == fit_source and previous["arm"] == arm
                    and previous["checkpoint_sha256"] == sha(path) and previous["steps"] == steps,
                    "Saved model used a different training experiment")
            schedules.add(previous["training_schedule_sha256"])
            continue
        target = teachers["directions"][INTERNAL if arm == CONTROL else arm]
        alpha = 0. if arm == CONTROL else .5
        torch.manual_seed(SEED)
        learner = make_encoder(torch.device("cuda")).float().train()
        learner.load_state_dict(initial)
        g = torch.Generator(device="cuda").manual_seed(SEED)
        optimizer = torch.optim.AdamW(learner.backbone.parameters(), lr=7e-5, betas=(.9, .995), weight_decay=.05)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=8e-6)
        trace = hashlib.sha256()
        history = []
        accumulator = torch.zeros(2, device="cuda")
        step = 0
        for _ in range(EPOCHS):
            for ids in epoch_batches(members, g, batch_size=BATCH):
                step += 1
                trace.update(ids.cpu().numpy().astype("<i8").tobytes())
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    estimate = learner.backbone(eeg[ids]).float()
                exact = photo_loss(estimate, photo[ids], loc, variance)
                distill = auxiliary(estimate - loc, target[ids], prototypes, weight[ids])
                (exact + alpha * distill).backward()
                torch.nn.utils.clip_grad_norm_(learner.backbone.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                accumulator += torch.stack((exact.detach(), distill.detach()))
                if step % 100 == 0 or step == steps:
                    last = history[-1]["step"] if history else 0
                    entry = dict(step=step, terms=(accumulator / (step - last)).tolist())
                    history.append(entry)
                    accumulator.zero_()
                    print(f"participant {person:02d} {arm}", entry, flush=True)
        require(step == steps, "Training batch count changed")
        state = {k: v.detach().cpu() for k, v in learner.state_dict().items()}
        files.verify()
        atomic_file(path, lambda f: torch.save(state, f))
        fingerprint = trace.hexdigest()
        save_json(receipt_path, dict(participant=person, arm=arm, steps=steps, inputs=fit_source,
                    checkpoint_sha256=sha(path), training_schedule_sha256=fingerprint, history=history, weight=alpha))
        schedules.add(fingerprint)
    require(len(schedules) == 1, "Matched training treatments have different sample order")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="task", required=True)
    prep = commands.add_parser("build")
    prep.add_argument("subjects", nargs="+", type=int, choices=SUBJECTS)
    train = commands.add_parser("fit")
    train.add_argument("subject", type=int, choices=SUBJECTS)
    args = cli.parse_args()
    if args.task == "build":
        build(args.subjects)
    else:
        fit(args.subject)
