"""Fine-tune a native personal encoder with matched physical-repeat and training-name supervision."""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, require, save_json, sha

ARMS = {
    "photo": (0., 0.),
    "repeat": (.25, 0.),
    "language": (0., .25),
    "language-repeat": (.25, .25),
}
SEED = 20261128
EPOCHS, OBJECTS_PER_BATCH = 8, 96
TEMPERATURE = .07


def split_trials(sessions, orientations):
    order = sessions.argsort(dim=1, stable=True).reshape(-1, 2, 2)
    first = order.gather(2, orientations.unsqueeze(-1)).squeeze(-1)
    second = order.gather(2, (1 - orientations).unsqueeze(-1)).squeeze(-1)
    return torch.stack((first, second), dim=1)


def paired_batches(members, epochs, objects_per_batch, seed):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.random(members.shape).argsort(axis=1)
        photos = np.take_along_axis(members, order, axis=1).reshape(len(members), -1, 2)
        for pair in photos.transpose(1, 0, 2):
            shuffle = rng.permutation(len(pair))
            for start in range(0, len(shuffle), objects_per_batch):
                yield pair[shuffle[start:start + objects_per_batch]]


def matched_margin(first, second):
    a, b = [F.normalize(x.float(), dim=-1) for x in (first, second)]
    difference = ((a[:, 0] - a[:, 1]) * (b[:, 0] - b[:, 1])).sum(dim=-1)
    return F.softplus(-difference / (2 * TEMPERATURE)).mean()


def apply_language_targets(names, object_ids):
    prototypes, mapping = torch.unique(names, dim=0, return_inverse=True)
    return prototypes, mapping[object_ids]


def photo_loss(prediction, photo, center, scale):
    logits = F.normalize(prediction - center) @ F.normalize(photo - center).T / TEMPERATURE
    index = torch.arange(len(photo), device=photo.device)
    return .5 * (F.cross_entropy(logits, index) + F.cross_entropy(logits.T, index)) + F.mse_loss(prediction, photo) / scale


def select_candidate(stats):
    native = stats["native-paper"]
    native_total = sum(item["full_object"] for item in native.values())
    native_photos = sum(item["same_object_photo"] for item in native.values())
    control = sum(item["full_object"] for item in stats["photo"].values())
    passed = []
    for arm in ("repeat", "language", "language-repeat"):
        table = stats[arm]
        names = sum(item["full_object"] for item in table.values())
        photos = sum(item["same_object_photo"] for item in table.values())
        if (all(table[s]["full_object"] > native[s]["full_object"] for s in native)
                and names >= native_total + 4 and names > control and photos >= native_photos - 4):
            passed.append((names, photos, -sum(x > 0 for x in ARMS[arm]), arm))
    return max(passed)[-1] if passed else None


def training_data(subject, sessions_root, files):
    from common import PREP, RUNS
    from paper_instance_residual import make_encoder
    from paper_oof_conditions import registered_rows
    from paper_render import physical_eeg
    from paper_soft_text import ids_sha
    from paper_token_evidence import ridge_weights

    require(RUNS.name == f"personal-{subject:02d}", "Select the participant's own Brain-HiVE directory")
    rows, labels, _ = registered_rows(PREP)
    concepts, objects, counts = np.unique(labels, return_inverse=True, return_counts=True)
    require(len(concepts) == 1323 and np.all(counts == 10), "Expected exactly ten training photographs per object")
    members = objects.argsort(kind="stable").reshape(-1, 10)
    source = PREP / "F_train.npy"
    split = PREP / "paper_split.json"
    names = RUNS.parent / "shared/enc/soft_text/fold14_space.pt"
    session_path = sessions_root / f"sub-{subject:02d}_train.npy"
    eeg_metadata = PREP / f"sub-{subject:02d}_train.json"
    photos_index = PREP / "train_ids.json"
    checkpoint = RUNS / f"enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
    raw, receipt = physical_eeg(subject, "train", files)
    proof = dict(eeg=receipt, base=files.pin(checkpoint), sessions=files.pin(session_path),
                 split=files.pin(split), photos=files.pin(source), names=files.pin(names),
                 labels=files.pin(PREP / "train_concept.npy"), code={})
    for source_function in (training_data, make_encoder, registered_rows, physical_eeg, ridge_weights, ids_sha):
        file = Path(inspect.getfile(source_function))
        proof["code"][file.name] = files.pin(file)
    require(proof["sessions"] == json.loads(eeg_metadata.read_bytes())["session_sha256"],
            "Physical order of sessions differs from the EEG preparation receipt")
    sessions = np.load(session_path, allow_pickle=False)[rows]
    sorted_sessions = np.sort(sessions, axis=1)
    require(sessions.shape == (len(rows), 4) and np.isin(sessions, np.arange(4)).all()
            and np.all(sorted_sessions[:, [0, 2]] == sorted_sessions[:, [1, 3]])
            and np.all(sorted_sessions[:, 0] != sorted_sessions[:, 2])
            and np.all(sorted_sessions[members] == sorted_sessions[members[:, :1]]),
            "Each object must have the same two sessions and each photo two repeats in each")
    text = torch.load(names, map_location="cuda", weights_only=True)
    index = json.loads(photos_index.read_bytes())["ids"]
    text_source = text["meta"]["inputs"]
    require(text_source["split"] == proof["split"]
            and text_source["training_concepts"] == ids_sha(concepts.tolist())
            and text_source["train_ids"] == ids_sha([index[i] for i in rows])
            and text["final_codes"].shape == (len(concepts), text["meta"]["rank"]),
            "The training names use a different set of registered photographs")
    prototypes, positives = apply_language_targets(text["final_codes"], torch.from_numpy(objects).cuda())
    photo = torch.from_numpy(np.load(source, mmap_mode="r", allow_pickle=False)[rows].copy()).cuda().float()
    photo_center, name_center = photo.mean(0), prototypes[positives].mean(0)
    transfer = ridge_weights(photo - photo_center, prototypes[positives] - name_center, .1)
    language = F.normalize(prototypes - name_center)
    session = torch.from_numpy(sessions.astype(np.int64, copy=False)).cuda()
    eeg = torch.cat([torch.from_numpy(raw[part].copy()).cuda() for part in np.array_split(rows, 55)])
    require(photo.shape == (len(rows), 1024) and torch.isfinite(eeg).all()
            and torch.isfinite(photo).all() and torch.isfinite(transfer).all(),
            "Invalid EEG, photograph or training-name tensors")
    data = dict(eeg=eeg, session=session, photos=photo, center=photo_center,
                variance=(photo - photo_center).square().mean().clamp_min(1e-6),
                name_transfer=transfer, names=language, name_positive=positives)
    initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
    files.verify()
    return data, members, proof, initial


def run_fit(subject, arms, sessions_root):
    from paper_instance_residual import make_encoder
    from paper_oof_conditions import study_root

    out = study_root() / f"fits/sub-{subject:02d}"
    out.mkdir(parents=True, exist_ok=True)
    files = PinnedFiles()
    data, members, proof, initial = training_data(subject, sessions_root, files)
    pairs_per_object = members.shape[1] // 2
    steps = EPOCHS * pairs_per_object * math.ceil(len(members) / OBJECTS_PER_BATCH)
    entries = {}
    for arm in arms:
        dest = out / f"{arm}.pt"
        receipt = dest.with_suffix(".json")
        if dest.is_file() and receipt.is_file():
            info = json.loads(receipt.read_bytes())
            require(info["inputs"] == proof and info["checkpoint_sha256"] == sha(dest), "Saved fit uses different inputs")
            entries[arm] = info
            continue
        contrast_weight, language_weight = ARMS[arm]
        torch.manual_seed(SEED)
        learner = make_encoder(torch.device("cuda")).float().train()
        learner.load_state_dict(initial)
        rng = np.random.default_rng(SEED + 1)
        optimizer = torch.optim.AdamW(learner.backbone.parameters(), lr=7e-5, betas=(.9, .995), weight_decay=.05)
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=8e-6)
        trace = hashlib.sha256()
        history = []
        totals = torch.zeros(3, device="cuda")
        batches = paired_batches(members, EPOCHS, OBJECTS_PER_BATCH, SEED)
        for step, pairs in enumerate(batches, 1):
            orientation = rng.integers(0, 2, size=(pairs.size, 2), dtype=np.int64)
            trace.update(pairs.astype("<i8").tobytes() + orientation.astype("<i8").tobytes())
            ids = torch.from_numpy(pairs.flatten()).cuda()
            raw = data["eeg"][ids].float()
            indices = split_trials(data["session"][ids], torch.from_numpy(orientation).cuda())
            halves = raw[torch.arange(len(ids), device="cuda")[:, None, None], indices].mean(2)
            trials = torch.cat((raw.mean(1), halves[:, 0], halves[:, 1]))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                full, left, right = learner.backbone(trials).float().chunk(3)
            photo = data["photos"][ids]
            center, variance = data["center"], data["variance"]
            exact = photo_loss(full, photo, center, variance)
            exact += .125 * (photo_loss(left, photo, center, variance) + photo_loss(right, photo, center, variance))
            contrast = matched_margin((left - center).reshape(-1, 2, 1024), (right - center).reshape(-1, 2, 1024))
            projected = F.normalize((full - center) @ data["name_transfer"])
            semantic = F.cross_entropy(projected @ data["names"].T / TEMPERATURE, data["name_positive"][ids])
            (exact + contrast_weight * contrast + language_weight * semantic).backward()
            torch.nn.utils.clip_grad_norm_(learner.backbone.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            decay.step()
            totals += torch.stack([x.detach() for x in (exact, contrast, semantic)])
            if step % 70 == 0 or step == steps:
                previous = history[-1]["step"] if history else 0
                item = dict(step=step, terms=(totals / (step - previous)).tolist())
                history.append(item)
                totals.zero_()
                print(f"participant {subject:02d} {arm}", item, flush=True)
        files.verify()
        state = {key: value.detach().cpu() for key, value in learner.state_dict().items()}
        atomic_file(dest, lambda f: torch.save(state, f))
        info = dict(arm=arm, participant=subject, inputs=proof, checkpoint_sha256=sha(dest),
                    seed=SEED, training_schedule_sha256=trace.hexdigest(), steps=steps,
                    repeat_coefficient=contrast_weight, language_coefficient=language_weight, history=history)
        save_json(receipt, info)
        entries[arm] = info
    require(len({item["training_schedule_sha256"] for item in entries.values()}) == 1,
            "Matched configurations used different batches or physical half-trial selections")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("subject", type=int, choices=range(1, 11))
    cli.add_argument("--sessions-root", type=Path, required=True)
    cli.add_argument("--arms", nargs="+", choices=ARMS, default=tuple(ARMS))
    args = cli.parse_args()
    run_fit(args.subject, args.arms, args.sessions_root)
