"""Fine-tune the D2 IP residual on condition-specific responses of a frozen four-step sampler."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import NamedTuple
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


SEED = 20261125
STEPS = 160
ANCHORS = 32
SETTINGS = dict(
    seed=SEED, steps=STEPS, anchors=ANCHORS, queries_per_step=2,
    anchor_source="32 distinct training concepts, alternating frozen D2 and mean training-F conditions",
    layout="D2 concept-OOF ridge coefficients, PCA basis and mean",
    sampler="original 4-step s375 Euler ancestral, low_keep=(8,.11,.02)",
    eeg="four native trials; .72 inverse-channel-RMS ERP features + .28 mean single-trial features",
    teacher="unmodified training-photo F -> frozen fusion adapter -> frozen IP projection",
    neutral="mean over the permitted training-photo F vectors, then adapter and projection",
    pairing="three cross-concept pairs then one within-concept pair; all anchor concepts excluded",
    distance=".5 cosine distance + .5 MSE / max(teacher mean square,1e-6), per response",
    positive="distance(U_student - U_neutral, U_teacher - U_neutral)",
    pair="distance(U_student_A - U_student_B, U_teacher_A - U_teacher_B)",
    pair_weight=.5, trust_weight=.02, residual_norm_weight=.002,
    regularization="mean squared token movement from D2, and from the frozen projection, divided by projected token power",
    optimizer="AdamW", lr=7.5e-5, minimum_lr=1e-5, betas=[.9, .995],
    weight_decay=.05, max_grad_norm=1., schedule="cosine", rank=128, max_relative_token_norm=.35,
    unet_dtype="float16", residual_dtype="float32", inference_unet_calls=4,
)


class Anchor(NamedTuple):
    row: int
    subject: int
    mode: str
    noise_seed: int


class Pair(NamedTuple):
    anchor: int
    time: int
    subject: int
    left: int
    right: int


def make_plan(labels, subjects):
    """All row indices are local to the ordered list of permitted training rows."""
    rng = np.random.default_rng(SEED)
    groups = {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}
    eligible = np.array([c for c, rows in groups.items() if len(rows) >= 2])
    if len(eligible) < ANCHORS + 2 or not subjects:
        raise ValueError("Need 32 donor concepts, two other multi-photo concepts and at least one subject")
    donors = rng.choice(eligible, ANCHORS, replace=False)
    targets = np.setdiff1d(eligible, donors)
    anchor_subjects = rng.permutation(np.resize(rng.permutation(subjects), ANCHORS))
    anchors = [
        Anchor(int(rng.choice(groups[c])), int(anchor_subjects[i]), "d2" if i % 2 == 0 else "neutral",
               int(rng.integers(0, 2**63 - 1)))
        for i, c in enumerate(donors)
    ]
    pair_subjects = rng.permutation(np.resize(rng.permutation(subjects), STEPS))
    pairs = []
    for group in range(STEPS // 4):
        if group % ANCHORS == 0:
            decks = [rng.permutation(ANCHORS) for _ in range(4)]
        if group % 4 == 0:
            within_times = rng.permutation(4)
        last = int(within_times[group % 4])
        times = [*rng.permutation(np.delete(np.arange(4), last)), last]
        for i, t in enumerate(times):
            if i == 3:
                left, right = rng.choice(groups[int(rng.choice(targets))], 2, replace=False)
            else:
                a, b = rng.choice(targets, 2, replace=False)
                left, right = rng.choice(groups[a]), rng.choice(groups[b])
            pairs.append(Pair(int(decks[t][group % ANCHORS]), int(t), int(pair_subjects[len(pairs)]), int(left), int(right)))
    return anchors, pairs


def response_distance(pred, target):
    pred, target = pred.float().flatten(1), target.float().flatten(1)
    power = target.square().mean(1).clamp_min(1e-6)
    relative_mse = (pred - target).square().mean(1) / power
    cosine = F.cosine_similarity(pred, target, dim=1, eps=1e-6)
    return .5 * (relative_mse + 1 - cosine).mean()


def response_terms(student, teacher, neutral):
    student, teacher, neutral = student.float(), teacher.float(), neutral.float()
    positive = response_distance(student - neutral, teacher - neutral)
    pair = response_distance(student[:1] - student[1:], teacher[:1] - teacher[1:])
    return positive, pair


def denoise(unet, state, time, tokens, prompts):
    """Broadcast one already normalized model input and its original scheduler timestep."""
    pe, pooled, dims = prompts
    n = len(tokens)
    return unet(
        state.expand(n, -1, -1, -1), time.expand(n), encoder_hidden_states=pe.expand(n, -1, -1),
        added_cond_kwargs={"text_embeds": pooled.expand(n, -1), "time_ids": dims.expand(n, -1)},
        cross_attention_kwargs={"ip_hidden_states": tokens},
    ).sample.float()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensors_hash(*tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        raw = tensor.detach().cpu().contiguous().numpy()
        h.update(str((raw.dtype.str, raw.shape)).encode())
        h.update(raw.tobytes())
    return h.hexdigest()


def read_pack(path, key):
    pack = torch.load(path, map_location="cpu", weights_only=True)
    if pack.get("request_sha256") != key or pack.get("complete") is not True:
        raise ValueError(f"Incomplete or incompatible artifact at {path}")
    return pack


def publish(path, pack):
    """Publish a complete file once on the group filesystem, including concurrent writers."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with tmp.open("xb") as stream:
            torch.save(pack, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return read_pack(path, pack["request_sha256"])
        return pack
    finally:
        tmp.unlink(missing_ok=True)


@torch.no_grad()
def eeg_features(model, trials, subject, cone, adapter, proj, dtype):
    from exp1_swapdiff_3arm import per_sample_cone
    from exp6_draft_intra4 import forward_branches_and_trunk
    from exp8_trial_precision_swapdiff import prec_erp_analytic

    sid = subject - 1
    f, h, _ = forward_branches_and_trunk(model, prec_erp_analytic(trials, p=.5), sid, 4.)
    ft, ht, _ = forward_branches_and_trunk(model, trials.flatten(0, 1), sid, 1.)
    f = .72 * f + .28 * ft.reshape(len(trials), 4, -1).mean(1)
    h = .72 * h + .28 * ht.reshape(len(trials), 4, -1).mean(1)
    c = per_sample_cone(f, cone["R"], cone["alpha"], cone["beta"], cone["u_tr"])
    return h, proj(adapter(c.to(dtype)))


@torch.no_grad()
def capture_trajectory(unet, run):
    """Read the exact normalized input and timestep passed by the existing renderer."""
    states, times = [], []

    def capture(module, args):
        state, time = args[:2]
        if state.shape[0] != 1 or time.shape != (1,):
            raise ValueError("A bank trajectory must contain exactly one donor query")
        states.append(state[0].detach().cpu().clone())
        times.append(time[0].detach().cpu().clone())

    hook = unet.register_forward_pre_hook(capture)
    try:
        run()
    finally:
        hook.remove()
    if len(states) != 4 or not all(torch.isfinite(s).all() for s in states):
        raise ValueError("The original sampler did not yield four finite denoiser inputs")
    return torch.stack(states), torch.stack(times)


@torch.no_grad()
def build_bank(anchors, oof, pipe, prompts, features, initial_residual, neutral):
    from exp6_draft_intra4 import rollout_4step_step_ips

    device = neutral.device
    basis = oof["V_pca"].to(device=device, dtype=torch.float32)
    mean = oof["my"].to(device=device, dtype=torch.float32)
    states, times = [], []
    for a in anchors:
        coeff = oof["Z_oof"][a.subject - 1, a.row:a.row + 1].to(device=device, dtype=torch.float32)
        if not torch.isfinite(coeff).all() or not coeff.any():
            raise ValueError(f"Missing OOF layout for subject {a.subject}, training row {a.row}")
        layout = (coeff @ basis.T + mean).reshape(1, 4, 64, 64)
        tokens = initial_residual(*features(a.subject, [a.row])) if a.mode == "d2" else neutral
        noise = torch.randn(5, 4, 64, 64, generator=torch.Generator().manual_seed(a.noise_seed)).to(device)
        init, *steps = noise.unsqueeze(0).unbind(1)
        state, time = capture_trajectory(pipe.unet, lambda: rollout_4step_step_ips(
            pipe.unet, pipe.scheduler, [tokens] * 4, layout, init, steps, *prompts, .375, (8, .11, .02),
        ))
        states.append(state)
        times.append(time)
    return torch.stack(states), torch.stack(times)


def train(subjects, positive_only=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from common import PREP, RUNS, concept_folds
    from exp1_swapdiff_3arm import ResidualIPCondLayer
    from exp6_draft_intra4 import DRAFT_DIR, precompute_or_load_tr14_z0_oof
    from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
    from paper_training import D2, SWAP, generator

    expected = f"personal-{subjects[0]:02d}" if len(subjects) == 1 else "shared"
    if RUNS.name != expected or "EEG_RUNS_ROOT" not in os.environ:
        raise ValueError(f"Set EEG_RUNS_ROOT to the {expected} group for this run")
    mode = "response_positive" if positive_only else "response"
    path = DRAFT_DIR / f"{mode}_s{SEED}.pt"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    plan_file = PREP / "paper_split.json"
    rows = np.asarray(json.loads(plan_file.read_text())["train_rows"], dtype=np.int64)
    concept = np.load(PREP / "train_concept.npy", mmap_mode="r")
    ok = np.load(PREP / "lat512_train_ok.npy", mmap_mode="r")
    expected_rows = np.flatnonzero((concept_folds(1654)[concept] != 0) & ok)
    if not np.array_equal(rows, expected_rows):
        raise ValueError("Permitted train rows disagree with the row ordering of the D2 OOF cache")
    eeg_ids = json.loads((PREP / "eeg_train_ids.json").read_text())["ids"]
    if eeg_ids != json.loads((PREP / "train_ids.json").read_text())["ids"]:
        raise ValueError("EEG and training-photo identities disagree")
    eeg, eeg_hashes = {}, {}
    for subject in subjects:
        file = PREP / f"sub-{subject:02d}_train.npy"
        expected_hash = json.loads(file.with_suffix(".json").read_text())["output_sha256"]
        if file_hash(file) != expected_hash:
            raise ValueError(f"Training EEG checksum mismatch in {file}")
        eeg[subject] = np.load(file, mmap_mode="r")
        if eeg[subject].shape != (len(eeg_ids), 4, 17, 250):
            raise ValueError(f"Unexpected physical repeat axes in {file}")
        eeg_hashes[str(subject)] = expected_hash
    teacher_file = PREP / "F_train.npy"
    train_f = torch.from_numpy(np.load(teacher_file, mmap_mode="r")[rows]).float()
    if train_f.shape != (len(rows), 1024) or not torch.isfinite(train_f).all():
        raise ValueError("Invalid training F targets")
    anchors, pairs = make_plan(concept[rows], subjects)
    donor_record = [dict(row=int(rows[a.row]), subject=a.subject, mode=a.mode, noise_seed=a.noise_seed) for a in anchors]
    schedule_hash = fingerprint([(p.anchor, p.time, p.subject, int(rows[p.left]), int(rows[p.right])) for p in pairs])
    oof = precompute_or_load_tr14_z0_oof(rows, concept, subjects=subjects)
    if oof["Z_oof"].shape != (10, len(rows), 256):
        raise ValueError("OOF layout inventory disagrees with the training split")
    inputs = dict(
        d2=file_hash(D2), swap=file_hash(SWAP), split=file_hash(plan_file), eeg=eeg_hashes,
        train_rows=fingerprint(rows.tolist()), train_F=tensors_hash(train_f),
        oof=file_hash(DRAFT_DIR / "z0_oof_tr14_256.pt"), code=file_hash(Path(__file__)),
    )
    bank_key = fingerprint([inputs, donor_record, SETTINGS["sampler"]])
    settings = dict(SETTINGS, effective_pair_weight=0. if positive_only else SETTINGS["pair_weight"])
    key = fingerprint([bank_key, schedule_hash, settings])
    if path.exists():
        read_pack(path, key)
        print("Verified existing", path, flush=True)
        return path

    d2 = torch.load(D2, map_location="cpu", weights_only=True)
    declared = Path(d2["backbone_ckpt"])
    if declared.name != SWAP.name or (declared.is_file() and not declared.samefile(SWAP)):
        raise ValueError("D2 points to a backbone outside the selected group's SWAP checkpoint")
    swap = torch.load(SWAP, map_location="cpu", weights_only=True)
    model = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=.20, in_drop=.15)
    model.load_state_dict(swap["state_dict"])
    del swap
    device, dtype = torch.device("cuda"), torch.float16
    model.to(device).eval().requires_grad_(False)
    residual = ResidualIPCondLayer().float().to(device)
    residual.load_state_dict(d2["cond_layer"])
    initial = copy.deepcopy(residual).eval().requires_grad_(False)
    cone = dict(d2["cone"], u_tr=d2["cone"]["u_tr"].float().to(device))
    pipe, adapter, proj, prompt, pooled, dims = generator(dtype)
    pipe.vae = None
    prompts = (prompt, pooled, dims)
    if pipe.scheduler.config.prediction_type != "epsilon" or pipe.scheduler.timesteps.tolist() != [999, 749, 499, 249]:
        raise ValueError("Unexpected SDXL-Turbo epsilon schedule")

    def features(subject, local_rows):
        x = torch.from_numpy(eeg[subject][rows[local_rows]]).to(device=device, dtype=torch.float32)
        return eeg_features(model, x, subject, cone, adapter, proj, dtype)

    with torch.no_grad():
        neutral = proj(adapter(train_f.mean(0, keepdim=True).to(device=device, dtype=dtype)))
    bank_path = DRAFT_DIR / f"response_bank_s{SEED}_{bank_key[:16]}.pt"
    if bank_path.exists():
        bank = read_pack(bank_path, bank_key)
    else:
        print("Building 32 independent train-only donor trajectories", flush=True)
        states, times = build_bank(anchors, oof, pipe, prompts, features, initial, neutral)
        bank = publish(bank_path, dict(
            states=states, times=times, donors=donor_record, sha256=tensors_hash(states, times),
            request_sha256=bank_key, complete=True,
        ))
    del oof
    states, times = bank["states"], bank["times"]
    if (states.shape != (ANCHORS, 4, 4, 64, 64) or states.dtype != dtype or times.shape != (ANCHORS, 4)
            or not torch.equal(times, pipe.scheduler.timesteps.cpu().expand_as(times))
            or not torch.isfinite(states).all() or tensors_hash(states, times) != bank["sha256"]):
        raise ValueError("Frozen sampler bank failed verification")

    opt = torch.optim.AdamW(residual.parameters(), lr=settings["lr"], betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=settings["minimum_lr"])
    records = []
    for step, pair in enumerate(pairs, 1):
        local = [pair.left, pair.right]
        state = states[pair.anchor, pair.time].unsqueeze(0).to(device)
        t = times[pair.anchor, pair.time].reshape(1).to(device)
        with torch.no_grad():
            h, projected = features(pair.subject, local)
            fixed = initial(h, projected).float()
            visual = proj(adapter(train_f[local].to(device=device, dtype=dtype)))
            oracle = denoise(pipe.unet, state, t, torch.cat((visual, neutral)), prompts)
        tokens = residual(h, projected)
        student = checkpoint(lambda ip: denoise(pipe.unet, state, t, ip, prompts), tokens,
                             use_reentrant=False, preserve_rng_state=False)
        positive, difference = response_terms(student, oracle[:2], oracle[2:])
        base_power = projected.float().square().mean((1, 2)).clamp_min(1e-6)
        trust = ((tokens.float() - fixed).square().mean((1, 2)) / base_power).mean()
        norm = ((tokens.float() - projected.float()).square().mean((1, 2)) / base_power).mean()
        loss = positive + settings["effective_pair_weight"] * difference + settings["trust_weight"] * trust + settings["residual_norm_weight"] * norm
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training response at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(residual.parameters(), settings["max_grad_norm"], error_if_nonfinite=True)
        if step == 1 and float(grad) == 0:
            raise RuntimeError("No U-Net response gradient reached the D2 residual")
        opt.step()
        scheduler.step()
        if step == 1 or step % 40 == 0:
            record = dict(step=step, loss=float(loss.detach()), positive=float(positive.detach()),
                          pair=float(difference.detach()), trust=float(trust.detach()), norm=float(norm.detach()), grad=float(grad))
            records.append(record)
            print(mode, json.dumps(record, allow_nan=False), flush=True)
    state = {name: tensor.detach().cpu() for name, tensor in residual.state_dict().items()}
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise FloatingPointError("Non-finite final conditional residual")
    publish(path, dict(
        complete=True, request_sha256=key, mode=mode, steps=STEPS, backbone_ckpt=str(SWAP.resolve()),
        cond_layer=state, cone=d2["cone"], subjects=list(subjects), settings=settings, input_sha256=inputs,
        bank_file=bank_path.name, bank_tensor_sha256=bank["sha256"], bank_donors=donor_record,
        training_pair_schedule_sha256=schedule_hash, training_record=records,
    ))
    print("Saved", path, flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, 11), help="personal group; omit for shared")
    parser.add_argument("--positive-only", action="store_true", help="same U-Net evaluations and pairs with zero pair-loss weight")
    args = parser.parse_args()
    train((args.subject,) if args.subject else tuple(range(1, 11)), args.positive_only)
