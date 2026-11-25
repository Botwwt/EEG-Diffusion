"""Training-only concept/photograph error calibration for a participant's native Brain-HiVE encoder."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import torch
import torch.nn.functional as F


SEED = 20261125
ARMS = ("instance-native", "instance-direct", "instance-residual")
GAINS = (0., .25, .5, .75, 1.)
RANK = 48
REMOVE_BETWEEN = 16
LAMBDA_OBJECT, LAMBDA_PHOTO = .5, 2.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def inner_partition(labels):
    """Fit, calibration and audit are disjoint concepts. Every row has a shadow that never saw its concept."""
    rng = np.random.default_rng(SEED)
    classes = rng.permutation(np.unique(labels))
    require(len(classes) >= 12, "Insufficient training concepts for three shadow encoders")
    hold = max(1, len(classes) // 10)
    parts = np.full(len(labels), "fit", dtype="U12")
    fold = np.full(len(labels), -1, dtype=np.int64)
    for tag, subset in (("calibration", classes[:hold]), ("audit", classes[hold:2 * hold]), ("fit", classes[2 * hold:])):
        for k in range(3):
            take = np.isin(labels, subset[k::3])
            parts[take], fold[take] = tag, k
    require((fold >= 0).all(), "Unassigned training row")
    return fold, parts


def photo_view(mean_eeg):
    """Posterior spatial contrasts in 30 nonoverlapping, 20-ms bins from the first 600 ms."""
    binned = mean_eeg[..., :150].reshape(-1, 17, 30, 5).mean(-1)
    return (binned - binned.mean(1, keepdim=True)).flatten(1)


def class_mean(x, labels):
    _, inv, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    means = x.new_zeros(len(counts), x.shape[1])
    means.index_add_(0, inv, x)
    return means / counts[:, None], inv


def reliability(a, b):
    # Each half contains two physically distinct trials. The variance of their mean's noise is E[(a-b)^2]/4.
    observed = ((a + b) * .5).var(0, unbiased=False).clamp_min(1e-8)
    noise = (a - b).square().mean(0) * .25
    return (1 - noise / observed).clamp(0, 1)


def axes(x, count):
    require(0 < count <= min(x.shape), "Insufficient rank for the registered feature map")
    _, vectors = torch.linalg.eigh(x.T @ x / len(x))
    return vectors[:, -count:]


def feature_map(raw, a, b, labels, rank, remove_between):
    raw, a, b = raw.double(), a.double(), b.double()
    rel = reliability(a, b)
    mean = raw.mean(0)
    scale = rel.clamp_min(.02).sqrt() / raw.std(0, unbiased=False).clamp_min(1e-6)
    x = (raw - mean) * scale
    means, inv = class_mean(x, labels)
    between = means - means.mean(0)
    if remove_between:
        nuisance = axes(between, remove_between)
        within = x - means[inv]
        restricted = within - (within @ nuisance) @ nuisance.T
        rotation = axes(restricted, rank)
        rotation = rotation - nuisance @ (nuisance.T @ rotation)
    else:
        rotation = axes(between, rank)
    projection = scale[:, None] * rotation
    scores = (raw - mean) @ projection
    projection /= scores.std(0, unbiased=False).clamp_min(1e-6)
    return {"mean": mean.float(), "projection": projection.float(), "reliability": rel.float()}


def project(pack, x):
    return (x - pack["mean"]) @ pack["projection"]


def ridge(x, y, penalty):
    x, y = x.double(), y.double()
    gram = x.T @ x / len(x)
    gram.diagonal().add_(torch.as_tensor(penalty, device=x.device, dtype=x.dtype))
    return torch.linalg.solve(gram, x.T @ y / len(x)).float()


def fit_heads(base, half_a, half_b, photo_a, photo_b, target, labels, rank=RANK, remove_between=REMOVE_BETWEEN):
    """Both heads see the same two label-free feature maps and have (2*rank+1)*1024 coefficients."""
    photo = (photo_a + photo_b) * .5
    sm = feature_map(base, half_a, half_b, labels, rank, 0)
    pm = feature_map(photo, photo_a, photo_b, labels, rank, remove_between)
    s, p, error = project(sm, base), project(pm, photo), target - base
    cs, inv = class_mean(s, labels)
    cp, _ = class_mean(p, labels)
    ce, _ = class_mean(error, labels)
    obj = ridge(cs - cs.mean(0), ce - ce.mean(0), LAMBDA_OBJECT)
    pic = ridge(p - cp[inv], error - ce[inv], LAMBDA_PHOTO)
    residual = dict(object=obj, photo=pic, bias=ce.mean(0) - cs.mean(0) @ obj)

    joint = torch.cat((s, p), dim=1)
    penalties = [LAMBDA_OBJECT] * rank + [LAMBDA_PHOTO] * rank
    coef = ridge(joint - joint.mean(0), error - error.mean(0), penalties)
    direct = dict(object=coef[:rank], photo=coef[rank:], bias=error.mean(0) - joint.mean(0) @ coef)
    return dict(semantic_map=sm, photo_map=pm, residual=residual, direct=direct)


def corrections(heads, mode, base, photo):
    bank = heads[mode]
    return (project(heads["semantic_map"], base) @ bank["object"] + bank["bias"],
            project(heads["photo_map"], photo) @ bank["photo"])


def gallery(real, labels):
    means, inv = class_mean(real, labels)
    sizes = torch.bincount(inv)
    require(int(sizes.min()) >= 2, "Within-concept calibration needs at least two photos per concept")
    candidates = torch.zeros(len(real), int(sizes.max()), dtype=torch.long, device=real.device)
    mask = torch.zeros_like(candidates, dtype=torch.bool)
    for i in range(len(sizes)):
        rows = torch.nonzero(inv == i).flatten()
        candidates[rows, :len(rows)] = rows[None, :]
        mask[rows, :len(rows)] = True
    return F.normalize(means), inv, sizes, F.normalize(real)[candidates], mask, F.normalize(real)


def measure(pred, gallery_data):
    prototypes, inv, sizes, siblings, mask, truth = gallery_data
    unit = F.normalize(pred)
    object_sim = unit @ prototypes.T
    row = torch.arange(len(inv), device=pred.device)
    true_object = object_sim[row, inv, None]
    other = torch.arange(len(prototypes), device=pred.device)[None, :] != inv[:, None]
    object_wins = (((object_sim < true_object).float() + .5 * (object_sim == true_object)) * other).sum(1)
    # Use the identical dot-product kernel for each candidate, including the matching photo.
    within_sim = torch.einsum("nd,nkd->nk", unit, siblings)
    local_index = torch.empty_like(inv)
    for i in range(len(sizes)):
        these = torch.nonzero(inv == i).flatten()
        local_index[these] = torch.arange(len(these), device=pred.device)
    true_photo = within_sim[row, local_index, None]
    within_wins = (((within_sim < true_photo).float() + .5 * (within_sim == true_photo)) * mask).sum(1) - .5
    w = 1 / (len(sizes) * sizes[inv].float())
    return dict(object_pair=float((object_wins * w).sum() / (len(sizes) - 1)),
                object_top1=float(((object_sim.argmax(1) == inv) * w).sum()),
                within_pair=float((within_wins * w / (sizes[inv] - 1)).sum()),
                target_cosine=float(((unit * truth).sum(1) * w).sum()))


def choose_gains(base, obj, photo, targets):
    first = [(g, measure(base + g * obj, targets)) for g in GAINS]
    go, ref = max(first, key=lambda r: (r[1]["object_pair"], r[1]["object_top1"], -r[0]))
    second = [(g, measure(base + go * obj + g * photo, targets)) for g in GAINS]
    eligible = [(g, m) for g, m in second if m["object_pair"] >= ref["object_pair"] - .002
                and m["object_top1"] >= ref["object_top1"] - .005]
    gp, chosen = max(eligible, key=lambda r: (r[1]["within_pair"], r[1]["object_pair"], -r[0]))
    return (go, gp), dict(object_scan=first, photo_scan=second, selected=chosen)


def locations(subject):
    from common import RUNS
    name = f"personal-{subject:02d}"
    personal = RUNS.parent / name if RUNS.name == "shared" else RUNS
    require(personal.name == name, f"EEG_RUNS_ROOT must select {name} or shared")
    native = personal / f"enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
    fitted = personal / f"enc/instance_residual/sub-{subject:02d}/fit.pt"
    return native, fitted


def make_encoder(device):
    from common import HIVE
    if str(HIVE) not in sys.path:
        sys.path.insert(0, str(HIVE))
    from main.models_brain import BrainEncoder
    return BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=250,
                        hidden_size=1024, embedding_dim=1440, dropout=.1).to(device)


def train_shadow(means, embeddings, indices, teacher, seed, device):
    torch.manual_seed(seed)
    model = make_encoder(device).train()
    temperature = torch.nn.Parameter(torch.tensor(2.6592, device=device))
    decay, clean = [temperature], []
    for name, par in model.named_parameters():
        (clean if name.endswith("bias") or ".norm." in name else decay).append(par)
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=.05), dict(params=clean, weight_decay=0.)],
                            lr=5e-4, betas=(.9, .995), eps=1e-8)
    total = 25 * math.ceil(len(indices) / 1024)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    rng = torch.Generator(device=device).manual_seed(seed)
    for _ in range(25):
        order = indices[torch.randperm(len(indices), generator=rng, device=device)]
        for batch in order.split(1024):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    image = F.normalize(teacher(**{k: v[batch] for k, v in embeddings.items()}), dim=-1)
                brain = F.normalize(model.backbone(means[batch]), dim=-1)
                sim = (brain @ image.T) * temperature.exp()
                target = torch.arange(len(batch), device=device)
                loss = .5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*model.parameters(), temperature], 1., error_if_nonfinite=True)
            opt.step()
            sched.step()
    return model.eval().requires_grad_(False)


def shadow_features(eeg, rows, fold, parts, means, embeddings, teacher, device):
    semantic = torch.empty(len(rows), 3, 1024)
    photos = torch.empty(len(rows), 2, 510)
    checks = []
    for k in range(3):
        train = np.flatnonzero((parts == "fit") & (fold != k))
        pred = np.flatnonzero(fold == k)
        require(not np.intersect1d(train, pred).size, "Shadow training and prediction rows overlap")
        model = train_shadow(means, embeddings, torch.as_tensor(train, device=device), teacher, 2025 + k, device)
        with torch.inference_mode():
            for pos in range(0, len(pred), 128):
                ix = pred[pos:pos + 128]
                x = torch.from_numpy(eeg[rows[ix]]).to(device=device, dtype=torch.float32)
                a, b = (x[:, 0] + x[:, 2]) * .5, (x[:, 1] + x[:, 3]) * .5
                m, left, right = model.backbone(torch.cat((x.mean(1), a, b))).chunk(3)
                semantic[ix] = torch.stack((m, left, right), 1).cpu()
                photos[ix] = torch.stack((photo_view(a), photo_view(b)), 1).cpu()
        checks.append(dict(fold=k, train_rows=len(train), predict_rows=len(pred),
                           train_rows_sha256=digest_array(rows[train]), predicted_rows_sha256=digest_array(rows[pred])))
        print("Completed concept-disjoint native shadow", k + 1, "/ 3", flush=True)
        del model
    return semantic, photos, checks


def train(subject, device):
    from common import EMB_KEYS, HIVE, PREP, PRIOR, concept_folds
    if str(HIVE) not in sys.path:
        sys.path.insert(0, str(HIVE))
    from main.models_adapter import FusionEncoderModel
    native, output = locations(subject)
    split_file = PREP / "paper_split.json"
    rows = np.asarray(json.loads(split_file.read_text())["train_rows"], dtype=np.int64)
    all_labels = np.load(PREP / "train_concept.npy", mmap_mode="r")
    ok = np.load(PREP / "lat512_train_ok.npy", mmap_mode="r")
    require(np.array_equal(rows, np.flatnonzero((concept_folds(1654)[all_labels] != 0) & ok)), "Incorrect outer training fold")
    train_ids = json.loads((PREP / "train_ids.json").read_text())["ids"]
    require(train_ids == json.loads((PREP / "eeg_train_ids.json").read_text())["ids"], "Training photograph and EEG order differ")
    raw_file = PREP / f"sub-{subject:02d}_train.npy"
    eeg_sha = json.loads(raw_file.with_suffix(".json").read_text())["output_sha256"]
    inputs = dict(native=sha(native), eeg=eeg_sha, train_rows=digest_array(rows), split=sha(split_file), code=sha(Path(__file__)),
                  train_F=sha(PREP / "F_train.npy"), embeddings=sha(PREP / "emb_train.pt"),
                  teacher=sha(PRIOR / "fusion_encoder/diffusion_pytorch_model.safetensors"))
    if output.is_file():
        saved = torch.load(output, weights_only=True, map_location="cpu")
        require(saved["complete"] and saved["input_sha256"] == inputs, "Existing correction pack uses different inputs")
        print("Verified existing", output, flush=True)
        return output
    require(sha(raw_file) == eeg_sha, "Training EEG differs from its source receipt")
    eeg = np.load(raw_file, mmap_mode="r")
    require(eeg.shape == (len(train_ids), 4, 17, 250), "Unexpected training EEG axes")
    labels = all_labels[rows]
    fold, parts = inner_partition(labels)
    device = torch.device(device)
    means = torch.empty(len(rows), 17, 250, device=device)
    for start in range(0, len(rows), 128):
        sl = slice(start, min(start + 128, len(rows)))
        means[sl] = torch.from_numpy(eeg[rows[sl]]).to(device=device, dtype=torch.float32).mean(1)
    emb = torch.load(PREP / "emb_train.pt", weights_only=True, map_location="cpu")
    require(set(emb) == set(EMB_KEYS), "Expected native H14, B32, VAE training targets")
    emb = {key: v[rows].float().to(device) for key, v in emb.items()}
    teacher = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.to(device)
    teacher.train().requires_grad_(False)
    for layer in teacher.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = .1
    semantic, photo, shadow_checks = shadow_features(eeg, rows, fold, parts, means, emb, teacher, device)
    del means, emb, teacher
    real = torch.from_numpy(np.load(PREP / "F_train.npy", mmap_mode="r")[rows]).float()
    require(real.shape == (len(rows), 1024) and torch.isfinite(real).all(), "Invalid native training F targets")
    cids = torch.as_tensor(labels.astype(np.int64))
    fit, cal, audit = (torch.from_numpy(parts == s) for s in ("fit", "calibration", "audit"))
    heads = fit_heads(*semantic[fit].unbind(1), *photo[fit].unbind(1), real[fit], cids[fit])
    cal_gallery, audit_gallery = gallery(real[cal], cids[cal]), gallery(real[audit], cids[audit])
    base = semantic[:, 0]
    gains, calibration = {}, {}
    audit_metrics = {"native": measure(base[audit], audit_gallery)}
    for mode in ("residual", "direct"):
        obj, pic = corrections(heads, mode, base, photo.mean(1))
        gains[mode], calibration[mode] = choose_gains(base[cal], obj[cal], pic[cal], cal_gallery)
        go, gp = gains[mode]
        audit_metrics[mode] = measure(base[audit] + go * obj[audit] + gp * pic[audit], audit_gallery)
    counts = {s: int(len(np.unique(labels[parts == s]))) for s in ("fit", "calibration", "audit")}
    reliabilities = {name: [float(v) for v in torch.quantile(heads[name]["reliability"], torch.tensor([.1,.5,.9]))]
                     for name in ("semantic_map", "photo_map")}
    info = dict(subject=subject, complete=True, input_sha256=inputs, concept_counts=counts, shadow_folds=shadow_checks,
                partition_sha256={s: digest_array(np.unique(labels[parts == s])) for s in counts},
                gains=gains, calibration=calibration, audit=audit_metrics, reliability_quantiles_10_50_90=reliabilities,
                rank_per_head=RANK, discarded_photo_between_concept_directions=REMOVE_BETWEEN,
                output_coefficients_per_arm=(2*RANK+1)*1024, ridge_penalties=[LAMBDA_OBJECT,LAMBDA_PHOTO],
                protocol="Seed 20261125 concept shuffle; separate 10% calibration and 10% audit, three concept-disjoint native shadows on the fit concepts. Shadows train 25 epochs with R4, stochastic frozen fusion vision teacher, and symmetric InfoNCE. Every head-fit, calibration and audit prediction comes from a shadow that never trained on its concept.",
                reliability="Disjoint repeats (0,2) vs (1,3); fit-only between-photo variance minus estimated R4 noise. Object uses reliable between-concept native F axes; photo uses within-concept posterior spatial-contrast EEG axes after removing the 16 dominant between-concept directions.",
                gain_rule="Object-first macro pairwise retrieval; then best within-class pairwise with object-pair loss <= .002 and top1 loss <= .005 on calibration concepts. Smallest gain wins remaining ties.",
                inference="Final locked personal fold-0 native model on the mean of the requested physical EEG repeats, with two fixed affine maps. Output is raw 1024-dimensional F.")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".fit-{uuid.uuid4().hex}.pt")
    try:
        torch.save(dict(info, heads=heads), tmp)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    receipt = output.with_suffix(".json")
    tmp = receipt.with_name(f".fit-{uuid.uuid4().hex}.json")
    try:
        tmp.write_text(json.dumps(dict(info, checkpoint_sha256=sha(output)), indent=2, allow_nan=False) + "\n")
        tmp.replace(receipt)
    finally:
        tmp.unlink(missing_ok=True)
    print(output, json.dumps(dict(gains=gains, inner_audit=audit_metrics)), flush=True)
    return output


@torch.inference_mode()
def apply_condition(arm, subject, x, group):
    """Return an EEG-only CPU float32 F matrix and the SHA-256 of its deployable checkpoint."""
    require(arm in ARMS, f"Unknown instance arm {arm}")
    native, path = locations(subject)
    native_sha = sha(native)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = make_encoder(device).eval().requires_grad_(False)
    enc.load_state_dict(torch.load(native, weights_only=True, map_location="cpu"))
    mode = arm.split("-", 1)[1]
    corrected = mode != "native"
    pack = torch.load(path, weights_only=True, map_location="cpu") if corrected else None
    if corrected:
        require(pack["complete"] and pack["subject"] == subject and pack["input_sha256"]["native"] == native_sha,
                "Correction pack and native participant checkpoint disagree")
    indices = np.asarray(group, dtype=np.int64).ravel()
    require(x.ndim == 4 and tuple(x.shape[2:]) == (17,250) and len(indices) > 0 and len(set(indices.tolist())) == len(indices)
            and int(indices.min()) >= 0 and int(indices.max()) < x.shape[1], "Invalid EEG or repeat selection")
    result = []
    for i in range(0,len(x),20):
        batch = x[i:i+20]
        raw = (batch if isinstance(batch,torch.Tensor) else torch.from_numpy(np.array(batch,copy=True))).to(device=device,dtype=torch.float32)
        mean = raw[:, indices.tolist()].mean(1)
        base = enc.backbone(mean).float().cpu()
        if corrected:
            obj, pic = corrections(pack["heads"], mode, base, photo_view(mean).cpu())
            go, gp = pack["gains"][mode]
            base = base + go*obj + gp*pic
        result.append(base)
    features = torch.cat(result) if result else torch.empty(0,1024)
    require(torch.isfinite(features).all(), "Non-finite native condition")
    return features, sha(path) if corrected else native_sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, required=True, choices=range(1,11))
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train(args.subject, args.device)
