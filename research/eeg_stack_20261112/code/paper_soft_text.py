"""Development spike: fit continuous SDXL text states from one person's EEG.

python paper_soft_text.py --build-space --device cuda
python paper_soft_text.py --subject 1 --device cuda
python paper_soft_text.py --self-test

The two development participants share only the training-name text space.
Their deployable packs contain affine maps and fixed text states, without a word bank.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid

import numpy as np
import torch
import torch.nn.functional as F


ARMS = ("soft-empty", "soft-mean", "soft-eeg", "soft-wrong")
SUBJECTS = (1, 3)
RANK, SEED = 48, 20261125
TEMPLATE = "a photograph of the {}"
STREAM_WEIGHTS = (.25, .25, .50)  # CLIP-L sequence, bigG sequence, projected bigG pool
RIDGES = (.1, 1., 10.)
MAX_GAIN, MAX_CAL_RMS_RATIO = 2., .75


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def clean_name(folder):
    match = re.fullmatch(r"\d+_(.+)", folder)
    require(match is not None, f"Invalid THINGS concept folder {folder!r}")
    name = re.sub(r"(?<=[a-z])\d+$", "", match.group(1).lower()).replace("_", " ").strip()
    require(bool(name), "Empty training concept name")
    return name


def partitions(names):
    # Identical text after normalizing THINGS homonym suffixes stays in the same partition.
    distinct = np.unique(names)
    require(len(distinct) >= 15, "Too few distinct training names for disjoint fit, calibration and audit")
    order = np.random.default_rng(SEED).permutation(distinct)
    hold = max(2, len(order) // 10)
    cal, audit = np.isin(names, order[:hold]), np.isin(names, order[hold:2*hold])
    return tuple(torch.from_numpy(x) for x in (~(cal | audit), cal, audit))


def registry():
    from common import PREP, concept_folds
    plan = PREP / "paper_split.json"
    rows = np.asarray(json.loads(plan.read_bytes())["train_rows"], dtype=np.int64)
    all_cids = np.load(PREP / "train_concept.npy", allow_pickle=False)
    ok = np.load(PREP / "lat512_train_ok.npy", allow_pickle=False)
    folds = concept_folds(1654)
    require(np.array_equal(rows, np.flatnonzero((folds[all_cids] != 0) & ok)), "Outer training rows changed")
    concepts, inverse = np.unique(all_cids[rows], return_inverse=True)
    require(np.array_equal(concepts, np.flatnonzero(folds != 0)) and len(concepts) == 1323, "Expected 1323 training concepts")
    photos = json.loads((PREP / "train_ids.json").read_bytes())
    eeg_ids = json.loads((PREP / "eeg_train_ids.json").read_bytes())["ids"]
    names = [None] * len(concepts)
    stems = []
    for i, j in zip(rows, inverse):
        p = PurePosixPath(photos["paths"][i].replace("\\", "/"))
        require(p.stem == photos["ids"][i] == eeg_ids[i], "Training photograph and EEG row IDs differ")
        name = clean_name(p.parent.name)
        require(names[j] in (None, name), "One concept ID maps to different text")
        stems.append(p.stem)
        names[j] = name
    provenance = dict(split=sha(plan), code=sha(__file__), train_ids=ids_sha(stems),
                      training_concepts=ids_sha(concepts.tolist()), training_names=ids_sha(names))
    return rows, torch.from_numpy(inverse).long(), names, provenance


def locations(subject=None):
    from common import RUNS
    require(RUNS.name == "shared" or re.fullmatch(r"personal-\d{2}", RUNS.name), "Expected shared or personal EEG_RUNS_ROOT")
    shared = RUNS.parent / "shared/enc/soft_text/fold14_space.pt"
    if subject is None:
        return shared
    require(subject in SUBJECTS and RUNS.name in ("shared", f"personal-{subject:02d}"), "Incorrect development participant or run root")
    return RUNS.parent / f"personal-{subject:02d}/enc/soft_text/sub-{subject:02d}/fit.pt"


def wrong_donors(n):
    require(n >= 2, "The wrong-EEG control needs at least two queries")
    return np.roll(np.arange(n, dtype=np.int64), n // 2)


def eeg_features(x, group, rows=None):
    """Mean physical repeats, then 40 fixed 20-ms bins in the first 800 ms at 250 Hz."""
    idx = np.asarray(group).ravel()
    require(x.ndim == 4 and tuple(x.shape[2:]) == (17, 250) and idx.dtype.kind in "iu" and len(idx) > 0
            and len(np.unique(idx)) == len(idx) and 0 <= idx.min() <= idx.max() < x.shape[1], "Invalid EEG axes or repeat group")
    n = len(x) if rows is None else len(rows)
    chunks = []
    for i in range(0, n, 128):
        b = x[i:i+128] if rows is None else x[rows[i:i+128]]
        b = b if torch.is_tensor(b) else torch.from_numpy(np.array(b, copy=True))
        erp = b[:, idx.tolist(), :, :200].float().mean(1)
        chunks.append(erp.reshape(-1, 17, 40, 5).mean(-1).flatten(1).cpu())
    result = torch.cat(chunks) if chunks else torch.empty(0, 680)
    require(torch.isfinite(result).all(), "Non-finite EEG")
    return result


@torch.inference_mode()
def encode_names(names, model_root, device):
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    prompts = [TEMPLATE.format(n) for n in names]
    blocks, empty, sources = [], [], {}
    for suffix, model_class in (("", CLIPTextModel), ("_2", CLIPTextModelWithProjection)):
        td, ed = (model_root / (s + suffix) for s in ("tokenizer", "text_encoder"))
        tokenizer = CLIPTokenizer.from_pretrained(str(td), local_files_only=True)
        require(tokenizer.model_max_length == 77, "Expected 77-position SDXL tokenizer")
        variant = "fp16" if any(p.is_file() and "fp16" in p.name for p in ed.iterdir()) else None
        files = sorted(p for p in td.iterdir() if p.is_file())
        files += sorted(p for p in ed.iterdir() if p.is_file() and
                        (p.name == "config.json" or (p.suffix in (".safetensors", ".bin", ".json")
                         and (("fp16" in p.name) == (variant is not None)))))
        sources.update({p.relative_to(model_root).as_posix(): sha(p) for p in files})
        model = model_class.from_pretrained(str(ed), local_files_only=True, variant=variant, torch_dtype=dtype).to(device).eval()
        def encode(batch):
            ids = tokenizer(batch, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            # Diffusers SDXL.encode_prompt passes input_ids only, including CLIP's padding/EOS positions.
            out = model(ids.to(device), output_hidden_states=True)
            seq = out.hidden_states[-2].to(device="cpu", dtype=torch.float32)
            return [seq, out.text_embeds.float().cpu()] if suffix else [seq]
        empty.extend(v[0] for v in encode([""]))
        pieces = [encode(prompts[i:i+16]) for i in range(0, len(prompts), 16)]
        blocks.extend(torch.cat(v) for v in zip(*pieces))
        del model, pieces
    require([tuple(x.shape[1:]) for x in blocks] == [(77,768), (77,1280), (1280,)], "Wrong staged SDXL text encoder dimensions")
    return tuple(blocks), tuple(empty), sources


def text_space(blocks, fit_mask, rank, device, decode):
    """Dual-Gram PCA. The metric gives 1/4 of the centered energy to each sequence and 1/2 to the pool."""
    fit = torch.nonzero(fit_mask).flatten().to(device)
    n = len(fit_mask)
    require(0 < rank < len(fit) and all(len(b) == n for b in blocks), "Invalid text PCA inputs")
    all_fit = torch.zeros(n, len(fit), device=device)
    means = []
    for b, weight in zip(blocks, STREAM_WEIGHTS):
        flat = b.flatten(1)
        gram = torch.zeros_like(all_fit)
        mean = []
        for chunk in flat.split(4096, dim=1):
            data = chunk.to(device=device, dtype=torch.float32)
            center = data[fit].mean(0)
            delta = data - center
            gram.addmm_(delta, delta[fit].T)
            mean.append(center.cpu())
        energy = gram[fit].diagonal().mean().item()
        require(np.isfinite(energy) and energy > 0, "Degenerate SDXL text stream")
        all_fit.add_(gram, alpha=weight / energy)
        means.append(torch.cat(mean).reshape(b.shape[1:]))
    eigen, vectors = torch.linalg.eigh(all_fit[fit].cpu().double())
    lam = eigen[-rank:].flip(0)
    require(lam[-1] > 1e-9, "Rank exceeds the positive text spectrum")
    dual = (vectors[:, -rank:].flip(1) / lam.sqrt()).float().to(device)
    coeff = (all_fit @ dual).cpu()
    fraction = float(lam.sum() / eigen.clamp_min(0).sum())
    if not decode:
        return None, coeff, fraction
    basis = []
    for b, mu in zip(blocks, means):
        maps = []
        for chunk, m in zip(b.flatten(1).split(4096, 1), mu.flatten().split(4096)):
            delta = chunk.to(device=device, dtype=torch.float32)[fit] - m.to(device)
            maps.append((dual.T @ delta).cpu())
        basis.append(torch.cat(maps, dim=1).reshape(rank, *b.shape[1:]))
    return dict(mean=tuple(means), basis=tuple(basis)), coeff, fraction


def balance(labels, dtype):
    _, inv, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inv].to(dtype))


def fit_decoder(x, y, labels, penalty):
    """Class-balanced ridge. Fold feature standardization and intercept into two deployment tensors."""
    x, y = x.double(), y.double()
    w = balance(labels, x.dtype)[:, None]
    center, target = (w*x).sum(0), (w*y).sum(0)
    centered = x - center
    scale = (w*centered.square()).sum(0).sqrt().clamp_min(1e-6)
    z = centered / scale
    cov = (z*w).T @ z
    cov.diagonal().add_(penalty)
    weight = torch.linalg.solve(cov, (z*w).T @ (y-target)) / scale[:, None]
    return dict(weight=weight.float().cpu(), bias=(target - center @ weight).float().cpu())


def predict_codes(model, x):
    return x @ model["weight"] + model["bias"]


def assess(pred, target_codes, cid):
    keys, inv, counts = torch.unique(cid, sorted=True, return_inverse=True, return_counts=True)
    require(len(keys) > 1 and counts.min() >= 2, "Calibration requires multiple concepts and photographs")
    target = target_codes[cid]
    sim = F.normalize(pred) @ F.normalize(target_codes[keys]).T
    row = torch.arange(len(cid))
    correct = sim[row, inv, None]
    other = torch.arange(len(keys))[None, :] != inv[:, None]
    w = balance(cid, pred.dtype)
    wins = ((sim < correct).float() + .5*(sim == correct)) * other
    # Each training photograph has the same name target as its siblings. Quantify variation of actual EEG predictions.
    means = torch.zeros(len(keys), pred.shape[1]).index_add_(0, inv, pred) / counts[:, None]
    within = ((pred-means[inv]).square().sum(1) * w).sum()
    between = (means-means.mean(0)).square().sum(1).mean()
    return dict(object_pair=float((wins.sum(1)*w).sum()/(len(keys)-1)),
                object_top1=float(((sim.argmax(1)==inv)*w).sum()),
                mse=float(((pred-target).square().sum(1)*w).sum()),
                within_photo_variance=float(within), across_object_variance=float(between))


def fit_person(x, cid, space, device):
    fit, cal, audit = (m[cid] for m in space["masks"])
    inner = space["inner_codes"]
    xf, yf, groups = x[fit].to(device), inner[cid[fit]].to(device), cid[fit].to(device)
    candidates = []
    for reg in RIDGES:
        head = fit_decoder(xf, yf, groups, reg)
        prediction = predict_codes(head, x[cal])
        candidates.append((reg, assess(prediction, inner, cid[cal]), head))
    reg, _, best = max(candidates, key=lambda a: (a[1]["object_pair"], -a[1]["mse"], -a[0]))
    p, t = predict_codes(best, x[cal]), inner[cid[cal]]
    w = balance(cid[cal], p.dtype)
    power = float((p.square().sum(1)*w).sum())
    true_power = float((t.square().sum(1)*w).sum())
    dot = float(((p*t).sum(1)*w).sum())
    gain = max(0., min(MAX_GAIN, dot/max(power, 1e-12), MAX_CAL_RMS_RATIO*(true_power/max(power, 1e-12))**.5))
    a = predict_codes(best, x[audit])
    final = fit_decoder(x.to(device), space["final_codes"][cid].to(device), cid.to(device), reg)
    report = dict(ridge=reg, gain=gain, ridge_scan=[(r,m) for r,m,_ in candidates],
                  calibration_pulse_rms_ratio=gain*(power/max(true_power, 1e-12))**.5,
                  audit_mean=assess(torch.zeros_like(a), inner, cid[audit]),
                  audit_eeg=assess(gain*a, inner, cid[audit]))
    return final, gain, report


def deploy_states(basis, empty):
    m1, m2, mp = basis["mean"]
    b1, b2, bp = basis["basis"]
    e1, e2, ep = empty
    return dict(mean_hidden=torch.cat((m1,m2), -1), mean_pooled=mp,
                hidden_basis=torch.cat((b1,b2), -1), pooled_basis=bp,
                empty_hidden=torch.cat((e1,e2), -1), empty_pooled=ep)


@contextmanager
def exclusive(path):
    """The two independently submitted Linux jobs can safely populate the common text pack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        import fcntl
        with path.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield
    else:
        # Windows is used for the single-process CPU fixture.
        yield


def atomic_write(path, write):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        write(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_receipt(path, info):
    text = json.dumps(dict(info, checkpoint_sha256=sha(path)), indent=2, allow_nan=False) + "\n"
    atomic_write(path.with_suffix(".json"), lambda tmp: tmp.write_text(text, encoding="utf-8"))


def shared_space(names, provenance, device):
    from common import TURBO
    path = locations()
    with exclusive(path):
        if path.is_file():
            pack = torch.load(path, map_location="cpu", weights_only=True)
            require(pack["meta"]["inputs"] == provenance, "Existing text space uses different training inputs")
        else:
            blocks, empty, sources = encode_names(names, TURBO, device)
            masks = partitions(names)
            _, inner, inner_fraction = text_space(blocks, masks[0], RANK, device, False)
            fitted, final, fraction = text_space(blocks, torch.ones_like(masks[0]), RANK, device, True)
            meta = dict(inputs=provenance, text_sources_sha256=sources, rank=RANK, template=TEMPLATE,
                        stream_weights=STREAM_WEIGHTS, retained_variance=dict(inner=inner_fraction, final=fraction),
                        outer_training_concepts=len(names), distinct_training_prompts=len(set(names)),
                        inner_concepts=dict(zip(("fit","calibration","audit"), [int(m.sum()) for m in masks])),
                        inner_name_hashes=[ids_sha([s for s,take in zip(names,m.tolist()) if take]) for m in masks],
                        supervision="Object names from training folds 1-4 are additional language supervision.",
                        training="Three blocks: penultimate CLIP-L and bigG states at all 77 positions and the projected bigG pooled state. Center per block and give the two sequences 25% each and pooled 50% of aggregate squared variation before rank-48 PCA. Inner PCA fits only inner-fit names. Final PCA fits all 1323 training concepts.",
                        version=dict(torch=str(torch.__version__)))
            pack = dict(meta=meta, masks=masks, inner_codes=inner, final_codes=final,
                        states=deploy_states(fitted, empty))
            atomic_write(path, lambda tmp: torch.save(pack, tmp))
        # An interruption after the checkpoint rename can be recovered by repeating the CLI.
        if not path.with_suffix(".json").is_file():
            save_receipt(path, pack["meta"])
    return pack, sha(path)


def train(subject, device):
    from common import PREP
    rows, cid, names, source = registry()
    space, space_sha = shared_space(names, source, device)
    if subject is None:
        print(locations(), space_sha, flush=True)
        return
    raw = PREP / f"sub-{subject:02d}_train.npy"
    expected = json.loads(raw.with_suffix(".json").read_bytes())["output_sha256"]
    inputs = dict(code=source["code"], text_space=space_sha, training_eeg=expected, training_row_ids=source["train_ids"])
    path = locations(subject)
    with exclusive(path):
        if path.is_file():
            pack = torch.load(path, map_location="cpu", weights_only=True)
            require(pack["meta"]["inputs"] == inputs and pack["subject"] == subject, "Existing participant fit uses different inputs")
        else:
            require(sha(raw) == expected, "Training EEG checksum disagrees with preparation receipt")
            eeg = np.load(raw, mmap_mode="r", allow_pickle=False)
            require(eeg.shape == (16540,4,17,250), "Incorrect physical training EEG shape")
            x = eeg_features(eeg, [[0,1,2,3]], rows)
            head, gain, report = fit_person(x, cid, space, device)
            meta = dict(inputs=inputs, subject=subject, supervision=space["meta"]["supervision"],
                        training_photographs=len(rows), inner_concepts=space["meta"]["inner_concepts"],
                        template=TEMPLATE, rank=RANK, training=report, split_seed=SEED,
                        protocol="Per-person mean of the four physical EEG repeats; 17 posterior electrodes, 40 contiguous 20 ms means in [0,800 ms). Class-balanced ridge predicts continuous dual-encoder PCA coordinates. Choose ridge on held-out training concepts by macro cross-object pair wins, then lower squared error, then smaller ridge. Fit one fixed nonnegative gain by held-out least squares, capped at 2 and 75% of target RMS on calibration. The separate audit also measures between-concept and within-concept variation over distinct photos. Refit on all folds 1-4 after locking both scalars.",
                        rendering="All four steps receive the returned states. Use native-paper IP of the recipient for all arms; soft-wrong cyclically moves only the EEG text coordinates by floor(panel size/2).",
                        source_models="Original staged SDXL-Turbo text_encoder and text_encoder_2. Empty is encoded separately as a one-prompt batch. Mean is the equal-concept full training prompt average.")
            pack = dict(subject=subject, meta=meta, states=space["states"], head=head, gain=gain)
            atomic_write(path, lambda tmp: torch.save(pack, tmp))
        if not path.with_suffix(".json").is_file():
            save_receipt(path, pack["meta"])
    print(path, json.dumps(pack["meta"]["training"]), flush=True)


@torch.inference_mode()
def apply_pack(arm, pack, x, group):
    require(arm in ARMS, f"Unknown text arm {arm}")
    s, n = pack["states"], len(x)
    if arm in ("soft-empty","soft-mean"):
        prefix = arm.split("-")[1]
        return s[prefix+"_hidden"].expand(n,-1,-1), s[prefix+"_pooled"].expand(n,-1)
    code = predict_codes(pack["head"], eeg_features(x, group)) * pack["gain"]
    if arm == "soft-wrong":
        code = code[torch.from_numpy(wrong_donors(n))]
    hidden = (code @ s["hidden_basis"].flatten(1)).reshape(n,*s["mean_hidden"].shape) + s["mean_hidden"]
    pooled = code @ s["pooled_basis"] + s["mean_pooled"]
    require(torch.isfinite(hidden).all() and torch.isfinite(pooled).all(), "Non-finite decoded SDXL text states")
    return hidden, pooled


def apply_soft(arm, subject, x, group):
    """CPU float32 (N,77,2048), (N,1280), and the personal pack SHA. Pass the full panel for soft-wrong."""
    path = locations(subject)
    pack = torch.load(path, weights_only=True, map_location="cpu")
    require(pack["subject"] == subject, "Soft text participant mismatch")
    hidden, pooled = apply_pack(arm, pack, x, group)
    require(hidden.shape == (len(x),77,2048) and pooled.shape == (len(x),1280), "Invalid deployed SDXL text states")
    return hidden, pooled, sha(path)


def self_test():
    gen = torch.Generator().manual_seed(33)
    n, rank = 48, 4
    z = torch.randn(n,rank,generator=gen)
    blocks = tuple((z @ torch.randn(rank,15,generator=gen)).reshape(n,3,5) for _ in range(2)) + (
        z @ torch.randn(rank,7,generator=gen),)
    names = [f"train {i//2}" for i in range(n)]
    masks = partitions(names)
    require(all(len({int(mask[i]) for i,s in enumerate(names) if s==name})==1 for mask in masks for name in set(names)),
            "Duplicate prompts were split across calibration folds")
    b, c, retained = text_space(blocks, masks[0], rank, "cpu", True)
    require(retained > .9999, "Lost known toy rank")
    for real, directions, mean in zip(blocks,b["basis"],b["mean"]):
        torch.testing.assert_close((c @ directions.flatten(1)).reshape_as(real)+mean, real, rtol=2e-4, atol=5e-5)
    shifted = tuple(v+(~masks[0]).reshape(n,*([1]*(v.ndim-1))).float()*20 for v in blocks)
    _, shifted_codes, _ = text_space(shifted, masks[0], rank, "cpu", False)
    # Match PCA sign by comparing Gram entries on the fit-name rows.
    torch.testing.assert_close(c[masks[0]] @ c[masks[0]].T, shifted_codes[masks[0]] @ shifted_codes[masks[0]].T)
    cid = torch.arange(n).repeat_interleave(3)
    class_eeg = torch.randn(n,17,250,generator=gen)
    raw = (class_eeg[cid,None]+.1*torch.randn(len(cid),4,17,250,generator=gen)).half()
    feat = eeg_features(raw, [[0,1,2,3]])
    torch.testing.assert_close(feat, eeg_features(raw.numpy(), [[0,2],[1,3]]))
    decoder, gain, report = fit_person(feat, cid, dict(masks=masks,inner_codes=c,final_codes=c), "cpu")
    require(0 <= gain <= MAX_GAIN and report["calibration_pulse_rms_ratio"] <= MAX_CAL_RMS_RATIO+1e-6, "Unbounded gain")
    pack = dict(states=deploy_states(b, tuple(torch.zeros_like(a[0]) for a in blocks)), head=decoder, gain=.75)
    signal = raw[:21]
    h,p = apply_pack("soft-eeg", pack, signal, [[0,1,2,3]])
    wh,wp = apply_pack("soft-wrong", pack, signal, [[0,1,2,3]])
    donor = wrong_donors(len(signal))
    require(np.all(donor != np.arange(len(signal))) and len(np.unique(donor)) == len(signal), "Wrong donors are not a derangement")
    torch.testing.assert_close(wh,h[donor])
    torch.testing.assert_close(wp,p[donor])
    single = apply_pack("soft-eeg",pack,signal[4:5], [[0,1,2,3]])
    torch.testing.assert_close(single[0][0],h[4],rtol=1e-5,atol=1e-5)
    mean, _ = apply_pack("soft-mean",pack,signal, [[0,1,2,3]])
    empty, _ = apply_pack("soft-empty",pack,signal, [[0,1,2,3]])
    require(mean[0].equal(pack["states"]["mean_hidden"]) and not bool(empty.any()), "Fixed controls changed")
    require(clean_name("00042_fire_extinguisher") == "fire extinguisher", "Invalid word cleaning")
    print("CPU self-test passed: fit-only text PCA, two sequence blocks and pool, bounded calibration, four arms and label-blind donors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--subject", type=int, choices=SUBJECTS)
    action.add_argument("--build-space", action="store_true")
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", choices=("cuda","cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if args.self_test:
        self_test()
    else:
        train(args.subject, args.device)
