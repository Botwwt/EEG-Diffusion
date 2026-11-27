"""Render one image per EEG query with a stimulus-blind seed and fixed sample budget."""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import PREP, PRIOR, ROOT, RUNS, TURBO
from exp1_swapdiff_3arm import ResidualIPCondLayer, per_sample_cone
from exp6_draft_intra4 import DRAFT_DIR, forward_branches_and_trunk, rollout_4step_step_ips
from exp8_trial_precision_swapdiff import prec_erp_analytic
from main.models_brain import BrainEncoder
from o13_xbrain_shared_trunk import CrossSubjectHorizonEncoder
from paper_training import D1, D2, MATCHED, REC, SWAP, generator
from paper_render_cache import PinnedFiles, atomic_file, cached_layout, key, render_transaction, require, sha, track_for

BUDGETS = ("official-r4", "official-r4-balanced", "official-r80", "fresh-r4", "dev-r4")
ARMS = {"matched": MATCHED, "rec": REC, "swap": SWAP, "d1": D1, "d2": D2, "wrong": D2}
ARMS.update({"response": D2.parent / "response_s20261125.pt", "response-positive": D2.parent / "response_positive_s20261125.pt"})
ARMS.update({"d2-positive-half": DRAFT_DIR / "d2_positive_half_cond_st120_s2025.pt", "d2-mean-erp": D2})
# The two crossed arms reuse the identical condition checkpoints of their parent arms.
FACTOR_CONDITION = {"native-layout": "native-paper", "d2-no-layout": "d2", "d2-reference": "d2", "token-reliable-wrong": "token-reliable"}
INSTANCE_ARMS = ("instance-native", "instance-direct", "instance-residual")
TOKEN_ARMS = ("token-photo", "token-object", "token-reliable")
NO_LAYOUT = {"native-code", "native-paper", "d2-no-layout", "token-reliable-wrong", *TOKEN_ARMS, *INSTANCE_ARMS}
WINDOWS = ((12, 62), (10, 75), (15, 90), (10, 125), (0, 250))


def query_seed(panel, i):
    return int.from_bytes(hashlib.sha256(f"paper-20261125/{panel}/{i}".encode()).digest()[:8], "big") % (2**63 - 1)


def groups(budget):
    if budget == "official-r80":
        return np.arange(80).reshape(20, 4).tolist()
    return [[0, 20, 40, 60]] if budget == "official-r4-balanced" else [[0, 1, 2, 3]]


def precision_mean(x, indices):
    # Average the 20 R4 predictions before the single R80 render. A linear ridge
    # commutes with this group average; nonlinear semantic predictions are averaged later.
    output = []
    for start in range(0, len(x), 64):
        b = torch.from_numpy(np.array(x[start:start + 64], copy=True)).cuda().float()
        output.append(torch.stack([prec_erp_analytic(b[:, group], p=.5) for group in indices]).mean(0).cpu())
    return torch.cat(output)


def low(x, f):
    return F.interpolate(F.avg_pool2d(x.reshape(-1, 4, 64, 64), f), size=64, mode="bilinear", align_corners=False).flatten(1)


def _fit_layout(sub, plan, train):
    idx = np.asarray(plan["train_rows"])
    concepts = np.load(PREP / "train_concept.npy")[idx]
    calib = np.isin(concepts, np.unique(concepts)[::8])
    assert 0 < calib.sum() < len(idx)
    x = precision_mean(train[idx], [[0, 1, 2, 3]])
    y = torch.from_numpy(np.load(PREP / "lat512_train.npy", mmap_mode="r")[idx]).cuda().float().flatten(1)
    mu = y.mean(0)
    y -= mu
    torch.manual_seed(20261115 + sub)
    _, _, vec = torch.pca_lowrank(y + 1.5 * low(y, 4) + .75 * low(y, 2), q=288, center=False, niter=4)
    basis = torch.linalg.qr(vec[:, :256].double(), mode="reduced")[0]
    tgt = y.double() @ basis
    ref_rms = F.avg_pool2d(y[calib].view(-1, 4, 64, 64), 4).flatten(1).square().sum(1).mean().sqrt()
    windows, pred_cal = [], []
    for start, stop in WINDOWS:
        a = x[:, :, start:stop].flatten(1).cuda().double()
        for final in (False, True):
            fit, t = (a, tgt) if final else (a[~calib], tgt[~calib])
            center = fit.mean(0)
            scale = fit.std(0).clamp_min(1e-6)
            a_fit = (fit - center) / scale
            cov = a_fit.T @ a_fit / len(fit)
            ridge = 3 * cov.diagonal().mean()
            cov.diagonal().add_(ridge)
            w = torch.linalg.solve(cov, a_fit.T @ t / len(fit))
            if final:
                windows.append((center.float().cpu(), scale.float().cpu(), w.float().cpu()))
            else:
                pred_cal.append(((a[calib] - center) / scale @ w).float())
    cal_img = (torch.stack(pred_cal).mean(0) @ basis.float().T).view(-1, 4, 64, 64)
    pred_rms = F.avg_pool2d(cal_img, 4).flatten(1).square().sum(1).mean().sqrt()
    assert pred_rms > 0
    pack = {"windows": windows, "basis": basis.float().cpu(), "mean": mu.cpu(),
            "gamma": float(.80 * ref_rms / pred_rms), "calibration_rows": int(calib.sum())}
    return pack


def physical_eeg(sub, phase, files):
    base = PREP / f"sub-{sub:02d}_{phase}"
    path, receipt = base.with_suffix(".npy"), base.with_suffix(".json")
    eeg_ids, photo_ids = PREP / f"eeg_{phase}_ids.json", PREP / f"{phase}_ids.json"
    proof = {"split": phase, "subject": sub, "array_sha256": files.pin(path), "receipt_sha256": files.pin(receipt),
             "eeg_ids_sha256": files.pin(eeg_ids), "photo_ids_sha256": files.pin(photo_ids)}
    meta = json.loads(receipt.read_bytes())
    ids = json.loads(eeg_ids.read_bytes())["ids"]
    require(ids == json.loads(photo_ids.read_bytes())["ids"] and len(set(ids)) == len(ids), f"Physical EEG/photo identities disagree for {base}")
    require(int(str(meta.get("subject", sub)).removeprefix("sub-")) == sub and meta.get("split", phase) == phase,
            f"Acquisition receipt identifies another participant or split: {receipt}")
    require(meta["output_sha256"] == proof["array_sha256"], f"Physical EEG array differs from its acquisition receipt: {path}")
    if "ids_sha256" in meta:
        require(meta["ids_sha256"] == proof["eeg_ids_sha256"], f"Physical EEG identity receipt differs: {base}")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    require(x.shape == (len(ids), 80 if phase == "test" else 4, 17, 250) and x.dtype.kind == "f", f"Unexpected EEG axes at {path}")
    if "shape" in meta:
        require(list(x.shape) == meta["shape"], f"EEG shape and acquisition receipt differ: {path}")
    require(meta.get("dtype", str(x.dtype)) == str(x.dtype) and meta.get("sample_rate_hz", 250) == 250
            and meta.get("axes", ["stimulus", "physical_repeat", "channel", "sample"]) == ["stimulus", "physical_repeat", "channel", "sample"],
            f"EEG type or physical axis labels differ: {receipt}")
    expected = ROOT / "expected-inputs.json"
    if expected.is_file():
        proof["expected_inputs_sha256"] = files.pin(expected)
        item = json.loads(expected.read_bytes())[receipt.name]
        require(item["output_sha256"] == proof["array_sha256"] and item["source_sha256"] == meta.get("source_sha256")
                and item["shape"] == list(x.shape) and item["bytes"] == path.stat().st_size,
                f"EEG bytes differ from the independently registered acquisition at {path}")
    proof.update(shape=list(x.shape), dtype=str(x.dtype), source_sha256_as_recorded=meta.get("source_sha256"))
    files.verify()
    return x, proof


def runtime():
    import diffusers
    import transformers
    return {"torch": str(torch.__version__), "numpy": np.__version__, "pillow": Image.__version__, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(), "default_dtype": str(torch.get_default_dtype()),
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(), "math_sdp": torch.backends.cuda.math_sdp_enabled(),
            "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _validate_layout(pack):
    require(len(pack["windows"]) == len(WINDOWS) and tuple(pack["basis"].shape) == (4 * 64 * 64, 256)
            and tuple(pack["mean"].shape) == (4 * 64 * 64,) and math.isfinite(pack["gamma"]) and pack["gamma"] > 0
            and pack["calibration_rows"] > 0, "Invalid fitted ridge layout")
    tensors = [pack["basis"], pack["mean"]]
    for (start, stop), (center, scale, weight) in zip(WINDOWS, pack["windows"], strict=True):
        width = 17 * (stop - start)
        require(center.shape == scale.shape == (width,) and weight.shape == (width, 256) and (scale > 0).all(),
                "Invalid fitted ridge window")
        tensors.extend([center, scale, weight])
    require(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors), "Nonfinite or incorrect ridge tensor type")


def fit_layout(sub, plan, train, *, files=None, train_proof=None, with_source=False):
    files = PinnedFiles() if files is None else files
    if train_proof is None:
        bound_train, train_proof = physical_eeg(sub, "train", files)
        require(np.shares_memory(train, bound_train) or (getattr(train, "filename", None) is not None
                and Path(train.filename).resolve() == Path(bound_train.filename).resolve()), "Layout needs the registered physical EEG array")
        train = bound_train
    split_path = PREP / "paper_split.json"
    split_hash = files.pin(split_path)
    require(json.loads(split_path.read_bytes()) == plan, "Layout training plan differs from the registered split")
    rows = np.asarray(plan["train_rows"])
    require(rows.ndim == 1 and rows.dtype.kind in "iu" and len(np.unique(rows)) == len(rows) > 0
            and min(rows) >= 0 and max(rows) < len(train), "Invalid layout training rows")
    request = {"cache_version": 2, "subject": sub, "train_eeg": train_proof, "split_sha256": split_hash,
               "registered_train_rows_sha256": key(rows.tolist()), "train_concept_sha256": files.pin(PREP / "train_concept.npy"),
               "latent_sha256": files.pin(PREP / "lat512_train.npy"), "windows": [list(w) for w in WINDOWS], "pca_seed": 20261115 + sub,
               "algorithm_sha256": key([inspect.getsource(f) for f in (_fit_layout, precision_mean, prec_erp_analytic, low)]),
               "runtime": runtime()}
    pack, receipt = cached_layout(ROOT / "layouts" / f"sub-{sub:02d}", request, lambda: _fit_layout(sub, plan, train),
                                  _validate_layout, files.verify)
    folder = ROOT / "layouts" / f"sub-{sub:02d}" / "v2" / receipt["request_sha256"]
    require(files.pin(folder / "ridge.pt") == receipt["ridge_sha256"], "Layout was replaced during loading")
    source = {"checkpoint": (folder / "ridge.pt").relative_to(ROOT).as_posix(), "ridge_sha256": receipt["ridge_sha256"],
              "recipe_sha256": receipt["request_sha256"], "receipt_sha256": files.pin(folder / "ridge.json"), "training": request}
    return (pack, source) if with_source else pack


def layout(pack, x, group):
    erp = precision_mean(x, group)
    z = []
    for (start, stop), (center, scale, w) in zip(WINDOWS, pack["windows"]):
        z.append(((erp[:, :, start:stop].flatten(1).cuda() - center.cuda()) / scale.cuda()) @ w.cuda())
    return ((torch.stack(z).mean(0) @ pack["basis"].cuda().T) * pack["gamma"] + pack["mean"].cuda()).reshape(-1, 4, 64, 64).cpu()


def condition_files(arm, sub, files):
    track = track_for(RUNS, sub)
    nested = None
    fit_receipt = None
    if arm in TOKEN_ARMS:
        require(track == "personal", "Object-token conditions need the participant's personal fit")
        primary = RUNS / f"enc/object_tokens/sub-{sub:02d}/{arm}.pt"
        fit_receipt = primary.with_suffix(".json")
        files.pin(fit_receipt)
        fit = json.loads(fit_receipt.read_bytes())
        require(fit["subject"] == sub and fit["mode"] == arm and fit["checkpoint_sha256"] == files.pin(primary),
                "Object-token fit receipt does not match its participant, mode or encoder bytes")
    elif arm in INSTANCE_ARMS:
        from paper_instance_residual import locations
        native, fitted = locations(sub)
        primary = native if arm == "instance-native" else fitted
        nested = native if arm != "instance-native" else None
    elif arm.startswith("native"):
        fold = -1 if RUNS.name.startswith("native-full-") else 0
        runs = RUNS.parent / f"personal-{sub:02d}" if RUNS.name == "shared" else RUNS
        primary = runs / f"enc/native/sub-{sub:02d}/fold{fold}_seed2025/brain_model.pt"
    else:
        primary = ARMS[arm]
        files.pin(primary)
        header = torch.load(primary, map_location="meta", weights_only=True)
        expected_mode = {"d2": "d2_compete4_cond", "wrong": "d2_compete4_cond", "d2-mean-erp": "d2_compete4_cond",
                         "d2-positive-half": "d2_positive_half_cond", "d1": "d1_simonly_cond"}.get(arm)
        if expected_mode and "mode" in header:
            require(header["mode"] == expected_mode and header.get("steps", 120) == 120,
                    f"Expected {expected_mode} at the registered final training step: {primary}")
        if "state_dict" not in header:
            nested = Path(header["backbone_ckpt"])
    def source(path):
        if path is None:
            return None
        physical = path.absolute()
        name = physical.relative_to(ROOT.absolute()).as_posix() if physical.is_relative_to(ROOT.absolute()) else str(physical)
        return {"path": name, "sha256": files.pin(path)}
    proof = {"semantic_arm": arm, "condition_checkpoint": source(primary), "nested_backbone": source(nested)}
    if fit_receipt is not None:
        proof["training_receipt"] = source(fit_receipt)
    files.verify()
    return primary, nested, proof


def semantic_branch(enc, bb, sub, arm):
    if arm == "d2-mean-erp":
        f, h, _ = forward_branches_and_trunk(enc, bb.mean(1), sub - 1, 4.)
        return f, h
    f, h, _ = forward_branches_and_trunk(enc, prec_erp_analytic(bb), sub - 1, 4.)
    ft, ht, _ = forward_branches_and_trunk(enc, bb.flatten(0, 1), sub - 1, 1.)
    return .72 * f + .28 * ft.view(len(bb), 4, -1).mean(1), .72 * h + .28 * ht.view(len(bb), 4, -1).mean(1)


def condition(arm, sub, x, group, adapter=None, proj=None, *, features_only=False, files=None, prepared=None, return_source=False):
    files = PinnedFiles() if files is None else files
    primary, nested, proof = condition_files(arm, sub, files) if prepared is None else prepared
    track_for(RUNS, sub)
    digest = proof["condition_checkpoint"]["sha256"]
    def completed(value):
        files.verify()
        return (value, digest, proof) if return_source else (value, digest)
    if arm in INSTANCE_ARMS:
        from paper_instance_residual import apply_condition
        features, used = apply_condition(arm, sub, x, group)
        require(used == digest, "Instance condition checkpoint changed during use")
        if features_only:
            return completed(features)
        return completed(torch.cat([proj(adapter(batch.cuda().half())).cpu() for batch in features.split(20)]))
    if arm.startswith("native") or arm in TOKEN_ARMS:
        enc = BrainEncoder(backbone="brain_mlp", brain_channels=17, brain_sequence_length=250, hidden_size=1024, embedding_dim=1440, dropout=.1)
        enc.load_state_dict(torch.load(primary, map_location="cpu", weights_only=True))
        enc.cuda().float().eval()
    else:
        meta = torch.load(primary, map_location="cpu", weights_only=True)
        require((nested is None and "state_dict" in meta) or (nested is not None and Path(meta["backbone_ckpt"]) == nested),
                "Condition checkpoint now names a different semantic backbone")
        base = meta if nested is None else torch.load(nested, map_location="cpu", weights_only=True)
        enc = CrossSubjectHorizonEncoder(n_subs=10, num_layers=3, dropout=.20, in_drop=.15).cuda().float().eval()
        enc.load_state_dict(base["state_dict"])
        if not features_only:
            residual = ResidualIPCondLayer().cuda().float().eval()
            residual.load_state_dict(meta["cond_layer"])
        cone = meta["cone"]
    ips = []
    for start in range(0, len(x), 20):
        b = torch.from_numpy(np.array(x[start:start + 20], copy=True)).cuda().float()
        if arm.startswith("native") or arm in TOKEN_ARMS:
            c = enc.backbone(b[:, np.asarray(group).ravel().tolist()].mean(1))
            ip = c if features_only else proj(adapter(c.half()) if arm == "native-paper" or arm in TOKEN_ARMS else c.half())
        else:
            fs, hs = [], []
            for g in group:
                bb = b[:, g]
                f, h = semantic_branch(enc, bb, sub, arm)
                fs.append(f)
                hs.append(h)
            c = per_sample_cone(torch.stack(fs).mean(0), cone["R"], cone["alpha"], cone["beta"], cone["u_tr"].cuda().float())
            ip = c if features_only else residual(torch.stack(hs).mean(0), proj(adapter(c.half())))
        ips.append(ip.cpu())
    return completed(torch.cat(ips))


def draw_images(out, pipe, vae, prompts, tokens, seeds, z0s=None, *, batch_starts=None, batch_complete=None):
    prompt, pooled, dims = prompts
    if prompt.shape[0] != pooled.shape[0] or prompt.shape[0] not in (1, len(seeds)):
        raise ValueError("Expected a shared prompt or a separate text condition for each EEG query")
    with_layout = z0s is not None
    if not with_layout:
        assert math.isclose(float(pipe.scheduler.init_noise_sigma), float(pipe.scheduler.sigmas[0]), rel_tol=1e-6)
    starts = range(0, len(seeds), 8) if batch_starts is None else batch_starts
    for start in starts:
        require(0 <= start < len(seeds) and start % 8 == 0, "Render repair must preserve original batch boundaries")
        batch = slice(start, start + 8)
        noise = [torch.randn(5, 4, 64, 64, generator=torch.Generator().manual_seed(s)) for s in seeds[batch]]
        eps, *steps = torch.stack(noise).cuda().unbind(1)
        ip = tokens[batch].cuda()
        z0 = z0s[batch].cuda() if with_layout else torch.zeros_like(eps)
        text, pooled_text = (prompt, pooled) if len(prompt) == 1 else (prompt[batch], pooled[batch])
        text = text.to(device=ip.device, dtype=ip.dtype)
        pooled_text = pooled_text.to(device=ip.device, dtype=ip.dtype)
        z = rollout_4step_step_ips(pipe.unet, pipe.scheduler, [ip] * 4, z0, eps, steps, text, pooled_text, dims,
                                   .375 if with_layout else 0., (8, .11, .02) if with_layout else None)
        image = (vae.decode(z.float() / vae.config.scaling_factor).sample.float() * .5 + .5).clamp(0, 1)
        assert torch.isfinite(image).all()
        pixels = image.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for j, pixel in enumerate(pixels, start):
            path = out / f"{j:05d}.png"
            atomic_file(path, lambda f: Image.fromarray(pixel).save(f, format="PNG"))
        if batch_complete is not None:
            batch_complete(start, min(start + 8, len(seeds)))


def generator_inputs(files):
    artifacts = {"turbo/model_index.json": files.pin(TURBO / "model_index.json")}
    for root, prefix, components in [
        (TURBO, "turbo", ("unet", "vae", "scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2")),
        (PRIOR, "prior", ("fusion_encoder", "attn_adapter", "proj")),
    ]:
        for name in components:
            directory = root / name
            paths = sorted(p for p in directory.rglob("*") if p.is_file() and not any(s.startswith(".") for s in p.relative_to(directory).parts))
            require(paths, f"Missing staged generator component {directory}")
            for path in paths:
                artifacts[f"{prefix}/{path.relative_to(root).as_posix()}"] = files.pin(path)
    return artifacts


def render_code(files, arm):
    from o11_tri_stream_bigg import _lp_op
    functions = [condition, cached_layout, generator, BrainEncoder, CrossSubjectHorizonEncoder, ResidualIPCondLayer,
                 forward_branches_and_trunk, prec_erp_analytic, per_sample_cone, _lp_op]
    from main.models_adapter import FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel
    functions.extend([FusionEncoderModel, IPAttnAdapterModel, IPProjectionModel])
    if arm in INSTANCE_ARMS:
        from paper_instance_residual import apply_condition
        functions.append(apply_condition)
    paths = {Path(inspect.getfile(f)) for f in functions}
    return {p.name: files.pin(p) for p in sorted(paths)}


def tensor_fingerprint(t):
    t = t.detach().cpu().contiguous()
    return {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()}


def donor_rows(arm, panel, labels):
    rows = np.arange(len(labels))
    if arm == "wrong":
        return np.roll(rows, 1) if panel == "official" else np.roll(rows.reshape(-1, 10), 1, axis=1).ravel()
    if arm != "token-reliable-wrong":
        return rows
    labels = np.asarray(labels)
    if panel == "official":
        donors = np.roll(rows, len(rows) // 2)
    else:
        blocks = rows.reshape(-1, 10)
        require(all(len(set(labels[b])) == 1 for b in blocks) and len(set(labels[blocks[:, 0]])) == len(blocks),
                "Cross-object token control requires contiguous ten-photo concept groups")
        donors = np.roll(blocks, len(blocks) // 2, axis=0).ravel()
    require(np.all(labels != labels[donors]), "Token wrong-EEG control contains a same-object donor")
    return donors


@torch.inference_mode()
def render(a):
    output_root = a.output_root or (ROOT / "images")
    sub = a.subject
    track = track_for(RUNS, sub)
    if track == "native-full" and (any(not b.startswith("official-") for b in a.budgets)
                                   or any(arm not in ("native-paper", "native-code") for arm in a.arms)):
        raise ValueError("The full 1,654-concept native fit is for the official 200-stimulus split")
    for arm in a.arms:
        for budget in a.budgets:
            final = output_root / f"{track}/{arm}/{budget}/sub-{sub:02d}/render.json"
            if final.is_file() and not final.parent.with_name("." + final.parent.name + ".render-v2").exists():
                require(json.loads(final.read_bytes()).get("render_cache_version") == 2,
                        f"Historical render has no v2 input binding; preserve this folder: {final.parent}")
    files = PinnedFiles()
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    ref = json.loads(split.read_bytes())
    need_layout = any(arm not in NO_LAYOUT for arm in a.arms)
    phases = {"test" if b.startswith("official-") else "train" for b in a.budgets}
    if need_layout:
        phases.add("train")
    eeg = {phase: physical_eeg(sub, phase, files) for phase in sorted(phases)}
    ridge, ridge_source = (fit_layout(sub, ref, eeg["train"][0], files=files, train_proof=eeg["train"][1], with_source=True)
                           if need_layout else (None, None))
    generator_files = generator_inputs(files)
    from diffusers import AutoencoderKL
    pipe, adapter, proj, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    fixed = {"generator_files_sha256": generator_files, "runtime": runtime(), "scheduler": type(pipe.scheduler).__name__,
             "timesteps": pipe.scheduler.timesteps.detach().cpu().tolist(), "sigmas": pipe.scheduler.sigmas.detach().cpu().tolist(),
             "initial_noise_sigma": float(pipe.scheduler.init_noise_sigma), "initial_and_step_noise_shape_per_query": [5, 4, 64, 64],
             "rng": "CPU torch.Generator manual_seed per query", "render_batch": 8, "semantic_batch": 20, "precision_ERP_batch": 64,
             "unet_dtype": str(pipe.unet.dtype), "vae_dtype": str(vae.dtype), "vae_scaling_factor": float(vae.config.scaling_factor),
             "vae_slicing": True, "classifier_free_guidance": False, "output_rgb": [512, 512, 3],
             "prompt_embedding": tensor_fingerprint(prompt), "pooled_prompt": tensor_fingerprint(pooled),
             "added_time_ids": tensor_fingerprint(dims)}
    files.verify()
    for budget in a.budgets:
        panel = budget.split("-")[0]
        phase = "test" if panel == "official" else "train"
        raw, physical = eeg[phase]
        dev_sha = None
        if panel == "dev":
            dev = PREP / "paper_development.json"
            dev_sha = files.pin(dev)
            qrows = json.loads(dev.read_bytes())["dev_rows"]
        else:
            qrows = ref["fresh_rows"] if panel == "fresh" else ref["official_rows"]
        require(len(qrows) == len(set(qrows)) == (500 if panel == "fresh" else 200) and
                all(type(r) is int and 0 <= r < len(raw) for r in qrows), "Invalid registered render rows")
        require(panel != "official" or qrows == list(range(len(raw))), "Official EEG rows/order differ from the complete test panel")
        x = raw if panel == "official" else raw[qrows]
        photo_paths = json.loads((PREP / f"{phase}_ids.json").read_bytes())["paths"]
        labels = [Path(photo_paths[r]).parent.name.split("_", 1)[1] for r in qrows]
        g = groups(budget)
        zero = None
        seeds = [query_seed(panel, i) for i in range(len(x))]
        assert len(set(seeds)) == len(x)
        for arm in a.arms:
            out = output_root / f"{track}/{arm}/{budget}/sub-{sub:02d}"
            source_arm = FACTOR_CONDITION.get(arm, arm)
            prepared = condition_files(source_arm, sub, files)
            cp = prepared[2]
            donors = donor_rows(arm, panel, labels)
            with_layout = arm not in NO_LAYOUT
            mode = "arithmetic per R4, direct effective repeats 4" if source_arm == "d2-mean-erp" else (
                "arithmetic selected repeats" if source_arm.startswith(("native", "instance")) or source_arm in TOKEN_ARMS
                else "M1a precision ERP plus individual trials")
            request = dict(render_cache_version=2, training=track, subject=sub, arm=arm, budget=budget, n=len(x), prompt="",
                           draws=1, unet_calls_per_draw=4, repeats=g, row=qrows, eeg_donor=donors.tolist(), seeds=seeds,
                           checkpoint_sha256=cp["condition_checkpoint"]["sha256"], condition_sources=cp, physical_eeg=physical,
                           split_sha256=split_sha, development_split_sha256=dev_sha, semantic_condition=source_arm, semantic_EEG=mode,
                           layout_mode="ridge-shell" if with_layout else "noise", layout_source=ridge_source if with_layout else None,
                           render_code_sha256=render_code(files, source_arm), sampling={**fixed, "smap_c": .375 if with_layout else 0.,
                           "low_keep": [8, .11, .02] if with_layout else None})

            def generate(stage, starts, complete):
                nonlocal zero
                if with_layout and zero is None:
                    zero = layout(ridge, x, g)
                landmark, used = condition(source_arm, sub, x, g, adapter, proj, files=files, prepared=prepared)
                require(used == request["checkpoint_sha256"], "Rendered semantic checkpoint differs from request")
                draw_images(stage, pipe, vae, (prompt, pooled, dims), landmark[donors], seeds, zero[donors] if with_layout else None,
                            batch_starts=starts, batch_complete=complete)

            wrote = render_transaction(out, request, generate, files.verify)
            print("rendered" if wrote else "verified", out, len(x), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, required=True, choices=range(1, 11))
    p.add_argument("--arms", nargs="+", choices=("native-code", "native-paper", *ARMS, *FACTOR_CONDITION, *INSTANCE_ARMS, *TOKEN_ARMS), required=True)
    p.add_argument("--budgets", nargs="+", choices=BUDGETS, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    render(p.parse_args())
