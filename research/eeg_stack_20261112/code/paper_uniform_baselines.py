"""Run shared-training EEG encoders with the registered paper image generator."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import HIVE, PREP, PRIOR, ROOT, TURBO
from paper_render import condition, condition_files, draw_images, generator_inputs, groups, layout, physical_eeg, query_seed, runtime
from paper_render_cache import PinnedFiles, atomic_file, require, render_transaction, save_json
from paper_training import generator

from main.models_adapter import FusionEncoderModel
from main.models_brain import BrainEncoder

BASELINES = {
    "brain-hive": ("brain_mlp", 1440, 0.1),
    "atm": ("atm", 1440, 0.25),
    "nice": ("nice", 1440, 0.5),
    "eegnet": ("eeg_net", 1248, 0.5),
}
METHODS = (*BASELINES, "ridge")
RENDER_METHODS = (*METHODS, "m1a-d2")
BUDGETS = ("official-r4", "official-r80")
PEOPLE = 10
PASSES = 25
BATCH_SIZE = 256
SEED = 2025


def registered_split(files):
    path = PREP / "paper_split.json"
    digest = files.pin(path)
    split = json.loads(path.read_bytes())
    rows = np.asarray(split["train_rows"], dtype=np.int64)
    concepts = np.load(PREP / "train_concept.npy", allow_pickle=False)
    files.pin(PREP / "train_concept.npy")
    require(len(rows) == len(np.unique(rows)) == 13230, "Expected registered training photo rows")
    require(len(np.unique(concepts[rows])) == 1323, "Unexpected training concept count")
    require(not np.intersect1d(rows, split["fresh_rows"]).size, "Fresh test rows entered training")
    require(split["official_rows"] == list(range(200)), "Invalid official test order")
    return split, rows, digest


def prepare(run):
    files = PinnedFiles()
    _, rows, split_hash = registered_split(files)
    run.mkdir(parents=True, exist_ok=True)
    cache = run / "train-erp.npy"
    receipt = run / "train-erp.json"
    if receipt.exists():
        used = json.loads(receipt.read_bytes())
        require(used["split_sha256"] == split_hash and used["erp_sha256"] == files.pin(cache), "Training ERP cache differs")
        return
    temporary = run / ".train-erp.partial.npy"
    erp = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(PEOPLE, len(rows), 17, 250))
    sources = []
    for subject in range(1, PEOPLE+1):
        trials, source = physical_eeg(subject, "train", files)
        for start in range(0, len(rows), 256):
            selected = rows[start:start+256]
            erp[subject-1, start:start+len(selected)] = trials[selected].mean(axis=1, dtype=np.float32)
        sources.append(source)
        print(f"Bound training EEG for participant {subject:02d}", flush=True)
    erp.flush()
    del erp
    files.verify()
    temporary.replace(cache)
    save_json(receipt, {"split_sha256": split_hash, "erp_sha256": files.pin(cache), "sources": sources,
                        "aggregation": "arithmetic mean of recorded training presentations 0,1,2,3"})
    print(f"Prepared {PEOPLE*len(rows)} registered EEG photo averages", flush=True)


class JointEncoder(nn.Module):
    def __init__(self, method):
        super().__init__()
        backbone, dim, dropout = BASELINES[method]
        self.encoder = BrainEncoder(backbone=backbone, embedding_dim=dim, dropout=dropout)
        self.subject_offset = nn.Embedding(PEOPLE, 1024)
        nn.init.zeros_(self.subject_offset.weight)
        self.log_scale = nn.Parameter(torch.tensor(2.6592))

    def forward(self, eeg, subject):
        return self.encoder(eeg, subject) + self.subject_offset(subject)


def training_reference(rows, files):
    source = PREP / "emb_train.pt"
    emb_hash = files.pin(source)
    visual_root = PRIOR / "fusion_encoder"
    teacher_hashes = {p.name: files.pin(p) for p in sorted(visual_root.iterdir()) if p.is_file()}
    teacher = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model
    teacher = teacher.cuda().eval().requires_grad_(False)
    embeddings = torch.load(source, map_location="cpu", weights_only=True)
    embeddings = {k: v[rows].cuda() for k, v in embeddings.items()}
    require(embeddings and all(len(v) == len(rows) for v in embeddings.values()), "Frozen teacher row mismatch")
    with torch.no_grad():
        y = torch.cat([teacher(**{k: v[i:i+256].float() for k, v in embeddings.items()}).float()
                       for i in range(0, len(rows), 256)])
    require(y.shape == (len(rows), 1024) and bool(torch.isfinite(y).all()), "Invalid frozen image targets")
    return y, {"embeddings_sha256": emb_hash, "fusion_teacher_sha256": teacher_hashes}


def fit_ridge(x, y):
    # Use all training examples. Center each participant with training EEG only.
    count = x.size(0) * x.size(1)
    width = x.shape[2] * x.shape[3]
    matrix = torch.zeros((width, width), device="cuda")
    rhs = torch.zeros((width, y.shape[-1]), device="cuda")
    centers = []
    centered_y = y - y.mean(0)
    for person in x:
        a = person.flatten(1).float()
        mu = a.mean(0)
        a = a - mu
        centers.append(mu.cpu())
        matrix.addmm_(a.T, a, alpha=1 / count)
        rhs.addmm_(a.T, centered_y, alpha=1 / count)
    matrix.diagonal().add_(3 * matrix.diagonal().mean())
    weight = torch.linalg.solve(matrix, rhs)
    require(bool(torch.isfinite(weight).all()), "Nonfinite ridge weights")
    return {"centers": torch.stack(centers), "weight": weight.cpu(), "target_center": y.mean(0).cpu()}


def fit_encoder(method, x, y):
    model = JointEncoder(method).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.995), weight_decay=0.05)
    rnd = torch.Generator(device="cuda").manual_seed(SEED)
    n = len(y)
    offsets = torch.randint(PEOPLE, (n,), generator=rnd, device="cuda")
    n_updates = PASSES * ((n + BATCH_SIZE - 1) // BATCH_SIZE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_updates)
    goal = F.normalize(y)
    log = []
    for epoch in range(PASSES):
        indices = torch.randperm(n, generator=rnd, device="cuda")
        total = torch.zeros((), device="cuda")
        for photo in indices.split(BATCH_SIZE):
            subject = (offsets[photo] + epoch) % PEOPLE
            eeg = x[subject, photo].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = model(eeg, subject)
            logits = F.normalize(value.float()) @ goal[photo].T * model.log_scale.clamp(max=4.60517).exp()
            labels = torch.arange(len(photo), device="cuda")
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            schedule.step()
            total += loss.detach() * len(photo)
        log.append(float(total / n))
        require(np.isfinite(log[-1]), f"Nonfinite {method} training loss")
        print(f"{method} pass {epoch+1}/{PASSES} loss {log[-1]:.6f}", flush=True)
    return {"weights": {k: v.cpu() for k, v in model.state_dict().items()}, "loss": log, "updates": n_updates}


def fit(run, method):
    folder = run / "fits" / method
    folder.mkdir(parents=True, exist_ok=True)
    files = PinnedFiles()
    _, rows, split_hash = registered_split(files)
    prep = run / "train-erp.npy"
    prep_meta = run / "train-erp.json"
    erp_proof = json.loads(prep_meta.read_bytes())
    require(erp_proof["split_sha256"] == split_hash and erp_proof["erp_sha256"] == files.pin(prep), "Training ERP cache differs")
    prep_hash = files.pin(prep_meta)
    cp, receipt = folder / "checkpoint.pt", folder / "fit.json"
    code_path = Path(__file__)
    model_code = {"experiment_sha256": files.pin(code_path),
                  "encoders_sha256": files.pin(HIVE / "main/models_brain.py"),
                  "layers_sha256": files.pin(HIVE / "main/layers.py")}
    if method == "atm":
        model_code["atm_sha256"] = {p.name: files.pin(p) for p in sorted((HIVE / "main/atm").rglob("*.py"))}
    if receipt.exists():
        old = json.loads(receipt.read_bytes())
        require(old["checkpoint_sha256"] == files.pin(cp) and old["train_cache_sha256"] == prep_hash
                and old["code"] == model_code and old["method"] == method, "Existing fit does not match this protocol")
        return
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arr = np.load(prep, allow_pickle=False)
    require(arr.shape == (PEOPLE, len(rows), 17, 250), "Unexpected training EEG axes")
    x = torch.from_numpy(arr).cuda()
    del arr
    target, target_proof = training_reference(rows, files)
    state = fit_ridge(x, target) if method == "ridge" else fit_encoder(method, x, target)
    files.verify()
    atomic_file(cp, lambda f: torch.save(state, f))
    save_json(receipt, {"method": method, "subjects": list(range(1, PEOPLE+1)), "train_concepts": 1323,
                        "train_photos": len(rows), "split_sha256": split_hash, "train_cache_sha256": prep_hash,
                        "checkpoint_sha256": files.pin(cp), "code": model_code, "target": target_proof,
                        "aggregation": "train arithmetic R4", "seed": SEED,
                        "optimizer": "ridge closed form" if method == "ridge" else "AdamW cosine",
                        "passes": 0 if method == "ridge" else PASSES, "loss": state.get("loss")})
    print("Bound baseline checkpoint", method, flush=True)


def reference_layout(subject, split_hash, files):
    reference = ROOT / "images/shared/d2-reference/official-r4" / f"sub-{subject:02d}" / "render.json"
    files.pin(reference)
    saved = json.loads(reference.read_bytes())
    require(saved["subject"] == subject and saved["split_sha256"] == split_hash, "Wrong registered layout reference")
    source = saved["layout_source"]
    relative = Path(source["checkpoint"])
    path = ROOT / relative
    require(not relative.is_absolute() and path.resolve().is_relative_to((ROOT / "layouts").resolve()),
            "Layout checkpoint outside the registered layout directory")
    require(source["ridge_sha256"] == files.pin(path)
            and source["receipt_sha256"] == files.pin(path.with_suffix(".json")), "Layout files differ from the reference")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["request_sha256"] == source["recipe_sha256"], "Layout recipe differs from the reference")
    return payload["layout"], source


@torch.inference_mode()
def render(run, image_root, subject, methods, budgets):
    from diffusers import AutoencoderKL

    files = PinnedFiles()
    split, _, split_hash = registered_split(files)
    x, test_source = physical_eeg(subject, "test", files)
    ridge, layout_source = reference_layout(subject, split_hash, files)
    generator_files = generator_inputs(files)
    pipe, adapter, proj, prompt, pooled, dims = generator(torch.float16)
    pipe.vae = None
    vae = AutoencoderKL.from_pretrained(str(TURBO), subfolder="vae", variant="fp16", torch_dtype=torch.float32).cuda().eval()
    vae.enable_slicing()
    participant = subject - 1
    rows = split["official_rows"]
    seeds = [query_seed("official", i) for i in rows]
    generator_receipt = {"files_sha256": generator_files, "runtime": runtime(),
                         "timesteps": pipe.scheduler.timesteps.cpu().tolist(), "rgb": [512, 512, 3],
                         "render_batch": 8, "vae_slicing": True, "smap_c": 0.375, "low_keep": [8, 0.11, 0.02]}
    code = {"experiment": files.pin(__file__), "renderer": files.pin(__import__("paper_render").__file__)}
    erps = {}
    latents = {}
    for budget in budgets:
        repeat_groups = groups(budget)
        physical = np.asarray(repeat_groups).ravel().tolist()
        erps[budget] = torch.from_numpy(x[:, physical].mean(axis=1, dtype=np.float32)).cuda()
        latents[budget] = layout(ridge, x, repeat_groups)
    for method in methods:
        if method == "m1a-d2":
            prepared = condition_files("d2", subject, files)
            cp_hash = prepared[2]["condition_checkpoint"]["sha256"]
            training = {"condition_sources": prepared[2]}
        else:
            paths = run / "fits" / method
            cp = paths / "checkpoint.pt"
            fit_path = paths / "fit.json"
            provenance = json.loads(fit_path.read_bytes())
            training = {"train_fit_sha256": files.pin(fit_path)}
            cp_hash = files.pin(cp)
            require(provenance["split_sha256"] == split_hash and provenance["checkpoint_sha256"] == cp_hash
                    and provenance["method"] == method and provenance["subjects"] == list(range(1, PEOPLE+1)),
                    "Mismatched baseline checkpoint")
            state = torch.load(cp, map_location="cpu", weights_only=True)
            if method != "ridge":
                model = JointEncoder(method).cuda().eval()
                model.load_state_dict(state["weights"])
        for budget in budgets:
            out = image_root / method / budget / f"sub-{subject:02d}"
            if method == "m1a-d2":
                tokens, used = condition("d2", subject, x, groups(budget), adapter, proj, files=files, prepared=prepared)
                require(used == cp_hash, "Wrong M1a+D2 checkpoint during rendering")
            else:
                eeg = erps[budget]
                if method == "ridge":
                    centered = eeg.flatten(1) - state["centers"][participant].cuda()
                    features = centered @ state["weight"].cuda() + state["target_center"].cuda()
                else:
                    features = torch.cat([model(b, torch.full((len(b),), participant, device="cuda", dtype=torch.long))
                                          for b in eeg.split(20)])
                tokens = torch.cat([proj(adapter(b.half())).cpu() for b in features.split(20)])
            require(len(tokens) == 200 and bool(torch.isfinite(tokens).all()), "Invalid baseline conditions")
            request = dict(render_cache_version=2, training="shared", subject=subject, arm=method, budget=budget,
                           n=len(rows), prompt="", draws=1, unet_calls_per_draw=4, row=rows, eeg_donor=rows,
                           repeats=groups(budget), seeds=seeds, checkpoint_sha256=cp_hash, **training,
                           physical_eeg=test_source, split_sha256=split_hash, layout_mode="ridge-shell",
                           layout_source=layout_source, code=code, sampling=generator_receipt)

            def generate(stage, starts, complete):
                draw_images(stage, pipe, vae, (prompt, pooled, dims), tokens, seeds, latents[budget],
                            batch_starts=starts, batch_complete=complete)

            rendered = render_transaction(out, request, generate, files.verify)
            print("rendered" if rendered else "verified", out, len(tokens), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("method", choices=METHODS)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--image-root", type=Path, help="Place large PNG and score writes on node-local storage")
    render_parser.add_argument("--subject", type=int, required=True, choices=range(1, PEOPLE+1))
    render_parser.add_argument("--methods", nargs="+", choices=RENDER_METHODS, default=RENDER_METHODS)
    render_parser.add_argument("--budgets", nargs="+", choices=BUDGETS, default=("official-r4",))
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "Run EEG arrays and GPU models on a Slurm compute node")
    if args.command == "prepare":
        prepare(args.run)
    elif args.command == "fit":
        fit(args.run, args.method)
    else:
        render(args.run, args.image_root or args.run / "images", args.subject, args.methods, args.budgets)


if __name__ == "__main__":
    main()
