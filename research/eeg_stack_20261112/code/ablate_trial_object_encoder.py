"""Cross reliability pooling and within-photo supervision on 96 held generic objects."""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from fit_personal_session_metric import similarities
from fit_shared_photo_eeg import require
from fit_trial_object_encoder import TrialObjectEncoder, get_records, source_data
from register_fresh_photo_panel import freeze, read, sha
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity"
DESIGN = PUBLIC / "trial_object_ablation_design.json"
DESIGN_SHA = "51ae4dd7c897975822e91241b93818d3943781fba527750142ec11eaaea781d4"
ARMS = ("weighted_pair", "weighted_none", "mean_pair", "mean_none")


class PoolingEncoder(TrialObjectEncoder):
    def __init__(self, reliability):
        super().__init__()
        self.reliability = reliability

    def aggregate(self, trials):
        if self.reliability:
            return super().aggregate(trials)
        return F.normalize(self.head(trials.mean(dim=1)), dim=-1)


def loss_for(model, samples, permutations, full_query, coefficient):
    features = model.trials(samples.flatten(0, 1))
    index = torch.arange(len(features), device=features.device)[:, None]
    left = model.aggregate(features[index, permutations[:, :2]]).reshape(12, 10, 96)
    right = model.aggregate(features[index, permutations[:, 2:]]).reshape(12, 10, 96)
    query = model.aggregate(features).reshape(12, 10, 96) if full_query else right
    supports = left.reshape(12, 5, 2, 96)
    centers = F.normalize(left.sum(1)[:, None] - supports.sum(2), dim=-1)
    scores = torch.einsum("cfpd,kfd->cfpk", query.reshape(12, 5, 2, 96), centers)
    objects = torch.arange(12, device=scores.device).repeat_interleave(10)
    photo_scores = torch.bmm(left, right.transpose(1, 2))
    photos = torch.arange(10, device=scores.device).repeat(12)
    return F.cross_entropy(scores.reshape(-1, 12) * 12, objects) + coefficient * F.cross_entropy(
        photo_scores.reshape(-1, 10) * 12, photos
    )


def fit(data, sessions, reliability, coefficient, spec, device):
    torch.manual_seed(spec["seed"])
    rng = np.random.default_rng(spec["seed"])
    model = PoolingEncoder(reliability).to(device)
    initial, episodes, noise = (hashlib.sha256() for _ in range(3))
    for k, v in model.state_dict().items():
        initial.update(k.encode())
        initial.update(v.detach().cpu().contiguous().numpy().tobytes())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec["updates"])
    cells = [np.flatnonzero(sessions == i) for i in np.unique(sessions)]
    require(min(map(len, cells)) >= 12, "Insufficient same-session training objects")
    losses = []
    for step in range(spec["updates"]):
        classes = rng.choice(cells[step % len(cells)], size=12, replace=False)
        photos = np.stack([rng.permutation(10) for _ in classes])
        sample = data[
            torch.as_tensor(classes, device=device)[:, None], torch.as_tensor(photos, device=device)
        ]
        splits = torch.rand(120, 4, device=device).argsort(dim=1)
        for v in (classes, photos, splits.cpu().numpy()):
            episodes.update(v.tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, sample, splits, step % 2 == 1, coefficient)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
        noise.update(state.numpy().tobytes())
        losses.append(float(loss.detach()))
        if (step + 1) % 150 == 0:
            print(
                "updates", step + 1, "mean training loss", float(np.mean(losses[-150:])), flush=True
            )
    require(np.isfinite(losses).all(), "Nonfinite training loss")
    return model.eval(), dict(
        initial_state=initial.hexdigest(),
        episode_stream=episodes.hexdigest(),
        post_update_rng_stream=noise.hexdigest(),
    )


def score_embeddings(original, test_ids, features):
    raw = (10 * test_ids[:, None] + np.arange(10)).ravel()
    z = features.reshape(-1, 96).astype(float)
    result = {}
    for name in ("generic_4", "generic_12"):
        with np.load(original / (name + ".npz"), allow_pickle=False) as old:
            rows, support, candidates, fisher = (
                old[k] for k in ("rows", "support_rows", "candidate_ids", "fisher_scores")
            )
            require(
                len(rows) == len(np.unique(rows)) == 960 and np.array_equal(np.sort(rows), raw),
                "Wrong generic photo list",
            )
            qi = np.searchsorted(raw, rows)
            si = np.searchsorted(raw, support)
            require(
                np.array_equal(raw[si], support)
                and np.all(support // 10 == candidates[:, :, None])
                and not (support == rows[:, None, None]).any(),
                "Invalid eight-support gallery",
            )
            deep = np.empty_like(fisher)
            for i in range(0, len(rows), 2):
                require(np.array_equal(si[i], si[i + 1]), "Generic pair menu differs")
                deep[i : i + 2] = similarities(z[qi[i : i + 2]], z[si[i]])
            index = np.arange(len(rows))
            v = dict(
                rows=rows,
                truth=old["truth_id"],
                deep=candidates[index, deep.argmax(1)],
                mixed=candidates[index, (fisher + deep).argmax(1)],
            )
            result[name] = v
    return result


def train(a, spec):
    source = a.inputs / "sub-04"
    official = a.original / "generic"
    baseline = read(official / "receipt.json")
    require(
        sha(official / "receipt.json") == spec["original_full_generic_training_receipt"]
        and sha(official / "score.json") == spec["original_full_generic_score"],
        "Archived generic run differs",
    )
    train_ids, test_ids = (np.asarray(baseline[k]) for k in ("train_classes", "test_classes"))
    require(
        len(train_ids) == 416
        and len(test_ids) == 96
        and not np.intersect1d(train_ids, test_ids).size,
        "Wrong train/test objects",
    )
    records = get_records(source, ("generic512",))
    require(
        baseline["sources"]
        == {str(i): records[str(i)]["output_sha256"] for i in np.r_[train_ids, test_ids]}
        and sha(a.sessions) == baseline["session_manifest_sha256"],
        "Different input waveforms or sessions",
    )
    device = torch.device(a.device)
    data = source_data(source, train_ids, records, device)
    sd, mu = torch.std_mean(data, dim=(0, 1, 2, 4), unbiased=False)
    sd = sd.clamp_min(1e-5)[:, None]
    mu = mu[:, None]
    data.sub_(mu).div_(sd).clamp_(-12, 12)
    dates = np.array(read(a.sessions)["subjects"]["sub-04"]["session_pairs"])
    _, sessions = np.unique(dates[train_ids], axis=0, return_inverse=True)
    test = source_data(source, test_ids, records, device)
    for arm in a.arms:
        folder = a.out / arm
        if (folder / "receipt.json").exists():
            r = read(folder / "receipt.json")
            require(
                r["source_sha256"] == sha(Path(__file__))
                and r["design_sha256"] == sha(DESIGN)
                and all(sha(folder / f) == h for f, h in r["files"].items()),
                "Cached ablation changed",
            )
            continue
        model, trace = fit(data, sessions, *spec["arms"][arm], spec, device)
        with torch.inference_mode():
            y = torch.cat(
                [
                    model(((block - mu) / sd).clamp(-12, 12))
                    for block in test.flatten(0, 1).split(128)
                ]
            )
        y = y.reshape(len(test_ids), 10, 96).cpu().numpy()
        require(np.allclose(np.linalg.norm(y, axis=-1), 1, atol=1e-5), "Invalid prediction")
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / "model.safetensors"
        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        state.update(input_mean=mu.cpu(), input_scale=sd.cpu())
        save_file(state, p)
        np.savez(folder / "embeddings.npz", ids=test_ids, features=y)
        files = {"model.safetensors": sha(p), "embeddings.npz": sha(folder / "embeddings.npz")}
        data_cases = score_embeddings(official, test_ids, y)
        replay = None
        if arm == "weighted_pair":
            with np.load(official / "embeddings.npz", allow_pickle=False) as old:
                err = float(np.abs(y - old["features"]).max())
            old_state = load_file(official / "model.safetensors")
            weight_err = max(float((state[k] - old_state[k]).abs().max()) for k in state)
            replay = dict(
                max_embedding_absolute_error=err,
                max_weight_absolute_error=weight_err,
                exact_predictions={},
            )
        for case, v in data_cases.items():
            if replay is not None:
                with np.load(official / (case + ".npz"), allow_pickle=False) as old:
                    require(
                        np.array_equal(old["rows"], v["rows"])
                        and np.array_equal(old["truth_id"], v["truth"])
                        and np.array_equal(old["mixed"], v["mixed"])
                        and np.array_equal(old["encoder"], v["deep"]),
                        "Full arm did not reproduce the original decisions",
                    )
                    replay["exact_predictions"][case] = len(v["rows"])
            path = folder / (case + ".npz")
            np.savez(path, **v)
            files[path.name] = sha(path)
        freeze(
            folder / "receipt.json",
            dict(
                subject="sub-04",
                arm=arm,
                source_sha256=sha(Path(__file__)),
                design_sha256=sha(DESIGN),
                files=files,
                trace_sha256=trace,
                full_replay=replay,
            ),
        )
        print(arm, "frozen", trace, flush=True)


def contrasted(values, draws):
    return dict(
        delta=float(values.mean()), ci95=np.quantile(values[draws].mean(1), [0.025, 0.975]).tolist()
    )


def summarize(a, spec):
    reports = {arm: read(a.out / arm / "receipt.json") for arm in ARMS}
    trace = reports["weighted_pair"]["trace_sha256"]
    for arm, r in reports.items():
        require(
            r["source_sha256"] == sha(Path(__file__))
            and r["design_sha256"] == sha(DESIGN)
            and r["trace_sha256"] == trace
            and all(sha(a.out / arm / f) == h for f, h in r["files"].items()),
            "Ablation arms differ in RNG, design, source or output artifacts",
        )
    result = dict(
        design_sha256=sha(DESIGN),
        source_sha256=sha(Path(__file__)),
        full_replay=reports["weighted_pair"]["full_replay"],
        participant="sub-04",
        training_objects=416,
        held_objects=96,
        updates=spec["updates"],
        same_initialization_and_random_streams=trace,
        intervals="Absolute differences in top-1 accuracy, paired by 96 whole held objects. 10000 percentile draws, seed 260552. Positive favors the full/weighted/photo-loss direction as named. The original full-run performance was known before this post hoc component study.",
        model_weights_sha256={arm: r["files"]["model.safetensors"] for arm, r in reports.items()},
        training_receipt_sha256={arm: sha(a.out / arm / "receipt.json") for arm in ARMS},
        cases={},
    )
    for case in ("generic_12", "generic_4"):
        raw = []
        for arm in ARMS:
            with np.load(a.out / arm / (case + ".npz"), allow_pickle=False) as z:
                raw.append(dict(z))
        truth = raw[0]["truth"]
        require(
            all(
                np.array_equal(d["rows"], raw[0]["rows"]) and np.array_equal(d["truth"], truth)
                for d in raw
            ),
            "Ablation query alignment differs",
        )
        ids, counts = np.unique(truth, return_counts=True)
        require(len(ids) == 96 and (counts == 10).all(), "Wrong class denominator")
        draws = np.random.default_rng(spec["bootstrap_seed"]).integers(
            96, size=(spec["bootstrap_draws"], 96)
        )
        result["cases"][case] = {}
        for mode in ("deep", "mixed"):
            # Each object contributes its ten photos as one cluster.
            rates = np.array([[(d[mode][truth == v] == v).mean() for v in ids] for d in raw])
            a11, a10, a01, a00 = rates
            result["cases"][case][mode] = dict(
                n=960,
                hits={k: int((v[mode] == truth).sum()) for k, v in zip(ARMS, raw, strict=True)},
                full_minus={k: contrasted(a11 - rates[i], draws) for i, k in enumerate(ARMS) if i},
                marginal_weighted_minus_mean=contrasted(((a11 - a01) + (a10 - a00)) / 2, draws),
                marginal_pair_minus_no_pair=contrasted(((a11 - a10) + (a01 - a00)) / 2, draws),
                interaction=contrasted((a11 - a10) - (a01 - a00), draws),
            )
    freeze(a.out / "result.json", result)
    freeze(PUBLIC / "trial_object_ablation.json", result)
    print(result["cases"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "summarize"))
    p.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for k, v in dict(
        inputs="outputs/trial-object-inputs",
        original="outputs/trial-object-encoder",
        sessions="outputs/object-transfer/sessions.json",
        out="outputs/trial-object-ablation",
    ).items():
        p.add_argument("--" + k, type=Path, default=Path(v))
    a = p.parse_args()
    spec = read(DESIGN)
    require(
        sha(DESIGN) == DESIGN_SHA
        and sha(CODE / "fit_trial_object_encoder.py") == spec["original_train_sha256"]
        and sha(CODE / "score_trial_object_encoder.py") == spec["original_scorer_sha256"]
        and sha(PUBLIC / "trial_object_encoder_design.json") == spec["original_design_sha256"],
        "Frozen training or evaluation specification changed",
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if a.stage == "train":
        train(a, spec)
    else:
        summarize(a, spec)


if __name__ == "__main__":
    main()
