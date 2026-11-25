"""Cross external object count and training updates for a fixed sub-04 trial encoder."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from ablate_trial_object_encoder import fit, score_embeddings
from fit_trial_object_encoder import DESIGN as ORIGINAL
from prepare_scale_trial import DESIGN, PUBLIC, metadata
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from safetensors.torch import load_file, save_file

CELLS = tuple((n, steps) for n in (416, 928) for steps in (750, 1500))


def rows(a, classes, old, extra, device):
    values = np.empty((len(classes), 10, 4, 63, 125), np.float32)
    for i, c in enumerate(classes):
        key = str(c)
        if key in old:
            folder, record = a.data / "trial-object-inputs/sub-04", old[key]
        else:
            folder, record = getattr(a, "extra_dir", a.out / "extra"), extra[key]
        path = folder / f"{c:04d}.npy"
        check(sha(path) == record["output_sha256"], f"Modified trial input for object {c}")
        v = np.load(path, allow_pickle=False)
        check(
            v.dtype == np.float32 and v.shape == values.shape[1:] and np.isfinite(v).all(),
            f"Bad trial crop for object {c}",
        )
        values[i] = v
    return torch.from_numpy(values).to(device)


def material(a, count):
    original, additions, native = metadata(a.data)
    old = read(a.data / "trial-object-inputs/sub-04/generic512.json")["blocks"]
    check(
        {k: v["output_sha256"] for k, v in old.items()} == original["sources"],
        "Wrong historical four-trial inputs",
    )
    initial, held = (np.array(original[k]) for k in ("train_classes", "test_classes"))
    ext = {}
    if count > 416:
        extra = read(getattr(a, "extra_dir", a.out / "extra") / "receipt.json")
        check(
            extra["design_sha256"] == sha(DESIGN)
            and extra["object_ids"] == additions
            and len(extra["blocks"]) == len(additions)
            and all(
                extra["blocks"][str(c)]["source_raw_sha256"] == native["raw_sha256"][str(c)]
                for c in additions
            ),
            "Incomplete or different additional EEG",
        )
        ext = extra["blocks"]
    if count == 416:
        ids = initial
    elif count == 928:
        ids = np.sort(np.r_[initial, additions])
    elif count == 1024:
        result = read(PUBLIC / "scale_trial.json")
        check(
            result["advance_to_personal12"]
            and result["design_sha256"] == sha(DESIGN)
            and read(a.out / "summary.json")["cases"] == result["cases"],
            "Generic gate did not pass",
        )
        ids = np.sort(np.r_[initial, held, additions])
    else:
        raise ValueError("Unexpected training count")
    check(
        len(np.unique(ids)) == count
        and np.all(np.diff(ids) > 0)
        and (count == 1024 or not np.intersect1d(ids, held).size),
        "Wrong fit/evaluation split",
    )
    return original, old, ext, ids, held


def cell(a, count, updates):
    original, old, ext, ids, held = material(a, count)
    folder = a.out / f"objects{count}_steps{updates}"
    folder.mkdir(parents=True, exist_ok=True)
    baseline = a.data / "trial-object-encoder/generic"
    spec = read(ORIGINAL) | {"updates": updates}
    check(sha(ORIGINAL) == original["design_sha256"], "Original training schedule changed")
    signature = dict(
        design_sha256=sha(DESIGN),
        driver_sha256=sha(Path(__file__)),
        training_objects=ids.tolist(),
        updates=updates,
        validation_objects=held.tolist() if count < 1024 else [],
    )
    receipt = folder / "receipt.json"
    if receipt.exists():
        r = read(receipt)
        check(
            all(r[k] == v for k, v in signature.items())
            and all(sha(folder / k) == v for k, v in r["files"].items()),
            "Changed cell",
        )
        print("verified", folder.name, flush=True)
        return
    device = torch.device(a.device)
    data = rows(a, ids, old, ext, device)
    sd, mu = torch.std_mean(data, dim=(0, 1, 2, 4), unbiased=False)
    mu, sd = mu[:, None], sd.clamp_min(1e-5)[:, None]
    data.sub_(mu).div_(sd).clamp_(-12, 12)
    schedule = np.array(
        read(a.data / "object-transfer/sessions.json")["subjects"]["sub-04"]["session_pairs"]
    )
    _, sessions = np.unique(schedule[ids], axis=0, return_inverse=True)
    model, trace = fit(data, sessions, True, 0.2, spec, device)
    del data
    weights = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
    weights.update(input_mean=mu.cpu(), input_scale=sd.cpu())
    save_file(weights, folder / "model.safetensors")
    files = {"model.safetensors": sha(folder / "model.safetensors")}
    replay = None
    if count < 1024:
        test = rows(a, held, old, ext, device)
        with torch.inference_mode():
            y = torch.cat(
                [
                    model(((block - mu) / sd).clamp(-12, 12))
                    for block in test.flatten(0, 1).split(128)
                ]
            )
        y = y.cpu().numpy().reshape(96, 10, 96)
        check(np.allclose(np.linalg.norm(y, axis=-1), 1, atol=1e-5), "Invalid encoder embeddings")
        np.savez(folder / "embeddings.npz", ids=held, features=y)
        files["embeddings.npz"] = sha(folder / "embeddings.npz")
        expected = read(baseline / "score.json")["predictions_sha256"]
        check(
            all(sha(baseline / (k + ".npz")) == v for k, v in expected.items()),
            "Original menus changed",
        )
        scores = score_embeddings(baseline, held, y)
        for case, vals in scores.items():
            name = case + ".npz"
            np.savez(folder / name, **vals)
            files[name] = sha(folder / name)
        if (count, updates) == (416, 750):
            check(
                sha(baseline / "model.safetensors") == original["weights_sha256"]
                and sha(baseline / "embeddings.npz") == original["packet_sha256"],
                "Original fit changed",
            )
            ref = load_file(baseline / "model.safetensors")
            with np.load(baseline / "embeddings.npz") as z:
                err = float(np.abs(z["features"] - y).max())
            state_error = max(float((weights[k] - ref[k]).abs().max()) for k in ref)
            check(err < 2e-5 and state_error < 2e-5, "Historical fit replay failed")
            for k, v in scores.items():
                with np.load(baseline / (k + ".npz")) as z:
                    check(
                        np.array_equal(z["mixed"], v["mixed"])
                        and np.array_equal(z["encoder"], v["deep"]),
                        "Historical individual decisions changed",
                    )
            replay = dict(
                max_embedding_abs_error=err,
                max_state_abs_error=state_error,
                matched_mixed_decisions=1920,
                matched_encoder_decisions=1920,
            )
    freeze(receipt, dict(**signature, files=files, trace=trace, replay=replay))
    print("finished", folder.name, replay, flush=True)


def contrast(values, draws):
    return dict(
        delta_pp=float(100 * values.mean()),
        ci95_pp=(100 * np.quantile(values[draws].mean(1), [0.025, 0.975])).tolist(),
    )


def summarize(a):
    metadata(a.data)
    folders = [a.out / f"objects{n}_steps{u}" for n, u in CELLS]
    reports = [read(folder / "receipt.json") for folder in folders]
    for folder, r in zip(folders, reports, strict=True):
        check(
            r["design_sha256"] == sha(DESIGN)
            and r["driver_sha256"] == sha(Path(__file__))
            and all(sha(folder / k) == v for k, v in r["files"].items()),
            "Cell changed",
        )
    check(len({v["trace"]["initial_state"] for v in reports}) == 1, "Initialization differed")
    result = dict(
        design_sha256=sha(DESIGN),
        driver_sha256=sha(Path(__file__)),
        extra_crop_receipt_sha256=sha(a.extra_dir / "receipt.json"),
        participant="sub-04",
        held_objects=96,
        distinct_photos=960,
        presentations_per_photo=4,
        cells={
            f"{n}_{u}": sha(folder / "receipt.json")
            for (n, u), folder in zip(CELLS, folders, strict=True)
        },
        baseline_replay=reports[0]["replay"],
        cases={},
    )
    draws = np.random.default_rng(260552).integers(96, size=(10000, 96))
    for k in (12, 4):
        data = []
        for folder in folders:
            with np.load(folder / f"generic_{k}.npz") as z:
                data.append(dict(z))
        truth = data[0]["truth"]
        classes, ns = np.unique(truth, return_counts=True)
        check(
            len(classes) == 96
            and (ns == 10).all()
            and all(
                np.array_equal(v["rows"], data[0]["rows"]) and np.array_equal(v["truth"], truth)
                for v in data
            ),
            "Different queries across cells",
        )
        case = {}
        for mode in ("mixed", "deep"):
            hits = np.stack([v[mode] == truth for v in data])
            rates = np.stack([hits[:, truth == c].mean(1) for c in classes], axis=1)

            def paired(i, j, hits=hits, rates=rates):
                return dict(
                    gains=int((hits[i] & ~hits[j]).sum()),
                    losses=int((~hits[i] & hits[j]).sum()),
                    net=int(hits[i].sum() - hits[j].sum()),
                    **contrast(rates[i] - rates[j], draws),
                )

            r00, r01, r10, r11 = rates
            case[mode] = dict(
                hits={f"{n}_{u}": int(h.sum()) for (n, u), h in zip(CELLS, hits, strict=True)},
                scaled_minus_historical=paired(3, 0),
                data_at_1500=paired(3, 1),
                data_at_750=paired(2, 0),
                updates_at_416=paired(1, 0),
                updates_at_928=paired(3, 2),
                average_data_effect=contrast((r10 - r00 + r11 - r01) / 2, draws),
                average_update_effect=contrast((r01 - r00 + r11 - r10) / 2, draws),
                interaction=contrast(r11 - r10 - r01 + r00, draws),
            )
        result["cases"][str(k)] = case
    primary, short = (result["cases"][k]["mixed"] for k in ("12", "4"))
    result["advance_to_personal12"] = bool(
        primary["scaled_minus_historical"]["net"] > 0
        and primary["data_at_1500"]["net"] > 0
        and primary["data_at_1500"]["ci95_pp"][0] > 0
        and short["scaled_minus_historical"]["net"] >= 0
        and short["data_at_1500"]["net"] >= 0
    )
    freeze(a.out / "summary.json", result)
    historical = read(PUBLIC / "scale_trial.json")
    check(
        all(result[k] == v for k, v in historical.items() if k not in ("cells", "driver_sha256")),
        "Replayed scientific results differ from the historical release",
    )
    print(primary, "advance", result["advance_to_personal12"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("old", "expanded", "score", "final"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/scale-trial/replay"))
    p.add_argument("--extra-dir", type=Path, default=Path("outputs/scale-trial/extra"))
    args = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.stage == "score":
        summarize(args)
    elif args.stage == "final":
        cell(args, 1024, 1500)
    else:
        for steps in (750, 1500):
            cell(args, 416 if args.stage == "old" else 928, steps)
