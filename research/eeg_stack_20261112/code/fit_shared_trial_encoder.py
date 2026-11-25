"""Generic-only cross-person photo supervision. Run from the repository root with the original EEG caches."""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path("research/eeg_stack_20261112/code").resolve()))
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha

CODE = Path("research/eeg_stack_20261112/code")
PUBLIC = CODE.parent / "results/object_fidelity"
SUBJECTS = ("sub-04", "sub-05", "sub-01")
ARMS = ("individual", "shared_only", "cross_sham", "cross_matched")
LOCK = "ca64956a955ed185db3fee39de104932135815e8da7381fdfa0b27f8fd44ae26"


def inputs(a):
    design = a.root / "design.json"
    check(sha(design) == LOCK, "Unexpected registration")
    d = read(design)
    o = a.data
    refs = dict(
        original_fit=CODE / "fit_trial_object_encoder.py",
        original_protocol=PUBLIC / "trial_object_encoder_design.json",
        menu_driver=CODE / "fit_hard_object_pairs.py",
        baseline_driver=CODE / "fit_episodic_object.py",
        native_score=CODE / "score_trial_object_encoder.py",
        archived_04=o / "trial-object-encoder/generic/embeddings.npz",
        archived_04_score=o / "trial-object-encoder/generic/score.json",
        schedules=o / "object-transfer/sessions.json",
    )
    check({k: sha(v) for k, v in refs.items()} == d["inputs"], "Changed prior protocol")
    result = []
    schedules = read(refs["schedules"])["subjects"]
    for s in SUBJECTS:
        trials = o / (
            f"trial-object-inputs/{s}" if s == "sub-04" else f"trial-object-replication/{s}/trials"
        )
        files = dict(
            trials=trials / ("generic512.json" if s == "sub-04" else "receipt.json"),
            split=o / "episodic-object" / s / "model.npz",
            expanded=o / "photo-structure/expanded/features" / (s + ".json"),
            previous=o / "episodic-object" / s / "receipt.json",
        )
        check(
            {k: sha(v) for k, v in files.items()} == d["per_subject"][s],
            "Participant sources changed",
        )
        with np.load(files["split"], allow_pickle=False) as z:
            ids, held = z["train_ids"], z["generic_heldout_ids"]
        train = ids[~np.isin(ids, held)]
        check(
            len(ids) == 512
            and len(held) == 96
            and len(train) == 416
            and not set(train) & set(held)
            and (
                not result
                or np.array_equal(train, result[0].train)
                and np.array_equal(held, result[0].held)
            ),
            "Different train/test identities",
        )
        session = np.array(schedules[s]["session_pairs"])
        _, si = np.unique(session[ids], axis=0, return_inverse=True)
        result.append(
            SimpleNamespace(
                subject=s,
                folder=trials,
                blocks=read(files["trials"])["blocks"],
                train=train,
                held=held,
                session=si[~np.isin(ids, held)],
                eval_session=si[np.isin(ids, held)],
                previous=read(files["previous"]),
            )
        )
    return d, result


def torch_setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch


def shared_model(device):
    from fit_trial_object_encoder import TrialObjectEncoder
    from torch import nn

    m = nn.ModuleList([TrialObjectEncoder().to(device) for _ in SUBJECTS])
    for net in list(m)[1:]:
        net.temporal = m[0].temporal
        net.log_precision = m[0].log_precision
        net.head = m[0].head
    check(
        len({id(v.spatial.weight) for v in m}) == 3 and len({id(v.log_precision) for v in m}) == 1,
        "Wrong sharing graph",
    )
    return m


def standardized(info, device):
    from fit_trial_object_encoder import source_data

    torch = torch_setup()
    raw = source_data(info.folder, info.train, info.blocks, device)
    sd, mu = torch.std_mean(raw, dim=(0, 1, 2, 4), unbiased=False)
    mu = mu[:, None]
    sd = sd.clamp_min(1e-5)[:, None]
    return raw.sub_(mu).div_(sd).clamp_(-12, 12), mu, sd


def shared_fit(armspec, data, info, seed, device):
    from fit_trial_object_encoder import episode_loss
    from torch.nn import functional as F

    torch = torch_setup()
    torch.manual_seed(seed)
    nets = shared_model(device)
    opt = torch.optim.AdamW(nets.parameters(), lr=0.0006, weight_decay=0.02)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=750)
    streams = [np.random.default_rng(seed) for _ in info]
    sham_stream = np.random.default_rng(seed + 1)
    cells = [[np.flatnonzero(e.session == j) for j in np.unique(e.session)] for e in info]
    logs = []
    target = torch.arange(10, device=device)[None].expand(12, 10)

    def within(x):
        x = x.reshape(12, 10, 96)
        return F.normalize(x - x.mean(1, keepdim=True), dim=2)

    for step in range(750):
        opt.zero_grad(set_to_none=True)
        val = 0
        for i, net in enumerate(nets):
            which = streams[i].choice(cells[i][step % len(cells[i])], 12, replace=False)
            order = np.stack([streams[i].permutation(10) for _ in which])
            objects = torch.as_tensor(which, device=device)[:, None]
            photos = torch.as_tensor(order, device=device)
            x = data[i][objects, photos]
            splits = torch.rand(120, 4, device=device).argsort(dim=1)
            nets.train()
            loss = episode_loss(net, x, splits, full_query=step % 2 == 1)
            nets.eval()
            anchor = within(net(x.flatten(0, 1)))
            cross = 0
            for j, peer in enumerate(nets):
                if i == j:
                    continue
                with torch.no_grad():
                    other = within(peer(data[j][objects, photos].flatten(0, 1)))
                shifts = torch.as_tensor(sham_stream.integers(1, 10, size=12), device=device)[
                    :, None
                ]
                label = (target + shifts) % 10 if armspec == "cross_sham" else target
                logits = 10 * torch.bmm(anchor, other.transpose(1, 2))
                cross += F.cross_entropy(logits.reshape(-1, 10), label.reshape(-1)) / 2
            loss = (loss + (0.0 if armspec == "shared_only" else 0.2) * cross) / 3
            loss.backward()
            val += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(nets.parameters(), 5)
        opt.step()
        lr.step()
        logs.append(val)
        if (step + 1) % 150 == 0:
            print(armspec, step + 1, float(np.mean(logs[-150:])), flush=True)
    check(np.isfinite(logs).all(), "Nonfinite optimization")
    return nets.eval()


def save_model(path, net, mean, scale):
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
    state.update(input_mean=mean.cpu(), input_scale=scale.cpu())
    save_file(state, path)


def train(a, design, info):
    from fit_trial_object_encoder import fit

    torch = torch_setup()
    device = torch.device(a.device)
    sets = [standardized(e, device) for e in info]
    data, means, scales = map(list, zip(*sets, strict=True))
    files = {}
    for arm in ARMS:
        dest = a.root / arm
        dest.mkdir(parents=True, exist_ok=True)
        rec = dest / "fit.json"
        if rec.exists():
            r = read(rec)
            check(
                r["design_sha256"] == LOCK
                and r["driver_sha256"] == sha(Path(__file__))
                and r["weights"] == {s: sha(dest / (s + ".safetensors")) for s in SUBJECTS},
                "Stale fit",
            )
            files[arm] = sha(rec)
            continue
        if arm == "individual":
            nets = [fit(data[i], e.session, design, device)[0] for i, e in enumerate(info)]
        else:
            nets = shared_fit(arm, data, info, design["seed"], device)
        for i, s in enumerate(SUBJECTS):
            save_model(dest / (s + ".safetensors"), nets[i], means[i], scales[i])
        freeze(
            rec,
            dict(
                design_sha256=LOCK,
                driver_sha256=sha(Path(__file__)),
                arm=arm,
                updates=750,
                torch=str(torch.__version__),
                training_objects=416,
                seed=design["seed"],
                weights={s: sha(dest / (s + ".safetensors")) for s in SUBJECTS},
            ),
        )
        files[arm] = sha(rec)
    freeze(a.root / "fits.json", dict(design_sha256=LOCK, models=files))


def evaluate(a, d, info):
    from fit_episodic_object import SEED, apply_projection, projection
    from fit_hard_object_pairs import episodes
    from fit_personal_session_metric import similarities
    from fit_shared_photo_eeg import load_photos
    from fit_trial_object_encoder import TrialObjectEncoder, source_data
    from run_prospective_readout import frozen_npz
    from safetensors.torch import load_file

    torch = torch_setup()
    device = torch.device(a.device)
    fits = read(a.root / "fits.json")
    check(
        fits["design_sha256"] == LOCK
        and all(fits["models"][arm] == sha(a.root / arm / "fit.json") for arm in ARMS),
        "Incomplete model freeze",
    )
    data = {}
    checks = {}
    for i, e in enumerate(info):
        held = e.held
        target = source_data(e.folder, held, e.blocks, device).flatten(0, 1)
        features = {}
        for arm in ARMS:
            model = TrialObjectEncoder().to(device)
            file = a.root / arm / (e.subject + ".safetensors")
            check(
                sha(file) == read(a.root / arm / "fit.json")["weights"][e.subject], "Changed model"
            )
            w = load_file(file, device=str(device))
            mu, sd = w.pop("input_mean"), w.pop("input_scale")
            model.load_state_dict(w, strict=True)
            model.eval()
            with torch.inference_mode():
                f = torch.cat([model(((v - mu) / sd).clamp(-12, 12)) for v in target.split(128)])
            features[arm] = f.cpu().numpy().astype(float)
        if i == 0:
            with np.load(a.data / "trial-object-encoder/generic/embeddings.npz") as z:
                check(
                    np.array_equal(held, z["ids"])
                    and np.array_equal(features["individual"], z["features"].reshape(-1, 96)),
                    "Archived sub-04 neural embeddings failed exact replay",
                )
        meta = SimpleNamespace(expanded=a.data / "photo-structure/expanded")
        tr = load_photos(meta, e.subject, e.train)
        ev = load_photos(meta, e.subject, held)
        fish = apply_projection(ev, projection(tr, e.session))[..., :96].reshape(-1, 96)
        raw = (held[:, None] * 10 + np.arange(10)).ravel()
        for k in (4, 12):
            pred = {name: [] for name in (*ARMS, "fisher")}
            truth = []
            rows = []
            for _, _, _, label, q, _, sup in episodes(
                raw.reshape(-1, 10), held, e.eval_session, k, SEED + 1000 + k
            ):
                qi = np.searchsorted(raw, q)
                si = np.searchsorted(raw, sup)
                check(
                    not np.isin(q, sup).any()
                    and len(np.unique(e.eval_session[si[:, 0] // 10])) == 1,
                    "Unsafe menu",
                )
                native = similarities(fish[qi], fish[si])
                pred["fisher"].extend(native.argmax(1))
                for arm, v in features.items():
                    pred[arm].extend((native + similarities(v[qi], v[si])).argmax(1))
                rows.extend(q)
                truth.extend([label, label])
            hist = a.data / "episodic-object" / e.subject / f"generic_{k}.npz"
            check(
                sha(hist) == e.previous["prediction_sha256"][f"generic_{k}"],
                "Changed historical scores",
            )
            with np.load(hist) as old:
                check(
                    np.array_equal(truth, old["truth"])
                    and np.array_equal(np.array(rows) // 10, old["concept"])
                    and np.array_equal(pred["fisher"], old["native"]),
                    "Historical Fisher/menu mismatch",
                )
            if i == 0:
                oldfile = a.data / "trial-object-encoder/generic" / f"generic_{k}.npz"
                chk = read(a.data / "trial-object-encoder/generic/score.json")[
                    "predictions_sha256"
                ][f"generic_{k}"]
                check(sha(oldfile) == chk, "Different sub-04 original generic decisions")
                with np.load(oldfile) as old:
                    check(
                        np.array_equal(old["rows"], rows)
                        and np.array_equal(old["mixed_slot"], pred["individual"]),
                        "Old sub-04 mixed decisions changed",
                    )
            packed = dict(rows=np.asarray(rows), truth=np.asarray(truth)) | {
                arm: np.asarray(v) for arm, v in pred.items()
            }
            path = a.root / "decisions" / e.subject / f"generic_{k}.npz"
            frozen_npz(path, packed)
            checks[f"{e.subject}/{k}"] = sha(path)
            order = np.argsort(rows)
            check(np.array_equal(np.array(rows)[order], raw), "Missing rows")
            data[e.subject, k] = np.array(
                [
                    (v == packed["truth"])[order].reshape(96, 10).sum(1)
                    for v in (packed[n] for n in ARMS + ("fisher",))
                ]
            )

    def metrics(k):
        result = {
            s: dict(zip(ARMS + ("fisher",), data[s, k].sum(1).tolist(), strict=True))
            for s in SUBJECTS
        }
        summed = sum(data[s, k] for s in SUBJECTS)
        draws = np.random.default_rng(d["bootstrap_seed"]).integers(
            96, size=(d["bootstrap_draws"], 96)
        )
        comparison = {}
        for j, other in enumerate(ARMS[:3]):
            diff = summed[3] - summed[j]
            comparison[other] = dict(
                hit_difference=int(diff.sum()),
                delta=float(diff.sum() / 2880),
                ci95=np.quantile(diff[draws].sum(1) / 2880, [0.025, 0.975]).tolist(),
            )
        return dict(
            n=2880,
            per_person=result,
            hits=dict(zip(ARMS + ("fisher",), summed.sum(1).tolist(), strict=True)),
            matched_minus=comparison,
        )

    m12, m4 = metrics(12), metrics(4)
    gate = all(
        m12["matched_minus"][n]["ci95"][0] > 0 and m4["matched_minus"][n]["hit_difference"] >= 0
        for n in ("individual", "cross_sham")
    )
    out = dict(
        protocol_sha256=LOCK,
        driver_sha256=sha(Path(__file__)),
        fit_receipts_sha256=fits["models"],
        held_out_objects=96,
        distinct_photos=960,
        participant_photo_queries_per_menu=2880,
        individual_sub04_exact_embedding_and_predictions_replay=True,
        decision_sha256=checks,
        twelve_way=m12,
        four_way=m4,
        advance=gate,
    )
    freeze(a.root / "result.json", out)
    print("12-way", m12, "4-way", m4, "advance", gate, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["train", "evaluate"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--root", type=Path, default=Path("outputs/shared-trial-encoder"))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    d, info = inputs(a)
    (train if a.stage == "train" else evaluate)(a, d, info)
