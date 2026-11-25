"""Export or run the sub-04 1,024-object decoder using one query and its labeled supports."""

import argparse
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import torch
from calibrate_photo_angles import rotate
from fit_double_calibration import training
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_within_photo_embedding import generative, images, unit
from infer_personal_object import CHANNEL_NAMES, KEYS, PersonalModel
from prepare_expanded_personal import extract
from prepare_scale_trial import PUBLIC
from prepare_trial_object_inputs import check
from register_fresh_mix import LOCK, SUB
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz


def export(a):
    score = read(PUBLIC / "fresh_mix.json")
    fit = a.root / "fit/final1024"
    r = read(fit / "receipt.json")
    mixture = read(a.root / "fit/mixture.json")
    check(
        sha(LOCK) == score["design_sha256"]
        and sha(fit / "receipt.json") == score["fit_sha256"]
        and sha(fit / "model.safetensors") == r["weights_sha256"]
        and sha(a.root / "fit/mixture.json") == score["weight_sha256"]
        and mixture["neural_weight"] == score["weight"],
        "Different fitted encoder or scalar",
    )
    parent = read(PUBLIC / "single_query_personal_parity.json")["participants"][SUB]
    check(
        sha(a.visual_bundle / "heads.npz") == parent["cached_heads_sha256"],
        "Different frozen visual readout",
    )
    with np.load(a.visual_bundle / "heads.npz", allow_pickle=False) as z:
        head = dict(z)
    _, x = training(
        SimpleNamespace(
            expanded=a.data / "photo-structure/expanded", out=a.data / "double-calibration"
        ),
        SUB,
    )
    # Evaluating zero and each unit vector gives the exact fitted affine map.
    basis = np.vstack([np.zeros((1, 1003)), np.eye(1003)])
    fx = fisher(x.reshape(1024, 10, 1003), basis)
    head.update(fisher_weight=fx[1:] - fx[0], fisher_bias=fx[0], neural_weight=score["weight"])
    model = a.model_dir
    model.mkdir(parents=True, exist_ok=True)
    frozen_npz(model / "heads.npz", head)
    shutil.copyfile(fit / "model.safetensors", model / "model.safetensors")
    freeze(
        model / "receipt.json",
        dict(
            subject=SUB,
            heads_sha256=sha(model / "heads.npz"),
            weights_sha256=r["weights_sha256"],
            design_sha256=sha(LOCK),
            fit_sha256=score["fit_sha256"],
            mixture_sha256=score["weight_sha256"],
            legacy_visual_bundle_sha256=parent["cached_heads_sha256"],
        ),
    )


class FreshModel(PersonalModel):
    def __init__(self, folder, device):
        super().__init__(folder, device)
        # The prospective CUDA fit used cuDNN's TF32 convolution and FP32 matmul.
        if device == "cuda":
            torch.backends.cudnn.allow_tf32 = True
        check(
            self.receipt["design_sha256"] == sha(LOCK)
            and str(self.head["subject"]) == SUB
            and self.head["fisher_weight"].shape == (1003, 96)
            and self.head["visual_weight"].shape == (1003, 1024)
            and 0 <= float(self.head["neural_weight"]) <= 1,
            "Wrong portable decoder",
        )

    def infer(self, d):
        check(set(d) == KEYS and str(d["subject"]) == SUB, "Expected one label-blind sub-04 query")
        names = d["channels"].tolist()
        ids, query, words = d["support_ids"], d["query_id"], d["candidate_words"]
        check(
            len(names) == 63
            and set(names) == set(CHANNEL_NAMES)
            and d["times"].shape == (250,)
            and np.allclose(d["times"], np.arange(250) / 250, atol=1e-7, rtol=0),
            "Invalid montage/times",
        )
        check(
            ids.shape == (12, 8)
            and ids.dtype.kind in "iuU"
            and len(set(ids.astype(str).flat)) == 96
            and query.ndim == 0
            and query.dtype.kind in "iuU"
            and str(query) not in ids.astype(str)
            and words.shape == (12,)
            and words.dtype.kind == "U"
            and len(set(words)) == 12
            and all(str(w).strip() for w in words),
            "Repeated or invalid word/query/support identity",
        )
        q, s, y = d["query_eeg"], d["support_eeg"], d["support_h14"]
        check(
            q.shape == (4, 63, 250)
            and s.shape == (12, 8, 4, 63, 250)
            and y.shape == (12, 8, 1024)
            and all(v.dtype.kind == "f" and np.isfinite(v).all() for v in (q, s, y))
            and (np.linalg.norm(y, axis=-1) > 1e-8).all(),
            "Invalid four-trial input or H14 supports",
        )
        waves = np.concatenate((q[None], s.reshape(96, 4, 63, 250))).astype(np.float32)
        waves = np.take(waves, [names.index(c) for c in CHANNEL_NAMES], axis=2)
        # Match design_block's contiguous stored-order mean for this prospective protocol.
        x = extract(np.take(waves, self.pick, axis=2).mean(1, dtype=np.float32)).astype(float)
        fx = x @ self.head["fisher_weight"] + self.head["fisher_bias"]
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(waves[:, :, :, :125])).to(self.device)
            deep = self.model(((t - self.mu) / self.sd).clamp(-12, 12)).cpu().numpy().astype(float)
        fs, ds = (similarities(z[:1], z[1:].reshape(12, 8, 96))[0] for z in (fx, deep))
        w = self.head["neural_weight"]
        mixed = (1 - w) * fs + w * ds
        winner = int(mixed.argmax())
        # Only the selected class needs the visual affine map.
        sx = x[1:].reshape(12, 8, 1003)[winner]
        hv = np.vstack((x[:1], sx)) @ self.head["visual_weight"] + self.head["visual_bias"]
        mean = unit(y[winner].astype(float)).mean(0, keepdims=True)
        check(np.linalg.norm(mean) > 1e-8, "Selected support vectors cancel")
        prior = unit(mean).astype(np.float32)
        native = generative(mean, hv[:1] - hv[1:].mean(0)).astype(np.float32)
        own = rotate(prior, native, self.head)[0]
        check(
            abs(np.linalg.norm(own) - 1) < 1e-5 and np.isfinite(own).all(),
            "Invalid native direction",
        )
        return dict(
            subject=d["subject"],
            query_id=query,
            word=words[winner],
            candidate_words=words,
            mixed_scores=mixed,
            fisher_scores=fs,
            trial_scores=ds,
            support_ids=ids[winner],
            own=own,
            uniform=prior[0],
            radius=self.head["radius"],
        )


def replay(a):
    model = FreshModel(a.model_dir, a.device)
    design = read(LOCK)
    receipt = read(a.root / "eeg/sub-04.json")
    visual = images(a.data / "official-object-readout/sub-03")
    counts = {}
    maxes = {"own": 0.0, "uniform": 0.0, "mixed_scores": 0.0}
    for name, menu in design["panels"].items():
        reg = read(a.root / "references" / (name + "_ablation.json"))
        waves = {}
        for c in menu["object_ids"]:
            path = a.root / "eeg/raw" / f"{c:04d}.bin"
            check(sha(path) == receipt["raw_sha256"][str(c)], "Full waveform differs")
            x = np.fromfile(path, "<f8").reshape(10, 4, 63, 250).astype(np.float32)
            waves.update({10 * c + i: v for i, v in enumerate(x)})
        packets = a.root / "render_inputs/learned"
        ref = packets / "conditions" / SUB / name / "conditions_512.npz"
        check(
            sha(ref) == read(packets / "design.json")["packets_sha256"][f"{SUB}/{name}"],
            "Condition packet differs",
        )
        checks = 0
        hits = 0
        with (
            np.load(ref, allow_pickle=False) as z,
            np.load(a.root / "decisions" / (name + ".npz")) as scores,
        ):
            w = float(model.head["neural_weight"])
            reference = (1 - w) * scores["scores"][0, 2] + w * scores["scores"][0, 3]
            for f in range(5):
                if a.fold is not None and f != a.fold:
                    continue
                ids = np.asarray(reg["calibration_rows_by_fold_and_word"][f])
                payload = dict(
                    subject=np.array(SUB),
                    channels=np.array(CHANNEL_NAMES),
                    times=np.arange(250) / 250,
                    support_ids=ids,
                    support_eeg=np.stack([waves[int(k)] for k in ids.ravel()]).reshape(
                        12, 8, 4, 63, 250
                    ),
                    candidate_words=np.array(reg["words"]),
                    support_h14=np.asarray(visual[ids]),
                )
                for i in np.flatnonzero(np.array(reg["folds"]) == f):
                    row = int(reg["rows"][i])
                    payload.update(query_id=np.array(row), query_eeg=waves[row])
                    if not counts and checks == 0 and a.example:
                        a.example.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(a.example, **payload)
                    got = model.infer(payload)
                    check(
                        np.array_equal(got["support_ids"], z["support_rows"][i]),
                        "Different support class",
                    )
                    hits += int(str(got["word"]) == z["predicted_words"][i])
                    checks += 1
                    for key in ("uniform", "own", "mixed_scores"):
                        target = reference[i] if key == "mixed_scores" else z[key][i]
                        maxes[key] = max(maxes[key], float(np.abs(got[key] - target).max()))
        check(checks == hits == (120 if a.fold is None else 24), "Different single-query word")
        counts[name] = dict(queries=checks, word_matches=hits, packet_sha256=sha(ref))
    check(
        max(maxes["own"], maxes["uniform"]) < 2e-6
        and (a.device == "cpu" or maxes["mixed_scores"] < 2e-5),
        f"Numerical parity failed: {maxes}",
    )
    summary = dict(
        design_sha256=sha(LOCK),
        driver_sha256=sha(Path(__file__)),
        heads_sha256=model.receipt["heads_sha256"],
        weights_sha256=model.receipt["weights_sha256"],
        subject=SUB,
        device=a.device,
        fold=a.fold,
        queries=sum(v["queries"] for v in counts.values()),
        menus=counts,
        max_abs_error=maxes,
    )
    dest = a.model_dir / ("parity.json" if a.fold is None else f"parity_fold{a.fold}.json")
    freeze(dest, summary)
    if a.publish:
        check(a.fold is None, "Public parity requires all 480 queries")
        freeze(PUBLIC / "fresh_mix_single_query.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("export", "predict", "replay"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--root", type=Path, default=Path("outputs/fresh-mix"))
    p.add_argument(
        "--visual-bundle", type=Path, default=Path("outputs/personal-object-inference/sub-04")
    )
    p.add_argument("--model-dir", type=Path, default=Path("outputs/fresh-mix/portable"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--input", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--fold", type=int, choices=range(5))
    p.add_argument("--example", type=Path)
    p.add_argument("--publish", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if a.stage == "export":
        export(a)
    elif a.stage == "replay":
        replay(a)
    else:
        if not a.input or not a.out:
            p.error("predict requires --input and --out")
        model = FreshModel(a.model_dir, a.device)
        with np.load(a.input, allow_pickle=False) as z:
            got = model.infer(z)
        frozen_npz(
            a.out,
            got
            | dict(
                input_sha256=sha(a.input),
                weights_sha256=model.receipt["weights_sha256"],
                heads_sha256=model.receipt["heads_sha256"],
            ),
        )
        print(str(got["word"]), flush=True)
