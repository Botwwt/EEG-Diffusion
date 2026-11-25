"""Predict one calibrated participant's word and native H/14 condition from four EEG trials."""

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from calibrate_photo_angles import rotate
from decode_eeg_layout import CHANNELS
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_spatial_photo import save
from fit_within_photo_embedding import bases, generative, images, predict, training, unit
from prepare_expanded_personal import extract
from prepare_registered_subjects import OFFSET, REV, ROW_BYTES, SHAPE
from render_within_photo_ip import check, read, sha
from run_prospective_readout import frozen_npz

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
CHANNEL_NAMES = (
    "Fp1 Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 F2 F4 F6 F8 FT9 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 FT10 T7 C5 C3 C1 Cz C2 C4 C6 T8 TP9 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 TP10 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2"
).split()
KEYS = {
    "subject",
    "query_id",
    "query_eeg",
    "support_ids",
    "support_eeg",
    "support_h14",
    "candidate_words",
    "channels",
    "times",
}
SELECTION = "914c9a527049783f23e25d1bdd2c3137e5fdd4ed69d18b1ce617853f9d53e79a"
ANGLE = "06e8f4d5c34554dd0e076347c519aa24a2b0f5bd89d785cce9632b0721dcd60d"


def prepare(a):
    first = read(PUBLIC / "trial_object_personal12_design.json")["inputs"]
    repeated = read(PUBLIC / "trial_object_personal12_replication_design.json")
    if a.subject == "sub-04":
        fit = a.data / "trial-object-encoder/target"
        check(sha(fit / "receipt.json") == first["fit_receipt"], "Different frozen trial fit")
        classes_key = "train_classes"
    else:
        fit = a.data / "trial-object-replication" / a.subject / "fit"
        check(
            read(fit / "receipt.json")["replication_design_sha256"]
            == sha(PUBLIC / "trial_object_personal12_replication_design.json"),
            "Different trial replication",
        )
        classes_key = "train_ids"
    fit_meta = read(fit / "receipt.json")
    selection = a.data / "photo-structure/within_photo/selection.json"
    angular = a.data / "photo-structure/residual_angle/scale.json"
    radius = a.data / "object-head-confirmation/render_inputs/radius.json"
    checkpoint = fit / "model.safetensors"
    check(
        sha(selection) == SELECTION
        and sha(angular) == ANGLE
        and sha(radius) == repeated["inputs"]["radius"]
        and fit_meta["subject"] == a.subject
        and fit_meta["weights_sha256"] == sha(checkpoint),
        "Wrong angle calibration, radius, or participant checkpoint",
    )
    expanded = a.data / "photo-structure/expanded"
    classes, x, y = training(
        SimpleNamespace(expanded=expanded),
        a.subject,
        512,
        images(a.data / "official-object-readout/sub-03"),
    )
    check(
        set(classes) == set(fit_meta[classes_key]),
        "Visual and trial heads use different training objects",
    )
    # Evaluating the original affine transforms on zero and the coordinate basis preserves
    # both their learned slopes and intercepts without shipping the 5,120 training examples.
    basis = np.vstack([np.zeros((1, x.shape[-1])), np.eye(x.shape[-1])])
    fx = fisher(x, basis)
    components = bases(x, y, basis)
    h = predict(components, read(selection)["selected"]) @ components[2].T
    scale = read(angular)
    a.model_dir.mkdir(parents=True, exist_ok=True)
    cache = a.model_dir / "heads.npz"
    frozen_npz(
        cache,
        dict(
            subject=a.subject,
            channels=np.asarray(CHANNEL_NAMES),
            posterior=np.asarray(CHANNELS),
            fisher_weight=fx[1:] - fx[0],
            fisher_bias=fx[0],
            visual_weight=h[1:] - h[0],
            visual_bias=h[0],
            gain=scale["gain"],
            max_angle=scale["max_angle"],
            radius=read(radius)["median_norm"],
        ),
    )
    shutil.copyfile(checkpoint, a.model_dir / "model.safetensors")
    save(
        a.model_dir / "receipt.json",
        dict(
            subject=a.subject,
            heads_sha256=sha(cache),
            weights_sha256=sha(checkpoint),
            select_sha256=SELECTION,
            angle_sha256=ANGLE,
            radius_sha256=sha(radius),
            fit_receipt_sha256=sha(fit / "receipt.json"),
            training_receipt_sha256=sha(expanded / "features" / (a.subject + ".json")),
        ),
    )


class PersonalModel:
    def __init__(self, folder, device):
        import torch
        from fit_trial_object_encoder import TrialObjectEncoder
        from safetensors.torch import load_file

        self.receipt = read(folder / "receipt.json")
        head = folder / "heads.npz"
        deep = folder / "model.safetensors"
        check(
            sha(head) == self.receipt["heads_sha256"]
            and sha(deep) == self.receipt["weights_sha256"],
            "Model bundle differs from its receipts",
        )
        with np.load(head, allow_pickle=False) as d:
            self.head = {k: d[k] for k in d.files}
        check(
            str(self.head["subject"]) == self.receipt["subject"]
            and self.head["channels"].tolist() == CHANNEL_NAMES
            and self.head["posterior"].tolist() == CHANNELS,
            "Invalid model subject or EEG axes",
        )
        self.pick = [CHANNEL_NAMES.index(c) for c in CHANNELS]
        weights = load_file(deep, device=device)
        self.mu = weights.pop("input_mean")
        self.sd = weights.pop("input_scale")
        self.model = TrialObjectEncoder().to(device)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval().requires_grad_(False)
        self.device = device
        if device == "cpu":
            torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def features(self, waves):
        import torch

        # Two actual early repetitions and two late repetitions in the public stored order.
        post = waves[:, :, self.pick, :]
        mean = (post[:, [1, 3]].mean(1) + post[:, [0, 2]].mean(1)) * 0.5
        x = extract(mean).astype(float)
        fx = x @ self.head["fisher_weight"] + self.head["fisher_bias"]
        h = x @ self.head["visual_weight"] + self.head["visual_bias"]
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(waves[:, :, :, :125])).to(self.device)
            deep = self.model(((t - self.mu) / self.sd).clamp(-12, 12)).cpu().numpy().astype(float)
        return fx, deep, h

    def infer(self, data):
        check(
            set(data) == KEYS and str(data["subject"]) == self.receipt["subject"],
            "Expected the documented label-blind single-query schema and model participant",
        )
        ch = data["channels"].tolist()
        check(
            len(ch) == 63
            and set(ch) == set(CHANNEL_NAMES)
            and data["times"].shape == (250,)
            and np.allclose(data["times"], np.arange(250) / 250, atol=1e-7, rtol=0),
            "Wrong montage or sample times",
        )
        ids = data["support_ids"]
        query = data["query_id"]
        vocab = data["candidate_words"]
        check(
            ids.shape == (12, 8)
            and ids.dtype.kind in "iuU"
            and query.ndim == 0
            and query.dtype.kind in "iuU"
            and len(set(ids.astype(str).flat)) == 96
            and str(query) not in ids.astype(str)
            and vocab.shape == (12,)
            and vocab.dtype.kind == "U"
            and len(set(vocab)) == 12
            and all(str(w).strip() for w in vocab),
            "Duplicate query/support IDs or candidate words",
        )
        q, s, y = (data[k] for k in ("query_eeg", "support_eeg", "support_h14"))
        check(
            q.shape == (4, 63, 250)
            and s.shape == (12, 8, 4, 63, 250)
            and y.shape == (12, 8, 1024)
            and all(v.dtype.kind == "f" and np.isfinite(v).all() for v in (q, s, y))
            and np.all(np.linalg.norm(y, axis=2) > 1e-8),
            "Malformed physical trials or support features",
        )
        order = [ch.index(k) for k in CHANNEL_NAMES]
        waves = np.concatenate([q[None], s.reshape(96, 4, 63, 250)]).astype(np.float32)[
            :, :, order, :
        ]
        fx, deep, h = self.features(waves)
        scores = np.stack([similarities(v[:1], v[1:].reshape(12, 8, 96))[0] for v in (fx, deep)])
        mixed = scores.mean(0)
        winner = int(mixed.argmax())
        mean = unit(y[winner].astype(float)).mean(0, keepdims=True)
        check(np.linalg.norm(mean) > 1e-8, "Selected support directions cancel")
        prior = unit(mean).astype(np.float32)
        native = generative(mean, h[:1] - h[1:].reshape(12, 8, 1024)[winner].mean(0)).astype(
            np.float32
        )
        own = rotate(prior, native, self.head)[0]
        check(
            np.isfinite(own).all() and abs(np.linalg.norm(own) - 1) < 1e-5,
            "Invalid native H/14 direction",
        )
        return dict(
            subject=data["subject"],
            query_id=query,
            word=vocab[winner],
            candidate_words=vocab,
            mixed_scores=mixed,
            fisher_scores=scores[0],
            trial_scores=scores[1],
            support_ids=ids[winner],
            own=own,
            uniform=prior[0],
            radius=self.head["radius"],
        )


def replay(a):
    root = a.data / "object-head-confirmation"
    subject = a.subject
    reg = read(root / "references" / f"personal12_{subject}_ablation.json")
    rec = read(root / "eeg" / (subject + ".json"))
    table = {}
    for block in rec["blocks"]:
        c = int(block["concept_id"])
        lo = OFFSET + c * 10 * ROW_BYTES
        f = root / "ranges" / REV / subject / f"{lo}-{lo + 10 * ROW_BYTES - 1}.bin"
        check(sha(f) == block["sha256"], "Changed stimulus EEG")
        raw = np.fromfile(f, "<f8").reshape(10, *SHAPE[1:]).astype(np.float32)
        table.update({c * 10 + j: w for j, w in enumerate(raw)})
    full = a.data / (
        "trial-object-encoder/personal12"
        if subject == "sub-04"
        else f"trial-object-replication/{subject}"
    )
    target = full / "render_inputs/conditions" / subject / "personal12/conditions_512.npz"
    bank = images(a.data / "official-object-readout/sub-03")
    examined = 0
    correct = 0
    max_error = dict(own=0.0, uniform=0.0)
    model = PersonalModel(a.model_dir, a.device)
    with np.load(target, allow_pickle=False) as old:
        check(
            sha(target) == read(full / "receipt.json")["mixed_packet_sha256"],
            "Historical packet differs",
        )
        for i, (row, fold) in enumerate(zip(reg["rows"], reg["folds"], strict=True)):
            if a.fold is not None and fold != a.fold:
                continue
            ids = np.array(reg["calibration_rows_by_fold_and_word"][fold])
            payload = dict(
                subject=np.array(subject),
                query_id=np.asarray(row),
                channels=np.asarray(CHANNEL_NAMES),
                times=np.arange(250) / 250,
                query_eeg=table[row],
                candidate_words=np.asarray(reg["words"]),
                support_ids=ids,
                support_h14=np.asarray(bank[ids]),
                support_eeg=np.stack([table[r] for r in ids.ravel()]).reshape(12, 8, 4, 63, 250),
            )
            if not examined and a.save_example:
                a.save_example.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(a.save_example, **payload)
            result = model.infer(payload)
            correct += int(old["predicted_words"][i] == result["word"])
            check(
                np.array_equal(old["support_rows"][i], result["support_ids"]),
                "Selected calibration bank changed",
            )
            for name in max_error:
                max_error[name] = max(
                    max_error[name], float(abs(result[name] - old[name][i]).max())
                )
            examined += 1
    report = dict(
        subject=subject,
        rows=examined,
        fold=a.fold,
        word_matches=correct,
        max_abs_error=max_error,
        model_sha256=model.receipt["weights_sha256"],
        cached_heads_sha256=model.receipt["heads_sha256"],
        comparison_packet_sha256=sha(target),
        single_query_driver_sha256=sha(Path(__file__)),
    )
    check(
        examined == (120 if a.fold is None else 24)
        and correct == examined
        and max(max_error.values()) < 2e-6,
        f"Single-query parity failed: {report}",
    )
    save(
        a.model_dir / ("replay_all.json" if a.fold is None else f"replay_fold{a.fold}.json"), report
    )
    print(report, flush=True)


def audit(a):
    cases = {}
    for subject in ("sub-04", "sub-05", "sub-01"):
        folder = a.model_dir / subject
        r = read(folder / "replay_all.json")
        check(
            r["subject"] == subject
            and r["rows"] == r["word_matches"] == 120
            and r["fold"] is None
            and max(r["max_abs_error"].values()) < 2e-6
            and r["single_query_driver_sha256"] == sha(Path(__file__))
            and r["model_sha256"] == sha(folder / "model.safetensors")
            and r["cached_heads_sha256"] == sha(folder / "heads.npz"),
            "Reproduce each participant's 120 independent queries before pooling the parity check",
        )
        cases[subject] = r
    save(
        a.out or a.model_dir / "replay_summary.json",
        dict(
            task="Agreement of single-query decisions and native H/14 conditions with each participant's frozen mixed-head rendering inputs.",
            compared_queries=360,
            participants=cases,
            units="Maximum absolute difference over float32 coordinates. Each case uses four query trials and 12 by 8 different labeled support photos with their own four trials.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "predict", "replay", "audit"))
    p.add_argument("--subject", choices=("sub-04", "sub-05", "sub-01"), default="sub-04")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--input", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--fold", choices=range(5), type=int)
    p.add_argument("--save-example", type=Path)
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(a)
    elif a.stage == "replay":
        replay(a)
    elif a.stage == "audit":
        audit(a)
    else:
        if not a.input or not a.out:
            p.error("predict requires --input and --out")
        model = PersonalModel(a.model_dir, a.device)
        with np.load(a.input, allow_pickle=False) as d:
            result = model.infer(d)
        result.update(
            input_sha256=sha(a.input),
            weights_sha256=model.receipt["weights_sha256"],
            heads_sha256=model.receipt["heads_sha256"],
        )
        a.out.parent.mkdir(exist_ok=True, parents=True)
        frozen_npz(a.out, result)
        print(
            "Predicted word",
            str(result["word"]),
            "unit H/14 shape",
            result["own"].shape,
            flush=True,
        )
