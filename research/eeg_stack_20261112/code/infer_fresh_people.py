"""Export, run or audit the portable sub-05/sub-01 expanded EEG decoder on one unlabeled query."""

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
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import check
from run_prospective_readout import frozen_npz

PUB = Path(__file__).resolve().parents[1] / "results/object_fidelity"
PEOPLE = ("sub-05", "sub-01")
WORD_SHA = "4f21516d46783a49fad4fde682134cf9b8b731687f1893a709ae0d959791941d"
DESIGN_SHA = "2900cd504dafadd1483b9894f617cee0d10204334626e3c0955159903fcc3f83"


def frozen_models():
    words = PUB / "fresh_people_words.json"
    lock = PUB / "fresh_people_design.json"
    check(sha(words) == WORD_SHA and sha(lock) == DESIGN_SHA, "Frozen model source changed")
    return read(words), read(lock)


def export(a):
    score, design = frozen_models()
    study = a.root / a.subject
    fit = study / "fit"
    receipt = read(fit / "receipt.json")
    check(
        sha(fit / "receipt.json") == score["trained"][a.subject]
        and receipt["subject"] == a.subject
        and receipt["design_sha256"] == DESIGN_SHA
        and sha(fit / "model.safetensors") == receipt["weights_sha256"]
        and score["weight"] == design["neural_weight"],
        "Different fit, identity or fixed weight",
    )
    old = a.data / "personal-object-inference" / a.subject / "heads.npz"
    check(
        sha(old)
        == read(PUB / "single_query_personal_parity.json")["participants"][a.subject][
            "cached_heads_sha256"
        ],
        "Changed frozen visual model",
    )
    with np.load(old, allow_pickle=False) as z:
        head = dict(z)
    _, x = training(
        SimpleNamespace(
            expanded=a.data / "photo-structure/expanded", out=a.data / "double-calibration"
        ),
        a.subject,
    )
    ids = read(PUB / "double_calibration_design.json")
    check(
        receipt["train_ids"] == sorted(ids["old_objects"] + ids["extra_objects"])
        and x.shape == (10240, 1003),
        "Unexpected Fisher training population",
    )
    basis = np.vstack((np.zeros((1, 1003)), np.eye(1003)))
    fx = fisher(x.reshape(1024, 10, 1003), basis)
    head.update(fisher_weight=fx[1:] - fx[0], fisher_bias=fx[0], neural_weight=score["weight"])
    p = a.model_dir
    frozen_npz(p / "heads.npz", head)
    dest = p / "model.safetensors"
    if dest.exists():
        check(sha(dest) == receipt["weights_sha256"], "Existing model differs")
    else:
        shutil.copyfile(fit / "model.safetensors", dest)
    freeze(
        p / "receipt.json",
        dict(
            subject=a.subject,
            heads_sha256=sha(p / "heads.npz"),
            weights_sha256=sha(dest),
            word_model_sha256=WORD_SHA,
            design_sha256=DESIGN_SHA,
            fit_sha256=score["trained"][a.subject],
            old_visual_heads_sha256=sha(old),
            neural_weight=score["weight"],
        ),
    )
    print(a.subject, "export", sha(p / "receipt.json"), flush=True)


class ExpandedModel(PersonalModel):
    def __init__(self, folder, device, subject):
        super().__init__(folder, device)
        check(
            self.receipt["subject"] == subject
            and subject in PEOPLE
            and self.receipt["design_sha256"] == DESIGN_SHA
            and self.receipt["word_model_sha256"] == WORD_SHA
            and float(self.head["neural_weight"])
            == self.receipt["neural_weight"]
            == 0.5907766654022144
            and self.head["fisher_weight"].shape == (1003, 96)
            and self.head["visual_weight"].shape == (1003, 1024),
            "Wrong expanded model or mixture",
        )
        if device == "cuda":
            torch.backends.cudnn.allow_tf32 = True

    def infer(self, d):
        check(
            set(d) == KEYS
            and np.asarray(d["subject"]).shape == ()
            and str(d["subject"]) == self.receipt["subject"],
            "Expected one query for this participant with the documented input keys",
        )
        q, s, y = (d[k] for k in ("query_eeg", "support_eeg", "support_h14"))
        names = d["channels"].tolist()
        words = d["candidate_words"]
        ids = d["support_ids"]
        qid = d["query_id"]
        check(
            d["channels"].shape == (63,)
            and set(names) == set(CHANNEL_NAMES)
            and d["times"].shape == (250,)
            and np.allclose(d["times"], np.arange(250) / 250, atol=1e-7, rtol=0),
            "Invalid montage/times",
        )
        check(
            words.shape == (12,)
            and words.dtype.kind == "U"
            and len(set(words)) == 12
            and all(w.strip() for w in words)
            and ids.shape == (12, 8)
            and ids.dtype.kind in "iuU"
            and len(set(ids.astype(str).flat)) == 96
            and all(x.strip() for x in ids.astype(str).flat)
            and qid.shape == ()
            and qid.dtype.kind in "iuU"
            and str(qid).strip()
            and str(qid) not in ids.astype(str),
            "Invalid or overlapping support/query identities or words",
        )
        check(
            q.shape == (4, 63, 250)
            and s.shape == (12, 8, 4, 63, 250)
            and y.shape == (12, 8, 1024)
            and all(z.dtype.kind == "f" and np.isfinite(z).all() for z in (q, s, y))
            and np.all(np.linalg.norm(y, axis=2) > 1e-8),
            "Bad trial or support-image input",
        )
        waves = np.take(
            np.concatenate((q[None], s.reshape(96, 4, 63, 250))).astype(np.float32),
            [names.index(ch) for ch in CHANNEL_NAMES],
            axis=2,
        )
        x = extract(np.take(waves, self.pick, axis=2).mean(1, dtype=np.float32)).astype(float)
        fx = x @ self.head["fisher_weight"] + self.head["fisher_bias"]
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(waves[:, :, :, :125])).to(self.device)
            deep = self.model(((t - self.mu) / self.sd).clamp(-12, 12)).cpu().numpy().astype(float)
        fs, ds = (similarities(z[:1], z[1:].reshape(12, 8, 96))[0] for z in (fx, deep))
        w = float(self.head["neural_weight"])
        scores = (1 - w) * fs + w * ds
        check(np.isfinite(scores).all(), "Nonfinite model score")
        win = int(scores.argmax())
        xr = np.vstack((x[:1], x[1:].reshape(12, 8, 1003)[win]))
        hv = xr @ self.head["visual_weight"] + self.head["visual_bias"]
        mean = unit(y[win].astype(float)).mean(0, keepdims=True)
        check(np.linalg.norm(mean) > 1e-8, "Selected support-image directions cancel")
        prior = unit(mean).astype(np.float32)
        native = generative(mean, hv[:1] - hv[1:].mean(0)).astype(np.float32)
        own = rotate(prior, native, self.head)[0]
        check(
            np.isfinite(own).all() and abs(np.linalg.norm(own) - 1) < 1e-5,
            "Nonfinite/invalid H14 condition",
        )
        return dict(
            subject=d["subject"],
            query_id=qid,
            word=words[win],
            candidate_words=words,
            mixed_scores=scores,
            fisher_scores=fs,
            trial_scores=ds,
            support_ids=ids[win],
            own=own,
            uniform=prior[0],
            radius=self.head["radius"],
        )


def robustness(model, d):
    other = next(s for s in PEOPLE if s != str(d["subject"]))

    def mutated(key, fn):
        v = d[key].copy()
        fn(v)
        return {key: v}

    errors = {
        "query_truth_field": {"truth": np.array("unavailable")},
        "wrong_participant": {"subject": np.array(other)},
        "duplicate_word": mutated("candidate_words", lambda v: v.__setitem__(1, v[0])),
        "duplicate_support": mutated("support_ids", lambda v: v.__setitem__((0, 1), v[0, 0])),
        "query_reused_as_support": {"query_id": np.asarray(d["support_ids"][0, 0])},
        "wrong_montage": mutated("channels", lambda v: v.__setitem__(1, v[0])),
        "shifted_time": {"times": d["times"] + 0.004},
        "missing_repeat": {"query_eeg": d["query_eeg"][:3]},
        "nonfinite_eeg": mutated("query_eeg", lambda v: v.__setitem__((0, 0, 0), np.nan)),
        "invalid_support_image": mutated("support_h14", lambda v: v.__setitem__((0, 0), 0)),
    }
    rejected = []
    for k, v in errors.items():
        try:
            model.infer(d | v)
        except ValueError:
            rejected.append(k)
        else:
            raise ValueError(f"Malformed input accepted: {k}")
    expected = model.infer(d)
    index = np.arange(62, -1, -1)
    reordered = model.infer(
        d
        | dict(
            channels=d["channels"][index],
            query_eeg=d["query_eeg"][:, index],
            support_eeg=d["support_eeg"][:, :, :, index],
        )
    )
    check(
        all(np.array_equal(v, reordered[k]) for k, v in expected.items()),
        "Permuted channel names and data should give identical output",
    )
    return dict(rejected=rejected, channel_reorder_exact=True)


def replay(a):
    r, design = frozen_models()
    s = a.subject
    root = a.root / s
    m = ExpandedModel(a.model_dir, a.device, s)
    visual = images(a.data / "official-object-readout/sub-03")
    receipt = read(root / "eeg" / (s + ".json"))
    check(receipt["subject"] == s and receipt["design_sha256"] == DESIGN_SHA, "Wrong EEG source")
    maximum = {k: 0.0 for k in ("own", "uniform", "mixed_scores", "fisher_scores", "trial_scores")}
    counts = {}
    bad = None
    render = root / "render_inputs/learned"
    for panel, menu in design["panels"][s].items():
        rp = root / "references" / (panel + "_ablation.json")
        p = root / "decisions" / (panel + ".npz")
        zfile = render / "conditions" / s / panel / "conditions_512.npz"
        check(
            sha(rp) == menu["registration_sha256"]
            and sha(p) == r["decisions"][s][panel]
            and sha(zfile) == read(render / "design.json")["packets_sha256"][f"{s}/{panel}"],
            "Original decision/packet/row set changed",
        )
        reg = read(rp)
        waves = {}
        for obj in menu["object_ids"]:
            path = root / "eeg/raw" / f"{obj:04d}.bin"
            check(sha(path) == receipt["raw_sha256"][str(obj)], "Source raw EEG differs")
            raw = np.fromfile(path, "<f8").reshape(10, 4, 63, 250).astype(np.float32)
            waves.update({10 * obj + i: v for i, v in enumerate(raw)})
        n = 0
        exact = {"word": 0, "supports": 0, "own": 0, "uniform": 0}
        with np.load(p, allow_pickle=False) as dec, np.load(zfile, allow_pickle=False) as z:
            check(
                np.array_equal(z["rows"], reg["rows"])
                and np.array_equal(z["folds"], reg["folds"])
                and np.array_equal(z["predicted_words"], dec["words"][dec["learned"]]),
                "Frozen prediction is misaligned",
            )
            fs, ds = dec["scores"][0, 2:4]
            w = float(m.head["neural_weight"])
            mixed = (1 - w) * fs + w * ds
            for f in range(5):
                ids = np.asarray(reg["calibration_rows_by_fold_and_word"][f])
                idx = np.flatnonzero(z["folds"] == f)
                d = dict(
                    subject=np.array(s),
                    channels=np.array(CHANNEL_NAMES),
                    times=np.arange(250) / 250,
                    support_ids=ids,
                    support_eeg=np.stack([waves[i] for i in ids.flat]).reshape(12, 8, 4, 63, 250),
                    support_h14=np.asarray(visual[ids]),
                    candidate_words=np.asarray(reg["words"]),
                )
                for i in idx:
                    row = z["rows"][i]
                    payload = d | dict(query_eeg=waves[row], query_id=np.asarray(row))
                    if bad is None:
                        bad = robustness(m, payload)
                        if a.example:
                            a.example.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(a.example, **payload)
                    got = m.infer(payload)
                    n += 1
                    exact["word"] += int(got["word"] == z["predicted_words"][i])
                    exact["supports"] += int(
                        np.array_equal(got["support_ids"], z["support_rows"][i])
                    )
                    for k, truth in dict(
                        own=z["own"][i],
                        uniform=z["uniform"][i],
                        mixed_scores=mixed[i],
                        fisher_scores=fs[i],
                        trial_scores=ds[i],
                    ).items():
                        if k in ("own", "uniform"):
                            exact[k] += int(np.array_equal(got[k], truth))
                        maximum[k] = max(maximum[k], float(np.max(np.abs(got[k] - truth))))
        check(n == exact["word"] == exact["supports"] == 120, "Selected word or supports differs")
        counts[panel] = dict(
            queries=n, exact=exact, packet_sha256=sha(zfile), decision_sha256=sha(p)
        )
        print(s, panel, exact, maximum, flush=True)
    check(
        maximum["mixed_scores"] < 2e-4 and maximum["own"] < 2e-6 and maximum["uniform"] < 2e-6,
        "Exceeded portable decoder tolerance",
    )
    out = dict(
        subject=s,
        device=a.device,
        queries=sum(v["queries"] for v in counts.values()),
        menus=counts,
        max_abs_error=maximum,
        model_receipt_sha256=sha(a.model_dir / "receipt.json"),
        weights_sha256=m.receipt["weights_sha256"],
        heads_sha256=m.receipt["heads_sha256"],
        robustness=bad,
        driver_sha256=sha(Path(__file__)),
    )
    freeze(a.model_dir / ("parity_" + a.device + ".json"), out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("export", "predict", "replay", "summarize"))
    p.add_argument("--subject", choices=PEOPLE)
    p.add_argument("--root", type=Path, default=Path("outputs/fresh-people"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--input", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--example", type=Path)
    a = p.parse_args()
    if a.stage == "summarize":
        reports = {
            s: read(a.root / s / "portable" / ("parity_" + a.device + ".json")) for s in PEOPLE
        }
        check(
            all(
                v["subject"] == s
                and v["driver_sha256"] == sha(Path(__file__))
                and v["queries"] == 480
                for s, v in reports.items()
            ),
            "Missing participant/model replay",
        )
        freeze(
            a.out or PUB / "fresh_people_single_query.json",
            dict(
                queries=960,
                device=a.device,
                design_sha256=DESIGN_SHA,
                word_result_sha256=WORD_SHA,
                participants=reports,
            ),
        )
        return
    if not a.subject:
        p.error("Select the model participant with --subject")
    a.model_dir = a.model_dir or a.root / a.subject / "portable"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if a.stage == "export":
        export(a)
    elif a.stage == "replay":
        replay(a)
    else:
        if not a.input or not a.out:
            p.error("predict requires --input and --out")
        m = ExpandedModel(a.model_dir, a.device, a.subject)
        with np.load(a.input, allow_pickle=False) as z:
            pred = m.infer(z)
        frozen_npz(
            a.out,
            pred
            | dict(
                input_sha256=sha(a.input),
                weights_sha256=m.receipt["weights_sha256"],
                heads_sha256=m.receipt["heads_sha256"],
            ),
        )
        print(str(pred["word"]), flush=True)


if __name__ == "__main__":
    main()
