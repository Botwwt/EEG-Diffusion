"""Select a generated JPEG from frozen H/14 and within-object EEG pose, then score sealed choices."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from fit_personal_pose import constrain
from PIL import Image
from prepare_expanded_personal import extract
from run_prospective_readout import frozen_npz
from scipy.stats import rankdata
from score_personal12_pose import moments, ort_session
from select_within_photo_candidates import bank_metadata, unit

CODE = Path(__file__).resolve().parent
PUBLIC = CODE.parent / "results/object_fidelity"
DESIGN = PUBLIC / "pose_candidate_design.json"
LOCK = "48daa7d3d4938f9737201851ad3cd7e697183b382f09837422406165dd1b8482"
ARMS = ("joint", "wrong", "mean", "pose_only", "old_h14", "old_uniform", "fixed0")
PEOPLE = ("sub-04", "sub-05", "sub-01")
PANELS = ("registered", "replicate")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def frozen(p, value):
    blob = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if p.exists() and p.read_bytes() != blob:
        raise ValueError(f"Conflicting immutable receipt: {p}")
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_bytes(blob)


def check(ok, why):
    if not ok:
        raise ValueError(why)


def bank_pose(a, p, spec, session):
    folder, records, words, _, allowed, _ = bank_metadata(a, p)
    check(sha(folder / "settings.json") == spec["bank_manifests"][p], "Generated bank changed")
    values = []
    hashes = []
    for item in records:
        raw = (folder / item["path"]).read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        check(h == item["sha256"], "JPEG bank hash mismatch")
        with Image.open(io.BytesIO(raw)) as im:
            check(im.format == "JPEG" and im.size == (512, 512), "Unexpected bank image format")
            values.append(moments(im.convert("RGB"), session))
        hashes.append(h)
    check(len(set(hashes)) == 480, "Repeated generated images")
    return (
        words,
        allowed,
        np.asarray(values).reshape(5, 12, 8, 5),
        np.array(hashes).reshape(5, 12, 8),
    )


def physical(a, s, files):
    import torch
    from fit_trial_object_encoder import TrialObjectEncoder
    from safetensors.torch import load_file

    root = a.data / "object-transfer"
    receipt = root / (s + ".json")
    hp = root / "halves" / (s + ".npz")
    check(
        sha(receipt) == files["physical_ranges"] and sha(hp) == files["posterior_packet"],
        "Unregistered target EEG",
    )
    rec = read(receipt)
    check(
        rec["subject"] == s
        and rec["serialized_shape"] == [16540, 4, 63, 250]
        and rec["dtype"] == "<f8",
        "Wrong physical axis or subject",
    )
    with np.load(hp, allow_pickle=False) as z:
        rows = z["rows"]
        halves = z["half_eeg"]
        names = z["channels"].tolist()
        slots = z["stored_slots"]
    eeg = extract(halves.mean(0)).astype(float)
    axes = np.load(root / (s + ".npz"), allow_pickle=False)
    channels = axes["channels"].tolist()
    axes.close()
    picks = [channels.index(v) for v in names]
    raw = []
    allrows = []
    for item in rec["row_ranges"]:
        lo, hi = item["bytes"]
        start, end = item["rows"]
        blob = (root / "ranges" / rec["revision"] / s / f"{lo}-{hi}.bin").read_bytes()
        check(
            len(blob) == hi - lo + 1 and hashlib.sha256(blob).hexdigest() == item["sha256"],
            "Different full trial range",
        )
        trials = np.frombuffer(blob, "<f8").reshape(-1, 4, 63, 250)
        post = np.take(trials, picks, axis=2).astype(np.float32)
        ii = np.arange(start, end + 1)
        loc = np.searchsorted(rows, ii)
        rebuilt = np.stack([post[:, sl].mean(1) for sl in slots])
        check(
            np.array_equal(ii, rows[loc]) and np.array_equal(halves[:, loc], rebuilt),
            "Full-head and posterior readouts differ physically",
        )
        allrows.extend(ii)
        raw.append(np.ascontiguousarray(trials[:, :, :, :125], np.float32))
    check(np.array_equal(allrows, rows) and len(rows) == 240, "Different trials/order")
    modeldir = a.data / (
        "trial-object-encoder/target" if s == "sub-04" else f"trial-object-replication/{s}/fit"
    )
    weights = modeldir / "model.safetensors"
    check(sha(weights) == files["trial_encoder"], "Changed trial encoder")
    pars = load_file(weights, device=a.device)
    mu, sd = pars.pop("input_mean"), pars.pop("input_scale")
    net = TrialObjectEncoder().to(a.device)
    net.load_state_dict(pars)
    net.eval()
    with torch.inference_mode():
        x = torch.from_numpy(np.concatenate(raw)).to(a.device)
        deep = torch.cat([net(((b - mu) / sd).clamp(-12, 12)) for b in x.split(128)]).cpu().numpy()
    return rows, eeg, deep.astype(float)


def select(a, spec):
    check(sha(a.bank / "design.json") == spec["bank_design"], "Wrong eight-candidate bank design")
    session = ort_session(a)
    bank = {p: bank_pose(a, p, spec, session) for p in PANELS}
    sealed = {}
    for s in PEOPLE:
        sf = spec["sources"][s]
        sr, post, deep = physical(a, s, sf)
        model = a.data / "personal-pose/models" / s / "model.npz"
        check(
            sha(model) == sf["pose_model"]
            and sha(model.parent / "receipt.json") == sf["pose_model_receipt"],
            "Different generic pose model",
        )
        with np.load(model, allow_pickle=False) as m:
            gains, sd, w0, w1, ids = (
                m[k] for k in ("gains", "scale", "posterior", "deep", "fit_classes")
            )
        check(
            len(set(ids)) == 512 and not set(ids).intersection(sr // 10),
            "EEG training object in evaluation",
        )
        for panel in PANELS:
            words, allowed, bankpose, hs = bank[panel]
            folder = a.bank / "selected/renders" / panel / s
            historical = folder / "selection.json"
            cfg = read(folder / "settings.json")
            p = a.data / "photo-structure/within_photo" / s / panel / "conditions_512.npz"
            check(
                sha(historical) == sf["old_selection"][panel]
                and sha(p) == sf["H14_prediction"][panel]
                and sha(p) == cfg["packet_sha256"],
                "Different existing H/14 or candidate choices",
            )
            old = read(historical)
            with np.load(p, allow_pickle=False) as z:
                rows, fold, term, support, donors = [
                    z[k] for k in ("rows", "folds", "predicted_words", "support_rows", "donor_rows")
                ]
            col = np.array([words.tolist().index(w) for w in term])
            si = np.searchsorted(sr, support)
            qi = np.searchsorted(sr, rows)
            check(
                np.array_equal(sr[si], support)
                and np.array_equal(sr[qi], rows)
                and np.array_equal(np.sort(allowed[fold, col], axis=1), np.sort(support, axis=1))
                and all(not np.intersect1d(allowed[f], rows[fold == f]).size for f in range(5)),
                "Wrong calibration, query leakage or row order",
            )
            lookup = {r: j for j, r in enumerate(rows)}
            di = np.array([lookup[r] for r in donors])
            check(
                np.array_equal(term, term[di])
                and np.array_equal(fold, fold[di])
                and np.array_equal(support, support[di]),
                "Invalid wrong EEG",
            )
            candidate = bankpose[fold, col]
            prior = candidate.mean(1)
            delta = ((post[qi] - post[si].mean(1)) @ w0) * gains[0] + (
                (deep[qi] - deep[si].mean(1)) @ w1
            ) * gains[1]
            target = constrain(prior + delta)
            neutral = constrain(prior)

            def pose(t, images=candidate, dimensions=sd):
                return -np.sum(((images - t[:, None]) / dimensions) ** 2, axis=2)

            h0, h1, hm = [np.asarray(old["scores"][k]) for k in ("own", "wrong", "uniform")]
            check(
                np.allclose(h0[di], h1, atol=1e-6, rtol=0),
                "Historical H14 negative control changed",
            )

            def fused(h, p):
                return (rankdata(h, axis=1) + rankdata(p, axis=1)).argmax(1)

            own, wrong, mean = pose(target), pose(target[di]), pose(neutral)
            choices = dict(
                joint=fused(h0, own),
                wrong=fused(h1, wrong),
                mean=fused(hm, mean),
                pose_only=own.argmax(1),
                old_h14=h0.argmax(1),
                old_uniform=hm.argmax(1),
                fixed0=np.zeros(120, int),
            )
            check(
                np.array_equal(choices["joint"][di], choices["wrong"]),
                "Wrong selections do not replay the registered donors",
            )
            for arm, k in [("old_h14", "own"), ("old_uniform", "uniform"), ("fixed0", "fixed0")]:
                check(
                    np.array_equal(choices[arm], old["indices"][k]), "Historical bank pick changed"
                )
            out = a.out / panel / s / "choices.npz"
            out.parent.mkdir(exist_ok=True, parents=True)
            frozen_npz(
                out,
                dict(
                    rows=rows,
                    folds=fold,
                    words=term,
                    donor_rows=donors,
                    support_rows=support,
                    candidate_hashes=hs[fold, col],
                    predicted_pose=target,
                    candidate_mean_pose=prior,
                    **choices,
                ),
            )
            sealed[panel + "/" + s] = sha(out)
    frozen(
        a.out / "selections.json",
        dict(design_sha256=LOCK, selector_sha256=sha(Path(__file__)), cases=sealed, arms=ARMS),
    )


def credit(d):
    return (d > 1e-12).astype(float) + 0.5 * (np.abs(d) <= 1e-12)


def score(a, spec):
    from score_official_objects import topk_credit
    from score_reconstruction_jpegs import registered

    seal = read(a.out / "selections.json")
    check(
        seal["design_sha256"] == LOCK and seal["selector_sha256"] == sha(Path(__file__)),
        "Selections not frozen with this protocol/program",
    )
    matrices = {}
    allarms = (*ARMS, "oracle")
    for b in ("dino", "clip"):
        path = a.bank / (b + "_bank.npz")
        check(sha(path) == spec["evaluation_files"][b], f"Independent {b} feature cache changed")
        with np.load(path, allow_pickle=False) as z:
            matrices[b] = dict(z)
    for z in matrices.values():
        check(
            z["features"].shape[0] == len(set(z["jpeg_sha256"])) == 960,
            "Different generated image universe",
        )
    clusters = {}
    results = {}
    evidence = {}
    differ = {}
    old = {b: read(PUBLIC / f"generated_bank_{b}.json")["subjects"] for b in matrices}
    for p in PANELS:
        reg = registered(a.data / "object-fidelity", p)
        check(reg["sha"] == spec["registration"][p], "Changed independent evaluation gallery")
        rows = reg["rows"]
        folds = reg["folds"]
        labels = reg["labels"]
        groups = [np.flatnonzero(labels == w) for w in labels]
        irow = np.arange(120)
        for s in PEOPLE:
            key = p + "/" + s
            f = a.out / p / s / "choices.npz"
            check(sha(f) == seal["cases"][key], "Chosen generated photos changed")
            with np.load(f, allow_pickle=False) as z:
                chosen = {k: z[k] for k in ARMS}
                h = z["candidate_hashes"]
                word = z["words"]
                check(
                    np.array_equal(z["rows"], rows) and np.array_equal(z["folds"], folds),
                    "Evaluation rows/folds differ",
                )
            pairs = [
                np.flatnonzero((labels == t) & (folds == fold))
                for t in np.unique(labels)
                for fold in range(5)
            ]
            pair = np.array([v for v in pairs if np.array_equal(labels[v], word[v])])
            first, second = pair.T
            clusters[key] = {"shots": labels, "pairs": labels[first]}
            results[key] = {}
            checks = {}
            d = matrices["dino"]
            ix = {v: i for i, v in enumerate(d["jpeg_sha256"])}
            idx = np.array([[ix[hh] for hh in v] for v in h])
            dino = unit(d["features"][idx].astype(float))
            orig = unit(d[p].astype(float))
            sim = dino @ orig.T
            margins = np.stack(
                [sim[i, :, i] - sim[i][:, g[g != i]].max(1) for i, g in enumerate(groups)]
            )
            oracle = margins.argmax(1)
            chosen["oracle"] = oracle
            differ[key] = {k: int(np.sum(chosen["joint"] != chosen[k])) for k in allarms}
            unique_cells = {}
            for i in range(120):
                cand = dino[i]
                unique_cells[(int(folds[i]), str(word[i]))] = float(
                    (cand @ cand.T)[np.triu_indices(8, 1)].mean()
                )
            checks["mean_candidate_pair_cosine_used_cells"] = float(
                np.mean(list(unique_cells.values()))
            )
            checks["used_word_fold_cells"] = len(unique_cells)
            checks["oracle_attainable_under_fixed_wrong_word"] = int(
                np.sum((word != labels) & (margins.max(1) > 0))
            )
            checks["wrong_words"] = int(np.sum(word != labels))
            checks["oracle_attainable_under_fixed_correct_word"] = int(
                np.sum((word == labels) & (margins.max(1) > 0))
            )
            checks["correct_words"] = int(np.sum(word == labels))
            oracle_file = a.out / p / s / "oracle_evaluation_only.npz"
            frozen_npz(
                oracle_file,
                dict(rows=rows, index=oracle, sha256=h[irow, oracle], margin=margins[irow, oracle]),
            )
            checks["oracle_sha256"] = sha(oracle_file)
            for b, bank in matrices.items():
                hmm = {v: i for i, v in enumerate(bank["jpeg_sha256"])}
                check(
                    np.array_equal(bank[p + "_rows"], rows),
                    "Independent original gallery order mismatch",
                )
                y = bank[p]
                metrics = {}
                for k in allarms:
                    v = bank["features"][[hmm[z] for z in h[irow, chosen[k]]]]
                    cos = unit(v.astype(float)) @ unit(y.astype(float)).T
                    rank = (unit(v - v.mean(0)) @ unit(y - y.mean(0)).T) if b == "clip" else cos
                    top = np.array(
                        [
                            float(rank[i, i] == rank[i, g].max())
                            / sum(rank[i, g] == rank[i, g].max())
                            for i, g in enumerate(groups)
                        ]
                    )
                    m1 = cos[first, first] - cos[first, second]
                    m2 = cos[second, second] - cos[second, first]
                    vals = dict(
                        within_class_photo=top,
                        paired_assignment=credit(m1 + m2),
                        single_two_choice=(credit(m1) + credit(m2)) / 2,
                        cosine_to_original=cos.diagonal(),
                    )
                    if b == "clip":
                        ids = bank["labels"].tolist()
                        logits = (unit(v) @ bank["text"].T)[:, bank["inverse"]]
                        gt = np.array([ids.index(t) for t in labels])
                        vals["full1854"] = topk_credit(logits, 1)[irow, gt]
                    for name, vv in vals.items():
                        metrics.setdefault(name, {})[k] = vv
                    if k in ("old_h14", "old_uniform", "fixed0"):
                        was = {"old_h14": "own", "old_uniform": "uniform", "fixed0": "fixed0"}[k]
                        for name in (
                            "within_class_photo",
                            "paired_assignment",
                            "single_two_choice",
                        ):
                            obs = old[b][s]["panels"][p]["metrics"][name]["arms"][was]["credit"]
                            check(
                                float(np.sum(vals[name])) == obs,
                                "Original bank results failed parity",
                            )
                results[key][b] = metrics
            evidence[key] = checks

    def report(keys):
        out = {}
        for b in matrices:
            out[b] = {}
            for metric in results[keys[0]][b]:
                pair = metric in ("paired_assignment", "single_two_choice")
                labels = np.concatenate([clusters[k]["pairs" if pair else "shots"] for k in keys])
                vals = np.stack(
                    [np.concatenate([results[k][b][metric][arm] for k in keys]) for arm in allarms],
                    axis=1,
                )
                universe = sorted(set(np.concatenate([clusters[k]["shots"] for k in keys])))
                count = np.array([sum(labels == c) for c in universe])
                by = np.stack([vals[labels == c].sum(0) for c in universe])
                boot = np.random.default_rng(260559).multinomial(
                    len(universe), np.ones(len(universe)) / len(universe), size=10000
                )
                den = boot @ count
                good = den > 0
                sam = (boot[good] @ by) / den[good, None]
                avg = vals.mean(0)
                item = dict(
                    n=len(labels),
                    contributing_objects=int(np.sum(count > 0)),
                    means=dict(zip(allarms, avg.tolist(), strict=True)),
                    joint_minus={},
                )
                if metric != "cosine_to_original":
                    item["credit"] = dict(zip(allarms, vals.sum(0).tolist(), strict=True))
                for arm in ("old_h14", "wrong", "mean", "fixed0"):
                    j = allarms.index(arm)
                    item["joint_minus"][arm] = {
                        "delta": float(avg[0] - avg[j]),
                        "ci95": np.quantile(sam[:, 0] - sam[:, j], [0.025, 0.975]).tolist(),
                    }
                out[b][metric] = item
        return out

    full = report(list(results))
    primary = full["dino"]["within_class_photo"]
    check(
        primary["n"] == 720
        and primary["contributing_objects"] == 24
        and full["dino"]["paired_assignment"]["n"] == 170,
        "Unexpected registered denominators",
    )
    passed = all(primary["joint_minus"][k]["ci95"][0] > 0 for k in ("old_h14", "wrong", "mean"))
    result = dict(
        design_sha256=LOCK,
        scorer_sha256=sha(Path(__file__)),
        selection_seal_sha256=sha(a.out / "selections.json"),
        pass_gate=passed,
        counts=spec["n"],
        arms=allarms,
        pooled=full,
        by_person={s: report([k for k in results if k.endswith("/" + s)]) for s in PEOPLE},
        by_panel={p: report([k for k in results if k.startswith(p + "/")]) for p in PANELS},
        choices_differ_from_joint=differ,
        oracle_and_diversity=evidence,
    )
    frozen(a.out / "summary.json", result)
    if a.release:
        frozen(a.release, result)
    print("All-query DINO", primary["credit"], "gate", passed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["select", "score"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--bank", type=Path, default=Path("outputs/photo-structure/h14_bank"))
    p.add_argument("--out", type=Path, default=Path("outputs/pose-candidate"))
    p.add_argument("--mask-model", type=Path, default=Path("outputs/photo-structure/u2netp.onnx"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--provider", default="DmlExecutionProvider")
    p.add_argument("--release", type=Path)
    a = p.parse_args()
    check(sha(DESIGN) == LOCK, "Changed pose candidate selection registration")
    (select if a.stage == "select" else score)(a, read(DESIGN))


if __name__ == "__main__":
    main()
