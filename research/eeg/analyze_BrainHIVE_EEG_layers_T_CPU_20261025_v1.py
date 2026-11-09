"""Finite CPU-only, T-only held-concept readout diagnostics.

This does not fit a new EEG encoder or generate/score images. Every readout is
fit and evaluated *inside the held pool of one frozen OOF encoder*. Pooling the
stitched OOF bank before an outer probe split is deliberately forbidden.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

SCHEMA = "BrainHIVE_decisive_EEG_layers_T_CPU_v1"
METHODS = ("c", "h0", "hpre", "c_cstats", "c_prestats",
           "c_c", "c_h0", "c_h0_shuffled")
METRICS = ("mse", "cosine_correct", "cosine_wrong", "cosine_margin")
CONTRASTS = (("h0", "c"), ("hpre", "c"), ("c_prestats", "c_cstats"),
             ("c_h0", "c_c"), ("c_h0", "c_h0_shuffled"), ("c_h0", "c_prestats"))


def stable_order(values, seed, namespace):
    return sorted(values, key=lambda x: (
        hashlib.sha256(f"{seed}|{namespace}|{x}".encode()).hexdigest(), str(x)))


def held_probe_split(rows, concept_by_row, seed=1729):
    """Two fixed concept partitions; independent of features and targets."""
    concepts = stable_order({concept_by_row[r] for r in rows}, seed, "probe-concepts")
    assert len(concepts) >= 4
    side = {c: i % 2 for i, c in enumerate(concepts)}
    a = [r for r in rows if side[concept_by_row[r]] == 0]
    b = [r for r in rows if side[concept_by_row[r]] == 1]
    assert a and b and not set(a) & set(b) and set(a) | set(b) == set(rows)
    assert not {concept_by_row[r] for r in a} & {concept_by_row[r] for r in b}
    return a, b


def different_concept_permutation(rows, concept_by_row, seed, namespace):
    """Positions forming a bijection, never a label-chosen hard negative."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(concept_by_row[row], []).append(i)
    assert len(groups) > 1
    ordered = []
    for concept in stable_order(groups, seed, namespace + "|concept"):
        ordered.extend(stable_order(groups[concept], seed, namespace + "|" + concept))
    shift = max(map(len, groups.values()))
    assert shift * 2 <= len(rows)
    permutation = [-1] * len(rows)
    for i, position in enumerate(ordered):
        permutation[position] = ordered[(i + shift) % len(ordered)]
    assert sorted(permutation) == list(range(len(rows)))
    assert all(concept_by_row[rows[i]] != concept_by_row[rows[j]]
               for i, j in enumerate(permutation))
    return permutation


def validate_scopes(all_T, encoder_fit, encoder_held, pool, probe_fit, probe_eval,
                    concept_by_row):
    for values in (all_T, encoder_fit, encoder_held, pool, probe_fit, probe_eval):
        assert len(values) == len(set(values)), "Duplicate rows"
    assert not set(encoder_fit) & set(encoder_held)
    assert set(encoder_fit) | set(encoder_held) == set(all_T)
    assert set(pool) == set(encoder_held), "Only the actual encoder-held pool is permitted"
    assert set(probe_fit) | set(probe_eval) == set(pool)
    assert not set(probe_fit) & set(probe_eval)
    assert not set(probe_fit + probe_eval) & set(encoder_fit)
    train_concepts = {concept_by_row[r] for r in probe_fit}
    eval_concepts = {concept_by_row[r] for r in probe_eval}
    encoder_concepts = {concept_by_row[r] for r in encoder_fit}
    assert not train_concepts & eval_concepts
    assert not (train_concepts | eval_concepts) & encoder_concepts


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return None
    return None


class Budget:
    def __init__(self, limits, output):
        self.started = time.monotonic()
        self.limits = limits
        self.output = Path(output)
        self.baseline_rss = rss_bytes()
        assert self.baseline_rss is not None, "Actual analysis requires Linux RSS accounting"
        self.peak_extra_rss = 0

    def check(self):
        elapsed = time.monotonic() - self.started
        assert elapsed <= self.limits["maximum_seconds"], "Finite analysis timeout"
        current = rss_bytes()
        assert current is not None
        extra = max(0, current - self.baseline_rss)
        self.peak_extra_rss = max(self.peak_extra_rss, extra)
        assert extra <= self.limits["maximum_extra_RSS_bytes"], "Extra RSS budget"
        size = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
        assert size <= self.limits["maximum_output_bytes"], "Output budget"
        return {"seconds": elapsed, "extra_RSS_bytes": extra,
                "peak_extra_RSS_bytes": self.peak_extra_rss, "output_bytes": size}


def file_sha(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
            if budget:
                budget.check()
    return digest.hexdigest()


class Inputs:
    def __init__(self, config, budget):
        self.config = config
        self.budget = budget
        self.verified = {}

    def ref(self, key):
        return self.config.get("refs", self.config)[key]

    def checked_path(self, ref):
        assert set(("path", "sha256")) <= set(ref)
        path = str(Path(ref["path"]).resolve())
        assert len(ref["sha256"]) == 64
        if path not in self.verified:
            actual = file_sha(path, self.budget)
            assert actual == ref["sha256"], ("Input SHA mismatch", path)
            self.verified[path] = actual
        assert self.verified[path] == ref["sha256"]
        return Path(path)

    def json(self, key):
        return json.loads(self.checked_path(self.ref(key)).read_bytes())


def native_taps(torch, eeg, state):
    expected = {"proj_in.weight", "proj_in.bias", "block.layers.0.ff.w1.weight",
                "block.layers.0.ff.w2.weight", "block.layers.0.norm.weight",
                "block.layers.0.norm.bias"}
    assert set(state) == expected
    assert sum(v.numel() for v in state.values()) == 6452224
    assert all(v.device.type == "cpu" and v.dtype == torch.float32 for v in state.values())
    f = torch.nn.functional
    h0 = f.linear(eeg.reshape(len(eeg), -1), state["proj_in.weight"], state["proj_in.bias"])
    hidden = f.gelu(f.linear(h0, state["block.layers.0.ff.w1.weight"]))
    hpre = h0 + f.linear(hidden, state["block.layers.0.ff.w2.weight"])
    c = f.layer_norm(hpre, (1024,), state["block.layers.0.norm.weight"],
                     state["block.layers.0.norm.bias"], 1e-6)
    assert h0.shape == hpre.shape == c.shape == (len(eeg), 1024)
    assert all(torch.isfinite(v).all() for v in (h0, hpre, c))
    return h0, hpre, c


def row_stats(torch, x):
    return torch.stack((x.mean(1), (x.var(1, unbiased=False) + 1e-6).sqrt().log()), 1)


def raw_feature(torch, taps, method, positions, permutation):
    c, h0, hpre = (taps[name][positions].double() for name in ("c", "h0", "hpre"))
    if method in ("c", "h0", "hpre"):
        return {"c": c, "h0": h0, "hpre": hpre}[method]
    if method == "c_cstats":
        return torch.cat((c, row_stats(torch, c)), 1)
    if method == "c_prestats":
        return torch.cat((c, row_stats(torch, hpre)), 1)
    if method == "c_c":
        return torch.cat((c, c), 1)
    if method == "c_h0":
        return torch.cat((c, h0), 1)
    if method == "c_h0_shuffled":
        return torch.cat((c, h0[permutation]), 1)
    raise ValueError(method)


def standardize(torch, fit, evaluation, method):
    mean = fit.mean(0)
    scale = fit.std(0, unbiased=False)
    scale = torch.where(scale >= 1e-6, scale, torch.ones_like(scale))
    x = (fit - mean) / scale
    z = (evaluation - mean) / scale
    # Each half of a concatenation is standardized on probe-fit rows only.
    # Duplicate-c then has exactly the c-only kernel and regularization scale.
    if method in ("c_c", "c_h0", "c_h0_shuffled"):
        x = x / math.sqrt(2)
        z = z / math.sqrt(2)
    return x, z, mean, scale


def ridge_predict(torch, x, y, z, strength):
    assert strength > 0 and x.dtype == y.dtype == z.dtype == torch.float64
    target_mean = y.mean(0)
    centered = y - target_mean
    gram = x.T @ x
    gram.diagonal().add_(len(x) * strength)
    rhs = x.T @ centered
    chol = torch.linalg.cholesky(gram)
    coefficient = torch.cholesky_solve(rhs, chol)
    prediction = z @ coefficient + target_mean
    assert torch.isfinite(prediction).all()
    return prediction, coefficient


def prediction_metrics(torch, prediction, target, wrong):
    f = torch.nn.functional
    correct = f.cosine_similarity(prediction, target, dim=1, eps=1e-12)
    incorrect = f.cosine_similarity(prediction, wrong, dim=1, eps=1e-12)
    return torch.stack(((prediction-target).square().mean(1), correct,
                        incorrect, correct-incorrect), 1)


def summarize_contrasts(torch, all_metrics):
    result = {}
    for method, baseline in CONTRASTS:
        a, b = all_metrics[method], all_metrics[baseline]
        result[f"{method}_minus_{baseline}"] = {
            "mean_MSE_reduction": float((b[:, 0] - a[:, 0]).mean()),
            "relative_MSE_reduction": float((b[:, 0].mean()-a[:, 0].mean()) /
                                           b[:, 0].mean().clamp_min(1e-12)),
            "mean_cosine_correct_gain": float((a[:, 1]-b[:, 1]).mean()),
            "mean_cosine_margin_gain": float((a[:, 3]-b[:, 3]).mean())}
    return result


def decide(summary, thresholds, lambdas, required_contrasts=None):
    """Conjunctive practical rule; no candidate/architecture auto-launch."""
    required = required_contrasts or ("c_h0_minus_c_c", "c_h0_minus_c_h0_shuffled")
    reasons = []
    for strength in lambdas:
        for contrast in required:
            row = summary[strength][contrast]
            if row["relative_MSE_reduction"] < thresholds["minimum_relative_MSE_reduction"]:
                reasons.append([strength, contrast, "relative_MSE_below_practical_floor"])
            if row["mean_cosine_margin_gain"] < thresholds["minimum_absolute_cosine_margin_gain"]:
                reasons.append([strength, contrast, "margin_below_practical_floor"])
            if row["subjects_joint_positive"] < thresholds["minimum_subjects_positive"]:
                reasons.append([strength, contrast, "subject_consistency_below_floor"])
    return {"status": "SUPPORTED" if not reasons else "NOT_SUPPORTED",
            "interpretation": "Fixed readout contrasts jointly pass the practical filter"
            if not reasons else "Fixed readout contrasts do not jointly pass the practical filter",
            "required_contrasts": list(required),
            "failed_conditions": reasons, "automatic_new_candidate_authorized": False,
            "scope": "Fixed linear-readout family and held concepts only; not an information-theoretic or image-quality claim"}


def validate_config(config):
    assert config["schema"] == SCHEMA
    assert config["seed"] == 1729
    assert config["ridge_lambdas"] == [0.1, 1.0], "Both strengths reported, never winner-selected"
    limits = config["limits"]
    assert limits["threads"] == 4
    assert 0 < limits["maximum_seconds"] <= 14400
    assert 0 < limits["maximum_extra_RSS_bytes"] <= 16 * (1 << 30)
    assert 0 < limits["maximum_output_bytes"] <= 4 * (1 << 30)
    thresholds = config["thresholds"]
    assert 0 < thresholds["minimum_relative_MSE_reduction"] < 1
    assert 0 < thresholds["minimum_absolute_cosine_margin_gain"] < 2
    assert 1 <= thresholds["minimum_subjects_positive"] <= 10
    return limits


def run(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_bytes())
    limits = validate_config(config)
    assert config["analysis_source_sha256"] == file_sha(__file__)
    output.mkdir(parents=True, exist_ok=True)
    assert not any((output/name).exists() for name in
                   ("completion.json", "progress.json", "failure.json", "per_query.csv.gz"))
    budget = Budget(limits, output)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(limits["threads"])
    import torch
    torch.set_num_threads(limits["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    inputs = Inputs(config, budget)
    contract = inputs.json("input_contract")
    assert contract["status"] == "ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT"
    split = inputs.json("split_manifest")
    bindings = inputs.json("input_bindings")
    assert contract["split_manifest_sha256"] == inputs.ref("split_manifest")["sha256"]
    assert bindings["split_manifest_sha256"] == inputs.ref("split_manifest")["sha256"]
    assert bindings["image_manifest_sha256"] == inputs.ref("image_manifest")["sha256"]
    registrations = {role: inputs.json("parent_registration_" + role) for role in ("oof0", "oof1")}
    all_T = list(split["image_rows"]["T"])
    T_set = set(all_T)
    assert len(all_T) == len(set(all_T)) == 13230
    manifest = inputs.checked_path(inputs.ref("image_manifest"))
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = {int(r["array_row"]): r for r in csv.DictReader(stream)
                if r["dataset_split"] == "train" and int(r["array_row"]) in T_set}
    assert set(rows) == set(all_T)
    concepts = {r: rows[r]["concept_id"] for r in all_T}
    assert len(set(concepts.values())) == 1323
    assert all(sum(concepts[r] == c for r in all_T) == 10 for c in set(concepts.values()))
    selected = list(bindings["selected_indices"])
    assert len(selected) == len(set(selected)) == 17
    assert len(contract["subjects"]) == 10
    target_path = inputs.checked_path({"path": contract["target_cache_path"],
                                      "sha256": contract["target_cache_sha256"]})
    target_all = torch.load(target_path, weights_only=True, map_location="cpu", mmap=True)
    assert target_all.shape == (16540, 1024)
    expected_subjects = [s["subject"] for s in contract["subjects"]]
    assert len(set(expected_subjects)) == 10
    expected_keys = {(s, r) for s in expected_subjects for r in all_T}
    coverage = {str(a): set() for a in config["ridge_lambdas"]}
    grouped = {}
    completed_units = []
    numerical_checks = []
    plan = {"schema": SCHEMA, "config_sha256": file_sha(config_path),
            "source_sha256": file_sha(__file__), "expected_queries_per_lambda": 132300,
            "expected_encoder_roles": 20, "expected_probe_units": 40,
            "methods": list(METHODS), "lambdas": config["ridge_lambdas"],
            "feature_normalization": "probe-fit coordinate mean/std only; concatenations divided by sqrt(2)",
            "readout_objective": "||XW-Y||_F^2/n + lambda*||W||_F^2; intercept from probe-fit target mean",
            "forbidden_reads": ["Dfit numerical rows", "Dcheck numerical rows", "V numerical rows", "official test", "image scores"],
            "no_GPU_context": True, "no_new_EEG_training": True}
    json_write(output/"plan.json", plan)
    metric_path = output/"per_query.csv.gz"
    with gzip.open(metric_path, "wt", encoding="utf-8", newline="") as metric_stream:
        fieldnames = ["subject", "encoder_role", "probe_direction", "array_row", "concept_id",
                      "stimulus_id", "ridge_lambda", "shuffled_h0_array_row", "wrong_target_array_row"] + [
                          f"{m}_{q}" for m in METHODS for q in METRICS]
        writer = csv.DictWriter(metric_stream, fieldnames=fieldnames)
        writer.writeheader()
        for subject in contract["subjects"]:
            name = subject["subject"]
            raw_binding = bindings["subjects"][name]
            raw_ref = {"path": raw_binding["train_path"], "sha256": raw_binding["train_sha256"]}
            raw = torch.load(inputs.checked_path(raw_ref), weights_only=True,
                             map_location="cpu", mmap=True)
            assert raw.shape == (16540, 63, 250) and raw.dtype == torch.float32
            for role in ("oof0", "oof1"):
                budget.check()
                source = next(r for r in subject["sources"] if r["role"] == role)
                registration_ref = inputs.ref("parent_registration_" + role)
                assert source["parent_registration_sha256"] == registration_ref["sha256"]
                assert Path(source["parent_registration_path"]).resolve() == Path(registration_ref["path"]).resolve()
                checkpoint = inputs.checked_path({"path": source["checkpoint_path"],
                                                   "sha256": source["checkpoint_sha256"]})
                cache_path = inputs.checked_path({"path": source["condition_cache_path"],
                                                   "sha256": source["condition_cache_sha256"]})
                cache = torch.load(cache_path, weights_only=True, map_location="cpu", mmap=True)
                assert cache["model_sha256"] == source["checkpoint_sha256"] and cache["role"] == role
                payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
                assert payload["registration_sha256"] == source["parent_registration_sha256"]
                state = payload["model"]
                del payload
                registered = registrations[role]
                assert registered["role"] == role
                pool = [int(r["array_row"]) for r in source["groups"]["component_train"]]
                assert set(pool) == set(registered["excluded_oof_rows"])
                assert all(split["oof_fold_by_concept"][concepts[r]] == int(role[-1]) for r in pool)
                pos = {row: i for i, row in enumerate(pool)}
                cache_positions = [int(r["cache_row"]) for r in source["groups"]["component_train"]]
                assert cache["array_rows"][cache_positions].tolist() == pool
                taps = {key: [] for key in ("h0", "hpre", "c")}
                with torch.no_grad():
                    for start in range(0, len(pool), 256):
                        budget.check()
                        ids = pool[start:start+256]
                        batch = raw[ids][:, selected, :].clone()
                        for key, value in zip(("h0", "hpre", "c"), native_taps(torch, batch, state)):
                            taps[key].append(value)
                    taps = {key: torch.cat(value) for key, value in taps.items()}
                    reference = cache["condition"][cache_positions].float()
                    max_error = float((reference-taps["c"]).abs().max())
                    mean_error = float((reference-taps["c"]).abs().mean())
                    minimum_cosine = float(torch.nn.functional.cosine_similarity(reference, taps["c"], dim=1).min())
                    tolerances = config.get("numerical_tolerances", {})
                    assert max_error <= tolerances.get("maximum_cache_absolute_error", 0.002)
                    assert minimum_cosine >= tolerances.get("minimum_cache_cosine", 0.99999)
                    numerical_checks.append({"subject": name, "role": role, "max_cache_error": max_error,
                                             "mean_cache_error": mean_error, "minimum_cache_cosine": minimum_cosine})
                del cache, reference, state
                a, b = held_probe_split(pool, concepts, config["seed"])
                for direction, (fit_rows, eval_rows) in enumerate(((a, b), (b, a))):
                    validate_scopes(all_T, registered["training_array_rows"], registered["excluded_oof_rows"],
                                    pool, fit_rows, eval_rows, concepts)
                    fit_pos = torch.tensor([pos[r] for r in fit_rows])
                    eval_pos = torch.tensor([pos[r] for r in eval_rows])
                    namespace = f"{name}|{role}|{direction}"
                    fit_permutation = different_concept_permutation(fit_rows, concepts, config["seed"], namespace+"|fit")
                    eval_permutation = different_concept_permutation(eval_rows, concepts, config["seed"], namespace+"|eval")
                    wrong_permutation = different_concept_permutation(eval_rows, concepts, config["seed"], namespace+"|wrong_target")
                    y = target_all[fit_rows].double()
                    truth = target_all[eval_rows].double()
                    # An independent mapping prevents mechanically rewarding the
                    # shuffled branch for predicting the selected wrong teacher.
                    wrong = truth[wrong_permutation]
                    assert torch.isfinite(y).all() and torch.isfinite(truth).all()
                    metrics_by_lambda = {str(v): {} for v in config["ridge_lambdas"]}
                    coefficient_hashes = {}
                    with torch.no_grad():
                        for method in METHODS:
                            budget.check()
                            fit = raw_feature(torch, taps, method, fit_pos, fit_permutation)
                            evaluation = raw_feature(torch, taps, method, eval_pos, eval_permutation)
                            x, z, mean, scale = standardize(torch, fit, evaluation, method)
                            for strength in config["ridge_lambdas"]:
                                budget.check()
                                prediction, coefficient = ridge_predict(torch, x, y, z, strength)
                                key = str(strength)
                                metrics_by_lambda[key][method] = prediction_metrics(torch, prediction, truth, wrong)
                                coefficient_hashes[f"{key}|{method}"] = hashlib.sha256(
                                    coefficient.contiguous().numpy().tobytes()).hexdigest()
                                del prediction, coefficient
                            del fit, evaluation, x, z, mean, scale
                    for strength, values in metrics_by_lambda.items():
                        assert set(values) == set(METHODS)
                        # Same kernel: c-c duplication is not a larger effective baseline.
                        duplicate_error = float((values["c"]-values["c_c"]).abs().max())
                        assert duplicate_error < 1e-7, ("Duplicate-c equivalence failed", duplicate_error)
                        scalar_values = {m: v.tolist() for m, v in values.items()}
                        for i, row in enumerate(eval_rows):
                            key = (name, row)
                            assert key not in coverage[strength]
                            coverage[strength].add(key)
                            record = {"subject": name, "encoder_role": role, "probe_direction": direction,
                                      "array_row": row, "concept_id": concepts[row],
                                      "stimulus_id": rows[row]["stimulus_id"], "ridge_lambda": strength,
                                      "shuffled_h0_array_row": eval_rows[eval_permutation[i]],
                                      "wrong_target_array_row": eval_rows[wrong_permutation[i]]}
                            for method in METHODS:
                                for q, value in zip(METRICS, scalar_values[method][i]):
                                    record[f"{method}_{q}"] = format(value, ".17g")
                            writer.writerow(record)
                        for method, baseline in CONTRASTS:
                            contrast = f"{method}_minus_{baseline}"
                            for i, row in enumerate(eval_rows):
                                gkey = (strength, contrast, name, concepts[row])
                                accumulator = grouped.setdefault(gkey, [0.0] * 7)
                                xrow, brow = scalar_values[method][i], scalar_values[baseline][i]
                                increment = [1, brow[0], xrow[0], brow[0]-xrow[0],
                                             xrow[1]-brow[1], xrow[3]-brow[3], xrow[2]-brow[2]]
                                for j, value in enumerate(increment):
                                    accumulator[j] += value
                    unit = {"subject": name, "role": role, "direction": direction,
                            "fit_queries": len(fit_rows), "held_queries": len(eval_rows),
                            "fit_rows_sha256": hashlib.sha256(json.dumps(fit_rows).encode()).hexdigest(),
                            "held_rows_sha256": hashlib.sha256(json.dumps(eval_rows).encode()).hexdigest(),
                            "coefficient_sha256": coefficient_hashes,
                            "contrasts": {k: summarize_contrasts(torch, v) for k, v in metrics_by_lambda.items()},
                            "resource": budget.check()}
                    completed_units.append(unit)
                    metric_stream.flush()
                    json_write(output/"progress.json", {"status": "RUNNING_T_ONLY_CPU_PROBES",
                        "units_completed": len(completed_units), "expected_units": 40,
                        "covered_queries_per_lambda": {k: len(v) for k, v in coverage.items()},
                        "expected_queries_per_lambda": 132300, "last_unit": unit,
                        "resource": budget.check(), "generated_images": 0, "GPU_used": False})
                    print(json.dumps({"unit": len(completed_units), "subject": name, "role": role,
                                      "direction": direction, **budget.check()}), flush=True)
                del taps
            del raw
    assert len(completed_units) == 40 and len(numerical_checks) == 20
    assert all(values == expected_keys for values in coverage.values())
    assert not torch.cuda.is_initialized()
    aggregate_file = output/"paired_concept_subject.csv.gz"
    with gzip.open(aggregate_file, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ridge_lambda", "contrast", "subject", "concept_id", "queries",
                         "baseline_mse_sum", "method_mse_sum", "MSE_reduction_sum",
                         "cosine_correct_gain_sum", "cosine_margin_gain_sum", "cosine_wrong_gain_sum"])
        for key in sorted(grouped):
            writer.writerow(list(key) + grouped[key])
    summary = {}
    subject_summary = {}
    for strength in map(str, config["ridge_lambdas"]):
        summary[strength] = {}
        subject_summary[strength] = {}
        for method, baseline in CONTRASTS:
            contrast = f"{method}_minus_{baseline}"
            subject_summary[strength][contrast] = {}
            total = [0.0]*7
            for subject in expected_subjects:
                v = [0.0]*7
                for (alpha, item, name, concept), values in grouped.items():
                    if (alpha, item, name) == (strength, contrast, subject):
                        v = [a+b for a, b in zip(v, values)]
                assert v[0] == 13230
                row = {"queries": int(v[0]), "mean_MSE_reduction": v[3]/v[0],
                       "relative_MSE_reduction": v[3]/max(v[1], 1e-12),
                       "mean_cosine_correct_gain": v[4]/v[0],
                       "mean_cosine_margin_gain": v[5]/v[0]}
                subject_summary[strength][contrast][subject] = row
                total = [a+b for a, b in zip(total, v)]
            assert total[0] == 132300
            summary[strength][contrast] = {
                "queries": 132300, "mean_MSE_reduction": total[3]/total[0],
                "relative_MSE_reduction": total[3]/max(total[1], 1e-12),
                "mean_cosine_correct_gain": total[4]/total[0],
                "mean_cosine_margin_gain": total[5]/total[0],
                "subjects_joint_positive": sum(
                    r["mean_MSE_reduction"] > 0 and r["mean_cosine_margin_gain"] > 0
                    for r in subject_summary[strength][contrast].values())}
    decision = decide(summary, config["thresholds"], list(map(str, config["ridge_lambdas"])))
    branches = {"beyond_preLN_statistics": decide(summary, config["thresholds"],
                   list(map(str, config["ridge_lambdas"])), ("c_h0_minus_c_prestats",)),
                "preLN_statistics": decide(summary, config["thresholds"],
                   list(map(str, config["ridge_lambdas"])), ("c_prestats_minus_c_cstats",))}
    result = {"schema": SCHEMA, "status": "COMPLETE_T_ONLY_CPU_HELD_CONCEPT_PROBES",
              "config_sha256": file_sha(config_path), "source_sha256": file_sha(__file__),
              "queries_per_lambda": 132300, "subjects": 10, "concepts": 1323,
              "encoder_roles": 20, "probe_units": 40, "ridge_lambdas": config["ridge_lambdas"],
              "thresholds": config["thresholds"], "contrasts": summary,
              "per_subject": subject_summary, "main_decision": decision, "branches": branches,
              "numerical_checks": numerical_checks, "units": completed_units,
              "verified_input_SHA": inputs.verified, "resource": budget.check(),
              "GPU_used": False, "new_EEG_encoder_fit": False, "generated_images": 0,
              "no_V": True, "no_D": True, "no_test": True,
              "V_or_test_scores_read": False, "paired_intervals_computed": False,
              "limits": ["Fixed linear readouts do not prove strict information loss.",
                         "hpre gains can reflect per-query scale/mean.",
                         "OOF role probes do not establish full-encoder deployment transfer.",
                         "No image benefit, perceptual quality or SOTA conclusion."]}
    json_write(output/"summary.json", result)
    json_write(output/"progress.json", {"status": result["status"], "units_completed": 40,
               "expected_units": 40, "covered_queries_per_lambda": {k: len(v) for k, v in coverage.items()},
               "expected_queries_per_lambda": 132300, "main_decision": decision,
               "branches": branches,
               "GPU_used": False, "no_V": True, "no_D": True, "no_test": True,
               "resource": budget.check()})
    json_write(output/"completion.json", {"status": result["status"], "queries_per_lambda": 132300,
               "summary": {"path": str(output/"summary.json"), "sha256": file_sha(output/"summary.json")},
               "per_query": {"path": str(metric_path), "sha256": file_sha(metric_path)},
               "paired_concept_subject": {"path": str(aggregate_file), "sha256": file_sha(aggregate_file)},
               "main_decision": decision, "no_V": True, "no_D": True, "no_test": True,
               "branches": branches,
               "GPU_used": False, "resource": budget.check()})
    return result


def self_check():
    from check_BrainHIVE_EEG_layers_T_CPU_20261025_v1 import run_checks
    return run_checks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert not args.config and not args.output
        self_check()
        return
    assert args.config and args.output, "--config and --output required"
    try:
        run(args.config, args.output)
    except Exception as exc:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        json_write(output/"failure.json", {"status": "UNKNOWN_TECHNICAL_FAILURE",
                   "main_decision": {"status": "UNKNOWN", "automatic_new_candidate_authorized": False},
                   "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "automatic_new_candidate_authorized": False})
        raise


if __name__ == "__main__":
    main()
