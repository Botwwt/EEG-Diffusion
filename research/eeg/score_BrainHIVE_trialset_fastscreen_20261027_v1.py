"""Locked DINOv3-S/16 score for the completed single-subject T80 image pilot.

Exactly 160 generated images and 80 distinct targets, each encoded once.  The
80-image global gallery and ten-image same-concept galleries are diagnostic
screens, never the full-V gallery.  Eight-concept bootstrap intervals describe
this fixed pilot and do not establish training stability or SOTA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
SCHEMA = "BrainHIVE_trialset_fastscreen_DINO_T80_20261027_v1"
GEN_SCHEMA = "BrainHIVE_trialset_fastscreen_native_T80_20261027_v1"
ARMS = ("mean", "each_mean")
DINO_SHA256 = "08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d"
MAX_SECONDS = 600
OUTPUT_CAP = 64 << 20


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    assert not path.exists() and not pending.exists(), "Never overwrite pilot evidence"
    with pending.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def checked_ref(ref):
    path = Path(ref["path"])
    assert path.is_file() and sha(path) == ref["sha256"], f"SHA mismatch: {path}"
    return path


def validate_inputs(completion_path, check_pixels=True):
    completion_path = Path(completion_path).resolve()
    complete = read(completion_path)
    assert complete["status"] == "COMPLETE_FIXED_T80_TWO_ARM_NATIVE_IMAGES"
    assert complete["schema"] == GEN_SCHEMA
    assert complete["queries"] == 80 and complete["images"] == 160
    assert complete["arms"] == list(ARMS)
    assert complete["per_arm_images"] == {arm: 80 for arm in ARMS}
    assert complete["native_NFE_per_image"] == 4 and complete["resolution"] == [512, 512]
    assert complete["no_best_of_N"] is True and complete["normal_inference_target_access"] is False
    assert complete["target_pixels_loaded_after_generation"] is True
    manifest = checked_ref(complete["manifest"])
    checked_ref(complete["inference_manifest"])
    panel_path = checked_ref(complete["panel"])
    checked_ref(complete["input_provenance"])
    panel = read(panel_path)
    assert panel["status"] == "FIXED_BLIND_SINGLE_SUBJECT_T80_PANEL" and panel["expected_queries"] == 80
    panel_queries = {row["query_id"]: row for row in panel["queries"]}
    assert len(panel_queries) == 80
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 160
    grouped = {}
    generated_paths = set()
    generated_inodes = set()
    for row in rows:
        assert row["arm"] in ARMS and row["query_id"] in panel_queries
        assert row["target_mapping_attached_after_inference"] is True
        assert row["normal_inference_target_access"] is False and row["oracle_normalization"] is False
        assert row["one_image_per_query"] is True and row["repeat_protocol"] == "R4"
        assert row["UNet_NFE"] == 4
        query = panel_queries[row["query_id"]]
        for key in ("row", "source_row", "array_row", "subject_id", "concept_id", "stimulus_id", "fold", "image_seed"):
            assert row[key] == query[key], f"Panel identity mismatch in {key}"
        group = grouped.setdefault(row["query_id"], {})
        assert row["arm"] not in group, "Duplicate arm/query"
        group[row["arm"]] = row
        path = Path(row["generated_path"]).resolve()
        assert str(path) not in generated_paths, "Cross-arm generated path reuse"
        generated_paths.add(str(path))
        info = path.stat()
        assert (info.st_dev, info.st_ino) not in generated_inodes, "Cross-arm hardlink reuse"
        generated_inodes.add((info.st_dev, info.st_ino))
        assert sha(path) == row["generated_sha256"]
        assert Path(row["target_image_path"]).resolve() == Path(row["score_target_path"]).resolve()
        if check_pixels:
            with Image.open(path) as image:
                assert image.size == (512, 512)
                pixels = np.asarray(image.convert("RGB"))
                assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["generated_pixel_sha256"]
    assert set(grouped) == set(panel_queries)
    queries = sorted(panel_queries.values(), key=lambda row: (int(row["fold"]), str(row["concept_id"]), int(row["array_row"])))
    assert len({row["subject_id"] for row in queries}) == 1
    assert len({row["stimulus_id"] for row in queries}) == 80
    assert len({row["array_row"] for row in queries}) == 80
    concepts = {}
    targets = []
    ordered = []
    for query in queries:
        group = grouped[query["query_id"]]
        assert set(group) == set(ARMS)
        first, second = group["mean"], group["each_mean"]
        for key in ("target_image_path", "target_image_sha256", "initial_noise_latents_sha256", "initial_UNet_input_sha256"):
            assert first[key] == second[key], f"Paired identity/noise mismatch: {key}"
        target = Path(first["target_image_path"]).resolve()
        assert sha(target) == first["target_image_sha256"]
        if check_pixels:
            with Image.open(target) as image:
                assert min(image.size) > 0
                image.convert("RGB").load()
        targets.append(target)
        concepts.setdefault(query["concept_id"], []).append(query)
        ordered.extend(group[arm] for arm in ARMS)
    assert len({str(path) for path in targets}) == 80
    assert len(concepts) == 8 and all(len(group) == 10 for group in concepts.values())
    assert {fold: len([group for group in concepts.values() if group[0]["fold"] == fold]) for fold in (0, 1)} == {0: 4, 1: 4}
    assert all(len({row["fold"] for row in group}) == 1 for group in concepts.values())
    return complete, manifest, panel_path, queries, ordered, targets


def metric_arrays(query_features, target_features, queries):
    from locked_metrics import fixed_gallery_metrics
    ids = [query["stimulus_id"] for query in queries]
    global_scores = fixed_gallery_metrics(query_features, target_features, ids, ids)
    result = {"target_cosine": global_scores["correct_cosine"].numpy().astype(np.float64),
              "global80_two_afc": global_scores["two_afc"].numpy().astype(np.float64),
              "global80_top1": global_scores["top1"].numpy().astype(np.float64)}
    within_top1, within_afc = np.zeros(len(queries)), np.zeros(len(queries))
    concepts = {}
    for position, query in enumerate(queries):
        concepts.setdefault(query["concept_id"], []).append(position)
    for positions in concepts.values():
        selected = torch.tensor(positions, dtype=torch.long)
        local_ids = [ids[position] for position in positions]
        local = fixed_gallery_metrics(query_features[selected], target_features[selected], local_ids, local_ids)
        within_top1[positions] = local["top1"].numpy()
        within_afc[positions] = local["two_afc"].numpy()
    result["same_concept10_top1"] = within_top1
    result["same_concept10_two_afc"] = within_afc
    return result


def paired_bootstrap(metrics, queries, repetitions=1000, seed=1729):
    names = list(metrics["mean"])
    concepts = sorted({query["concept_id"] for query in queries})
    positions = {concept: [i for i, query in enumerate(queries) if query["concept_id"] == concept] for concept in concepts}
    differences = {name: metrics["each_mean"][name] - metrics["mean"][name] for name in names}
    group_values = np.asarray([[differences[name][positions[concept]].mean() for name in names] for concept in concepts])
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(concepts), size=(repetitions, len(concepts)))
    values = group_values[sampled].mean(1)
    return {name: {"each_mean_minus_mean": float(differences[name].mean()),
                   "descriptive_concept_bootstrap95": np.quantile(values[:, index], [.025, .975]).tolist(),
                   "positive_concepts": int((group_values[:, index] > 0).sum()),
                   "concepts": len(concepts)} for index, name in enumerate(names)}


def feature_statistics(generated_features, target_features, queries):
    assert generated_features.shape == (160, 768) and target_features.shape == (80, 768)
    assert torch.isfinite(generated_features).all() and torch.isfinite(target_features).all()
    assert (generated_features.norm(dim=1) > 0).all() and (target_features.norm(dim=1) > 0).all()
    metrics = {arm: metric_arrays(generated_features[index::2], target_features, queries) for index, arm in enumerate(ARMS)}
    self_metrics = metric_arrays(target_features, target_features, queries)
    assert np.allclose(self_metrics["target_cosine"], 1, atol=2e-5)
    assert np.all(self_metrics["global80_two_afc"] >= 1 - 1e-6), "Target self retrieval failed or target features are duplicated"
    # Fixed feature / fixed gallery control; no additional model/image calls.
    rng = np.random.default_rng(1729)
    permutation = rng.permutation(80)
    while np.any(permutation == np.arange(80)):
        permutation = rng.permutation(80)
    permuted = metric_arrays(target_features[torch.tensor(permutation)], target_features, queries)
    return metrics, {"original_selfcheck": {key: float(value.mean()) for key, value in self_metrics.items()},
                     "deterministic_derangement": permutation.tolist(),
                     "deranged_original_target_labels": {key: float(value.mean()) for key, value in permuted.items()},
                     "permutation_is_diagnostic_not_a_significance_test": True}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-registration", type=Path)
    parser.add_argument("--gpu-job-name")
    parser.add_argument("--paused-host-proof", type=Path)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    assert 0 < args.max_seconds <= MAX_SECONDS and 0 < args.batch <= 32
    invoked = time.monotonic()
    complete, manifest, panel_path, queries, ordered, targets = validate_inputs(args.completion)
    completion_sha, manifest_sha = sha(args.completion), sha(manifest)
    if args.preflight_only:
        print(json.dumps({"status": "CPU_COMPLETE_PAIRED_T80_INPUTS_VERIFIED", "queries": 80, "images": 160, "GPU_started": False}), flush=True)
        return 0
    assert args.native_registration and args.gpu_job_name
    assert not args.output.exists(), "Preserve every prior scoring attempt"
    args.output.mkdir(parents=True)
    try:
        import brainhive_paired_study_contract_20261025_v1 as C
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        config = C.read(args.native_registration)
        C.validate(config)
        C.setup_runtime(config)
        D.torch = torch
        document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds,
                        maximum_output_bytes=OUTPUT_CAP, maximum_extra_CPU_bytes=10 << 30,
                        maximum_GPU_bytes=20 << 30, minimum_CPU_headroom_bytes=10 << 30,
                        resource_authorization=config["resource_authorization"])
        if args.paused_host_proof:
            from run_gpu_job_BrainHIVE_paused_host_20261027_v1 import verify_wrapper
            wrapper = verify_wrapper(document, args.paused_host_proof)
        else:
            wrapper = D.verify_wrapper(document)
        budget = D.Budget(document)
        stopped = [False]
        signal.signal(signal.SIGTERM, lambda signum, frame: stopped.__setitem__(0, True))
        def check():
            assert not stopped[0] and time.monotonic() - invoked < args.max_seconds, "Finite scoring stop"
            budget.check()
            assert sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) <= OUTPUT_CAP

        torch.set_num_threads(2)
        torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        import image_evaluation as E
        assert E.DINO_SHA256 == DINO_SHA256
        assert E.DINO_WEIGHT.is_file(), "Existing locked DINO weight required; never download"
        provenance = dict(schema=SCHEMA, source={"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
                          generation_completion={"path": str(args.completion.resolve()), "sha256": completion_sha},
                          generation_manifest={"path": str(manifest.resolve()), "sha256": manifest_sha},
                          panel={"path": str(panel_path.resolve()), "sha256": sha(panel_path)},
                          image_evaluation_source_sha256=sha(ROOT / "image_evaluation.py"),
                          locked_metrics_source_sha256=sha(ROOT / "locked_metrics.py"),
                          DINO_weight={"path": str(E.DINO_WEIGHT), "sha256": DINO_SHA256},
                          dino_features="layer11 normalized CLS + mean patch, concatenated and L2-normalized; 256 bilinear antialias; historical real CPU / generated CUDA resize routes",
                          all_generated_encoded_once=True, all_targets_encoded_once=True,
                          image_counts={"generated": 160, "targets": 80}, wrapper=wrapper,
                          gallery="fixed 80 targets; separate same-concept ten targets", development_screen_only=True)
        atomic_json(args.output / "provenance.json", provenance)
        check()
        model = E.dino_model()
        check()
        def extract(paths, real_images):
            features = []
            for start in range(0, len(paths), args.batch):
                check()
                features.append(E.dino_features(paths[start:start + args.batch], model, batch_size=args.batch, real_images=real_images))
            return torch.cat(features).float()
        generated_features = extract([Path(row["generated_path"]) for row in ordered], False)
        target_features = extract(targets, True)
        del model
        torch.cuda.empty_cache()
        check()
        metrics, controls = feature_statistics(generated_features, target_features, queries)
        paired = paired_bootstrap(metrics, queries)
        feature_path = args.output / "DINO_features.pt"
        torch.save(dict(generated=generated_features, targets=target_features,
                        generated_keys=[(row["query_id"], row["arm"]) for row in ordered],
                        target_stimulus_ids=[query["stimulus_id"] for query in queries],
                        DINO_weight_sha256=DINO_SHA256), feature_path)
        per_query = args.output / "per_query.csv"
        names = list(metrics["mean"])
        with per_query.open("x", newline="", encoding="utf-8") as stream:
            fields = ["query_id", "subject_id", "array_row", "concept_id", "stimulus_id", "fold", "arm", "generated_path", "generated_sha256", "target_image_path", "target_image_sha256", *names]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, query in enumerate(queries):
                for arm_index, arm in enumerate(ARMS):
                    row = ordered[index * 2 + arm_index]
                    writer.writerow({**{key: row[key] for key in fields if key not in names},
                                     **{name: float(metrics[arm][name][index]) for name in names}})
        concept_table = {concept: {arm: {name: float(values[[i for i, query in enumerate(queries) if query["concept_id"] == concept]].mean())
                                      for name, values in metrics[arm].items()} for arm in ARMS}
                         for concept in sorted({query["concept_id"] for query in queries})}
        assert sha(args.completion) == completion_sha and sha(manifest) == manifest_sha
        for row in ordered:
            assert sha(row["generated_path"]) == row["generated_sha256"]
        for index, target in enumerate(targets):
            assert sha(target) == ordered[index * 2]["target_image_sha256"]
        check()
        summary = dict(status="COMPLETE_FIXED_T80_TWO_ARM_DINO_SCREEN", schema=SCHEMA,
                       subjects=1, concepts=8, queries=80, generated_images=160, distinct_target_images=80,
                       rows_per_arm=80, full_registered_panel_denominator=True,
                       means={arm: {name: float(values.mean()) for name, values in metrics[arm].items()} for arm in ARMS},
                       paired_each_mean_minus_mean=paired, per_concept=concept_table, controls=controls,
                       bootstrap=dict(seed=1729, draws=1000, resampling_unit="concept (eight groups of ten paired queries)",
                                      gallery="original fixed 80 targets; bootstrap does not rebuild gallery", descriptive_only=True),
                       original_generation_NFE=4, original_generation_resolution=[512, 512],
                       no_best_of_N=True, full_V_evaluation=False, SOTA_established=False,
                       claim_limit="single-subject eight-concept T-only development image screen; no full16600/full-V, mechanism, independent confirmation, or SOTA conclusion",
                       generated_features_shape=list(generated_features.shape), target_features_shape=list(target_features.shape),
                       features={"path": str(feature_path.resolve()), "sha256": sha(feature_path)},
                       per_query={"path": str(per_query.resolve()), "sha256": sha(per_query)},
                       provenance_sha256=sha(args.output / "provenance.json"), seconds=time.monotonic() - invoked,
                       peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes, peak_GPU_bytes=budget.peak_GPU_bytes)
        atomic_json(args.output / "summary.json", summary)
        atomic_json(args.output / "completion.json", dict(status=summary["status"], schema=SCHEMA,
                    summary={"path": str((args.output / "summary.json").resolve()), "sha256": sha(args.output / "summary.json")},
                    generation_completion_sha256=completion_sha, generation_manifest_sha256=manifest_sha,
                    expected_images=160, observed_images=160, expected_targets=80, observed_targets=80,
                    per_query_sha256=sha(per_query), features_sha256=sha(feature_path), seconds=time.monotonic() - invoked))
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        atomic_json(args.output / "failure.json", dict(status="FAILED_FIXED_DENOMINATOR_NO_AUTOMATIC_RETRY",
                    error=repr(error), traceback=traceback.format_exc(), expected_images=160, expected_targets=80,
                    seconds=time.monotonic() - invoked, no_effect_claim=True))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
