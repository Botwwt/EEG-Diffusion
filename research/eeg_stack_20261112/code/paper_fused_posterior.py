"""Infer a whole-photo condition with a training-photo prior and measured EEG consistency."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_forward_guidance import REPEATS, binned_eeg, forward_weights
from paper_fused_prior import from_coordinates, load_prior, to_coordinates
from paper_oof_conditions import SOURCES, concept_splits, registered_rows, study_root
from paper_render_cache import PinnedFiles, require, save_json


ANCHOR_WEIGHT = .35
SCHEDULE = (1., .7, .5, .35, .25, .15)
MEASUREMENT_WEIGHTS = (.3, 1., 3., 10.)
MODES = ("linear", "gaussian", "learned")
ARMS = ("native-paper", "linear", "gaussian", "learned", "learned-prior-only", "learned-wrong")


def data_term(predictor, observed):
    return predictor @ predictor.T, observed @ predictor.T


def anchored_reconstruction(original, gram, rhs, strength, mode, denoiser=None):
    """Alternate a visual estimate and a quadratic update using the measured EEG."""
    require(mode in MODES and strength >= 0 and (mode != "learned" or denoiser is not None),
            "Invalid visual prior or measurement weight")
    solution = original
    data_matrix = gram * strength
    data_matrix.diagonal().add_(1.)
    factor = torch.linalg.cholesky(data_matrix)
    anchor = ANCHOR_WEIGHT * original + strength * rhs
    for sigma in SCHEDULE:
        if mode == "gaussian":
            prior = solution / (1. + sigma ** 2)
        elif mode == "learned":
            current = solution.new_full((len(solution),), sigma)
            with torch.autocast(solution.device.type, dtype=torch.bfloat16, enabled=solution.is_cuda):
                prior = denoiser(solution, current).float()
        else:
            prior = solution
        proposal = anchor + (1. - ANCHOR_WEIGHT) * prior
        solution = torch.cholesky_solve(proposal.T.contiguous(), factor).T
    return solution


def measure_observation(observed, weights):
    predictor = weights["weighted"] / weights["visual_scale"][:, None]
    standard = (observed - weights["eeg_center"]) / weights["eeg_scale"]
    white = torch.linalg.solve_triangular(weights["noise_factor"], standard.T, upper=False).T
    return data_term(predictor, white + weights["visual_center"] @ predictor)


def limit_feature_update(original, predicted, center):
    update = predicted - original
    radius = ((original - center).norm(dim=1, keepdim=True) * .85).clamp_min(1e-4)
    return original + update / (update.norm(dim=1, keepdim=True) / radius).clamp_min(1.)


def summarize_trials(data, rows):
    eeg = binned_eeg(data, rows, REPEATS)
    return eeg.mean(1), (eeg[:, :2].mean(1) - eeg[:, 2:].mean(1)) * .5


@torch.inference_mode()
def calibrate(source_root):
    """Lock a single measurement weight on four concept-held training folds."""
    from common import PREP
    from paper_object_margin_encoder import visual_transfer
    from paper_render import physical_eeg

    root = study_root()
    files = PinnedFiles()
    split_sha = files.pin(PREP / "paper_split.json")
    photo_path = PREP / "F_train.npy"
    photo_sha = files.pin(photo_path)
    rows, labels, folds = registered_rows(PREP)
    dino_path = source_root / "object-identity/training-photo-dino.pt"
    dino_sha = files.pin(dino_path)
    dino = torch.load(dino_path, map_location="cpu", weights_only=True)
    require(dino["proof"]["split"] == split_sha and np.array_equal(dino["rows"].numpy(), rows)
            and np.array_equal(dino["concepts"][dino["labels"]].numpy(), labels),
            "Visual calibration features disagree with the registered training photos")
    photos = torch.from_numpy(np.load(photo_path, allow_pickle=False, mmap_mode="r")[rows].copy()).cuda().float()
    photo_dino = dino["image_features"].cuda().float()
    object_sums, ids = dino["sums"].cuda(), dino["labels"].cuda()
    measurements, sources = {}, {}
    for subject in SOURCES:
        raw, proof = physical_eeg(subject, "train", files)
        file = source_root / f"conditions-sub-{subject:02d}.pt"
        condition_sha = files.pin(file)
        cache = torch.load(file, map_location="cpu", weights_only=True)
        require(cache["sources"]["subject"] == subject and cache["sources"]["split"] == split_sha
                and cache["sources"]["images"] == photo_sha
                and cache["sources"]["eeg"]["array_sha256"] == proof["array_sha256"]
                and np.array_equal(cache["rows"].numpy(), rows) and np.array_equal(cache["folds"].numpy(), folds),
                "Native out-of-fold conditions are paired to other input bytes")
        observed, gaps = summarize_trials(raw, rows)
        measurements[subject] = dict(observed=observed, gaps=gaps, original=cache["out_of_fold"].cuda().float())
        sources[subject] = dict(physical_eeg=proof, oof_native_sha256=condition_sha)
    prior_proofs, synthetic, fold_metrics = {}, {}, {}
    totals = {"native": 0, "learned-prior-only": 0,
              **{f"{mode}/{lam}": 0 for mode in MODES for lam in MEASUREMENT_WEIGHTS}}
    per_person = {s: totals.copy() for s in SOURCES}
    total_target_cos = {k: 0. for k in totals}
    for held, tr, va in concept_splits(labels, folds):
        basis, prior, digest = load_prior(root, held, "cuda", files, split_sha, photo_sha)
        prior_proofs[held] = digest
        raw_target = photos[va]
        target = to_coordinates(raw_target, basis)
        target_train = to_coordinates(photos[tr], basis)
        center, visual_center, lift = visual_transfer(photos[tr], photo_dino[tr])
        classes, inverse = torch.unique(ids[va], return_inverse=True)
        negatives = F.normalize(F.normalize(object_sums[classes]) - visual_center)
        positives = F.normalize(F.normalize(object_sums[ids[va]] - photo_dino[va]) - visual_center)

        def measure(features):
            estimate = F.normalize((features - center) @ lift)
            other = (estimate @ negatives.T).scatter_(1, inverse[:, None], float("-inf"))
            hit = (estimate * positives).sum(1) > other.max(1).values
            return int(hit.sum()), float(F.cosine_similarity(features - center, raw_target - center).sum())

        sigma = .7
        rng = torch.Generator(device="cuda").manual_seed(1440 + held)
        noisy = target + sigma * torch.randn(target.shape, generator=rng, device="cuda")
        gaussian = noisy / (1. + sigma**2)
        learned = prior(noisy, noisy.new_tensor([sigma]))
        synthetic[held] = dict(n=len(va), sigma=sigma, gaussian_mse=float((gaussian - target).square().mean()),
                               learned_mse=float((learned - target).square().mean()))
        fold_values = {}
        for subject in SOURCES:
            entry = measurements[subject]
            fit = forward_weights(target_train, entry["observed"][tr], entry["gaps"][tr])
            gram, rhs = measure_observation(entry["observed"][va], fit)
            original = entry["original"][va]
            coords = to_coordinates(original, basis)
            results = {"native": measure(original)}
            trials = [(f"{mode}/{lam}", mode, lam) for mode in MODES for lam in MEASUREMENT_WEIGHTS]
            trials.append(("learned-prior-only", "learned", 0.))
            for key, mode, strength in trials:
                state = anchored_reconstruction(coords, gram, rhs, strength, mode, prior)
                features = limit_feature_update(original, from_coordinates(state, basis), basis["center"])
                results[key] = measure(features)
            for key, (hit, centered_cos) in results.items():
                totals[key] += hit
                per_person[subject][key] += hit
                total_target_cos[key] += centered_cos
            fold_values[subject] = {key: value[0] for key, value in results.items()}
            print("photo posterior calibration", held, subject, fold_values[subject], flush=True)
        fold_metrics[held] = fold_values
    best = max(MEASUREMENT_WEIGHTS, key=lambda lam: (totals[f"learned/{lam}"], -lam))
    n = len(rows) * len(SOURCES)
    files.verify()
    output = dict(n=n, people=list(SOURCES), calibration_mode="whole-concept-held DINO leave-one-photo-out prototypes",
                  anchor_weight=ANCHOR_WEIGHT, schedule=list(SCHEDULE), measurement_weights=list(MEASUREMENT_WEIGHTS),
                  selected_strength=best, correct=totals, per_person=per_person, fold_metrics=fold_metrics,
                  centered_cos={k: value / n for k, value in total_target_cos.items()}, synthetic_corruption=synthetic,
                  sources=sources, prior_sha256=prior_proofs, split_sha256=split_sha, photo_sha256=photo_sha,
                  dino_sha256=dino_sha, code_sha256=files.pin(Path(__file__)),
                  forward_code_sha256=files.pin(Path(inspect.getfile(forward_weights))))
    save_json(root / "calibration.json", output)
    print("photo posterior calibration total", totals, "selected", best, flush=True)


@torch.inference_mode()
def render(subject, arms):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_render import condition, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_token_evidence import project_tokens, render_arms, score_outputs
    from paper_visual_ceiling import official_features

    root = study_root()
    files = PinnedFiles()
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    rows = json.loads(split.read_bytes())["official_rows"]
    require(rows == list(range(200)), "Use the 200 official stimulus rows")
    locked = root / "calibration.json"
    lock_sha = files.pin(locked)
    calibration = json.loads(locked.read_bytes())
    code_sha = files.pin(Path(__file__))
    forward_sha = files.pin(Path(inspect.getfile(forward_weights)))
    photo_path = PREP / "F_train.npy"
    photo_sha = files.pin(photo_path)
    require((calibration["split_sha256"], calibration["photo_sha256"], calibration["code_sha256"],
             calibration["forward_code_sha256"]) == (split_sha, photo_sha, code_sha, forward_sha)
            and calibration["people"] == list(SOURCES) and subject not in SOURCES,
            "Calibration must be locked before inference on the independent participant")
    strength = calibration["selected_strength"]
    require(strength in MEASUREMENT_WEIGHTS, "Invalid locked measurement weight")
    basis, denoiser, prior_sha = load_prior(root, 0, "cuda", files, split_sha, photo_sha)
    train_rows, _, _ = registered_rows(PREP)
    train, training_proof = physical_eeg(subject, "train", files)
    eeg, physical = physical_eeg(subject, "test", files)
    train_photo = torch.from_numpy(np.load(photo_path, allow_pickle=False, mmap_mode="r")[train_rows].copy()).cuda().float()
    observed, gaps = summarize_trials(train, train_rows)
    measurement = forward_weights(to_coordinates(train_photo, basis), observed, gaps)
    selected, _ = summarize_trials(eeg, rows)
    gram, rhs = measure_observation(selected, measurement)
    repeats = groups("official-r4")
    require(repeats == [REPEATS], "Measurement and native encoder must use the same four trials")
    native, checkpoint = condition("native-paper", subject, eeg[rows], repeats, features_only=True, files=files)
    native = native.cuda().float()
    coords = to_coordinates(native, basis)
    order = np.arange(len(rows))
    wrong = np.roll(order, len(order) // 2)
    seeds = [query_seed("official", index) for index in order]
    controls = {
        "linear": ("linear", strength, order),
        "gaussian": ("gaussian", strength, order),
        "learned": ("learned", strength, order),
        "learned-prior-only": ("learned", 0., order),
        "learned-wrong": ("learned", strength, wrong),
    }
    generator_assets = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipeline
    shared = dict(subject=subject, budget="official-r4", repeats=repeats, split_sha256=split_sha,
                  physical_eeg=physical, fit_eeg=training_proof, native_checkpoint_sha256=checkpoint,
                  generator_files_sha256=generator_assets, photo_features_sha256=photo_sha,
                  photo_prior_sha256=prior_sha, calibration_sha256=lock_sha,
                  code_sha256=code_sha, forward_code_sha256=forward_sha,
                  render_source_sha256=files.pin(Path(inspect.getfile(draw_images))))
    variants, scored_features = [], {}
    for arm in arms:
        if arm == "native-paper":
            features = native
            details = dict(visual_prior=None, measurement_strength=0., measurement_donor=None, posterior_steps=0)
        else:
            mode, weight, donor = controls[arm]
            updated = anchored_reconstruction(coords, gram, rhs[donor], weight, mode, denoiser)
            features = limit_feature_update(native, from_coordinates(updated, basis), basis["center"])
            details = dict(visual_prior=mode, measurement_strength=weight,
                           measurement_donor=donor.tolist() if weight else None, posterior_steps=len(SCHEDULE))
        require(torch.isfinite(features).all(), "Invalid render condition")
        variables = project_tokens(features, adapter, projection).cpu().half()
        variants.append((arm, variables, details))
        scored_features[arm] = features
    destinations = render_arms(root / "images" / f"sub-{subject:02d}", variants, rows, seeds, shared, files, pipeline)
    del pipeline, adapter, projection
    torch.cuda.empty_cache()
    score_outputs(destinations, "official")
    # Only the independent scoring phase opens any held reference features.
    reference, reference_rows, gallery_proof = official_features(PREP, files)
    require(reference_rows == rows, "Wrong official feature row registration")
    reference = F.normalize(reference.cuda() - basis["center"])
    numeric = {}
    for arm, features in scored_features.items():
        scores = F.normalize(features - basis["center"]) @ reference.T
        numeric[arm] = dict(centered_visual_cosine=float(scores.diagonal().mean()),
                            exact_visual=int((scores.argmax(1) == torch.arange(len(rows), device="cuda")).sum()),
                            rms_update=float((features - native).square().mean().sqrt()))
    files.verify()
    save_json(root / f"pre-gen-sub-{subject:02d}.json", dict(n=len(rows), scores=numeric, gallery=gallery_proof))
    print("whole-photo feature probe", subject, numeric, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    calibration = actions.add_parser("calibrate")
    calibration.add_argument("--source-root", type=Path, required=True)
    inference = actions.add_parser("render")
    inference.add_argument("subject", type=int, choices=range(5, 11))
    inference.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    if args.action == "calibrate":
        calibrate(args.source_root)
    else:
        render(args.subject, args.arms)
