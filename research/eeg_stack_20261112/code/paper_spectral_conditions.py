"""Fit equal-budget EEG visual readouts for local repeat spectral moments."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_oof_conditions import SOURCES, registered_rows, study_root
from paper_render_cache import PinnedFiles, require, save_json
import paper_trial_spectra
from paper_trial_spectra import MOMENTS, REPEATS, read_spectra


WAVE_WEIGHTS = (0., .25, .5)
MOMENT_WEIGHTS = (.25, .5, 1.)
ARMS = ("native-paper", "wave", *MOMENTS, "residual-wrong", "cross-wrong")


def fit_normalization(x):
    center = x.mean(0)
    return center, x.std(0).clamp_min(1e-5)


def standardize(x, state):
    return (x - state[0]) / state[1]


def ridge(inputs, target, strength):
    gram = inputs.T @ inputs / len(inputs)
    gram.diagonal().add_(strength * gram.diagonal().mean().clamp_min(1e-8))
    return torch.linalg.solve(gram, inputs.T @ target / len(inputs))


def fit_readouts(features, photos):
    mean = features["mean"]
    norm = fit_normalization(mean)
    base = standardize(mean, norm)
    photo_mean = photos.mean(0)
    photo = photos - photo_mean
    wave_matrix = ridge(base, photo, .3)
    err = photo - base @ wave_matrix
    state = dict(wave_norm=norm, photo_mean=photo_mean, wave_matrix=wave_matrix, moments={})
    for k in MOMENTS:
        feature = features[k]
        scale = feature.abs().median(0).values.clamp_min(1e-8)
        transformed = (feature / scale).asinh()
        stats = fit_normalization(transformed)
        power = standardize(transformed, stats)
        expected = ridge(base, power, .3)
        remainder = power - base @ expected
        matrix = ridge(remainder, err, .8)
        state["moments"][k] = dict(input_scale=scale, norm=stats, expected=expected, matrix=matrix)
    return state


def readouts(features, state):
    base = standardize(features["mean"], state["wave_norm"])
    results = {"wave": state["photo_mean"] + base @ state["wave_matrix"]}
    for k, pack in state["moments"].items():
        power = standardize((features[k] / pack["input_scale"]).asinh(), pack["norm"])
        results[k] = (power - base @ pack["expected"]) @ pack["matrix"]
    require(all(torch.isfinite(x).all() for x in results.values()), "Invalid spectral readout")
    return results


def bounded(original, adjusted, photo_mean):
    dev = adjusted - original
    bound = .85 * (original - photo_mean).norm(dim=1, keepdim=True).clamp_min(1e-6)
    return original + dev / (dev.norm(dim=1, keepdim=True) / bound).clamp_min(1.)


@torch.inference_mode()
def screen(prep, eeg_root, photo_file, source, output):
    """Evaluate EEG spectral readouts on four fully held-out training concept folds."""
    from paper_object_margin_encoder import visual_transfer

    files = PinnedFiles()
    split = prep / "paper_split.json"
    split_sha = files.pin(split)
    photo_sha = files.pin(photo_file)
    dino_path = source / "object-identity/training-photo-dino.pt"
    dino_sha = files.pin(dino_path)
    bank = torch.load(dino_path, map_location="cpu", weights_only=True)
    rows = bank["rows"].numpy()
    require(len(rows) == 13230 and rows.tolist() == json.loads(split.read_bytes())["train_rows"]
            and split_sha == bank["proof"]["split"], "Train-only photo bank is not registered")
    dino, ids, sums = bank["image_features"].cuda(), bank["labels"].cuda(), bank["sums"].cuda()
    photos = torch.from_numpy(np.load(photo_file, allow_pickle=False, mmap_mode="r")[rows].copy()).cuda()
    sources, folded = {}, {}
    keys = ["native", *[f"wave/{beta}" for beta in WAVE_WEIGHTS],
            *[f"{k}/{beta}/{weight}" for k in MOMENTS for beta in WAVE_WEIGHTS for weight in MOMENT_WEIGHTS]]
    hit_totals = dict.fromkeys(keys, 0)
    for subject in SOURCES:
        array = eeg_root / f"sub-{subject:02d}_train.npy"
        native_path = source / f"conditions-sub-{subject:02d}.pt"
        eeg_sha = files.pin(array)
        native_sha = files.pin(native_path)
        saved = torch.load(native_path, map_location="cpu", weights_only=True)
        proof = saved["sources"]
        require((proof["subject"], proof["split"], proof["images"], proof["eeg"]["array_sha256"])
                == (subject, split_sha, photo_sha, eeg_sha) and np.array_equal(saved["rows"].numpy(), rows),
                "The EEG arrays and native conditions have different provenance")
        folds = saved["folds"].numpy()
        require(set(folds) == {1, 2, 3, 4}, "Invalid held-out training folds")
        native = saved["out_of_fold"].cuda()
        packed = read_spectra(np.load(array, mmap_mode="r", allow_pickle=False), rows)
        sources[subject] = dict(physical_eeg_sha256=eeg_sha, native_conditions_sha256=native_sha)
        local = dict.fromkeys(keys, 0)
        for held in (1, 2, 3, 4):
            tr, va = folds != held, folds == held
            require(not np.intersect1d(ids[tr].cpu().numpy(), ids[va].cpu().numpy()).size,
                    "A visual concept entered the validation fit")
            center, visual_center, transform = visual_transfer(photos[tr], dino[tr])
            cats, inverse = torch.unique(ids[va], return_inverse=True)
            negatives = F.normalize(F.normalize(sums[cats]) - visual_center)
            positives = F.normalize(F.normalize(sums[ids[va]] - dino[va]) - visual_center)

            def score(x):
                directions = F.normalize((x - center) @ transform)
                other = (directions @ negatives.T).scatter_(1, inverse[:, None], float("-inf"))
                return int(((directions * positives).sum(1) > other.max(1).values).sum())

            trained = fit_readouts({k: v[tr] for k, v in packed.items()}, photos[tr])
            offsets = readouts({k: v[va] for k, v in packed.items()}, trained)
            original = native[va]
            local["native"] += score(original)
            for beta in WAVE_WEIGHTS:
                waveform = original.lerp(offsets["wave"], beta)
                local[f"wave/{beta}"] += score(bounded(original, waveform, center))
                for k in MOMENTS:
                    for weight in MOMENT_WEIGHTS:
                        vector = waveform + weight * offsets[k]
                        local[f"{k}/{beta}/{weight}"] += score(bounded(original, vector, center))
        print("local spectral calibration", subject, local, flush=True)
        for name, count in local.items():
            hit_totals[name] += count
        folded[subject] = local
    beta = max(WAVE_WEIGHTS, key=lambda weight: (hit_totals[f"wave/{weight}"], -weight))
    weights = {k: max(MOMENT_WEIGHTS, key=lambda weight: (hit_totals[f"{k}/{beta}/{weight}"], -weight)) for k in MOMENTS}
    files.verify()
    report = dict(n=len(rows) * len(SOURCES), people=list(SOURCES), selected_wave=beta, selected_moments=weights,
                  hits=hit_totals, per_subject=folded, sources=sources, split_sha256=split_sha, photo_sha256=photo_sha,
                  dino_sha256=dino_sha, code_sha256=files.pin(Path(__file__)),
                  spectral_code_sha256=files.pin(Path(paper_trial_spectra.__file__)))
    save_json(output, report)
    print("local spectral calibration total", hit_totals, "selected", beta, weights, flush=True)


@torch.inference_mode()
def render(subject, registration, arms):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_render import condition, draw_images, generator_inputs, groups, physical_eeg, query_seed
    from paper_token_evidence import project_tokens, render_arms, score_outputs
    from paper_visual_ceiling import official_features

    files = PinnedFiles()
    split = PREP / "paper_split.json"
    split_sha = files.pin(split)
    rows = json.loads(split.read_bytes())["official_rows"]
    require(rows == list(range(200)), "The strict four-repeat comparison uses the official set")
    photo_file = PREP / "F_train.npy"
    photo_sha = files.pin(photo_file)
    proof = files.pin(registration)
    locked = json.loads(registration.read_bytes())
    code_sha = files.pin(Path(__file__))
    wavelet_sha = files.pin(Path(paper_trial_spectra.__file__))
    require((locked["split_sha256"], locked["photo_sha256"], locked["code_sha256"], locked["spectral_code_sha256"])
            == (split_sha, photo_sha, code_sha, wavelet_sha)
            and locked["people"] == list(SOURCES) and subject not in SOURCES, "Locked calibration does not match")
    beta, moment_weights = locked["selected_wave"], locked["selected_moments"]
    require(beta in WAVE_WEIGHTS and all(moment_weights[k] in MOMENT_WEIGHTS for k in MOMENTS),
            "Invalid calibrated mixture weights")
    train_indices, _, _ = registered_rows(PREP)
    train, train_proof = physical_eeg(subject, "train", files)
    test, test_proof = physical_eeg(subject, "test", files)
    photo = torch.from_numpy(np.load(photo_file, mmap_mode="r", allow_pickle=False)[train_indices].copy()).cuda()
    state = fit_readouts(read_spectra(train, train_indices), photo)
    prediction = readouts(read_spectra(test, rows), state)
    repeats = groups("official-r4")
    require(repeats == [list(REPEATS)], "Native and spectral encoders need the same presentations")
    native, checkpoint = condition("native-paper", subject, test[rows], repeats, features_only=True, files=files)
    native = native.cuda().float()
    wave = native.lerp(prediction["wave"], beta)
    own = np.arange(len(rows))
    wrong = np.roll(own, len(rows) // 2)
    scores = {arm: native if arm == "native-paper" else wave if arm == "wave" else
              wave + moment_weights[arm.removesuffix("-wrong")]
              * prediction[arm.removesuffix("-wrong")][wrong if arm.endswith("-wrong") else own]
              for arm in arms}
    assets = generator_inputs(files)
    pipeline = load_generator(torch.float16, False)
    _, adapter, projection, _, vae = pipeline
    variants, conditions = [], {}
    for arm, value in scores.items():
        value = bounded(native, value, state["photo_mean"])
        tokens = project_tokens(value, adapter, projection).cpu().half()
        mode = arm.removesuffix("-wrong")
        variants.append((arm, tokens, dict(wave_weight=0. if arm == "native-paper" else beta,
                                           spectral_moment=mode if mode in MOMENTS else None,
                                           moment_weight=moment_weights.get(mode),
                                           moment_donor=(wrong.tolist() if arm.endswith("-wrong") else own.tolist())
                                           if mode in MOMENTS else None)))
        conditions[arm] = value
    common = dict(subject=subject, budget="official-r4", repeats=repeats, split_sha256=split_sha,
                  physical_eeg=test_proof, training_eeg=train_proof, photo_sha256=photo_sha,
                  brain_hive_checkpoint_sha256=checkpoint, calibration_sha256=proof, generator_files_sha256=assets,
                  code_sha256=code_sha, spectral_code_sha256=wavelet_sha,
                  renderer_sha256=files.pin(Path(inspect.getfile(draw_images))), diagnostic_only=False)
    root = study_root()
    starts = [query_seed("official", i) for i in rows]
    locations = render_arms(root / "images" / f"sub-{subject:02d}", variants, rows, starts, common, files, pipeline)
    del pipeline, adapter, projection, vae
    torch.cuda.empty_cache()
    score_outputs(locations, "official")
    target, target_rows, reference_proof = official_features(PREP, files)
    require(target_rows == rows, "The scoring reference uses different query rows")
    ref = F.normalize(target.cuda() - state["photo_mean"])
    diagnostic = {}
    for arm, value in conditions.items():
        similarity = F.normalize(value - state["photo_mean"]) @ ref.T
        diagnostic[arm] = dict(visual_match=int((similarity.argmax(1) == torch.arange(len(rows), device="cuda")).sum()),
                               mean_cosine=float(similarity.diagonal().mean()),
                               rms_condition_delta=float((value - native).square().mean().sqrt()))
    files.verify()
    save_json(root / f"condition-probe-sub-{subject:02d}.json",
              dict(n=len(rows), gallery=reference_proof, metrics=diagnostic))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    calibration = commands.add_parser("screen")
    for name in ("prep", "eeg-root", "photos", "source-root", "output"):
        calibration.add_argument("--" + name, type=Path, required=True)
    images = commands.add_parser("render")
    images.add_argument("subject", type=int, choices=range(5, 11))
    images.add_argument("--calibration", type=Path, required=True)
    images.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    if args.action == "screen":
        screen(args.prep, args.eeg_root, args.photos, args.source_root, args.output)
    else:
        render(args.subject, args.calibration, args.arms)
