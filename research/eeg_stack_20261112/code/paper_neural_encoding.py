"""Measure held-out image-to-ERP scores on the complete official EEG-EditBench."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular


SHRINKAGE = .25
TRAIN_SHAPE = (16540, 4, 17, 250)
TEST_SHAPE = (200, 80, 17, 250)
WINDOW_BINS, BIN_SAMPLES = 40, 5


def fit_templates(images, trials, groups):
    """Fit unconstrained photo and cross-photo concept templates in the same ERP coordinates."""
    _, counts = np.unique(groups, return_counts=True)
    if len(counts) < 2 or counts.min() < 2 or counts.min() != counts.max():
        raise ValueError("Visual concepts need equally many distinct training photographs")
    eeg = trials.mean(axis=1)
    center = eeg.mean(axis=0)
    eeg -= center
    v = images - images.mean(axis=0)
    grouped = np.argsort(groups, kind="stable").reshape(len(counts), counts[0])
    eeg_by_concept = eeg[grouped]
    eeg_concept = eeg_by_concept.mean(axis=1)
    models = {}
    for name, x, y in (("photo", v, eeg), ("concept", v[grouped].mean(axis=1), eeg_concept)):
        gram = x.T @ x
        gram.flat[::len(gram) + 1] += SHRINKAGE * gram.diagonal().mean()
        models[name] = cho_solve(cho_factor(gram), x.T @ y)

    repeats = (trials - trials.mean(axis=1, keepdims=True)).reshape(-1, trials.shape[-1])
    within = (eeg_by_concept - eeg_concept[:, None]).reshape(-1, eeg.shape[-1])
    trial_cov = (repeats.T @ repeats) / (len(trials) * (trials.shape[1] - 1))
    within_cov = (within.T @ within) / (len(trials) - len(counts))
    return center, models, trial_cov, within_cov


def neural_cosine(eeg, templates, covariance):
    """Compare an observed ERP and a predicted one in a training-defined metric."""
    lower = cholesky(covariance, lower=True)
    whitened = [solve_triangular(lower, x.T, lower=True).T for x in (eeg, templates)]
    scaled = [x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10) for x in whitened]
    return scaled[0] @ scaled[1].T


def other_photo_scores(scores, labels):
    """Leave one photo per candidate out, including the query photo itself."""
    _, key, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(labels, kind="stable").reshape(len(counts), counts[0])
    position = np.empty_like(key)
    position[order] = np.arange(counts[0])
    gallery = scores[:, order]
    excluded = np.take_along_axis(gallery, position[:, None, None], axis=-1).squeeze(-1)
    return (gallery.sum(-1) - excluded) / (counts[0] - 1), key


def readout_scores(scores, names, labels, targets):
    from paper_raw_semantic_readout import top_credit

    cross_photo, key = other_photo_scores(scores, labels)
    own = np.arange(len(labels))
    same = np.where(labels[:, None] == labels[None, :], scores, -np.inf)
    return np.column_stack((top_credit(names, targets, 1), top_credit(cross_photo, key, 1),
                            top_credit(scores, own, 1), top_credit(same, own, 1)))


def held_native(study, subject, held, train_sha, pins, files):
    from paper_raw_semantic_readout import normalize, read_json
    from paper_render_cache import require

    directory = study / 'horizon-readout-20261127/native-h14'
    path = directory / f'sub-{subject:02d}.h14.npz'
    receipt = read_json(directory / f'sub-{subject:02d}.json')
    proof = receipt['proof']
    require(proof['subject'] == subject and proof['eeg']['array_sha256'] == train_sha
            and proof['repeats'] == list(range(4)) and proof['views'][0] == 'native'
            and proof['registration']['split'] == pins['training_split']
            and proof['registration']['development'] == pins['development_split']
            and proof['registration']['labels'] == pins['training_labels']
            and receipt['h14_sha256'] == files.pin(path), 'R4 baseline has a different fit, panel or physical trial budget')
    with np.load(path, allow_pickle=False) as data:
        require(np.array_equal(data['row'], held) and data['embedding'].shape == (len(held), 4, 1024),
                'R4 baseline feature order differs from the trial order')
        return normalize(data['embedding'][:, 0]), receipt['h14_sha256']


def evaluate_held(held, labels, names, records, bootstrap_ci):
    size = 200
    dev = slice(0, size)
    fresh = slice(size, len(held))
    labels_dev, labels_fresh = labels[held[dev]], labels[held[fresh]]
    selection = {}
    for weight in (0., .125, .25, .50):
        credits = []
        for native, head in zip(records['native'], records['concept-within'], strict=True):
            scores = (1 - weight) * native[0][dev, dev] + weight * head[0][dev, dev]
            scores, target = other_photo_scores(scores, labels_dev)
            truth = scores[np.arange(len(scores)), target]
            credits.append((1 / (1 + (scores > truth[:, None]).sum(axis=1))).mean())
        selection[weight] = np.mean(credits)
    chosen = max(selection, key=selection.get)
    records['calibrated-combination'] = [tuple((1 - chosen) * a + chosen * b for a, b in zip(native, head, strict=True))
                                         for native, head in zip(records['native'], records['concept-within'], strict=True)]
    outcomes = {arm: np.stack([readout_scores(image[fresh, fresh], text[fresh], labels_fresh, names[fresh])
                               for image, text in evidence]) for arm, evidence in records.items()}
    paired = {}
    keys = ('name1854', 'other_nine_50way', 'original_500way', 'within_concept_10way')
    grouped = np.argsort(labels_fresh, kind='stable').reshape(50, 10)
    for name in outcomes:
        if name != 'native':
            delta = (outcomes[name] - outcomes['native'])[:, grouped].mean(axis=2)
            lo, hi = bootstrap_ci(delta)
            paired[name + ' minus native'] = dict(zip(keys, np.column_stack((delta.mean((0, 1)), lo, hi)).tolist(), strict=True))
    hits = {name: dict(zip(keys, record.sum((0, 1)).tolist(), strict=True)) for name, record in outcomes.items()}
    return dict(n=5000, hits=hits, paired=paired, development_reciprocal_rank=selection, calibrated_weight=chosen), outcomes


def pool_trials(raw, rows):
    features = np.empty((len(rows), raw.shape[1], 17 * WINDOW_BINS), dtype=np.float32)
    for start in range(0, len(rows), 160):
        selected = rows[start:start + 160]
        segments = raw[selected, :, :, :WINDOW_BINS * BIN_SAMPLES].astype(np.float32)
        binned = segments.reshape(len(selected), raw.shape[1], 17, WINDOW_BINS, BIN_SAMPLES).mean(-1)
        features[start:start + len(selected)] = binned.reshape(len(selected), raw.shape[1], -1)
    return features


def load_eeg(root, subject, split, shape, ids_digest, files):
    from paper_render_cache import require
    from paper_raw_semantic_readout import read_json

    path = root / f"sub-{subject:02d}_{split}.npy"
    proof = read_json(path.with_suffix(".json"))
    digest = files.pin(path)
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    require(proof["output_sha256"] == digest and proof["ids_sha256"] == ids_digest
            and proof["checked_physical_photo_ids"] == shape[0] * shape[1] and proof["sample_rate_hz"] == 250
            and raw.shape == shape and raw.dtype == np.float16,
            f"Unexpected physical {split} trials for participant {subject}")
    return raw, digest


def run(benchmark, output, export_subjects=()):
    from paper_aggregate import bootstrap_ci
    from paper_editbench import FAMILIES, TOTALS, cluster_edits, edit_outcomes, point_metrics
    from paper_raw_semantic_readout import normalize, read_json
    from paper_render_cache import PinnedFiles, atomic_file, require, save_json

    files = PinnedFiles()
    code_sha256 = files.pin(Path(__file__))
    prep = benchmark / "verified/data/prep"
    registered = benchmark / "inputs/image_set"
    raw = benchmark / "inputs/prep17"
    teacher = benchmark / "native_baseline_candidate/enigma/targets"
    study = benchmark / "semantic-energy-check"
    edits = study / "background-invariant-20261127"
    path = teacher / "train_h14.npz"
    train_file = registered / "train_photos.csv"
    test_file = registered / "test_photos.csv"
    sources = {"training_split": prep / "paper_split.json", "training_labels": prep / "train_concept.npy",
               "development_split": prep / "paper_development.json", "names": study / "h14-text-bank.npz",
               "train_stimuli": train_file, "test_stimuli": test_file, "photo_features": path,
               "train_ids": raw / "train_ids.json", "test_ids": raw / "test_ids.json",
               "edit_features": edits / "editbench-visual.npz", "native_scores": edits / "frozen-r80-editbench.npz"}
    pins = {key: files.pin(p) for key, p in sources.items()}
    teacher_proof = read_json(teacher / "receipt.json")
    visual_proof = read_json(sources["edit_features"].with_suffix(".json"))
    native_proof = read_json(sources["native_scores"].with_suffix(".json"))
    require(teacher_proof["pack_sha256"] == pins["photo_features"]
            and teacher_proof["train_photo_index_sha256"] == pins["train_stimuli"]
            and teacher_proof["text_encoder_sha256"] == visual_proof["sources"]["weights"]
            and visual_proof["scope"] == "evaluation-only" and visual_proof["array_sha256"] == pins["edit_features"]
            and visual_proof["sources"]["registry"] == pins["test_stimuli"]
            and native_proof["arrays"] == pins["native_scores"]
            and native_proof["proof"]["references"] == pins["edit_features"], "Inconsistent H/14 or EEG-EditBench provenance")
    tables = []
    for name, registry in (("train", train_file), ("test", test_file)):
        with registry.open(encoding="utf-8-sig", newline="") as stream:
            entries = list(csv.DictReader(stream))
        require([int(row["row0"]) for row in entries] == list(range(len(entries)))
                and [row["image_id"] for row in entries] == read_json(sources[name + "_ids"])["ids"],
                "The EEG stimulus order must equal the image order")
        tables.append(entries)
    train, test = tables
    split = read_json(sources["training_split"])
    rows = np.asarray(split["train_rows"])
    labels = np.load(sources["training_labels"], allow_pickle=False)
    require(rows.shape == (13230,) and rows.min() >= 0 and rows.max() < len(train) and np.all(np.diff(rows) > 0)
            and len(np.unique(labels[rows])) == 1323 and tuple(split["official_rows"]) == tuple(range(200))
            and tuple(split["trial_indices_R80"]) == tuple(range(80))
            and set(row["unique_id"] for row in test).isdisjoint(train[i]["unique_id"] for i in rows),
            "The official 200 concepts must be disjoint from the 1,323 fitting concepts")
    dev = np.asarray(read_json(sources['development_split'])['dev_rows'])
    fresh = np.asarray(split['fresh_rows'])
    held = np.concatenate((dev, fresh))
    require(np.array_equal(split['trial_indices_R4'], np.arange(4)) and len(np.unique(held)) == 700
            and set(labels[dev]).isdisjoint(labels[fresh]) and set(labels[held]).isdisjoint(labels[rows])
            and all(np.array_equal(np.unique(labels[panel], return_counts=True)[1], np.full(count, 10))
                    for panel, count in ((dev, 20), (fresh, 50))), 'R4 comparison requires independent whole-concept panels')
    with np.load(path, allow_pickle=False) as data:
        require(np.array_equal(data["photo_ids"], [row["image_id"] for row in train])
                and np.array_equal(data["class_index"], labels)
                and np.array_equal(data["category"][data["class_index"]], [row["unique_id"] for row in train]),
                "Training visual rows are out of order")
        training = normalize(data["images"][rows].astype(np.float32))
        held_photos = normalize(data['images'][held].astype(np.float32))
        text_mean = normalize(data['texts'][np.unique(labels[rows])].astype(np.float32)).mean(axis=0)
    names_proof = read_json(sources['names'].with_suffix('.json'))
    require(names_proof['bank_sha256'] == pins['names'] and names_proof['model']['weights'] == teacher_proof['text_encoder_sha256'],
            'R4 naming bank must come from the pinned H/14 weights')
    with np.load(sources['names'], allow_pickle=False) as data:
        words = normalize(data['text'])
        dictionary = {name: i for i, name in enumerate(data['concept_id'])}
        require(len(dictionary) == 1854 and words.shape == (1854, 1024), 'R4 naming requires the complete 1854-word bank')
        targets = np.array([dictionary[train[i]['unique_id']] for i in held])
    with np.load(sources["edit_features"], allow_pickle=False) as data:
        origin, family, ids = (data[name] for name in ("origin", "family", "query_id"))
        references, canonical = (normalize(data[name]) for name in ("embedding", "canonical"))
        require(tuple(np.bincount(family)) == TOTALS and references.shape == (2337, 1024)
                and list(ids) == [f"{int(row['split_class_id']):05d}" for row in test], "Invalid edit reference order")
    with np.load(sources["native_scores"], allow_pickle=False) as data:
        require(np.array_equal(data["query_id"], ids) and np.array_equal(data["origin"], origin)
                and np.array_equal(data["family"], family), "Baseline edit scores do not use the same reference pairs")
        native = data["native"]
    require(native.shape == (10, 200, 2337), "The native comparison requires a complete ten-person replay")

    centered = np.vstack((references, canonical)) - training.mean(axis=0)
    held_centered = np.vstack((held_photos - training.mean(axis=0), words - text_mean))
    proof = {}
    results, details = {}, {}
    held_evidence = {'native': [], 'photo-within': [], 'concept-within': []}
    render_weights = {}
    for person in range(1, 11):
        train_eeg, train_sha = load_eeg(raw, person, "train", TRAIN_SHAPE, pins["train_ids"], files)
        test_eeg, test_sha = load_eeg(raw, person, "test", TEST_SHAPE, pins["test_ids"], files)
        require(test_sha == native_proof["proof"]["physical"][str(person)], "Baseline used different physical R80 trials")
        proof[str(person)] = dict(train=train_sha, test=test_sha)
        center, heads, trial_cov, within_cov = fit_templates(training, pool_trials(train_eeg, rows), labels[rows])
        observed = pool_trials(test_eeg, np.arange(200)).mean(axis=1) - center
        held_observed = pool_trials(train_eeg, held).mean(axis=1) - center
        held_h14, held_digest = held_native(study, person, held, train_sha, pins, files)
        proof[str(person)]['held_h14'] = held_digest
        held_evidence['native'].append((held_h14 @ held_photos.T, held_h14 @ words.T))
        metrics = {"native": native[person - 1]}
        for covariance in (trial_cov, within_cov):
            covariance.flat[::len(covariance) + 1] += SHRINKAGE * covariance.diagonal().mean()
        if person in export_subjects:
            lower = cholesky(within_cov, lower=True)
            render_weights[person] = dict(eeg_center=center, visual_center=training.mean(axis=0),
                                          lower=lower, projection=solve_triangular(lower, heads['concept'].T, lower=True).T)
        for label, weights in heads.items():
            predicted = centered @ weights
            held_scores = neural_cosine(held_observed, held_centered @ weights, within_cov)
            held_evidence[f'{label}-within'].append((held_scores[:, :len(held)], held_scores[:, len(held):]))
            metrics[f"{label}-trial"] = neural_cosine(observed, predicted[:2337], trial_cov)
            metrics[f"{label}-within"] = neural_cosine(observed, predicted[:2337], within_cov)
            if label == "concept":
                scores = neural_cosine(observed, predicted, within_cov)
                metrics["concept-within-canonical"] = np.concatenate((scores[:, 2337:], scores[:, 200:2337]), axis=1)
                metrics["concept-within-wrong"] = np.roll(metrics["concept-within"], 1, axis=0)
        metrics["native+concept-within"] = (metrics["native"] + metrics["concept-within"]) / 2
        metrics["native+wrong-concept-within"] = (metrics["native"] + metrics["concept-within-wrong"]) / 2
        for arm, values in metrics.items():
            require(np.isfinite(values).all(), f"Invalid neural scores for {arm}")
            outcome = edit_outcomes(values, 200, origin)
            results.setdefault(arm, []).append(outcome)
            details.setdefault(arm, []).append(np.column_stack((*outcome[:2], cluster_edits(outcome[-1], family, origin, 200))))
        print(f"Scored all physical trials and public edits for participant {person:02d}", flush=True)

    summaries = {arm: point_metrics(tuple(np.stack(values) for values in zip(*records, strict=True)), family)
                 for arm, records in results.items()}
    comparisons = {}
    for candidate, control in (("concept-within", "photo-within"), ("concept-within", "concept-trial"),
                               ("concept-within", "concept-within-wrong"), ("concept-within", "native"),
                               ("native+concept-within", "native"), ("native+concept-within", "native+wrong-concept-within"),
                               ("concept-within-canonical", "concept-within")):
        delta = np.asarray(details[candidate]) - np.asarray(details[control])
        lo, hi = bootstrap_ci(delta)
        comparisons[f"{candidate} minus {control}"] = dict(zip(("200way_original", "ep_top1", *FAMILIES),
                                                               np.column_stack((delta.mean((0, 1)), lo, hi)).tolist(), strict=True))
    held_report, held_values = evaluate_held(held, labels, targets, held_evidence, bootstrap_ci)
    files.verify()
    directory = output.parent
    directory.mkdir(parents=True, exist_ok=True)
    archive = output.with_suffix(".npz")
    details = {arm: np.stack(records) for arm, records in details.items()}
    atomic_file(archive, lambda stream: np.savez_compressed(stream, **details, query_id=ids,
                                                           fresh_row=fresh, **{'fresh_' + k: v for k, v in held_values.items()}))
    save_json(output, dict(training_photos=len(rows), training_concepts=1323, train_repeats=4, test_repeats=80,
                           subjects=10, features="17 posterior electrodes, 0–800 ms in 20 ms bins", shrinkage=SHRINKAGE,
                           sources=pins, physical_eeg=proof, code_sha256=code_sha256,
                           summary=summaries, paired=comparisons, fresh_r4=held_report, observations_sha256=files.pin(archive)))
    for person, weights in render_weights.items():
        target = directory / f'sub-{person:02d}-neural-guide.npz'
        atomic_file(target, lambda stream: np.savez_compressed(stream, **weights))
        save_json(target.with_suffix('.json'), dict(subject=person, weights_sha256=files.pin(target),
                                                   training_split=pins['training_split'], training_eeg=proof[str(person)]['train'],
                                                   training_visual=pins['photo_features'], training_only=True,
                                                   mixing=held_report['calibrated_weight'], study_sha256=files.pin(output)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument('--export-subjects', nargs='+', choices=range(1, 11), type=int, default=())
    args = parser.parse_args()
    run(args.benchmark, args.output, args.export_subjects)
